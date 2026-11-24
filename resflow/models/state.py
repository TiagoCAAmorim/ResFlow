"""State model: one time step of the reservoir state, as cell volume increments.

`StateModel` wraps the graph network with everything around it: feature computation
from the physical state, linear input normalisation (with grouped statistics), the asinh
compression of a few heavy-tailed inputs, the well-type embedding, the output
de-normalisation into standard-volume increments (dVo, dVg, dVw per cell), and the
volume -> state inversion of the physics (`step`).

The normalisation statistics, the PVT / relative-permeability tables and the pressure
estimator of the inversion are part of the state dict, so a model is fully defined by a
config dict and its weights (see `resflow.models.bundle`).

Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from ..data.normalization import LinearScaler, MIN_INFORMATIVE_SCALE
from ..physics.transform import Transform
from .features import (TARGET_NORM_NAMES, WELL_BHP_WI_NAMES, all_feature_names,
                       compute_features, con_data_as_grid, dynamic_well_type,
                       edge_feature_names, grouped_names, node_feature_names,
                       state_with_volumes)
from .layers import GraphNet

# Network hyper-parameters (config keys) and their defaults (the final 2D model).
NETWORK_DEFAULTS = {
    'hidden_dims': [128, 64],
    'enc_nodes_dims': [128, 128, 128],
    'enc_edges_dims': [64, 64],
    'dec_nodes_dims': [128, 128],
    'kernel_dims': [128],
    'edge_mlp_dims': [64, 64],
    'n_blocks': 12,
    'conv_rank': 8,
    'activation': 'gelu',
    'decoder_concat_input': True,
    'use_bn': True,
    'add_inner_skip': True,
    'dense_layers': False,
    'init_scale': [],
    'init_type': 'xavier_uniform',
}
OUTPUT_NAMES = ['dVo_std', 'dVg_std', 'dVw_std']


def make_scaler(device='cpu') -> LinearScaler:
    """Linear scaler over the feature registry, with the registry's grouping."""
    return LinearScaler(trainable=False, device=device, group_fn=grouped_names)


def select_state_features(scaler: LinearScaler, use_beta_ig: bool = False,
                          use_beta_iw: bool = True) -> Tuple[List[str], List[str]]:
    """Node and edge features of a new state model: every registry feature except the
    BHP / well-index columns (the well controls enter as volumes), the injection
    mobilities not asked for, and the features whose (grouped) scale is degenerate in
    the fitted statistics (constant in the training set, e.g. depth on a 2D model)."""
    exclude = set(WELL_BHP_WI_NAMES)
    if not use_beta_ig:
        exclude.add('beta_ig')
    if not use_beta_iw:
        exclude.add('beta_iw')
    nodes = [n for n in node_feature_names() if n not in exclude]
    nodes = scaler.drop_degenerate_scale(nodes, MIN_INFORMATIVE_SCALE, grouped=True)
    edges = scaler.drop_degenerate_scale(edge_feature_names(), MIN_INFORMATIVE_SCALE,
                                         grouped=True)
    return nodes, edges


def transform_from_state(state: Dict[str, torch.Tensor], prefix: str = 'transform.',
                         with_vols2p: bool = True, g: float = 1e-4,
                         z_mean: Optional[float] = None) -> Transform:
    """Physics module rebuilt from the tensors of a state dict (no dataset needed)."""
    from ..physics.pvt import PVT_KEYS  # pylint: disable=import-outside-toplevel
    from ..physics.vols2p import build_vols2p_net  # pylint: disable=import-outside-toplevel
    pvt = {k: state[f'{prefix}pvt.{k}'] for k in PVT_KEYS if f'{prefix}pvt.{k}' in state}
    wo = state[f'{prefix}krel.wo_table'].T
    go = state[f'{prefix}krel.go_table'].T
    vols2p = build_vols2p_net(state, f'{prefix}vols2p_nn.') if with_vols2p else None
    return Transform(pvt, wo, go, vols2p_nn=vols2p, g=g, z_mean=z_mean)


class InputAsinh(nn.Module):
    """asinh(z / c) / k on some already normalised channels (zero stays zero)."""

    def __init__(self, names: Sequence[str], config: Dict[str, Sequence[float]]):
        super().__init__()
        cfg = {n: ck for n, ck in (config or {}).items() if n in names}
        names = list(names)
        self.register_buffer('pos', torch.tensor([names.index(n) for n in cfg], dtype=torch.long),
                             persistent=False)
        self.register_buffer('c', torch.tensor([float(c) for c, _ in cfg.values()]),
                             persistent=False)
        self.register_buffer('inv_k', torch.tensor([1.0 / float(k) for _, k in cfg.values()]),
                             persistent=False)

    def forward(self, x):
        if self.pos.numel() == 0:
            return x
        sel = x.index_select(-1, self.pos)
        return x.index_copy(-1, self.pos, torch.asinh(sel / self.c) * self.inv_k)


class StateModel(nn.Module):
    """The state model (paper Sections 3.4-3.7).

    Args:
        config: model config (see `resflow.models.bundle`): ``network`` (layer sizes),
            ``node_features`` / ``edge_features`` (ordered input names), ``input_asinh``,
            ``well_type_embedding`` ({'num_types', 'dim', 'active_eps'}).
        transform: physics (PVT, relative permeability, vols2p estimator, G).
        data_norm_in, data_norm_out: fitted scalers (inputs; output de-normalisation and
            the loss).
    """

    def __init__(self, config: dict, transform: Transform, data_norm_in: LinearScaler,
                 data_norm_out: LinearScaler):
        super().__init__()
        self.config = config
        self.data_norm_in = data_norm_in
        self.data_norm_out = data_norm_out
        self.transform = transform
        for module in (self.data_norm_in, self.data_norm_out, self.transform):
            for p in module.parameters():
                p.requires_grad = False
        self.node_props = list(config['node_features'])
        self.edge_props = list(config['edge_features'])
        self.input_asinh = InputAsinh(self.node_props, config.get('input_asinh'))
        self.input_asinh_e = InputAsinh(self.edge_props, config.get('input_asinh'))
        emb = config.get('well_type_embedding', {'num_types': 3, 'dim': 4, 'active_eps': 1e-4})
        self.well_active_eps = float(emb['active_eps'])
        self.well_type_embedding = nn.Embedding(int(emb['num_types']), int(emb['dim']))
        net = {**NETWORK_DEFAULTS, **config.get('network', {})}
        self.model = GraphNet(
            input_dims=(len(self.node_props) + int(emb['dim']), len(self.edge_props)),
            output_dim=len(OUTPUT_NAMES), **net)
        # Exact (run-to-run identical) aggregation of the edge flows outside training.
        self.exact_aggregation = True
        self.model_batch_limit: Optional[int] = config.get('model_batch_limit', 16)

    # MARK: Construction
    @classmethod
    def from_state(cls, config: dict, state: Dict[str, torch.Tensor],
                   device='cpu') -> "StateModel":
        """Model with saved weights: scalers, physics and network all from `state`."""
        names = config.get('normalization', {}).get('feature_names') or all_feature_names()
        norm_in, norm_out = make_scaler(), make_scaler()
        norm_in.restore_from_state(state, 'data_norm_in.', names)
        norm_out.restore_from_state(state, 'data_norm_out.', names)
        phys = config.get('physics', {})
        transform = transform_from_state(state, g=float(phys.get('g', 1e-4)),
                                         z_mean=phys.get('z_mean'))
        model = cls(config, transform, norm_in, norm_out)
        model.load_state_dict(state, strict=True)
        return model.to(device)

    # MARK: Forward
    def _inputs(self, batch, dv_con: Optional[torch.Tensor]):
        dv_given = dv_con is not None
        if dv_con is None:
            dv_con, _ = con_data_as_grid(batch)
        well_type = dynamic_well_type(dv_con, eps=self.well_active_eps)
        x, e = compute_features(
            batch, self.transform, self.node_props, self.edge_props,
            con_delta_override=dv_con if dv_given else None,
            deterministic=(not self.training) and self.exact_aggregation)
        x = self.input_asinh(self.data_norm_in.norm(x, self.node_props, grouped=True))
        e = self.input_asinh_e(self.data_norm_in.norm(e, self.edge_props, grouped=True))
        x = torch.cat([x, self.well_type_embedding(well_type)], dim=-1)
        return x, e

    def _network(self, x, edge_index, e):
        limit = self.model_batch_limit
        if limit is None or x.ndim < 3 or x.shape[0] <= limit:
            return self.model(x, edge_index, e)
        return torch.cat([self.model(x[i:i + limit], edge_index, e[i:i + limit])
                          for i in range(0, x.shape[0], limit)], dim=0)

    def forward(self, batch, dv_con: Optional[torch.Tensor] = None,
                return_raw: bool = False):
        """Volume increments [..., N, 3] (dVo, dVg, dVw at standard conditions per cell).

        dv_con: per-cell well volumes of the step ([..., N, 5]: dNp, dGp, dWp, dGi, dWi)
            replacing the sample's (rate mode). BHPs are always the sample's.
        return_raw: also return the network output before de-normalisation.
        """
        x, e = self._inputs(batch, dv_con)
        raw = self._network(x, batch.edge_index, e)
        dvols = self.data_norm_out.denorm(raw, OUTPUT_NAMES, grouped=True)
        return (dvols, raw) if return_raw else dvols

    @torch.no_grad()
    def step(self, batch, dv_con: Optional[torch.Tensor] = None,
             max_iter: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """One step from batch.x: returns ([..., N, 8] Rs, P, So, Sg, Sw, Vo, Vg, Vw and
        [..., N, 4] Rs, P, So, Sg) at the end of the step."""
        vpor = batch.stat[:, 1]
        dvols = self(batch, dv_con=dv_con)
        v1 = self.transform.t2vols(vpor, batch.x) + dvols
        t1 = self.transform.vols2t(vpor, v1, max_iter=max_iter)
        return state_with_volumes(self.transform, vpor, t1, v1), t1

    def target_norm(self, y: torch.Tensor) -> torch.Tensor:
        """Normalise [..., 8] state increments with the output statistics (loss space)."""
        return self.data_norm_out.norm(y, TARGET_NORM_NAMES, grouped=True)
