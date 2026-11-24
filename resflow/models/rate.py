"""Rate model: well volumes of a step from the reservoir state and the well controls.

For each well cell, the network predicts a non-negative multiplier (softplus) of a
Peaceman-type baseline ``WI * mobility * drawdown * K`` per phase (paper Section 3.8):
oil and water production use the producer BHP and the oil / water mobilities, water
injection the injector BHP and the total mobility of the injected water. Gas production
is free gas (own multiplier on the free-gas baseline) plus dissolved gas Rs * oil.

Inputs: the state features (as the state model) plus BHP and well index per cell, with
their own (non-grouped) statistics, and a static well-type embedding (none, producer,
gas injector, water injector) from the connection lists, so the target is not leaked.

Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..data.normalization import LinearScaler, MIN_INFORMATIVE_SCALE
from ..physics.transform import K, Transform
from .features import (CON_DELTA_NAMES, all_feature_names, compute_features,
                       con_data_as_grid, con_wi_as_grid, edge_feature_names, n_nodes_of,
                       node_feature_names)
from .layers import GraphNet
from .state import make_scaler, transform_from_state

RATE_TARGET_NAMES = ["dNp", "dGp", "dWp", "dWi"]
WELL_TYPE_NONE, WELL_TYPE_PRODUCER, WELL_TYPE_GAS_INJECTOR, WELL_TYPE_WATER_INJECTOR = 0, 1, 2, 3
N_WELL_TYPES = 4

NETWORK_DEFAULTS = {
    'hidden_dims': [16, 8],
    'enc_nodes_dims': [16, 16],
    'enc_edges_dims': [8, 8],
    'dec_nodes_dims': [16, 16],
    'kernel_dims': [16],
    'edge_mlp_dims': [8],
    'n_blocks': 2,
    'conv_rank': 4,
    'activation': 'gelu',
    'decoder_concat_input': False,
    'use_bn': True,
    'add_inner_skip': True,
    'dense_layers': True,
    'init_scale': 0.001,
    'init_type': 'xavier_uniform',
}


def select_rate_features(scaler: LinearScaler, use_beta_ig: bool = False,
                         use_beta_iw: bool = True):
    """Node and edge features of a new rate model: every registry feature except the
    well volumes (the targets) and the injection mobilities not asked for, without the
    features whose own (non-grouped) scale is degenerate."""
    exclude = set(CON_DELTA_NAMES)
    if not use_beta_ig:
        exclude.add('beta_ig')
    if not use_beta_iw:
        exclude.add('beta_iw')
    nodes = [n for n in node_feature_names() if n not in exclude]
    nodes = scaler.drop_degenerate_scale(nodes, MIN_INFORMATIVE_SCALE, grouped=False)
    edges = scaler.drop_degenerate_scale(edge_feature_names(), MIN_INFORMATIVE_SCALE,
                                         grouped=False)
    return nodes, edges


def static_well_type(batch) -> torch.Tensor:
    """Per-cell well type from the connection lists: 0 none, 1 producer, 2 gas injector,
    3 water injector (producer wins, then gas injector)."""
    well_type = torch.zeros(n_nodes_of(batch), dtype=torch.long, device=batch.stat.device)
    for attr, type_id in (('conw_index', WELL_TYPE_WATER_INJECTOR),
                          ('cong_index', WELL_TYPE_GAS_INJECTOR),
                          ('conp_index', WELL_TYPE_PRODUCER)):
        idx = getattr(batch, attr, None)
        if idx is not None and idx.numel() > 0:
            well_type[idx] = type_id
    return well_type


def rate_targets(batch) -> torch.Tensor:
    """True per-cell well volumes of the step [..., N, 4] (dNp, dGp, dWp, dWi)."""
    con_delta, _ = con_data_as_grid(batch)
    return con_delta[..., [CON_DELTA_NAMES.index(n) for n in RATE_TARGET_NAMES]]


def rate_baseline(batch, transform: Transform, step_size: float = 1.0) -> Dict[str, torch.Tensor]:
    """Peaceman-type estimates per cell (volumes per step, clamped at zero: no crossflow).

    dNp, dWp, dGp_free = WI_p * beta_{o,w,g} * (P - BHP_p) * K * step;
    dWi = WI_w * beta_w_inj * (BHP_w - P) * K * step. P is the cell pressure at the start of
    the step, BHP the control at its end.
    """
    t = batch.x
    p = t[..., 1]
    beta = transform.get_mobility(t, expanded=True)
    _, con_bhp = con_data_as_grid(batch)
    wi = con_wi_as_grid(batch)
    bhp_p, bhp_w = con_bhp[..., 0], con_bhp[..., 2]
    wi_p, wi_w = wi[..., 0], wi[..., 2]
    return {
        'dNp': (wi_p * beta[..., 0] * (p - bhp_p) * K * step_size).clamp(min=0.0),
        'dWp': (wi_p * beta[..., 2] * (p - bhp_p) * K * step_size).clamp(min=0.0),
        'dWi': (wi_w * beta[..., 4] * (bhp_w - p) * K * step_size).clamp(min=0.0),
        'dGp_free': (wi_p * beta[..., 1] * (p - bhp_p) * K * step_size).clamp(min=0.0),
    }


class RateModel(nn.Module):
    """The rate model.

    Args:
        config: ``network``, ``node_features``, ``edge_features``, ``step_size`` (days per
            step: scales the per-day baseline), ``well_type_embedding`` ({'dim'}).
        transform: physics (PVT and relative permeability).
        data_norm: fitted scaler (inputs and targets, non-grouped).
    """

    def __init__(self, config: dict, transform: Transform, data_norm: LinearScaler):
        super().__init__()
        self.config = config
        self.data_norm = data_norm
        self.transform = transform
        for module in (self.data_norm, self.transform):
            for p in module.parameters():
                p.requires_grad = False
        self.feats: List[str] = list(config['node_features'])
        self.edge_feats: List[str] = list(config['edge_features'])
        self.target_feats = RATE_TARGET_NAMES
        self.step_size = float(config.get('step_size', 1.0))
        dim = int(config.get('well_type_embedding', {}).get('dim', 4))
        self.well_type_embedding = nn.Embedding(N_WELL_TYPES, dim)
        net = {**NETWORK_DEFAULTS, **config.get('network', {})}
        self.model = GraphNet(input_dims=(len(self.feats) + dim, len(self.edge_feats)),
                              output_dim=len(RATE_TARGET_NAMES), **net)
        self.exact_aggregation = True

    @classmethod
    def from_state(cls, config: dict, state: Dict[str, torch.Tensor], device='cpu') -> "RateModel":
        """Model with saved weights (scaler and physics from `state`)."""
        names = config.get('normalization', {}).get('feature_names') or all_feature_names()
        norm = make_scaler()
        norm.restore_from_state(state, 'data_norm.', names)
        phys = config.get('physics', {})
        transform = transform_from_state(state, with_vols2p=False, g=float(phys.get('g', 1e-4)),
                                         z_mean=phys.get('z_mean'))
        model = cls(config, transform, norm)
        model.load_state_dict(state, strict=True)
        return model.to(device)

    def inputs(self, batch):
        """Normalised node (with the well-type embedding) and edge inputs."""
        x, e = compute_features(batch, self.transform, self.feats, self.edge_feats,
                                deterministic=(not self.training) and self.exact_aggregation)
        x = self.data_norm.norm(x, self.feats, grouped=False)
        e = self.data_norm.norm(e, self.edge_feats, grouped=False)
        emb = self.well_type_embedding(static_well_type(batch))
        if x.ndim == 3:
            emb = emb.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((x, emb), dim=-1), e

    def forward(self, batch, return_raw: bool = False):
        """Per-cell well volumes of the step [..., N, 4] (dNp, dGp, dWp, dWi)."""
        x, e = self.inputs(batch)
        raw = self.model(x, batch.edge_index, e)
        mult = F.softplus(raw)
        base = rate_baseline(batch, self.transform, self.step_size)
        qo = mult[..., 0] * base['dNp']
        qg = mult[..., 1] * base['dGp_free'] + batch.x[..., 0] * qo
        out = torch.stack((qo, qg, mult[..., 2] * base['dWp'], mult[..., 3] * base['dWi']), dim=-1)
        return (out, raw) if return_raw else out

    def target_norm(self, y: torch.Tensor) -> torch.Tensor:
        """Normalise [..., 4] well volumes with their own statistics (loss space)."""
        return self.data_norm.norm(y, self.target_feats, grouped=False)

    @staticmethod
    def to_con_delta(rates: torch.Tensor) -> torch.Tensor:
        """[..., 4] (dNp, dGp, dWp, dWi) -> the [..., 5] layout of `con_data_as_grid`
        (dGi = 0: no gas injection is predicted)."""
        zeros = torch.zeros_like(rates[..., :1])
        return torch.cat((rates[..., 0:3], zeros, rates[..., 3:4]), dim=-1)
