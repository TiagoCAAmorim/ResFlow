"""Input features of the graph models, computed from a sample's state and statics.

The feature *registry* lists every feature the models can use, by group, in a fixed
order. The normalisation statistics stored with the weights hold one (shift, scale) per
registry entry (`all_feature_names`: node, then delta, then edge features; 90 in total).
A model selects some of them by name (its config lists them in order).

Node features (per cell, at the start of a step):
    init: Rs, P;  s: So, Sg, Sw;  v_std: Vo_std, Vg_std, Vw_std (standard volumes);
    v_res: Vo_res, Vg_free_res, Vw_res (reservoir volumes);  phi: Phi_o, Phi_g, Phi_w
    (phase potentials P - (z - z_ref) rho G);  por_vol: PorVol_ref, PorVol_res;
    beta: beta_o, beta_g, beta_w, beta_ig, beta_iw (mobilities; ig/iw: injection);
    gf, gf_pos, gf_neg: G(f_*) sums of the edge flows into the cell (all, inflow only,
    outflow only);  con_data: dNp, dGp, dWp, dGi, dWi (well volumes of the step per cell)
    and BHPp, BHPg, BHPw (bottom-hole pressure at the end of the step);
    stat_data: z, por, perm_i, perm_j, perm_k and their logs;  con_wi: WIp, WIg, WIw.
Delta features (targets): dRs, dP, dSo, dSg, dSw, dVo_std, dVg_std, dVw_std, dVo_res,
    dVg_free_res, dVw_res, dPhi_o, dPhi_g, dPhi_w.
Edge features: upwind flows (fo, fg_free, fw, fg_dissolved, fg_g_inj, fg_w_inj), their
    positive parts, potential differences, transmissibility, distances (+ logs).

"Grouped" names: several features share the statistics of a representative one (for
example all saturations use So's). `grouped_names` maps names to representatives.

Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

NODE_NAMES: Dict[str, List[str]] = {
    "init": ["Rs", "P"],
    "s": ["So", "Sg", "Sw"],
    "v_std": ["Vo_std", "Vg_std", "Vw_std"],
    "v_res": ["Vo_res", "Vg_free_res", "Vw_res"],
    "phi": ["Phi_o", "Phi_g", "Phi_w"],
    "por_vol": ["PorVol_ref", "PorVol_res"],
    "beta": ["beta_o", "beta_g", "beta_w", "beta_ig", "beta_iw"],
    "gf": ["G(f_o)", "G(f_g)", "G(f_w)", "G(f_g_inj)", "G(f_w_inj)"],
    "gf_pos": ["G(f_o>0)", "G(f_g>0)", "G(f_w>0)", "G(f_g_inj>0)", "G(f_w_inj>0)"],
    "gf_neg": ["G(f_o<0)", "G(f_g<0)", "G(f_w<0)", "G(f_g_inj<0)", "G(f_w_inj<0)"],
    "con_data": ["dNp", "dGp", "dWp", "dGi", "dWi", "BHPp", "BHPg", "BHPw"],
    "stat_data": ["z", "por", "perm_i", "perm_j", "perm_k",
                  "log(perm_i)", "log(perm_j)", "log(perm_k)"],
    "con_wi": ["WIp", "WIg", "WIw"],
}
NODE_NAMES_GROUPED: Dict[str, List[str]] = {
    "init": ["Rs", "P"],
    "s": ["So", "So", "So"],
    "v_std": ["Vo_std", "Vg_std", "Vo_std"],
    "v_res": ["Vo_std", "Vo_std", "Vo_std"],
    "phi": ["P", "P", "P"],
    "por_vol": ["PorVol_ref", "PorVol_ref"],
    "beta": ["beta_o", "beta_o", "beta_o", "beta_o", "beta_o"],
    "gf": ["G(f_o)", "G(f_g)", "G(f_o)", "G(f_g)", "G(f_o)"],
    "gf_pos": ["G(f_o)", "G(f_g)", "G(f_o)", "G(f_g)", "G(f_o)"],
    "gf_neg": ["G(f_o)", "G(f_g)", "G(f_o)", "G(f_g)", "G(f_o)"],
    "con_data": ["dVo_std", "dVg_std", "dVo_std", "dVg_std", "dVo_std", "P", "P", "P"],
    "stat_data": ["z", "por", "perm_i", "perm_i", "perm_i",
                  "log(perm_i)", "log(perm_i)", "log(perm_i)"],
    "con_wi": ["WIp", "WIp", "WIp"],
}
DELTA_NAMES: Dict[str, List[str]] = {
    "dinit": ["dRs", "dP"],
    "ds": ["dSo", "dSg", "dSw"],
    "dv_std": ["dVo_std", "dVg_std", "dVw_std"],
    "dv_res": ["dVo_res", "dVg_free_res", "dVw_res"],
    "dphi": ["dPhi_o", "dPhi_g", "dPhi_w"],
}
DELTA_NAMES_GROUPED: Dict[str, List[str]] = {
    "dinit": ["dRs", "dP"],
    "ds": ["dSo", "dSo", "dSo"],
    "dv_std": ["dVo_std", "dVg_std", "dVo_std"],
    "dv_res": ["dVo_std", "dVo_std", "dVo_std"],
    "dphi": ["dP", "dP", "dP"],
}
EDGE_NAMES: Dict[str, List[str]] = {
    "f": ["fo", "fg_free", "fw", "fg_dissolved", "fg_g_inj", "fg_w_inj"],
    "f_pos": ["fo>0", "fg_free>0", "fw>0", "fg_dissolved>0", "fg_g_inj>0", "fg_w_inj>0"],
    "d_phi": ["delta_phi_o", "delta_phi_g", "delta_phi_w"],
    "tran": ["Transmissibility", "log(Transmissibility)"],
    "dist": ["edge_dist", "log(edge_dist)"],
    "dist_mod": ["edge_dist_mod", "log(edge_dist_mod)"],
}
EDGE_NAMES_GROUPED: Dict[str, List[str]] = {
    "f": ["fo", "fg_dissolved", "fo", "fg_dissolved", "fg_dissolved", "fo"],
    "f_pos": ["fo", "fg_dissolved", "fo", "fg_dissolved", "fg_dissolved", "fo"],
    "d_phi": ["delta_phi_o", "delta_phi_o", "delta_phi_o"],
    "tran": ["Transmissibility", "log(Transmissibility)"],
    "dist": ["edge_dist_mod", "log(edge_dist_mod)"],
    "dist_mod": ["edge_dist_mod", "log(edge_dist_mod)"],
}

# State and target channels: the state model's output space.
STATE_NAMES = ['Rs', 'P', 'So', 'Sg', 'Sw', 'Vo_std', 'Vg_std', 'Vw_std']
TARGET_NAMES = ['dRs', 'dP', 'dSo', 'dSg', 'dSw', 'dVo_std', 'dVg_std', 'dVw_std']
# Statistics used to normalise each target channel (dSg uses dSo's: dSg is ~0 here).
TARGET_NORM_NAMES = ['dRs', 'dP', 'dSo', 'dSo', 'dSw', 'dVo_std', 'dVg_std', 'dVw_std']
# Well-volume columns of `con_data_as_grid` (per node, per step).
CON_DELTA_NAMES = ["dNp", "dGp", "dWp", "dGi", "dWi"]
WELL_RATE_NAMES = CON_DELTA_NAMES
WELL_BHP_WI_NAMES = ["BHPp", "BHPg", "BHPw", "WIp", "WIg", "WIw"]


def _flatten(*dicts) -> List[str]:
    return [n for d in dicts for names in d.values() for n in names]


def all_feature_names() -> List[str]:
    """Every registry feature, in the order of the stored normalisation statistics."""
    return _flatten(NODE_NAMES, DELTA_NAMES, EDGE_NAMES)


def node_feature_names() -> List[str]:
    """Every node feature, in registry order."""
    return _flatten(NODE_NAMES)


def edge_feature_names() -> List[str]:
    """Every edge feature, in registry order."""
    return _flatten(EDGE_NAMES)


_ALL = all_feature_names()
_ALL_GROUPED = _flatten(NODE_NAMES_GROUPED, DELTA_NAMES_GROUPED, EDGE_NAMES_GROUPED)


def grouped_names(names: Sequence[str]) -> List[str]:
    """Representative feature (whose statistics are used) of each name."""
    out = []
    for name in names:
        try:
            out.append(_ALL_GROUPED[_ALL.index(name)])
        except ValueError as exc:
            raise ValueError(f"Unknown feature '{name}'.") from exc
    return out


# MARK: Wells
def scatter_to_nodes(values: torch.Tensor, index: torch.Tensor, n_nodes: int,
                     dim: int = 0) -> torch.Tensor:
    """Sum per-connection `values` onto their cells along `dim` (a cell with several
    connections of the same type gets their sum)."""
    shape = (*values.shape[:dim], n_nodes, *values.shape[dim + 1:])
    return values.new_zeros(shape).index_add_(dim, index, values)


def n_nodes_of(batch) -> int:
    """Number of cells of a sample."""
    return batch.stat.shape[-2]


def con_data_as_grid(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-cell well volumes of each step and BHP at its end.

    Returns (con_delta [..., N, 5] = dNp, dGp, dWp, dGi, dWi (step increments of the
    cumulative connection volumes), con_bhp [..., N, 3] = BHPp, BHPg, BHPw at the end of
    the step). Works on single-step (no time axis) and multi-step samples.
    """
    if getattr(batch, 'conp_y', None) is not None:
        pairs = [(batch.conp, batch.conp_y), (batch.cong, batch.cong_y), (batch.conw, batch.conw_y)]
    else:
        pairs = [(c[:-1], c[1:]) for c in (batch.conp, batch.cong, batch.conw)]
    single = pairs[0][0].ndim == 2
    n = n_nodes_of(batch)
    grids = []
    for (c0, c1), idx in zip(pairs, (batch.conp_index, batch.cong_index, batch.conw_index)):
        if single:
            c0, c1 = c0.unsqueeze(0), c1.unsqueeze(0)
        grids.append((scatter_to_nodes(c0, idx, n, dim=1), scatter_to_nodes(c1, idx, n, dim=1)))
    (p0, p1), (g0, g1), (w0, w1) = grids
    con_delta = torch.cat(((p1 - p0)[..., :3], (g1 - g0)[..., :1], (w1 - w0)[..., :1]), dim=-1)
    con_bhp = torch.stack((p1[..., -1], g1[..., -1], w1[..., -1]), dim=-1)
    if single:
        return con_delta[0], con_bhp[0]
    return con_delta, con_bhp


def con_wi_as_grid(batch) -> torch.Tensor:
    """Per-cell well index by type [N, 3] (WIp, WIg, WIw; static)."""
    n = n_nodes_of(batch)
    return torch.stack([scatter_to_nodes(stat[1], idx, n) for stat, idx in (
        (batch.conp_stat, batch.conp_index), (batch.cong_stat, batch.cong_index),
        (batch.conw_stat, batch.conw_index))], dim=-1)


def net_well_volumes(con_delta: torch.Tensor) -> torch.Tensor:
    """Net well contribution per cell (O, G, W) of a step: -dNp, dGi - dGp, dWi - dWp."""
    return torch.stack((-con_delta[..., 0], con_delta[..., 3] - con_delta[..., 1],
                        con_delta[..., 4] - con_delta[..., 2]), dim=-1)


def dynamic_well_type(con_delta: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
    """Per-cell well type from the step's well volumes: 0 none, 1 producer, 2 injector
    (producer wins if both)."""
    prod = (con_delta[..., :3].abs() > eps).any(dim=-1)
    inj = (con_delta[..., 3:5].abs() > eps).any(dim=-1)
    well_type = torch.zeros_like(prod, dtype=torch.long)
    well_type[inj] = 2
    well_type[prod] = 1
    return well_type


def well_node_mask(batch) -> torch.Tensor:
    """Cells with a well connection of any type."""
    mask = torch.zeros(n_nodes_of(batch), dtype=torch.bool, device=batch.stat.device)
    for attr in ('conp_index', 'cong_index', 'conw_index'):
        idx = getattr(batch, attr, None)
        if idx is not None and idx.numel() > 0:
            mask[idx] = True
    return mask


# MARK: Feature assembly
_NODE_GROUP_OF = {n: g for g, names in NODE_NAMES.items() for n in names}
_EDGE_GROUP_OF = {n: g for g, names in EDGE_NAMES.items() for n in names}


def _expand_static(t: torch.Tensor, lead: Tuple[int, ...]) -> torch.Tensor:
    return t.expand(*lead, *t.shape) if lead else t


def compute_features(batch, transform, node_names: Sequence[str], edge_names: Sequence[str],
                     x: Optional[torch.Tensor] = None,
                     con_delta_override: Optional[torch.Tensor] = None,
                     deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Node [..., N, len(node_names)] and edge [..., E, len(edge_names)] features.

    Args:
        batch: a sample (single step: x [N, 4]; or several steps: x [T, N, 4]).
        transform: the physics (`resflow.physics.Transform`).
        x: state (Rs, P, So, Sg) to use instead of batch.x.
        con_delta_override: per-cell well volumes of the step ([..., N, 5], layout of
            `con_data_as_grid`) replacing the sample's (e.g. a rate model's estimate);
            BHP stays the sample's.
        deterministic: sum the edge flows onto the cells with the exact, run-to-run
            reproducible (CPU) sum instead of the GPU scatter (not differentiable).
    """
    groups = {_NODE_GROUP_OF[n] for n in node_names} | {_EDGE_GROUP_OF[n] for n in edge_names}
    t = batch.x if x is None else x
    por_vol_0 = batch.stat[:, 1]
    con_delta, con_bhp = con_data_as_grid(batch)
    if con_delta_override is not None:
        if con_delta_override.shape != con_delta.shape:
            raise ValueError(f"con_delta_override has shape {tuple(con_delta_override.shape)}, "
                             f"expected {tuple(con_delta.shape)}.")
        con_delta = con_delta_override
    if t.shape[-1] != 4:
        raise ValueError(f"The state must have 4 columns (Rs, P, So, Sg), got {t.shape[-1]}.")
    if t.ndim == 3 and t.shape[0] == con_delta.shape[0] + 1:
        t = t[:-1]  # a stored series: T + 1 states, T steps of well data
    lead = tuple(t.shape[:-2])

    need_gf = bool(groups & {"gf", "gf_pos", "gf_neg"})
    need_f = need_gf or bool(groups & {"f", "f_pos"})
    need_phi = need_f or bool(groups & {"phi", "d_phi"})
    need_beta = need_f or "beta" in groups

    node = {"init": t[..., :2],
            "s": torch.cat((t[..., 2:4], (1.0 - t[..., 2] - t[..., 3]).unsqueeze(-1)), dim=-1)}
    edge = {}
    if "por_vol" in groups:
        vpor = transform.get_vpor(por_vol_0, t[..., 1])
        node["por_vol"] = torch.stack((por_vol_0.expand_as(vpor), vpor), dim=-1)
    pvt_table = None
    if need_phi or need_beta or "v_std" in groups:
        pvt_table = transform.pvt.get(t[..., 0], t[..., 1])
    if "v_std" in groups:
        node["v_std"] = transform.t2vols(por_vol_0, t, pvt_table=pvt_table)
    if "v_res" in groups:
        node["v_res"] = transform.t2vres(por_vol_0, t)
    beta = phi = None
    if need_beta:
        beta = transform.get_mobility(t, pvt_table, expanded=True)
        node["beta"] = beta
    if need_phi:
        phi = transform.get_phi(t, z=batch.stat[:, 0], pvt_table=pvt_table)
        node["phi"] = phi
    if need_f:
        f = transform.t2f(batch, t, beta=beta, phi=phi, pvt_table=pvt_table, inj_flow=True)
        edge["f"] = f
        f_pos = F.relu(f)
        edge["f_pos"] = f_pos
        if need_gf:
            gather = transform.gather_f_exact if deterministic else transform.gather_f
            if "gf" in groups:
                node["gf"] = gather(batch, f)
            if "gf_pos" in groups:
                node["gf_pos"] = gather(batch, f_pos)
            if "gf_neg" in groups:
                node["gf_neg"] = gather(batch, F.relu(-f))
    if "tran" in groups:
        tran = batch.edge_tran[..., :1]
        edge["tran"] = _expand_static(torch.cat((tran, torch.log(tran + 1e-8)), dim=-1), lead)
    for group, attr in (("dist", "edge_dist"), ("dist_mod", "edge_dist_mod")):
        if group in groups:
            d = getattr(batch, attr).unsqueeze(-1)
            edge[group] = _expand_static(torch.cat((d, torch.log(d + 1e-8)), dim=-1), lead)
    if "d_phi" in groups:
        edge["d_phi"] = (phi.index_select(-2, batch.edge_index[0])
                         - phi.index_select(-2, batch.edge_index[1]))
    if "stat_data" in groups:
        st = batch.stat[..., [0, 2, 3, 4, 5]]
        node["stat_data"] = _expand_static(torch.cat((st, torch.log(st[..., 2:] + 1e-8)), dim=-1),
                                           lead)
    if "con_wi" in groups:
        node["con_wi"] = _expand_static(con_wi_as_grid(batch), lead)
    node["con_data"] = torch.cat((con_delta, con_bhp), dim=-1)

    def _assemble(blocks, names, registry, group_of):
        cols = []
        for name in names:
            g = group_of[name]
            cols.append(blocks[g][..., registry[g].index(name)])
        return torch.stack(cols, dim=-1)

    return (_assemble(node, node_names, NODE_NAMES, _NODE_GROUP_OF),
            _assemble(edge, edge_names, EDGE_NAMES, _EDGE_GROUP_OF))


def delta_features(transform, batch, t: torch.Tensor) -> torch.Tensor:
    """Delta features (registry order) between consecutive states of a series t [T+1, N, 4]."""
    por_vol_0 = batch.stat[:, 1]
    full = torch.cat((t, (1.0 - t[..., 2] - t[..., 3]).unsqueeze(-1),
                      transform.t2vols(por_vol_0, t), transform.t2vres(por_vol_0, t),
                      transform.get_phi(t, z=batch.stat[:, 0])), dim=-1)
    return full[1:] - full[:-1]


def fit_values(batch, transform):
    """(node, delta, edge) features of every registry entry for a stored series (the
    columns fitted by `fit-norm`, in the order of `all_feature_names`)."""
    node, edge = compute_features(batch, transform, node_feature_names(), edge_feature_names())
    return node, delta_features(transform, batch, batch.x), edge


def state_with_volumes(transform, por_vol_0: torch.Tensor, t: torch.Tensor,
                       v: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[..., N, 8] = Rs, P, So, Sg, Sw, Vo_std, Vg_std, Vw_std of a state (volumes from
    `t2vols` unless given)."""
    if v is None:
        v = transform.t2vols(por_vol_0, t)
    sw = (1.0 - t[..., 2] - t[..., 3]).unsqueeze(-1)
    return torch.cat((t, sw, v), dim=-1)
