"""State <-> standard volume transformations, potentials, mobilities and edge flows.

Copyright (c) Tiago Amorim. MIT License.

Cell state tensors ``t`` hold (rs, p, so, sg) in the last dimension. Standard-volume
tensors hold (vo, vg, vw) (the "vols" of the method names), or (rs, p, vo, vg, vw).

Units follow the raw data (IMEX modified SI): pressure kgf/cm2, volumes m3 at standard
conditions, depth m, viscosity cP. `G` converts a density times a depth difference
(kg/m3 * m) into kgf/cm2: g[m/s2] * (kgf/cm2 / Pa) = 9.80665 / 98066.5.
"""
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from .krel import KRELModel
from .pvt import PVTModel

EPS = 1e-6  # Small value to avoid division by zero
G = 0.0001  # m/s2 * (kgf/cm2 / Pa)
K = 8.362147e-03  # Flow conversion to m3/d, from (mD.m * kgf/cm2 / cP)

PVT_FILE = "pvt{:02d}.pth"
KREL_WO_FILE = "krel{:02d}.wo.pth"
KREL_GO_FILE = "krel{:02d}.go.pth"


def find_model_folder(root: Union[str, Path]) -> Path:
    """First folder (in sorted order) at or under `root` that holds a ``model.json``.

    `root` may be a dataset root, a split folder, a ``raw`` folder or a single raw model folder.
    """
    root = Path(root)
    if (root / "model.json").is_file():
        return root
    if root.is_dir():
        for path in sorted(root.rglob("model.json")):
            return path.parent
    raise FileNotFoundError(f"Could not find 'model.json' under '{root}'.")


def load_pvt_krel(folder: Union[str, Path], table: int = 1):
    """Read the PVT dictionary and the oil-water / gas-oil relative permeability tables
    written by the converters.

    One naming convention: ``pvtNN.pth`` holds the PVT dictionary, and the relative
    permeability of table NN is split in ``krelNN.wo.pth`` (Sw, Krw, Krow) and
    ``krelNN.go.pth`` (Sg, Krg, Krog). There is no ``krelNN.pth``.
    """
    folder = Path(folder)
    pvt_data = torch.load(folder / PVT_FILE.format(table), weights_only=True)
    wo_krel = torch.load(folder / KREL_WO_FILE.format(table), weights_only=True)
    go_krel = torch.load(folder / KREL_GO_FILE.format(table), weights_only=True)
    return pvt_data, wo_krel, go_krel


class Transform(nn.Module):
    """Transforms simulation data using the PVT and relative permeability of a model.

    Args:
        pvt_data: PVT dictionary of tensors (see PVTModel).
        wo_krel: (n, 3) oil-water table (Sw, Krw, Krow).
        go_krel: (n, 3) gas-oil table (Sg, Krg, Krog).
        g, z_mean: G and the depth reference of the potentials (parameters, never read from a
            dataset; D2 stores them in the model config / weights bundle).
        vols2p_nn: optional pressure estimator (see `resflow.physics.vols2p`). When present
            it gives the initial pressure of `vols2p` / `vols2t`, which then use 5 Ridder
            iterations instead of 100.

    Build it from a raw model folder with `Transform.from_folder`. When saved weights are
    loaded, the PVT, relative permeability and estimator tensors are overwritten by the
    weights file (they are part of the state dict), so the folder contents do not matter.
    G is a constant (``G``, non persistent) and the depth reference of the potentials is
    the mean depth of each graph unless `set_z_mean` is called.
    """

    def __init__(self, pvt_data: dict, wo_krel: torch.Tensor, go_krel: torch.Tensor,
                 vols2p_nn: Optional[nn.Module] = None, g: float = G,
                 z_mean: Optional[float] = None) -> None:
        super().__init__()
        # Non-persistent buffers: they follow the module across devices but are not part
        # of the state dict (G and z_mean are constants of the method, not of a training).
        self.register_buffer('_g', torch.tensor(g), persistent=False)
        self.register_buffer('_z_mean', None if z_mean is None else torch.tensor(z_mean),
                             persistent=False)
        self.add_module('pvt', PVTModel(pvt_data))
        self.add_module('krel', KRELModel(wo_krel, go_krel))
        self.add_module('vols2p_nn', vols2p_nn)

    @classmethod
    def from_folder(cls, folder: Union[str, Path], table: int = 1,
                    vols2p_nn: Optional[nn.Module] = None, g: float = G,
                    z_mean: Optional[float] = None) -> "Transform":
        """Transform from the ``pvtNN.pth`` / ``krelNN.{wo,go}.pth`` files of a raw model
        folder (or of the first one under it, see find_model_folder)."""
        pvt_data, wo_krel, go_krel = load_pvt_krel(find_model_folder(folder), table)
        return cls(pvt_data, wo_krel, go_krel, vols2p_nn=vols2p_nn, g=g, z_mean=z_mean)

    @property
    def device(self) -> torch.device:
        """Device of the module."""
        return self._g.device

    # MARK: G and z_mean
    def find_optimal_g(self, g, z_ref=None):
        """Fit G on a graph `g` in equilibrium (day 0 states) and set it.

        Minimises the squared residuals of phi = p - (z - z_ref) * rhoo * G around its mean,
        over the cells with so > 0.1. Only needed to calibrate a new dataset: the shipped
        models use the default G.

        Args:
            g: graph with `x` [n, 4] (rs, p, so, sg) and `stat` [n, >=1] (depth in column 0).
            z_ref: reference depth; the mean depth of the used cells if None.

        Returns:
            The fitted G (not set if it is NaN or infinite).
        """
        sel = g.x[:, 2] > 0.1
        rs = g.x[sel][:, 0]
        p = g.x[sel][:, 1]
        z = g.stat[sel][:, 0]
        rhoo = self.pvt.get(rs, p)[:, 4]

        if z_ref is None:
            z_ref = z.mean()
        beta = (z - z_ref) * rhoo
        p_prime = p - p.mean()
        beta_prime = beta - beta.mean()
        g_opt = torch.sum(p_prime * beta_prime) / torch.sum(beta_prime ** 2)
        if torch.isnan(g_opt) or torch.isinf(g_opt):
            return g_opt
        self._g = g_opt
        return g_opt

    def get_g(self):
        """Current G."""
        return self._g

    def set_g(self, g: Union[float, torch.Tensor]):
        """Set G."""
        if isinstance(g, float):
            g = torch.tensor(g, device=self.device)
        self._g = g

    def get_z_mean(self):
        """Current depth reference (None: the mean depth of each graph is used)."""
        return self._z_mean

    def set_z_mean(self, z_mean: Union[float, torch.Tensor]):
        """Set the depth reference of the potentials."""
        if isinstance(z_mean, float):
            self._z_mean = torch.tensor(z_mean, device=self.device)
        else:
            self._z_mean = z_mean

    # MARK: Node calculations
    def get_vpor(self, vpor_0: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Cell pore volume at pressure p: vpor_0 * (1 + cpor (p - prpor) + dcpor (p - prpor)^2).

        vpor_0 is (n,); p is (n,) or (m, n).
        """
        assert vpor_0.ndim == 1, "vpor_0 must be a 1D tensor"
        assert p.ndim in [1, 2], "p must be a 1D or 2D tensor"

        por_mod = self.pvt.get_por_mod(p)
        if por_mod.ndim == 2:
            return vpor_0.unsqueeze(0) * por_mod
        return vpor_0 * por_mod

    def t2vres(self, vpor_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Reservoir-condition volumes per cell: (..., n, 3) with vo_res, vg_res, vw_res."""
        vpor = self.get_vpor(vpor_0, t[..., 1])

        vo = vpor * t[..., 2]
        vg = vpor * t[..., 3]
        vw = vpor * (1 - t[..., 2] - t[..., 3])
        return torch.stack((vo, vg, vw), dim=-1)

    def t2vols(self, vpor_0: torch.Tensor, t: torch.Tensor,
               pvt_table: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """Standard volumes per cell: (..., n, 3) with vo, vg, vw.

        vo and vw are the oil and water standard volumes; vg includes the dissolved gas
        (free gas + rs * vo).

        Args:
            vpor_0: (n,) reference pore volume (m3).
            t: (..., n, 4) rs, p, so, sg.
            pvt_table: optional (..., n, >=2) PVT table (bo, eg, ...) of the cells, as
                given by `pvt.get`; calculated if None.
        """
        if pvt_table is None:
            psat = self.pvt.get_psat(t[..., 0])
            eg = self.pvt.get_eg(t[..., 1])
            bo = self.pvt.get_bo(t[..., 0], t[..., 1], psat)
        else:
            bo = pvt_table[..., 0]
            eg = pvt_table[..., 1]
        bw = self.pvt.get_bw(t[..., 1])

        v_res = self.t2vres(vpor_0, t)

        vo = v_res[..., 0] / torch.clamp(bo, min=EPS)
        vg = v_res[..., 1] * eg + vo * t[..., 0]
        vw = v_res[..., 2] / torch.clamp(bw, min=EPS)
        return torch.stack((vo, vg, vw), dim=-1)

    def t2v(self, vpor_0: torch.Tensor, t: torch.Tensor,
            pvt_table: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """Like t2vols, with rs and p kept: (..., n, 5) with rs, p, vo, vg, vw."""
        vols = self.t2vols(vpor_0, t, pvt_table)
        return torch.stack((t[..., 0], t[..., 1], vols[..., 0], vols[..., 1], vols[..., 2]), dim=-1)

    def vols2p_quick(self, vpor_0: torch.Tensor, vols: torch.Tensor, clip: bool = True):
        """Pressure estimate from the volumes with the neural estimator `vols2p_nn`.

        `vols` is (..., n, 3) (vo, vg, vw) or (..., n, 5) (rs, p, vo, vg, vw); the result
        is clipped to the PVT table pressure range when `clip` is True.
        """
        if self.vols2p_nn is None:
            raise ValueError("vols2p_nn model is not available in this Transform instance.")

        i = 0 if (vols.shape[-1] == 3) else 2
        vol_input = vols[..., i:i + 3]

        original_shape = vol_input.shape[:-1]
        if vol_input.ndim == 3:
            vol_input = vol_input / vpor_0.unsqueeze(0).unsqueeze(-1)
        else:
            vol_input = vol_input / vpor_0.unsqueeze(-1)

        with torch.no_grad():
            p = self.vols2p_nn(vol_input.reshape(-1, 3))
        p = p.view(*original_shape)
        if clip:
            p = torch.clamp(p, min=self.pvt.sat[0, 0], max=self.pvt.sat[-1, 0])
        return p

    def vols2p(self, vpor_0: torch.Tensor, vols: torch.Tensor, max_iter=None, tol=1e-5,
               p_init=None, p_init_bracket=None) -> torch.Tensor:
        """Equilibrium pressure from the standard volumes (equilibrium assumed).

        Args:
            vpor_0: (n,) reference pore volume (m3).
            vols: (..., n, 3) vo, vg, vw or (..., n, 5) rs, p, vo, vg, vw (rs, p ignored).
            max_iter: Ridder iterations. None resolves to 5 when `vols2p_nn` is available
                (its estimate warm-starts the solve) and to 100 otherwise.
            tol: tolerance on the pressure interval width.
            p_init: (..., n) initial pressure; from `vols2p_nn` when None and available.
            p_init_bracket: half-bracket around p_init (default 5 kgf/cm2 with the estimator).
        """
        if p_init is None and self.vols2p_nn is not None:
            p_init = self.vols2p_quick(vpor_0, vols)
            if max_iter is None:
                max_iter = 5
            if p_init_bracket is None:
                p_init_bracket = 5.0
        elif max_iter is None:
            max_iter = 100

        i = 0 if (vols.shape[-1] == 3) else 2
        return self.pvt.find_equilibrium(
            vols[..., i], vols[..., i + 1], vols[..., i + 2], vpor_0,
            max_iter=max_iter, tol=tol, p_init=p_init, p_init_bracket=p_init_bracket)

    def vols2t(self, vpor_0: torch.Tensor, vols: torch.Tensor, max_iter=None, tol=1e-5,
               p_init=None, p_init_bracket=None, quick=False) -> torch.Tensor:
        """State (rs, p, so, sg) from the standard volumes: (..., n, 4).

        Equilibrium is assumed. `quick=True` takes the pressure from `vols2p_nn` without
        refinement. Other arguments as in `vols2p`.
        """
        if quick:
            p = self.vols2p_quick(vpor_0, vols)
        else:
            p = self.vols2p(vpor_0, vols, max_iter=max_iter, tol=tol, p_init=p_init,
                            p_init_bracket=p_init_bracket)

        i = 0 if (vols.shape[-1] == 3) else 2
        vo = vols[..., i]
        vg = vols[..., i + 1]
        vw = vols[..., i + 2]

        rs_psat = self.pvt.get_rs_psat(vo, vg, p)
        rs = rs_psat[..., 0]
        psat = rs_psat[..., 1]

        eg = self.pvt.get_eg(p)
        bo = self.pvt.get_bo(rs, p, psat)
        bw = self.pvt.get_bw(p)

        vo_res = vo * bo
        vg_res = (vg - vo * rs) / torch.clamp(eg, min=EPS)
        vw_res = vw * bw

        vpor = vo_res + vg_res + vw_res
        so = vo_res / torch.clamp(vpor, min=EPS)
        sg = vg_res / torch.clamp(vpor, min=EPS)

        return torch.stack((rs, p, so, sg), dim=-1)

    # MARK: Flow calculations
    def get_phi(self, t: torch.Tensor, z: torch.Tensor,
                z_ref: Union[None, float, torch.Tensor] = None,
                pvt_table: Union[None, torch.Tensor] = None,
                rhow: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """Phase potentials per node: (..., n, 3) with phi_o, phi_g, phi_w.

        phi = p - (z - z_ref) * rho * G.

        Args:
            t: (..., n, 4) rs, p, so, sg.
            z: (n,) depth (positive downwards).
            z_ref: reference depth; the stored z_mean if set, else the mean of z.
            pvt_table: optional (..., n, 6) table from `pvt.get` (rhoo, rhog in columns 4, 5).
            rhow: optional water density; calculated from p if None.
        """
        if pvt_table is None:
            psat = self.pvt.get_psat(t[..., 0])
            bo = self.pvt.get_bo(t[..., 0], t[..., 1], psat)
            rhog = self.pvt.get_rhog(t[..., 1])
            rhoo = self.pvt.get_rhoo(t[..., 0], t[..., 1], bo, psat)
        else:
            rhoo = pvt_table[..., 4]
            rhog = pvt_table[..., 5]

        if rhow is None:
            bw = self.pvt.get_bw(t[..., 1])
            rhow = self.pvt.get_rhow(t[..., 1], bw)
        rho = torch.stack((rhoo, rhog, rhow), dim=-1)

        if z_ref is None:
            if self._z_mean is not None:
                z_ref = self._z_mean.to(z.device)
            else:
                z_ref = z.mean()
        z_rel = z - z_ref

        if rho.ndim == 3:
            z_rel = z_rel.unsqueeze(0)
        return t[..., 1].unsqueeze(-1) - z_rel.unsqueeze(-1) * rho * self._g

    def get_kr(self, t: torch.Tensor) -> torch.Tensor:
        """Relative permeabilities (..., n, 3): kro, krg, krw, clamped to [0, 1]."""
        sw = 1.0 - t[..., 2] - t[..., 3]
        sg = t[..., 3]
        return self.krel.get(sw, sg).clamp(min=0, max=1)

    def get_mobility(self, t: torch.Tensor, pvt_table: Union[None, torch.Tensor] = None,
                     expanded: bool = False) -> torch.Tensor:
        """Phase mobility beta = kr / (fvf * mu): (..., n, 3) with beta_o, beta_g, beta_w.

        With `expanded` the injection mobilities are appended: (..., n, 5) with
        beta_g_inj, beta_w_inj (total mobility over the injected phase volume factor).

        `pvt_table` is an optional (..., n, 6) table from `pvt.get` (bo, eg, uo, ug, ...).
        """
        if pvt_table is None:
            psat = self.pvt.get_psat(t[..., 0])
            bo = self.pvt.get_bo(t[..., 0], t[..., 1], psat)
            eg = self.pvt.get_eg(t[..., 1])
            uo = self.pvt.get_uo(t[..., 0], t[..., 1], psat)
            ug = self.pvt.get_ug(t[..., 1])
        else:
            bo = pvt_table[..., 0]
            eg = pvt_table[..., 1]
            uo = pvt_table[..., 2]
            ug = pvt_table[..., 3]

        bw = self.pvt.get_bw(t[..., 1])
        uw = self.pvt.get_uw(t[..., 1])
        kr = self.get_kr(t)

        lambda_o = kr[..., 0] / torch.clamp(uo, min=EPS)
        lambda_g = kr[..., 1] / torch.clamp(ug, min=EPS)
        lambda_w = kr[..., 2] / torch.clamp(uw, min=EPS)

        beta_o = lambda_o / torch.clamp(bo, min=EPS)
        beta_g = lambda_g * eg
        beta_w = lambda_w / torch.clamp(bw, min=EPS)

        if expanded:
            lambda_total = lambda_o + lambda_g + lambda_w
            beta_g_inj = lambda_total * eg
            beta_w_inj = lambda_total / torch.clamp(bw, min=EPS)
            return torch.stack((beta_o, beta_g, beta_w, beta_g_inj, beta_w_inj), dim=-1)

        return torch.stack((beta_o, beta_g, beta_w), dim=-1)

    def t2f(self, g, t: torch.Tensor, pvt_table: Union[None, torch.Tensor] = None,
            z_ref: Union[None, float] = None, phi: Union[None, torch.Tensor] = None,
            beta: Union[None, torch.Tensor] = None, inj_flow: bool = False) -> torch.Tensor:
        """Edge flows in standard volumes (upwind two-point flux, per edge of g.edge_index).

        The gas flow is returned as free and dissolved gas.

        Args:
            g: graph with `edge_index`, `edge_tran` and `stat` (depth in column 0).
            t: (..., n, 4) rs, p, so, sg.
            pvt_table, z_ref, phi, beta: optional precomputed inputs (see get_phi and
                get_mobility; beta needs 5 columns when `inj_flow`).
            inj_flow: append the flows with the injection mobilities.

        Returns:
            (..., m, 4) fo, fg_free, fw, fg_dissolved; or (..., m, 6) with fg_g_inj,
            fg_w_inj appended when `inj_flow`.
        """
        if pvt_table is None:
            pvt_table = self.pvt.get(t[..., 0], t[..., 1])

        if phi is None:
            phi = self.get_phi(t, z=g.stat[:, 0], z_ref=z_ref, pvt_table=pvt_table)
        edges = g.edge_index

        if phi.ndim == 3:
            dphi = phi[:, edges[0], :] - phi[:, edges[1], :]
        else:
            dphi = phi[edges[0]] - phi[edges[1]]

        if beta is None or (inj_flow and beta.shape[-1] < 5):
            beta = self.get_mobility(t, pvt_table, expanded=inj_flow)
        if inj_flow:
            dphi = torch.concat((dphi, dphi[..., 1:]), dim=-1)

        up_edges = torch.where(dphi > 0, edges[:1].T, edges[1:].T)
        beta_up = beta.gather(dim=-2, index=up_edges)
        flow = g.edge_tran[:, :1] * beta_up * dphi * K

        fg_diss = flow[..., 0] * t[..., 0].gather(dim=-1, index=up_edges[..., 0])
        flow_out = torch.concat((flow[..., 0:3], fg_diss.unsqueeze(-1)), dim=-1)

        if inj_flow:
            flow_out = torch.concat((flow_out, flow[..., -2:]), dim=-1)
        return flow_out

    @staticmethod
    def _combine_gas(f_sum: torch.Tensor) -> torch.Tensor:
        """Sum free and dissolved gas of per-node flows: (fo, fg_free, fw, fg_diss[, inj...])
        -> (fo, fg, fw[, inj...])."""
        n = f_sum.shape[-1]
        if n in (3, 5):
            return f_sum
        f_out = torch.stack(
            (f_sum[..., 0], f_sum[..., 1] + f_sum[..., 3], f_sum[..., 2]), dim=-1)
        if n == 4:
            return f_out
        return torch.concat((f_out, f_sum[..., -2:]), dim=-1)

    def gather_f(self, g, f: torch.Tensor) -> torch.Tensor:
        """Sum the edge flows `f` ((m, F) or (B, m, F)) on the receiving node (edge_index[1]).

        Differentiable (index_add); on GPU the summation order is not deterministic.
        Gas flow combines free and dissolved gas: returns (n, 3) fo, fg, fw (plus the two
        injection columns for F = 6).
        """
        n_nodes = g.stat.shape[0]
        index = g.edge_index[1]
        if f.ndim == 2:
            f_sum = torch.zeros((n_nodes, f.shape[-1]), device=f.device, dtype=f.dtype)
            f_sum = f_sum.index_add(0, index, f)
        else:
            f_sum = torch.zeros((f.shape[0], n_nodes, f.shape[-1]), device=f.device, dtype=f.dtype)
            f_sum = f_sum.index_add(1, index, f)
        return self._combine_gas(f_sum)

    def gather_f_exact(self, g, f: torch.Tensor) -> torch.Tensor:
        """Like gather_f, with exact, deterministic sums (CPU, torch.bincount). Not differentiable.

        Batched inputs are summed step by step so that every result is bit-identical to its
        single-step equivalent.
        """
        if f.ndim == 3:
            return torch.stack([self.gather_f_exact(g, f[i]) for i in range(f.shape[0])], dim=0)

        n_nodes = g.stat.shape[0]
        edges_cpu = g.edge_index[1].cpu()
        f_cpu = f.cpu()
        f_sum_cpu = torch.zeros((n_nodes, f.shape[1]), dtype=f.dtype)
        for i in range(f.shape[1]):
            f_sum_cpu[:, i] = torch.bincount(edges_cpu, weights=f_cpu[:, i], minlength=n_nodes)
        return self._combine_gas(f_sum_cpu.to(f.device))
