"""Black-oil PVT model (tables and equilibrium solver).

Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Union

import torch
from torch import nn

from .interp import alt_interp1d, interp2d, search_interval

EPS = 1e-10
P_INIT_BRACKET = 5.0  # kgf/cm2: half-bracket around p_init when none is given
MAX_BRACKET_ATTEMPTS = 10

# Buffers registered from the PVT dictionary (names are part of the saved state dicts).
PVT_KEYS = ("sat", "bo", "uo", "denoil", "dengas", "denwat",
            "cpor", "prpor", "dcpor", "refpw", "bwi", "cw", "vwi", "cvw")


class PVTModel(nn.Module):
    """Model that calculates PVT properties.

    Args:
        data: PVT tables (tensors) with the keys:
            - 'sat': saturated table (Pres = Psat). Columns: P (kgf/cm2), Rs (Sm3/Sm3),
              Bo (m3/Sm3), Eg=1/Bg (Sm3/m3), Uo (cP), Ug (cP).
            - 'bo', 'uo': undersaturated Bo and Uo tables (Pres > Psat).
            - 'denoil', 'dengas', 'denwat': densities (kg/m3).
            - 'cpor', 'prpor', 'dcpor': rock compressibility; 'refpw', 'bwi', 'cw':
              water Bw model; 'vwi', 'cvw': water viscosity model.
    """

    def __init__(self, data: dict):
        super().__init__()
        for p in PVT_KEYS:
            if p in data:
                self.register_buffer(p, data[p].contiguous())
            else:
                print(f"PVT import: property '{p}' not found in data")

        # Contiguous columns of the saturated table (avoid strided views in interpolation).
        if "sat" in data:
            self.register_buffer("_sat_p", self.sat[:, 0].clone())
            self.register_buffer("_sat_rs", self.sat[:, 1].clone())
            self.register_buffer("_sat_bo", self.sat[:, 2].clone())
            self.register_buffer("_sat_eg", self.sat[:, 3].clone())
            self.register_buffer("_sat_uo", self.sat[:, 4].clone())
            self.register_buffer("_sat_ug", self.sat[:, 5].clone())

            # Plain floats (no host sync when used as interpolation switch points).
            self._p_min_f = self.sat[0, 0].item()
            self._p_max_f = self.sat[-1, 0].item()
            self._rs_min_f = self.sat[0, 1].item()

        # Which table axes coincide, so get() can search each axis once (see _bo_uo_search).
        self._bo_psat_is_sat_p = False
        self._uo_axes_as_bo = False
        if "sat" in data and "bo" in data:
            bo_psat, bo_sub = self._undersat_axes(self.bo)
            self._bo_psat_is_sat_p = torch.equal(bo_psat, self._sat_p)
            if "uo" in data:
                uo_psat, uo_sub = self._undersat_axes(self.uo)
                self._uo_axes_as_bo = torch.equal(uo_psat, bo_psat) and (
                    (bo_sub is None and uo_sub is None)
                    or (bo_sub is not None and uo_sub is not None
                        and torch.equal(uo_sub, bo_sub)))

    @staticmethod
    def _undersat_axes(data):
        """(psat axis, sub-saturation axis or None) of an undersaturated Bo/Uo table."""
        if data.shape[0] == 3:
            return data[0], None
        return data[1:, 0].contiguous(), data[0, 2:]

    # MARK: Rock and water
    def get_por_mod(self, p):
        """Pore volume modifier: 1 + cpor (p - prpor) + dcpor (p - prpor)^2."""
        return 1 + self.cpor * (p - self.prpor) + self.dcpor * (p - self.prpor) ** 2

    def get_bw(self, p):
        """Water formation volume factor: bwi [1 - cw (p - refpw)]."""
        return self.bwi * (1 - self.cw * (p - self.refpw))

    def get_uw(self, p):
        """Water viscosity: vwi + cvw (p - refpw)."""
        return self.vwi + self.cvw * (p - self.refpw)

    def get_rhow(self, p, bw=None):
        """Water density (kg/m3); `bw` is calculated from p if not given."""
        if bw is None:
            bw = self.get_bw(p)
        return self.denwat / bw

    # MARK: Gas
    def _sat_lookup(self, y, p, idx=None, x_inversion=None, extrapolate=True):
        """Interpolate column y of the saturated table at pressure p (1/y-linear below the
        minimum table pressure for the columns that need it)."""
        p_input = p.view(-1)
        out = alt_interp1d(
            x=self._sat_p, y=y, xp=p_input,
            x_inversion=self._p_min_f if x_inversion is None else x_inversion,
            inverse_smaller=True, extrapolate=extrapolate, idx=idx)
        return out.view_as(p)

    def get_eg(self, p: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """Gas expansion factor (=1/Bg) at pressure p, from the saturated table.

        `idx` is an optional precomputed search_interval(_sat_p, p.view(-1)).
        """
        return self._sat_lookup(self._sat_eg, p, idx)

    def get_ug(self, p: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """Gas viscosity (cP) at pressure p, from the saturated table."""
        return self._sat_lookup(self._sat_ug, p, idx)

    def get_rhog(self, p: torch.Tensor) -> torch.Tensor:
        """Gas density (kg/m3) at pressure p."""
        return self.dengas * self.get_eg(p)

    # MARK: Psat, Rs
    def get_psat(self, rs: torch.Tensor) -> torch.Tensor:
        """Saturation pressure for the solution gas-oil ratio rs."""
        psat = alt_interp1d(
            x=self._sat_rs, y=self._sat_p, xp=rs.view(-1),
            x_inversion=self._rs_min_f, inverse_smaller=True, extrapolate=True)
        return psat.view_as(rs)

    def get_rs(self, psat: torch.Tensor) -> torch.Tensor:
        """Solution gas-oil ratio at saturation pressure psat."""
        return self._sat_lookup(self._sat_rs, psat)

    # MARK: Oil saturated
    def get_bo_sat(self, psat: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """Oil formation volume factor at saturation pressure psat (saturated table)."""
        return self._sat_lookup(self._sat_bo, psat, idx)

    def get_uo_sat(self, psat: torch.Tensor, idx: torch.Tensor = None) -> torch.Tensor:
        """Oil viscosity (cP) at saturation pressure psat (saturated table)."""
        return self._sat_lookup(self._sat_uo, psat, idx,
                                x_inversion=self._p_max_f, extrapolate=False)

    # MARK: Oil undersaturated
    def _get_bo_uo(self, p, rs, col_name, data, psat=None, search=None):
        """Bo or Uo for any condition (undersaturated correction over the saturated value).

        If data.shape[0] == 3 the correction is 1D, beta=f(psat); otherwise 2D,
        beta=f(psat, p - psat).

        search: optional dict of precomputed search_interval results ('sat': psat on the
        saturated table, 'psat': psat on this table's psat axis, 'sub': p - psat on its
        sub-saturation axis), as built by _bo_uo_search. Missing entries are computed here.
        """
        orig_shape = p.shape
        if psat is None:
            psat = self.get_psat(rs)
        if search is None:
            search_bo, search_uo = self._bo_uo_search(p, psat)
            search = search_bo if col_name.lower() == "bo" else search_uo
        p = p.reshape(-1)
        rs = rs.reshape(-1)
        psat = psat.reshape(-1)
        sub_sat = p - psat

        if col_name.lower() == "bo":
            vsat = self.get_bo_sat(psat, idx=search.get("sat"))
        elif col_name.lower() == "uo":
            vsat = self.get_uo_sat(psat, idx=search.get("sat"))
        else:
            raise ValueError(f"Unknown column name: {col_name}. Expected 'bo' or 'uo'.")

        psat_table, sub_sat_table = self._undersat_axes(data)
        idx_psat = search.get("psat")
        if idx_psat is None:
            idx_psat = search_interval(psat_table, psat)
        if data.shape[0] == 3:
            alphas = data[1]
            betas = data[2]
            alpha = alt_interp1d(psat_table, alphas, psat, extrapolate=False, idx=idx_psat)
            beta = alt_interp1d(psat_table, betas, psat, extrapolate=False, idx=idx_psat)
        else:
            alphas = data[1:, 1]
            betas = data[1:, 2:]

            alpha = alt_interp1d(psat_table, alphas, psat, extrapolate=False, idx=idx_psat)

            beta = interp2d(
                x1=psat_table,
                x2=sub_sat_table,
                y=betas,
                xp=torch.stack((psat, sub_sat), dim=1),
                idx1=idx_psat,
                idx2=search.get("sub"),
            )

        s = alpha * sub_sat * beta + 1
        s = torch.where(s < EPS, -EPS / sub_sat, s)  # avoid negative values inside the power
        out = vsat * torch.pow(s, 1 / alpha)
        return out.reshape(orig_shape)

    def get_bo(self, rs: torch.Tensor, p: torch.Tensor,
               psat: Union[None, torch.Tensor] = None,
               _search: Union[None, dict] = None) -> torch.Tensor:
        """Oil formation volume factor (m3/Sm3). `psat` is calculated from rs if not given."""
        bo = self._get_bo_uo(p, rs, "BO", self.bo, psat, search=_search)
        return bo.view_as(rs)

    def get_uo(self, rs: torch.Tensor, p: torch.Tensor,
               psat: Union[None, torch.Tensor] = None,
               _search: Union[None, dict] = None) -> torch.Tensor:
        """Oil viscosity (cP). `psat` is calculated from rs if not given."""
        uo = self._get_bo_uo(p, rs, "UO", self.uo, psat, search=_search)
        return uo.view_as(rs)

    def get_rhoo(self, rs: torch.Tensor, p: torch.Tensor,
                 bo: Union[None, torch.Tensor] = None,
                 psat: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """Oil density (kg/m3). `bo` (and `psat`) are calculated if not given."""
        if bo is None:
            if psat is None:
                psat = self.get_psat(rs)
            bo = self.get_bo(rs, p, psat)
        return (rs * self.dengas + self.denoil) / bo

    # MARK: Equilibrium
    def get_rs_psat(self, vo_std, vg_std, p, rs=None, psat=None, is_sat=None):
        """Solution gas-oil ratio and saturation pressure from standard volumes, at pressure p.

        Equilibrium is assumed. Returns a tensor [..., 2] with (rs, psat).
        """
        if rs is None:
            rs = vg_std / (vo_std + EPS)

        if is_sat is None:
            psat = self.get_psat(rs=rs)
            is_sat = psat > p

        if is_sat.any():
            rs = torch.where(is_sat, self.get_rs(psat=p), rs)
            psat = torch.where(is_sat, p, psat)

        return torch.stack((rs, psat), dim=-1)

    def get_vol_error(self, vo_std, vg_std, vw_std, vpor_ref, p, rs=None, psat=None,
                      is_sat=None, has_oil=None, rs_const=None, psat_const=None):
        """Volume error at pressure p: (vo + vg + vw) / vpor - 1 (reservoir volumes).

        If `is_sat` is given the saturation condition is assumed; if `has_oil` is given,
        cells without oil skip the Bo evaluation; if `rs_const`, `psat_const` and `is_sat`
        are given, the optimized path (no get_rs_psat call) is used.
        """
        if has_oil is None:
            has_oil = vo_std > 1e-10

        if rs_const is not None and psat_const is not None and is_sat is not None:
            # Saturated cells: rs = get_rs(psat=p), psat = p. Undersaturated: constants.
            if is_sat.any():
                if is_sat.all():
                    rs = self.get_rs(psat=p)
                    psat = p
                else:
                    rs = torch.where(is_sat, self.get_rs(psat=p), rs_const)
                    psat = torch.where(is_sat, p, psat_const)
            else:
                rs = rs_const
                psat = psat_const
        else:
            rs_psat = self.get_rs_psat(vo_std, vg_std, p, rs=rs, psat=psat, is_sat=is_sat)
            rs = rs_psat[..., 0]
            psat = rs_psat[..., 1]

        vpor = vpor_ref * self.get_por_mod(p=p)
        if has_oil.any():
            vo = vo_std * self.get_bo(p=p, rs=rs, psat=psat)
        else:
            vo = torch.zeros_like(vo_std)
        eg = self.get_eg(p=p)
        vg = (vg_std - rs * vo_std) / eg
        vw = vw_std * self.get_bw(p=p)
        v_total = vo + vg + vw
        return v_total / vpor - 1

    def find_equilibrium(self, vo_std, vg_std, vw_std, vpor_ref, max_iter=50, tol=1e-3,
                         p_init=None, p_init_bracket=None):
        """Equilibrium pressure from standard volumes, with Ridder's method.

        The saturation pressure is tested first, to tell whether the system is saturated;
        Ridder's method then solves the volume balance in the corresponding pressure range.

        Args:
            vo_std, vg_std, vw_std: standard oil, gas and water volumes.
            vpor_ref: reference pore volume.
            max_iter: maximum number of Ridder iterations.
            tol: tolerance on the pressure interval width.
            p_init: optional initial pressure guess (tightens the first bracket).
            p_init_bracket: half-bracket around p_init (float, or tensor per sample).
                Default P_INIT_BRACKET.

        Returns:
            Equilibrium pressure, same shape as the volumes.
        """
        # Cells with negligible oil are always undersaturated (rs is irrelevant for them).
        has_oil = vo_std > 1e-10

        rs = torch.where(
            has_oil, vg_std / (vo_std + EPS), torch.full_like(vo_std, self._rs_min_f))

        rs_limits = self.get_rs_limits()
        rs = rs.clamp(min=rs_limits[0], max=rs_limits[1])
        psat = self.get_psat(rs=rs)

        error_ = self.get_vol_error(
            vo_std=vo_std, vg_std=vg_std, vw_std=vw_std, vpor_ref=vpor_ref,
            p=psat, rs=rs, psat=psat,
            is_sat=torch.zeros_like(psat, dtype=torch.bool), has_oil=has_oil)
        is_sat = (error_ < 0) & has_oil

        # Constant terms of the volume error (avoid repeated get_rs_psat calls).
        rs_const = rs
        psat_const = psat

        def f(p):
            return self.get_vol_error(
                vo_std=vo_std, vg_std=vg_std, vw_std=vw_std, vpor_ref=vpor_ref, p=p,
                is_sat=is_sat, has_oil=has_oil, rs_const=rs_const, psat_const=psat_const)

        # Initial bracket bounds
        p_low = torch.where(is_sat, self._sat_p[0], psat)
        p_high = torch.where(is_sat, psat, self._sat_p[-1])

        if p_init is None:
            p_center = 0.5 * (p_low + p_high)
        else:
            p_center = torch.clamp(p_init, min=p_low, max=p_high)

        if p_init_bracket is None:
            bracket_half = torch.full_like(p_center, P_INIT_BRACKET)
        elif isinstance(p_init_bracket, (float, int)):
            bracket_half = torch.full_like(p_center, float(p_init_bracket))
        else:
            bracket_half = p_init_bracket.to(device=p_center.device, dtype=p_center.dtype)
            if bracket_half.shape != p_center.shape:
                bracket_half = torch.broadcast_to(bracket_half, p_center.shape)
        bracket_half = torch.clamp(bracket_half, min=tol)

        f_center = f(p_center)

        x_left = torch.clamp(p_center - bracket_half, min=p_low, max=p_high)
        x_right = torch.clamp(p_center + bracket_half, min=p_low, max=p_high)
        x_right = torch.where(
            x_right <= x_left, torch.minimum(p_high, x_left + bracket_half), x_right)

        f_left = f(x_left)
        f_right = f(x_right)

        # Use p_center if it brackets the root better than x_left or x_right
        x_left = torch.where(
            f_center * f_left > 0,
            torch.where(f_left.abs() < f_center.abs(), x_left, p_center),
            x_left)
        f_left = torch.where(
            f_center * f_left > 0,
            torch.where(f_left.abs() < f_center.abs(), f_left, f_center),
            f_left)
        x_right = torch.where(
            f_center * f_right > 0,
            torch.where(f_right.abs() < f_center.abs(), x_right, p_center),
            x_right)
        f_right = torch.where(
            f_center * f_right > 0,
            torch.where(f_right.abs() < f_center.abs(), f_right, f_center),
            f_right)

        # The interval must bracket the root (opposite signs); expand it if it does not.
        for _ in range(MAX_BRACKET_ATTEMPTS):
            needs_expansion = (f_left * f_right) > 0
            if not needs_expansion.any():
                break

            x_left_mod = torch.where(f_left.abs() < f_right.abs(), x_left / 2.0, x_left)
            x_right_mod = torch.where(f_left.abs() > f_right.abs(), x_right * 2.0, x_right)
            x_left = torch.where(needs_expansion, x_left_mod, x_left)
            x_right = torch.where(needs_expansion, x_right_mod, x_right)

            f_left = torch.where(needs_expansion, f(x_left), f_left)
            f_right = torch.where(needs_expansion, f(x_right), f_right)

        # Samples that still do not bracket the root use the best point (smallest |f|).
        product = f_left * f_right
        if (product > 0).any():
            x_bad = torch.where(f_left.abs() < f_right.abs(), x_left, x_right)
            f_bad = torch.where(f_left.abs() < f_right.abs(), f_left, f_right)
            x_left = torch.where(product > 0, x_bad, x_left)
            x_right = torch.where(product > 0, x_bad, x_right)
            f_left = torch.where(product > 0, f_bad, f_left)
            f_right = torch.where(product > 0, f_bad, f_right)

        # Ridder's method
        converged = torch.zeros_like(x_left, dtype=torch.bool)
        for _ in range(max_iter):
            converged = converged | (torch.abs(x_right - x_left) < tol)

            if converged.all():
                return torch.where(torch.abs(f_left) < torch.abs(f_right), x_left, x_right)

            active = ~converged
            x_mid = 0.5 * (x_left + x_right)
            f_mid = f(x_mid)
            discriminant = f_mid ** 2 - f_left * f_right
            s = torch.sqrt(torch.clamp(discriminant, min=0) + EPS)

            sign = torch.sign(f_left - f_right)
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            x_ridder = x_mid + (x_mid - x_left) * sign * f_mid / s

            # Bisection if the discriminant was bad or x_ridder is out of bounds
            out_of_bounds = (x_ridder <= x_left) | (x_ridder >= x_right)
            use_bisection = (discriminant <= 0) | out_of_bounds

            x_new = torch.where(use_bisection, x_mid, x_ridder)
            if use_bisection.all():
                f_new = f_mid
            elif (~use_bisection).all():
                f_new = f(x_ridder)
            else:
                f_new = torch.where(use_bisection, f_mid, f(x_ridder))

            use_left = (f_new * f_left) > 0
            use_left_active = use_left & active
            use_right_active = (~use_left) & active
            x_left = torch.where(use_left_active, x_new, x_left)
            f_left = torch.where(use_left_active, f_new, f_left)
            x_right = torch.where(use_right_active, x_new, x_right)
            f_right = torch.where(use_right_active, f_new, f_right)

        return torch.where(torch.abs(f_left) < torch.abs(f_right), x_left, x_right)

    # MARK: Limits
    def get_rs_limits(self) -> torch.Tensor:
        """Rs limits of the saturated table."""
        return torch.stack([self.sat[0, 1], self.sat[-1, 1]], dim=0)

    # MARK: All
    def get(self, rs: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """PVT properties for 1D rs and p, as [..., 6]: Bo (m3/Sm3), Eg (Sm3/m3), Uo (cP),
        Ug (cP), Rhoo (kg/m3), Rhog (kg/m3)."""
        psat = self.get_psat(rs)
        # Eg and Ug are both looked up at p on the saturated table: search it once.
        idx_p = search_interval(self._sat_p, p.reshape(-1))
        eg = self.get_eg(p, idx=idx_p)
        ug = self.get_ug(p, idx=idx_p)
        search_bo, search_uo = self._bo_uo_search(p, psat)
        bo = self.get_bo(rs, p, psat, _search=search_bo)
        uo = self.get_uo(rs, p, psat, _search=search_uo)
        rhog = self.dengas * eg
        rhoo = self.get_rhoo(rs, p, bo, psat)
        return torch.stack((bo, eg, uo, ug, rhoo, rhog), dim=-1)

    def _bo_uo_search(self, p, psat):
        """Table searches shared by the Bo and Uo lookups (see _get_bo_uo's `search`).

        psat is searched on the saturated table once for Bo_sat and Uo_sat, and on the
        undersaturated axes once when they coincide with it or between the Bo and Uo
        tables (the usual layout, checked in __init__).
        """
        psat_flat = psat.reshape(-1)
        idx_sat = search_interval(self._sat_p, psat_flat)
        search_bo = {"sat": idx_sat}
        search_uo = {"sat": idx_sat}
        if self._bo_psat_is_sat_p:
            search_bo["psat"] = idx_sat
        if self._uo_axes_as_bo:
            if "psat" not in search_bo:
                search_bo["psat"] = search_interval(self._undersat_axes(self.bo)[0], psat_flat)
            search_uo["psat"] = search_bo["psat"]
            sub_sat_table = self._undersat_axes(self.bo)[1]
            if sub_sat_table is not None:
                sub_sat = p.reshape(-1) - psat_flat
                search_bo["sub"] = search_uo["sub"] = search_interval(sub_sat_table, sub_sat)
        return search_bo, search_uo

    def forward(self, x):
        """x: [..., 2] with (rs, p). Returns the output of `get`."""
        return self.get(x[..., 0], x[..., 1])
