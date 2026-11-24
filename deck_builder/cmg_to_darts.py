"""Run a CMG IMEX (MODSI, black-oil, corner-point) .dat deck in open-DARTS (TU Delft, `pip install open-darts`).

Same deck family and parsing as cmg_to_opm.py (single-layer Cartesian grid, BOT/VOT black oil, SWT relperm, BHP
wells changed by DATE / TARGET BHP). DARTS has no deck reader for this, so the deck is turned into a DARTS model
in Python and run in-process. See DARTS.md for the mapping, the checks and the comparison with OPM / IMEX.

Fluid: two-component dead oil (water, oil). Every producer BHP of these decks is above the bubble point (PB CON), so the
oil stays undersaturated at Rs(Pb) and its Bo and viscosity follow the undersaturated branch of the IMEX table at Pb.
The oil component carries its dissolved gas mass: oil density = (rho_o_sc + Rs * rho_g_sc) / Bo(p). Surface volumes:
oil = oil mass / (rho_o_sc + Rs rho_g_sc), gas = Rs x oil, water = water mass / rho_w_sc. The run stops if any BHP is
at or below Pb.

Usage:
    python cmg_to_darts.py path/to/case.dat [-o outdir] [--wi-scale 1.0] [--n-points 400]
Output (default <deck folder>_darts/<case>/): wells.csv (per well at every DATE of the deck, OPM summary names: WBHP bar,
WOPR/WWPR/WGPR/WWIR sm3/day, WOPT/WWPT/WGPT/WWIT sm3), summary.csv (final values, run time), stopped_wells.csv (wells
stopped because their flow reversed), run.log, output/well_data.h5 (DARTS well blocks) and, (unless --no-cells) output/static.h5 (grid, rock, perforations) and
output/reservoir_solution.h5 (pressure and water mass fraction of every cell at every DATE; cell_states() gives Sw).
"""
import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cmg_to_opm import (KGCM2_TO_BAR, find_include, all_includes, parse_corners, parse_kr, parse_pvt,
                        parse_wells_and_schedule, read_numbers, strip_comments)

DARCY_METRIC = 0.008527            # CF [cP.rm3/day/bar] = c * 2pi k[mD] h[m] / (ln(r0/rw) + s), as ECLIPSE / OPM
G_BAR_PER_M = 9.80665e-5           # bar per (kg/m3 * m)


# ---------------------------------------------------------------- deck reading
def _scalar(text, pattern):
    return float(re.search(pattern, text, re.I | re.M).group(1))


def cartesian_from_corners(X, Y, Z, nx, ny, nz, tol=1e-6):
    """Per-cell DX, DY, DZ and centre depth from CMG CORNERS; raises if a cell is not an axis-aligned box."""
    shape = (nz, 2, ny, 2, nx, 2)                  # k, top/bottom, j, south/north, i, west/east
    X, Y, Z = (a.reshape(shape) for a in (X, Y, Z))
    if not (np.ptp(X[..., 0], axis=(1, 3)).max() < tol and np.ptp(X[..., 1], axis=(1, 3)).max() < tol
            and np.ptp(Y[:, :, :, 0], axis=(1, 4)).max() < tol and np.ptp(Y[:, :, :, 1], axis=(1, 4)).max() < tol
            and np.ptp(Z[:, 0], axis=(2, 4)).max() < tol and np.ptp(Z[:, 1], axis=(2, 4)).max() < tol):
        raise ValueError("grid cells are not axis-aligned boxes")
    dx = X[:, 0, :, 0, :, 1] - X[:, 0, :, 0, :, 0]
    dy = Y[:, 0, :, 1, :, 0] - Y[:, 0, :, 0, :, 0]
    dz = Z[:, 1, :, 0, :, 0] - Z[:, 0, :, 0, :, 0]
    depth = Z.mean(axis=(1, 3, 5))
    if (dx <= 0).any() or (dy <= 0).any() or (dz <= 0).any():
        raise ValueError("non-positive cell size in CORNERS")
    if (nx > 1 and np.abs(X[:, 0, :, 0, 1:, 0] - X[:, 0, :, 0, :-1, 1]).max() > tol) or \
       (ny > 1 and np.abs(Y[:, 0, 1:, 0, :, 0] - Y[:, 0, :-1, 1, :, 0]).max() > tol):
        raise ValueError("gaps between neighbours")
    if (nx > 1 and np.abs(Z[:, 0, :, 0, 1:, 0] - Z[:, 0, :, 0, :-1, 1]).max() > tol) or \
       (ny > 1 and np.abs(Z[:, 0, 1:, 0, :, 0] - Z[:, 0, :-1, 1, :, 0]).max() > tol):
        raise ValueError("vertical throws between neighbours")
    return [a.reshape(-1) for a in (dx, dy, dz, depth)]       # i fastest, then j, then k


def peaceman_cf(kx, ky, dx, dy, h, rw, skin):
    """ECLIPSE/OPM Peaceman connection factor of a vertical well, cP.rm3/day/bar."""
    r0 = 0.28 * math.sqrt(math.sqrt(ky / kx) * dx ** 2 + math.sqrt(kx / ky) * dy ** 2) / ((ky / kx) ** 0.25 + (kx / ky) ** 0.25)
    return DARCY_METRIC * 2 * math.pi * math.sqrt(kx * ky) * h / (math.log(r0 / rw) + skin)


def undersaturated_oil(pv, pb):
    """(p [bar], Bo, mu [cP]) of the oil saturated at pb [kg/cm2] (BOT/VOT branch at that PVT row), and Rs(pb)."""
    P = pv["pvt"]
    i = np.flatnonzero(np.isclose(P[:, 0], pb))
    if not i.size or int(i[0]) >= len(pv["bot"]):
        raise ValueError(f"PB {pb} is not a PVT row with a BOT/VOT table")
    i = int(i[0])
    bt, vt = pv["bot"][i], pv["vot"][i]
    if not (np.isclose(bt[0, 0], pb) and np.isclose(bt[0, 1], P[i, 2]) and np.isclose(vt[0, 1], P[i, 4])):
        raise ValueError("BOT/VOT table does not start at the saturated point")
    return np.column_stack([bt[:, 0] * KGCM2_TO_BAR, bt[:, 1], vt[:, 1]]), P[i, 1]


def read_deck(dat_path):
    """Everything the DARTS model needs, in METRIC units (bar, m, mD, cP, kg/m3, days)."""
    dat_path = Path(dat_path).resolve()
    base = dat_path.parent
    text = strip_comments(dat_path.read_text())
    lines = text.splitlines()
    nx, ny, nz = map(int, re.search(r"^\s*CORNER\s+(\d+)\s+(\d+)\s+(\d+)", text, re.I | re.M).groups())
    if not re.search(r"^\s*VERTICAL\s+BLOCK_CENTER\s+WATER_OIL", text, re.I | re.M):
        raise ValueError("only VERTICAL BLOCK_CENTER WATER_OIL initialisation is converted")
    if re.search(r"^\s*TRANS[IJK]\b", text, re.I | re.M):
        raise ValueError("TRANSI/J/K multipliers are not converted")
    corner_line = next(i for i, l in enumerate(lines) if re.match(r"\s*CORNER\s", l, re.I))
    corners_inc = (base / re.match(r"\s*INCLUDE\s+'([^']+)'", lines[corner_line + 1], re.I).group(1)).resolve()
    d = dict(name=dat_path.stem, nx=nx, ny=ny, nz=nz)
    d["dx"], d["dy"], d["dz"], d["depth"] = cartesian_from_corners(*parse_corners(corners_inc, nx, ny, nz), nx, ny, nz)
    d["permx"] = read_numbers(find_include(lines, r"PERMI\s+ALL", base))
    d["permy"] = read_numbers(find_include(lines, r"PERMJ\s+ALL", base))
    assert d["permx"].size == nx * ny * nz and d["permy"].size == nx * ny * nz, "perm size mismatch"
    d["permz"] = _scalar(text, r"^\s*PERMK\s+CON\s+([-+0-9.eE]+)")
    d["poro"] = _scalar(text, r"^\s*POR\s+CON\s+([-+0-9.eE]+)")
    d["cpor"] = _scalar(text, r"^\s*CPOR\s+([-+0-9.eE]+)") / KGCM2_TO_BAR
    d["prpor"] = _scalar(text, r"^\s*PRPOR\s+([-+0-9.eE]+)") * KGCM2_TO_BAR
    d["refdepth"] = _scalar(text, r"^\s*REFDEPTH\s+([-+0-9.eE]+)")
    d["refpres"] = _scalar(text, r"^\s*REFPRES\s+([-+0-9.eE]+)") * KGCM2_TO_BAR
    d["dwoc"] = _scalar(text, r"^\s*DWOC\s+([-+0-9.eE]+)")
    pb = _scalar(text, r"^\s*PB\s+CON\s+([-+0-9.eE]+)")
    d["dtmax"] = _scalar(text, r"^\s*DTMAX\s+([-+0-9.eE]+)")
    d["start"], d["wells"], d["events"] = parse_wells_and_schedule(lines)
    lowest = min([w["bhp"] for w in d["wells"].values() if w["type"] == "PROD"]
                 + [b for _, k, w, b in d["events"] if k == "target" and d["wells"][w]["type"] == "PROD"], default=np.inf)
    if lowest <= pb:
        raise ValueError(f"a producer BHP ({lowest} kg/cm2) is at or below PB {pb}: free gas can form, not modelled")
    pvt_inc = next(p for p in all_includes(lines, base) if p.name.upper().startswith("PVT"))
    kr_inc = next(p for p in all_includes(lines, base) if p.name.upper().startswith("KR"))
    pv = parse_pvt(pvt_inc)
    d["oil_table"], d["rs"] = undersaturated_oil(pv, pb)
    d["pb"] = pb * KGCM2_TO_BAR
    d["rho_o_sc"], d["rho_w_sc"], d["rho_g_sc"] = pv["dens_o"], pv["dens_w"], pv["dens_g"]
    d["oil_mass_per_sm3"] = pv["dens_o"] + d["rs"] * pv["dens_g"]
    d["pvtw"] = (pv["refpw"] * KGCM2_TO_BAR, pv["bwi"], pv["cw"] / KGCM2_TO_BAR, pv["vwi"], pv["cvw"] / KGCM2_TO_BAR)
    swt, slt = parse_kr(kr_inc)
    if np.any(swt[:, 3] != 0):
        raise ValueError("water-oil capillary pressure is not converted")
    d["swof"] = swt[:, :3]
    return d


# ---------------------------------------------------------------- property evaluators (DARTS calls .evaluate)
class TableOil:
    """Oil of the undersaturated branch: density and viscosity vs pressure, linear interpolation (and extrapolation)."""
    def __init__(self, table, mass_per_sm3):
        self.p, self.bo, self.mu = table.T
        self.m = mass_per_sm3

    def _interp(self, p, y):
        if p <= self.p[0]:
            i = 0
        elif p >= self.p[-1]:
            i = len(self.p) - 2
        else:
            i = int(np.searchsorted(self.p, p)) - 1
        return y[i] + (p - self.p[i]) * (y[i + 1] - y[i]) / (self.p[i + 1] - self.p[i])

    def bo_at(self, p):
        return self._interp(p, self.bo)

    def mu_at(self, p):
        return self._interp(p, self.mu)


class OilDensity:
    def __init__(self, oil):
        self.oil = oil

    def evaluate(self, pressure, temperature=None, x=None):
        return self.oil.m / self.oil.bo_at(pressure)


class OilViscosity:
    def __init__(self, oil):
        self.oil = oil

    def evaluate(self, pressure, temperature=None, x=None, rho=None):
        return self.oil.mu_at(pressure)


class WaterDensity:
    """ECLIPSE PVTW: Bw = Bw_ref / (1 + X + X^2/2), X = cw (p - p_ref)."""
    def __init__(self, pvtw, rho_sc):
        self.pref, self.bw, self.cw = pvtw[:3]
        self.rho_sc = rho_sc

    def evaluate(self, pressure, temperature=None, x=None):
        X = self.cw * (pressure - self.pref)
        return self.rho_sc * (1 + X + X * X / 2) / self.bw


class WaterViscosity:
    def __init__(self, pvtw):
        self.pref, self.mu, self.cv = pvtw[0], pvtw[3], pvtw[4]

    def evaluate(self, pressure, temperature=None, x=None, rho=None):
        Y = -self.cv * (pressure - self.pref)
        return self.mu / (1 + Y + Y * Y / 2)


class TableRelPerm:
    """kr from the SWOF-like table (Sw, krw, krow); `phase` 'wat' gets Sw, 'oil' gets So (= 1 - Sw)."""
    def __init__(self, swof, phase):
        self.sw, self.krw, self.krow = swof.T
        self.phase = phase

    def evaluate(self, sat):
        if self.phase == "wat":
            return float(np.interp(sat, self.sw, self.krw))
        return float(np.interp(1.0 - sat, self.sw, self.krow))


# ---------------------------------------------------------------- DARTS model
def build_model(d, wi_scale=1.0, n_points=400, p_range=(100.0, 700.0), inj_mobility="total", dtmax=None,
                no_crossflow=True):
    from darts.models.darts_model import DartsModel
    from darts.physics.base.physics import PhysicsBase
    from darts.physics.dead_oil import DeadOilProperties
    from darts.physics.properties.basic import RockCompactionEvaluator
    from darts.reservoirs.struct_reservoir import StructReservoir

    oil = TableOil(d["oil_table"], d["oil_mass_per_sm3"])
    zero = 1e-12

    class Model(DartsModel):
        def __init__(self):
            super().__init__()
            self.timer.node["initialization"].start()
            nx, ny, nz = d["nx"], d["ny"], d["nz"]
            n = nx * ny * nz
            self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=d["dx"], dy=d["dy"], dz=d["dz"],
                                             permx=d["permx"], permy=d["permy"], permz=np.full(n, d["permz"]),
                                             poro=np.full(n, d["poro"]), depth=d["depth"])
            pc = DeadOilProperties(phases_name=["wat", "oil"], components_name=["w", "o"], Mw=np.ones(2),
                                   eps_z=zero, temperature=1.0)
            pc.density_ev = dict(wat=WaterDensity(d["pvtw"], d["rho_w_sc"]), oil=OilDensity(oil))
            pc.viscosity_ev = dict(wat=WaterViscosity(d["pvtw"]), oil=OilViscosity(oil))
            pc.rel_perm_ev = dict(wat=TableRelPerm(d["swof"], "wat"), oil=TableRelPerm(d["swof"], "oil"))
            pc.rock_compr_ev = RockCompactionEvaluator(pref=d["prpor"], compres=d["cpor"])
            self.physics = PhysicsBase(["w", "o"], ["wat", "oil"], self.timer,
                                       state_spec=PhysicsBase.StateSpecification.P,
                                       axes_step=[(p_range[1] - p_range[0]) / n_points, 1.0 / n_points],
                                       axes_origin=[p_range[0], zero], epsilon_z=zero, extrapolation_flag=True)
            self.physics.add_property_region(pc)
            self.pc = pc
            self.timer.node["initialization"].stop()

        def set_solver(self):
            self.ts_control.dt_first = 1e-3
            self.ts_control.dt_min = 1e-6
            self.ts_control.dt_mult = 2
            self.ts_control.dt_max = dtmax or d["dtmax"]
            super().set_solver()

        def set_wells(self, verbose=None):
            self.cf = {}
            nx, ny = d["nx"], d["ny"]
            for w in d["wells"].values():
                self.reservoir.add_well(w["name"])
                for (i, j, k) in w["perfs"]:
                    c = (i - 1) + (j - 1) * nx + (k - 1) * nx * ny
                    cf = peaceman_cf(d["permx"][c], d["permy"][c], d["dx"][c], d["dy"][c], d["dz"][c] * w["wfrac"],
                                     w["rw"], w["skin"]) * wi_scale
                    self.cf[w["name"]] = cf
                    self.reservoir.add_perforation(w["name"], res_cell_idx=(i, j, k), well_index=cf,
                                                   well_indexD=0.0, well_diameter=2 * w["rw"])
            super().set_wells(verbose)

        def set_initial_conditions(self):
            p0 = d["refpres"] + oil.m / oil.bo_at(d["refpres"]) * G_BAR_PER_M * (d["depth"] - d["refdepth"])
            if (d["depth"] > d["dwoc"]).any():
                raise ValueError("cells below DWOC: water-zone initialisation is not converted")
            sw = d["swof"][0, 0]
            rho_w = WaterDensity(d["pvtw"], d["rho_w_sc"]).evaluate(p0)
            rho_o = oil.m / np.array([oil.bo_at(p) for p in p0])
            zw = sw * rho_w / (sw * rho_w + (1 - sw) * rho_o)
            return self.physics.set_initial_conditions_from_array(
                mesh=self.reservoir.mesh, input_distribution={self.physics.vars[0]: p0, self.physics.vars[1]: zw})

        def set_well_controls(self):
            for w in self.reservoir.wells:
                self.control(w.name, None)

        # ---- injector mobility. DARTS upwinds the well-to-block flux from the well segment, so an injector sees the
        # mobility of pure water (krw(1)/muw). IMEX MOBWEIGHT and OPM use the total mobility of the perforated block.
        # With inj_mobility="total" the perforation transmissibility is rescaled after every converged time step:
        # T = CF * lambda_t(block) / lambda_w(Sw = 1), lagged by one step.
        def init_inj_mobility(self):
            mesh = self.reservoir.mesh
            self._tran = np.array(mesh.tran, copy=False)
            bm, bp = np.array(mesh.block_m, copy=False), np.array(mesh.block_p, copy=False)
            self._inj = []
            for w in self.reservoir.wells:
                if d["wells"][w.name]["type"] != "INJ":
                    continue
                seg = w.well_body_idx
                for (_, cell, wi, _) in w.perforations:
                    idx = np.flatnonzero(((bm == seg) & (bp == cell)) | ((bp == seg) & (bm == cell)))
                    self._inj.append((idx, cell, wi))
            self._water_d = WaterDensity(d["pvtw"], d["rho_w_sc"])
            self._water_mu = WaterViscosity(d["pvtw"])
            self._kr_w, self._kr_o = TableRelPerm(d["swof"], "wat"), TableRelPerm(d["swof"], "oil")
            self._krw_1 = float(np.interp(1.0, d["swof"][:, 0], d["swof"][:, 1]))

        def update_inj_mobility(self):
            if inj_mobility != "total" or not getattr(self, "_inj", None):
                return
            X = np.array(self.physics.engine.X, copy=False)
            for idx, cell, wi in self._inj:
                p, zw = X[2 * cell], X[2 * cell + 1]
                vw, vo = zw / self._water_d.evaluate(p), (1 - zw) * oil.bo_at(p) / oil.m
                sw = vw / (vw + vo)
                mu_w = self._water_mu.evaluate(p)
                lam_t = self._kr_w.evaluate(sw) / mu_w + self._kr_o.evaluate(1 - sw) / oil.mu_at(p)
                self._tran[idx] = wi * lam_t / (self._krw_1 / mu_w)

        def after_converged_timestep(self):
            super().after_converged_timestep()
            self.update_inj_mobility()
            if no_crossflow:
                self.stop_reversed_wells()

        # ---- no crossflow. A DARTS well under BHP control injects back through a producer whose BHP is above the
        # block pressure (and produces through such an injector). OPM stops the well instead and keeps it stopped until
        # its next control keyword. This mimics it: after every converged step, a well whose flow has reversed at all its
        # perforations (well-block pressure vs block pressure) is set to zero rate until its next TARGET. The step in which
        # the reversal happened keeps its (small) reversed flow.
        def stop_reversed_wells(self):
            X = np.array(self.physics.engine.X, copy=False)
            for w in self.reservoir.wells:
                if self.bhp_on.get(w.name) is None:
                    continue
                inj = d["wells"][w.name]["type"] == "INJ"
                p_seg = X[2 * w.well_body_idx]                 # well-block pressure next to the perforations
                p_cells = np.array([X[2 * c] for c in self.cells[w.name]])
                if (not inj and (p_seg >= p_cells).all()) or (inj and (p_seg <= p_cells).all()):
                    self.stop_log.append((float(self.physics.engine.t), w.name, self.bhp_on[w.name], float(p_cells.mean())))
                    self.control(w.name, None)

        def control(self, name, bhp):
            """bhp in bar; None = shut (zero total rate)."""
            from darts.engines import well_control_iface as wci
            if not hasattr(self, "bhp_on"):
                self.bhp_on, self.stop_log = {}, []
                nx, ny = d["nx"], d["ny"]
                self.cells = {n: [(i - 1) + (j - 1) * nx + (k - 1) * nx * ny for i, j, k in w["perfs"]]
                              for n, w in d["wells"].items()}
            self.bhp_on[name] = bhp
            well = next(w for w in self.reservoir.wells if w.name == name)
            inj = d["wells"][name]["type"] == "INJ"
            comp = [1.0 - zero] if inj else None
            if bhp is None:
                self.physics.set_well_controls(wctrl=well.control, control_type=wci.MASS_RATE, is_inj=inj,
                                               target=0.0, inj_composition=comp)
            else:
                self.physics.set_well_controls(wctrl=well.control, control_type=wci.BHP, is_inj=inj,
                                               target=bhp, inj_composition=comp)

    m = Model()
    return m, oil


def run(dat, out_dir=None, wi_scale=1.0, n_points=400, save_cells=True, verbose=0, inj_mobility="total", dtmax=None,
        no_crossflow=True):
    from darts.engines import redirect_darts_output

    dat = Path(dat).resolve()
    out_dir = Path(out_dir) if out_dir else dat.parent.parent / f"{dat.parent.name}_darts" / dat.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    d = read_deck(dat)
    redirect_darts_output(str(out_dir / "run.log"))
    m, oil = build_model(d, wi_scale, n_points, inj_mobility=inj_mobility, dtmax=dtmax, no_crossflow=no_crossflow)
    m.init(verbose=verbose)
    m.init_inj_mobility()
    m.update_inj_mobility()
    m.set_output(output_folder=str(out_dir / "output"), save_initial=False, verbose=0)

    t0 = time.perf_counter()
    report = []                       # (date, days) at every DATE
    t, prev = 0.0, d["start"]
    for date_, kind, wname, bhp in d["events"]:
        if kind == "date" and date_ != prev:
            dt = (date_ - prev).days
            m.run(dt, verbose=verbose, save_reservoir_data=save_cells)
            t += dt
            prev = date_
            report.append((date_, t))
        elif kind == "target":
            m.control(wname, bhp * KGCM2_TO_BAR)
        elif kind == "shut":
            for n_ in (list(d["wells"]) if wname == "*" else [wname]):
                m.control(n_, None)
        elif kind == "open":
            raise ValueError("OPEN without TARGET is not converted")
    elapsed = time.perf_counter() - t0
    if save_cells:
        save_static(out_dir, d)

    pd.DataFrame(getattr(m, "stop_log", []), columns=["days", "well", "bhp_bar", "block_p_bar"]).to_csv(
        out_dir / "stopped_wells.csv", index=False)
    series = well_series(m, d, report)
    series.to_csv(out_dir / "wells.csv", index=False)
    last = series.groupby("well").last().reset_index()
    last["darts_time_s"] = round(elapsed, 2)
    last[["well", "days", "WBHP", "WOPT", "WWPT", "WGPT", "WWIT", "darts_time_s"]].to_csv(out_dir / "summary.csv", index=False)
    return out_dir, series, elapsed


def well_series(m, d, report):
    """Per-well series at the report dates from DARTS well data: surface rates (sm3/day) and cumulatives (sm3)."""
    td = m.output.store_well_time_data(phase_molar_rate=False, phase_mass_rate=False, phase_volumetric_rate=False,
                                       component_molar_rate=False, component_mass_rate=True, energy_rate=False)
    t = np.asarray(td["time"])
    rows = []
    days_rep = np.array([r[1] for r in report])
    for name, w in d["wells"].items():
        mw = np.asarray(td[f"well_{name}_mass_rate_w_at_wh"])      # kg/day, through the wellhead connection
        mo = np.asarray(td[f"well_{name}_mass_rate_o_at_wh"])
        bhp = np.asarray(td[f"well_{name}_BHP"])
        # DARTS sign: flow into the reservoir > 0. Below, producers report production > 0 and injectors injection > 0;
        # a reversed flow (before a well is stopped) shows as a negative rate.
        qo = mo / d["oil_mass_per_sm3"]
        qw = mw / d["rho_w_sc"]
        dt = np.diff(t, prepend=0.0)
        df = pd.DataFrame(dict(days=t, BHP=bhp, qo=qo, qw=qw, dt=dt))
        rows.append((name, w, df))
    out = []
    for name, w, df in rows:
        inj = w["type"] == "INJ"
        sgn = 1.0 if inj else -1.0
        qo, qw = sgn * df.qo.to_numpy(), sgn * df.qw.to_numpy()
        cum_o, cum_w = np.cumsum(qo * df.dt.to_numpy()), np.cumsum(qw * df.dt.to_numpy())
        idx = np.searchsorted(df.days.to_numpy(), days_rep - 1e-9)
        idx = np.clip(idx, 0, len(df) - 1)
        for k, (date_, days) in zip(idx, report):
            out.append(dict(well=name, date=date_.isoformat(), days=days, WBHP=df.BHP.iloc[k],
                            WOPR=0.0 if inj else qo[k], WWPR=0.0 if inj else qw[k], WGPR=0.0 if inj else qo[k] * d["rs"],
                            WWIR=qw[k] if inj else 0.0,
                            WOPT=0.0 if inj else cum_o[k], WWPT=0.0 if inj else cum_w[k],
                            WGPT=0.0 if inj else cum_o[k] * d["rs"], WWIT=cum_w[k] if inj else 0.0))
    return pd.DataFrame(out)


def save_static(out_dir, d):
    """output/static.h5: grid and rock per cell (flattened, i fastest) and the well perforations."""
    import h5py
    n = d["nx"] * d["ny"] * d["nz"]
    with h5py.File(Path(out_dir) / "output" / "static.h5", "w") as f:
        f.attrs.update(nx=d["nx"], ny=d["ny"], nz=d["nz"], units="m, mD, bar; well ijk are 1-based")
        for k in ("dx", "dy", "dz", "depth", "permx", "permy"):
            f[k] = np.broadcast_to(np.asarray(d[k], float), n)
        for k in ("permz", "poro"):
            f[k] = np.full(n, d[k])
        for name, w in d["wells"].items():
            g = f.create_group(f"wells/{name}")
            g.attrs["type"] = w["type"]
            g["ijk"] = np.array(w["perfs"]) if "perfs" in w else np.zeros((0, 3), int)


def cell_states(out_dir, dat):
    """(days, pressure [bar], Sw) arrays of shape (n_dates,) and (n_dates, n_cells) from a run with cell data."""
    import h5py
    d = read_deck(dat)
    oil = TableOil(d["oil_table"], d["oil_mass_per_sm3"])
    water = WaterDensity(d["pvtw"], d["rho_w_sc"])
    with h5py.File(Path(out_dir) / "output" / "reservoir_solution.h5", "r") as f:
        t, X = f["dynamic/time"][:], f["dynamic/X"][:]
    p, zw = X[..., 0], X[..., 1]
    bo = np.interp(p, oil.p, oil.bo)                    # inside the table range: same as TableOil
    vw = zw / np.vectorize(water.evaluate)(p)
    vo = (1 - zw) * bo / oil.m
    return t, p, vw / (vw + vo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dat")
    ap.add_argument("-o", "--out-dir")
    ap.add_argument("--wi-scale", type=float, default=1.0, help="multiplier on the Peaceman well indices")
    ap.add_argument("--n-points", type=int, default=400, help="OBL interpolation points per axis (accuracy vs speed)")
    ap.add_argument("--no-cells", action="store_true",
                    help="do not save the cell data (output/static.h5 and output/reservoir_solution.h5, saved by default)")
    ap.add_argument("--inj-mobility", choices=["total", "water"], default="total",
                    help="injector mobility: total mobility of the perforated block (IMEX MOBWEIGHT, OPM) or DARTS default "
                         "(pure injected water)")
    ap.add_argument("--allow-crossflow", action="store_true",
                    help="let a producer inject (and an injector produce) when its BHP is on the wrong side of the block "
                         "pressure (DARTS default); by default the well is stopped until its next target, as OPM does")
    ap.add_argument("--dtmax", type=float, help="largest time step, days (default: DTMAX of the deck)")
    ap.add_argument("-v", "--verbose", type=int, default=0)
    a = ap.parse_args()
    out, series, el = run(a.dat, a.out_dir, a.wi_scale, a.n_points, not a.no_cells, a.verbose, a.inj_mobility, a.dtmax,
                          not a.allow_crossflow)
    print(series.groupby("well").last()[["days", "WBHP", "WOPT", "WWPT", "WWIT"]].to_string())
    print(f"{el:.1f} s -> {out}")
