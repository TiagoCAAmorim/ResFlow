"""Convert the results of an OPM Flow run into a ResFlow raw folder.

The folder has the same files, layout, dtypes and units as the one written by the IMEX ``.sr3`` converter::

    model.json  res_static.pt  res_dynamic/{pres,so,sg,rs}.pth  pvt01.pth  krel01.{wo,go}.pth
    well_static.json  well_dynamic/{np,wp,gp,wi,p_prod,p_winj}.pth
    con_static.json   con_dynamic/{np,wp,gp,wi,p_prod,p_winj}.pth

Units: pressure kgf/cm2 (OPM bar / 0.980665), volumes sm3 (cumulative), depth m, well index md.m
(OPM ``CTFAC`` / 0.00852702), transmissibility md.m (OPM ``TRAN`` / 0.00852702).

The PVT and KR tables, the well order and the BHP depths are read from the IMEX ``.dat`` deck the OPM deck was
made from (OPM does not output them). Every restart (report) step of the OPM run is a grid output; the one-day
series are chosen as in the IMEX converter. Needs per-connection summary vectors CTFAC, COPT, CWPT, CWIT, CGPT, CPR
(requested by ``deck_builder/cmg_to_opm.py``) and ``RPTRST BASIC=2`` for the restart file.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import dat_tables
from .reader import DARCY_METRIC, KGCM2_TO_BAR, Summary, find_output_stem, read_grid, read_restart
from .static import build_static

TIME_TOL = 1e-4  # days: tolerance to match summary entries with restart steps
SHUT_TOL = 1.0  # sm3: a well with all cumulative volumes below this at the last output never flowed


class OpmRunDiscarded(ValueError):
    """The run is unusable (no well produced or injected, e.g. it stopped before the wells opened)."""


def select_outputs(values, step=1.0, tol=1e-6):
    """Select the outputs that form series spaced exactly ``step`` apart (same rule as the IMEX converter).

    A series starts at an output and moves to the next one at least ``step`` later; outputs closer than ``step`` are
    skipped until the elapsed time reaches ``step`` (within ``tol``); if it overshoots, the series ends. The next
    series starts at the output after the one where the previous ended; single-output series are discarded.
    Returns ``(kept, groups)``: kept indices into ``values`` and series as ``[start, end]`` indices into ``kept``.
    """
    n = len(values)
    kept, groups, start = [], [], 0
    while start < n - 1:
        members, cur, j = [start], start, start + 1
        while j < n:
            diff = values[j] - values[cur]
            if diff < step - tol:
                j += 1
            elif diff <= step + tol:
                members.append(j)
                cur = j
                j += 1
            else:
                break
        if len(members) > 1:
            groups.append([len(kept), len(kept) + len(members) - 1])
            kept.extend(members)
        start = cur + 1
    return kept, groups


def _ijk(num, nx, ny):
    n = num - 1
    return n % nx + 1, (n // nx) % ny + 1, n // (nx * ny) + 1


def _sample(vec, sum_times, times):
    """Value of a summary vector at each restart time; time 0 (no summary entry) is zero."""
    idx = {round(t / TIME_TOL): i for i, t in enumerate(sum_times)}
    out = np.zeros(len(times))
    for n, t in enumerate(times):
        i = idx.get(round(t / TIME_TOL))
        if i is None:
            if abs(t) < TIME_TOL:
                continue
            raise ValueError(f"No summary entry at day {t}: summary and restart steps do not match")
        out[n] = vec[i]
    return out


def _save(t, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor(np.nan_to_num(t, nan=0.0), dtype=torch.float32), path)


def convert_opm_run(opm_case, imex_dat_path, out_dir, step_size=1.0, verbose=False, wells_open_day=4.0):
    """Write the raw folder ``out_dir`` for one OPM run and return its path.

    ``opm_case``: the OPM ``.DATA`` file, its case folder (results in ``out/``) or the result stem.
    ``imex_dat_path``: the IMEX ``.dat`` the OPM deck was made from (PVT, KR, well data).
    ``step_size``: days between the samples of one series (1.0 for the 1-day models, 10.0 for the 10-day ones).
    ``wells_open_day``: a run whose last kept output is at or before this day never opened its wells and is
    discarded. Raises :class:`OpmRunDiscarded` (and leaves no folder) also when no well produced or injected.
    """
    out_dir, dat = Path(out_dir), Path(imex_dat_path)
    stem = find_output_stem(opm_case)
    try:
        _convert(stem, dat, out_dir, step_size, verbose, wells_open_day)
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise
    return out_dir


def _convert(stem, dat, out_dir, step_size, verbose, wells_open_day):
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = read_grid(stem)
    nx, ny = grid["nx"], grid["ny"]
    act_index = np.full(nx * ny * grid["nz"], -1)
    act_index[grid["actnum"]] = np.arange(int(grid["actnum"].sum()))
    ini = grid["init"]

    times, rst = read_restart(stem)
    if len(times) < 2:
        raise OpmRunDiscarded(f"Only one restart step in {stem}")
    summ = Summary(stem)
    for k in ("CTFAC", "COPT", "CWPT", "CWIT", "CGPT", "CPR", "WBHP"):
        if not summ.has(k):
            raise ValueError(f"Summary vector {k} missing: rebuild the OPM deck with deck_builder/cmg_to_opm.py")
    kept, steps = select_outputs([float(t) for t in times], step=step_size)
    if not kept:
        raise OpmRunDiscarded(f"No pair of outputs {step_size} day(s) apart in {stem}")
    tk = times[kept]
    if tk[-1] <= wells_open_day:
        raise OpmRunDiscarded(f"Run ended at day {tk[-1]:g} (<= {wells_open_day:g}): the wells never opened")

    # ---- wells and connections (order: .dat well order; connections by cell number, i.e. by layer)
    dat_order = dat_tables.well_order(dat)
    wells_in_run = {n for n, _ in summ.entries("CTFAC")}
    wells = [w for w in dat_order if w in wells_in_run] + sorted(wells_in_run - set(dat_order))
    cons = {}
    for w in wells:
        cons[w] = sorted(c for n, c in summ.entries("CTFAC") if n == w)
    bhp_depth = dat_tables.bhp_depths(dat)
    missing = [w for w in wells if w not in bhp_depth]
    if missing:
        raise ValueError(f"No BHPDEPTH in the .dat for wells {missing}")

    def cum(key, w, num):
        return _sample(summ.get(key, w, num), summ.times, tk)

    cum_all = {k: {(w, c): cum(k, w, c) for w in wells for c in cons[w]} for k in ("COPT", "CWPT", "CWIT", "CGPT")}

    # ---- which wells count (same rule as the IMEX converter, on the cumulative volumes at the last output)
    last = {}
    for w in wells:
        last[w] = {k: sum(float(_sample(summ.get(k, w, c), summ.times, summ.times[-1:])[0]) for c in cons[w])
                   for k in ("COPT", "CWPT", "CWIT", "CGPT")}
    prod = [w for w in wells if last[w]["COPT"] > SHUT_TOL]
    winj = [w for w in wells if last[w]["COPT"] < SHUT_TOL < last[w]["CWIT"]]
    if verbose:
        shut = [w for w in wells if all(v < SHUT_TOL for v in last[w].values())]
        print(f"  {len(prod)} producers, {len(winj)} water injectors, {len(shut)} shut-in wells")
    if not prod and not winj:
        raise OpmRunDiscarded("No production or injection wells found (all wells shut-in)")

    # ---- model.json
    (out_dir / "model.json").write_text(json.dumps({
        "name": Path(stem).name + ".DATA",
        "grid_type": "1phi1k",
        "model_type": "blackoil",
        "n_cells": int(grid["actnum"].sum()),
        "days": [float(t) for t in tk],
        "steps": steps,
        "prod": prod, "winj": winj, "ginj": [],
        "n_krel": 1, "n_pvt": len(dat_tables.read_pvt(dat)),
    }, indent=4))
    pvt = dat_tables.read_pvt(dat)
    for i, p in enumerate(pvt):
        torch.save(dat_tables.pvt_to_tensors(p), out_dir / f"pvt{i + 1:02}.pth")
    wo, go = dat_tables.krel_tensors(*dat_tables.read_krel(dat))
    torch.save(wo, out_dir / "krel01.wo.pth")
    torch.save(go, out_dir / "krel01.go.pth")

    # ---- static grid
    graph, _ = build_static(grid)
    torch.save(graph, out_dir / "res_static.pt")

    # ---- dynamic grid (kgf/cm2; So = 1 - Sw - Sg)
    pres = rst["PRESSURE"][kept].astype(np.float64) / KGCM2_TO_BAR
    sw, sg = rst["SWAT"][kept].astype(np.float64), rst["SGAS"][kept].astype(np.float64)
    fields = {"pres": pres, "so": np.clip(1 - sw - sg, 0, 1), "sg": np.clip(sg, 0, 1), "rs": rst["RS"][kept]}
    for name, v in fields.items():
        v = np.array(v, dtype=np.float64)
        v[v < 1e-15] = 0
        _save(v, out_dir / "res_dynamic" / f"{name}.pth")

    # ---- static connections and wells
    con_static = {"prod": {}, "winj": {}, "ginj": {}}
    for t, lst in (("prod", prod), ("winj", winj)):
        for w in lst:
            con_static[t][w] = {}
            for c in cons[w]:
                i, j, k = _ijk(c, nx, ny)
                cell = int(act_index[c - 1]) + 1
                if cell == 0:
                    continue
                ctfac = float(summ.get("CTFAC", w, c).max())
                con_static[t][w][f"{w}{{{i},{j},{k}}}"] = {
                    "cell": cell, "depth": float(ini["DEPTH"][cell - 1]), "wi": ctfac / DARCY_METRIC}
    (out_dir / "con_static.json").write_text(json.dumps(con_static, indent=4))
    well_static = {"prod": {w: bhp_depth[w] for w in prod}, "winj": {w: bhp_depth[w] for w in winj}, "ginj": {}}
    (out_dir / "well_static.json").write_text(json.dumps(well_static, indent=4))

    # ---- dynamic connections and wells
    prop = {"prod": {"np": "COPT", "wp": "CWPT", "gp": "CGPT"}, "winj": {"wi": "CWIT"}}
    pname = {"prod": "p_prod", "winj": "p_winj"}
    targets = dat_tables.bhp_targets(dat)
    for t, lst in (("prod", prod), ("winj", winj)):
        if not lst:
            continue  # no such wells: no files (as in the IMEX converter)
        keys = [(w, c) for w in lst for c in cons[w] if act_index[c - 1] >= 0]
        for fname, key in prop[t].items():
            v = np.stack([cum_all[key][k] for k in keys], axis=1)
            _save(v, out_dir / "con_dynamic" / f"{fname}.pth")
            vw = np.stack([sum(cum_all[key][(w, c)] for c in cons[w] if act_index[c - 1] >= 0) for w in lst], axis=1)
            _save(vw, out_dir / "well_dynamic" / f"{fname}.pth")
        # pressures: open connections/wells from OPM. A shut one (OPM reports 0, no crossflow) gets what IMEX
        # reports: the BHP target in force (the cell pressure before the first target, i.e. before the wells open)
        pc, pw = [], []
        for w in lst:
            ok = [c for c in cons[w] if act_index[c - 1] >= 0]
            tgt = dat_tables.target_at(targets.get(w, []), tk)
            wb = _sample(summ.get("WBHP", w, 0), summ.times, tk) / KGCM2_TO_BAR
            c0 = ok[0]
            pw.append(np.where(wb > 0, wb, np.where(np.isnan(tgt), pres[:, act_index[c0 - 1]], tgt)))
            for c in ok:
                p = _sample(summ.get("CPR", w, c), summ.times, tk) / KGCM2_TO_BAR
                both = (p > 0) & (wb > 0)
                head = float(np.median((p - wb)[both])) if both.any() else 0.0  # static well-to-layer head
                shut = np.where(np.isnan(tgt), pres[:, act_index[c - 1]], tgt + head)
                pc.append(np.where(p > 0, p, shut))
        # connection columns follow `keys` (wells in order, connections by layer)
        _save(np.stack(pc, axis=1), out_dir / "con_dynamic" / f"{pname[t]}.pth")
        _save(np.stack(pw, axis=1), out_dir / "well_dynamic" / f"{pname[t]}.pth")


def organize_opm_dataset(folder, output_dir, split_csv, step_size=1.0, resume=False, overwrite=False,
                         split_column="set", name_column="deck", wells_open_day=4.0, opm_dir=None):
    """Convert the OPM runs of a deck folder into train/val/test raw folders (hook of ``resflow convert --format opm``).

    ``folder``: the IMEX deck folder (``<deck>.dat``, as for the IMEX converter). The OPM cases are in ``opm_dir``
    (default ``<folder>_opm``, the output of ``deck_builder/run_opm_batch.py``) as ``<opm_dir>/<deck>/`` with the
    results in ``out/``. Each deck of ``split_csv`` goes to ``<output_dir>/<split>/raw/<deck>``; decks without OPM
    results or without ``.dat`` are skipped and reported. Log in ``<output_dir>/processing.log``.
    Returns messages ``"<deck>: <reason>"`` of the decks not converted; raises RuntimeError if none converted.
    """
    from ...data.convert import already_converted, prepare_output_dir, read_split_index
    folder = Path(folder)
    opm_dir = Path(opm_dir) if opm_dir else folder.parent / f"{folder.name}_opm"
    output_dir = prepare_output_dir(output_dir, resume=resume, overwrite=overwrite)
    split_of = read_split_index(split_csv, split_column, name_column)
    messages, n_ok = [], 0
    with open(output_dir / "processing.log", "a" if resume else "w", buffering=1, encoding="utf-8") as log:
        log.write(f"OPM conversion: {len(split_of)} indexed decks, step {step_size:g} d, "
                  f"runs ending at day <= {wells_open_day:g} discarded\n")
        for deck, split in split_of.items():
            if resume and already_converted(output_dir, deck):
                log.write(f"skipped (already converted): {deck}\n")
                continue
            dat = folder / f"{deck}.dat"
            try:
                if not dat.is_file():
                    raise FileNotFoundError(f"{dat.name} not found in {folder}")
                find_output_stem(opm_dir / deck)
                convert_opm_run(opm_dir / deck, dat, output_dir / split / "raw" / deck, step_size=step_size,
                                verbose=False, wells_open_day=wells_open_day)
                n_ok += 1
                log.write(f"converted: {deck}\n")
            except (OpmRunDiscarded, FileNotFoundError, ValueError, NotImplementedError) as e:
                messages.append(f"{deck}: {e}")
                log.write(f"not converted: {deck}: {e}\n")
    if n_ok == 0 and messages:
        raise RuntimeError(f"No simulation could be converted. First failure: {messages[0]}")
    return messages
