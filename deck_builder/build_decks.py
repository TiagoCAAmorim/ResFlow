"""Write the IMEX .dat decks (and optionally OPM Flow decks) for the layouts and BHP schedules.

    python build_decks.py --patches _output_patches --layouts _layouts --out _decks --template my_template.tpl
    python build_decks.py --patches P --layouts L --out _decks_opm --template T --format opm

One deck per (case, schedule): case = patch x well count (case_id in well_locations.csv). Needs the patches in
IMEX format (generate_spe10_patches.py default). --format opm also converts every deck with cmg_to_opm.py into
<out>/opm/<deck name>/ (the IMEX decks are kept). Output folder must not exist or be empty.

Layout of --out: inc/ (KR.inc, PVT.inc, CORNERS_<ni>x<nj>x<nk>.inc), perm/<family_id>.inc, decks/<case>_<schedule>.dat,
decks_index.csv. The template is an input (--template); templates/ holds an example (syntax: template_engine.py).
"""
import argparse
import re
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

import bhp_schedule as bhp
import fluid_files
import time_grid
from defaults import DEFAULT_SEED, out_dir_in_use
from report_dates import report_dates
from grid_geometry import write_corners_file
from template_engine import render

HERE = Path(__file__).resolve().parent
PROD_LIMIT, INJ_LIMIT = bhp.PROD_RANGE[0], bhp.INJ_RANGE[1]   # OPERATE MIN BHP / MAX BHP limits


def ymd(d):
    return f"{d.year} {d.month:02d} {d.day:02d}"


def wells_of(row):
    """[(name, i, j)] of the used wells of a well_locations.csv row, producers first."""
    out = []
    for letter in "PI":
        k = 1
        while f"{letter}{k}I" in row and row[f"{letter}{k}I"] > 0:
            out.append((f"{letter}{k}", int(row[f"{letter}{k}I"]), int(row[f"{letter}{k}J"])))
            k += 1
    return out


def date_blocks(sched, extra_dates):
    """[{'date': 'YYYY MM DD', 'controls': [{'well', 'bhp'}]}] from the long schedule rows of one deck plus the
    report dates (dates without controls)."""
    by_date = {d: [] for d in extra_dates}
    for r in sched.itertuples():
        by_date.setdefault(date.fromisoformat(r.date), []).append(dict(well=r.well, bhp=r.bhp_kgfcm2))
    return [dict(date=ymd(d), controls=by_date[d]) for d in sorted(by_date)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches", required=True, help="output folder of generate_spe10_patches.py (IMEX format)")
    p.add_argument("--layouts", required=True, help="folder with well_locations.csv and bhp_schedules.csv")
    p.add_argument("--out", required=True, help="output folder (must not exist or be empty)")
    p.add_argument("--format", choices=["imex", "opm"], default="imex",
                   help="imex = write the IMEX decks; opm = also convert each one for OPM Flow (default: %(default)s)")
    p.add_argument("--template", required=True,
                   help="deck template file (required; see template_engine.py for the syntax)")
    p.add_argument("--schedules", default=None,
                   help="write decks only for these schedules, among those that exist for the case (default: fixed,variable; "
                        "with --date-step also staged)")
    p.add_argument("--cache-dir", default=str(HERE / "_inputs"),
                   help="where the UNISIM downloads (KR, PVT) are kept (default: %(default)s)")
    p.add_argument("--cell-size", type=float, nargs=3, default=[50.0, 50.0, 5.0], metavar=("DX", "DY", "DZ"),
                   help="cell size in metres for CORNERS.inc; must match generate_well_layouts.py (default: 50 50 5)")
    p.add_argument("--dtmax", type=float, default=10.0, help="DTMAX of the NUMERICAL section: the largest time step the simulator may take between two DATEs, in days. "
                        "Smaller is more accurate and slower; it only matters when dates are far apart (default: %(default)s)")
    p.add_argument("--report-dates", default="example",
                   help="extra DATE lines without commands that force output on that day: example (pattern of the "
                        "paper decks), daily, controls (none), or a file with one YYYY-MM-DD per line "
                        "(default: %(default)s). See report_dates.py")
    p.add_argument("--test-report-dates", default="daily",
                   help="same as --report-dates, for the decks of the test set only (default: %(default)s)")
    p.add_argument("--sets", default="train,validation,test",
                   help="only write decks of these sets (default: %(default)s)")
    p.add_argument("--date-step", type=int, default=None, metavar="DAYS",
                   help="long-horizon decks: a DATE line every DAYS days for all sets, wells open on the first one. "
                        "Replaces --report-dates, --test-report-dates and bhp_schedules.csv (BHP values are drawn here, "
                        "with --seed); needs --bhp-step. See time_grid.py")
    p.add_argument("--bhp-step", type=int, default=None, metavar="DAYS",
                   help="with --date-step: BHP changes every DAYS days (a multiple of --date-step; 0 = fixed BHP). "
                        "Test decks also get a 'staged' schedule that opens one well per --open-step")
    p.add_argument("--open-step", type=int, default=None, metavar="DAYS",
                   help="with --date-step: the staged schedule opens one well every DAYS days (a multiple of --date-step; "
                        "default: --bhp-step, or --date-step if --bhp-step is 0)")
    p.add_argument("--sim-days", type=int, default=None, metavar="DAYS",
                   help="with --date-step: the run lasts DAYS days from the start date instead of until 2025-01-01 "
                        "(default: the whole run, 1828 days)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="with --date-step: seed of the BHP values (default: %(default)s)")
    p.add_argument("--no-restart", action="store_true", help="OPM only: do not write grid results (pressure, saturation) at each DATE; faster, well and field results only")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    dense = a.date_step is not None
    if a.sim_days is not None and not dense:
        sys.exit("--sim-days needs --date-step")
    sim_end = bhp.SIM_START + timedelta(days=a.sim_days) if a.sim_days else bhp.SIM_END
    if dense != (a.bhp_step is not None):
        sys.exit("--date-step and --bhp-step go together")
    if dense:
        try:
            all_dates, controls = time_grid.date_grid(a.date_step, a.bhp_step, end=sim_end)
            opens = time_grid.open_dates(a.date_step, a.open_step or a.bhp_step or a.date_step, end=sim_end)
        except ValueError as e:
            sys.exit(str(e))
    out, patches, layouts = Path(a.out), Path(a.patches), Path(a.layouts)
    if out_dir_in_use(out):
        sys.exit(f"{out} is not empty; choose a new or an empty folder")
    for folder, files in ((patches, ("config_used.yaml", "manifest.csv", "perm")),
                          (layouts, ("well_locations.csv",) if dense else ("well_locations.csv", "bhp_schedules.csv"))):
        missing = [f for f in files if not (folder / f).exists()]
        if missing:
            sys.exit(f"{folder}: missing {', '.join(missing)}. Run the previous steps first "
                     "(generate_spe10_patches.py, generate_well_layouts.py, bhp_schedule.py); see docs/02_build_decks.md.")
    pcfg = yaml.safe_load((patches / "config_used.yaml").read_text())
    if pcfg.get("format") != "imex":
        sys.exit(f"{patches} holds {pcfg.get('format')} patches; the decks need IMEX patch files")
    man = pd.read_csv(patches / "manifest.csv").set_index("family_id")
    wl = pd.read_csv(layouts / "well_locations.csv")
    sched = None if dense else pd.read_csv(layouts / "bhp_schedules.csv")
    wanted = (a.schedules or ("fixed,variable,staged" if dense else "fixed,variable")).split(",")
    template = Path(a.template).read_text()

    for sub in ("inc", "perm", "decks"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fluid_files.build_kr_inc(a.cache_dir, out / "inc" / "KR.inc")
    fluid_files.build_pvt_inc(a.cache_dir, out / "inc" / "PVT.inc")

    # report dates plus every control date, so that the fixed and variable decks share one time grid
    def dates_for(mode):
        return sorted(set(report_dates(mode, bhp.CONTROL_DATES, bhp.SIM_START, bhp.SIM_END)) | set(bhp.CONTROL_DATES))
    extra = {False: dates_for(a.report_dates), True: dates_for(a.test_report_dates)} if not dense else None   # keyed by "is test set"
    sets = a.sets.split(",")
    index = []
    for row in wl.to_dict("records"):
        fid = row["family_id"]
        m = man.loc[fid]
        if m.set not in sets:
            continue
        ni, nj, nk = int(m.n_i), int(m.n_j), int(m.get("n_k", 1))
        corners = out / "inc" / f"CORNERS_{ni}x{nj}x{nk}.inc"
        if not corners.exists():
            write_corners_file((ni, nj, nk), tuple(a.cell_size), (0.0, 0.0), 5450.0, corners,
                               include_comments=False)
        if not (out / "perm" / f"{fid}.inc").exists():
            shutil.copyfile(patches / "perm" / f"{fid}.inc", out / "perm" / f"{fid}.inc")
        wells = wells_of(row)
        assert len(wells) == row["n_prod"] + row["n_inj"], row["case_id"]
        prod = [dict(well=n, i=i, j=j, bhp_limit=PROD_LIMIT) for n, i, j in wells if n[0] == "P"]
        inj = [dict(well=n, i=i, j=j, bhp_limit=INJ_LIMIT) for n, i, j in wells if n[0] == "I"]
        for kind in (time_grid.schedules_for(m.set, row["n_inj"], a.bhp_step) if dense else wanted):
            if kind not in wanted:
                continue
            if dense:
                try:
                    rows = time_grid.schedule_table(kind, row["case_id"], wells, controls, a.seed, tuple(a.cell_size[:2]), opens)
                except ValueError as e:
                    sys.exit(str(e))
            else:
                rows = sched[(sched.case_id == row["case_id"]) & (sched.schedule == kind)]
            if rows.empty:      # e.g. no fixed schedule for train cases: see bhp_schedule.schedules_for
                continue
            name = f"{row['case_id']}_{kind}"
            ctx = dict(case_id=row["case_id"], schedule=kind, n_prod=row["n_prod"], n_inj=row["n_inj"],
                       ni=ni, nj=nj, nk=nk, corners_file=f"../inc/{corners.name}", perm_file=f"../perm/{fid}.inc",
                       pvt_file="../inc/PVT.inc", kr_file="../inc/KR.inc", dtmax=a.dtmax,
                       start_date=ymd(bhp.SIM_START), end_date=ymd(sim_end), producers=prod,
                       injectors=inj, dates=date_blocks(rows, all_dates if dense else extra[m.set == "test"]))
            (out / "decks" / f"{name}.dat").write_text(render(template, ctx))
            index.append(dict(deck=name, case_id=row["case_id"], schedule=kind, family_id=fid, set=m.set,
                              size=m["size"], n_prod=row["n_prod"], n_inj=row["n_inj"]))
    pd.DataFrame(index).to_csv(out / "decks_index.csv", index=False)
    (out / "config_used.yaml").write_text(yaml.safe_dump({**vars(a), "argv": sys.argv[1:] if argv is None else list(argv)}))   # seed of the BHP values etc., to rerun
    print(f"{len(index)} IMEX decks -> {out / 'decks'}")

    if a.format == "opm":
        from cmg_to_opm import convert
        for r in index:
            convert(out / "decks" / f"{r['deck']}.dat", out / "opm" / r["deck"], restart=not a.no_restart)
        print(f"{len(index)} OPM decks -> {out / 'opm'}")


if __name__ == "__main__":
    main()
