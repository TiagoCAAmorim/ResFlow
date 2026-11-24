"""BHP schedules for the dataset: 'fixed' (one value per well, whole run) and 'variable' (a new value
per well at every control date). Units are kgf/cm2 (IMEX MODSI).

    python bhp_schedule.py --well-locations _layouts/well_locations.csv --out _layouts/bhp_schedules.csv

Output (long format): case_id, schedule, well, date, bhp_kgfcm2. For the fixed schedule there is one row
per well, at the first control date; the value is kept until the end of the run.
"""
import argparse
import re
import sys
import zlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from defaults import DEFAULT_SEED

PROD_RANGE = (180.0, 260.0)
INJ_RANGE = (500.0, 550.0)
SIM_START = date(2020, 1, 1)
SIM_END = date(2025, 1, 1)
# Control dates of the previous dataset versions: Jan 5, then every 4 months (day 2) up to Sep 2024.
CONTROL_DATES = [date(2020, 1, 5)] + [date(y, m, 2) for y in range(2020, 2025) for m in (1, 5, 9)][1:]
SCHEDULES = ("fixed", "variable", "staged")   # staged: only made by time_grid.py (wells open one per control date)


def schedules_for(set_name, n_inj, fixed_sets=("test",)):
    """Schedules simulated for a case: 'variable' for every case, 'fixed' only for the sets in `fixed_sets`
    (default: test, as a generalisation test). A producer-only case (no injector) is constant-BHP only."""
    if n_inj == 0:
        return ["fixed"]
    return ["variable"] + (["fixed"] if set_name in fixed_sets else [])


def make_schedule(kind, case_id, n_prod, n_inj, seed=DEFAULT_SEED, dates=CONTROL_DATES):
    """Rows (well, date, bhp) of one schedule; reproducible for (seed, case_id, kind). 'staged' returns all wells at
    all dates, like 'variable' (time_grid.py drops the rows before each well opens)."""
    if kind not in SCHEDULES:
        raise ValueError(f"unknown schedule {kind!r}")
    rng = np.random.default_rng([seed, zlib.crc32(case_id.encode()), SCHEDULES.index(kind)])
    wells = [(f"P{p}", PROD_RANGE) for p in range(1, n_prod + 1)] + \
            [(f"I{i}", INJ_RANGE) for i in range(1, n_inj + 1)]
    use_dates = dates[:1] if kind == "fixed" else dates
    rows = []
    for name, (lo, hi) in wells:
        for d in use_dates:
            rows.append((name, d, round(float(rng.uniform(lo, hi)), 1)))
    return rows


def build_table(well_locations, seed=DEFAULT_SEED, fixed_sets=("test",)):
    rows = []
    for r in well_locations.itertuples(index=False):
        for kind in schedules_for(r.set, r.n_inj, fixed_sets):
            for well, d, bhp in make_schedule(kind, r.case_id, r.n_prod, r.n_inj, seed):
                rows.append(dict(case_id=r.case_id, schedule=kind, well=well, date=d.isoformat(),
                                 bhp_kgfcm2=bhp))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--well-locations", required=True, help="well_locations.csv written by generate_well_layouts.py")
    p.add_argument("--out", required=True, help="output CSV (must not exist)")
    p.add_argument("--fixed-sets", default="test",
                   help="sets that also get the fixed (constant BHP) schedule; the others get variable only "
                        "(default: %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        sys.exit(f"{out} already exists")
    table = build_table(pd.read_csv(a.well_locations), a.seed, tuple(a.fixed_sets.split(",")))
    table.to_csv(out, index=False)
    print(f"{table.case_id.nunique()} cases, {table.groupby(['case_id', 'schedule']).ngroups} schedules -> {out}")


if __name__ == "__main__":
    main()
