"""Regular time grid for long-horizon decks (build_decks.py --date-step / --bhp-step).

  DATE lines every `date_step` days from the start of the run (the last one is the end date, which only carries STOP).
  BHP control dates: every `bhp_step` days (a multiple of `date_step`), the first one on the first DATE line, which is
  where the wells open (they are shut before it). No control on the end date. bhp_step = 0: one control only, so the
  BHP is fixed.

Schedules per case (the same DATE lines for all of them):
  fixed     one BHP per well, set on the first control date (the only schedule when bhp_step = 0)
  variable  a new BHP per well on every control date
  staged    the wells open one per opening date (every `open_step` days, default bhp_step, or date_step if bhp_step = 0):
            P1, the injector closest to the open producers, P2, the next closest injector, ... (alternating; when
            one kind is used up the other follows). A well gets its BHP on its opening date (IMEX opens a shut well
            when it gets a BHP target) and, if bhp_step > 0, a new one on every later control date.
Which cases get which: train / validation = variable; test = variable, fixed and staged; a case without injector
(1P0I) = fixed only. With bhp_step = 0: variable is replaced by fixed (train / validation = fixed; test = fixed and staged).
"""
from datetime import timedelta
from math import hypot

import pandas as pd

import bhp_schedule as bhp


def date_grid(date_step, bhp_step, start=bhp.SIM_START, end=bhp.SIM_END):
    """(dates, controls): sorted DATE lines before the end date, and the BHP control dates among them."""
    if date_step < 1:
        raise ValueError("--date-step must be at least 1 day")
    if bhp_step < 0 or (bhp_step and bhp_step % date_step):
        raise ValueError(f"--bhp-step ({bhp_step}) must be 0 or a multiple of --date-step ({date_step})")
    n_days = (end - start).days
    dates = [start + timedelta(days=d) for d in range(date_step, n_days, date_step)]
    if not dates:
        raise ValueError(f"--date-step {date_step} leaves no DATE line before the end date")
    if bhp_step == 0:
        controls = dates[:1]
    else:
        controls = [start + timedelta(days=d) for d in range(date_step, n_days, bhp_step)]
    return dates, controls


def open_dates(date_step, open_step, start=bhp.SIM_START, end=bhp.SIM_END):
    """Dates on which the staged schedule opens its wells: every `open_step` days from the first DATE line."""
    if open_step < 1 or open_step % date_step:
        raise ValueError(f"--open-step ({open_step}) must be a positive multiple of --date-step ({date_step})")
    return [start + timedelta(days=d) for d in range(date_step, (end - start).days, open_step)]


def open_order(wells, cell_size=(50.0, 50.0)):
    """Order in which the wells open in the staged schedule. wells = [(name, i, j)], producers first (P1 first).
    P1, then alternately the nearest unopened injector to any open producer and the next producer; when one
    kind is used up the other one follows."""
    prods = [w for w in wells if w[0][0] == "P"]
    injs = [w for w in wells if w[0][0] == "I"]
    order, want_prod = [], True
    while prods or injs:
        if (want_prod and prods) or not injs:
            order.append(prods.pop(0))
        else:
            open_p = [w for w in order if w[0][0] == "P"]
            dist = lambda w: min(hypot((w[1] - p[1]) * cell_size[0], (w[2] - p[2]) * cell_size[1]) for p in open_p)
            nearest = min(injs, key=dist)
            injs.remove(nearest)
            order.append(nearest)
        want_prod = not want_prod
    return [w[0] for w in order]


def schedules_for(set_name, n_inj, bhp_step):
    if n_inj == 0:
        return ["fixed"]
    if bhp_step == 0:
        return ["fixed", "staged"] if set_name == "test" else ["fixed"]
    return ["variable", "fixed", "staged"] if set_name == "test" else ["variable"]


def schedule_table(kind, case_id, wells, controls, seed, cell_size=(50.0, 50.0), opens=None):
    """DataFrame (date as ISO text, well, bhp_kgfcm2) of one schedule; `wells` = [(name, i, j)]. `opens` = the
    opening dates of the staged schedule (see open_dates)."""
    n_prod = sum(w[0][0] == "P" for w in wells)
    if kind != "staged":
        rows = bhp.make_schedule(kind, case_id, n_prod, len(wells) - n_prod, seed, controls)
    else:
        order = open_order(wells, cell_size)
        if len(order) > len(opens):
            raise ValueError(f"{case_id}: staged schedule needs {len(order)} opening dates, only {len(opens)} fit in "
                             "the run; use a smaller --open-step (or --bhp-step)")
        first = {w: opens[k] for k, w in enumerate(order)}
        # a well gets a BHP on its opening date (this opens it) and on every later control date
        rows = [r for r in bhp.make_schedule(kind, case_id, n_prod, len(wells) - n_prod, seed,
                                             sorted(set(controls) | set(opens[:len(order)])))
                if r[1] == first[r[0]] or (r[1] > first[r[0]] and r[1] in controls)]
    return pd.DataFrame([dict(date=d.isoformat(), well=w, bhp_kgfcm2=b) for w, d, b in rows])
