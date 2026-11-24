"""Report dates: DATE lines without any command, which force IMEX/OPM to write results on that day.

Modes (build_decks.py --report-dates):
  example   the report-date pattern of the paper's train and validation decks. For every BHP control date c: c+1..c+3 days, the 1st of the next
            four months, and days 2-5 of the second of those months. Before the first control every day is
            reported, and the first control is followed by 4 daily dates instead of 3.
  daily     every day from the start to the end of the run
  controls  only the BHP control dates (no extra dates)
  <file>    a text file with one ISO date (YYYY-MM-DD) per line
The dates are built from the full control-date grid, so the fixed and variable decks of a case share them.
"""
from datetime import date, timedelta
from pathlib import Path


def add_months(d, n):
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def example_dates(controls, start, end):
    out = set()
    first = controls[0]
    out.update(start + timedelta(days=k) for k in range(1, (first - start).days))
    for n, c in enumerate(controls):
        out.update(c + timedelta(days=k) for k in range(1, 5 if n == 0 else 4))
        for m in range(1, 5):
            out.add(add_months(c, m))
        out.update(add_months(c, 2) + timedelta(days=k) for k in range(1, 5))
    return {d for d in out if start < d < end}


def report_dates(mode, controls, start, end):
    """Sorted list of the report dates (without the start date, the end date and the control dates)."""
    if mode == "controls":
        extra = set()
    elif mode == "daily":
        extra = {start + timedelta(days=k) for k in range(1, (end - start).days)}
    elif mode == "example":
        extra = example_dates(list(controls), start, end)
    else:
        extra = {date.fromisoformat(l.strip()) for l in Path(mode).read_text().splitlines() if l.strip()}
        bad = [d for d in extra if not start < d < end]
        if bad:
            raise ValueError(f"{mode}: dates outside ({start}, {end}): {sorted(bad)[:3]}")
    return sorted(extra - set(controls))
