"""Run every CMG .dat in a folder in open-DARTS and compare with IMEX (.sr3) or OPM Flow (.UNSMRY).

Usage:
    python run_darts_batch.py $DATA/decks/decks [-j 4] [--tol 0.10] [--n-points 400]
Output: <folder>_darts/<case>/ (wells.csv, summary.csv, stopped_wells.csv, run.log) and <folder>_darts/comparison.csv
Reference for each case, first found: <case>.sr3 next to the deck (IMEX); <folder>/../opm/<case>/ (build_decks.py
--format opm, after running it) or <folder>_opm/<case>/out/ (run_opm_batch.py), holding <CASE>.UNSMRY.
darts_time_s = wall time of the DARTS run (time stepping only), -j cases in parallel.
"""
import argparse
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd


def find_reference(dat):
    dat = Path(dat)
    if dat.with_suffix(".sr3").exists():
        return dat.with_suffix(".sr3")
    for folder in (dat.parent.parent / "opm" / dat.stem, dat.parent.parent / "opm" / dat.stem / "out",
                   dat.parent.parent / f"{dat.parent.name}_opm" / dat.stem / "out"):
        f = folder / f"{dat.stem.upper()}.UNSMRY"
        if f.exists():
            return f
    return None


def compare(wells_csv, ref, tol=0.10):
    """Final cumulatives: rows (well, key, ref, darts, rel); ok if every producer's oil is within tol."""
    from compare_cmg_opm import read_cmg, read_opm
    ref = Path(ref)
    rf = read_cmg(ref) if ref.suffix.lower() == ".sr3" else read_opm(ref)
    dd = pd.read_csv(wells_csv).groupby("well").last()
    rows = []
    for w, r in dd.iterrows():
        inj = r["WWIT"] > 0 or (r["WOPT"] == 0 and r["WWPT"] == 0)
        for k in (("WWIT",) if inj else ("WOPT", "WWPT")):
            c, a = float(rf[w][k][-1]), float(r[k])
            rows.append((w, k, c, a, (a - c) / c if c else float("nan")))
    ok = all(abs(rel) <= tol for w, k, c, a, rel in rows if k == "WOPT" and c > 0)
    return ok, rows


def run_case(args):
    dat, tol, n_points = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from cmg_to_darts import run
    dat = Path(dat)
    row = dict(case=dat.stem, darts_time_s="", status="", reference="", oil_diff_pct="", water_diff_pct="",
               inj_diff_pct="", worst_well_oil_diff_pct="", detail="")
    try:
        out, series, elapsed = run(dat, n_points=n_points)
        row["darts_time_s"] = f"{elapsed:.2f}"
        ref = find_reference(dat)
        if ref is None:
            row["status"] = "NO_REFERENCE"
            return row
        row["reference"] = "IMEX" if ref.suffix.lower() == ".sr3" else "OPM"
        ok, rows = compare(out / "wells.csv", ref, tol)
        with open(out / "comparison.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["well", "quantity", row["reference"], "DARTS", "rel_diff"])
            wr.writerows(rows)
        tot = {k: [sum(x[2] for x in rows if x[1] == k), sum(x[3] for x in rows if x[1] == k)] for k in ("WOPT", "WWPT", "WWIT")}
        for key, k in (("oil_diff_pct", "WOPT"), ("water_diff_pct", "WWPT"), ("inj_diff_pct", "WWIT")):
            row[key] = f"{(tot[k][1] / tot[k][0] - 1) * 100:.2f}" if tot[k][0] else ""
        worst = max(((abs(x[4]), x[0]) for x in rows if x[1] == "WOPT" and x[2] > 0), default=(0, ""))
        row["worst_well_oil_diff_pct"] = f"{worst[0] * 100:.2f}"
        row["detail"] = worst[1]
        row["status"] = "OK" if ok else "OUT_OF_TOL"
    except Exception as e:
        row.update(status="ERROR", detail=f"{type(e).__name__}: {e}"[:200])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--n-points", type=int, default=400, help="OBL points per axis")
    a = ap.parse_args()
    folder = Path(a.folder).resolve()
    decks = sorted(folder.glob("*.dat"))
    print(f"{len(decks)} decks")
    with ProcessPoolExecutor(a.j) as ex:
        rows = list(ex.map(run_case, [(d, a.tol, a.n_points) for d in decks]))
    out = folder.parent / f"{folder.name}_darts" / "comparison.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(Counter(r["status"] for r in rows))
    print(out)
