"""Compare final cumulative well volumes between a CMG .sr3 and an OPM Flow run.

Usage:
    python compare_cmg_opm.py case.sr3 opm_out/CASE.UNSMRY [--tol 0.10]
Producers: CMG OILVOLSC / WATVOLSC vs OPM WOPT / WWPT (sm3).
Injectors: CMG stores injected water in WATVOLSC; compared with OPM WWIT.
Pass/fail: every producer with CMG oil > 0 must have final oil within tol.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import resfo


def read_cmg(sr3):
    with h5py.File(sr3, "r") as f:
        w = f["TimeSeries/WELLS"]
        table = w["WellTable"][:]
        names = [r[2].decode().strip() for r in table]
        is_inj = [r[3].decode().startswith("INJ") for r in table]
        vars_ = [v.decode().strip() for v in w["Variables"][:]]
        data, t = w["Data"][:], w["Timesteps"][:]
    out = {}
    for j, n in enumerate(names):
        wat = data[:, vars_.index("WATVOLSC"), j]
        out[n] = dict(days=t, injector=is_inj[j], WOPT=data[:, vars_.index("OILVOLSC"), j],
                      WWPT=np.zeros_like(wat) if is_inj[j] else wat, WWIT=np.abs(wat) if is_inj[j] else np.zeros_like(wat))
    return out


def read_opm(unsmry):
    unsmry = Path(unsmry)
    spec = {k.strip(): v for k, v in resfo.read(unsmry.with_suffix(".SMSPEC")) if k.strip() in ("KEYWORDS", "WGNAMES")}
    kw = [x.decode().strip() for x in spec["KEYWORDS"]]
    nm = [x.decode().strip() for x in spec["WGNAMES"]]
    params = [v for k, v in resfo.read(unsmry) if k.strip() == "PARAMS"]
    arr = np.array(params)
    out = {}
    for i, (k, n) in enumerate(zip(kw, nm)):
        if k in ("WOPT", "WWPT", "WWIT"):
            out.setdefault(n, {})[k] = arr[:, i]
    return out


def compare(sr3, unsmry, tol=0.10):
    """Return (ok, rows); rows = (well, quantity, cmg, opm, rel_diff) for producers (WOPT, WWPT) and injectors (WWIT)."""
    cmg, opm = read_cmg(sr3), read_opm(unsmry)
    rows = []
    for w in cmg:
        for k in (("WWIT",) if cmg[w]["injector"] else ("WOPT", "WWPT")):
            c, o = float(cmg[w][k][-1]), float(opm[w][k][-1])
            rows.append((w, k, c, o, (o - c) / c if c else float("nan")))
    for w, k, c, o, rel in rows:
        print(f"{w:4s} {k}  CMG {c:14.1f}  OPM {o:14.1f}  diff {rel * 100:+6.2f}%")
    ok = all(abs(rel) <= tol for w, k, c, o, rel in rows if k == "WOPT" and c > 0)
    return ok, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sr3"); ap.add_argument("unsmry"); ap.add_argument("--tol", type=float, default=0.10)
    a = ap.parse_args()
    ok, _ = compare(a.sr3, a.unsmry, a.tol)
    sys.exit(0 if ok else 1)
