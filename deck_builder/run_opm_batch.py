"""Convert every CMG .dat in a folder to OPM Flow, run them in parallel, and compare to the CMG .sr3.

Usage:
    python run_opm_batch.py $DATA/decks/decks [-j 16] [--tol 0.10] [--no-restart]
Output: <folder>_opm/<case>/ (deck + out/), and <folder>_opm/comparison.csv
Engine (--engine): "wheel" (default) runs the OPM Python wheels (`pip install opm opm-simulators`, Linux x86_64 only, no MPI)
in a subprocess per deck; "flow" runs the `flow` binary. Both write the results to <case>/out/. --timeout is per deck.
--python selects the interpreter that has the wheels (default: the current one).
opm_time_s = wall time of the `flow` process only (single thread, run on native disk, -j cases in parallel).
"""
import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cmg_to_opm import convert
from compare_cmg_opm import compare


def run_case(args):
    dat, tol, restart, engine, timeout, python = args
    dat = Path(dat)
    row = dict(case=dat.stem, opm_time_s="", imex_time_s="", status="", oil_diff_pct="", water_diff_pct="", inj_diff_pct="", detail="")
    perf = dat.parent / "performance.csv"
    if perf.exists():
        with open(perf) as f:
            for r_ in csv.DictReader(f):
                if r_["simulation"] == dat.stem:
                    row["imex_time_s"] = r_["elapsed_time_seconds"]
    try:
        data = convert(dat, restart=restart)
        env = {"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"}
        if engine == "wheel":
            env["HOME"] = str(Path.home())
        # Run in a local temporary folder (Flow writes at every report step, slow on network or mounted drives); copy results back after.
        with tempfile.TemporaryDirectory(prefix=f"opm_{dat.stem}_") as tmp:
            tmp = Path(tmp)
            for f in data.parent.iterdir():
                if f.is_file():
                    shutil.copy2(f, tmp / f.name)
            t0 = time.perf_counter()
            if engine == "flow":
                cmd = ["flow", data.name, "--output-dir=out", "--enable-terminal-output=false"]
            else:
                cmd = [python, "-c", f"from opm.simulators import BlackOilSimulator; BlackOilSimulator('{data.name}').run()"]
            try:
                r = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout,
                                   stdin=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                row.update(opm_time_s=f"{time.perf_counter() - t0:.2f}", status="TIMEOUT", detail=f"> {timeout} s")
                return row
            if engine == "wheel":   # the wheel writes next to the .DATA: collect the results in out/
                (tmp / "out").mkdir(exist_ok=True)
                for f in tmp.iterdir():
                    if f.is_file() and f.name not in {p.name for p in data.parent.iterdir()}:
                        shutil.move(str(f), tmp / "out" / f.name)
            row["opm_time_s"] = f"{time.perf_counter() - t0:.2f}"
            if (data.parent / "out").exists():
                shutil.rmtree(data.parent / "out")
            if (tmp / "out").exists():
                shutil.copytree(tmp / "out", data.parent / "out")
        (data.parent / "run.log").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            row.update(status="FLOW_FAILED", detail=(r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1][:200])
            return row
        sr3 = dat.with_suffix(".sr3")
        if not sr3.exists():
            row["status"] = "NO_SR3"
            return row
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ok, rows = compare(sr3, data.parent / "out" / f"{dat.stem.upper()}.UNSMRY", tol)
        tot = {k: [sum(r[2] for r in rows if r[1] == k), sum(r[3] for r in rows if r[1] == k)] for k in ("WOPT", "WWPT", "WWIT")}
        for key, k in (("oil_diff_pct", "WOPT"), ("water_diff_pct", "WWPT"), ("inj_diff_pct", "WWIT")):
            row[key] = f"{(tot[k][1] / tot[k][0] - 1) * 100:.2f}" if tot[k][0] else ""
        worst = max(((abs(r[4]), r[0]) for r in rows if r[1] == "WOPT" and r[2] > 0), default=(0, ""))
        row["detail"] = f"worst producer oil diff {worst[0] * 100:.1f}% ({worst[1]})"
        row["status"] = "OK" if ok else "OUT_OF_TOL"
    except Exception as e:
        row.update(status="CONVERT_ERROR", detail=f"{type(e).__name__}: {e}"[:200])
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("-j", type=int, default=16); ap.add_argument("--tol", type=float, default=0.10)
    ap.add_argument("--engine", choices=["wheel", "flow"], default="wheel", help="OPM wheels (default) or the flow binary")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds per deck")
    ap.add_argument("--python", default=sys.executable, help="interpreter with the opm wheels (engine wheel)")
    ap.add_argument("--only", nargs="*", help="run only these deck names (default: every .dat of the folder)")
    ap.add_argument("--no-restart", action="store_true", help="skip per-step grid restart output (faster)")
    a = ap.parse_args()
    folder = Path(a.folder).resolve()
    decks = sorted(folder.glob("*.dat"))
    if a.only:
        decks = [d for d in decks if d.stem in a.only]
    print(f"{len(decks)} decks")
    with ProcessPoolExecutor(a.j) as ex:
        rows = list(ex.map(run_case, [(d, a.tol, not a.no_restart, a.engine, a.timeout, a.python) for d in decks]))
    out = folder.parent / f"{folder.name}_opm" / "comparison.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(Counter(r["status"] for r in rows)); print(out)
