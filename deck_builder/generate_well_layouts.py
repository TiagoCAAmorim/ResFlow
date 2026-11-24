"""Well layouts for the patches written by generate_spe10_patches.py.

    python generate_well_layouts.py --patches _output_patches --out _layouts
    python generate_well_layouts.py --patches _output_patches --out _layouts_test --sets test \
        --well-counts 2/1,8/4 --max-patches 3 --jobs 4

Reads <patches>/manifest.csv and <patches>/perm/<family_id>.inc (IMEX or OPM format). Writes in --out
(must not exist or be empty): well_locations.csv, skipped_cases.csv, config_used.yaml, grids/ (one CORNERS include and
base deck per grid size, shared by all patches; a patch of nk > 1 layers is placed on its layer-averaged
map, see well_layout.collapse_layers). See well_layout.py for the rules.
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

import well_layout as wl
from defaults import DEFAULT_SEED, out_dir_in_use


def _work(job):
    return job[0], wl.place_patch(*job[:-1], **job[-1])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--patches", required=True, help="output folder of generate_spe10_patches.py")
    p.add_argument("--out", required=True, help="output folder (must not exist or be empty)")
    p.add_argument("--well-counts", default=",".join(f"{p}/{i}" for p, i in wl.DEFAULT_COUNTS),
                   help="P/I pairs, comma separated (default: %(default)s)")
    p.add_argument("--test-well-counts", default="1/0",
                   help="extra P/I pairs for the test set only; 1/0 = a single producer, no injector "
                        "(default: %(default)s; empty string for none)")
    p.add_argument("--sets", default="train,validation,test", help="which sets to process (default: %(default)s)")
    p.add_argument("--families", help="regex on family_id")
    p.add_argument("--max-patches", type=int, help="only the first N patches (testing)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    p.add_argument("--d-min", type=float, default=500.0, help="minimum distance between any two wells, in metres (default: %(default)s)")
    p.add_argument("--inj-factor", type=float, default=1.5, help="preferred minimum distance between injectors = factor * d-min (default: %(default)s)")
    p.add_argument("--cell-size", type=float, nargs=3, default=[50.0, 50.0, 5.0], metavar=("DX", "DY", "DZ"),
                   help="cell size in metres; must be the same in build_decks.py (default: 50 50 5)")
    p.add_argument("--n-iter", type=int, default=2000, help="simulated-annealing iterations per layout; more is better but slower (default: %(default)s)")
    p.add_argument("--rate", type=float, default=0.10,
                   help="discount rate of the drained-volume index Y_at: volume far from a well counts less (default: %(default)s)")
    p.add_argument("--x-base", type=float, default=15.0,
                   help="distance over which the discount --rate applies once, in equivalent metres (default: %(default)s)")
    p.add_argument("--w-bal", type=float, default=0.25,
                   help="weight of the penalty for wells draining very different volumes (default: %(default)s)")
    p.add_argument("--k-min", type=float, default=0.0,
                   help="minimum cell permeability in mD for a well; a layout with a well below it is invalid, 0 = off (default: %(default)s)")
    p.add_argument("--score-min-frac", type=float, default=0.0,
                   help="a layout is invalid if a well's Y_at is below this fraction of the mean Y_at of the other wells, 0 = off (default: %(default)s)")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="annealing runs (new seed each) tried for a valid layout before the well count is skipped (default: %(default)s)")
    p.add_argument("--jobs", type=int, default=1, help="patches processed in parallel (default: %(default)s)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    out, patches = Path(a.out), Path(a.patches).resolve()
    if out_dir_in_use(out):
        sys.exit(f"{out} is not empty; choose a new or an empty folder")
    if not (patches / "manifest.csv").exists():
        sys.exit(f"{patches}: no manifest.csv. Run generate_spe10_patches.py first; see docs/02_build_decks.md.")
    man = pd.read_csv(patches / "manifest.csv")
    man = man[man.set.isin(a.sets.split(","))]
    if a.families:
        man = man[man.family_id.str.contains(a.families, regex=True)]
    if a.max_patches:
        man = man.head(a.max_patches)
    if man.empty:
        sys.exit("no patches selected")
    counts = wl.parse_counts(a.well_counts)
    test_counts = [c for c in wl.parse_counts(a.test_well_counts) if c not in counts]
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(cell=tuple(a.cell_size), d_min=a.d_min, rate=a.rate, x_base=a.x_base, w_bal=a.w_bal,
              n_iter=a.n_iter, seed=a.seed, inj_factor=a.inj_factor,
              k_min=a.k_min, score_min_frac=a.score_min_frac, max_attempts=a.max_attempts)
    grid_dir = out / "grids"
    if "n_k" not in man:
        man["n_k"] = 1
    for ni, nj, nk in sorted(set(zip(man.n_i.astype(int), man.n_j.astype(int), man.n_k.astype(int)))):
        wl.ensure_grid(grid_dir, ni, nj, a.cell_size, nk)    # created once, before the workers start
    jobs = [(r.family_id, patches / "perm" / f"{r.family_id}.inc", int(r.n_i), int(r.n_j),
             counts + (test_counts if r.set == "test" else []),
             grid_dir, {**kw, "nk": int(r.n_k)}) for r in man.itertuples()]
    sets = dict(zip(man.family_id, man.set))

    rows, skipped = [], []
    with ProcessPoolExecutor(a.jobs) as ex:
        for fid, (r, s, room) in ex.map(_work, jobs):
            for row in r:
                row["set"] = sets[fid]
            rows += r
            skipped += [dict(family_id=fid, set=sets[fid], requested=f"{q[0]}/{q[1]}", reason=why,
                             room=room) for q, why in s]
            print(f"{fid}: room for {room} wells, {len(r)} cases, {len(s)} skipped", flush=True)

    df = pd.DataFrame(rows)
    pcols = sorted([c for c in df.columns if c[0] == "P" and c[-1] in "IJ" and c[1:-1].isdigit()],
                   key=lambda c: (int(c[1:-1]), c[-1]))
    icols = sorted([c for c in df.columns if c[0] == "I" and c[-1] in "IJ" and c[1:-1].isdigit()],
                   key=lambda c: (int(c[1:-1]), c[-1]))
    head = ["family_id", "set", "case_id", "n_prod", "n_inj", "requested_prod", "requested_inj", "reduced",
            "inj_spacing_relaxed", "inj_behind_injector", "score"]
    df = df[head + pcols + icols]
    df[pcols + icols] = df[pcols + icols].fillna(0).astype(int)   # 0 = well not used
    df.to_csv(out / "well_locations.csv", index=False)
    pd.DataFrame(skipped, columns=["family_id", "set", "requested", "reason", "room"]).to_csv(
        out / "skipped_cases.csv", index=False)
    (out / "config_used.yaml").write_text(yaml.safe_dump({**vars(a), "well_counts_parsed": counts, "test_well_counts_parsed": test_counts}))
    print(f"{len(df)} cases ({int(df.reduced.sum())} reduced), {len(skipped)} skipped -> {out}")


if __name__ == "__main__":
    main()
