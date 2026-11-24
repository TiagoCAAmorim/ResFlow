"""Cut SPE10 permeability patches, assign them to train / validation / test by whole layer, and
write them as CMG IMEX (default) or OPM Flow include files.

    python generate_spe10_patches.py --out output_patches            # downloads SPE10 if needed
    python generate_spe10_patches.py --out out --perm-file PERMI.inc --format opm

See docs/02_build_decks.md. No reservoir simulation is run here.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from defaults import DEFAULT_SEED, out_dir_in_use  # noqa: E402
import layer_split as ls  # noqa: E402
import patch_layout as pl  # noqa: E402
import perm_io  # noqa: E402

HERE = Path(__file__).resolve().parent
LAYERS = list(range(1, 36))
SIZE_COLORS = {"32x32": "tab:red", "60x60": "tab:blue", "120x60": "tab:green"}
MANIFEST_COLS = ["family_id", "set", "layer", "layer_end", "size", "n_i", "n_j", "n_k", "n_cells", "x_start", "y_start",
                 "size_x_original", "size_y_original", "rotation", "mean_log10_kx", "std_log10_kx",
                 "min_kx", "max_kx"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (must not exist or be empty)")
    p.add_argument("--perm-file", required=True,
                   help="PERMI.inc (one block) or spe_perm.dat (three blocks); download_spe10.py fetches it")
    p.add_argument("--layer-split", help="CSV with columns layer,set (train / validation / test / excluded), e.g. "
                                         "split.csv of layer_correlation.py. Replaces the built-in Tarbert split "
                                         "(layers 1-35); excluded layers are not used")
    p.add_argument("--nk", type=int, default=1,
                   help="layers per patch: each patch is a 32x32xNK (etc.) block of NK consecutive layers; the sets are "
                        "made of whole blocks, so each needs runs of at least NK consecutive layers (default: %(default)s). "
                        "With NK > 1 and no --layer-split the built-in split is test 31-35, validation 26-30 (or "
                        "--val-layers), train the rest of 1-30, and the validation search is not run")
    p.add_argument("--train-stride", type=int, default=None,
                   help="train blocks start every this many layers inside a run of train layers; 1 = sliding blocks "
                        "that share layers (default: NK, i.e. disjoint blocks). Validation and test blocks never overlap")
    p.add_argument("--format", choices=["imex", "opm"], default="imex", help="patch file format: imex = bare value list (needed by build_decks.py), opm = PERMX block (default: %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="do not write patch files")
    p.add_argument("--debug-npy", action="store_true", help="also write .npy copies of the patches")
    p.add_argument("--n-train", type=int, default=14, help="32x32 patches cut from each train layer (default: %(default)s)")
    p.add_argument("--n-eval", type=int, default=6, help="32x32 patches cut from each validation and test layer (default: %(default)s)")
    p.add_argument("--max-overlap", type=float, default=0.5, help="largest allowed overlap between two train patches of the same layer, as a fraction of the patch area (default: %(default)s)")
    p.add_argument("--max-corr-val", type=float, default=0.5,
                   help="validation search: max whole-layer correlation of log10(Kx) between a train and a validation layer (default: %(default)s)")
    p.add_argument("--max-corr-test", type=float, default=None, help="validation search: max correlation between a train and a test layer (default: the value of the split train 1-25 / val 26-30, recomputed = 0.628)")
    p.add_argument("--std-tol", type=float, default=0.2,
                   help="validation search: the std of log10(Kx) of the validation layers may differ from train by less than this fraction (default: %(default)s)")
    p.add_argument("--val-layers", help="skip the search; comma list, e.g. 26,27,28,29,30 (not with --layer-split)")
    p.add_argument("--rotation-k", type=int, default=1, help="the 120x60 patches are rotated with numpy.rot90(k) (k=1: counterclockwise; default: %(default)s)")
    p.add_argument("--avoid-colocated", type=float, default=None, metavar="R",
                   help="resample val/test 32x32 x offsets whose correlation with a co-located train "
                        "patch exceeds R (up to 50 tries)")
    return p.parse_args(argv)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- planning
def window(logk, patch, start, nk):
    """log10(Kx) of a window over layers start .. start+nk-1, averaged over the layers (x, y)."""
    x, y, w, h = patch
    return logk[x:x + w, y:y + h, start - 1:start - 1 + nk].mean(axis=2).ravel()


def colocation(logk, patch, start, train_starts, nk=1):
    """Pearson correlation of log10(Kx) of a window with the same window in each train block."""
    a = window(logk, patch, start, nk)
    return np.array([np.corrcoef(a, window(logk, patch, t, nk))[0, 1] for t in train_starts])


def runs_of(layers):
    """[(first, last)] of the runs of consecutive layers in `layers`."""
    out = []
    for k in sorted(layers):
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [tuple(r) for r in out]


def block_starts(layers, nk, stride):
    """First layers of the blocks of nk consecutive layers cut from `layers`: in each run of consecutive layers
    from its start, every `stride` layers while the block fits."""
    return [a for first, last in runs_of(layers) for a in range(first, last - nk + 2, stride)]


def read_layer_split(path, nk):
    """(assign, val, train) from a CSV with columns layer,set. Layers marked 'excluded' are left out."""
    df = pd.read_csv(path)
    if not {"layer", "set"} <= set(df.columns):
        raise RuntimeError(f"{path}: needs the columns layer and set")
    bad = sorted(set(df["set"]) - {"train", "validation", "test", "excluded"})
    if bad:
        raise RuntimeError(f"{path}: unknown set(s) {bad}")
    if df["layer"].duplicated().any():
        raise RuntimeError(f"{path}: a layer appears twice")
    if not df["layer"].between(1, nk).all():
        raise RuntimeError(f"{path}: layers must be between 1 and {nk}")
    df = df[df["set"] != "excluded"]
    assign = {int(k): s for k, s in zip(df["layer"], df["set"])}
    for s in ("train", "validation", "test"):
        if s not in assign.values():
            raise RuntimeError(f"{path}: no layer in set {s}")
    val = sorted(k for k, s in assign.items() if s == "validation")
    train = sorted(k for k, s in assign.items() if s == "train")
    return assign, val, train


def build_plan(args, K, logk):
    """Layer assignment and patch positions. Deterministic for a given seed."""
    if args.layer_split:
        if args.val_layers:
            raise RuntimeError("--layer-split and --val-layers cannot be used together")
        assign, val, train = read_layer_split(args.layer_split, K.shape[2])
        search, X = None, None
    elif args.val_layers or args.nk > 1:
        X = ls.layer_log_matrix(K, LAYERS)
        val = sorted(int(v) for v in args.val_layers.split(",")) if args.val_layers else list(ls.REF_VAL)
        train = [k for k in range(1, 31) if k not in val]
        search = None
    else:
        X = ls.layer_log_matrix(K, LAYERS)
        search = ls.search_validation(X, args.max_corr_val, args.max_corr_test, args.std_tol)
        val, train = search.val_layers, search.train_layers
    if not args.layer_split:
        assign = {**{k: "train" for k in train}, **{k: "validation" for k in val},
                  **{k: "test" for k in ls.TEST_LAYERS}}

    nk = args.nk
    stride = args.train_stride or nk
    notes, patches = [], []   # patches: dict(layer = first layer of the block, nk, set, size, x, y, w, h, rotation)

    # blocks of nk consecutive layers; a block is the unit that a single layer is when nk = 1
    blocks = {}
    for s in ("train", "validation", "test"):
        layers_s = [k for k, v in assign.items() if v == s]
        starts = block_starts(layers_s, nk, stride if s == "train" else nk)
        if not starts:
            raise RuntimeError(f"set {s}: no run of {nk} consecutive layers in {sorted(layers_s)}")
        blocks.update({a: s for a in starts})
        used = {k for a in starts for k in range(a, a + nk)}
        if nk > 1 and used != set(layers_s):
            notes.append(f"{s}: layers {sorted(set(layers_s) - used)} are in no block of {nk} consecutive layers "
                         "and are not used")
    train = sorted(a for a, s in blocks.items() if s == "train")   # first layers of the train blocks

    def add(layer, size, box, rot):
        patches.append(dict(layer=layer, nk=nk, set=blocks[layer], size=size, x=box[0], y=box[1],
                            w=box[2], h=box[3], rotation=rot))

    coloc_rows = []
    for layer in sorted(blocks):
        s = blocks[layer]
        if s == "train":
            boxes = pl.place_train(pl.layer_rng(args.seed, layer), args.n_train, 32, args.max_overlap)
            if len(boxes) < args.n_train:
                notes.append(f"layer {layer}: placed {len(boxes)} of {args.n_train} train patches "
                             f"under max_overlap={args.max_overlap}")
            for b in boxes:
                add(layer, "32x32", b, "none")
        else:
            rng = pl.layer_rng(args.seed, layer)
            boxes = pl.place_spread(rng, args.n_eval, 32)
            rng2 = pl.layer_rng(args.seed, layer, 1)
            for idx, b in enumerate(boxes):
                cor = colocation(logk, b, layer, train, nk)
                if args.avoid_colocated is not None and cor.max() > args.avoid_colocated:
                    best, best_cor = b, cor
                    for _ in range(50):
                        cand = pl.resample_x(rng2, b)
                        c2 = colocation(logk, cand, layer, train, nk)
                        if c2.max() < best_cor.max():
                            best, best_cor = cand, c2
                        if c2.max() <= args.avoid_colocated:
                            break
                    if best_cor.max() > args.avoid_colocated:
                        notes.append(f"layer {layer} 32x32 #{idx}: no position under R="
                                     f"{args.avoid_colocated}; kept least-correlated "
                                     f"({best_cor.max():.3f})")
                    b, cor = best, best_cor
                boxes[idx] = b
                coloc_rows.append(dict(layer=layer, set=s, index=idx, x_start=b[0], y_start=b[1],
                                       max_corr=cor.max(), max_corr_train_layer=train[int(cor.argmax())]))
            for b in boxes:
                add(layer, "32x32", b, "none")
        if s == "test":
            for b in pl.POS_60:
                add(layer, "60x60", b, "none")
            for b in pl.POS_120:
                add(layer, "120x60", b, f"rot90_k{args.rotation_k}")

    counters = {}
    for p in patches:
        key = (p["layer"], p["size"])
        span = f"{p['layer']:02d}" if nk == 1 else f"{p['layer']:02d}-{p['layer'] + nk - 1:02d}"
        p["family_id"] = f"L{span}_{p['size']}_{counters.get(key, 0):02d}"
        counters[key] = counters.get(key, 0) + 1
    return dict(assign=assign, search=search, patches=patches, notes=notes, nk=nk, blocks=blocks,
                coloc=pd.DataFrame(coloc_rows), val=val, train=sorted(k for k, v in assign.items() if v == "train"),
                X=X, layers=sorted(assign))


def extract(K, p, rotation_k):
    """Patch array G[i, j, k] as written to the file (rotated in the i-j plane if requested)."""
    g = K[p["x"]:p["x"] + p["w"], p["y"]:p["y"] + p["h"], p["layer"] - 1:p["layer"] - 1 + p["nk"]]
    return np.rot90(g, k=rotation_k, axes=(0, 1)) if p["rotation"] != "none" else g.copy()


def build_manifest(K, plan, rotation_k):
    rows, arrays = [], {}
    for p in plan["patches"]:
        g = extract(K, p, rotation_k)
        arrays[p["family_id"]] = g
        lg = np.log10(g)
        rows.append(dict(family_id=p["family_id"], set=p["set"], layer=p["layer"], layer_end=p["layer"] + p["nk"] - 1,
                         size=p["size"], n_i=g.shape[0], n_j=g.shape[1], n_k=g.shape[2], n_cells=g.size,
                         x_start=p["x"], y_start=p["y"], size_x_original=p["w"], size_y_original=p["h"],
                         rotation=p["rotation"], mean_log10_kx=lg.mean(), std_log10_kx=lg.std(),
                         min_kx=g.min(), max_kx=g.max()))
    return pd.DataFrame(rows, columns=MANIFEST_COLS), arrays


def plan_checksums(args, K, logk):
    plan = build_plan(args, K, logk)
    man, _ = build_manifest(K, plan, args.rotation_k)
    assign = pd.DataFrame(sorted(plan["assign"].items()), columns=["layer", "set"])
    return (hashlib.sha256(man.to_csv(index=False).encode()).hexdigest(),
            hashlib.sha256(assign.to_csv(index=False).encode()).hexdigest()), plan


# ---------------------------------------------------------------- checks
def overlap_stats(plan):
    rows = []
    by = {}
    for p in plan["patches"]:
        by.setdefault((p["layer"], p["size"]), []).append((p["x"], p["y"], p["w"], p["h"]))
    for (layer, size), boxes in sorted(by.items()):
        fr = [pl.overlap_frac(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]]
        rows.append(dict(layer=layer, set=plan["assign"][layer], size=size, n_patches=len(boxes),
                         max_overlap=max(fr, default=0.0), mean_overlap=float(np.mean(fr)) if fr else 0.0))
    return pd.DataFrame(rows)


def run_checks(args, K, plan, man, arrays, ovl, files_dir):
    """Returns a list of (check, status, detail). Assertion-type checks raise on failure."""
    res = []
    assign = plan["assign"]

    sets = {s: sorted(k for k, v in assign.items() if v == s) for s in ("train", "validation", "test")}
    assert sum(len(v) for v in sets.values()) == len(set(sum(sets.values(), []))), "layer in two sets"
    assert sorted(sum(sets.values(), [])) == plan["layers"], "sets do not cover the used layers exactly"
    if args.layer_split:
        res.append(("1 layer disjointness", "PASS", f"no layer in two sets; sets as in {args.layer_split} "
                    f"(excluded layers unused)"))
    else:
        assert plan["layers"] == LAYERS, "sets do not cover layers 1-35 exactly"
        assert sets["test"] == ls.TEST_LAYERS, "test layers are not 31-35"
        if args.nk > 1:
            res.append(("1b blocks", "PASS", f"every set is cut into blocks of {args.nk} consecutive layers"))
        res.append(("1 layer disjointness", "PASS", "no layer in two sets; layers 1-35 covered; test = 31-35"))

    for p in plan["patches"]:
        assert 0 <= p["x"] and p["x"] + p["w"] <= pl.NX and 0 <= p["y"] and p["y"] + p["h"] <= pl.NY, p
    exp_shape = {"32x32": (32, 32), "60x60": (60, 60), "120x60": (120, 60)}
    for r in man.itertuples():
        assert (r.n_i, r.n_j) == exp_shape[r.size], (r.family_id, r.n_i, r.n_j)
        assert arrays[r.family_id].shape == (r.n_i, r.n_j, r.n_k)
        assert r.n_k == args.nk and r.n_i * r.n_j * r.n_k == r.n_cells
        if r.rotation == "none":
            assert (r.size_x_original, r.size_y_original) == (r.n_i, r.n_j)
        else:
            assert (r.size_x_original, r.size_y_original) == (r.n_j, r.n_i)
    res.append(("2 geometry", "PASS", "all patches inside their layer; shapes as requested; n_i/n_j match grids"))

    o = ovl.set_index(["size", "set"], drop=False)
    for (size, s), g in ovl.groupby(["size", "set"]):
        if size == "60x60" or (size == "32x32" and s != "train"):
            assert g.max_overlap.max() == 0, f"{size} {s} patches overlap"
        if size == "32x32" and s == "train":
            assert g.max_overlap.max() <= args.max_overlap + 1e-12, "train overlap above limit"
    for layer in sorted(a for a, s in plan["blocks"].items() if s == "test"):
        a, b = [(p["x"], p["y"], p["w"], p["h"]) for p in plan["patches"]
                if p["layer"] == layer and p["size"] == "120x60"]
        assert pl.overlap_frac(a, b) * a[2] * a[3] == 60 * 20, "120x60 overlap is not 20 rows"
    res.append(("3 overlap rules", "PASS", "60x60 and val/test 32x32 disjoint; train within "
                f"{args.max_overlap}; 120x60 pairs overlap by exactly 20 cells"))

    rng = np.random.default_rng(args.seed)
    ids = list(man.family_id)
    sample = [ids[i] for i in rng.choice(len(ids), size=min(100, len(ids)), replace=False)]
    byid = {p["family_id"]: p for p in plan["patches"]}
    ext = {"imex": ".inc", "opm": ".inc"}[args.format]
    for fid in sample:
        g, p = arrays[fid], byid[fid]
        assert np.isfinite(g).all() and (g > 0).all()
        src = K[p["x"]:p["x"] + p["w"], p["y"]:p["y"] + p["h"], p["layer"] - 1:p["layer"] - 1 + p["nk"]]
        undone = np.rot90(g, k=-args.rotation_k, axes=(0, 1)) if p["rotation"] != "none" else g
        assert np.array_equal(undone, src), f"{fid}: values differ from source"
        if files_dir is not None:
            back = perm_io.parse_values(files_dir / f"{fid}{ext}")
            if args.format == "opm":
                back = back[:g.size]
            assert back.size == g.size and np.array_equal(back, g.T.ravel()), f"{fid}: file differs"
    res.append(("4 values", "PASS", f"{len(sample)} sampled patches positive, finite, identical to source"
                + ("" if files_dir is None else " (files read back)")))
    return res


# ---------------------------------------------------------------- reporting
def plot_maps(path, logk, plan):
    layers = plan["layers"]
    ncol = 7
    nrow = -(-len(layers) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 4.8 * nrow), squeeze=False)
    vmin, vmax = np.percentile(logk[:, :, [k - 1 for k in layers]], [1, 99])
    for ax in axes.ravel()[len(layers):]:
        ax.axis("off")
    for ax, layer in zip(axes.ravel(), layers):
        ax.imshow(logk[:, :, layer - 1].T, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        for p in plan["patches"]:
            if p["layer"] <= layer < p["layer"] + p["nk"]:
                ax.add_patch(Rectangle((p["x"] - 0.5, p["y"] - 0.5), p["w"], p["h"], fill=False,
                                       ec=SIZE_COLORS[p["size"]], lw=0.8))
        ax.set_title(f"layer {layer} ({plan['assign'][layer]})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("log10(Kx) with patch rectangles: red 32x32, blue 60x60, green 120x60 (before rotation)")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def distribution_report(args, K, man, arrays):
    cells = {s: np.concatenate([np.log10(arrays[f]).ravel() for f in man.family_id[man.set == s]])
             for s in ("train", "validation", "test")}
    tr = cells["train"]
    rows = []
    for s, v in cells.items():
        rows.append(dict(set=s, n_cells=v.size, mean=v.mean(), std=v.std(),
                         diff_vs_train_in_train_std=(v.mean() - tr.mean()) / tr.std(),
                         ks_vs_train=0.0 if s == "train" else stats.ks_2samp(v, tr).statistic))
    return pd.DataFrame(rows)


def expected_counts(args, blocks):
    n = {s: sum(v == s for v in blocks.values()) for s in ("train", "validation", "test")}
    return {("train", "32x32"): n["train"] * args.n_train, ("validation", "32x32"): n["validation"] * args.n_eval,
            ("test", "32x32"): n["test"] * args.n_eval, ("test", "60x60"): 3 * n["test"],
            ("test", "120x60"): 2 * n["test"]}


def md_table(df, floatfmt=".4f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(path, args, K, plan, man, ovl, dist, checks, count_df, coloc, note_input):
    s = plan["search"]
    L = ["# SPE10 patch generation summary", "",
         "Numbers only; no physical interpretation.", "", "## Validation layers", ""]
    if args.layer_split:
        X = ls.layer_log_matrix(K, plan["layers"])
        pos = {k: i for i, k in enumerate(plan["layers"])}
        C = np.corrcoef(X)
        tr = [pos[k] for k in plan["train"]]
        ev = [pos[k] for k in plan["layers"] if plan["assign"][k] != "train"]
        L += [f"Layer sets read from `{args.layer_split}` (no search); layers not listed there or marked excluded are unused.",
              f"- train {plan['train']}", f"- validation {plan['val']}",
              f"- test {sorted(k for k, v in plan['assign'].items() if v == 'test')}",
              f"- max correlation of log10(Kx) between a train layer and a validation/test layer: "
              f"{C[np.ix_(tr, ev)].max():.3f}"]
    elif s is None:
        met = ls.set_metrics(plan["X"], plan["val"])
        L += [f"Validation layers given with --val-layers (no search): {plan['val']}",
              f"- mean difference (val - train): {met['mean_diff']:+.3f} train std; std relative difference "
              f"{met['std_rel']:+.3f}; KS {met['ks']:.3f}",
              f"- max correlation train-val {met['max_corr_tv']:.3f}; train-test {met['max_corr_tt']:.3f}"]
    else:
        top = s.table.iloc[0]
        L += [f"Chosen by exhaustive search over C(30,5) = 142506 combinations: **{plan['val']}**",
              f"- combinations satisfying all constraints: {s.n_feasible}",
              f"- thresholds: max corr train-val < {s.max_corr_val}; max corr train-test <= "
              f"{s.max_corr_test:.4f}; |std relative difference| < {s.std_tol}",
              f"- mean difference (val - train): {top.mean_diff_in_train_std:+.3f} train std; "
              f"std relative difference {top.std_rel_diff:+.3f}; KS {top.ks:.3f}",
              f"- max correlation train-val {top.max_corr_train_val:.3f}; train-test "
              f"{top.max_corr_train_test:.3f}",
              f"- split A (26-30) reference: mean difference {s.ref['mean_diff']:+.3f} train std, "
              f"max corr train-val {s.ref['max_corr_tv']:.3f}, train-test {s.ref['max_corr_tt']:.3f}"]
        L += [f"- NOTE: {n}" for n in s.notes]
    L += ["", "## Counts per set and size (actual vs expected)", "", md_table(count_df), "",
          "## Overlap statistics (pairwise fraction of patch area, same layer and size)", "",
          "Per set:", "",
          md_table(ovl.groupby(["set", "size"]).agg(max_overlap=("max_overlap", "max"),
                                                    mean_overlap=("mean_overlap", "mean")).reset_index()),
          "", "Per layer: see the table in `overlap_by_layer.csv`.", "",
          "## Distribution of log10(Kx) in the patches", "", md_table(dist), "",
          "## Checks", ""]
    L += [f"- {n}: **{st}** - {d}" for n, st, d in checks]
    L += ["", "## Deviations, notes and rotation", "",
          f"- Rotation of the 120x60 patches: `numpy.rot90(patch[x, y], k={args.rotation_k})` "
          f"({'counterclockwise' if args.rotation_k % 4 == 1 else 'k=' + str(args.rotation_k)}). "
          "The sense of rotation was not specified by the project owner; k=1 is the default.",
          f"- Layers per patch: nk = {args.nk}" + ("" if args.nk == 1 else f" (blocks of consecutive layers; train stride "
          f"{args.train_stride or args.nk}); `layer` in the manifest is the first layer of the block, `layer_end` the last") + ".",
          f"- Input: {note_input or 'single block of 1,122,000 values'}",
          "- File format: compact include format (a bare list of values) for IMEX, plus an OPM variant (`--format opm`).",
          "- Std in this folder is the population standard deviation (ddof=0)."]
    L += [f"- {n}" for n in plan["notes"]]
    if args.max_corr_test is not None or args.val_layers or args.avoid_colocated is not None \
            or args.n_train != 14 or args.n_eval != 6 or args.max_overlap != 0.5 \
            or args.max_corr_val != 0.5 or args.std_tol != 0.2:
        L.append("- NON-DEFAULT parameters in use: see `config_used.yaml`.")
    if len(coloc):
        L += ["", "## Co-location correlation (report only)", "",
              f"Max over validation/test 32x32 patches: {coloc.max_corr.max():.3f}; "
              f"mean of per-patch max: {coloc.max_corr.mean():.3f} (`colocation_corr.csv`)."]
    Path(path).write_text("\n".join(L) + "\n")


README = """# SPE10 permeability patches

Generated by `generate_spe10_patches.py`.

Run: `python generate_spe10_patches.py {cmdline}`

## Conventions
- Source: SPE10 Model 2 Kx (md), grid 60 x 220 x 85, x fastest; layers are 1-based; Tarbert = layers 1-35.
- Ky = Kx; Kz is defined in the simulation input file. Only Kx is written.
- Values are cropped (and, for 120x60, rotated) only; no rescaling. The project simulations use
  50 x 50 x 5 m cells; the script does nothing about cell size.
- A patch has n_k = {nk} layers (consecutive; `layer` in the manifest = first, `layer_end` = last). Files hold
  `n_i * n_j * n_k` values, i fastest, then j, then k.
- 120x60: a (60, 120) crop [x, y] is rotated with `numpy.rot90(patch, k={k})`, giving (120, 60) [i, j].
- Format `{fmt}`: {fmt_text}
- Sets are assigned by whole layer; always read the set of a family from `manifest.csv`.

## Files
manifest.csv, layer_assignment.csv, validation_search.csv, colocation_corr.csv, overlap_by_layer.csv,
config_used.yaml, summary.md, patch_map_layers.png, `perm/<family_id>.inc`.

## Versions
python {py}, numpy {np}, pandas {pd}, scipy {sp}, matplotlib {mpl}, pyyaml {yml}
"""
FMT_TEXT = {"imex": "bare value list with `**` header comments; use as `PERMI ALL` + `INCLUDE 'file'` "
                    "and the same for `PERMJ ALL`.",
            "opm": "`PERMX ... /` block followed by `COPY PERMX PERMY /`; INCLUDE it in the GRID section."}


# ---------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if args.nk < 1 or (args.train_stride is not None and args.train_stride < 1):
        sys.exit("--nk and --train-stride must be at least 1")
    if out_dir_in_use(out):
        sys.exit(f"{out} is not empty; choose a new or an empty folder")

    perm_file = Path(args.perm_file)
    K, note_input = perm_io.load_kx(perm_file)
    logk = np.log10(K)

    out.mkdir(parents=True, exist_ok=True)
    try:
        (c1, c2), plan = plan_checksums(args, K, logk)
    except RuntimeError as e:
        sys.exit(f"stopped: {e}")
    (d1, d2), _ = plan_checksums(args, K, logk)
    assert (c1, c2) == (d1, d2), "two runs with the same seed gave different outputs"
    checks_repro = ("5 reproducibility", "PASS", f"manifest sha256 {c1[:12]}, layer_assignment sha256 {c2[:12]}")

    man, arrays = build_manifest(K, plan, args.rotation_k)
    assign_df = pd.DataFrame(sorted(plan["assign"].items()), columns=["layer", "set"])
    ovl = overlap_stats(plan)

    files_dir = None
    if not args.dry_run:
        files_dir = out / "perm"
        files_dir.mkdir()
        info = {p["family_id"]: p for p in plan["patches"]}
        for fid, g in arrays.items():
            p = info[fid]
            perm_io.write_patch(files_dir / f"{fid}.inc", g,
                                dict(family_id=fid, set=p["set"], layer=p["layer"], x_start=p["x"],
                                     y_start=p["y"], rotation=p["rotation"], layer_end=p["layer"] + p["nk"] - 1),
                                fmt=args.format)
            if args.debug_npy:
                (out / "debug_npy").mkdir(exist_ok=True)
                np.save(out / "debug_npy" / f"{fid}.npy", g)

    checks = run_checks(args, K, plan, man, arrays, ovl, files_dir) + [checks_repro]

    dist = distribution_report(args, K, man, arrays)
    checks.append(("6 distribution report", "REPORT", "see table above; split not changed"))
    coloc = plan["coloc"]
    checks.append(("7 co-location correlation", "REPORT",
                   f"max {coloc.max_corr.max():.3f} over {len(coloc)} patches"
                   + (f"; --avoid-colocated {args.avoid_colocated} active" if args.avoid_colocated else "")))

    exp = expected_counts(args, plan["blocks"])
    act = man.groupby(["set", "size"]).size()
    count_df = pd.DataFrame([dict(set=k[0], size=k[1], actual=int(act.get(k, 0)), expected=v)
                             for k, v in exp.items()])
    counts_ok = bool((count_df.actual == count_df.expected).all())
    checks.append(("8 expected counts", "PASS" if counts_ok else "FAIL",
                   "match spec section 8" if counts_ok else "MISMATCH, see counts table"))

    man.to_csv(out / "manifest.csv", index=False)
    assign_df.to_csv(out / "layer_assignment.csv", index=False)
    ovl.to_csv(out / "overlap_by_layer.csv", index=False)
    coloc.to_csv(out / "colocation_corr.csv", index=False)
    if args.layer_split:
        shutil.copy(args.layer_split, out / "layer_split_used.csv")
    elif plan["search"] is not None:
        plan["search"].table.to_csv(out / "validation_search.csv", index=False)
    else:
        pd.DataFrame([dict(label="given_by_user", validation_layers=" ".join(map(str, plan["val"])),
                           **{k: v for k, v in ls.set_metrics(plan["X"], plan["val"]).items()})]
                     ).to_csv(out / "validation_search.csv", index=False)
    plot_maps(out / "patch_map_layers.png", logk, plan)
    write_summary(out / "summary.md", args, K, plan, man, ovl, dist, checks, count_df, coloc, note_input)

    cfg = {k: v for k, v in vars(args).items()}
    cfg.update(perm_file=str(perm_file.resolve()), perm_file_sha256=sha256(perm_file))
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    (out / "README.md").write_text(README.format(
        cmdline=" ".join(sys.argv[1:] if argv is None else argv), k=args.rotation_k, nk=args.nk, fmt=args.format,
        fmt_text=FMT_TEXT[args.format], py=sys.version.split()[0], np=np.__version__, pd=pd.__version__,
        sp=scipy.__version__, mpl=matplotlib.__version__, yml=yaml.__version__))

    print(f"validation layers: {plan['val']}")
    print(f"{len(man)} patches -> {out}")
    for n, st, d in checks:
        print(f"[{st}] {n}: {d}")
    if not counts_ok:
        sys.exit("expected counts not met; see summary.md")


if __name__ == "__main__":
    main()
