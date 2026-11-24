"""Correlation between the layers of a permeability grid, and a train / validation / test split of a range of layers.

Works on any grid given as a list of values (i fastest, then j, then k), e.g. SPE10 (60 x 220 x 85). Used to
decide which layers the patches of a 2D-simulation dataset are cut from.

    python layer_correlation.py --out out --perm-file spe_perm.dat --ni 60 --nj 220 --nk 85 --k-init 36 --k-end 70
    python layer_correlation.py ... --window 32 --n-val 4 --n-test 6     # also report co-located 32x32 windows

What it does:
1. Pearson correlation of log10(Kx) between every pair of layers of the range (`layer_corr.csv`), and its
   mean / max as a function of the layer distance (`corr_vs_lag.csv`, `corr_vs_lag.png`).
2. Random search (seeded) for a split: n_val validation and n_test test layers such that no train layer
   has a correlation above --max-corr with a validation or test layer. Train layers that would break this
   are `excluded` (not used). Among the splits that keep the log10(Kx) mean and spread of the eval layers
   close to the train layers, the one with most train layers is chosen (`split.csv`, `split_candidates.csv`).
3. Only if --window is given: for the chosen split, the correlation of the same window between an eval layer and the train
   layers (`colocation_corr.csv`): this is what matters when patches are cut at the same place.

All statistics use log10(Kx) pooled over cells, population std (ddof=0). Layers are 1-based.
No reservoir simulation is run here.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from defaults import DEFAULT_SEED, out_dir_in_use  # noqa: E402
import perm_io  # noqa: E402

SET_COLORS = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:red", "excluded": "0.6"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output folder (must not exist or be empty)")
    p.add_argument("--perm-file", required=True,
                   help="permeability include file, values with i fastest, then j, then k; one block, "
                        "or three blocks (Kx, Ky, Kz) of which the first is used")
    p.add_argument("--ni", type=int, required=True, help="grid cells in i (x)")
    p.add_argument("--nj", type=int, required=True, help="grid cells in j (y)")
    p.add_argument("--nk", type=int, required=True, help="grid cells in k (layers)")
    p.add_argument("--k-init", type=int, required=True, help="first layer of the range, 1-based")
    p.add_argument("--k-end", type=int, required=True, help="last layer of the range, 1-based, included")
    p.add_argument("--n-val", type=int, default=4, help="validation layers (default: %(default)s)")
    p.add_argument("--n-test", type=int, default=4, help="test layers (default: %(default)s)")
    p.add_argument("--max-corr", type=float, default=0.3,
                   help="largest correlation of log10(Kx) allowed between a train layer and a validation/test layer, "
                        "and between a validation and a test layer (default: %(default)s)")
    p.add_argument("--min-train", type=int, default=12, help="smallest number of train layers accepted (default: %(default)s)")
    p.add_argument("--mean-tol", type=float, default=0.25,
                   help="|mean(eval) - mean(train)| must be below this many train std (default: %(default)s)")
    p.add_argument("--std-tol", type=float, default=0.2,
                   help="std of an eval set may differ from train by less than this fraction (default: %(default)s)")
    p.add_argument("--n-samples", type=int, default=200_000, help="random splits tried (default: %(default)s)")
    p.add_argument("--window", type=int, default=None,
                   help="patch size in cells (square). If given, report the correlation of the same window "
                        "between eval and train layers (default: no co-location analysis)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (default: %(default)s)")
    return p.parse_args(argv)


# ---------------------------------------------------------------- statistics
def load_grid(path, ni, nj, nk):
    """Permeability array K[i, j, k] (shape ni, nj, nk) from an include file; first block if there are three."""
    vals = perm_io.parse_values(path)
    n = ni * nj * nk
    if vals.size == 3 * n:
        vals = vals[:n]
    if vals.size != n:
        raise ValueError(f"{path}: found {vals.size} values, expected {n} (or {3 * n}) for {ni}x{nj}x{nk}")
    if not (np.isfinite(vals).all() and (vals > 0).all()):
        raise ValueError(f"{path}: non-positive or non-finite permeability found")
    return np.ascontiguousarray(vals.reshape(nk, nj, ni).transpose(2, 1, 0))


def log_matrix(K, layers):
    """log10(Kx) of the given 1-based layers as a (n_layers, n_cells) array."""
    return np.stack([np.log10(K[:, :, k - 1]).ravel() for k in layers])


def lag_table(C):
    """Mean / median / max correlation of layer pairs at each distance (in layers)."""
    rows = []
    for lag in range(1, len(C)):
        d = np.diag(C, lag)
        rows.append(dict(lag=lag, n_pairs=d.size, mean=d.mean(), median=np.median(d), max=d.max(), min=d.min()))
    return pd.DataFrame(rows)


def search_split(X, C, args):
    """Random search. Returns (candidates DataFrame sorted best first, best (val, test, train, excluded) as 0-based index lists)."""
    rng = np.random.default_rng(args.seed)
    n, n_cells = X.shape
    n_eval = args.n_val + args.n_test
    if n_eval >= n:
        sys.exit(f"{n_eval} eval layers do not fit in {n} layers")
    m1, m2 = X.mean(1), (X ** 2).mean(1)            # per-layer first and second moments (equal cell counts)
    rows, best_key, chunk = [], None, 20_000
    for _ in range(0, args.n_samples, chunk):
        perm = np.argsort(rng.random((chunk, n)), axis=1)[:, :n_eval]     # distinct layers per sample
        val, test = perm[:, :args.n_val], perm[:, args.n_val:]
        ev = perm
        is_eval = np.zeros((chunk, n), bool)
        np.put_along_axis(is_eval, ev, True, axis=1)
        # eval layers must differ from each other (val vs test); val-val and test-test pairs are free
        cvt = C[val[:, :, None], test[:, None, :]].reshape(chunk, -1).max(1)
        # a non-eval layer is excluded if it correlates too much with any eval layer
        cmax = C[:, ev].transpose(1, 0, 2).max(2)                           # (chunk, n): max corr of layer l with the eval set
        excluded = ~is_eval & (cmax > args.max_corr)
        train_mask = ~is_eval & ~excluded
        n_train = train_mask.sum(1)
        ok = (cvt <= args.max_corr) & (n_train >= args.min_train)
        if not ok.any():
            continue
        w = train_mask / np.maximum(n_train, 1)[:, None]
        mt = (w * m1).sum(1)
        st = np.sqrt(np.maximum((w * m2).sum(1) - mt ** 2, 1e-12))
        for name, sel in (("val", val), ("test", test)):
            mv = m1[sel].mean(1)
            sv = np.sqrt(np.maximum(m2[sel].mean(1) - mv ** 2, 1e-12))
            dm, ds = (mv - mt) / st, (sv - st) / st
            ok &= (np.abs(dm) < args.mean_tol) & (np.abs(ds) < args.std_tol)
            if name == "val":
                dm_v, ds_v = dm, ds
            else:
                dm_t, ds_t = dm, ds
        for i in np.flatnonzero(ok):
            rows.append(dict(val=" ".join(str(int(x)) for x in np.sort(val[i])),
                             test=" ".join(str(int(x)) for x in np.sort(test[i])),
                             n_train=int(n_train[i]), n_excluded=int(excluded[i].sum()),
                             mean_diff_val=dm_v[i], mean_diff_test=dm_t[i],
                             std_rel_val=ds_v[i], std_rel_test=ds_t[i],
                             max_corr_val_test=cvt[i],
                             max_corr_train_eval=float(np.where(train_mask[i][:, None], C[:, ev[i]], -1).max())))
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates(["val", "test"])
    df["abs_mean_diff"] = df.mean_diff_val.abs() + df.mean_diff_test.abs()
    df = df.sort_values(["n_train", "abs_mean_diff"], ascending=[False, True], kind="stable").reset_index(drop=True)
    return df


def colocation(K, layers_eval, layers_train, set_of, window, seed):
    """Max / mean correlation of the same window in an eval layer and in each train layer.

    Windows: x offsets {0, ni - window} times y offsets spread over the layer (non-overlapping in j).
    """
    nx, ny = K.shape[:2]
    ys = np.round(np.linspace(0, ny - window, ny // window)).astype(int)
    xs = [0, nx - window]
    rows = []
    for le in layers_eval:
        for x in xs:
            for y in ys:
                a = np.log10(K[x:x + window, y:y + window, le - 1]).ravel()
                cs = np.array([np.corrcoef(a, np.log10(K[x:x + window, y:y + window, lt - 1]).ravel())[0, 1]
                               for lt in layers_train])
                rows.append(dict(layer=le, set=set_of[le], x_start=x, y_start=y, max_corr=cs.max(),
                                 max_corr_train_layer=layers_train[int(cs.argmax())], mean_corr=cs.mean()))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plots
def plot_matrix(path, C, layers, assign):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower",
                   extent=[layers[0] - .5, layers[-1] + .5, layers[0] - .5, layers[-1] + .5])
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title("correlation of log10(Kx) between layers")
    if assign is not None:
        for t in (ax.get_xticklabels() + ax.get_yticklabels()):
            t.set_color("k")
        for k in layers:
            c = SET_COLORS[assign[k]]
            ax.plot(k, layers[0] - 1.8, "s", color=c, ms=5, clip_on=False)
            ax.plot(layers[0] - 1.8, k, "s", color=c, ms=5, clip_on=False)
        for s, c in SET_COLORS.items():
            ax.plot([], [], "s", color=c, label=s)
        ax.legend(loc="upper left", bbox_to_anchor=(1.18, 1))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_lag(path, lags, max_corr):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags["lag"], lags["mean"], "o-", label="mean")
    ax.fill_between(lags["lag"], lags["min"], lags["max"], alpha=0.2, label="min-max")
    ax.axhline(max_corr, color="k", ls="--", lw=0.8, label=f"--max-corr {max_corr}")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("distance between layers")
    ax.set_ylabel("correlation of log10(Kx)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def md_table(df, floatfmt=".3f"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if out_dir_in_use(out):
        sys.exit(f"{out} is not empty; choose a new or an empty folder")
    if not 1 <= args.k_init < args.k_end <= args.nk:
        sys.exit(f"layers must satisfy 1 <= k-init < k-end <= nk ({args.nk})")
    if args.window is not None and args.window > min(args.ni, args.nj):
        sys.exit(f"--window {args.window} does not fit in the {args.ni}x{args.nj} layer")

    perm_file = Path(args.perm_file)
    K = load_grid(perm_file, args.ni, args.nj, args.nk)
    layers = list(range(args.k_init, args.k_end + 1))
    X = log_matrix(K, layers)
    C = np.corrcoef(X)
    lags = lag_table(C)
    out.mkdir(parents=True, exist_ok=True)

    lab = [f"L{k}" for k in layers]
    pd.DataFrame(C, index=lab, columns=lab).to_csv(out / "layer_corr.csv", float_format="%.4f")
    lags.to_csv(out / "corr_vs_lag.csv", index=False)
    plot_lag(out / "corr_vs_lag.png", lags, args.max_corr)

    cand = search_split(X, C, args)
    if cand is None:
        plot_matrix(out / "layer_corr.png", C, layers, None)
        sys.exit(f"stopped: no split of {len(layers)} layers with {args.n_val} val + {args.n_test} test layers satisfies "
                 f"--max-corr {args.max_corr}, --min-train {args.min_train}, --mean-tol, --std-tol "
                 f"in {args.n_samples} samples. Relax a threshold or use fewer eval layers. "
                 f"Correlations were written to {out}.")
    cand.head(50).to_csv(out / "split_candidates.csv", index=False)
    best = cand.iloc[0]
    val = [layers[i] for i in map(int, best.val.split())]
    test = [layers[i] for i in map(int, best.test.split())]
    ev = set(val) | set(test)
    ei = [layers.index(k) for k in sorted(ev)]
    assign = {}
    for k in layers:
        if k in val:
            assign[k] = "validation"
        elif k in test:
            assign[k] = "test"
        else:
            assign[k] = "excluded" if C[layers.index(k), ei].max() > args.max_corr else "train"
    train = [k for k in layers if assign[k] == "train"]
    excluded = [k for k in layers if assign[k] == "excluded"]
    cmax = [C[i, ei].max() if assign[k] in ("train", "excluded") else np.nan for i, k in enumerate(layers)]
    split = pd.DataFrame(dict(layer=layers, set=[assign[k] for k in layers],
                              mean_log10_kx=X.mean(1), std_log10_kx=X.std(1), max_corr_with_eval=cmax))
    split.to_csv(out / "split.csv", index=False)
    plot_matrix(out / "layer_corr.png", C, layers, assign)

    col = None
    if args.window is not None:
        col = colocation(K, sorted(ev), train, assign, args.window, args.seed)
        col.to_csv(out / "colocation_corr.csv", index=False)

    cfg = dict(vars(args), perm_file=str(perm_file.resolve()), layers=layers)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    pooled = {s: X[[layers.index(k) for k in layers if assign[k] == s]].ravel() for s in ("train", "validation", "test")}
    dist = pd.DataFrame([dict(set=s, n_layers=sum(a == s for a in assign.values()), mean=v.mean(), std=v.std(),
                              diff_vs_train_in_train_std=(v.mean() - pooled["train"].mean()) / pooled["train"].std(),
                              ks_vs_train=0.0 if s == "train" else stats.ks_2samp(v, pooled["train"]).statistic)
                         for s, v in pooled.items()])
    off = C[~np.eye(len(C), dtype=bool)]
    first_below = lags[lags["mean"] < args.max_corr]
    L = ["# Layer correlation and split", "", "Numbers only; no physical interpretation.", "",
         f"Layers {layers[0]}-{layers[-1]} ({len(layers)} layers), correlation of log10(Kx) over all {X.shape[1]} cells.", "",
         f"- off-diagonal correlation: mean {off.mean():.3f}, median {np.median(off):.3f}, 99th percentile "
         f"{np.percentile(off, 99):.3f}, max {off.max():.3f}",
         f"- mean correlation by layer distance: " + ", ".join(f"{r.lag}: {r.mean:.2f}" for r in lags.head(6).itertuples()),
         f"- mean correlation first falls below --max-corr ({args.max_corr}) at distance "
         f"{int(first_below.lag.iloc[0]) if len(first_below) else 'never'}; largest single pair at distance >= 4: "
         f"{max(np.diag(C, l).max() for l in range(4, len(C))):.3f}", "",
         "## Split", "",
         f"Random search, {args.n_samples} samples, seed {args.seed}; {len(cand)} splits met every constraint "
         f"(max corr {args.max_corr}, min train {args.min_train}, |mean diff| < {args.mean_tol} train std, "
         f"|std rel diff| < {args.std_tol}). Chosen: most train layers, then smallest mean difference.", "",
         f"- train ({len(train)}): {train}", f"- validation ({len(val)}): {sorted(val)}",
         f"- test ({len(test)}): {sorted(test)}", f"- excluded ({len(excluded)}): {excluded}",
         f"- max correlation train-eval {best.max_corr_train_eval:.3f}; validation-test {best.max_corr_val_test:.3f}", "",
         md_table(dist), "",
         "## Co-located windows", ""]
    if col is not None:
        L.append(f"Same {args.window}x{args.window} window in an eval layer and in each train layer ({len(col)} windows): "
                 f"max {col.max_corr.max():.3f}, mean of per-window max {col.max_corr.mean():.3f}, "
                 f"mean over all train layers {col.mean_corr.mean():.3f} (`colocation_corr.csv`).")
    else:
        L.append("Not computed (no --window).")
    L += ["", "## Notes", ""]
    L.append("- A random search is not exhaustive; `split_candidates.csv` lists the best 50 splits found.")
    (out / "summary.md").write_text("\n".join(L) + "\n")

    print(f"train {train}\nvalidation {sorted(val)}\ntest {sorted(test)}\nexcluded {excluded}")
    print(f"max corr train-eval {best.max_corr_train_eval:.3f}"
          + (f"; co-located window max {col.max_corr.max():.3f}" if col is not None else ""))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
