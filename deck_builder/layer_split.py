"""Layer-wise train / validation / test split: whole-layer correlations and the validation search.

All statistics use log10(Kx) pooled over cells, population standard deviation (ddof=0).
Layers are 1-based in every input and output; arrays are 0-based internally.
"""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

N_POOL = 30          # layers 1-30 are split between train and validation
TEST_LAYERS = list(range(31, 36))
REF_VAL = list(range(26, 31))  # split A


def layer_log_matrix(K, layers):
    """log10(Kx) of the given 1-based layers as a (n_layers, n_cells) array."""
    return np.stack([np.log10(K[:, :, k - 1]).ravel() for k in layers])


@dataclass
class SearchResult:
    val_layers: list
    train_layers: list
    table: pd.DataFrame      # top-20 combinations plus the split-A reference row
    ref: dict                # metrics of split A
    n_feasible: int
    max_corr_val: float
    max_corr_test: float
    std_tol: float
    notes: list


def _metrics(X, C, combos):
    """Metrics of each validation combination (rows of `combos`, 0-based layer indices < N_POOL).

    X: (35, n_cells) log10 values of layers 1-35; C: its correlation matrix.
    """
    n = X.shape[1]
    N, m = combos.shape
    in_v = np.zeros((N, N_POOL), bool)
    in_v[np.arange(N)[:, None], combos] = True

    max_tv = np.full(N, -np.inf)
    for c in range(m):
        max_tv = np.maximum(max_tv, np.where(in_v, -np.inf, C[:N_POOL, :N_POOL][combos[:, c]]).max(1))
    max_tt = np.full(N, -np.inf)
    for j in range(len(TEST_LAYERS)):
        max_tt = np.maximum(max_tt, np.where(in_v, -np.inf, C[:N_POOL, N_POOL + j][None, :]).max(1))

    X0 = X[:N_POOL] - X[:N_POOL].mean()
    s1, s2 = X0.sum(1), (X0 ** 2).sum(1)
    nv, nt = m * n, (N_POOL - m) * n
    v1, v2 = s1[combos].sum(1), s2[combos].sum(1)
    t1, t2 = s1.sum() - v1, s2.sum() - v2
    mv, mt = v1 / nv, t1 / nt
    sv = np.sqrt(v2 / nv - mv ** 2)
    st = np.sqrt(t2 / nt - mt ** 2)
    return dict(max_corr_tv=max_tv, max_corr_tt=max_tt,
                mean_diff=(mv - mt) / st, std_rel=(sv - st) / st)


def _ks(X, combo):
    v = X[list(combo)].ravel()
    t = X[[i for i in range(N_POOL) if i not in combo]].ravel()
    return stats.ks_2samp(v, t).statistic


def search_validation(X, max_corr_val=0.5, max_corr_test=None, std_tol=0.2, n_val=5,
                      ref_check=True):
    """Exhaustive search for the validation layers (see spec section 4). Deterministic."""
    C = np.corrcoef(X)
    notes = []

    ref_combo = np.array([[k - 1 for k in REF_VAL]])
    r = _metrics(X, C, ref_combo)
    ref = {k: float(v[0]) for k, v in r.items()}
    ref["ks"] = float(_ks(X, tuple(ref_combo[0])))
    if ref_check and (round(ref["max_corr_tv"], 3) != 0.391 or round(ref["max_corr_tt"], 3) != 0.628):
        raise RuntimeError("split A correlations do not match the spec (0.391, 0.628): "
                           f"got {ref['max_corr_tv']:.3f}, {ref['max_corr_tt']:.3f}")
    if max_corr_test is None:
        max_corr_test = ref["max_corr_tt"]

    combos = np.array(list(itertools.combinations(range(N_POOL), n_val)))
    m = _metrics(X, C, combos)
    ok1 = m["max_corr_tv"] < max_corr_val
    ok2 = m["max_corr_tt"] <= max_corr_test + 1e-9
    ok3 = np.abs(m["std_rel"]) < std_tol
    feas = ok1 & ok2 & ok3
    absdiff = np.abs(m["mean_diff"])

    if feas.any():
        pool = np.flatnonzero(feas)
        pool = pool[np.argsort(absdiff[pool], kind="stable")][:50]
        rows = [(round(absdiff[i], 12), _ks(X, tuple(combos[i])), m["max_corr_tv"][i], i) for i in pool]
        rows.sort()
        ranked = [r_[3] for r_ in rows]
        ks_of = {r_[3]: r_[1] for r_ in rows}
        chosen = ranked[0]
    else:
        notes.append("NO combination satisfies all three constraints.")
        for name, arr, ok in (("constraint 1 (max corr train-val)", m["max_corr_tv"], ok1),
                              ("constraint 2 (max corr train-test)", m["max_corr_tt"], ok2),
                              ("constraint 3 (std spread)", np.abs(m["std_rel"]), ok3)):
            i = int(np.argmin(arr))
            notes.append(f"least violating {name}: layers {[int(c) + 1 for c in combos[i]]}, value {arr[i]:.4f}")
        pool = np.flatnonzero(ok1 & ok2)
        if pool.size == 0:
            raise RuntimeError("no combination satisfies constraints 1 and 2; " + "; ".join(notes))
        pool = pool[np.argsort(absdiff[pool], kind="stable")]
        ranked = [int(i) for i in pool[:20]]
        ks_of = {i: _ks(X, tuple(combos[i])) for i in ranked}
        chosen = ranked[0]
        notes.append(f"chosen: lowest mean difference satisfying constraints 1 and 2: "
                     f"{[int(c) + 1 for c in combos[chosen]]}")

    def row(label, layers, mt):
        return dict(label=label, validation_layers=" ".join(map(str, layers)),
                    mean_diff_in_train_std=mt["mean_diff"], std_rel_diff=mt["std_rel"], ks=mt["ks"],
                    max_corr_train_val=mt["max_corr_tv"], max_corr_train_test=mt["max_corr_tt"])

    table = [row(f"rank_{n + 1}", [int(c) + 1 for c in combos[i]],
                 dict(mean_diff=m["mean_diff"][i], std_rel=m["std_rel"][i], ks=ks_of[i],
                      max_corr_tv=m["max_corr_tv"][i], max_corr_tt=m["max_corr_tt"][i]))
             for n, i in enumerate(ranked[:20])]
    table.append(row("split_A_reference", REF_VAL,
                     dict(mean_diff=ref["mean_diff"], std_rel=ref["std_rel"], ks=ref["ks"],
                          max_corr_tv=ref["max_corr_tv"], max_corr_tt=ref["max_corr_tt"])))

    val = sorted(int(c) + 1 for c in combos[chosen])
    train = [k for k in range(1, N_POOL + 1) if k not in val]
    return SearchResult(val, train, pd.DataFrame(table), ref, int(feas.sum()),
                        max_corr_val, max_corr_test, std_tol, notes)


def set_metrics(X, val_layers):
    """Metrics for an arbitrary validation set (used with --val-layers)."""
    C = np.corrcoef(X)
    idx = np.array([[k - 1 for k in val_layers]])
    r = _metrics(X, C, idx)
    out = {k: float(v[0]) for k, v in r.items()}
    out["ks"] = float(_ks(X, tuple(idx[0])))
    return out
