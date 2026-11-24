"""Readers for the rollout files: ``rollout_<case>.sr3.npz`` and ``rollout_cumulative_<case>.sr3.csv``.

npz keys used: ``days pred pred_days true var_names ijk stat well_marker_ij well_marker_type well_marker_name``
(``pred``/``true`` are [days, cells, variables]; ``stat[:, 3]`` is kx in md; ``ijk`` is 1-based). The pressure of
the files is in kgf/cm2; `state_fields` in maps.py converts it to MPa.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KGF_CM2_TO_MPA = 0.0980665


@dataclass
class Rollout:
    """The arrays of one rollout needed for plotting. `true` and `pred` hold only the days in `days`."""
    ijk: np.ndarray                 # [cells, 3], 1-based
    kx: np.ndarray                  # [cells], md
    var_names: list
    days: np.ndarray                # days held in true / pred
    true: np.ndarray | None         # [len(days), cells, variables]
    pred: np.ndarray | None
    marker_ij: np.ndarray           # [wells, 2], 1-based cell coordinates (may be fractional)
    marker_type: np.ndarray         # 'producer' or 'injector'
    marker_name: np.ndarray         # short names, e.g. 'P1', 'I1'
    extra: dict = field(default_factory=dict)

    def var(self, name: str) -> int:
        return list(self.var_names).index(name)


def read_rollout_npz(path, days=None, keys_extra=()) -> Rollout:
    """Read a rollout file. With `days` (iterable of day numbers) only those days of true/pred are kept in
    memory; with ``days=None`` the maps are not read (permeability and wells only)."""
    z = np.load(path, allow_pickle=True)
    ijk = z["ijk"]
    out = dict(ijk=ijk, kx=np.asarray(z["stat"][:, 3], dtype=float), var_names=[str(v) for v in z["var_names"]],
               marker_ij=np.asarray(z["well_marker_ij"], dtype=float), marker_type=np.asarray(z["well_marker_type"]),
               marker_name=np.asarray(z["well_marker_name"]))
    if days is None:
        true = pred = None
        got = np.zeros(0)
    else:
        days = np.asarray(list(days), dtype=float)
        all_true, all_pred = z["days"], z["pred_days"]
        it = np.array([int(np.argmin(np.abs(all_true - d))) for d in days])
        ip = np.array([int(np.argmin(np.abs(all_pred - d))) for d in days])
        if not (np.allclose(all_true[it], days) and np.allclose(all_pred[ip], days)):
            raise ValueError(f"days {days} not all in the rollout (true {all_true[[0, -1]]}, pred {all_pred[[0, -1]]})")
        true = np.asarray(z["true"][it])
        pred = np.asarray(z["pred"][ip])
        got = days
    extra = {k: z[k] for k in keys_extra if k in z.files}
    return Rollout(days=got, true=true, pred=pred, extra=extra, **out)


def read_cumulative_csv(path) -> dict:
    """``rollout_cumulative_<case>.sr3.csv`` -> {(component, well): (days, true, estimate)} (numpy arrays).
    Columns used: component, well, day, True, Est."""
    acc = defaultdict(lambda: ([], [], []))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            a = acc[(r["component"], r["well"])]
            a[0].append(float(r["day"]))
            a[1].append(float(r["True"]))
            a[2].append(float(r["Est"]))
    return {k: tuple(np.asarray(x) for x in v) for k, v in acc.items()}
