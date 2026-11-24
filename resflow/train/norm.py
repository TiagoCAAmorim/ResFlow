"""``fit-norm``: normalisation statistics of a training set, for a training from scratch.

For up to ``max_samples`` stored series of the train split (in a seeded random order),
every registry feature (node, delta and edge features; `resflow.models.features`) is
computed and each gets a linear (shift, scale) pair: the mean and standard deviation, with
a zero shift for flows, deltas, potentials, mobilities and saturations. G is fixed (1e-4)
and the depth reference of the potentials is the mean depth of each graph, as in the
models; neither is fitted.

The statistics are written next to the dataset (``analysis/stats/lin_scaler.pt``) and are
read only when a training starts from scratch: a trained model carries its own copy.

Copyright (c) Tiago Amorim. MIT License.
"""
from pathlib import Path
from typing import Optional

import torch

from ..data.dataset import SimDataset
from ..data.normalization import MAX_DATA_POINTS, MAX_SAMPLES, stats_path
from ..models.features import all_feature_names, fit_values
from ..models.state import make_scaler
from ..physics.transform import G, Transform


def fit_norm(dataset_root, out: Optional[Path] = None, max_samples: int = MAX_SAMPLES,
             max_points: int = MAX_DATA_POINTS, seed: int = 0, device: Optional[str] = None) -> Path:
    """Fit and save the statistics of ``<dataset_root>/train``; returns the file path."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(dataset_root)
    ds = SimDataset(root=root / 'train', min_day=0.0, max_day=99999.0)
    torch.manual_seed(seed)
    ds = ds.shuffle()
    transform = Transform.from_folder(root / 'train' / 'raw', g=G, z_mean=None).to(device)
    scaler = make_scaler(device)
    scaler.fit(ds, all_feature_names(), fit_values, max_samples=max_samples,
               max_stats_samples=max_points, transform=transform)
    out = Path(out) if out is not None else stats_path(root)
    scaler.save_stats(out)
    return out
