"""Linear (standard) normalisation of the model inputs and outputs.

`LinearScaler` holds one (shift, scale) pair per named feature: ``norm(x) = (x - shift) /
scale``. Its tensors are stored in the model weights (``scales``, ``shifts``, ``coords``),
so anything that loads saved weights restores them with `restore_from_state` and never
needs dataset statistics. The statistics are fitted from a dataset only when a new
training starts from scratch (`LinearScaler.fit`, then `save_stats` next to the dataset).

The names of the features and their grouping (several features sharing the statistics of
a representative one) belong to the model code: it passes the feature names and a
``group_fn`` that maps names to their representatives.

Copyright (c) Tiago Amorim. MIT License.
"""
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import torch
from torch import nn
from tqdm import tqdm

from .scalers import MIN_SCALE, fit_linear
from . import scalers as _scalers

MAX_SAMPLES = 1000  # samples (processed files) read to fit
MAX_DATA_POINTS = 1_000_000  # rows kept per tensor to fit
# Scale threshold of drop_degenerate_scale: well above the zero-variance floor of the fit
# (~1e-8), well below the smallest real scale (~0.02).
MIN_INFORMATIVE_SCALE = 1e-6
# Features whose names start with these letters get a zero shift (they are flows, deltas,
# potentials, mobilities, saturations: zero is a meaningful value).
ZERO_SHIFT_PREFIXES = ("d", "f", "G", "B", "b", "S")

STATS_FILE = "lin_scaler.pt"


def stats_path(dataset_root: Union[str, Path]) -> Path:
    """Where the fitted statistics of a dataset root are stored."""
    return Path(dataset_root) / "analysis" / "stats" / STATS_FILE


class LinearScaler(nn.Module):
    """Per-feature linear normalisation (x - shift) / scale.

    Args:
        trainable: if True, scales and shifts are trainable parameters.
        device: device of the tensors.
        group_fn: callable mapping a list of feature names to the names whose statistics
            are used (e.g. several well-rate features sharing the statistics of one). Used by
            `norm`, `denorm` and `drop_degenerate_scale` when called with ``grouped=True``.
    """

    def __init__(self, trainable: bool = False, device: Union[str, torch.device] = "cpu",
                 group_fn: Optional[Callable[[List[str]], List[str]]] = None):
        super().__init__()
        self.device = torch.device(device)
        self.trainable = trainable
        self.group_fn = group_fn
        self.fitted = False
        self.feature_names: List[str] = []
        self._idx_tensor_cache: dict = {}

        self.scales = nn.Parameter(torch.tensor([]), requires_grad=trainable)
        self.shifts = nn.Parameter(torch.tensor([]), requires_grad=trainable)
        self.coords = nn.Parameter(torch.tensor([]), requires_grad=False)

    # MARK: Restore / save
    def restore_from_state(self, state: dict, prefix: str, feature_names: Sequence[str]) -> None:
        """Make this scaler fitted from the tensors of a weights file (no dataset needed).

        Args:
            state: state dict holding ``<prefix>scales``, ``<prefix>shifts``, ``<prefix>coords``.
            prefix: key prefix, e.g. ``"data_norm_in."``.
            feature_names: the feature names, in the order of the saved scales.
        Raises:
            ValueError if the number of names differs from the number of saved scales.
        """
        scales = state[f'{prefix}scales']
        if len(feature_names) != len(scales):
            raise ValueError(
                f"'{prefix}scales' has {len(scales)} entries but {len(feature_names)} feature "
                "names were given. Refusing to guess the feature order.")

        def _param(key, trainable):
            return nn.Parameter(
                state[f'{prefix}{key}'].detach().clone().to(self.device), requires_grad=trainable)

        self.scales = _param('scales', self.trainable)
        self.shifts = _param('shifts', self.trainable)
        self.coords = _param('coords', False)
        self.feature_names = list(feature_names)
        self._idx_tensor_cache = {}
        self.fitted = True

    def save_stats(self, path: Union[str, Path]) -> None:
        """Save the fitted statistics to a file (see `stats_path` for the dataset location)."""
        if not self.fitted:
            raise RuntimeError("Cannot save statistics: scaler has not been fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"scales": self.scales, "shifts": self.shifts, "coords": self.coords,
                    "feature_names": self.feature_names}, path)
        print(f"Statistics saved to {path}")

    def load_stats(self, path: Union[str, Path]) -> bool:
        """Load statistics saved by `save_stats`; returns False if the file is missing or
        invalid. Only for a new training from scratch."""
        path = Path(path)
        if not path.is_file():
            return False
        stats = torch.load(path, weights_only=True, map_location=self.device)
        for key in ("scales", "shifts", "feature_names", "coords"):
            if key not in stats:
                print(f"Error loading stats: missing key '{key}'")
                return False
        n = len(stats["feature_names"])
        if len(stats["scales"]) != n or len(stats["shifts"]) != n:
            print("Error loading stats: dimension mismatch.")
            return False
        self.scales = nn.Parameter(stats["scales"], requires_grad=self.trainable)
        self.shifts = nn.Parameter(stats["shifts"], requires_grad=self.trainable)
        self.coords = nn.Parameter(stats["coords"], requires_grad=False)
        self.feature_names = list(stats["feature_names"])
        self._idx_tensor_cache = {}
        self.fitted = True
        return True

    # MARK: Fit
    def fit(self, dataset, feature_names: Sequence[str], get_values_fn: Callable,
            max_samples: Optional[int] = MAX_SAMPLES,
            max_stats_samples: Optional[int] = MAX_DATA_POINTS, **kwargs) -> None:
        """Fit the statistics from a dataset (new training from scratch only).

        For each sample of the dataset, ``get_values_fn(sample, **kwargs)`` returns a
        tensor (or tuple of tensors) [..., n_features]; the columns of all tensors,
        concatenated in order, are named by `feature_names`. Each feature gets its mean as
        shift (zero for the names starting with one of ZERO_SHIFT_PREFIXES) and its
        standard deviation as scale. At most `max_samples` samples are read, and at most
        `max_stats_samples` rows per tensor are kept (random subset).

        `coords` (min and max of node coordinates) is a placeholder [0, 0]: the datasets
        carry no coordinates.
        """
        print("Fitting LinearScaler to dataset...")
        feature_names = list(feature_names)
        n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

        tensor_data: List[List[torch.Tensor]] = []
        row_counts: List[int] = []
        for i, data in enumerate(tqdm(dataset, total=n_samples, desc="Loading data")):
            if i >= n_samples:
                break
            data = data.to(device=self.device)
            result = get_values_fn(data, **kwargs)
            if not isinstance(result, tuple):
                result = (result,)
            if not tensor_data:
                tensor_data = [[] for _ in result]
                row_counts = [0 for _ in result]
            if len(result) != len(tensor_data):
                raise ValueError(
                    f"Inconsistent number of tensors: expected {len(tensor_data)}, got {len(result)}")

            for k, tensor in enumerate(result):
                if tensor is None:
                    continue
                if tensor.ndim == 3:
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                elif tensor.ndim == 1:
                    tensor = tensor.unsqueeze(-1)
                elif tensor.ndim != 2:
                    raise ValueError(f"Tensor {k} has {tensor.ndim} dims, expected 1/2/3")
                tensor_data[k].append(tensor.cpu())
                row_counts[k] += tensor.shape[0]
                if max_stats_samples is not None and row_counts[k] > 2 * max_stats_samples:
                    compacted = _scalers.subset(
                        torch.cat(tensor_data[k], dim=0), max_n=max_stats_samples, random=True)
                    tensor_data[k] = [compacted]
                    row_counts[k] = compacted.shape[0]

        if not tensor_data or all(len(lst) == 0 for lst in tensor_data):
            raise ValueError("No data was collected from the dataset")

        all_scales, all_shifts = [], []
        i_names = 0
        for k, lists in enumerate(tensor_data):
            if len(lists) == 0:
                continue
            data = torch.cat(lists, dim=0)
            if max_stats_samples is not None and data.shape[0] > max_stats_samples:
                data = data[torch.randperm(data.shape[0])[:max_stats_samples]]
            n_feats = data.shape[1]
            names = feature_names[i_names:i_names + n_feats]
            i_names += n_feats

            scales = torch.zeros(n_feats)
            shifts = torch.zeros(n_feats)
            for i in tqdm(range(n_feats), desc=f"Fitting tensor {k} features"):
                column = data[:, i]
                column = column[torch.isfinite(column)]
                if column.numel() == 0:
                    print(f"Warning: tensor {k}, feature {names[i]} has no valid values")
                    scales[i], shifts[i] = 1.0, 0.0
                    continue
                zero_shift = 0.0 if names[i][0] in ZERO_SHIFT_PREFIXES else None
                scales[i], shifts[i] = fit_linear(column, shift=zero_shift)
            all_scales.append(scales)
            all_shifts.append(shifts)

        if i_names != len(feature_names):
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) does not match the number of "
                f"features collected ({i_names})")
        scales = torch.cat(all_scales)
        if (scales <= 0).any():
            raise ValueError("Some scales are non-positive.")

        self.scales = nn.Parameter(scales, requires_grad=self.trainable)
        self.shifts = nn.Parameter(torch.cat(all_shifts), requires_grad=self.trainable)
        self.coords = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.feature_names = feature_names
        self._idx_tensor_cache = {}
        self.fitted = True

    # MARK: Use
    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "Scaler has not been fitted. Restore it from a weights file, load the "
                "statistics of a dataset, or call fit().")

    def _names(self, feature_names, grouped):
        if feature_names is None:
            return None
        if grouped:
            if self.group_fn is None:
                raise ValueError("grouped=True needs a group_fn (feature name -> representative).")
            return self.group_fn(list(feature_names))
        return list(feature_names)

    def _indices(self, feature_names: Optional[List[str]]) -> List[int]:
        if feature_names is None:
            return list(range(len(self.feature_names)))
        out = []
        for name in feature_names:
            try:
                out.append(self.feature_names.index(name))
            except ValueError as exc:
                raise ValueError(
                    f"Feature '{name}' not found in the scaler. Available: {self.feature_names}"
                ) from exc
        return out

    def _indices_tensor(self, feature_names, grouped: bool, device) -> torch.Tensor:
        """Cached device tensor of the indices (avoids a host-to-device copy per call)."""
        key = (tuple(feature_names) if feature_names is not None else None, grouped, device)
        idx = self._idx_tensor_cache.get(key)
        if idx is None:
            idx = torch.tensor(self._indices(self._names(feature_names, grouped)),
                               dtype=torch.long, device=device)
            self._idx_tensor_cache[key] = idx
        return idx

    def get_scale(self, feature_names: Optional[List[str]] = None) -> torch.Tensor:
        """Scales of the given features (all if None)."""
        self._check_fitted()
        if feature_names is None:
            return self.scales.clone()
        return self.scales[self._indices(feature_names)].clone()

    def get_shift(self, feature_names: Optional[List[str]] = None) -> torch.Tensor:
        """Shifts of the given features (all if None)."""
        self._check_fitted()
        if feature_names is None:
            return self.shifts.clone()
        return self.shifts[self._indices(feature_names)].clone()

    def get_coords(self) -> torch.Tensor:
        """Coordinate statistics (min and max)."""
        self._check_fitted()
        return self.coords.clone()

    def drop_degenerate_scale(self, feature_names: List[str], min_scale: float = MIN_INFORMATIVE_SCALE,
                              grouped: bool = False) -> List[str]:
        """`feature_names` without the features whose scale is at the zero-variance floor.

        A column that is constant in the fitted dataset (e.g. depth on a single-layer 2D
        model) gets a tiny scale; normalising with it would blow up noise. With
        `grouped`, the scale of the representative feature is the one checked.
        """
        scales = self.get_scale(self._names(feature_names, grouped))
        return [n for n, s in zip(feature_names, scales.tolist()) if s > min_scale]

    def norm(self, x: torch.Tensor, feature_names: Optional[List[str]] = None,
             grouped: bool = True) -> torch.Tensor:
        """(x - shift) / scale for the columns named `feature_names` (all, in order, if None)."""
        self._check_fitted()
        indices = self._indices_tensor(feature_names, grouped, self.scales.device)
        scales = self.scales[indices].to(x.device)
        shifts = self.shifts[indices].to(x.device)
        return (x - shifts) / scales.clamp(min=MIN_SCALE)

    def denorm(self, x: torch.Tensor, feature_names: Optional[List[str]] = None,
               grouped: bool = True) -> torch.Tensor:
        """Inverse of norm: x * scale + shift."""
        self._check_fitted()
        indices = self._indices_tensor(feature_names, grouped, self.scales.device)
        scales = self.scales[indices].to(x.device)
        shifts = self.shifts[indices].to(x.device)
        return x * scales.clamp(min=MIN_SCALE) + shifts
