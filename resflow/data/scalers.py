"""Scalers used to fit statistics: mode estimation, asinh and standard (linear) scalers.

Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Union

import torch
from torch import nn

X_MIN = 0.2483  # Approximate value where |asinh(x)/x - 1| = 0.01
MIN_SCALE = 1e-12  # Minimum absolute value for scale parameters to avoid numerical issues
MAX_SCALE_CHANGE = 2  # Change in log10 scale to search for the optimal scale parameter


# MARK: Mode
def subset(x: torch.Tensor, max_n: int, random: bool = False) -> torch.Tensor:
    """At most `max_n` rows of x (first dimension), random or uniformly spaced."""
    n = x.shape[0]
    if n > max_n:
        if random:
            indices = torch.randperm(n)[:max_n]
            return x[indices]
        step = int(n // max_n)
        return x[::step]
    return x


def _freedman_diaconis_bins(x: torch.Tensor) -> int:
    """Number of bins by the Freedman-Diaconis rule: (max - min) / (2 IQR n^(-1/3))."""
    x = x.flatten().to(torch.float64)
    n = x.numel()
    if n < 2:
        return 1

    x_sub = subset(x, max_n=1_000_000)
    q1 = x_sub.quantile(0.25)
    q3 = x_sub.quantile(0.75)
    iqr = q3 - q1

    x_min, x_max = x.min(), x.max()
    if x_min == x_max:
        return 1

    if iqr == 0:
        # fallback proportional to sqrt(n) when the IQR is zero
        bins = int(torch.ceil(torch.sqrt(torch.tensor(float(n)))).item())
        return max(bins, 1)

    h = 2.0 * iqr * n ** (-1.0 / 3.0)
    if h == 0:
        return 1

    bins = int(torch.ceil((x_max - x_min) / h).item())
    return max(bins, 1)


def _binned_mode_edges(x: torch.Tensor, bins: Union[int, None] = None):
    """(left_edge, right_edge) of the most populous bin of a histogram of x."""
    x = x.contiguous()
    x_min = x.min()
    x_max = x.max()
    if x_min == x_max:
        return x_min, x_max

    if bins is None:
        bins = _freedman_diaconis_bins(x)
    if bins < 1:
        raise ValueError("bins must be >= 1")

    edges = torch.linspace(x_min, x_max, bins + 1, dtype=x.dtype, device=x.device)
    idx = torch.bucketize(x, edges, right=False) - 1
    idx = idx.clamp(0, bins - 1)

    counts = torch.bincount(idx, minlength=bins)
    mode_bin = torch.argmax(counts)
    return edges[mode_bin], edges[mode_bin + 1]


def binned_mode_refined(x: torch.Tensor, bins: Union[int, None] = None,
                        refine_bins: int = 32) -> torch.Tensor:
    """Mode of all elements of x via a coarse histogram plus a refinement of its top bin.

    The number of coarse bins is estimated with Freedman-Diaconis when `bins` is None.
    Returns a float64 scalar tensor.
    """
    x = x.to(torch.float64).flatten()
    if x.numel() == 0:
        raise ValueError("Empty tensor")

    coarse_left, coarse_right = _binned_mode_edges(x, bins)
    if coarse_left == coarse_right:
        return coarse_left

    mask = (x >= coarse_left) & (x <= coarse_right)
    x_local = x[mask]
    if x_local.numel() == 0:
        return (coarse_left + coarse_right) / 2

    fine_left, fine_right = _binned_mode_edges(x_local, bins=refine_bins)
    return (fine_left + fine_right) / 2


def uniformity_score(tensor) -> float:
    """Score from 0 to 1 (1: perfectly uniform) of a tensor of counts or frequencies."""
    probs = tensor / tensor.sum()
    probs = probs[probs > 0]
    entropy = -(probs * torch.log(probs)).sum()
    max_entropy = torch.log(torch.tensor(len(probs), dtype=probs.dtype, device=probs.device))
    return (entropy / (max_entropy + 1e-10)).item()


# MARK: Asinh
class AsinhScaler(nn.Module):
    """Inverse hyperbolic sine scaler: y = asinh((x - shift) / scale) / y_scale.

    Fitted on samples: `shift` is the (histogram) mode, `scale` maximises the uniformity of
    the histogram of the transformed data (coarse then fine search over powers of 10,
    starting from an almost linear transform), `y_scale` is the standard deviation of the
    transformed data. Used by the dataset builder to flatten the distribution of the volume
    increments before the histogram resampling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.y_scale = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def fit(self, x: torch.Tensor) -> None:
        """Fit shift, scale and y_scale to the samples x."""
        self.shift = nn.Parameter(binned_mode_refined(x.flatten()), requires_grad=False)
        self._fit_scale(x)
        self._fit_y_scale(x)

    def _fit_scale(self, x: torch.Tensor) -> None:
        x = x - self.shift

        if x.std() < MIN_SCALE:
            self._set_scale(1.0)
            return

        scale_min = x.abs().max() / X_MIN

        def evaluate(scale_value) -> float:
            x_trans = torch.asinh(x / scale_value)
            return uniformity_score(torch.histc(x_trans, bins=50))

        us_best = evaluate(scale_min)
        best_scale = scale_min

        init_scale = torch.log10(scale_min).floor().item() + 1
        end_scale = torch.log10(x.abs().max()).floor().item() - MAX_SCALE_CHANGE
        end_scale = min(0.0, end_scale)  # Don't search above scale=1.0
        end_scale = min(end_scale, init_scale - MAX_SCALE_CHANGE)

        def test_scales(scales):
            nonlocal us_best, best_scale
            for i in scales:
                test_scale = torch.tensor([10.0 ** i])
                us = evaluate(test_scale)
                if us > us_best:
                    best_scale = test_scale
                    us_best = us

        test_scales(list(range(int(init_scale), int(end_scale) - 1, -1)))

        log_best = torch.log10(best_scale).item()
        test_scales(torch.linspace(log_best - 1.0, log_best + 1.0, 21))

        log_best = torch.log10(best_scale).item()
        test_scales(torch.linspace(log_best - 0.1, log_best + 0.1, 21))

        self._set_scale(best_scale)

    def _set_scale(self, value) -> None:
        if isinstance(value, float):
            value = torch.tensor([value])
        else:
            value = value.reshape(1)
        self.scale = nn.Parameter(value, requires_grad=False)

    def _fit_y_scale(self, x: torch.Tensor) -> None:
        y = torch.asinh((x.flatten() - self.shift) / self.scale)
        y_std = y.std()
        if y_std < MIN_SCALE:
            y_std = torch.tensor([1.0])
        self.y_scale = nn.Parameter(y_std, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """asinh((x - shift) / scale) / y_scale."""
        return torch.asinh((x - self.shift) / self.scale) / self.y_scale

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit to x and transform it."""
        self.fit(x)
        return self.forward(x)


def asinh_fit_transform(t: torch.Tensor) -> torch.Tensor:
    """Fit an AsinhScaler on the flattened-per-feature data `t` ([..., n_features]) and
    return the transformed data with the same shape."""
    t_ = t.reshape(-1, t.shape[-1])
    transformed = AsinhScaler().fit_transform(t_)
    return transformed.reshape(t.shape)


# MARK: Linear
def fit_linear(samples: torch.Tensor, shift: Union[None, float] = None):
    """Fit a 'standard' linear scaler (x - shift) / scale to 1D samples.

    shift is the mean unless given; scale is the standard deviation (floor 1e-8).
    Returns (scale, shift) as float values.
    """
    shift_t = samples.mean(dim=0) if shift is None else torch.tensor([shift])
    scale_t = (samples - shift_t).std(dim=0).clone().detach().clamp(min=1e-8)
    return scale_t.item(), shift_t.item()
