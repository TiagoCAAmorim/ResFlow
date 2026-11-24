"""Histogram-based resampling: draws a subset of the data that flattens its distribution.

The dataset builder uses it to choose which cell/step samples are marked for training
(`train_mask`) and with what `weights` (1 / number drawn from the bin).

Copyright (c) Tiago Amorim. MIT License.
"""
import numpy as np
import torch


def estimate_bins(data, max_bins=200, min_bins=2):
    """Bin edges of a histogram of 1D `data` by the Freedman-Diaconis rule.

    bin width = 2 IQR / n^(1/3); the number of bins is limited to [min_bins, max_bins].
    """
    data = np.asarray(data)
    if len(data) < 2:
        return np.array([data.min(), data.max()])

    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    bin_width = 2 * iqr / (len(data) ** (1 / 3))

    if bin_width <= 0:
        num_bins = min_bins
    else:
        num_bins = max(1, int(np.ceil((data.max() - data.min()) / bin_width)))
    num_bins = min(num_bins, max_bins)
    num_bins = max(num_bins, min_bins)
    return np.linspace(data.min(), data.max(), num_bins + 1)


def reduce_product_to_limit(integers, max_product):
    """Reduce the larger integers of a list (one by one, by 1) until their product fits
    in `max_product`. Used to bound the number of cells of an n-dimensional histogram."""
    if not integers:
        return integers

    result = list(integers)

    def _product(values):
        p = 1
        for val in values:
            p *= val
        return p

    current_product = _product(result)
    while current_product > max_product:
        max_idx = result.index(max(result))
        if result[max_idx] <= 1:
            break
        result[max_idx] -= 1
        current_product = _product(result)
    return result


def _compute_draw_counts(hist, num_samples, uniform_ratio):
    """Number of draws from each bin.

    Assumes a minimum number of draws per bin and redistributes the remaining draws
    according to the histogram. Bins with counts below the minimum draw all their elements.
    The minimum is iteratively adjusted to be at least `uniform_ratio` times the average
    number of draws per bin above the minimum.
    """
    pmin = 1
    for _ in range(10):
        n_min = torch.clamp(hist, max=pmin).sum()
        if n_min > num_samples and pmin <= 1:
            # Too many bins for the samples: one draw from each bin with samples.
            p_new = torch.clamp(hist, max=1)
            break
        n_delta = torch.clamp(hist - pmin, min=0).sum()
        a = (num_samples - n_min) / n_delta
        p_new = torch.clamp(hist, max=pmin) + a * torch.clamp(hist - pmin, min=0)
        j = p_new >= pmin
        pmin_new = uniform_ratio * p_new[j].sum() / j.sum()
        if abs(pmin_new - pmin) < 1:
            pmin = pmin_new
            break
        pmin = pmin_new
    return torch.round(p_new).to(torch.int32)


def _compute_bin_indices(data, edges):
    """Map the samples to their histogram bins."""
    for i in range(data.shape[1]):
        edges[i][0] = edges[i][0] - (edges[i][1] - edges[i][0]) / 100.0
    dt = data.T.contiguous()

    bin_indices = []
    for i in range(data.shape[1]):
        indices = torch.searchsorted(edges[i][:-1], dt[i, :], right=False) - 1
        indices = torch.clamp(indices, 0, len(edges[i]) - 2)
        bin_indices.append(indices)

    return torch.stack(bin_indices, dim=1)


def _draw_samples(hist, hist_flat, draw_counts, bin_indices):
    """Boolean mask and weights of the drawn samples (random draw without replacement)."""
    num_samples = bin_indices.shape[0]
    mask = torch.zeros(num_samples, dtype=torch.bool, device=bin_indices.device)
    weight = torch.zeros(num_samples, dtype=torch.float32, device=bin_indices.device)
    hist_shape = hist.shape
    for idx, count in enumerate(draw_counts):
        if count == 0 or hist_flat[idx] == 0:
            continue
        multi_idx = torch.tensor(np.unravel_index(idx, hist_shape), device=bin_indices.device)
        in_bin = (bin_indices == multi_idx).all(dim=1)
        region_indices = torch.nonzero(in_bin, as_tuple=False).squeeze()
        if region_indices.numel() == 0:
            continue
        if region_indices.ndim == 0:
            region_indices = region_indices.unsqueeze(0)
        chosen = region_indices[
            torch.randperm(region_indices.size(0), device=bin_indices.device)[:count]
        ]
        mask[chosen] = True
        weight[chosen] = 1.0 / count

    return mask, weight


def resample(data, num_samples, uniform_ratio=0.5, bins=10):
    """Resample data to about `num_samples` samples with a flatter distribution.

    The data is binned in an n-dimensional histogram; each bin gets a number of draws
    (see _compute_draw_counts) and the drawn samples get weight 1 / draws of their bin.
    Uses the global torch random generator.

    Args:
        data: (n, m) tensor.
        num_samples: number of samples to draw (<= n).
        uniform_ratio: 0 < u < 1; closer to 1 gives a flatter distribution.
        bins: number of bins per dimension (int or list of m ints).
    Returns:
        mask (n,) bool of drawn samples, weights (n,) float32.
    """
    if data.ndim == 1:
        data = data.unsqueeze(1)
    assert data.ndim == 2, "Data must be a 2D tensor."

    bins = 10 if bins is None else bins
    if isinstance(bins, int):
        bins = [bins] * data.shape[1]
    assert all(b > 0 for b in bins), "Number of bins must be positive."
    assert data.shape[1] == len(bins), "Length of bins must match data dimension."

    assert 0 < uniform_ratio < 1, "Uniform ratio must be in (0, 1)."
    assert num_samples >= 1, "num_samples must be positive."
    assert num_samples <= data.shape[0], "num_samples must be <= number of data points."
    assert data.shape[0] > 0, "Data must have at least one point."

    hist, edges = torch.histogramdd(data, bins=bins)
    hist_flat = hist.flatten()

    draw_counts = _compute_draw_counts(hist_flat, num_samples, uniform_ratio)
    bin_indices = _compute_bin_indices(data, edges)
    return _draw_samples(hist, hist_flat, draw_counts, bin_indices)
