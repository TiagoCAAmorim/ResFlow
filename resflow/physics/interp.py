"""Piecewise-linear interpolation in torch (1D, 1D with inverse blending, and 2D).

Used by the PVT and relative-permeability tables.
Copyright (c) Tiago Amorim. MIT License.
"""
import torch

EPS = 1e-10  # Small value to avoid division by zero in inverse interpolation


def _take(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """table[idx] for a 1D table and an index tensor of any shape (via index_select)."""
    return table.index_select(0, idx.reshape(-1)).view(idx.shape)


def search_interval(x: torch.Tensor, xp: torch.Tensor) -> torch.Tensor:
    """Index of the interval of x holding each xp (-1 below x[0], len(x)-1 at or above x[-1]).

    Callers that look up several columns on the same axis at the same points can compute
    it once and pass it to the interpolators as `idx` (the results are identical).
    """
    return torch.searchsorted(x, xp.contiguous(), right=True) - 1


def _interp1d_components(x, xp, extrapolate, idx=None):
    """Compute reusable interpolation components for 1D interpolation."""
    if idx is None:
        idx = search_interval(x, xp)

    xp_interp = xp if extrapolate else xp.clamp(x[0], x[-1])

    idx_clamped = idx.clamp(0, x.shape[0] - 2)
    x0 = _take(x, idx_clamped)
    dx = _take(torch.clamp(x[1:] - x[:-1], min=EPS), idx_clamped)
    wx = (xp_interp - x0) / dx

    below = idx < 0
    above = idx >= x.shape[0] - 1
    return idx_clamped, wx, below, above


def _interp1d_apply(x, y, xp, idx_clamped, wx, below, above, extrapolate):
    """Apply interpolation using precomputed components."""
    y0 = _take(y, idx_clamped)
    dy = _take(y[1:] - y[:-1], idx_clamped)
    interp = y0 + wx * dy

    if extrapolate:
        # One host sync for both checks instead of one each.
        any_below, any_above = torch.stack((below.any(), above.any())).tolist()

        if any_below:
            slope_below = (y[1] - y[0]) / torch.clamp(x[1] - x[0], min=EPS)
            interp = torch.where(below, y[0] + slope_below * (xp - x[0]), interp)

        if any_above:
            slope_above = (y[-1] - y[-2]) / torch.clamp(x[-1] - x[-2], min=EPS)
            interp = torch.where(above, y[-1] + slope_above * (xp - x[-1]), interp)

    return interp


def interp1d(x: torch.Tensor, y: torch.Tensor, xp: torch.Tensor, extrapolate: bool = True,
             idx: torch.Tensor = None) -> torch.Tensor:
    """Linear interpolation for 1D data.

    x is assumed to be sorted in ascending order. With `extrapolate=False` the query
    points are clamped to the table; otherwise the end segments are extended linearly.
    `idx` is an optional precomputed `search_interval(x, xp)`.
    """
    idx_clamped, wx, below, above = _interp1d_components(x, xp, extrapolate, idx)
    return _interp1d_apply(x, y, xp, idx_clamped, wx, below, above, extrapolate)


def alt_interp1d(x: torch.Tensor, y: torch.Tensor, xp: torch.Tensor, *,
                 x_inversion: float = -torch.inf, inverse_smaller: bool = True,
                 extrapolate: bool = True, idx: torch.Tensor = None) -> torch.Tensor:
    """1D interpolation, part linear and part linear in the inverse of y.

    Points with xp < x_inversion (or xp >= x_inversion when `inverse_smaller` is False)
    are interpolated linearly in 1/y (the gas expansion factor and viscosities of the
    PVT tables behave this way); the others linearly in y. `x_inversion` must be a
    plain float: with +-inf the plain linear interpolation is used.
    """
    if x_inversion in [-torch.inf, torch.inf]:
        return interp1d(x, y, xp, extrapolate=extrapolate, idx=idx)

    mask_inv = xp < x_inversion if inverse_smaller else xp >= x_inversion

    # Fast paths: all points of the same type (one host sync for both checks).
    n_inv = int(mask_inv.sum())
    if n_inv == 0:
        return interp1d(x, y, xp, extrapolate=extrapolate, idx=idx)
    if n_inv == mask_inv.numel():
        y_inv = 1.0 / (y + EPS)
        return 1.0 / interp1d(x, y_inv, xp, extrapolate=extrapolate, idx=idx)

    # Mixed case: some points need inversion, some do not.
    idx_clamped, wx, below, above = _interp1d_components(x, xp, extrapolate, idx)
    interp_lin = _interp1d_apply(x, y, xp, idx_clamped, wx, below, above, extrapolate)

    y_inv = 1.0 / (y + EPS)
    interp_inv = _interp1d_apply(x, y_inv, xp, idx_clamped, wx, below, above, extrapolate)
    interp_inv = 1.0 / (interp_inv + EPS)

    return torch.where(mask_inv, interp_inv, interp_lin)


def interp2d(x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, xp: torch.Tensor,
             idx1: torch.Tensor = None, idx2: torch.Tensor = None) -> torch.Tensor:
    """Bilinear interpolation for 2D data.

    xp values are clamped to the grid bounds (no extrapolation). x1 and x2 are sorted
    in ascending order.

    Args:
        x1: (n1,) coordinates of the first axis.
        x2: (n2,) coordinates of the second axis.
        y: (n1, n2) values at the grid points.
        xp: (n, 2) query points [[x1', x2'], ...].
        idx1, idx2: optional precomputed search_interval on each axis.
    Returns:
        (n,) interpolated values.
    """
    x1p = xp[:, 0].clamp(x1[0], x1[-1])
    x2p = xp[:, 1].clamp(x2[0], x2[-1])

    if idx1 is None:
        idx1 = search_interval(x1, x1p)
    if idx2 is None:
        idx2 = search_interval(x2, x2p)
    idx1 = idx1.clamp(0, x1.shape[0] - 2)
    idx2 = idx2.clamp(0, x2.shape[0] - 2)

    x1_0 = _take(x1, idx1)
    x2_0 = _take(x2, idx2)
    dx1 = _take(torch.clamp(x1[1:] - x1[:-1], min=EPS), idx1)
    dx2 = _take(torch.clamp(x2[1:] - x2[:-1], min=EPS), idx2)

    wx1 = (x1p - x1_0) / dx1
    wx2 = (x2p - x2_0) / dx2

    n2 = y.shape[1]
    flat = idx1 * n2 + idx2
    y_flat = y.reshape(-1)
    dy_flat = (y[1:] - y[:-1]).reshape(-1)
    y00 = _take(y_flat, flat)
    y01 = _take(y_flat, flat + 1)
    dy0 = _take(dy_flat, flat)  # y10 - y00
    dy1 = _take(dy_flat, flat + 1)  # y11 - y01

    return (y00 + wx1 * dy0) * (1 - wx2) + (y01 + wx1 * dy1) * wx2
