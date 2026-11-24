"""
Interpolation functions.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


EPS = 1e-10  # Small value to avoid division by zero in inverse interpolation

@staticmethod
def interp_extrap(x, y, new_x, extrap=True):
    """
    Perform linear interpolation and extrapolation.

    Extrapolates using the two lowest or two largest points if new_x is outside x range.

    Will try to invert x and y if x is not strictly increasing. If that fails,
    will sort x and y according to x.

    Parameters
    ----------
    x : array_like
        Known x-coordinates.
    y : array_like
        Known y-coordinates corresponding to x.
    new_x : array_like
        New x-coordinates where interpolation/extrapolation is desired.
    extrap : bool, optional
        If True, extrapolate outside the range of x. Default is True.

    Returns
    -------
    y_interp : ndarray
        Interpolated or extrapolated y-coordinates corresponding to new_x.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if not np.all(np.diff(x) > 0):
        x = x[::-1]
        if not np.all(np.diff(x) > 0):
            sort_indices = np.argsort(x)
            x = x[sort_indices]
            if not np.all(np.diff(x) > 0):
                raise ValueError("Input x must be strictly increasing.")
            y = y[sort_indices]
        else:
            y = y[::-1]

    new_x = np.asarray(new_x)
    y_interp = np.interp(new_x, x, y)
    if not extrap:
        return y_interp

    below = new_x < x[0]
    if np.any(below):
        slope = (y[1] - y[0]) / (x[1] - x[0])
        y_interp[below] = y[0] + slope * (new_x[below] - x[0])

    above = new_x > x[-1]
    if np.any(above):
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        y_interp[above] = y[-1] + slope * (new_x[above] - x[-1])
    return y_interp


@staticmethod
def alt_interp1d(x, y, x_new, x_inversion=-np.inf, inverse_smaller=True, extrap=True):
    """
    1D interpolation part linear, part linear to the inverse.

    Parameters
    ----------
    x : array_like
        The x-coordinates of the data points.
    y : array_like
        The y-coordinates of the data points.
    x_new : array_like
        The x-coordinates where the interpolation is evaluated.
    x_inversion : float, optional
        The x-coordinate where the interpolation switches from linear to inverse.
        Default is -np.inf, meaning no switch.
    inverse_smaller : bool, optional
        If True, the inverse interpolation is applied for x values smaller than x_inversion.
        If False, it is applied for x values larger than x_inversion.
        Default is True.
    extrap : bool, optional
        If True, extrapolation is performed for x values outside the range of x.
        If False, values outside the range will not be extrapolated.
        Default is True.
    Returns
    -------
    y_interp : array_like
        The interpolated y-coordinates corresponding to the input x values.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x_new = np.asarray(x_new)

    if inverse_smaller:
        mask = x_new < x_inversion
    else:
        mask = x_new > x_inversion

    y_interp = np.zeros_like(x_new)
    if mask.any():
        y_interp[mask] = 1/interp_extrap(x, 1/(y+EPS), x_new[mask], extrap=extrap)
    if (~mask).any():
        y_interp[~mask] = interp_extrap(x, y, x_new[~mask], extrap=extrap)

    return y_interp


@staticmethod
def interp2d(x, y, new_x, interpolator=None, extrap=True):
    """
    Control extrapolation behavior for 2D interpolation.

    Parameters
    ----------
    x : list of array_like
        Known x-coordinates for each dimension.
    y : array_like
        Known y-coordinates corresponding to the grid defined by x.
        Ignored if interpolator is provided.
    new_x : array_like
        New x-coordinates where interpolation/extrapolation is desired.
    interpolator : RegularGridInterpolator, optional
        An instance of RegularGridInterpolator to use for interpolation.
        If None, a new interpolator will be created.
    extrap : bool, optional
        If True, extrapolate outside the range of x. An array of booleans
        can also be provided to specify extrapolation for each dimension.
        Default is True.
    Returns
    -------
    y_interp : ndarray
        Interpolated or extrapolated y-coordinates corresponding to new_x.
    """
    x_ = []
    for xi in x:
        x_.append(np.asarray(xi))

    extrap = np.asarray(extrap, dtype=bool)
    if extrap.ndim == 0:
        extrap = np.full(len(x_), extrap, dtype=bool)
    elif extrap.shape[0] == 1:
        extrap = np.full(len(x_), extrap, dtype=bool)

    new_x = np.asarray(new_x).copy()

    if interpolator is None:
        y = np.asarray(y)
        interpolator = RegularGridInterpolator(x_, y, bounds_error=False, fill_value=None)

    for i in range(new_x.shape[1]):
        if not extrap[i]:
            below = new_x[:,i] < x_[i][0]
            if np.any(below):
                new_x[below,i] = x_[i][0]
            above = new_x[:,i] > x_[i][-1]
            if np.any(above):
                new_x[above,i] = x_[i][-1]

    return interpolator(new_x)
