"""Cell geometry in metres -- the single source of coordinates for the package.

Every downstream module (transmissibility, distances, optimisation) reads coordinates
from :class:`CellGeometry` and never from ``(i, j, k)`` indices. That is what lets the
same code serve a regular cartesian grid and an irregular, rotated ``CORNERS`` grid
with only the builder changing.

Cell ordering follows the CMG ``.inc`` convention: ``n = (k*nj + j)*ni + i`` with ``i``
varying fastest.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import get_logger


# MARK: Index helpers
def ijk_to_index(shape: tuple[int, int, int], i: int, j: int, k: int) -> int:
    """Convert 1-based ``(I, J, K)`` to a 0-based linear cell index.

    Args:
        shape: ``(ni, nj, nk)``.
        i: 1-based I index.
        j: 1-based J index.
        k: 1-based K index.

    Returns:
        The 0-based linear index.

    Raises:
        ValueError: If any index falls outside the grid.
    """
    ni, nj, nk = shape
    if not (1 <= i <= ni and 1 <= j <= nj and 1 <= k <= nk):
        raise ValueError(f"({i}, {j}, {k}) is outside a {ni}x{nj}x{nk} grid.")
    return ((k - 1) * nj + (j - 1)) * ni + (i - 1)


def index_to_ijk(shape: tuple[int, int, int], index) -> tuple:
    """Convert 0-based linear cell indices to 1-based ``(I, J, K)``.

    Args:
        shape: ``(ni, nj, nk)``.
        index: A single index or an array of indices.

    Returns:
        ``(I, J, K)`` as ints for a scalar input, or as three arrays otherwise.
    """
    ni, nj, _ = shape
    index = np.asarray(index)
    i = index % ni
    j = (index // ni) % nj
    k = index // (ni * nj)
    if index.ndim == 0:
        return int(i) + 1, int(j) + 1, int(k) + 1
    return i + 1, j + 1, k + 1


# MARK: CellGeometry
@dataclass(frozen=True)
class CellGeometry:
    """Per-cell geometry in metres.

    Attributes:
        centre: ``(ncell, 3)`` cell-centre coordinates ``x, y, z`` in metres. ``z`` is
            depth, positive downwards.
        bulk_volume: ``(ncell,)`` bulk (gross) cell volume in m3.
        corners_xy: ``(ncell, 4, 2)`` map-view corner coordinates in metres, in ring
            order. This is the only geometric input used by the plots, so a rotated or
            irregular grid draws correctly with no special case.
        active: ``(ncell,)`` boolean mask of active cells.
        shape: ``(ni, nj, nk)``, kept only for index <-> IJK conversion and for the
            regular-neighbour connection builder.
        cell_size: ``(dI, dJ, dK)`` in metres when the grid is regular, otherwise
            ``None``. Used as a convenience by the regular connection builder.
    """

    centre: np.ndarray
    bulk_volume: np.ndarray
    corners_xy: np.ndarray
    active: np.ndarray
    shape: tuple[int, int, int]
    cell_size: tuple[float, float, float] | None = None

    @property
    def n_cells(self) -> int:
        """Total number of cells, active or not."""
        return int(self.centre.shape[0])

    @property
    def is_regular(self) -> bool:
        """Whether the grid is a regular cartesian box aligned with the axes."""
        return self.cell_size is not None

    def extent(self) -> tuple[float, float, float, float]:
        """Return ``(x_min, x_max, y_min, y_max)`` in metres over all cells."""
        xy = self.corners_xy.reshape(-1, 2)
        return (
            float(xy[:, 0].min()),
            float(xy[:, 0].max()),
            float(xy[:, 1].min()),
            float(xy[:, 1].max()),
        )

    def layer_slice(self, k_layer: int) -> np.ndarray:
        """Return the linear indices of every cell in a 1-based layer.

        Args:
            k_layer: 1-based layer number.

        Returns:
            ``(ni*nj,)`` array of linear indices.

        Raises:
            ValueError: If the layer is outside the grid.
        """
        ni, nj, nk = self.shape
        if not 1 <= k_layer <= nk:
            raise ValueError(f"k_layer must be in 1..{nk}, got {k_layer}.")
        start = (k_layer - 1) * ni * nj
        return np.arange(start, start + ni * nj)

    def typical_cell_width(self, cells=None) -> np.ndarray:
        """Return a representative horizontal cell width in metres.

        Used to size the well markers on the maps so that they follow the local cell
        size instead of a fixed number of points.

        Args:
            cells: Optional linear indices. Defaults to every cell.

        Returns:
            Array of widths, one per requested cell.
        """
        idx = np.arange(self.n_cells) if cells is None else np.asarray(cells)
        corners = self.corners_xy[idx]
        spans = corners.max(axis=1) - corners.min(axis=1)
        return spans.mean(axis=1)


# MARK: Builders
def build_regular_geometry(
    shape: tuple[int, int, int],
    cell_size: tuple[float, float, float],
    active: np.ndarray | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CellGeometry:
    """Build the geometry of a regular cartesian grid.

    Args:
        shape: ``(ni, nj, nk)``.
        cell_size: ``(dI, dJ, dK)`` in metres.
        active: Optional ``(ncell,)`` boolean mask. Defaults to all active.
        origin: ``(x, y, z)`` of the grid corner at ``(I, J, K) = (1, 1, 1)``.

    Returns:
        The corresponding :class:`CellGeometry`.
    """
    ni, nj, nk = (int(v) for v in shape)
    d_i, d_j, d_k = (float(v) for v in cell_size)
    n_cells = ni * nj * nk

    # Broadcast in (k, j, i) order so that ravel() matches the .inc cell ordering.
    x_c = origin[0] + (np.arange(ni) + 0.5) * d_i
    y_c = origin[1] + (np.arange(nj) + 0.5) * d_j
    z_c = origin[2] + (np.arange(nk) + 0.5) * d_k
    grid = (nk, nj, ni)
    centre = np.stack(
        [
            np.broadcast_to(x_c[None, None, :], grid).ravel(),
            np.broadcast_to(y_c[None, :, None], grid).ravel(),
            np.broadcast_to(z_c[:, None, None], grid).ravel(),
        ],
        axis=1,
    )

    half_i, half_j = 0.5 * d_i, 0.5 * d_j
    offsets = np.array(
        [[-half_i, -half_j], [half_i, -half_j], [half_i, half_j], [-half_i, half_j]]
    )
    corners_xy = centre[:, None, :2] + offsets[None, :, :]

    if active is None:
        active = np.ones(n_cells, dtype=bool)
    else:
        active = np.asarray(active, dtype=bool).ravel()
        if active.size != n_cells:
            raise ValueError(
                f"active has {active.size} entries but the grid has {n_cells} cells."
            )

    return CellGeometry(
        centre=centre,
        bulk_volume=np.full(n_cells, d_i * d_j * d_k, dtype=float),
        corners_xy=corners_xy,
        active=active,
        shape=(ni, nj, nk),
        cell_size=(d_i, d_j, d_k),
    )


def read_corners_geometry(
    inc_path: str | Path,
    shape: tuple[int, int, int],
    active: np.ndarray | None = None,
    *,
    tolerance: float = 1e-6,
    verbose: bool = True,
) -> CellGeometry:
    """Build the geometry from a CMG ``CORNERS`` include file.

    The file lists, for every coordinate in turn (x, then y, then z), the eight corners
    of each cell as top-face then bottom-face rows -- the layout produced by
    :func:`grid_geometry.write_corners_file`. Corner order within a cell is
    ``NW-T, NE-T, SW-T, SE-T, NW-B, NE-B, SW-B, SE-B``.

    Args:
        inc_path: Path to the ``.inc`` file holding the ``CORNERS`` data.
        shape: ``(ni, nj, nk)``.
        active: Optional ``(ncell,)`` boolean mask. Defaults to all active.
        tolerance: Relative tolerance used to decide whether the grid is regular.
        verbose: Print progress messages.

    Returns:
        The corresponding :class:`CellGeometry`. ``cell_size`` is filled in when the grid
        turns out to be regular and axis-aligned, and is ``None`` otherwise.

    Raises:
        ValueError: If the file does not hold ``24 * ni * nj * nk`` values.
    """
    from .model_io import read_inc_values  # local import: avoids a circular import

    log = get_logger(verbose)
    ni, nj, nk = (int(v) for v in shape)
    n_cells = ni * nj * nk
    values = read_inc_values(inc_path)
    if values.size != 24 * n_cells:
        raise ValueError(
            f"CORNERS file {inc_path} holds {values.size} values; a {ni}x{nj}x{nk} grid "
            f"needs {24 * n_cells} (8 corners x 3 coordinates)."
        )

    # Split into the three coordinate blocks, then into (nk, 2, nj, 2, ni, 2):
    # layer, top/bottom, J index, N/S row, I index, W/E column.
    blocks = values.reshape(3, n_cells * 8)
    coords = np.empty((n_cells, 8, 3), dtype=float)
    for c in range(3):
        block = blocks[c].reshape(nk, 2, nj, 2, ni, 2)
        # (top/bottom, N/S, W/E) -> the 8-corner order used by CellGeometry consumers.
        for corner, (tb, ns, we) in enumerate(
            [
                (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
                (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
            ]
        ):
            coords[:, corner, c] = block[:, tb, :, ns, :, we].ravel()

    centre = coords.mean(axis=1)
    # Map-view ring order from the NW/NE/SW/SE top corners.
    corners_xy = coords[:, [0, 1, 3, 2], :2]

    # Bulk volume: map-view shoelace area times the mean corner thickness.
    x = corners_xy[:, :, 0]
    y = corners_xy[:, :, 1]
    area = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    )
    thickness = (coords[:, 4:, 2] - coords[:, :4, 2]).mean(axis=1)
    bulk_volume = area * np.abs(thickness)

    if active is None:
        active = np.ones(n_cells, dtype=bool)
    else:
        active = np.asarray(active, dtype=bool).ravel()

    cell_size = _infer_regular_cell_size(coords, tolerance)
    if cell_size is None:
        log.info(
            "CORNERS grid is not a regular axis-aligned box; using per-cell geometry. "
            "Connection building for irregular grids is not implemented yet "
            "(see ARCHITECTURE.md)."
        )
    else:
        log.info(
            "CORNERS grid is regular: dI=%.3f m, dJ=%.3f m, dK=%.3f m.", *cell_size
        )

    return CellGeometry(
        centre=centre,
        bulk_volume=bulk_volume,
        corners_xy=corners_xy,
        active=active,
        shape=(ni, nj, nk),
        cell_size=cell_size,
    )


def _infer_regular_cell_size(
    coords: np.ndarray, tolerance: float
) -> tuple[float, float, float] | None:
    """Return ``(dI, dJ, dK)`` if every cell is the same axis-aligned box, else ``None``.

    Args:
        coords: ``(ncell, 8, 3)`` corner coordinates.
        tolerance: Relative tolerance on the spread of each edge length.

    Returns:
        The common cell size, or ``None`` when the grid is irregular or rotated.
    """
    spans = coords.max(axis=1) - coords.min(axis=1)  # (ncell, 3)
    reference = spans[0]
    if np.any(reference <= 0):
        return None
    if not np.allclose(spans, reference, rtol=tolerance, atol=tolerance):
        return None

    # Axis alignment: every corner must sit exactly on a face of the centred box, i.e.
    # each coordinate is centre +/- span/2. A rotated cell fails this even though its
    # bounding box is the right size.
    centre = coords.mean(axis=1)[:, None, :]
    offset = np.abs(coords - centre)
    atol = tolerance * max(float(reference.max()), 1.0)
    if not np.allclose(offset, reference[None, None, :] / 2.0, rtol=tolerance, atol=atol):
        return None
    return tuple(float(v) for v in reference)  # type: ignore[return-value]
