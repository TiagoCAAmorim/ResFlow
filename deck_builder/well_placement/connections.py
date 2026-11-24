"""Flow connections between cells -- the neutral layer the physics is built on.

Nothing downstream of this module indexes neighbours as ``(i+1, j, k)``. Every
calculation consumes a :class:`ConnectionTable`, which is just a list of "cell A talks to
cell B through this face". Regular cartesian neighbours are one way to fill it; an
irregular corner-point grid with faults and pinch-outs is another, and connections
between cells that are not neighbours at all (NNCs) need no special handling anywhere
else in the package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CellGeometry

#: Direction codes. They select which permeability to use and whether NTG applies.
DIR_I, DIR_J, DIR_K, DIR_IJ = 0, 1, 2, 3
DIRECTION_NAMES = {DIR_I: "I", DIR_J: "J", DIR_K: "K", DIR_IJ: "IJ"}

#: Directions in which net-to-gross multiplies the transmissibility (K is excluded).
#: The in-layer diagonal is horizontal flow, so NTG applies to it as well.
NTG_DIRECTIONS = (DIR_I, DIR_J, DIR_IJ)


# MARK: ConnectionTable
@dataclass(frozen=True)
class ConnectionTable:
    """Flow connections between pairs of cells.

    By convention ``cell_a < cell_b``; the graph built from the table is undirected.
    Connections touching an inactive cell may stay in the table -- they are filtered when
    the graph is built.

    Attributes:
        cell_a: ``(nconn,)`` linear index of the first cell.
        cell_b: ``(nconn,)`` linear index of the second cell.
        area: ``(nconn,)`` flow-face area in m2.
        dist_a: ``(nconn,)`` distance from the centre of A to the face, in metres.
        dist_b: ``(nconn,)`` distance from the centre of B to the face, in metres.
        direction: ``(nconn,)`` direction code, ``0=I``, ``1=J``, ``2=K``.
    """

    cell_a: np.ndarray
    cell_b: np.ndarray
    area: np.ndarray
    dist_a: np.ndarray
    dist_b: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        sizes = {
            name: np.asarray(getattr(self, name)).size
            for name in ("cell_a", "cell_b", "area", "dist_a", "dist_b", "direction")
        }
        if len(set(sizes.values())) > 1:
            raise ValueError(f"ConnectionTable fields have different lengths: {sizes}.")

    @property
    def n_connections(self) -> int:
        """Number of connections in the table."""
        return int(self.cell_a.size)

    @classmethod
    def concat(cls, *tables: "ConnectionTable") -> "ConnectionTable":
        """Concatenate connection tables.

        This is how non-neighbour connections are added to a set of structured ones:
        build them separately and concatenate.

        Args:
            *tables: Tables to join.

        Returns:
            A single table holding every connection.
        """
        tables = tuple(t for t in tables if t.n_connections)
        if not tables:
            empty_i = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=float)
            return cls(empty_i, empty_i.copy(), empty_f, empty_f.copy(),
                       empty_f.copy(), np.empty(0, dtype=np.int8))
        return cls(
            cell_a=np.concatenate([t.cell_a for t in tables]),
            cell_b=np.concatenate([t.cell_b for t in tables]),
            area=np.concatenate([t.area for t in tables]),
            dist_a=np.concatenate([t.dist_a for t in tables]),
            dist_b=np.concatenate([t.dist_b for t in tables]),
            direction=np.concatenate([t.direction for t in tables]),
        )

    def counts_by_direction(self) -> dict[str, int]:
        """Return the number of connections per direction, for logging and QC."""
        return {
            name: int((self.direction == code).sum())
            for code, name in DIRECTION_NAMES.items()
        }


# MARK: Builders
def build_regular_connections(geometry: CellGeometry) -> ConnectionTable:
    """Build the I, J and K neighbour connections of a regular cartesian grid.

    Face areas and centre-to-face distances follow the specification: for the I
    direction ``area = dJ*dK`` and ``dist = dI/2``, and likewise for J and K.

    Args:
        geometry: A regular :class:`CellGeometry` (``cell_size`` must be set).

    Returns:
        The connection table, with ``cell_a < cell_b`` on every row.

    Raises:
        ValueError: If the geometry is not regular.
    """
    if geometry.cell_size is None:
        raise ValueError(
            "build_regular_connections needs a regular grid. For an irregular "
            "CORNERS grid use build_corner_connections (see ARCHITECTURE.md)."
        )
    ni, nj, nk = geometry.shape
    d_i, d_j, d_k = geometry.cell_size
    index = np.arange(ni * nj * nk).reshape(nk, nj, ni)

    layout = [
        (DIR_I, np.s_[:, :, :-1], np.s_[:, :, 1:], d_j * d_k, 0.5 * d_i),
        (DIR_J, np.s_[:, :-1, :], np.s_[:, 1:, :], d_i * d_k, 0.5 * d_j),
        (DIR_K, np.s_[:-1, :, :], np.s_[1:, :, :], d_i * d_j, 0.5 * d_k),
    ]
    parts = []
    for direction, slice_a, slice_b, area, dist in layout:
        cell_a = index[slice_a].ravel()
        cell_b = index[slice_b].ravel()
        if cell_a.size == 0:
            continue
        n = cell_a.size
        parts.append(
            ConnectionTable(
                cell_a=cell_a,
                cell_b=cell_b,
                area=np.full(n, area, dtype=float),
                dist_a=np.full(n, dist, dtype=float),
                dist_b=np.full(n, dist, dtype=float),
                direction=np.full(n, direction, dtype=np.int8),
            )
        )
    return ConnectionTable.concat(*parts)


def build_diagonal_connections(geometry: CellGeometry) -> ConnectionTable:
    """Build the in-layer diagonal connections of a regular cartesian grid.

    With only the I and J neighbours, the shortest path between two cells is a staircase
    and the equivalent-distance metric is Manhattan: its unit "circle" is a diamond
    aligned with the grid, so a diagonal path measures up to ``sqrt(2)`` -- 41% -- longer
    than the straight line, while an axis-aligned one is exact. Every downstream result
    inherits that anisotropy, and it is worst exactly where the injector work needs it
    least: the set of cells lying on a shortest path between two wells degenerates to a
    line segment when the wells are grid-aligned and swells to a whole rectangle when
    they are diagonal.

    Adding both diagonals of each cell turns the metric octagonal and cuts the worst-case
    error to about 8%. Nothing downstream changes -- this is a plain
    :class:`ConnectionTable` -- which is the point of the neutral connection layer.

    Args:
        geometry: A regular :class:`CellGeometry` (``cell_size`` must be set).

    Returns:
        The connection table of the ``(i+1, j+1)`` and ``(i+1, j-1)`` neighbours, with
        ``direction = DIR_IJ``.

    Raises:
        ValueError: If the geometry is not regular.
    """
    if geometry.cell_size is None:
        raise ValueError("build_diagonal_connections needs a regular grid.")
    ni, nj, nk = geometry.shape
    d_i, d_j, d_k = geometry.cell_size
    index = np.arange(ni * nj * nk).reshape(nk, nj, ni)

    # Half the centre-to-centre diagonal, so a diagonal step costs sqrt(dI^2 + dJ^2) in
    # the reference model -- the true straight-line distance, which is the whole point.
    half = 0.5 * float(np.hypot(d_i, d_j))
    # The face a diagonal "sees" is not a real grid face. Only T_ab uses it, and stage 3
    # (distances.py) uses distance ratios, not T_ab directly, so the mean of the two real
    # face areas is an adequate stand-in.
    area = 0.5 * (d_j * d_k + d_i * d_k)

    parts = []
    for slice_a, slice_b in (
        (np.s_[:, :-1, :-1], np.s_[:, 1:, 1:]),   # (i, j) <-> (i+1, j+1)
        (np.s_[:, 1:, :-1], np.s_[:, :-1, 1:]),   # (i, j) <-> (i+1, j-1)
    ):
        cell_a = index[slice_a].ravel()
        cell_b = index[slice_b].ravel()
        if cell_a.size == 0:
            continue
        low = np.minimum(cell_a, cell_b)
        high = np.maximum(cell_a, cell_b)
        n = low.size
        parts.append(
            ConnectionTable(
                cell_a=low,
                cell_b=high,
                area=np.full(n, area, dtype=float),
                dist_a=np.full(n, half, dtype=float),
                dist_b=np.full(n, half, dtype=float),
                direction=np.full(n, DIR_IJ, dtype=np.int8),
            )
        )
    return ConnectionTable.concat(*parts)


def build_corner_connections(geometry: CellGeometry) -> ConnectionTable:
    """Build connections for an irregular corner-point grid. **Not implemented yet.**

    The intended behaviour is documented in ``context/ARCHITECTURE.md``: return the same
    :class:`ConnectionTable` structure, including connections between cells that are not
    ``(i+-1, j+-1, k+-1)`` neighbours (faults, pinch-outs). Because every downstream
    module already consumes the table, implementing this function is the only work
    needed to support irregular grids.

    Args:
        geometry: The corner-point cell geometry.

    Raises:
        NotImplementedError: Always.
    """
    raise NotImplementedError(
        "Irregular CORNERS connections are not implemented yet. Fill in this builder "
        "following the contract in context/ARCHITECTURE.md; nothing downstream needs "
        "to change. For a regular CORNERS grid, load_model already detects regularity "
        "and build_regular_connections works as-is."
    )


def build_connections(
    geometry: CellGeometry, *, diagonal: bool = False
) -> ConnectionTable:
    """Build the connection table appropriate for the given geometry.

    Args:
        geometry: The cell geometry.
        diagonal: Also add the in-layer diagonal neighbours
            (:func:`build_diagonal_connections`), which makes the shortest-path metric a
            much better approximation of straight-line distance. Regular grids only.

    Returns:
        Connections from :func:`build_regular_connections` for a regular grid, or from
        :func:`build_corner_connections` otherwise.
    """
    if not geometry.is_regular:
        return build_corner_connections(geometry)
    table = build_regular_connections(geometry)
    if diagonal:
        table = ConnectionTable.concat(table, build_diagonal_connections(geometry))
    return table


def make_custom_connection(
    geometry: CellGeometry,
    cell_a: int,
    cell_b: int,
    area: float,
    direction: int = DIR_I,
) -> ConnectionTable:
    """Build a single connection between two arbitrary cells (an NNC).

    Centre-to-face distances are taken as half the straight-line distance between the
    two cell centres, i.e. the face is assumed to sit midway between them.

    Args:
        geometry: The cell geometry, used for the centre coordinates.
        cell_a: Linear index of the first cell.
        cell_b: Linear index of the second cell.
        area: Flow-face area in m2.
        direction: Direction code deciding which permeability applies and whether NTG
            is used.

    Returns:
        A one-row :class:`ConnectionTable`, ready to be concatenated with others.
    """
    low, high = (cell_a, cell_b) if cell_a < cell_b else (cell_b, cell_a)
    half = 0.5 * float(np.linalg.norm(geometry.centre[high] - geometry.centre[low]))
    return ConnectionTable(
        cell_a=np.array([low], dtype=np.int64),
        cell_b=np.array([high], dtype=np.int64),
        area=np.array([float(area)], dtype=float),
        dist_a=np.array([half], dtype=float),
        dist_b=np.array([half], dtype=float),
        direction=np.array([direction], dtype=np.int8),
    )
