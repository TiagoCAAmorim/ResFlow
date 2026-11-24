"""Stage 2 -- pore volume and connection transmissibilities.

Both quantities are computed from a :class:`~well_placement.connections.ConnectionTable`,
so the same code serves a regular grid and an irregular one. For one side ``s`` of a
connection in direction ``d``::

    T_s  = k_s[d] * NTG_s * area / dist_s        (NTG only applies to I and J)
    T_ab = 1 / (1/T_a + 1/T_b)                   (harmonic mean)

which is the scalar form of ``T_i = k_i * NTG_i * (A.D)/(D.D)`` when the face normal is
parallel to the centre-to-face vector.

Units are mD*m2/m. :data:`TRANS_UNIT_FACTOR` exists for documentation only: the
equivalent distance uses ratios against a reference model, so any constant factor
cancels out.
"""
from __future__ import annotations

import numpy as np

from .connections import NTG_DIRECTIONS, ConnectionTable
from .model_io import ReservoirModel

#: Constant that would convert mD*m2/m into simulator transmissibility units. Left at
#: 1.0 because stage 3 only uses ratios, where it cancels.
TRANS_UNIT_FACTOR = 1.0


def pore_volume(model: ReservoirModel) -> np.ndarray:
    """Return the oil-bearing pore volume of every cell.

    ``V = Vbulk * POR * NTG * (1 - Swi)``. With the default ``Swi = 0`` this is plain
    pore volume. Inactive cells get zero.

    Args:
        model: The resolved reservoir model.

    Returns:
        ``(ncell,)`` volume in m3, in linear cell order.
    """
    volume = (
        model.geometry.bulk_volume
        * model.flat("por")
        * model.flat("ntg")
        * (1.0 - model.swi)
    )
    return np.where(model.geometry.active, volume, 0.0)


def half_transmissibility(
    model: ReservoirModel, connections: ConnectionTable
) -> tuple[np.ndarray, np.ndarray]:
    """Return the half transmissibility of each side of every connection.

    Args:
        model: The resolved reservoir model.
        connections: The connection table.

    Returns:
        ``(T_a, T_b)``, each ``(nconn,)`` in mD*m2/m. A side with zero permeability or
        zero NTG gets ``T = 0``, which makes the connection non-conductive.
    """
    ntg = model.flat("ntg")
    t_a = np.zeros(connections.n_connections, dtype=float)
    t_b = np.zeros_like(t_a)

    for direction in np.unique(connections.direction):
        mask = connections.direction == direction
        perm = model.permeability(int(direction))
        factor = ntg if int(direction) in NTG_DIRECTIONS else np.ones_like(ntg)
        cell_a = connections.cell_a[mask]
        cell_b = connections.cell_b[mask]
        area = connections.area[mask]
        t_a[mask] = perm[cell_a] * factor[cell_a] * area / connections.dist_a[mask]
        t_b[mask] = perm[cell_b] * factor[cell_b] * area / connections.dist_b[mask]

    return TRANS_UNIT_FACTOR * t_a, TRANS_UNIT_FACTOR * t_b


def pair_transmissibility(
    model: ReservoirModel, connections: ConnectionTable
) -> np.ndarray:
    """Return the harmonic pair transmissibility of every connection.

    Exported for QC and reporting. The equivalent distance of stage 3 uses the two half
    transmissibilities separately, not this value.

    Args:
        model: The resolved reservoir model.
        connections: The connection table.

    Returns:
        ``(nconn,)`` array of ``T_ab = 1/(1/T_a + 1/T_b)``, zero where either side is
        non-conductive.
    """
    t_a, t_b = half_transmissibility(model, connections)
    total = t_a + t_b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(total > 0, t_a * t_b / total, 0.0)
    return result
