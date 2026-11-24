"""Stage 3 -- equivalent distance between cells.

The geometric distance between cell centres is stretched by a factor ``alpha`` that
compares the cell against a reference model with the same geometry, ``POR = 0.10`` and
``k = 100 mD`` everywhere::

    alpha = sqrt( (PV/PV_ref) / (T/T_ref) )

Because the reference shares the geometry, ``Vbulk``, ``area`` and ``dist`` cancel and
``alpha`` reduces to a closed form that depends only on the rock::

    alpha_s = sqrt( (POR_s/0.10) * (100 / (k_s[d] * NTG_s)) )      # NTG absent for K

The equivalent distance of a connection is ``alpha_a*dist_a + alpha_b*dist_b``, and the
equivalent distance between two arbitrary cells is the shortest path over those weights.

Since ``alpha`` is a ratio of the cell to the reference, the reference model itself gives
``alpha == 1`` and equivalent distance == physical distance. That identity is the
package's strongest sanity check and is covered by a test.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterator

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra

from .config import REFERENCE_PERMEABILITY, REFERENCE_POROSITY, get_logger
from .connections import NTG_DIRECTIONS, ConnectionTable
from .model_io import ReservoirModel


# MARK: Alpha
def alpha_per_side(
    model: ReservoirModel, connections: ConnectionTable
) -> tuple[np.ndarray, np.ndarray]:
    """Return the distance-stretch factor of each side of every connection.

    Args:
        model: The resolved reservoir model.
        connections: The connection table.

    Returns:
        ``(alpha_a, alpha_b)``, each ``(nconn,)``. A side that cannot conduct -- zero
        permeability, zero porosity, zero NTG, or an inactive cell -- gets ``inf``,
        which suppresses the connection when the graph is built.
    """
    por = model.flat("por")
    ntg = model.flat("ntg")
    active = model.geometry.active

    alpha_a = np.full(connections.n_connections, np.inf, dtype=float)
    alpha_b = np.full_like(alpha_a, np.inf)

    for direction in np.unique(connections.direction):
        mask = connections.direction == direction
        perm = model.permeability(int(direction))
        factor = ntg if int(direction) in NTG_DIRECTIONS else np.ones_like(ntg)
        conductivity = perm * factor  # what stands in for k*NTG in the ratio
        for cells, out in (
            (connections.cell_a[mask], alpha_a),
            (connections.cell_b[mask], alpha_b),
        ):
            ok = active[cells] & (conductivity[cells] > 0) & (por[cells] > 0)
            value = np.full(cells.size, np.inf, dtype=float)
            value[ok] = np.sqrt(
                (por[cells][ok] / REFERENCE_POROSITY)
                * (REFERENCE_PERMEABILITY / conductivity[cells][ok])
            )
            out[np.flatnonzero(mask)] = value

    return alpha_a, alpha_b


def connection_weights(
    model: ReservoirModel, connections: ConnectionTable
) -> np.ndarray:
    """Return the equivalent distance of every connection, in equivalent metres.

    Args:
        model: The resolved reservoir model.
        connections: The connection table.

    Returns:
        ``(nconn,)`` array of ``alpha_a*dist_a + alpha_b*dist_b``. Non-conductive
        connections come out as ``inf``.
    """
    alpha_a, alpha_b = alpha_per_side(model, connections)
    return alpha_a * connections.dist_a + alpha_b * connections.dist_b


# MARK: Graph
def build_graph(
    model: ReservoirModel,
    connections: ConnectionTable,
    verbose: bool = True,
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Build the undirected equivalent-distance graph over the active cells.

    Args:
        model: The resolved reservoir model.
        connections: The connection table. Rows touching inactive cells, and rows with
            infinite weight, are dropped here.
        verbose: Print a summary of the graph.

    Returns:
        A tuple ``(graph, node_of_cell, cell_of_node)``:

        - ``graph``: ``(nnode, nnode)`` sparse matrix of equivalent distances.
        - ``node_of_cell``: ``(ncell,)`` graph node of each cell, ``-1`` if inactive.
        - ``cell_of_node``: ``(nnode,)`` linear cell index of each graph node.

    Raises:
        ValueError: If the model has no active cells.
    """
    log = get_logger(verbose)
    active = model.geometry.active
    cell_of_node = np.flatnonzero(active)
    if cell_of_node.size == 0:
        raise ValueError("The model has no active cells; nothing to connect.")

    node_of_cell = np.full(model.n_cells, -1, dtype=np.int64)
    node_of_cell[cell_of_node] = np.arange(cell_of_node.size)

    weights = connection_weights(model, connections)
    keep = (
        np.isfinite(weights)
        & (weights > 0)
        & active[connections.cell_a]
        & active[connections.cell_b]
    )
    rows = node_of_cell[connections.cell_a[keep]]
    cols = node_of_cell[connections.cell_b[keep]]
    data = weights[keep]

    n_nodes = cell_of_node.size
    graph = coo_matrix(
        (
            np.concatenate([data, data]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n_nodes, n_nodes),
    ).tocsr()

    dropped = connections.n_connections - int(keep.sum())
    log.info(
        "Graph: %d active cells, %d connections (%d dropped as non-conductive).",
        n_nodes,
        int(keep.sum()),
        dropped,
    )
    return graph, node_of_cell, cell_of_node


# MARK: DistanceEngine
class DistanceEngine:
    """Shortest equivalent distances, computed lazily and never stored in full.

    The full ``ncell x ncell`` distance matrix is impractical beyond a few thousand
    cells, so two access patterns are offered instead:

    - :meth:`iter_chunks` streams blocks of source rows for the histogram of stage 4;
      each block is consumed and discarded.
    - :meth:`rows` returns single-source rows with an LRU cache, for the annealing of
      stage 5, which revisits the same candidate cells many times.

    Distances are in equivalent metres. Unreachable cells come back as ``inf``.
    """

    def __init__(
        self,
        model: ReservoirModel,
        connections: ConnectionTable,
        *,
        limit: float | None = None,
        cache_size: int = 4096,
        verbose: bool = True,
    ) -> None:
        """Build the graph and prepare the caches.

        Args:
            model: The resolved reservoir model.
            connections: The connection table.
            limit: Optional shortest-path cutoff in equivalent metres. Cells further
                than this are reported as ``inf``, which prunes the search.
            cache_size: Maximum number of single-source rows kept in memory.
            verbose: Print progress messages.
        """
        self.model = model
        self.connections = connections
        self.limit = limit
        self.verbose = verbose
        self._log = get_logger(verbose)
        self.graph, self.node_of_cell, self.cell_of_node = build_graph(
            model, connections, verbose=verbose
        )
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = int(cache_size)

    @property
    def n_nodes(self) -> int:
        """Number of active cells, i.e. graph nodes."""
        return int(self.cell_of_node.size)

    def _dijkstra(self, nodes: np.ndarray) -> np.ndarray:
        """Run Dijkstra from the given graph nodes, honouring ``limit``."""
        kwargs = {"directed": False, "indices": nodes}
        if self.limit is not None:
            kwargs["limit"] = self.limit
        return dijkstra(self.graph, **kwargs)

    def iter_chunks(
        self, chunk: int = 256, sources: np.ndarray | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield blocks of shortest-distance rows, in node space.

        Args:
            chunk: Number of source nodes per block.
            sources: Optional graph nodes to use as sources. Defaults to every node.

        Yields:
            ``(nodes, distances)`` where ``nodes`` is ``(m,)`` graph nodes and
            ``distances`` is ``(m, nnode)`` in equivalent metres.
        """
        nodes = np.arange(self.n_nodes) if sources is None else np.asarray(sources)
        for start in range(0, nodes.size, chunk):
            block = nodes[start : start + chunk]
            yield block, np.atleast_2d(self._dijkstra(block))

    def rows(self, cells) -> np.ndarray:
        """Return shortest distances from the given cells to every **cell**.

        Unlike :meth:`iter_chunks`, this works in cell space: the returned columns are
        indexed by linear cell index, and inactive cells are ``inf``.

        Args:
            cells: A single linear cell index or a sequence of them. Every cell must be
                active.

        Returns:
            ``(ncell,)`` for a scalar input, otherwise ``(len(cells), ncell)``.

        Raises:
            ValueError: If any requested cell is inactive.
        """
        scalar = np.ndim(cells) == 0
        cell_list = np.atleast_1d(np.asarray(cells, dtype=np.int64))
        nodes = self.node_of_cell[cell_list]
        if np.any(nodes < 0):
            bad = cell_list[nodes < 0]
            raise ValueError(f"Cells {bad.tolist()} are inactive and have no distances.")

        found = {}
        missing = []
        for c in dict.fromkeys(cell_list.tolist()):
            if c in self._cache:
                found[c] = self._touch(c)
            else:
                missing.append(c)
        if missing:
            block = np.atleast_2d(self._dijkstra(self.node_of_cell[np.asarray(missing)]))
            for cell, node_row in zip(missing, block):
                found[cell] = self._to_cell_space(node_row)
                self._store(cell, found[cell])

        out = np.stack([found[int(c)] for c in cell_list])
        return out[0] if scalar else out

    def _to_cell_space(self, node_row: np.ndarray) -> np.ndarray:
        """Expand a distance row from node space to cell space."""
        row = np.full(self.model.n_cells, np.inf, dtype=np.float32)
        row[self.cell_of_node] = node_row
        return row

    def _store(self, cell: int, row: np.ndarray) -> None:
        """Insert a row into the LRU cache, evicting the oldest entry if needed."""
        self._cache[cell] = row
        self._cache.move_to_end(cell)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _touch(self, cell: int) -> np.ndarray:
        """Return a cached row and mark it as recently used."""
        self._cache.move_to_end(cell)
        return self._cache[cell]

    def cache_info(self) -> dict[str, int]:
        """Return the current cache occupancy, for logging."""
        return {"cached_rows": len(self._cache), "capacity": self._cache_size}
