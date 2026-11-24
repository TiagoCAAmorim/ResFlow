"""Stage 4 -- the discounted volume index ``Y_at``.

Two steps, deliberately kept apart:

1. :func:`volume_distance_histogram` bins, for every source cell, the volume of all other
   cells by their minimum equivalent distance. This is the expensive part (one Dijkstra
   per cell) and it is cached on disk.
2. :func:`discounted_index` collapses that histogram into one number per cell by
   discounting each bin like a cash flow.

The split matters: the histogram depends on the model and on ``delta_x``, but **not** on
``rate`` or ``x_base``. Sweeping those two parameters is therefore a matrix-vector
product, which is what makes :func:`sweep_index` instant.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import IndexConfig, get_logger
from .connections import ConnectionTable
from .distances import DistanceEngine
from .model_io import ReservoirModel
from .transmissibility import pore_volume

#: Bump this whenever the meaning or layout of the cached histogram changes.
CACHE_VERSION = 1


# MARK: Cache key
def histogram_cache_key(
    model: ReservoirModel,
    connections: ConnectionTable,
    delta_x: float,
    limit: float | None,
) -> str:
    """Return a stable hash of everything the histogram depends on.

    ``rate`` and ``x_base`` are deliberately excluded -- they do not change the
    histogram, only how it is collapsed.

    Args:
        model: The resolved reservoir model.
        connections: The connection table.
        delta_x: Bin width in equivalent metres.
        limit: Shortest-path cutoff, or ``None``.

    Returns:
        A 16-character hexadecimal digest.
    """
    digest = hashlib.sha1()
    digest.update(str(CACHE_VERSION).encode())
    for array in (
        model.por, model.permi, model.permj, model.permk, model.ntg,
        model.geometry.active, model.geometry.centre, model.geometry.bulk_volume,
        connections.cell_a, connections.cell_b, connections.area,
        connections.dist_a, connections.dist_b, connections.direction,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(f"{model.swi}|{delta_x}|{limit}".encode())
    return digest.hexdigest()[:16]


# MARK: Histogram
def volume_distance_histogram(
    model: ReservoirModel,
    engine: DistanceEngine,
    delta_x: float,
    *,
    chunk: int = 256,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin the volume reachable from each cell by minimum equivalent distance.

    Bin edges start at ``X_0 = -eps`` so that the source cell itself, at distance zero,
    falls into the first bin -- as the specification requires. ``hist[s, m]`` is the
    ``deltaY`` of bin ``m`` seen from source ``s``; the cumulative curve of the
    specification is ``cumsum(hist[s])`` against ``x_edges[1:]``.

    Args:
        model: The resolved reservoir model.
        engine: The distance engine holding the graph.
        delta_x: Bin width in equivalent metres.
        chunk: Number of source cells processed per Dijkstra call.
        cache_dir: Directory for the ``.npz`` cache. ``None`` disables caching.
        verbose: Print progress messages.

    Returns:
        ``(hist, x_edges)`` where ``hist`` is ``(nnode, nbins)`` in node order (use
        ``engine.cell_of_node`` to map back to cells) and ``x_edges`` is ``(nbins+1,)``.
    """
    log = get_logger(verbose)
    cache_path = None
    if cache_dir is not None:
        key = histogram_cache_key(model, engine.connections, delta_x, engine.limit)
        cache_path = Path(cache_dir) / f"hist_{model.label}_{key}.npz"
        if cache_path.is_file():
            with np.load(cache_path) as data:
                log.info("Histogram loaded from cache: %s", cache_path.name)
                return data["hist"], data["x_edges"]

    volume = pore_volume(model)[engine.cell_of_node]
    n_nodes = engine.n_nodes
    eps = delta_x * 1e-6

    # Single pass over the Dijkstra blocks. The number of bins is not known in advance,
    # so the histogram grows on demand instead of costing a second full pass.
    log.info("Computing equivalent distances for %d cells...", n_nodes)
    hist = np.zeros((n_nodes, 1), dtype=float)
    max_distance = 0.0
    for nodes, block in engine.iter_chunks(chunk=chunk):
        reachable = np.isfinite(block)
        if not reachable.any():
            continue
        max_distance = max(max_distance, float(block[reachable].max()))
        bins = np.floor((block[reachable] + eps) / delta_x).astype(np.int64)
        needed = int(bins.max()) + 1
        if needed > hist.shape[1]:
            hist = np.pad(hist, ((0, 0), (0, needed - hist.shape[1])))
        n_bins = hist.shape[1]

        rows = np.broadcast_to(
            np.arange(nodes.size)[:, None], block.shape
        )[reachable]
        weights = np.broadcast_to(volume, block.shape)[reachable]
        counts = np.bincount(
            rows * n_bins + bins, weights=weights, minlength=nodes.size * n_bins
        )
        hist[nodes] += counts.reshape(nodes.size, n_bins)

    n_bins = hist.shape[1]
    x_edges = -eps + np.arange(n_bins + 1) * delta_x
    log.info(
        "Histogram: %d bins of %.4g equivalent m (max distance %.4g m).",
        n_bins, delta_x, max_distance,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, hist=hist, x_edges=x_edges)
        log.info("Histogram cached at %s", cache_path.name)
    return hist, x_edges


def cumulative_volume_curve(
    hist: np.ndarray, x_edges: np.ndarray, node: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(X, Y)`` distance-versus-cumulative-volume curve of one cell.

    Args:
        hist: The ``(nnode, nbins)`` histogram.
        x_edges: The ``(nbins+1,)`` bin edges.
        node: Graph node (not linear cell index) of the cell of interest.

    Returns:
        ``(x, y)`` with ``x = x_edges[1:]``. ``y`` starts at the cell's own volume and
        rises to the total reachable volume.
    """
    return x_edges[1:], np.cumsum(hist[node])


# MARK: Discounting
def discount_factors(x_edges: np.ndarray, rate: float, x_base: float) -> np.ndarray:
    """Return the discount factor applied to each histogram bin.

    Args:
        x_edges: The ``(nbins+1,)`` bin edges.
        rate: Discount rate per ``x_base`` of equivalent distance.
        x_base: Equivalent distance of one discount period.

    Returns:
        ``(nbins,)`` factors ``(1+rate)**(-x_right/x_base)``, evaluated at the right edge
        of each bin, matching ``Y_at_i = deltaY_i / (1+rate)**((X_i+deltaX)/X_base)``.
    """
    return np.power(1.0 + rate, -x_edges[1:] / x_base)


def discounted_index(
    hist: np.ndarray, x_edges: np.ndarray, rate: float, x_base: float
) -> np.ndarray:
    """Collapse the histogram into one discounted volume per cell.

    Args:
        hist: The ``(nnode, nbins)`` histogram.
        x_edges: The ``(nbins+1,)`` bin edges.
        rate: Discount rate per ``x_base`` of equivalent distance.
        x_base: Equivalent distance of one discount period.

    Returns:
        ``(nnode,)`` array of ``Y_at``, in node order.
    """
    return hist @ discount_factors(x_edges, rate, x_base)


def discounted_index_exact(
    model: ReservoirModel,
    engine: DistanceEngine,
    rate: float,
    x_base: float,
    *,
    chunk: int = 256,
    verbose: bool = True,
) -> np.ndarray:
    """Compute ``Y_at`` without binning -- the ``delta_x -> 0`` limit.

    Discounting is exponential in distance, so the binned sum is an approximation whose
    error depends on ``delta_x``. This variant discounts every cell individually and is
    the reference against which that error can be measured. It cannot reuse the cache.

    Args:
        model: The resolved reservoir model.
        engine: The distance engine.
        rate: Discount rate per ``x_base`` of equivalent distance.
        x_base: Equivalent distance of one discount period.
        chunk: Number of source cells per Dijkstra call.
        verbose: Print progress messages.

    Returns:
        ``(nnode,)`` array of ``Y_at``, in node order.
    """
    log = get_logger(verbose)
    log.info("Computing the exact (unbinned) index for %d cells...", engine.n_nodes)
    volume = pore_volume(model)[engine.cell_of_node]
    decay = np.log1p(rate) / x_base
    y_at = np.zeros(engine.n_nodes, dtype=float)
    for nodes, block in engine.iter_chunks(chunk=chunk):
        weights = np.where(np.isfinite(block), np.exp(-decay * np.nan_to_num(block)), 0.0)
        y_at[nodes] = weights @ volume
    return y_at


def index_to_cells(
    y_at_nodes: np.ndarray, engine: DistanceEngine, n_cells: int
) -> np.ndarray:
    """Expand a node-order index into a full ``(ncell,)`` array.

    Args:
        y_at_nodes: ``(nnode,)`` index in node order.
        engine: The distance engine providing the node/cell mapping.
        n_cells: Total number of cells in the grid.

    Returns:
        ``(ncell,)`` array with ``nan`` on inactive cells.
    """
    out = np.full(n_cells, np.nan, dtype=float)
    out[engine.cell_of_node] = y_at_nodes
    return out


def compute_index(
    model: ReservoirModel,
    engine: DistanceEngine,
    index_cfg: IndexConfig,
    *,
    chunk: int = 256,
    cache_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Run stage 4 end to end for one parameter set.

    Args:
        model: The resolved reservoir model.
        engine: The distance engine.
        index_cfg: Index parameters, including the ``binned``/``exact`` mode.
        chunk: Number of source cells per Dijkstra call.
        cache_dir: Directory for the histogram cache, or ``None``.
        verbose: Print progress messages.

    Returns:
        ``(y_at, hist, x_edges)``. ``y_at`` is ``(ncell,)`` with ``nan`` on inactive
        cells. ``hist`` and ``x_edges`` are ``None`` in ``exact`` mode.
    """
    if index_cfg.mode == "exact":
        nodes = discounted_index_exact(
            model, engine, index_cfg.rate, index_cfg.x_base,
            chunk=chunk, verbose=verbose,
        )
        return index_to_cells(nodes, engine, model.n_cells), None, None

    hist, x_edges = volume_distance_histogram(
        model, engine, index_cfg.delta_x,
        chunk=chunk, cache_dir=cache_dir, verbose=verbose,
    )
    nodes = discounted_index(hist, x_edges, index_cfg.rate, index_cfg.x_base)
    return index_to_cells(nodes, engine, model.n_cells), hist, x_edges


def sweep_index(
    hist: np.ndarray,
    x_edges: np.ndarray,
    rates,
    x_bases,
    engine: DistanceEngine,
    n_cells: int,
) -> dict[tuple[float, float], np.ndarray]:
    """Evaluate ``Y_at`` for a grid of ``(rate, x_base)`` values.

    Each combination costs one matrix-vector product, so a full sweep is essentially
    free once the histogram exists.

    Args:
        hist: The ``(nnode, nbins)`` histogram.
        x_edges: The ``(nbins+1,)`` bin edges.
        rates: Iterable of discount rates.
        x_bases: Iterable of discount base distances.
        engine: The distance engine providing the node/cell mapping.
        n_cells: Total number of cells.

    Returns:
        A dict mapping ``(rate, x_base)`` to the ``(ncell,)`` index array.
    """
    return {
        (float(rate), float(x_base)): index_to_cells(
            discounted_index(hist, x_edges, rate, x_base), engine, n_cells
        )
        for rate in np.atleast_1d(rates)
        for x_base in np.atleast_1d(x_bases)
    }


def single_well_volume_maps(
    model: ReservoirModel,
    engine: DistanceEngine,
    cell: int,
    rate: float,
    x_base: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact swept-volume and discounted-contribution maps of one source cell.

    Unlike the binned histogram, this uses the exact equivalent distance from ``cell``
    to every other cell (one Dijkstra row), so it is exact regardless of the index mode.

    Args:
        model: The resolved reservoir model.
        engine: The distance engine.
        cell: Linear cell index of the source (must be active).
        rate: Discount rate per ``x_base`` of equivalent distance.
        x_base: Discount base distance.

    Returns:
        ``(y_map, discounted_map)``, each ``(ncell,)`` with ``nan`` on unreachable or
        inactive cells.

        - ``y_map``: the cumulative swept volume, i.e. each cell's value is the total
          pore volume of every cell at or closer than its own equivalent distance from
          ``cell`` -- the spatial reading of the ``Y(X)`` curve.
        - ``discounted_map``: each cell's own pore volume discounted by its distance
          from ``cell``. ``np.nansum(discounted_map)`` equals
          :func:`discounted_index_exact` for that source cell, since both sum the same
          per-target discounted volume.
    """
    dist = engine.rows(cell)
    volume = pore_volume(model)
    reachable = np.isfinite(dist)

    y_map = np.full(model.n_cells, np.nan, dtype=float)
    targets = np.flatnonzero(reachable)[np.argsort(dist[reachable])]
    y_map[targets] = np.cumsum(volume[targets])

    decay = np.log1p(rate) / x_base
    discounted_map = np.full(model.n_cells, np.nan, dtype=float)
    discounted_map[reachable] = volume[reachable] * np.exp(-decay * dist[reachable])
    return y_map, discounted_map


#: Default percentile triple shown on the diagnostics panel and used in the sweep
#: report. Petroleum convention: **P0 is the high value** (the maximum, the 100th
#: statistical percentile), P10 the low one.
DEFAULT_PERCENTILES = (("P0", 100.0), ("P05", 95.0), ("P10", 90.0))


# MARK: Percentiles
def percentile_cells(
    y_at: np.ndarray, percentiles: tuple[tuple[str, float], ...] = DEFAULT_PERCENTILES,
) -> dict[str, int]:
    """Pick the cells representing the given statistical percentiles of the index.

    Args:
        y_at: ``(ncell,)`` index with ``nan`` on inactive cells.
        percentiles: ``(name, statistical_percentile)`` pairs, statistical percentile
            in ``[0, 100]``. Defaults to :data:`DEFAULT_PERCENTILES` (P05/P10/P30).

    Returns:
        A dict of ``{name: cell}``, each the active cell whose index is closest to the
        corresponding percentile.

    Raises:
        ValueError: If there is no active cell with a finite index.
    """
    valid = np.flatnonzero(np.isfinite(y_at))
    if valid.size == 0:
        raise ValueError("No active cell has a finite index; cannot pick percentiles.")
    values = y_at[valid]
    out = {}
    for name, statistical in percentiles:
        target = np.percentile(values, statistical)
        out[name] = int(valid[np.argmin(np.abs(values - target))])
    return out
