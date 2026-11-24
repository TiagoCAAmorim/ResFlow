"""Stage 5 -- place producer wells (minimum distance is a hard constraint here) with simulated annealing.

Every well here is a producer: there is no injector/role machinery, streamtube
coupling, or role-swap search. Which of the optimised positions become injectors is
decided afterwards, by ranking positions by Y_at (see `well_placement_batch.build_row`)
-- not by this module.

A well set is scored by how much discounted volume it drains. Every cell is assigned to
exactly one well -- the one it is closest to in equivalent distance -- so wells do not
double-count the same volume::

    contribution(c) = V_c * exp(-lambda * d(c, W))

Two penalties shape the result::

    F      = sum_W Y_at_W
    P_bal  = w_bal  * F * CV(Y_at_W)                              imbalance
    P_dist = w_dist * F * sum_pairs max(0, (d_min - d)/d_min)^2   crowding
    score  = F - P_bal - P_dist

Search: a greedy seed (wells on the best cells that respect ``d_min``), then annealing
over positions, then a deterministic 1-opt polish.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage

from .config import SAConfig, get_logger
from .distances import DistanceEngine
from .geometry import index_to_ijk
from .model_io import ReservoirModel
from .transmissibility import pore_volume


class InvalidLayoutError(ValueError):
    """No valid well layout was found within ``SAConfig.max_attempts`` attempts."""


# MARK: Result
@dataclass
class OptimizationResult:
    """Outcome of a well-placement run.

    Attributes:
        wells: ``(n_wells,)`` linear cell indices of the optimised positions.
        wells_ijk: 1-based ``(I, J, K)`` of each well.
        wells_xy: ``(n_wells, 2)`` well coordinates in metres.
        y_at_per_producer: Discounted volume drained by each well, aligned with ``wells``.
        y_at_total: Sum of ``y_at_per_producer``.
        score: Objective value, i.e. ``y_at_total`` minus the penalties.
        owner: ``(ncell,)`` index into ``y_at_per_producer`` of the well each cell
            belongs to, ``-1`` if the cell is inactive or unreachable.
        initial_wells: Linear indices of the greedy starting positions.
        initial_y_at_per_producer: Per-well index of the starting configuration.
        initial_owner: Cell assignment of the starting configuration.
        initial_score: Objective value of the starting configuration.
        d_min_used: Minimum distance enforced (never relaxed).
        d_min_relaxations: Always 0, kept for compatibility.
        history: Per-iteration ``(iteration, current score, best score)``.
        accepted: Number of accepted annealing moves.
    """

    wells: np.ndarray
    wells_ijk: list[tuple[int, int, int]]
    wells_xy: np.ndarray
    y_at_per_producer: np.ndarray
    y_at_total: float
    score: float
    owner: np.ndarray
    initial_wells: np.ndarray
    initial_y_at_per_producer: np.ndarray
    initial_owner: np.ndarray
    initial_score: float
    d_min_used: float
    d_min_relaxations: int = 0
    history: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    accepted: int = 0
    attempts: int = 1
    seed_used: int = 0


# MARK: Objective
class WellObjective:
    """Scores a set of well positions.

    Holds everything the score needs -- volumes, discount decay, the distance engine and
    the penalty weights -- so the annealing loop only passes candidate positions.
    """

    def __init__(
        self,
        model: ReservoirModel,
        engine: DistanceEngine,
        rate: float,
        x_base: float,
        sa_cfg: SAConfig,
    ) -> None:
        self.model = model
        self.engine = engine
        self.volume = pore_volume(model)
        self.rate = rate
        self.x_base = x_base
        self.decay = math.log1p(rate) / x_base
        self.cfg = sa_cfg
        self.centre = model.geometry.centre
        self.low_k = np.sqrt(model.permi.ravel() * model.permj.ravel()) < sa_cfg.k_min
        self.region = self._flow_regions(model, self.low_k) if sa_cfg.k_min > 0 else None

    @staticmethod
    def _flow_regions(model: ReservoirModel, low_k: np.ndarray) -> np.ndarray:
        """Label the cells that can communicate, I and J neighbours only.

        Cells below ``k_min`` and inactive cells are barriers; diagonals do not connect.
        Layers are never joined. Barrier cells get label ``0``.
        """
        ni, nj, nk = model.shape  # property arrays are (nk, nj, ni)
        open_cells = (model.geometry.active & ~low_k).reshape(nk, nj, ni)
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = structure[1, :, 1] = True  # I and J neighbours only
        labels, _ = ndimage.label(open_cells, structure=structure)
        return labels.ravel()

    def isolated(self, wells) -> np.ndarray:
        """Boolean per well: it shares no flow region with any other well.

        Always ``False`` for a single well or when ``k_min`` is off.
        """
        wells = np.asarray(wells)
        if self.region is None or wells.size < 2:
            return np.zeros(wells.size, dtype=bool)
        labels = self.region[wells]
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        return (labels == 0) | (counts[inverse] < 2)

    def assign(self, wells) -> tuple[np.ndarray, np.ndarray]:
        """Assign every cell to its nearest well in equivalent distance.

        Args:
            wells: Linear cell indices of every well.

        Returns:
            ``(owner, distance)``: the winning well index per cell (``-1`` when no well
            reaches the cell), and the equivalent distance to that well (``inf`` when
            unreachable).
        """
        rows = np.atleast_2d(self.engine.rows(np.asarray(wells)))
        owner = np.argmin(rows, axis=0)
        distance = rows[owner, np.arange(rows.shape[1])]
        owner = np.where(np.isfinite(distance), owner, -1)
        return owner, distance

    def per_producer_index(self, wells) -> tuple[np.ndarray, np.ndarray]:
        """Return the discounted volume drained by each well, and the assignment."""
        owner, distance = self.assign(wells)
        reached = owner >= 0
        cells = np.flatnonzero(reached)
        contribution = self.volume[cells] * np.exp(-self.decay * distance[cells])
        y_at = np.bincount(owner[cells], weights=contribution, minlength=len(wells))
        y_at[self.low_k[np.asarray(wells)]] = 0.0  # a well below k_min drains nothing
        y_at[self.isolated(wells)] = 0.0  # nor does one cut off from every other well
        return y_at, owner

    def validity_problems(self, wells, y_at) -> list[str]:
        """List why a final layout is not valid; empty when it is.

        A well is invalid on a cell below ``k_min``, or with ``Y_at`` under
        ``score_min_frac`` times the mean ``Y_at`` of the other wells.
        """
        problems = [
            f"well {w + 1} sits on a cell with k < {self.cfg.k_min:g} mD"
            for w in np.flatnonzero(self.low_k[np.asarray(wells)])
        ]
        problems += [
            f"well {w + 1} is isolated from the other wells"
            for w in np.flatnonzero(self.isolated(wells) & ~self.low_k[np.asarray(wells)])
        ]
        if self.region is not None and self.cfg.n_injectors > 0 and y_at.size > 1:
            n_regions = len(set(self.region[np.asarray(wells)].tolist()) - {0})
            if n_regions > min(self.cfg.n_injectors, self.cfg.n_wells - self.cfg.n_injectors):
                problems.append(
                    f"{n_regions} separate flow regions cannot each get a producer and an injector "
                    f"({self.cfg.n_wells - self.cfg.n_injectors}P{self.cfg.n_injectors}I)"
                )
        if self.cfg.score_min_frac > 0 and y_at.size > 1:
            for w in range(y_at.size):
                others = (y_at.sum() - y_at[w]) / (y_at.size - 1)
                if y_at[w] < self.cfg.score_min_frac * others:
                    problems.append(
                        f"well {w + 1} is {100 * y_at[w] / others:.1f}% the mean Y_at "
                        f"(limit={100 * self.cfg.score_min_frac:.1f}%)"
                    )
        return problems

    def spacing(self, wells) -> np.ndarray:
        """Return the pairwise distances between wells, in the configured metric."""
        wells = np.asarray(wells)
        if wells.size < 2:
            return np.empty(0, dtype=float)
        if self.cfg.metric == "equivalent":
            rows = np.atleast_2d(self.engine.rows(wells))
            iu = np.triu_indices(wells.size, k=1)
            return rows[iu[0], wells[iu[1]]]
        points = self.centre[wells]
        iu = np.triu_indices(wells.size, k=1)
        return np.linalg.norm(points[iu[0]] - points[iu[1]], axis=1)

    @staticmethod
    def _shortfall(gaps: np.ndarray, minimum: float) -> float:
        """Return the squared relative shortfall of every gap below ``minimum``."""
        if minimum <= 0 or gaps.size == 0:
            return 0.0
        shortfall = np.maximum(0.0, (minimum - gaps) / minimum)
        return float(np.sum(shortfall**2))

    def score(self, wells, d_min: float | None = None) -> tuple[float, np.ndarray, np.ndarray]:
        """Return the objective value of a well set.

        Returns:
            ``(score, y_at_per_producer, owner)``.
        """
        d_min = self.cfg.d_min if d_min is None else d_min
        y_at, owner = self.per_producer_index(wells)
        total = float(y_at.sum())
        if total <= 0:
            return 0.0, y_at, owner

        penalty = 0.0
        if y_at.size > 1:
            penalty += self.cfg.w_bal * float(y_at.std() / y_at.mean())
        penalty += self.cfg.w_dist * self._shortfall(self.spacing(wells), d_min)
        return total * (1.0 - penalty), y_at, owner


# MARK: Greedy seed
class NoRoomError(ValueError):
    """The requested number of wells cannot be placed with the minimum distance."""


def _greedy(ranked: np.ndarray, centre: np.ndarray, d_min: float, n_wells: int | None) -> list[int]:
    chosen: list[int] = []
    for cell in ranked:
        if n_wells is not None and len(chosen) == n_wells:
            break
        if chosen and d_min > 0:
            gaps = np.linalg.norm(centre[chosen] - centre[cell], axis=1)
            if gaps.min() < d_min:
                continue
        chosen.append(int(cell))
    return chosen


def max_wells_fit(y_at: np.ndarray, model: ReservoirModel, d_min: float) -> int:
    """Number of wells the greedy placement (best Y_at first) fits with the minimum distance."""
    candidates = np.flatnonzero(np.isfinite(y_at))
    ranked = candidates[np.argsort(y_at[candidates], kind="stable")[::-1]]
    return len(_greedy(ranked, model.geometry.centre, float(d_min), None))


def respects_d_min(wells: np.ndarray, centre: np.ndarray, d_min: float) -> bool:
    """True if every pair of wells is at least ``d_min`` metres apart (cell-centre distance)."""
    if d_min <= 0 or len(wells) < 2:
        return True
    pts = centre[np.asarray(wells)]
    iu = np.triu_indices(len(wells), k=1)
    return bool(np.linalg.norm(pts[iu[0]] - pts[iu[1]], axis=1).min() >= d_min)


def greedy_initial_wells(
    y_at: np.ndarray,
    model: ReservoirModel,
    sa_cfg: SAConfig,
    *,
    n_wells: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, float, int]:
    """Place wells greedily on the best cells that respect the minimum distance.

    Cells are tried in decreasing order of ``Y_at``. The minimum distance is never relaxed.

    Returns:
        ``(wells, d_min, 0)``.

    Raises:
        NoRoomError: If ``n_wells`` wells do not fit with the minimum distance.
    """
    n_wells = sa_cfg.n_wells if n_wells is None else int(n_wells)
    candidates = np.flatnonzero(np.isfinite(y_at))
    ranked = candidates[np.argsort(y_at[candidates], kind="stable")[::-1]]
    chosen = _greedy(ranked, model.geometry.centre, float(sa_cfg.d_min), n_wells)
    if len(chosen) < n_wells:
        raise NoRoomError(
            f"Only {len(chosen)} of {n_wells} wells fit with a {sa_cfg.d_min:.0f} m minimum distance."
        )
    return np.asarray(chosen, dtype=np.int64), float(sa_cfg.d_min), 0


# MARK: Annealing
def _optimize_once(
    model: ReservoirModel,
    engine: DistanceEngine,
    y_at: np.ndarray,
    rate: float,
    x_base: float,
    sa_cfg: SAConfig,
    *,
    verbose: bool = True,
) -> OptimizationResult:
    """Optimise well positions with simulated annealing.

    Returns:
        The :class:`OptimizationResult`, including the 1-based ``(I, J, K)`` positions.
    """
    log = get_logger(verbose)
    rng = np.random.default_rng(sa_cfg.seed)
    objective = WellObjective(model, engine, rate, x_base, sa_cfg)

    # Search aid: seed and move only on cells at or above k_min. The Y_at = 0 rule alone gives
    # the search no gradient to climb out of a tight region.
    open_y_at = np.where(objective.low_k, np.nan, y_at)
    if np.isfinite(open_y_at).sum() < sa_cfg.n_wells:
        open_y_at = y_at
    wells, d_min_used, relaxations = greedy_initial_wells(
        open_y_at, model, sa_cfg, verbose=verbose
    )
    init_score, init_y_at, init_owner = objective.score(wells, d_min_used)
    log.info(
        "Greedy seed: score %.4g, %d well(s) at %s.",
        init_score, sa_cfg.n_wells,
        [tuple(int(v) for v in index_to_ijk(model.shape, c)) for c in wells],
    )

    candidates = np.flatnonzero(np.isfinite(open_y_at))
    centre = model.geometry.centre
    extent = model.geometry.extent()
    max_radius = 0.5 * max(extent[1] - extent[0], extent[3] - extent[2])

    current = wells.copy()
    current_score = init_score
    best = current.copy()
    best_score = current_score
    best_y_at, best_owner = init_y_at, init_owner

    t_start = _calibrate_temperature(objective, candidates, sa_cfg, rng, d_min_used)
    t_end = max(t_start * sa_cfg.t_end_ratio, 1e-12)
    cooling = (t_end / t_start) ** (1.0 / sa_cfg.n_iter) if t_start > 0 else 1.0
    log.info("Annealing %d iterations from T=%.4g to T=%.4g.", sa_cfg.n_iter, t_start, t_end)

    history = np.empty((sa_cfg.n_iter, 3), dtype=float)
    temperature = t_start
    accepted = 0

    for iteration in range(sa_cfg.n_iter):
        radius = max_radius * (1.0 - iteration / sa_cfg.n_iter)
        proposal = _propose(current, candidates, centre, radius, sa_cfg, rng)
        if proposal is not None:
            score, y_at_w, owner = objective.score(proposal, d_min_used)
            delta = score - current_score
            if delta > 0 or rng.random() < math.exp(delta / temperature):
                current, current_score = proposal, score
                accepted += 1
                if score > best_score:
                    best = proposal.copy()
                    best_score = score
                    best_y_at, best_owner = y_at_w, owner
        history[iteration] = (iteration, current_score, best_score)
        temperature *= cooling

    if sa_cfg.polish:
        radius = sa_cfg.polish_radius
        if radius is None:
            radius = 3.0 * float(model.geometry.typical_cell_width(best).mean())
        best, best_score, best_y_at, best_owner = _polish(
            objective, best, best_score, candidates, d_min_used, radius, centre, log,
        )

    log.info(
        "Annealing done: score %.4g (%+.1f%% over the seed), %d/%d moves accepted.",
        best_score,
        100.0 * (best_score / init_score - 1.0) if init_score else float("nan"),
        accepted,
        sa_cfg.n_iter,
    )
    ijk = [tuple(int(v) for v in index_to_ijk(model.shape, c)) for c in best]

    return OptimizationResult(
        wells=best,
        wells_ijk=ijk,
        wells_xy=centre[best][:, :2],
        y_at_per_producer=best_y_at,
        y_at_total=float(best_y_at.sum()),
        score=float(best_score),
        owner=best_owner,
        initial_wells=wells,
        initial_y_at_per_producer=init_y_at,
        initial_owner=init_owner,
        initial_score=float(init_score),
        d_min_used=d_min_used,
        d_min_relaxations=relaxations,
        history=history,
        accepted=accepted,
        seed_used=sa_cfg.seed,
    )


def optimize_wells(
    model: ReservoirModel,
    engine: DistanceEngine,
    y_at: np.ndarray,
    rate: float,
    x_base: float,
    sa_cfg: SAConfig,
    *,
    verbose: bool = True,
) -> OptimizationResult:
    """Optimise well positions, retrying with ``seed + 1``, ... until the layout is valid.

    With ``k_min = 0`` and ``score_min_frac = 0`` the first run is always accepted.

    Raises:
        InvalidLayoutError: If all ``max_attempts`` runs gave an invalid layout.
    """
    log = get_logger(verbose)
    objective = WellObjective(model, engine, rate, x_base, sa_cfg)
    problems: list[str] = []
    for attempt in range(1, sa_cfg.max_attempts + 1):
        trial_cfg = replace(sa_cfg, seed=sa_cfg.seed + attempt - 1)
        result = _optimize_once(model, engine, y_at, rate, x_base, trial_cfg, verbose=verbose)
        problems = objective.validity_problems(result.wells, result.y_at_per_producer)
        if not problems:
            result.attempts = attempt
            return result
        log.info(
            "Attempt %d/%d (seed %d) gave an invalid layout: %s.",
            attempt, sa_cfg.max_attempts, trial_cfg.seed, "; ".join(problems),
        )
    raise InvalidLayoutError(
        f"no valid layout in {sa_cfg.max_attempts} attempt(s); last: " + "; ".join(problems)
    )


def _calibrate_temperature(
    objective: WellObjective,
    candidates: np.ndarray,
    sa_cfg: SAConfig,
    rng: np.random.Generator,
    d_min: float,
) -> float:
    """Estimate a starting temperature from the spread of random configurations."""
    if sa_cfg.burn_in < 2:
        return 1.0
    scores = [
        objective.score(
            rng.choice(candidates, size=sa_cfg.n_wells, replace=False), d_min
        )[0]
        for _ in range(sa_cfg.burn_in)
    ]
    spread = float(np.std(scores))
    return spread if spread > 0 else max(abs(float(np.mean(scores))) * 1e-3, 1e-9)


def _propose(
    current: np.ndarray,
    candidates: np.ndarray,
    centre: np.ndarray,
    radius: float,
    sa_cfg: SAConfig,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Propose a new configuration by relocating one well.

    Relocation is local within ``radius`` metres, shrinking as the run cools, with an
    occasional uniform jump to escape local optima.

    Returns:
        The proposed well positions, or ``None`` when the draw would duplicate a well.
    """
    which = int(rng.integers(current.size))
    if rng.random() < sa_cfg.jump_prob or radius <= 0:
        pool = candidates
    else:
        gaps = np.linalg.norm(centre[candidates] - centre[current[which]], axis=1)
        pool = candidates[gaps <= radius]
        if pool.size == 0:
            pool = candidates
    new_cell = int(rng.choice(pool))
    if new_cell in current:
        return None
    proposal = current.copy()
    proposal[which] = new_cell
    if not respects_d_min(proposal, centre, sa_cfg.d_min):
        return None
    return proposal


def _polish(
    objective: WellObjective,
    wells: np.ndarray,
    score: float,
    candidates: np.ndarray,
    d_min: float,
    radius: float,
    centre: np.ndarray,
    log,
    max_sweeps: int = 10,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Run a deterministic 1-opt sweep until no local move improves the score.

    Returns:
        ``(wells, score, y_at_per_producer, owner)`` after polishing.
    """
    wells = wells.copy()
    best_y_at, best_owner = objective.per_producer_index(wells)
    evaluations = 0
    for sweep in range(1, max_sweeps + 1):
        improved = False
        for which in range(wells.size):
            gaps = np.linalg.norm(centre[candidates] - centre[wells[which]], axis=1)
            for cell in candidates[gaps <= radius]:
                if cell in wells:
                    continue
                trial = wells.copy()
                trial[which] = cell
                if not respects_d_min(trial, centre, d_min):
                    continue
                trial_score, trial_y_at, trial_owner = objective.score(trial, d_min)
                evaluations += 1
                if trial_score > score:
                    wells, score = trial, trial_score
                    best_y_at, best_owner = trial_y_at, trial_owner
                    improved = True
        if not improved:
            break
    log.info(
        "1-opt polish: %d sweep(s), %d evaluations within %.0f m, final score %.4g.",
        sweep, evaluations, radius, score,
    )
    return wells, score, best_y_at, best_owner
