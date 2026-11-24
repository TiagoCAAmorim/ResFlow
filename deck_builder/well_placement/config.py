"""Configuration dataclasses, YAML (de)serialisation and verbosity control.

`SAConfig` only exposes producer-side parameters: this package always runs producer-only
optimisation (`optimize.py` has no injector/role machinery at all -- see its module
docstring).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

LOGGER_NAME = "well_placement"

#: Reference model used to normalise the equivalent distance (see `distances.py`).
REFERENCE_POROSITY = 0.10
REFERENCE_PERMEABILITY = 100.0  # mD


# MARK: Logging
def get_logger(verbose: bool = True) -> logging.Logger:
    """Return the package logger configured for the requested verbosity."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
    return logger


# MARK: Grid
@dataclass
class GridConfig:
    """Grid geometry and property overrides.

    Attributes:
        cell_size: ``(dI, dJ, dK)`` in metres, or the string ``'corners'`` to read the
            geometry from the ``CORNERS`` include referenced by the ``.dat`` file.
        swi: Initial water saturation used to turn pore volume into oil volume:
            ``V = Vbulk * POR * NTG * (1 - swi)``.
        overrides: Per-property overrides applied on top of whatever the ``.dat`` says.
            Keys are ``POR``, ``PERMI``, ``PERMJ``, ``PERMK``, ``NTG``, ``NULL``; values
            are either a number (constant) or a path to an ``.inc`` file.
        diagonal_connections: Also connect the in-layer diagonal neighbours.
    """

    cell_size: tuple[float, float, float] | Literal["corners"] = (100.0, 100.0, 5.0)
    swi: float = 0.0
    overrides: dict[str, Any] = field(default_factory=dict)
    diagonal_connections: bool = False

    def __post_init__(self) -> None:
        if self.cell_size != "corners":
            self.cell_size = tuple(float(v) for v in self.cell_size)  # type: ignore[assignment]
            if len(self.cell_size) != 3 or any(v <= 0 for v in self.cell_size):
                raise ValueError(
                    f"cell_size must be three positive numbers or 'corners', "
                    f"got {self.cell_size!r}."
                )
        if not 0.0 <= self.swi < 1.0:
            raise ValueError(f"swi must be in [0, 1), got {self.swi}.")


# MARK: Quality index
@dataclass
class IndexConfig:
    """Parameters of the discounted volume index (stage 4).

    Attributes:
        delta_x: Bin width of the distance/volume table, in equivalent metres.
        rate: Discount rate per ``x_base`` of equivalent distance, as a fraction.
        x_base: Equivalent distance over which one full discount period elapses.
        mode: ``'binned'`` follows the specification literally; ``'exact'`` discounts
            every cell individually.
        limit: Optional shortest-path cutoff in equivalent metres.
        k_layer: 1-based layer shown in the diagnostic maps.
    """

    delta_x: float = 50.0
    rate: float = 0.10
    x_base: float = 500.0
    mode: Literal["binned", "exact"] = "binned"
    limit: float | None = None
    k_layer: int = 1

    def __post_init__(self) -> None:
        if self.delta_x <= 0:
            raise ValueError(f"delta_x must be positive, got {self.delta_x}.")
        if self.rate < 0:
            raise ValueError(f"rate must be non-negative, got {self.rate}.")
        if self.x_base <= 0:
            raise ValueError(f"x_base must be positive, got {self.x_base}.")
        if self.mode not in ("binned", "exact"):
            raise ValueError(f"mode must be 'binned' or 'exact', got {self.mode!r}.")
        if self.limit is not None and self.limit <= 0:
            raise ValueError(f"limit must be positive or None, got {self.limit}.")
        if self.k_layer < 1:
            raise ValueError(f"k_layer is 1-based, got {self.k_layer}.")


# MARK: Optimisation
@dataclass
class SAConfig:
    """Simulated annealing and objective-function parameters (stage 5, producers only).

    Attributes:
        n_wells: Number of producer wells to place.
        n_injectors: How many of the ``n_wells`` will become injectors (roles are assigned after
            the annealing). Only used to check that every flow region can get a producer and an
            injector; ``0`` skips that check.
        k_min: Minimum cell permeability, in mD, for a well to be valid there
            (``sqrt(PERMI * PERMJ)``). During annealing a well below it scores ``Y_at = 0``;
            at the end the layout is rejected if any well sits on such a cell. Cells below
            it also act as barriers: with more than one well, a well that cannot reach
            another well through I/J-neighbour cells at or above ``k_min`` (no diagonals)
            is isolated, scores ``Y_at = 0`` and invalidates the layout. ``0`` disables.
        score_min_frac: A layout is rejected when any well's final ``Y_at`` is below this
            fraction of the mean ``Y_at`` of the **other** wells (``0.2``: below 20%).
            Not checked with a single well. ``0`` disables.
        max_attempts: Annealing runs (seeds ``seed``, ``seed + 1``, ...) tried before the
            layout is given up on.
        d_min: Minimum distance between wells, in metres (see ``metric``).
        metric: ``'physical'`` measures well spacing as the euclidean distance between
            cell centres; ``'equivalent'`` uses the shortest-path equivalent distance.
        w_bal: Weight of the imbalance penalty. The penalty is
            ``w_bal * F * CV(Y_at per well)``, so the weight is dimensionless.
        w_dist: Weight of the minimum-distance penalty,
            ``w_dist * F * sum(max(0, (d_min - d)/d_min)**2)``.
        n_iter: Number of annealing iterations.
        t_end_ratio: Final temperature as a fraction of the auto-calibrated ``T0``.
        burn_in: Number of random states used to calibrate ``T0``.
        jump_prob: Probability of proposing a uniform random relocation instead of a
            local move.
        polish: Run a deterministic 1-opt sweep after annealing.
        polish_radius: Radius in metres searched around each well during the polish.
            ``None`` picks three cell widths.
        seed: Random seed. Governs every stochastic decision, so runs are reproducible.
    """

    n_wells: int = 1
    n_injectors: int = 0
    k_min: float = 0.0
    score_min_frac: float = 0.0
    max_attempts: int = 3
    d_min: float = 500.0
    metric: Literal["physical", "equivalent"] = "physical"
    w_bal: float = 0.25
    w_dist: float = 1.0
    n_iter: int = 2000
    t_end_ratio: float = 1e-3
    burn_in: int = 50
    jump_prob: float = 0.1
    polish: bool = True
    polish_radius: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_wells < 1:
            raise ValueError(f"n_wells must be >= 1, got {self.n_wells}.")
        if self.k_min < 0:
            raise ValueError(f"k_min must be non-negative, got {self.k_min}.")
        if self.score_min_frac < 0:
            raise ValueError(
                f"score_min_frac must be non-negative, got {self.score_min_frac}."
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}.")
        if self.d_min < 0:
            raise ValueError(f"d_min must be non-negative, got {self.d_min}.")
        if self.metric not in ("physical", "equivalent"):
            raise ValueError(
                f"metric must be 'physical' or 'equivalent', got {self.metric!r}."
            )
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}.")
        if not 0.0 <= self.jump_prob <= 1.0:
            raise ValueError(f"jump_prob must be in [0, 1], got {self.jump_prob}.")


# MARK: Run
@dataclass
class RunConfig:
    """Everything needed to run the full pipeline on one model.

    Attributes:
        dat_path: Path to the CMG ``.dat`` file.
        grid: Grid geometry and property overrides.
        index: Quality-index parameters.
        sa: Optimisation parameters.
        cache_dir: Directory for the volume/distance histogram cache. ``None`` disables
            caching.
        verbose: Verbosity of every stage.
        label: Optional short name for this run, used in output file names.
    """

    dat_path: Path
    grid: GridConfig = field(default_factory=GridConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    sa: SAConfig = field(default_factory=SAConfig)
    cache_dir: Path | None = None
    verbose: bool = True
    label: str | None = None

    def __post_init__(self) -> None:
        self.dat_path = Path(self.dat_path)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
        if self.label is None:
            self.label = self.dat_path.stem

    # MARK: YAML
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Build a ``RunConfig`` from a plain dictionary (e.g. parsed YAML)."""
        data = dict(data)
        nested = {"grid": GridConfig, "index": IndexConfig, "sa": SAConfig}
        for key, klass in nested.items():
            value = data.get(key)
            if isinstance(value, dict):
                data[key] = klass(**value)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown configuration keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}."
            )
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """Read a ``RunConfig`` from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-friendly dictionary (paths become strings)."""

        def _clean(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items()}
            return value

        return {key: _clean(value) for key, value in asdict(self).items()}
