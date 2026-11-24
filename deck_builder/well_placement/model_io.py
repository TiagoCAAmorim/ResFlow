"""Stage 1 -- read a CMG ``.dat`` model: grid size and rock properties.

The parser only looks at what the placement algorithm needs: the ``CORNER``/``CART``
grid dimensions and the ``POR``, ``PERMI``, ``PERMJ``, ``PERMK``, ``NETGROSS`` and
``NULL`` arrays. Each property may be given as a constant (``POR CON 0.10``), as an
include file (``PERMI ALL`` followed by ``INCLUDE '...'``), as inline values, or as an
alias of another array (``PERMJ EQUALSI``). Any of them can also be overridden from the
configuration, so a model can be re-run with, say, a different porosity without editing
the ``.dat``.

Include paths are resolved **relative to the ``.dat`` file**, matching how the simulator
reads them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import GridConfig, get_logger
from .geometry import CellGeometry, build_regular_geometry, read_corners_geometry

#: Property keywords understood by the parser, mapped to their canonical name.
PROPERTY_KEYWORDS = {
    "POR": "POR",
    "PERMI": "PERMI",
    "PERMJ": "PERMJ",
    "PERMK": "PERMK",
    "NETGROSS": "NTG",
    "NTG": "NTG",
    "NULL": "NULL",
}

#: Keywords that end an inline value block without being properties themselves.
_BLOCK_TERMINATORS = {
    "GRID", "ROCKFLUID", "INITIAL", "NUMERICAL", "RUN", "MODEL", "CORNER", "CART",
    "CORNERS", "COORD", "ZCORN", "PINCHOUTARRAY", "CPOR", "PRPOR", "PVCUTOFF",
    "RTYPE", "PTYPE", "ITYPE", "EOSTYPE", "TRANSF", "DIFRAC", "DJFRAC", "DKFRAC",
}

_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eEdD][+-]?\d+)?$")


# MARK: Property specs
@dataclass
class PropertySpec:
    """How one grid property is defined in the ``.dat`` file.

    Attributes:
        kind: One of ``'const'``, ``'file'``, ``'inline'`` or ``'equalsi'``.
        value: The constant, for ``kind='const'``.
        path: The include path, for ``kind='file'``.
        values: The inline values, for ``kind='inline'``.
        source: Human-readable origin, used in log messages.
    """

    kind: str
    value: float | None = None
    path: Path | None = None
    values: np.ndarray | None = None
    source: str = ""

    def describe(self) -> str:
        """Return a short description for logging."""
        if self.kind == "const":
            return f"constant {self.value:g}"
        if self.kind == "file":
            return f"file {self.path.name}" if self.path else "file"
        if self.kind == "inline":
            n = 0 if self.values is None else self.values.size
            return f"{n} inline values"
        return "copy of PERMI"


@dataclass
class DatModel:
    """Raw contents of a ``.dat`` file, before properties are resolved to arrays.

    Attributes:
        dat_path: Path to the parsed file.
        shape: ``(ni, nj, nk)``.
        specs: Property name -> :class:`PropertySpec`.
        corners_path: Path to the ``CORNERS`` include, when the grid is corner-point.
    """

    dat_path: Path
    shape: tuple[int, int, int]
    specs: dict[str, PropertySpec] = field(default_factory=dict)
    corners_path: Path | None = None


@dataclass
class ReservoirModel:
    """A fully resolved model: property arrays plus geometry.

    Property arrays are ``(nk, nj, ni)``; use ``.ravel()`` to get the linear ordering
    used everywhere else in the package.

    Attributes:
        dat_path: Path of the source ``.dat``.
        geometry: Cell geometry in metres.
        por: Porosity, fraction.
        permi: Permeability along I, mD.
        permj: Permeability along J, mD.
        permk: Permeability along K, mD.
        ntg: Net-to-gross, fraction.
        null: Active flag as read from the model (1 = active).
        swi: Initial water saturation used for the oil-volume factor.
        label: Short name for the model.
    """

    dat_path: Path
    geometry: CellGeometry
    por: np.ndarray
    permi: np.ndarray
    permj: np.ndarray
    permk: np.ndarray
    ntg: np.ndarray
    null: np.ndarray
    swi: float = 0.0
    label: str = ""

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(ni, nj, nk)``."""
        return self.geometry.shape

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self.geometry.n_cells

    def flat(self, name: str) -> np.ndarray:
        """Return a property as a flat ``(ncell,)`` array in linear cell order.

        Args:
            name: One of ``por``, ``permi``, ``permj``, ``permk``, ``ntg``, ``null``.

        Returns:
            The flattened array.
        """
        return getattr(self, name).ravel()

    def permeability(self, direction: int) -> np.ndarray:
        """Return the flat permeability array for a connection direction.

        Args:
            direction: ``0`` for I, ``1`` for J, ``2`` for K, ``3`` for the in-layer
                diagonal.

        Returns:
            ``(ncell,)`` permeability in mD. The diagonal gets the **harmonic** mean of
            ``PERMI`` and ``PERMJ``: resistance to flow along a unit direction ``n`` goes
            as ``1/(n . K^-1 n)``, which at 45 degrees is exactly that mean. Cells where
            either component is zero come out as zero, i.e. still impermeable.
        """
        if direction == 3:
            ki, kj = self.permi.ravel(), self.permj.ravel()
            total = ki + kj
            return np.divide(
                2.0 * ki * kj, total, out=np.zeros_like(total), where=total > 0
            )
        return (self.permi, self.permj, self.permk)[direction].ravel()


# MARK: .inc reading
def read_inc_values(path: str | Path) -> np.ndarray:
    """Read the numeric values of a CMG include file.

    Handles the compact ``N*value`` notation, ``**`` comments and an optional leading
    keyword line.

    Args:
        path: Path to the include file.

    Returns:
        A flat ``float`` array of cell values, in ``.inc`` order (I fastest).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a token is not a number or a ``count*value`` group.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Include file not found: {path}")
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.split("**", 1)[0].split("--", 1)[0].strip()
            if not line:
                continue
            for token in line.split():
                if not (token[0].isdigit() or token[0] in "+-."):
                    continue  # keyword line, e.g. 'PERMI ALL'
                try:
                    if "*" in token:
                        count, value = token.split("*", 1)
                        values.extend([float(value)] * int(count))
                    else:
                        values.append(float(token))
                except ValueError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: cannot parse token {token!r}."
                    ) from exc
    if not values:
        raise ValueError(f"No numeric values found in {path}.")
    return np.asarray(values, dtype=float)


# MARK: .dat parsing
def parse_dat(dat_path: str | Path, verbose: bool = True) -> DatModel:
    """Parse the grid size and property definitions of a CMG ``.dat`` file.

    Args:
        dat_path: Path to the ``.dat`` file.
        verbose: Print what was found.

    Returns:
        The :class:`DatModel` with the grid shape and one :class:`PropertySpec` per
        property found.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If no ``CORNER``/``CART`` dimension line is found, or a property
            keyword is followed by an unsupported qualifier.
    """
    log = get_logger(verbose)
    dat_path = Path(dat_path)
    if not dat_path.is_file():
        raise FileNotFoundError(f"Model file not found: {dat_path}")

    lines = dat_path.read_text(encoding="utf-8", errors="replace").splitlines()
    shape: tuple[int, int, int] | None = None
    specs: dict[str, PropertySpec] = {}
    corners_path: Path | None = None
    pending: str | None = None  # property waiting for its INCLUDE or inline values
    inline: list[float] = []
    expect_corners_include = False

    def _flush_inline() -> None:
        nonlocal pending, inline
        if pending is not None and inline:
            specs[pending] = PropertySpec(
                kind="inline", values=np.asarray(inline, dtype=float), source=pending
            )
        pending, inline = None, []

    for lineno, raw in enumerate(lines, start=1):
        line = raw.split("**", 1)[0].split("--", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        keyword = tokens[0].lstrip("*").upper()

        # Values continuing an inline property block.
        if pending is not None and _NUMBER.match(tokens[0].split("*")[-1] or "x"):
            for token in tokens:
                if "*" in token:
                    count, value = token.split("*", 1)
                    inline.extend([float(value)] * int(count))
                else:
                    inline.append(float(token))
            continue

        if keyword in ("CORNER", "CART"):
            _flush_inline()
            if len(tokens) < 4:
                raise ValueError(
                    f"{dat_path}:{lineno}: expected '{keyword} ni nj nk', got {line!r}."
                )
            shape = tuple(int(v) for v in tokens[1:4])  # type: ignore[assignment]
            expect_corners_include = keyword == "CORNER"
            continue

        if keyword == "INCLUDE":
            include = _resolve_include(dat_path, line)
            if pending is not None:
                specs[pending] = PropertySpec(kind="file", path=include, source=pending)
                pending, inline = None, []
            elif expect_corners_include:
                corners_path = include
                expect_corners_include = False
            continue

        if keyword in PROPERTY_KEYWORDS:
            _flush_inline()
            expect_corners_include = False
            name = PROPERTY_KEYWORDS[keyword]
            qualifier = tokens[1].lstrip("*").upper() if len(tokens) > 1 else ""
            if qualifier == "CON":
                if len(tokens) < 3:
                    raise ValueError(
                        f"{dat_path}:{lineno}: '{keyword} CON' needs a value."
                    )
                specs[name] = PropertySpec(
                    kind="const", value=float(tokens[2]), source=name
                )
            elif qualifier == "EQUALSI":
                specs[name] = PropertySpec(kind="equalsi", source=name)
            elif qualifier in ("ALL", "IJK", ""):
                pending = name  # values arrive via INCLUDE or on the following lines
            else:
                raise ValueError(
                    f"{dat_path}:{lineno}: unsupported qualifier {qualifier!r} for "
                    f"{keyword}. Supported: CON, ALL, IJK, EQUALSI."
                )
            continue

        if keyword in _BLOCK_TERMINATORS or keyword.isalpha():
            _flush_inline()
            expect_corners_include = False

    _flush_inline()

    if shape is None:
        raise ValueError(
            f"No CORNER/CART grid dimensions found in {dat_path}. "
            "The parser needs a line like 'CORNER 32 32 1'."
        )

    log.info("Model %s: grid %dx%dx%d.", dat_path.name, *shape)
    for name, spec in specs.items():
        log.info("  %-6s <- %s", name, spec.describe())
    return DatModel(
        dat_path=dat_path, shape=shape, specs=specs, corners_path=corners_path
    )


def _resolve_include(dat_path: Path, line: str) -> Path:
    """Extract and resolve the path of an ``INCLUDE`` line, relative to the ``.dat``."""
    match = re.search(r"['\"](.+?)['\"]", line)
    raw = match.group(1) if match else line.split(maxsplit=1)[1].strip()
    include = Path(raw.replace("\\", "/"))
    return include if include.is_absolute() else (dat_path.parent / include).resolve()


# MARK: Resolution
def load_model(
    dat_path: str | Path,
    grid: GridConfig | None = None,
    *,
    label: str | None = None,
    verbose: bool = True,
) -> ReservoirModel:
    """Read a ``.dat`` file and resolve every property into an array.

    Args:
        dat_path: Path to the CMG ``.dat`` file.
        grid: Geometry and override settings. Defaults to a 100x100x5 m regular grid
            with no overrides and ``swi=0``.
        label: Short name for the model. Defaults to the file stem.
        verbose: Print progress messages.

    Returns:
        The resolved :class:`ReservoirModel`.

    Raises:
        ValueError: If a property array has the wrong number of values, or a required
            property is missing.
    """
    log = get_logger(verbose)
    grid = grid or GridConfig()
    dat = parse_dat(dat_path, verbose=verbose)
    ni, nj, nk = dat.shape
    n_cells = ni * nj * nk

    specs = dict(dat.specs)
    for name, override in (grid.overrides or {}).items():
        name = PROPERTY_KEYWORDS.get(name.upper(), name.upper())
        specs[name] = _spec_from_override(name, override, dat.dat_path)
        log.info("  %-6s <- override: %s", name, specs[name].describe())

    arrays: dict[str, np.ndarray] = {}
    for name in ("POR", "PERMI", "PERMJ", "PERMK", "NTG", "NULL"):
        arrays[name] = _resolve_spec(name, specs, n_cells, dat.shape, log)

    null = arrays["NULL"]
    active = (null != 0) & (arrays["POR"].ravel() > 0)
    inactive = int((~active).sum())
    if inactive:
        log.info("  %d of %d cells are inactive.", inactive, n_cells)

    if grid.cell_size == "corners":
        if dat.corners_path is None:
            raise ValueError(
                f"cell_size='corners' was requested but {dat.dat_path.name} has no "
                "CORNERS include to read the geometry from."
            )
        geometry = read_corners_geometry(
            dat.corners_path, dat.shape, active, verbose=verbose
        )
    else:
        geometry = build_regular_geometry(dat.shape, grid.cell_size, active)
        log.info("  Regular geometry: dI=%.1f m, dJ=%.1f m, dK=%.1f m.", *grid.cell_size)

    return ReservoirModel(
        dat_path=dat.dat_path,
        geometry=geometry,
        por=arrays["POR"].reshape(nk, nj, ni),
        permi=arrays["PERMI"].reshape(nk, nj, ni),
        permj=arrays["PERMJ"].reshape(nk, nj, ni),
        permk=arrays["PERMK"].reshape(nk, nj, ni),
        ntg=arrays["NTG"].reshape(nk, nj, ni),
        null=null.reshape(nk, nj, ni),
        swi=grid.swi,
        label=label or dat.dat_path.stem,
    )


def _spec_from_override(name: str, override: Any, dat_path: Path) -> PropertySpec:
    """Turn a configuration override into a :class:`PropertySpec`."""
    if isinstance(override, (int, float)):
        return PropertySpec(kind="const", value=float(override), source=f"{name} override")
    if isinstance(override, (str, Path)):
        path = Path(override)
        if not path.is_absolute():
            path = (dat_path.parent / path).resolve()
        return PropertySpec(kind="file", path=path, source=f"{name} override")
    raise ValueError(
        f"Override for {name} must be a number or a path, got {type(override).__name__}."
    )


def _resolve_spec(
    name: str,
    specs: dict[str, PropertySpec],
    n_cells: int,
    shape: tuple[int, int, int],
    log,
) -> np.ndarray:
    """Resolve one property into a flat ``(ncell,)`` array, applying defaults."""
    spec = specs.get(name)
    if spec is None:
        return _default_property(name, n_cells, shape, log)

    if spec.kind == "const":
        return np.full(n_cells, float(spec.value), dtype=float)
    if spec.kind == "equalsi":
        if "PERMI" not in specs:
            raise ValueError(f"{name} EQUALSI, but PERMI is not defined in the model.")
        return _resolve_spec("PERMI", specs, n_cells, shape, log)

    values = spec.values if spec.kind == "inline" else read_inc_values(spec.path)
    if values.size != n_cells:
        origin = spec.path.name if spec.path else "inline values"
        raise ValueError(
            f"{name}: {origin} holds {values.size} values but the "
            f"{shape[0]}x{shape[1]}x{shape[2]} grid needs {n_cells}."
        )
    return values.astype(float)


def _default_property(
    name: str, n_cells: int, shape: tuple[int, int, int], log
) -> np.ndarray:
    """Return the default array for a property missing from the model."""
    if name == "NTG":
        return np.ones(n_cells, dtype=float)
    if name == "NULL":
        return np.ones(n_cells, dtype=float)
    if name == "PERMK":
        if shape[2] > 1:
            log.warning(
                "PERMK is not defined and the grid has %d layers; assuming 0 mD, which "
                "makes the layers hydraulically isolated.",
                shape[2],
            )
        return np.zeros(n_cells, dtype=float)
    raise ValueError(
        f"Required property {name} is not defined in the model and has no default. "
        "Provide it in the .dat file or through GridConfig.overrides."
    )
