"""Read the binary results of an OPM Flow run (``EGRID``, ``INIT``, ``UNRST``, ``SMSPEC``/``UNSMRY``).

Only what the converter needs. Files are read with ``resfo``. All values are returned as written by OPM
(ECLIPSE METRIC units: pressure in bar, volumes in sm3, transmissibility in cP.rm3/day/bar).
"""
from pathlib import Path

import numpy as np
import resfo

DARCY_METRIC = 0.00852702  # transmissibility (cP.rm3/day/bar) = Darcy constant x permeability (md) x geometry (m)
KGCM2_TO_BAR = 0.980665  # 1 kgf/cm2 = 0.980665 bar


def find_output_stem(case):
    """Return the path stem (no suffix) of the OPM results of ``case``.

    ``case`` may be the ``.DATA`` file, the case folder (results in ``<folder>/out`` or in the folder itself) or a
    results stem. The ``UNRST`` file defines the stem.
    """
    case = Path(case)
    if case.suffix.upper() == ".DATA":
        cands = [case.parent / "out" / case.stem, case.parent / case.stem]
        cands += [c.with_name(c.name.upper()) for c in cands]
    elif case.is_dir():
        cands = [p.with_suffix("") for d in (case / "out", case) for p in sorted(d.glob("*.UNRST"))]
    else:
        cands = [case.with_suffix("")]
    for c in cands:
        if c.with_suffix(".UNRST").is_file() and c.with_suffix(".SMSPEC").is_file():
            return c
    raise FileNotFoundError(f"No OPM results (.UNRST and .SMSPEC) found for {case}")


def _s(a):
    return [x.decode().strip() if isinstance(x, bytes) else str(x).strip() for x in a]


def read_grid(stem):
    """Grid and static cell data (active cells only, in OPM active-cell order)."""
    out = {}
    for kw in resfo.lazy_read(f"{stem}.EGRID"):
        name = kw.read_keyword().strip()
        if name in ("GRIDHEAD", "ACTNUM", "COORD", "ZCORN", "NNC1", "NNC2"):
            out[name] = np.asarray(kw.read_array())
    gh = out["GRIDHEAD"]
    nx, ny, nz = int(gh[1]), int(gh[2]), int(gh[3])
    actnum = out["ACTNUM"].astype(bool)
    init = {}
    for kw in resfo.lazy_read(f"{stem}.INIT"):
        name = kw.read_keyword().strip()
        if name in ("PORV", "DEPTH", "PORO", "PERMX", "PERMY", "PERMZ", "NTG", "TRANX", "TRANY", "TRANZ"):
            init[name] = np.asarray(kw.read_array(), dtype=np.float64)
    if "NNC1" in out and out["NNC1"].size:
        raise NotImplementedError("Non-neighbour connections (NNC) are not supported by the converter")
    return dict(nx=nx, ny=ny, nz=nz, actnum=actnum, coord=out["COORD"].astype(np.float64),
                zcorn=out["ZCORN"].astype(np.float64), init=init)


def read_restart(stem, keys=("PRESSURE", "SWAT", "SGAS", "RS")):
    """Read the unified restart file. Returns (times in days, {key: [n_times, n_active] float32})."""
    times, data = [], {k: [] for k in keys}
    cur = {}
    t = None
    for kw in resfo.lazy_read(f"{stem}.UNRST"):
        name = kw.read_keyword().strip()
        if name == "SEQNUM":
            cur = {}
            t = None
        elif name == "DOUBHEAD":
            t = float(np.asarray(kw.read_array())[0])
        elif name in keys:
            cur[name] = np.asarray(kw.read_array(), dtype=np.float32)
        elif name == "ENDSOL":
            if t is None or any(k not in cur for k in keys):
                raise ValueError(f"Restart block without time or without {keys}")
            times.append(t)
            for k in keys:
                data[k].append(cur[k])
    if not times:
        raise ValueError("No restart blocks found")
    return np.array(times), {k: np.stack(v) for k, v in data.items()}


class Summary:
    """Summary vectors: ``get(keyword, name, num)`` returns one value per summary entry; ``times`` in days."""

    def __init__(self, stem):
        spec = {k.strip(): v for k, v in resfo.read(f"{stem}.SMSPEC")}
        self.keywords = _s(spec["KEYWORDS"])
        self.names = _s(spec["WGNAMES"])
        self.nums = [int(x) for x in spec["NUMS"]]
        params = [v for k, v in resfo.read(f"{stem}.UNSMRY") if k.strip() == "PARAMS"]
        self.params = np.array(params, dtype=np.float64).reshape(len(params), len(self.keywords))
        self.times = self.params[:, self.keywords.index("TIME")]
        self.index = {(k, n, c): i for i, (k, n, c) in enumerate(zip(self.keywords, self.names, self.nums))}

    def has(self, keyword):
        return keyword in self.keywords

    def entries(self, keyword):
        """[(name, num)] of every vector of this keyword, in file order."""
        return [(n, c) for k, n, c in zip(self.keywords, self.names, self.nums) if k == keyword]

    def get(self, keyword, name, num=0):
        return self.params[:, self.index[(keyword, name, num)]]
