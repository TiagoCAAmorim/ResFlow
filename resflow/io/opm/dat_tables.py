"""Static data that OPM does not output, read from the IMEX ``.dat`` the OPM deck was made from.

PVT (``pvt01.pth``), relative permeability (``krel01.{wo,go}.pth``), well order and BHP depths.
"""
import re
from pathlib import Path

import numpy as np
import torch

from ..cmg.datreader import dat_parser, dat_pvt, dat_run


def _strip(text):
    return "\n".join(line.split("**")[0] for line in text.splitlines())


def _full_text(path, _seen=None):
    """Text of a .dat with its INCLUDE files expanded (comments removed)."""
    path = Path(path).resolve()
    out = []
    for line in _strip(path.read_text(errors="replace")).splitlines():
        m = re.match(r"\s*INCLUDE\s+'([^']+)'", line, re.I)
        if m and (path.parent / m.group(1)).is_file():
            out.append(_full_text(path.parent / m.group(1)))
        else:
            out.append(line)
    return "\n".join(out)


def read_pvt(dat_path):
    """List of PVT dictionaries (one per table), as the IMEX converter reads them."""
    parser = dat_parser.DatParser(
        ignore=["ROCKFLUID", "INITIAL", "NUMERICAL", "RUN", "GRID_keys", "KREL_keys", "VFP_keys"])
    parser.process(Path(dat_path))
    if "TITLE1" not in parser.get() or "GRID" not in parser.get():
        raise ValueError(f"TITLE1 or GRID section not found in {dat_path}")
    return dat_pvt.get_from_dat_data(parser.get())


def pvt_to_tensors(data):
    """dict -> {name: tensor}; float64 becomes float32, non-array items are dropped."""
    import pandas as pd
    out = {}
    for k, v in data.items():
        if isinstance(v, pd.DataFrame):
            arr = v.values
        elif isinstance(v, np.ndarray):
            arr = v
        elif isinstance(v, (float, int)):
            arr = np.array(v)
        else:
            continue
        t = torch.from_numpy(arr)
        out[k] = t.float() if t.dtype == torch.float64 else t
    return out


def read_krel(dat_path):
    """Relative permeability tables of the .dat: list of (wo [n,3]: Sw Krw Krow, go [n,3]: Sg Krg Krog) arrays."""
    text = _full_text(dat_path)

    def blocks(name):
        res = []
        for m in re.finditer(rf"^\s*{name}\s*$(.*?)(?=^\s*(?:SLT|SWT|LIST|[A-Z][A-Z0-9_-]+\s*(?:[A-Z]+\s*)?$)|\Z)",
                             text, re.I | re.M | re.S):
            rows = []
            for l in m.group(1).splitlines():
                tok = l.split()
                if tok and all(re.fullmatch(r"[-+0-9.eE]+", t) for t in tok):
                    rows.append([float(x) for x in tok])
            if rows:
                res.append(np.array(rows))
        return res

    swt, slt = blocks("SWT"), blocks("SLT")
    if len(swt) != 1 or len(slt) != 1:
        raise NotImplementedError(f"Expected one SWT and one SLT table, found {len(swt)} and {len(slt)}")
    return swt[0], slt[0]


def krel_tensors(swt, slt):
    """Same processing as the IMEX converter: sort by saturation and add the (0,0,max) / (1,1,0) end points."""
    wo = torch.tensor(np.stack([swt[:, 0], swt[:, 1], swt[:, 2]]).T, dtype=torch.float32)
    go = torch.tensor(np.stack([1 - slt[:, 0], slt[:, 1], slt[:, 2]]).T, dtype=torch.float32)

    def fix(t):
        t = t[torch.argsort(t[:, 0])]
        if t[:, 0].min() > 0:
            t = torch.cat([torch.tensor([[0, 0, t[:, -1].max()]]), t])
        if t[:, 0].max() < 1:
            t = torch.cat([t, torch.tensor([[1, 1, 0]])])
        return t

    return fix(wo), fix(go)


def well_order(dat_path):
    """Well names in the order of their definition in the .dat (``WELL 'name' ...``)."""
    names = []
    for m in re.finditer(r"^\s*WELL\s+'([^']+)'", _strip(Path(dat_path).read_text(errors="replace")), re.I | re.M):
        if m.group(1) not in names:
            names.append(m.group(1))
    return names


def bhp_depths(dat_path):
    """{well: BHPDEPTH} from the .dat; raises if a well has none or several different values."""
    parser = dat_parser.DatParser(ignore=["TITLE1", "GRID", "ROCKFLUID", "INITIAL", "NUMERICAL",
                                          "GRID_keys", "KREL_keys", "VFP_keys", "FLUID_keys"])
    parser.process(Path(dat_path))
    rows = dat_run.get_well_key(parser.get(), keyword="BHPDEPTH", verbose=False)
    depths = {}
    for _, well, depth in rows:
        depths.setdefault(well, set()).add(depth)
    bad = {w: d for w, d in depths.items() if len(d) > 1}
    if bad:
        raise ValueError(f"Several BHPDEPTH values per well are not supported: {bad}")
    return {w: next(iter(d)) for w, d in depths.items()}


def bhp_targets(dat_path):
    """BHP targets of the schedule: ``{well: [(day, kgf/cm2)]}`` (day since the first ``DATE``), from ``TARGET BHP``.

    Used to report the BHP of a well that OPM shut (no crossflow): IMEX keeps the target as its BHP.
    """
    from datetime import date
    lines = _strip(Path(dat_path).read_text(errors="replace")).splitlines()
    t0, now, out, i = None, 0.0, {}, 0
    while i < len(lines):
        s = lines[i].strip()
        m = re.match(r"DATE\s+(\d+)\s+(\d+)\s+(\d+)", s, re.I)
        if m:
            d = date(*map(int, m.groups()))
            t0 = t0 or d
            now = float((d - t0).days)
        elif re.match(r"TARGET\s+BHP\b", s, re.I):
            names = re.findall(r"'([^']+)'", s)
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            vals = [float(v) for v in lines[j].split()]
            vals = vals * len(names) if len(vals) == 1 else vals
            for n, v in zip(names, vals):
                out.setdefault(n, []).append((now, v))
            i = j
        i += 1
    return out


def target_at(targets, times):
    """BHP target in force at each output time (NaN before the first target). A target set at day d applies to the
    step that ends after d, so the output at day d still shows the previous one."""
    out = np.full(len(times), np.nan)
    for k, (day, v) in enumerate(targets):
        end = targets[k + 1][0] if k + 1 < len(targets) else np.inf
        t = np.asarray(times)
        out[(t > day + 1e-9) & (t <= end + 1e-9)] = v
    return out
