"""Download, read and write SPE10 permeability data (CMG IMEX and OPM Flow include formats).

Grid arrays are indexed [i, j] (x, y) for a patch and [x, y, k] for the full model.
Files are written with i fastest, then j, which is the order both IMEX and OPM expect.
"""
import itertools
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

NX, NY, NZ = 60, 220, 85
N_CELLS = NX * NY * NZ

SPE_ZIP_URL = "https://www.spe.org/web/csp/datasets/por_perm_case2a.zip"
SPE_ZIP_MEMBER = "spe_perm.dat"  # three blocks: Kx, Ky, Kz
OPM_URL = "https://raw.githubusercontent.com/OPM/opm-data/master/spe10model2/SPE10MODEL2_PERM.INC"


# ---------------------------------------------------------------- download
def _download(url, dest, timeout=60):
    dest = Path(dest)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "spe10-dataset-builder"})
    with urllib.request.urlopen(req, timeout=timeout) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(dest)


def fetch_spe10_perm(cache_dir):
    """Return the path of the SPE10 permeability file, downloading it into cache_dir if needed.

    Tries the SPE.org zip first (only the permeability member is extracted), then the OPM data
    repository. Raises RuntimeError if both fail.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for cached in (cache_dir / SPE_ZIP_MEMBER, cache_dir / "SPE10MODEL2_PERM.INC"):
        if cached.exists():
            return cached

    errors = []
    try:
        zip_path = cache_dir / "por_perm_case2a.zip"
        print(f"Downloading {SPE_ZIP_URL}")
        _download(SPE_ZIP_URL, zip_path)
        target = cache_dir / SPE_ZIP_MEMBER
        with zipfile.ZipFile(zip_path) as zf, zf.open(zf.getinfo(SPE_ZIP_MEMBER)) as src, \
                target.with_name(target.name + ".part").open("wb") as dst:
            shutil.copyfileobj(src, dst)
        target.with_name(target.name + ".part").rename(target)
        return target
    except Exception as e:  # noqa: BLE001 - report every failure reason to the user
        errors.append(f"{SPE_ZIP_URL}: {e}")

    try:
        target = cache_dir / "SPE10MODEL2_PERM.INC"
        print(f"Downloading {OPM_URL}")
        _download(OPM_URL, target)
        return target
    except Exception as e:  # noqa: BLE001
        errors.append(f"{OPM_URL}: {e}")

    raise RuntimeError("Could not download the SPE10 permeability file:\n  " + "\n  ".join(errors)
                       + "\nDownload it manually and pass it with --perm-file.")


# ---------------------------------------------------------------- reading
def parse_values(path):
    """All numeric values of an IMEX/ECLIPSE include file as a flat float array.

    Skips '**' and '--' comments, keywords (alphabetic tokens) and '/' terminators, and expands
    'n*value' repeats.
    """
    vals = []
    with Path(path).open() as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("**", "--")):
                continue
            toks = s.split("--")[0].split("**")[0].split()
            try:
                row = list(map(float, toks))
            except ValueError:
                for t in toks:
                    if "*" in t:
                        n, v = t.split("*")
                        vals.extend([float(v)] * int(n))
                    elif t == "/" or t.isalpha():
                        continue
                    else:
                        vals.append(float(t))
            else:
                vals.extend(row)
    return np.array(vals)


def load_kx(path):
    """Read the SPE10 Kx field (md) as an array K[x, y, k] of shape (60, 220, 85).

    Accepts a single block (PERMI.inc style, exactly 1,122,000 values) or the original three-block
    file (Kx, Ky, Kz), in which case only the first block is used. Returns (K, note).
    """
    vals = parse_values(path)
    note = ""
    if vals.size == 3 * N_CELLS:
        vals = vals[:N_CELLS]
        note = "three-block file (Kx, Ky, Kz): first block used as Kx"
    if vals.size != N_CELLS:
        raise ValueError(f"{path}: found {vals.size} values, expected {N_CELLS} (or {3 * N_CELLS})")
    if not (np.isfinite(vals).all() and (vals > 0).all()):
        raise ValueError(f"{path}: non-positive or non-finite permeability found")
    return np.ascontiguousarray(vals.reshape(NZ, NY, NX).transpose(2, 1, 0)), note


# ---------------------------------------------------------------- writing
def _compact(values):
    """Values as strings, using 'n*value' for runs of equal values."""
    out = []
    for v, grp in itertools.groupby(values):
        n = len(list(grp))
        out.append(f"{v}" if n == 1 else f"{n}*{v}")
    return out


def _wrap(tokens, max_line=80):
    lines, cur = [], ""
    for t in tokens:
        if cur and len(cur) + 1 + len(t) > max_line:
            lines.append(cur)
            cur = t
        else:
            cur = f"{cur} {t}" if cur else t
    if cur:
        lines.append(cur)
    return lines


def write_patch(path, kx_patch, patch_info, fmt="imex"):
    """Write one patch (kx_patch[i, j] or kx_patch[i, j, k], md) as an include file with Ky = Kx.

    imex: bare value list, meant for `PERMI ALL` / `PERMJ ALL` followed by `INCLUDE`.
    opm:  PERMX block, then COPY PERMX -> PERMY; meant to be INCLUDEd in the GRID section.
    """
    if fmt not in ("imex", "opm"):
        raise ValueError(f"unknown format {fmt!r}")
    ni, nj, nk = kx_patch.shape if kx_patch.ndim == 3 else (*kx_patch.shape, 1)
    layers = patch_info["layer"] if nk == 1 else f"{patch_info['layer']}-{patch_info['layer_end']}"
    cmt = "**" if fmt == "imex" else "--"
    header = [
        f"{cmt} SPE10 Model 2 permeability patch (Kx = Ky), md",
        f"{cmt} family_id={patch_info['family_id']} set={patch_info['set']} layer={layers}"
        f" x_start={patch_info['x_start']} y_start={patch_info['y_start']} rotation={patch_info['rotation']}",
        f"{cmt} grid (ni, nj, nk) = ({ni}, {nj}, {nk}), i fastest, then j, then k",
    ]
    body = _wrap(_compact(kx_patch.T.ravel().tolist()))
    if fmt == "opm":
        body = ["PERMX"] + body + ["/", "", "COPY", "  PERMX PERMY /", "/"]
    Path(path).write_text("\n".join(header + body) + "\n")
