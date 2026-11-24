"""KR.inc (UNISIM-IV 2024, 'Nominal.rar' -> KR_est_1.inc) and PVT.inc (UNISIM-II-D BO, sliced from the .dat)."""
import re
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

KR_URL = "https://www.unisim.cepetro.unicamp.br/benchmarks/files/UNISIM-IV-2024/Nominal.rar"
KR_MEMBER = "KR_est_1.inc"
PVT_URL = "https://www.unisim.cepetro.unicamp.br/benchmarks/files/UNISIM-II-D-BO_deterministic.ZIP"
PVT_MEMBER = "UNISIM-II-D_IM80/UNISIM-II-D-BO_IM80.dat"

def _strip_stars(text):
    """Drop the leading '*' of CMG keywords (the OPM converter expects plain keywords)."""
    return "\n".join(re.sub(r"(?<!\*)\*(?=[A-Za-z])", "", l) for l in text.splitlines()) + "\n"


_RAR_TOOLS = {
    "unrar": lambda rar, member, out: ["unrar", "x", "-y", str(rar), member, f"{out}/"],
    "unar": lambda rar, member, out: ["unar", "-f", "-o", str(out), str(rar), member],
    "7z": lambda rar, member, out: ["7z", "x", "-y", f"-o{out}", str(rar), member],
}


def _download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "spe10-dataset-builder"})
    with urllib.request.urlopen(req, timeout=60) as r, part.open("wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(dest)
    return dest


def fetch_fluid_sources(cache_dir):
    """Download the KR (UNISIM-IV Nominal.rar) and PVT (UNISIM-II-D zip) sources into cache_dir; skipped if present.

    Returns (rar_path, zip_path).
    """
    return (_download(KR_URL, Path(cache_dir) / "Nominal.rar"),
            _download(PVT_URL, Path(cache_dir) / "UNISIM-II-D-BO_deterministic.ZIP"))


def build_kr_inc(cache_dir, dest):
    """Download Nominal.rar (if not in cache_dir) and extract KR_est_1.inc to `dest`. Needs unrar, unar or 7z."""
    rar = _download(KR_URL, Path(cache_dir) / "Nominal.rar")
    tried = []
    for tool, cmd in _RAR_TOOLS.items():
        if shutil.which(tool) is None:
            tried.append(f"{tool}: not installed")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            res = subprocess.run(cmd(rar, KR_MEMBER, tmp), capture_output=True, text=True)
            found = next(Path(tmp).rglob(KR_MEMBER), None)
            if res.returncode == 0 and found:
                Path(dest).write_text(_strip_stars(found.read_text(errors="replace")))
                return Path(dest)
            tried.append(f"{tool}: exit {res.returncode}: {res.stderr.strip()[:200]}")
    raise RuntimeError(f"Could not extract {KR_MEMBER} from {rar}:\n  " + "\n  ".join(tried)
                       + f"\nInstall unrar (or unar / 7z) with your package manager, or extract it manually to {dest}.")


def build_pvt_inc(cache_dir, dest):
    """Download the UNISIM-II-D BO zip and write the PVT block (*MODEL *BLACKOIL .. *CVW) to `dest`.

    """
    zpath = _download(PVT_URL, Path(cache_dir) / "UNISIM-II-D-BO_deterministic.ZIP")
    with zipfile.ZipFile(zpath) as zf:
        lines = zf.read(zf.getinfo(PVT_MEMBER)).decode(errors="replace").splitlines()
    start = next((i for i, l in enumerate(lines) if "*MODEL" in l and "*BLACKOIL" in l), None)
    end = next((i for i in range(start or 0, len(lines)) if lines[i].lstrip().startswith("*CVW")), None)
    if start is None or end is None:
        raise RuntimeError(f"PVT block markers not found in {PVT_MEMBER}")
    Path(dest).write_text(_strip_stars("\n".join(lines[start:end + 1])))
    return Path(dest)
