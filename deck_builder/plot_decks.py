"""Plot permeability (log colour scale) and well positions of cases, one PNG per case_id.

    python plot_decks.py decks_index.csv   L36_32x32_01 [more ...] [--out DIR]
    python plot_decks.py well_locations.csv L36_32x32_01_3P2I      [--patches DIR]

The input is the `decks_index.csv` of build_decks.py or the `well_locations.csv` of
generate_well_layouts.py (detected from the header). Each name is a case_id or the start of
one: `L36_32x32_01` plots every case of that patch (L36_32x32_01_3P2I, L36_32x32_01_4P2I, ...).
A case has the same map and wells under every BHP schedule, so decks_index.csv gives one
PNG per case, not per deck. A 3D case (nk > 1) gets one map per layer plus the mean over the layers, in one panel.
"""
import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from perm_io import parse_values  # noqa: E402


def read_deck_wells(path):
    """Return (perm_file, ni, nj, {well: (i, j)}) from an IMEX deck (1-based cells)."""
    text = path.read_text()
    ni, nj = map(int, re.search(r"^CORNER\s+(\d+)\s+(\d+)", text, re.M).groups())
    inc = re.search(r"^PERMI\s+ALL\s*\n\s*INCLUDE\s+'([^']+)'", text, re.M).group(1)
    wells = {m.group(1): (int(m.group(2)), int(m.group(3)))
             for m in re.finditer(r"PERF\s+GEOA\s+'([^']+)'\s*\n\s*(\d+)\s+(\d+)\s+\d+", text)}
    return path.parent / inc, ni, nj, wells


def cases_from_decks_index(index, prefixes):
    """{case_id: (perm_file, ni, nj, wells)}; the first deck of each case is read."""
    with index.open(newline="") as f:
        rows = list(csv.DictReader(f))
    first = {}
    for r in rows:
        first.setdefault(r["case_id"], r["deck"])
    cases = {}
    for case_id, deck in first.items():
        if case_id.startswith(prefixes):
            for sub in ("decks", "decks_test", "."):
                p = index.parent / sub / f"{deck}.dat"
                if p.exists():
                    cases[case_id] = read_deck_wells(p)
                    break
            else:
                raise FileNotFoundError(f"{deck}.dat not found under {index.parent}")
    return cases


def cases_from_well_locations(csv_path, prefixes, patches):
    """{case_id: (perm_file, ni, nj, wells)}; the patch file is <patches>/perm/<family_id>.inc."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    cases = {}
    for r in rows:
        if not r["case_id"].startswith(prefixes):
            continue
        ni, nj = map(int, re.search(r"_(\d+)x(\d+)_", r["family_id"]).groups())
        wells = {}
        for key, value in r.items():
            m = re.fullmatch(r"([PI]\d+)I", key)
            if m and int(value) > 0:
                wells[m.group(1)] = (int(value), int(r[m.group(1) + "J"]))
        cases[r["case_id"]] = (patches / "perm" / f"{r['family_id']}.inc", ni, nj, wells)
    return cases


def plot_case(case_id, perm_file, ni, nj, wells, out, max_cols=4):
    """One map per layer (nk = number of values / (ni * nj)) on a common log colour scale; wells on every
    layer, as they perforate all of them."""
    values = np.asarray(parse_values(perm_file), float)
    if values.size % (ni * nj):
        raise ValueError(f"{perm_file.name}: {values.size} values for a {ni}x{nj} grid")
    nk = values.size // (ni * nj)
    kx = values.reshape(nk, nj, ni)          # [k, j, i], i fastest
    maps = list(kx)
    titles = [f"k = {k + 1}" for k in range(nk)]
    if nk > 1:      # last panel: arithmetic mean over the layers, the map the well placement uses
        maps.append(kx.mean(axis=0))
        titles.append("mean of layers")
    npanel = len(maps)
    ncol = min(npanel, max_cols)
    nrow = -(-npanel // ncol)
    w, h = 6 * ncol, 6 * nj / ni * 0.8 * nrow + 1
    fig, axes = plt.subplots(nrow, ncol, figsize=(w if nk > 1 else 7, h), squeeze=False, sharex=True, sharey=True)
    norm = LogNorm(vmin=kx.min(), vmax=kx.max())
    for k, ax in enumerate(axes.ravel()):
        if k >= npanel:
            ax.axis("off")
            continue
        im = ax.imshow(maps[k], norm=norm, origin="lower", extent=(0.5, ni + 0.5, 0.5, nj + 0.5),
                       cmap="viridis", interpolation="nearest")
        for kind, marker, color in (("P", "v", "red"), ("I", "^", "white")):
            for name, (i, j) in wells.items():
                if name.startswith(kind):
                    ax.plot(i, j, marker, color=color, mec="black", ms=11)
                    ax.annotate(name, (i, j), xytext=(5, 5), textcoords="offset points",
                                color=color, fontsize=9, fontweight="bold")
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        if nk > 1:
            ax.set_title(titles[k])
    fig.suptitle(case_id)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Kx [md] (log scale)", shrink=0.8)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", type=Path, help="decks_index.csv or well_locations.csv")
    ap.add_argument("cases", nargs="+", help="case_id, or the start of case_ids (e.g. L36_32x32_01)")
    ap.add_argument("--out", type=Path, default=Path("deck_plots"), help="output folder (deck_plots)")
    ap.add_argument("--patches", type=Path,
                    help="well_locations.csv only: step 1 folder with perm/ (<csv folder>/../patches)")
    args = ap.parse_args()

    with args.table.open(newline="") as f:
        header = next(csv.reader(f))
    prefixes = tuple(args.cases)
    if "deck" in header:
        cases = cases_from_decks_index(args.table, prefixes)
    elif "P1I" in header:
        patches = args.patches or args.table.resolve().parent.parent / "patches"
        cases = cases_from_well_locations(args.table, prefixes, patches)
    else:
        ap.error(f"{args.table} is neither a decks_index.csv nor a well_locations.csv")
    if not cases:
        ap.error(f"no case_id starts with any of {', '.join(prefixes)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for case_id, (perm_file, ni, nj, wells) in cases.items():
        out = args.out / f"{case_id}.png"
        plot_case(case_id, perm_file, ni, nj, wells, out)
        print(out)


if __name__ == "__main__":
    main()
