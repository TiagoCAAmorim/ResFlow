"""Shared matplotlib style of the paper figures (single column 8.5 cm, double column 17 cm)."""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL1, COL2 = 8.5 / 2.54, 17.0 / 2.54  # inches
# Okabe-Ito colour-blind safe palette
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7")


def setup(dpi: int = 300) -> None:
    """Set the rcParams of the paper figures. `dpi` is the default save resolution."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "lines.linewidth": 1.0, "savefig.dpi": dpi, "figure.dpi": 150, "mathtext.default": "regular",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def panel_letter(ax, letter, dx=-0.0, dy=1.02, **kw):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="right" if dx < 0 else "left", **kw)


def save(fig, path, dpi: int | None = None, optimize: bool = True) -> None:
    """Save a figure as PNG; `optimize` asks Pillow for an optimized (smaller) PNG."""
    kw = {"pil_kwargs": {"optimize": True}} if optimize and str(path).lower().endswith(".png") else {}
    fig.savefig(path, dpi=dpi, **kw)
