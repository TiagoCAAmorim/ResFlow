"""Permeability map (paper Fig. 5) and state maps true / proxy / error (paper Figs. 6-7).

Orientation as in the paper: ``imshow(origin='lower')``, columns = i, rows = j, cells of `cell` metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from . import style as fs
from .io import KGF_CM2_TO_MPA, Rollout
from .wells import CELL, draw_wells


@dataclass
class MapVar:
    """One mapped variable: column of the npz, scale to plotting units, colour map, labels, tick formats."""
    var: str
    scale: float
    cmap: str
    cblabel: str
    errlabel: str
    fmt: str
    efmt: str
    short: tuple = ()   # (colour-bar label, error label) used when the panels are too low for the long labels


PRESSURE = MapVar("P", KGF_CM2_TO_MPA, "viridis", "Pressure (MPa)", "Pressure error (MPa)", "%.0f", "%.2f", ("P (MPa)", "P error (MPa)"))
SATURATION = MapVar("Sw", 1.0, "viridis", "Water saturation (-)", "Saturation error (-)", "%.2f", "%.3f", ("Sw (-)", "Sw error (-)"))


def cells_to_grid(v, ijk):
    """Cell values [cells] -> array [nj, ni] (rows = j, columns = i)."""
    ni, nj = int(ijk[:, 0].max()), int(ijk[:, 1].max())
    g = np.full((nj, ni), np.nan)
    g[ijk[:, 1] - 1, ijk[:, 0] - 1] = v
    return g


def _extent(ijk, cell):
    return [0, ijk[:, 0].max() * cell, 0, ijk[:, 1].max() * cell]


def _ticks(ext):
    step = 500 if ext <= 2000 else 1000 if ext <= 3500 else 2000
    return list(np.arange(0, ext + 1e-6, step))


def _axlabels(ax, ext, left=True, bottom=True):
    ax.set_xticks(_ticks(ext[1]))
    ax.set_yticks(_ticks(ext[3]))
    if bottom:
        ax.set_xlabel("x (m)")
    else:
        ax.tick_params(labelbottom=False)
    if left:
        ax.set_ylabel("y (m)")
    else:
        ax.tick_params(labelleft=False)


def plot_permeability(kx, ijk, markers, cell=CELL):
    """Permeability (md, log colour scale) with labelled wells. markers = (ij, type, name) arrays."""
    ext = _extent(ijk, cell)
    perm = cells_to_grid(kx, ijk)
    aspect = ext[3] / ext[1]
    fig, ax = plt.subplots(figsize=(fs.COL1, 0.9 + 2.0 * min(1.0, aspect)), constrained_layout=True)
    im = ax.imshow(perm, origin="lower", extent=ext, cmap="viridis", norm=LogNorm(np.nanmin(perm), np.nanmax(perm)))
    _axlabels(ax, ext)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("Permeability (md)")
    ax.legend(handles=[Line2D([], [], marker="^", ls="", mfc="white", mec="black", label="Producer"),
                       Line2D([], [], marker="v", ls="", mfc="black", mec="black", label="Injector")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18 / min(1.0, aspect)), ncol=2, frameon=False, handletextpad=0.1)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_wells(ax, *markers, cell=cell)
    return fig


def state_fields(ro: Rollout, specs=(PRESSURE, SATURATION)):
    """[(spec, true[days, cells], pred[days, cells])] in plotting units (pressure in MPa) from a Rollout."""
    out = []
    for s in specs:
        v = ro.var(s.var)
        out.append((s, ro.true[:, :, v] * s.scale, ro.pred[:, :, v] * s.scale))
    return out


def plot_state_maps(fields, days, ijk, markers, cell=CELL, row_height=None):
    """True / proxy / error maps, one row per (variable, day).

    fields: [(MapVar, true[len(days), cells], pred[len(days), cells])], values already in plotting units.
    The colour scale is shared by true and proxy in a row, the error scale is symmetric per row.
    Letters (a), (b), ... run over the panels. Wells are marked without names.
    """
    ext = _extent(ijk, cell)
    nd, nv = len(days), len(fields)
    if row_height is None:   # 1.55 in for square grids (paper); shorter rows for wide grids (panel width is 1.42 in)
        row_height = 1.55 - (1.0 - min(1.0, ext[3] / ext[1])) * 1.42
    # grid rows: nd rows per variable, a thin spacer row between variables (room for the column titles)
    compact = ext[3] / ext[1] < 0.75
    sp = 0.5 / row_height   # spacer row: 0.5 in
    ratios, pos = [], []
    for v in range(nv):
        if v:
            ratios.append(sp)
        for r in range(nd):
            pos.append(len(ratios))
            ratios.append(1.0)
    nrows_fig = nd * nv
    H = row_height * nrows_fig + 0.6 + (nv - 1) * 0.5
    W = [1.4, 0.2, 1.4, 0.8, 1.4, 0.8]
    fig = plt.figure(figsize=(fs.COL2, H))
    gs = fig.add_gridspec(len(ratios), 6, width_ratios=W, height_ratios=ratios, wspace=0.0,
                          hspace=0.12,
                          left=0.55 / fs.COL2, right=1 - 0.05 / fs.COL2, top=0.95, bottom=0.55 / H)
    late, k = [], 0
    for v, (spec, tt, pp) in enumerate(fields):
        for r, d in enumerate(days):
            gr = pos[v * nd + r]
            t, p = tt[r], pp[r]
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            e = p - t
            em = np.abs(e).max()
            a0, a1, a2 = (fig.add_subplot(gs[gr, c]) for c in (0, 2, 4))
            c1, c2 = a1.inset_axes([1.04, 0, 0.05, 1]), a2.inset_axes([1.04, 0, 0.05, 1])
            i0 = a0.imshow(cells_to_grid(t, ijk), origin="lower", extent=ext, cmap=spec.cmap, vmin=lo, vmax=hi)
            a1.imshow(cells_to_grid(p, ijk), origin="lower", extent=ext, cmap=spec.cmap, vmin=lo, vmax=hi)
            i2 = a2.imshow(cells_to_grid(e, ijk), origin="lower", extent=ext, cmap="RdBu_r", vmin=-em, vmax=em)
            last = (v == nv - 1) and (r == nd - 1)
            for c, a in enumerate((a0, a1, a2)):
                _axlabels(a, ext, left=(c == 0), bottom=(r == nd - 1))
                a.text(0.03, 0.97, f"({chr(97 + k)})", transform=a.transAxes, fontsize=7.5, fontweight="bold",
                       va="top", ha="left", color="white" if c < 2 else "black",
                       path_effects=[pe.withStroke(linewidth=1.2, foreground="black" if c < 2 else "white")])
                k += 1
                late.append(a)
            a0.set_ylabel(f"Day {int(d)}\ny (m)")
            cb = fig.colorbar(i0, cax=c1)
            cb.set_label(spec.short[0] if compact and spec.short else spec.cblabel)
            cb.ax.yaxis.set_major_formatter(mt.FormatStrFormatter(spec.fmt))
            cb2 = fig.colorbar(i2, cax=c2)
            cb2.set_label(spec.short[1] if compact and spec.short else spec.errlabel)
            cb2.ax.yaxis.set_major_formatter(mt.FormatStrFormatter(spec.efmt))
            if r == 0:
                for a, title in zip((a0, a1, a2), ("True", "Proxy", "Error (proxy $-$ true)")):
                    a.set_title(title)
    fig.canvas.draw()   # final layout before drawing markers
    for a in late:
        draw_wells(a, *markers, cell=cell, label=False, ms=3.5)
    return fig
