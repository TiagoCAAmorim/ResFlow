"""Well markers with name labels, and the cumulative-volume figure (paper Fig. 8)."""
from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from . import style as fs

CELL = 50.0  # m


def short_name(n) -> str:
    n = str(n)
    return n.replace("Producer ", "P").replace("Water Injector ", "I").replace("Injector ", "I")


def draw_wells(ax, marker_ij, marker_type, marker_name, cell=CELL, label=True, ms=4.5, fontsize=6, avoid=()):
    """Producers: white triangle up, black edge. Injectors: black triangle down, black edge.
    Labels: short names placed at the first candidate offset whose text box stays inside the axes and does not
    overlap any well marker, any other label or the artists in `avoid` (e.g. the panel letter)."""
    marker_ij = np.asarray(marker_ij, dtype=float)
    wx = (marker_ij[:, 0] - 0.5) * cell
    wy = (marker_ij[:, 1] - 0.5) * cell
    for x, y, t in zip(wx, wy, marker_type):
        if t == "producer":
            ax.plot(x, y, "^", ms=ms, mfc="white", mec="black", mew=0.7, zorder=5)
        else:
            ax.plot(x, y, "v", ms=ms, mfc="black", mec="black", mew=0.7, zorder=5)
    if not label:
        return
    fig = ax.figure
    r = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(r)
    pts = ax.transData.transform(np.c_[wx, wy])
    rad = ms * fig.dpi / 72 * 0.75
    obst = [Bbox([[px - rad, py - rad], [px + rad, py + rad]]) for px, py in pts]
    obst += [o.get_window_extent(r) for o in avoid]
    off = ms * 0.9 + 1.0   # points
    cands = [(off, 0, "left", "center"), (-off, 0, "right", "center"), (0, off, "center", "bottom"),
             (0, -off, "center", "top"), (off, off, "left", "bottom"), (-off, off, "right", "bottom"),
             (off, -off, "left", "top"), (-off, -off, "right", "top")]
    for x, y, n in zip(wx, wy, marker_name):
        best = None
        for dx, dy, ha, va in cands:
            tx = ax.annotate(short_name(n), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                             fontsize=fontsize, color="black", zorder=6,
                             path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
            bb = tx.get_window_extent(r)
            inside = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
            nover = sum(bb.overlaps(o) for o in obst)
            score = (0 if inside else 100) + nover
            if best is None or score < best[0]:
                if best is not None:
                    best[1].remove()
                best = (score, tx, bb)
            else:
                tx.remove()
            if score == 0:
                break
        obst.append(best[2])


COMPONENTS = [("Oil production", "Oil production (10$^3$ sm$^3$)", 1e-3),
              ("Water production", "Water production (10$^3$ sm$^3$)", 1e-3),
              ("Water injection", "Water injection (10$^3$ sm$^3$)", 1e-3)]


def _auto_err_ylim(vals, lo_min=-5.0, hi_min=5.0, cap=100.0):
    v = np.concatenate([x[np.isfinite(x)] for x in vals]) if vals else np.zeros(0)
    if v.size == 0:
        return (lo_min, hi_min)
    lo, hi = min(v.min(), lo_min), max(v.max(), hi_min)
    pad = 0.05 * (hi - lo)
    return (max(lo - pad, -cap), min(hi + pad, cap))


def plot_cumulative(series, tmax=1825, err_ylim=None, figsize=None):
    """Cumulative oil / water production per producer and water injection per injector (top: simulator solid,
    proxy dashed) and error relative to the well's own simulator volume (bottom).

    series: {(component, well): (days, true, estimate)} as from `read_cumulative_csv`.
    err_ylim: None (automatic, clipped to +-100 %) or three (lo, hi) pairs, one per column (the paper uses
    (-20, 5), (-20, 20), (-20, 5) for its detailed case).
    """
    import matplotlib.pyplot as plt
    short = short_name
    pal = [fs.OI[k] for k in ("blue", "verm", "green", "purple", "orange", "sky")] + ["#882255", "0.6"]
    # 8 producers all distinct; no black (black = True/Proxy keys)
    colr = {**{f"P{i + 1}": pal[i % 8] for i in range(40)}, **{f"I{i + 1}": pal[i % 8] for i in range(40)}}
    fig, axs = plt.subplots(2, 3, figsize=figsize or (fs.COL2, 5.6), constrained_layout=True, sharex=True)
    rels = [[], [], []]
    for c, (comp, lab, sc) in enumerate(COMPONENTS):
        wl = sorted({k[1] for k in series if k[0] == comp and k[1] != "Field"},
                    key=lambda w: (len(short(w)), short(w)))
        for w in wl:
            d, t, e = series[(comp, w)]
            axs[0][c].plot(d, t * sc, "-", color=colr[short(w)], label=short(w))
            axs[0][c].plot(d, e * sc, "--", color=colr[short(w)])
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(t > 0, 100 * (e - t) / t, np.nan)   # relative to the well's own volume
                if comp == "Water production":   # relative error of a near-zero volume before breakthrough is meaningless
                    rel = np.where(t >= 0.01 * t[-1], rel, np.nan)
            rels[c].append(rel)
            axs[1][c].plot(d, rel, "-", color=colr[short(w)], label=short(w))
        axs[0][c].set_ylabel(lab)
        axs[1][c].set_xlabel("Time (days)")
        axs[1][c].set_ylabel("Relative error (%)")
        axs[1][c].axhline(0, color="0.5", lw=0.5)
        if wl:
            axs[0][c].legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2)
        else:
            axs[0][c].text(0.5, 0.5, "no wells", transform=axs[0][c].transAxes, ha="center", va="center")
        axs[0][c].tick_params(labelbottom=False)
        for r in (0, 1):
            axs[r][c].text(0.0, 1.03, f"({chr(97 + 3 * r + c)})", transform=axs[r][c].transAxes, fontsize=8,
                           fontweight="bold", ha="right")
            axs[r][c].set_xlim(0, tmax)
    keys = [Line2D([], [], color="k", ls="-", label="True"), Line2D([], [], color="k", ls="--", label="Proxy")]
    axs[0][2].plot([], [], "k-", label="True")
    axs[0][2].plot([], [], "k--", label="Proxy")
    # (a): wells laid out as in (b) (4 columns), True/Proxy on a further line
    lw = axs[0][0].legend(frameon=False, ncol=4, loc="upper left", handlelength=1.2, fontsize=6.5, columnspacing=0.8)
    axs[0][0].add_artist(lw)
    fig.canvas.draw()
    bb = lw.get_window_extent().transformed(axs[0][0].transAxes.inverted())
    axs[0][0].legend(handles=keys, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(bb.x0, bb.y0 + 0.01),
                     handlelength=1.2, fontsize=6.5, columnspacing=0.8, borderaxespad=0)
    axs[0][2].legend(frameon=False, ncol=3, loc="upper left", handlelength=1.4, fontsize=6.5, columnspacing=0.8)
    axs[0][1].legend(frameon=False, ncol=4, loc="upper left", handlelength=1.2, fontsize=6.5, columnspacing=0.8)
    for c in range(3):
        lo, hi = axs[0][c].get_ylim()
        axs[0][c].set_ylim(lo, hi * (1.45 if c == 0 else 1.3))
        axs[1][c].set_ylim(*(err_ylim[c] if err_ylim is not None else _auto_err_ylim(rels[c])))
    return fig
