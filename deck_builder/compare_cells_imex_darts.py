"""Map comparison of IMEX (.sr3) and DARTS cell states at the last date: pressure [bar] and Sw, with differences.

Usage:
    python cmg_to_darts.py deck.dat -o darts_out
    python compare_cells_imex_darts.py deck.dat darts_out out.png          # <deck>.sr3 next to the deck
Sw of IMEX = 1 - SO - SG (OUTSRF GRID PRES SO SG); IMEX pressure is stored in kPa and converted to bar.
Well names are read from the PERF lines of the deck (circle = producer, triangle = injector).
"""
import sys, re, h5py, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cmg_to_darts import cell_states
dat, run_dir, out = sys.argv[1:4]
case = __import__("pathlib").Path(dat).stem
nx, ny = map(int, re.search(r"_(\d+)x(\d+)_", case).groups())
f = h5py.File(__import__("pathlib").Path(dat).with_suffix(".sr3"))
g = f["SpatialProperties"][sorted(k for k in f["SpatialProperties"] if k.isdigit())[-1]]
pi = g["PRES"][:] / 100.0                      # kPa -> bar
swi = 1 - g["SO"][:] - g["SG"][:]
t, p, sw = cell_states(run_dir, dat)
print("days", t[-1]); pd_, swd = p[-1], sw[-1]
# orientation check on initial state
print("n", pi.size, nx*ny)
txt = open(dat).read()
wells = [(m[0], int(m[1]), int(m[2])) for m in re.findall(r"PERF\s+GEOA\s+'(\w+)'\s*\n\s*(\d+)\s+(\d+)", txt)]
wells = [(n, i, j) for n, (i, j) in [(w[0], (w[1], w[2])) for w in wells]]
def pl(ax, a, title, cmap, vmin=None, vmax=None, lab=None):
    im = ax.imshow(a.reshape(ny, nx), origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for n, i, j in wells:
        ax.plot(i-1, j-1, "ko" if n[0]=="P" else "w^", ms=5, mec="k"); ax.text(i-1+1, j-1-1, n, fontsize=6, color="k", bbox=dict(fc="w", alpha=.6, pad=0.3, lw=0))
    plt.colorbar(im, ax=ax, shrink=.8, label=lab)
fig, axs = plt.subplots(2, 3, figsize=(13, 8))
pl_lo, pl_hi = min(pi.min(), pd_.min()), max(pi.max(), pd_.max())
pl(axs[0,0], pi, "IMEX pressure", "viridis", pl_lo, pl_hi, "bar")
pl(axs[0,1], pd_, "DARTS pressure", "viridis", pl_lo, pl_hi, "bar")
dp = pd_ - pi; m = np.abs(dp).max()
pl(axs[0,2], dp, f"DARTS - IMEX (max |d| {m:.2f} bar)", "RdBu_r", -m, m, "bar")
pl(axs[1,0], swi, "IMEX Sw", "Blues", 0, 1, "Sw")
pl(axs[1,1], swd, "DARTS Sw", "Blues", 0, 1, "Sw")
ds = swd - swi; m2 = np.abs(ds).max()
pl(axs[1,2], ds, f"DARTS - IMEX (max |d| {m2:.3f})", "RdBu_r", -m2, m2, "Sw")
fig.suptitle(f"{case}, final day ({t[-1]:.0f} d)")
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"P: mean abs {np.abs(dp).mean():.3f} bar, max {np.abs(dp).max():.3f}, IMEX range {pi.min():.1f}-{pi.max():.1f}")
print(f"Sw: mean abs {np.abs(ds).mean():.4f}, max {np.abs(ds).max():.4f}, rms {np.sqrt((ds**2).mean()):.4f}")
print("corr", np.corrcoef(pi, pd_)[0,1], np.corrcoef(swi, swd)[0,1], wells)
