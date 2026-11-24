# 9. Tutorial: roll out the shipped weights on the demo data

You will download OPM Flow results of six test decks, build the dataset of one of them, roll the
shipped models out in rate mode, plot the maps and well curves, compare with the simulator, and run the imposed mode. No
simulator licence and no GPU are needed. Install first ([00_install.md](00_install.md)); run everything from the repository
root with the virtual environment active. `_data/` is git-ignored.

## 1. Download the demo data

```bash
python scripts/download_demo.py --out _data/demo
```

The script fetches the files of the Zenodo record (one zip per deck and the deck index of each set), checks their md5, and
unzips them: `_data/demo/spe10/test/raw/<deck>/` for five SPE10 Tarbert decks and `_data/demo/spe10_fluvial/test/raw/<deck>/` for
one Upper Ness deck, plus `decks_index.csv` in each set. The zips stay in `_data/demo/_download` and can be deleted.
`--list` shows the files without downloading; `--from-dir <folder>` uses a local copy of the record. Rerunning skips what is
already there.

| Deck | Grid | Wells | Schedule |
|---|---|---|---|
| `L33_32x32_01_8P4I_variable` | 32x32 | 8 producers, 4 injectors | variable BHP (used below) |
| `L31_32x32_02_7P4I_fixed` | 32x32 | 7 producers, 4 injectors | fixed BHP |
| `L34_32x32_00_1P0I_fixed` | 32x32 | 1 producer, no injector | fixed BHP |
| `L32_60x60_01_8P4I_variable` | 60x60 | 8 producers, 4 injectors | variable BHP |
| `L31_120x60_00_5P3I_variable` | 120x60 | 5 producers, 3 injectors | variable BHP |
| `L59_32x32_00_4P2I_fixed` (fluvial) | 32x32 | 4 producers, 2 injectors | fixed BHP |

These raw folders are what `resflow convert` writes ([08_reference.md](08_reference.md), section 8.4), so the conversion step is
already done.

## 2. Build the dataset

To keep the tutorial short, work on one deck. Put its raw folder in a dataset root of its own and build:

```bash
mkdir -p _data/tutorial/test/raw
cp -r _data/demo/spe10/test/raw/L33_32x32_01_8P4I_variable _data/tutorial/test/raw/
resflow build --root _data/tutorial
```

It takes about 10 seconds and ends with `test: 1 processed files, 1,826 one-step samples`. It writes
`_data/tutorial/test/processed/data_0.pt` (about 40 MB) and `data.json`. To build the whole set, run
`resflow build --root _data/demo/spe10` (about 6 minutes, the 120x60 deck takes most of it).

## 3. Rate-mode rollout

```bash
resflow rollout --weights weights/state_2d_1d --rate-weights weights/rate_2d_1d \
    --dataset _data/tutorial --out _data/tutorial/rate --device cpu
```

The rollout starts from the true initial state and predicts 1,825 days from its own previous prediction; the well volumes come
from the rate model, which sees only the BHP schedule. It prints a progress line every 200 days and, at the end, a table.
It takes about 3 minutes on a CPU or a GPU (a 32x32 deck is too small for the GPU to help). Expected result on a CPU (other CPUs can differ in the last digits; on a GPU the numbers differ slightly, e.g. P MAE 0.101 MPa):

```
 sim                              mode  day   P_r2    P_shifted_r2  Sw_r2  P_mae_MPa  Sw_mae     MBe_oil_pct_abs  MBe_water_pct_abs
 L33_32X32_01_8P4I_VARIABLE.DATA  rate  1000  0.9996  0.9996        0.9991  0.07927  0.0006671  0.1175           0.01535
 L33_32X32_01_8P4I_VARIABLE.DATA  rate  end   0.9993  0.9994        0.9998  0.1049   0.0006032  0.1667           0.1597
```

`end` is day 1825. The simulation is called `L33_32X32_01_8P4I_VARIABLE.DATA` (upper case, from the OPM run); it is the name in the
output files. The files written are listed in [08_reference.md](08_reference.md), section 8.6.

> With a dataset of several decks, `--sims <deck>` selects one (deck names work for IMEX and OPM data).

## 4. Plot maps and well curves

```python
from resflow import plotting as P
from resflow.plotting import style

days = (365, 1000, 1825)
style.setup(110)
sim = "L33_32X32_01_8P4I_VARIABLE.DATA"
out = "_data/tutorial/rate"
ro = P.read_rollout_npz(f"{out}/rollout_{sim}.npz", days=days)
markers = (ro.marker_ij, ro.marker_type, ro.marker_name)
style.save(P.plot_permeability(ro.kx, ro.ijk, markers), "perm.png", 110)
style.save(P.plot_state_maps(P.state_fields(ro), days, ro.ijk, markers), "maps.png", 110)
style.save(P.plot_cumulative(P.read_cumulative_csv(f"{out}/rollout_cumulative_{sim}.csv")), "wells.png", 110)
```

Save it as a script (`python plot.py`) or paste it into a Python session. `perm.png` is the permeability with the wells
(producers as white triangles pointing up, injectors as black triangles pointing down), `maps.png` shows pressure (MPa) and
water saturation at the three days as simulator, proxy and error (the colour scale is shared by simulator and proxy, the error
scale is symmetric), `wells.png` the cumulative oil, gas and water production and water injection of each well and the field,
simulator against proxy, with the relative error. The days must exist in the rollout.

## 5. Compare with the true data

The rollout file holds the simulator's fields (`true`) next to the proxy's (`pred`), so every comparison is a few lines. The
metrics of section 3 are in `rollout_summary.csv`; the same number computed by hand from the raw simulation:

```python
import numpy as np, pandas as pd, torch

sim = "L33_32X32_01_8P4I_VARIABLE.DATA"
out = "_data/tutorial/rate"
raw = "_data/tutorial/test/raw/L33_32x32_01_8P4I_variable"

z = np.load(f"{out}/rollout_{sim}.npz", allow_pickle=True)
iP = list(z["var_names"]).index("P")
p_true = torch.load(f"{raw}/res_dynamic/pres.pth", weights_only=False)[1825].numpy()      # kgf/cm2, raw simulator output
p_pred = z["pred"][list(z["pred_days"]).index(1825), :, iP]
print("pressure MAE at day 1825 (MPa):", np.abs(p_pred - p_true).mean() * 0.0980665)       # 0.1049, as in the table

e = pd.read_csv(f"{out}/rollout_cumulative_errors_{sim}.csv")                              # final cumulative volumes
print(e[e.well.str.contains("Field")][["component", "true_cum", "pred_cum", "rel_err_pct"]])
```

Expected: a pressure MAE of 0.105 MPa and, for the field, cumulative errors of -1.7 % (oil), -1.7 % (gas), +1.0 % (water
production) and +1.2 % (water injection). Per-cell time series (pressure, saturations, Rs for 32 cells, the well cells first)
are in `rollout_<sim>.csv` (columns `day, cell, variable, True, Est`), the error per day over all cells in
`rollout_errors_<sim>.csv`. For the published results on IMEX data (the paper) see `results/final_model/` and
[06_run_trained_model.md](06_run_trained_model.md); an OPM rollout is expected to be close to them in rate mode, not
identical (different simulator, [04_opm_conversion.md](04_opm_conversion.md)).

## 6. Imposed mode, and the OPM water caveat

Without `--rate-weights` the simulator's own well volumes drive the state model:

```bash
resflow rollout --weights weights/state_2d_1d --dataset _data/tutorial --out _data/tutorial/imposed --device cpu
```

About 3 minutes again. For this deck the result is `P_mae_MPa` 0.4257 and `P_r2` 0.9943 at day 1825 (0.312 MPa and 0.9968 at day
1000), against 0.1049 MPa in rate mode. Plot it as in section 4 (change `out`; there is no cumulative file in this mode).

Read imposed-mode numbers on OPM data with care. The shipped weights were trained on IMEX results. In producers that make almost
no water, OPM produces about ten times more water than IMEX in the first months (about 0.01 % of the oil), the converted volumes
equal the OPM totals, and the state model is sensitive to such tiny volumes: on the OPM demo decks its imposed-mode pressure can end at a
shifted level (the 60x60 deck of the demo: P MAE 2.4 MPa against 0.4 MPa on IMEX data, R2 above 0.99 once the shift is removed;
`P_shifted_r2` in `rollout_summary.csv` shows it). The rate mode does not have this problem. Details in
[04_opm_conversion.md](04_opm_conversion.md). The deck used here, `L33`, is one of the better behaved.

## 7. Next steps

- Other decks: run section 2 on another folder of `_data/demo/spe10/test/raw/`, or `resflow rollout ... --dataset
  _data/demo/spe10` for all five (use `--device cuda` on a GPU).
- The fluvial deck needs the fine-tuned weights: `--weights weights/state_2d_1d_fluvial_ft --dataset _data/demo/spe10_fluvial`
  (imposed mode).
- Your own decks, simulators and training: [02](02_build_decks.md) to [05](05_train.md).
