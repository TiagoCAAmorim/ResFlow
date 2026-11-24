# 6. Run a trained model

`resflow rollout` rolls a state model out on the simulations of a dataset split (default `test`): it starts from the true state
of the first day, predicts every following day from its own previous prediction, and compares with the simulator. The dataset
is the one made in [05_train.md](05_train.md) (`resflow convert` + `resflow build`). Weights: the shipped bundles in `weights/`
or a run folder of `resflow train-state`.

## Two modes

**Imposed.** The simulator's well volumes drive the state model (the state model alone):

```bash
resflow rollout --weights weights/state_2d_1d --dataset $DATA/spe10/datasets \
    --sims L32_60x60_01_8P4I_variable --out $DATA/rollouts/imposed
```

**Rate.** The rate model predicts the well volumes from the state and the BHP schedule (add `--rate-weights`):

```bash
resflow rollout --weights weights/state_2d_1d --rate-weights weights/rate_2d_1d --dataset $DATA/spe10/datasets \
    --sims L32_60x60_01_8P4I_variable --out $DATA/rollouts/rate
```

Options: `--sims` (names with or without `.sr3`; default all simulations of the split), `--split`, `--min-day` / `--max-day`
(window, default 0 to 1827: the rollout predicts day 1 to day 1825), `--start-day`, `--report-days` (default 1000, besides the last day),
`--n-cells` (cells written to the per-cell CSV, default 32), `--device cpu|cuda`. A 1,825-day rollout is fast on a GPU
and takes minutes per deck on the CPU (about 45 s for 200 days of a 32x32 deck on a CPU with many cores).

Rollouts are deterministic by default (exact sums of the edge flows and deterministic kernels, same result on repeated runs of the same
device). `--fast` uses GPU atomics, which is faster but not reproducible.

## Outputs

One set of files per simulation `<sim>` (the name with `.sr3`), in `--out`:

| file | content |
|---|---|
| `rollout_<sim>.npz` | `days`, `true` and `pred` ([days, cells, variables]; `var_names` = Rs, P, So, Sg, Sw, Vo_std, Vg_std, Vw_std), `pred_days`, `ijk`, `stat` (static cell data; column 3 is Kx), `well_marker_ij/type/name`, `wellp`, `wellw`, `wellg` (true well volumes) and their `*_index`, `sim_model`; rate mode adds `well_pred_days`, `pred_producer_rates` (dNp, dGp, dWp per producer and step) and `pred_water_injector_rates` |
| `rollout_<sim>.csv` | `model, day, cell, variable, True, Est` for 32 cells (well cells, then evenly spaced), and the material-balance rows `MBe_<phase>` |
| `rollout_errors_<sim>.csv` | per day and variable over all cells: `mae, mse, min, p01 ... p99, max, mean, std, r2, shift, shifted_r2` |
| `rollout_thresholds_<sim>.csv` | % of cells above 10, 5 and 1 % relative error |
| `rollout_summary.csv` | one row per simulation, mode and report day: P R2, shifted P R2, Sw R2, P MAE (kgf/cm2 and MPa), Sw MAE, material-balance errors in %, and in rate mode the cumulative errors |
| `rollout_performance.yaml` | days, steps and run time per simulation |
| rate mode only: `rollout_wells_<sim>.csv`, `rollout_cumulative_<sim>.csv`, `rollout_cumulative_errors_<sim>.csv` | per-well volumes (`well, series, day, True, Est`), cumulative volumes per well and field (`component, well, day, True, Est, rel_err_pct`) and their final errors |

Pressures in the files are in kgf/cm2 (the unit of the data); only `rollout_summary.csv` also gives P MAE in MPa
(1 kgf/cm2 = 0.0980665 MPa).

## Plot

`resflow.plotting` makes the figures of the paper (needs `matplotlib`):

```python
from resflow import plotting as P
from resflow.plotting import style

days = (365, 1000, 1825)
style.setup(110)                                   # dpi
ro = P.read_rollout_npz("rollouts/rate/rollout_L32_60x60_01_8P4I_variable.sr3.npz", days=days)
markers = (ro.marker_ij, ro.marker_type, ro.marker_name)
style.save(P.plot_permeability(ro.kx, ro.ijk, markers), "perm.png", 110)
style.save(P.plot_state_maps(P.state_fields(ro), days, ro.ijk, markers), "maps.png", 110)     # P and Sw: simulator, proxy, error
cum = P.read_cumulative_csv("rollouts/rate/rollout_cumulative_L32_60x60_01_8P4I_variable.sr3.csv")   # rate mode
style.save(P.plot_cumulative(cum), "wells.png", 110)
```

The days must exist in the rollout. Pressure is plotted in MPa.

## Compare with the published results

`results/final_model/` holds the metrics and figures of the 65 test cases of the paper, in both modes (see its `README.md`).
Compare a new rollout with `all_rollouts.csv` (one row per case; columns `imposed_*` and `rate_*` at day 1825) or with the
`metrics.csv` in the case folder:

```python
import pandas as pd
pub = pd.read_csv("results/final_model/all_rollouts.csv")
new = pd.read_csv("rollouts/imposed/rollout_summary.csv")
new = new[new.day == "end"]                                    # last day (1825); the report days are numbers
case = "L32_60x60_01_8P4I_variable"
print(pub[pub.case == case][["imposed_P_r2", "imposed_P_mae_MPa", "imposed_Sw_r2"]])
print(new[new.sim == case + ".sr3"][["P_r2", "P_mae_MPa", "Sw_r2"]])
```

## Rerun noise: expect small differences

A rerun does not reproduce the published numbers digit by digit. The paper's rollouts were computed on a GPU with atomic
scatter operations, whose summation order changes from run to run, and the small rounding differences grow over 1,825
autoregressive steps (the old code differed from itself by up to 21 % in P MAE between two runs). This repository sums in a fixed order by default, so its
own reruns are identical, but its numbers still differ from the published draw.

Example (5 test decks, shipped weights, both modes, day 1825): the detailed case
`L32_60x60_01_8P4I_variable` in imposed mode gives **P MAE 0.398 MPa with this code and 0.339 MPa in the paper** (P R2 0.9955 vs
0.9958, Sw R2 0.9991 for both); in rate mode 0.141 vs 0.139 MPa. Across the 10 rollouts P MAE differs by 0.02 to 0.06 MPa and R2 by
at most 0.002 (one 120x60 imposed case: 0.908 vs 0.899). Material-balance errors near zero differ in relative terms but not in
absolute ones. The published files in `results/final_model/` are the draw of the paper and are not regenerated.
