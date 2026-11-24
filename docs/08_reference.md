# 8. Reference (user guide)

Inputs, outputs, options and expected behaviour of every command. The workflow pages ([00](00_install.md) to [07](07_extensions.md))
explain *when* to use them; this page explains *what they take and give*. Commands are written `resflow <command>`; if the
`resflow` script is not on your path, `python -m resflow <command>` is the same.

## 8.1 Command line

`resflow --help` lists the commands: `convert`, `build`, `fit-norm`, `train-state`, `train-rate`, `finetune`, `rollout`.
Every command prints its progress to the terminal and exits with a non-zero status and a one-line message on a usage error.

### `resflow convert`: simulation results to raw folders

| Option | Meaning | Default |
|---|---|---|
| `--format imex\|opm` | simulator that made the results (IMEX: `.sr3`, `.out`, `.dat` files; OPM: see [04](04_opm_conversion.md)) | `imex` |
| `--input` | folder with the results (for OPM: the IMEX decks folder, the OPM cases are in `<input>_opm/`) | required |
| `--index` | decks index CSV (columns `deck` and `set`, see 8.3) | required |
| `--output` | folder to write `train/`, `val/`, `test/` to | required |
| `--step-size` | days between the outputs used (1 daily decks, 10 for the 10-day decks) | 1 |
| `--wells-open-day` | a run that ends at or before this day never opened its wells and is discarded | 4 |
| `--split-column`, `--name-column` | index columns with the split and the deck name | `set`, `deck` |
| `--resume` / `--overwrite` | keep the output and skip what is converted / delete `train/ val/ test/` of `--output` first | neither |

Writes `<output>/<split>/raw/<deck>/` (8.4) and a `processing.log` listing discarded runs. A deck whose run stopped early is
kept up to the last day reached; a run that ended at or before `--wells-open-day` is discarded and logged.

### `resflow build`: raw folders to processed dataset

| Option | Meaning | Default |
|---|---|---|
| `--root` | dataset root with `<split>/raw/<deck>/` | required |
| `--step-size` | as in `convert` | 1 |
| `--samples-ratio` | fraction of the cell-step samples drawn for training by the histogram resampling | 0.7 |
| `--min-day`, `--max-day` | process only these days (the training window is chosen later, when the data is read) | all |
| `--seed` | seed of the resampling | 0 |
| `--resume` | skip simulations already processed | off |
| `--log` | progress log | `<root>/build.log` |

Writes `<root>/<split>/processed/data_<n>.pt` (one per simulation) and `<root>/<split>/data.json` (index of the processed
simulations). A progress bar shows one step per simulation.

### `resflow fit-norm`: normalisation statistics

Options: `--dataset` (required), `--out` (default `<dataset>/analysis/stats/lin_scaler.pt`), `--max-samples` (stored series
read, 1000), `--seed`, `--device`. Only needed before a training from scratch; rollouts read the statistics from the weights.

### `resflow train-state`, `resflow train-rate`

| Option | Meaning |
|---|---|
| `--config` | YAML config (8.2); a new run needs it |
| `--set KEY=VALUE ...` | override config entries, nested keys with dots: `--set dataset=data/spe10 train.epochs=1` |
| `--run-dir` | output folder (overrides `run_dir`) |
| `--device` | `cuda` or `cpu` (default: cuda if available) |
| `--resume RUN_DIR` | continue a run; `--set` after it changes entries (e.g. `train.epochs=14`) |

Output folder: `weights.pt` (EMA weights), `config.yaml`, `train_log.csv` (loss, clean loss, learning rate), `train_state.pt`
(optimizer, scheduler, EMA, random generators, position: what `--resume` needs). `weights.pt` + `config.yaml` is a weights
bundle (8.5). Checkpoints are written every `train.save_every` steps.

### `resflow finetune`

Same options as `train-state` plus `--weights BUNDLE` (required): the state bundle to start from. Architecture, features and
the input/output normalisation come from the bundle; the optimizer starts fresh. See [07](07_extensions.md).

### `resflow rollout`

| Option | Meaning | Default |
|---|---|---|
| `--weights` | state model: a bundle folder or a training run folder | required |
| `--rate-weights` | rate model bundle: **rate mode** (well volumes predicted from the BHP schedule); without it, **imposed mode** (the simulation's own well volumes) | imposed |
| `--dataset` | dataset root (output of `build`) | required |
| `--split` | split folder | `test` |
| `--sims` | simulation names; the `.sr3` suffix is optional (IMEX data). For OPM data use the exact name the dataset holds (see 8.6) | all of the split |
| `--out` | output folder | required |
| `--min-day`, `--max-day` | window of the simulation (end 1827: the daily decks are predicted to day 1825) | 0, 1827 |
| `--start-day` | start from the true state of this day | `--min-day` |
| `--report-days` | days of `rollout_summary.csv` besides the last day | 1000 |
| `--n-cells` | cells written to the per-cell CSV (well cells first, then evenly spaced) | 32 |
| `--fast` | GPU scatter sums, non-deterministic kernels: faster, not reproducible | off |
| `--device` | `cuda` or `cpu` | cuda if available |

## 8.2 Config keys (`configs/*.yaml`)

All files are plain YAML; every key can be overridden with `--set a.b=value`. Paths are relative to the working directory.

| Key | Meaning |
|---|---|
| `model` | `state` or `rate` (which trainer reads the file) |
| `dataset` | dataset root with `train/` (and `val/` or `test/` for the validation loss) |
| `run_dir` | output folder of the run |
| `seed` | seed of the data order and the initialisation |
| `norm_stats` | normalisation statistics file; `null` = `<dataset>/analysis/stats/lin_scaler.pt` from `fit-norm` |
| `loss_norm_stats` | (finetune) statistics that normalise the loss; `null` = those of the fine-tune dataset |
| `vols2p_weights` | weights file holding the frozen pressure estimator (volumes to pressure); the shipped state weights |
| `model_batch_limit` | time steps per network call in the validation loss (memory/speed) |
| `data.min_day`, `data.max_day` | days used for training samples (state: 0 to 1000; rate: all) |
| `physics.g` | constant `G` of the transmissibility term, 1e-4, never fitted; `physics.z_mean` mean depth (`null` = from the data) |
| `features.use_beta_ig`, `features.use_beta_iw` | gas / water injection mobility features |
| `well_type_embedding` / `well_type_embed_dim` | number of well types (3), embedding size (4), `active_eps` threshold of an active well |
| `network.hidden_dims` | processor hidden sizes (node, edge) |
| `network.enc_nodes_dims`, `enc_edges_dims`, `dec_nodes_dims` | encoder (nodes, edges) and decoder layer sizes |
| `network.kernel_dims`, `edge_mlp_dims` | kernel network of the convolution and the edge update MLP |
| `network.n_blocks` | processor blocks (12 state, 2 rate) |
| `network.conv_rank` | rank of the low-rank edge-conditioned convolution (8 state, 4 rate) |
| `network.activation`, `use_bn`, `add_inner_skip`, `decoder_concat_input`, `init_type`, `init_scale`, `dense_layers` | activation (gelu); LayerNorm after each block; skip inside a block; concatenate the input at the decoder; weight initialisation; initial scale of the learnable input affine (rate); dense layers on the pooled latent (rate) |
| `input_asinh` | `[c, k]` per heavy-tailed input: `asinh(z/c)/k` after normalisation |
| `train.epochs`, `train.steps_per_epoch` | epochs and steps per epoch (`null` = one pass over every step of the training set) |
| `train.lr`, `min_lr`, `warmup_steps`, `weight_decay`, `grad_clip` | AdamW learning rate, cosine floor and warm-up (rate), weight decay, gradient clipping |
| `train.scheduler` | state: ReduceLROnPlateau `{factor, patience, min_lr}` on the clean validation loss |
| `train.ema_decay` | exponential moving average of the weights (the saved ones); `null` = off |
| `train.block_schedule` | active processor blocks per epoch (the last value holds) |
| `train.pushforward` | `schedule` = largest chain length per epoch, `randomize_depth` = draw the chain length uniformly from 0 to the maximum |
| `train.loss_idxs` | outputs in the loss: indices into (Rs, P, So, Sg, Sw, Vo, Vg, Vw) |
| `train.train_mask` | `drop_frac`: probability of dropping a non-well cell from the loss; `seed` |
| `train.noise.std`, `train.noise_std` | input-state noise (state: 0; rate: 0.05 of the one-step change) |
| `train.clean_eval`, `train.clean_every`, `eval_sims` | clean one-step loss: every n steps on n random validation simulations |
| `train.save_every`, `log_every` | checkpoint and log intervals in steps |

## 8.3 Deck index CSV

One row per deck, written by `deck_builder/build_decks.py` (`decks_index.csv`):

| Column | Content |
|---|---|
| `deck` | deck name, `L<layer>_<grid>_<patch>_<n>P<m>I_<fixed\|variable>`; also the name of the raw folder |
| `case_id` | the deck without its schedule (a case has one `fixed` and one `variable` deck) |
| `schedule` | `fixed` or `variable` BHP schedule |
| `family_id` | layer, grid and patch (`L32_60x60_01`) |
| `set` | `train`, `validation` or `test`: the split `convert` writes to (`validation` goes to `val/`) |
| `size` | grid, `32x32`, `60x60`, `120x60` |
| `n_prod`, `n_inj` | numbers of producers and injectors |

## 8.4 Raw folder (output of `convert`, input of `build`; also the demo data)

`<root>/<split>/raw/<deck>/`. Arrays are PyTorch tensors (`.pth`, float32) with one row per output day (day 0 first; 1828 rows for
a daily 1825-day deck, the last rows are padding of the stored window), columns in the order of `model.json`.

| File | Content | Unit |
|---|---|---|
| `model.json` | `name`, `grid_type`, `model_type`, `n_cells`, `days` (output days), `steps` ([first, last] index ranges), well names `prod`, `winj`, `ginj`, number of PVT and KR tables | days |
| `res_static.pt` | PyG `Data`: `stat` [cells, 7] = depth, pore volume, porosity, permI, permJ, permK, KR table id; `edge_index` (both directions), `edge_tran`, `edge_dist`, `edge_dist_mod`, `ijk` (1-based), `mask` | m, m3, -, md; tran in md.m |
| `res_dynamic/pres.pth` | pressure per cell [days, cells] | kgf/cm2 |
| `res_dynamic/so.pth`, `sg.pth` | oil and gas saturation | - |
| `res_dynamic/rs.pth` | dissolved gas-oil ratio | sm3/sm3 |
| `well_dynamic/np.pth`, `gp.pth`, `wp.pth` | cumulative oil, gas, water production per producer [days, producers] | sm3 |
| `well_dynamic/wi.pth` | cumulative water injection per injector | sm3 |
| `well_dynamic/p_prod.pth`, `p_winj.pth` | BHP of producers / water injectors (the control value while the well is shut) | kgf/cm2 |
| `well_static.json` | BHP reference depth per well, groups `prod`, `winj`, `ginj` | m |
| `con_dynamic/*.pth` | the same series per well connection (perforation) | as above |
| `con_static.json` | per well and connection: `cell` (0-based), `depth`, `wi` (well index) | m, md.m |
| `pvt01.pth` | dict: `sat` (saturated PVT table), `bo`, `uo` (undersaturated tables), `denoil`, `dengas`, `denwat`, `cpor`, `prpor` | IMEX units |
| `krel01.wo.pth`, `krel01.go.pth` | water-oil and gas-oil relative permeability tables | - |

## 8.5 Processed dataset and weights bundle

**Processed** (`<root>/<split>/processed/data_<n>.pt`): a PyG `Data` per simulation with the graph (`edge_index`, `edge_tran`,
`edge_dist`, `edge_dist_mod`, `stat`, `ijk`, `mask`), the state history `x` [days, cells, 4] (Rs, P, So, Sg), the well and
connection series `wellp/wellw/wellg/conp/conw/cong` [days, wells, variables], their index and static tensors (`*_index`,
`*_stat`), `global_attr`, and for training `train_mask` and `weights` [steps, cells] (the histogram resampling). `data.json`
maps each file to its simulation (`name`, days, step size, well counts and names). The files are binary; read them with
`torch.load(..., weights_only=False)`.

**Weights bundle** (a folder): `weights.pt` (state dict: network, input/output normalisation, PVT and KR tables, pressure
estimator), `config.yaml` (`format_version`, architecture, ordered input features, normalisation, `input_asinh`, physics, `step_size`,
`units`, `training` summary), `MD5SUMS`, `README.md` (training summary). Rollout and fine-tuning read the normalisation
only from the bundle.

## 8.6 Rollout files

Written to `--out`, one set per simulation `<sim>` (the simulation name stored in the dataset: `<deck>.sr3` for IMEX data,
`<DECK>.DATA` in upper case for OPM data).

| File | Content |
|---|---|
| `rollout_<sim>.npz` | `days` and `pred_days`; `true`, `pred` [days, cells, variables] with `var_names` = Rs, P, So, Sg, Sw, Vo_std, Vg_std, Vw_std (P in kgf/cm2); `ijk`; `stat` (column 3 is Kx); `well_marker_ij`, `well_marker_type`, `well_marker_name`; `wellp`, `wellw`, `wellg` (true well volumes) and `*_index`; `sim_model`. Rate mode adds `well_pred_days`, `pred_producer_rates` (dNp, dGp, dWp per producer and step), `pred_water_injector_rates` |
| `rollout_<sim>.csv` | `model, day, cell, variable, True, Est` for `--n-cells` cells, plus `MBe_<phase>` rows |
| `rollout_errors_<sim>.csv` | per day and variable over all cells: `mae, mse, min, p01 ... p99, max, mean, std, r2, shift, shifted_r2` |
| `rollout_thresholds_<sim>.csv` | `model, day, variable, alpha, denom, pct_exceed`: % of cells above 10, 5, 1 % relative error |
| `rollout_summary.csv` | one row per simulation, mode and report day: `sim, mode, day` (`end` or a number), `day_value`, `P_r2, P_shifted_r2, P_mae_kgf_cm2, P_mae_MPa, Sw_r2, Sw_r2_clipped, Sw_mae, MBe_oil_pct_abs, MBe_water_pct_abs`, and in rate mode `cum_oil_rel_err_pct, cum_water_rel_err_pct, cum_winj_rel_err_pct, cum_water_err_pct_of_oil` |
| `rollout_performance.yaml` | per simulation: nodes, step size, mode, days simulated, steps, run time |
| rate mode: `rollout_wells_<sim>.csv` | `model, well, well_type, series, day, True, Est` |
| rate mode: `rollout_cumulative_<sim>.csv` | `model, component, well, day, True, Est, rel_err_pct, err_pct_of_oil` (cumulative volumes per well and field) |
| rate mode: `rollout_cumulative_errors_<sim>.csv` | final errors: `component, well, last_day, true_cum, pred_cum, rel_err_pct, mean_abs_rel_err_pct, err_pct_of_oil` |

Units: pressure kgf/cm2 in the files (MPa only in `P_mae_MPa`, factor 0.0980665), saturations dimensionless, volumes sm3.
R2 of a (nearly) constant field, such as Sw in a depletion case with one producer, can be very negative: judge such cases by the MAE.

## 8.7 Expected behaviour

A normal session prints (details vary):

- `convert`: one progress bar over the decks; ends with the counts per split. Look at `processing.log` for discarded decks.
- `build`: `Building dataset from <root>...`, `Logging to <root>/build.log...`, a bar over the simulations; a 32x32 deck takes
  seconds, a 120x60 deck one to three minutes. Writes `data_<n>.pt` of 1.3 times the size of the raw folder.
- `rollout`: nothing is printed until all simulations are done (a long CPU run looks idle); then a table with one row per
  simulation and the files above in `--out`. Typical CPU times for 1825 days: see the README, "Computational requirements".
  Rerunning the same command gives the same numbers (deterministic default).

Common errors:

| Message | Cause and fix |
|---|---|
| `No built split at <root>/test (run resflow build first)` | `build` did not run on that root, or the split name is wrong |
| `Not in <root>/test: ['<name>.sr3']` | `--sims` does not match a simulation of the split; the message lists what is missing. Names are those in `rollout_*` files or `data.json` (OPM data: upper case, ending `.DATA`) |
| `<path> is not a state model bundle` / `... rate model bundle` | `--weights` and `--rate-weights` swapped or a wrong folder |
| `The config needs 'dataset' ...`, `Give --config (new run) or --resume RUN_DIR.` | missing argument of a training command |
| `No normalisation statistics at ...: run resflow fit-norm` | training from scratch before `fit-norm` |
| `Override '...' is not key=value` | `--set` entry without `=` |
| `Unknown format '...'` | `--format` other than `imex` or `opm` |
| `DAT file ... does not exist`, `Well ... not found in SR3 file` | the `.dat`/`.sr3` pair of a deck is incomplete in the `convert` input folder |
| `Warning: the model was trained with step 1.0 day(s); the dataset has 10.0` | model and dataset step sizes differ; results are not meaningful |
| CUDA out of memory in `rollout` | use `--device cpu`, or a smaller grid; the paper's largest deck needed 15 GB of GPU memory |
