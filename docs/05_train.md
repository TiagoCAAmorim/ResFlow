# 5. Train the models

This page trains the final 2D, 1-day models of the paper from simulation results. The shipped weights (`weights/`) let you
skip it: go to [06_run_trained_model.md](06_run_trained_model.md).

Steps: decks ([02](02_build_decks.md)) -> simulator ([03](03_run_simulators.md), or [04](04_opm_conversion.md) for OPM)
-> `resflow convert` -> `resflow build` -> `resflow fit-norm` -> `resflow train-state` and `resflow train-rate`.
`$DATA` is the same data folder as in the other pages (any folder; `_data/` is git-ignored).

## 1. Prepare the dataset

IMEX results (`.sr3` and `.dat` next to each other in the decks folder):

```bash
resflow convert --input $DATA/decks/decks --index $DATA/decks/decks_index.csv --output $DATA/spe10/datasets
resflow build --root $DATA/spe10/datasets
```

For OPM results use `--format opm` in `convert` (see [04](04_opm_conversion.md)); `build` is the same. The result is a root with
`train/`, `val/` and `test/`. The configs read it from `dataset:`; set it with `--set dataset=$DATA/spe10/datasets`.

## 2. Normalisation statistics

```bash
resflow fit-norm --dataset $DATA/spe10/datasets
```

It reads the training split (1,000 stored series by default, `--max-samples`) and writes
`<dataset>/analysis/stats/lin_scaler.pt` (`--out` changes it). The state and the rate model use the same file. It takes
seconds to minutes. Only trainings from scratch need it: the shipped weights carry their own statistics.

## 3. State model

```bash
resflow train-state --config configs/state_2d_1d.yaml --set dataset=$DATA/spe10/datasets --run-dir runs/state_2d_1d
```

12 epochs, 12 blocks, learning rate 5e-5 with a plateau scheduler, EMA weights, one-step loss with a pushforward chain that grows
over the epochs (up to 4); the network and training settings are in the YAML (every key can be changed with `--set a.b=value`,
e.g. `--set train.epochs=1`). `--device cpu|cuda` (default: cuda if available).

The run folder holds `weights.pt` (EMA weights, the ones to use), `config.yaml`, `train_log.csv` (loss, clean loss, learning rate
every 10 steps) and `train_state.pt` (everything needed to resume). `weights.pt` and `config.yaml` form a weights bundle, so
the folder can be given to `resflow rollout --weights` directly.

## 4. Rate model

```bash
resflow train-rate --config configs/rate_2d_1d.yaml --set dataset=$DATA/spe10/datasets --run-dir runs/rate_2d_1d
```

7 epochs of one pass over every step of the training set, cosine learning rate 1e-4 to 1e-6. The rate model is small
(10,613 parameters); its run folder has the same files. It is independent of the state model, so the two can train in either order.

## Resume

```bash
resflow train-state --resume runs/state_2d_1d
resflow train-rate  --resume runs/rate_2d_1d       # same for the rate model
```

A run restarts at the saved position inside the epoch, with the optimizer, scheduler, EMA and random generators restored (the order of the
training files is drawn from seeded generators, so it is the same as in an uninterrupted run). Checkpoints are written every
`train.save_every` steps (2,000 for the state model). `--set` after `--resume` changes entries (e.g. `train.epochs=14` to train longer).

## Hardware and time

The paper's models were trained on one GPU (NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB): the state model (12 epochs of
108,000 steps) took about 4 days. A CPU is enough only for a smoke test of a setup:

```bash
resflow fit-norm --dataset $DATA/spe10/datasets --out $DATA/stats_test.pt --max-samples 20
resflow train-state --config configs/state_2d_1d.yaml --device cpu --run-dir $DATA/smoke \
    --set dataset=$DATA/spe10/datasets norm_stats=$DATA/stats_test.pt train.epochs=1 train.steps_per_epoch=20
```

## Differences with the shipped weights

- **State weights.** `weights/state_2d_1d` was trained with a mass-balance loss and a bias loss (weights 0.1 and 0.1) in addition
  to the one-step loss. This repository drops both terms (the paper recipe does not use them), so a retraining is not the same
  run as the shipped one.
- **Rate weights.** `weights/rate_2d_1d` came from a two-stage run (5 epochs of the cosine schedule, then 2 epochs at a constant
  1e-5). The config is one cosine run of 7 epochs, as in the paper's appendix table.
- Training from scratch is not bit-reproducible with the shipped weights (different seeds of the data order, GPU, the changes above).
  Expect similar, not identical, accuracy. Each weights folder has a `README.md` with its training summary.
