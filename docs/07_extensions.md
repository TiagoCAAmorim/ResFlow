# 7. Extensions: 10-day steps, 3D, fluvial fine-tune

Each recipe is a config in `configs/` and works with IMEX or OPM data: build the decks ([02](02_build_decks.md)), run them with
IMEX or OPM ([03](03_run_simulators.md), [04](04_opm_conversion.md)), `resflow convert` (add `--format opm` for OPM), `resflow build`,
`resflow fit-norm`, then train. Training, resume and options are as in [05_train.md](05_train.md); rollouts as in
[06_run_trained_model.md](06_run_trained_model.md). The configs have no shipped weights (except the fluvial fine-tune).

## 10-day steps (2D)

Decks: the 10-day variant of [02_build_decks.md](02_build_decks.md) (`build_decks.py ... --date-step 10 --bhp-step 60`, 3,502 decks).

```bash
resflow convert --input $DATA/decks_10d/decks --index $DATA/decks_10d/decks_index.csv --output $DATA/spe10_10d/datasets --step-size 10
resflow build --root $DATA/spe10_10d/datasets --step-size 10
resflow fit-norm --dataset $DATA/spe10_10d/datasets
resflow train-state --config configs/state_2d_10d.yaml --set dataset=$DATA/spe10_10d/datasets run_dir=runs/state_2d_10d
resflow rollout --weights runs/state_2d_10d --dataset $DATA/spe10_10d/datasets --out $DATA/rollouts/10d
```

The architecture and the settings are those of the 1-day model; the samples are 10 days apart (training window: days 0 to 1000).
Use the same `--step-size 10` in `convert` and `build`. The 10-day rollout is made with a 10-day model; the shipped 1-day weights
do not apply.

## 3D (5 layers)

Decks: the 3D 10-day decks of [02_build_decks.md](02_build_decks.md) (`decks_3d_10d`, 648 decks). A small model is trained.

```bash
resflow convert --input $DATA/decks_3d_10d/decks --index $DATA/decks_3d_10d/decks_index.csv --output $DATA/spe10_3d_10d/datasets --step-size 10
resflow build --root $DATA/spe10_3d_10d/datasets --step-size 10
resflow fit-norm --dataset $DATA/spe10_3d_10d/datasets
resflow train-state --config configs/state_3d_10d.yaml --set dataset=$DATA/spe10_3d_10d/datasets run_dir=runs/state_3d_10d
resflow rollout --weights runs/state_3d_10d --dataset $DATA/spe10_3d_10d/datasets --split val --out $DATA/rollouts/3d
```

The config has 3 epochs of 20,000 steps and no EMA. Use the split that exists in your dataset for `--split`
(`test` if the decks include the test layers 31-35).

## Fluvial fine-tune (Upper Ness)

The state model of the paper, trained on the Tarbert, is fine-tuned on the fluvial Upper Ness decks (layers 36-70, built as in
[02_build_decks.md](02_build_decks.md), "Fluvial decks"). The starting point is `weights/state_2d_1d`.

```bash
resflow convert --input $DATA/fl_decks/decks --index $DATA/fl_decks/decks_index.csv --output $DATA/spe10_fluvial/datasets
resflow build --root $DATA/spe10_fluvial/datasets
resflow fit-norm --dataset $DATA/spe10_fluvial/datasets        # statistics of the fine-tune data, used by the loss
resflow finetune --weights weights/state_2d_1d --config configs/finetune_fluvial.yaml \
    --set dataset=$DATA/spe10_fluvial/datasets run_dir=runs/finetune_fluvial
resflow rollout --weights runs/finetune_fluvial --dataset $DATA/spe10_fluvial/datasets --out $DATA/rollouts/fluvial
```

Settings (by design): learning rate 8e-6, 2 epochs, 12 blocks, pushforward chain up to 4 from the first step.

- **Statistics.** The loss is normalised with the fine-tune dataset's statistics, so run `fit-norm` on that dataset first (without
  it `finetune` stops with an error; `loss_norm_stats` in the config or `--set` points to another file). The model's own input
  and output scalers are not changed: they come from the weights.
- **Shipped fluvial weights.** `weights/state_2d_1d_fluvial_ft` came from the step-1,110,000 snapshot of the final run (not from the
  final weights) and had no pushforward in its first epoch. This recipe starts from `weights/state_2d_1d` and uses chain 4 from the
  start, so a rerun is not the same run.
- The validation loss uses the `val/` split, or `test/` if there is no `val/`.
