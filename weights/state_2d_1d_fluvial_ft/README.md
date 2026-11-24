# state_2d_1d_fluvial_ft: state model fine-tuned on the Upper Ness (fluvial) set

The state model of `state_2d_1d` fine-tuned on 2D decks of the fluvial Upper Ness formation of SPE10.

- **Trained on:** started from the final run's snapshot at step 1,110,000 (about 10 of its 12 Tarbert epochs),
  then 2 epochs on the Upper Ness training decks (32x32, daily steps, days 0-1000) with a new optimizer,
  lr 8e-6, all 12 blocks, pushforward off in the first epoch and chains up to 4 in the second, EMA 0.999. The normalisation statistics are those of
  the Tarbert model (kept frozen). The recipe applied to the shipped final weights: `configs/finetune_fluvial.yaml`.
- **Files:** `weights.pt`, `config.yaml`, `MD5SUMS`.
- **Metrics:** paper fine-tuning table; Upper Ness cases in `results/final_model/`.
