# state_2d_1d: final 2D state model, 1-day steps

The state model of the paper (encode-process-decode GNN, 12 low-rank convolution blocks, latent 128/64).
It predicts the standard-volume increments of oil, gas and water of every cell over one day; the physics
converts them into pressure and saturations.

- **Trained on:** IMEX results of the SPE10 Tarbert 2D training decks (32x32 grids), daily steps, days 0-1000.
  12 epochs x 108,000 steps, AdamW, lr 5e-5 with ReduceLROnPlateau, block curriculum 8/10/12, pushforward
  chains up to 4, EMA 0.999 (these are the averaged weights). The run used two extra loss terms (mass balance
  and bias, weight 0.1 each) that the published recipe `configs/state_2d_1d.yaml` and the paper do not use.
- **Files:** `weights.pt` (state dict: network, normalisation statistics, PVT and relative-permeability tables,
  vols2p pressure estimator), `config.yaml` (architecture, ordered input features, normalisation, asinh inputs,
  G = 1e-4, units), `MD5SUMS`.
- **Use:** `resflow rollout --weights weights/state_2d_1d --dataset <dataset root> --split test --out <folder>`
  (add `--rate-weights weights/rate_2d_1d` for rate mode). Loads on CPU or GPU; no dataset statistics needed.
- **Metrics:** paper Tables 4-7; per-case results in `results/final_model/`.
