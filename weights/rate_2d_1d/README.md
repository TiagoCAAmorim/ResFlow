# rate_2d_1d: rate model, 1-day steps

Predicts the well volumes of each step (oil, gas and water production, water injection per well cell) from the
reservoir state, the BHP controls and the well indices, as a non-negative multiplier of a Peaceman-type
estimate (2 low-rank convolution blocks, latent 16/8). Used in rate mode, where the state model reads its
predictions instead of the simulator's well volumes.

- **Trained on:** IMEX results of the SPE10 Tarbert 2D training decks, daily steps, all days (0-1827), with
  Gaussian noise (0.05 x the one-step change) on the input state. 7 epochs x 182,106 steps, AdamW, EMA 0.999
  (these are the averaged weights). Schedule of this run: 200 warm-up steps and cosine decay 1e-4 -> 1e-6
  over 5 epochs, then 2 epochs at a constant 1e-5. The published recipe `configs/rate_2d_1d.yaml` is the
  single cosine run of the paper (7 epochs, 1e-4 -> 1e-6).
- **Files:** `weights.pt`, `config.yaml`, `MD5SUMS` (as for the state model; no vols2p estimator).
- **Metrics:** paper rate-mode results (Tables 4-7, cumulative well errors); `results/final_model/`.
