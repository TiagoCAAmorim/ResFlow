# deck_builder

Scripts that build the dataset of small 2D (and 3D) waterflood simulations from the SPE10 permeability field, as CMG IMEX
decks, and convert them for OPM Flow and open-DARTS. Self-contained (Python 3.10+, `pip install -r requirements.txt`).
MIT licence, see the `LICENSE` file of the repository.

- How to build the decks (paper decks, 10-day, 3D and fluvial variants): [docs/02_build_decks.md](../docs/02_build_decks.md)
- How to run them (IMEX, OPM Flow, open-DARTS): [docs/03_run_simulators.md](../docs/03_run_simulators.md)
- Well placement algorithm: [WELL_LAYOUT.md](WELL_LAYOUT.md); open-DARTS notes: [DARTS.md](DARTS.md)

| Step | Script |
|---|---|
| 0. download inputs | `download_spe10.py` (and `layer_correlation.py` for other layer ranges) |
| 1. permeability patches | `generate_spe10_patches.py` |
| 2. well layouts, BHP schedules | `generate_well_layouts.py`, `bhp_schedule.py` |
| 3. decks | `build_decks.py` with a template from `templates/` |
| OPM Flow | `cmg_to_opm.py`, `run_opm_batch.py`, `compare_cmg_opm.py` |
| open-DARTS | `cmg_to_darts.py`, `run_darts_batch.py`, `compare_cells_imex_darts.py` |
