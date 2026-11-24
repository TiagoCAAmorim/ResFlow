# 3. Run the simulators

The decks of [02_build_decks.md](02_build_decks.md) are IMEX decks. This page shows how to run them in three simulators.

> **Warning: which results can become training data.**
> - **IMEX (CMG)** is the path of the paper: the shipped weights were trained on IMEX results. The `.sr3` files are converted to the training format by `resflow convert` (see docs/04_...).
> - **OPM Flow** results can also be converted: <!-- TODO: placeholder until the OPM converter is documented (docs/04_...) -->OPM results can be converted with `resflow convert --format opm` (see docs/04_...).
> - **open-DARTS output cannot be converted to training data.** Use it only to run and compare simulations.

The simulators agree closely on these decks (final cumulative oil within about 1% on a 32x32 case; see the numbers below),
but they are not identical, and the models in the paper saw IMEX only.

## IMEX (needs a CMG licence)

IMEX is commercial software and is not distributed with ResFlow. Run each deck with the IMEX command of your installation,
from the folder of the deck (the decks include `../inc/` and `../perm/` with relative paths, so keep the folder layout of
`$DATA/decks/`):

For example, with IMEX 2024.20 on Windows (the version used in the paper; adjust the path to your installation):

```bat
cd %DATA%\decks\decks
"C:\Program Files (x86)\CMG\IMEX\2024.20\Win_x64\EXE\mx202420.exe" -f L33_32x32_01_8P4I_variable.dat -wd . -wait
```

On Linux the executable has the same options; its name and folder depend on the CMG version.

Each run writes `<deck>.sr3` (results; the file used by the conversion) and `<deck>.out` (log), next to the deck. Check
that the log ends with a normal termination: a run that never opens its wells (ended at or before day 4) is discarded by the
converter, a run that stops later is kept up to the last day reached. A 32x32 deck with daily output takes seconds to
minutes; 3,142 decks are best run in parallel (one IMEX process per core, as many as the licence allows).

## OPM Flow (open source)

Two ways to run OPM Flow, with the same results (on a test deck the summary files of the two are bit-identical):

- **Python wheels (default of `run_opm_batch.py`, no MPI)**: `pip install opm opm-simulators`. PyPI has `opm-simulators` 2026.4
  wheels for Linux x86_64 only (CPython 3.8 to 3.14; `opm` also has macOS arm64 wheels, but the simulator does not), so
  Windows users run it in WSL. Run a deck with
  `python -c "from opm.simulators import BlackOilSimulator; BlackOilSimulator('CASE.DATA').run()"` in the case folder
  (results are written next to the `.DATA`).
- **The `flow` binary** (OPM packages, `flow` on the PATH; the checks here used `flow` 2026.04). It starts through Open MPI,
  which can hang on some systems (we saw it in WSL); then use the wheels.

Convert a deck and run it with the binary:

```bash
python deck_builder/cmg_to_opm.py $DATA/decks/decks/L33_32x32_01_8P4I_variable.dat -o $DATA/opm/L33_32x32_01_8P4I_variable
cd $DATA/opm/L33_32x32_01_8P4I_variable && flow L33_32X32_01_8P4I_VARIABLE.DATA --output-dir=out
```

(`--no-restart` skips the grid results at every date: much faster, but then no pressure or saturation maps.) The conversion also
runs inside the deck builder (`build_decks.py --format opm`, which writes `opm/<deck>/` next to the IMEX decks). A whole folder,
run in parallel and compared with the IMEX `.sr3` files that sit next to the decks:

```bash
pip install h5py resfo
python deck_builder/run_opm_batch.py $DATA/decks/decks -j 6        # writes <decks>_opm/<deck>/ and comparison.csv
# options: --engine wheel|flow (default wheel), --python PATH (interpreter with the wheels), --timeout SECONDS per deck,
#          --only DECK... (subset); results of every deck go to <decks>_opm/<deck>/out/
python deck_builder/compare_cmg_opm.py case.sr3 opm_out/CASE.UNSMRY      # one case
```

What `cmg_to_opm.py` writes (all pressures converted from kgf/cm2 to bar):

- `GRID.INC` (corner-point grid, permeability, porosity), `PROPS.INC` (PVT, relative permeability, rock), `SOLUTION.INC`, `SCHEDULE.INC` (wells and BHP controls by date) and `<CASE>.DATA`.
- **Output requested** (`SUMMARY`): field `FOPR FWPR FWIR FOPT FWPT FWIT FPR`, well `WBHP WOPR WWPR WWIR WOPT WWPT WWIT`, and **per-connection vectors for every connection of every well: `CTFAC COPT CWPT CWIT CGPT CPR`** (connection transmissibility factor, cumulative oil, water, injected water and gas, and connection pressure). `RPTRST BASIC=2` writes the grid state at every report date. The connection vectors are what the OPM-to-training-data converter reads.
- **No crossflow.** Every well is declared with crossflow `NO` (`WELSPECS` item 10). A well whose flow would reverse (for example a producer whose new BHP target is above the pressure of its block) is shut until its next control. Pass `--crossflow` to allow crossflow (item 10 = `YES`).

Check of the no-crossflow setting (deck `L33_32x32_01_8P4I_variable`, 8 producers and 4 injectors, 1,827 days, OPM Flow 2026.04):
the final cumulative oil, produced water and injected water of every well are identical with `NO` and with `YES`
(all 36 values equal to the printed precision, e.g. P1 oil 15,723.53 sm3, I4 injection 121,437.6 sm3): no well reversed in this
deck. The explicit `NO` makes the rule independent of the simulator default. The comparison with IMEX on the same deck
(final cumulatives, IMEX against OPM):

| Well | Oil | Water | Injected water |
|---|---|---|---|
| P1 | +0.39% | -0.05% | |
| P3 | -1.92% | -8.92% | |
| P4 | -1.80% | +9.90% | |
| P8 | +0.06% | | |
| I2 | | | -0.22% |
| I4 | | | -0.38% |

(OPM minus IMEX relative to IMEX; all producers' oil within 2%, all injectors' water within 1.4%. The water percentages of producers
with a few sm3 of water, for example P2, P5 to P8, are large and meaningless.)

## open-DARTS

[open-DARTS](https://gitlab.com/open-darts/open-darts) runs the same decks from Python (no licence, no separate executable). Use a
separate virtual environment: open-darts pins its own numpy and scipy.

```bash
python -m venv $DATA/venvs/darts && source $DATA/venvs/darts/bin/activate
pip install open-darts h5py resfo pyyaml
python deck_builder/cmg_to_darts.py $DATA/decks/decks/L33_32x32_01_8P4I_variable.dat -o $DATA/darts/L33_32x32_01_8P4I_variable   # about 20 s
python deck_builder/run_darts_batch.py $DATA/decks/decks -j 4              # a folder, compared with the IMEX .sr3 or OPM results
```

The converter stops a well when its flow reverses (no crossflow, as in the OPM deck). The output is `wells.csv`, `summary.csv`,
`stopped_wells.csv` and, unless `--no-cells`, `output/static.h5` and `output/reservoir_solution.h5`. The mapping, the settings that make
DARTS match IMEX and OPM, and the validation are in [`deck_builder/DARTS.md`](../deck_builder/DARTS.md). On the deck above,
open-DARTS and OPM agree on the final cumulative oil of every producer within 0.1%.

> **Warning.** open-DARTS output **cannot be converted to training data**. Use IMEX (paper path) or OPM Flow.
