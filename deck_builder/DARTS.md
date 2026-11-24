# Running the dataset decks in open-DARTS

> **Warning.** open-DARTS output (`wells.csv`, `output/*.h5`) **cannot be converted to training data** by ResFlow. The training data
> comes from IMEX results (the path of the paper). OPM Flow results can also be converted (see `docs/03_run_simulators.md`).
> Use DARTS only to run and compare simulations.

[open-DARTS](https://gitlab.com/open-darts/open-darts) (Delft Advanced Research Terra Simulator, TU Delft, Apache-2.0)
is an open-source reservoir simulator that uses operator-based linearization (OBL). It was written by former AD-GPRS
developers. Stanford's AD-GPRS itself is not publicly available, which is why this folder targets DARTS instead.

`cmg_to_darts.py` reads an IMEX deck from `build_decks.py` and builds and runs the same case as a DARTS model in Python.
On the quick-test decks, the results match OPM Flow within 0.5% on field cumulatives, and on the dataset decks
with fixed BHP they match IMEX within 0.4% on oil (see [Validation](#validation) and
[Comparison with IMEX](#comparison-with-imex)).

## Install

```bash
pip install open-darts                 # Python >= 3.10; wheels for Linux x86_64 and Windows x64
pip install pyyaml numpy pandas h5py resfo
```

Nothing else is needed: no licence and no separate executable. A virtual environment is a good idea, because
open-darts pins its own numpy/scipy versions:

```bash
python -m venv ~/venvs/darts && source ~/venvs/darts/bin/activate      # Windows: ~\venvs\darts\Scripts\activate
pip install open-darts h5py resfo pyyaml
```

## Run

```bash
# one deck: writes $DATA/decks/decks_darts/<deck>/
python cmg_to_darts.py $DATA/decks/decks/<deck>.dat                    # saves the cell data (static and dynamic) too
python cmg_to_darts.py $DATA/decks/decks/<deck>.dat --no-cells         # wells only (faster, much smaller)

# a folder, in parallel, compared with IMEX (<deck>.sr3 next to the deck) or OPM (see run_darts_batch.py)
python run_darts_batch.py $DATA/decks/decks -j 4
```

Outputs per deck (`<deck folder>_darts/<deck>/`):

| File | Content |
|---|---|
| `wells.csv` | per well at every `DATE` of the deck: `WBHP` (bar), `WOPR WWPR WGPR WWIR` (sm3/day), `WOPT WWPT WGPT WWIT` (sm3) |
| `summary.csv` | final values per well and the run time |
| `stopped_wells.csv` | wells stopped because their flow reversed (see below) |
| `comparison.csv` | (batch) final cumulatives vs the reference |
| `output/static.h5` | grid and rock per cell (`dx dy dz depth permx permy permz poro`, flattened with i fastest, m / mD / bar) and `wells/<name>/ijk` (1-based perforations); `nx ny nz` are attributes |
| `output/reservoir_solution.h5` | dynamic: pressure and water mass fraction per cell at every `DATE`; `cell_states()` converts it to (days, pressure [bar], Sw) |
| `run.log` | DARTS log |

The two `output/*.h5` cell files are skipped with `--no-cells`. A 60x60 case with daily dates (1827 dates) writes about 60 MB of dynamic data.

A 32x32 case takes 2-6 s on one core.

## How the deck is mapped

| IMEX (MODSI) | DARTS | Note |
|---|---|---|
| `CORNER` + `CORNERS` | `StructReservoir` with `dx dy dz depth` per cell | stops unless the cells are axis-aligned boxes touching their neighbours (true here: 50 x 50 x 5 m). Checked: DARTS transmissibilities equal the harmonic TPFA values |
| `PERMI/J`, `PERMK CON`, `POR CON` | `permx/y/z`, `poro` | |
| `CPOR`, `PRPOR` | `RockCompactionEvaluator` | kg/cm2 -> bar |
| `PVT`/`BOT`/`VOT`, `PB CON` | two-component dead oil (water, oil) | the oil stays undersaturated at Rs(Pb) because every producer BHP (>= 180 kg/cm2) is above Pb (150), so Bo(p) and mu(p) come from the BOT/VOT branch at Pb. The run stops if a BHP is at or below Pb |
| `DENSITY`, `BWI CW REFPW VWI CVW` | oil density (rho_o_sc + Rs rho_g_sc) / Bo, water as ECLIPSE `PVTW` | surface volumes: oil = oil mass / (rho_o_sc + Rs rho_g_sc), gas = Rs x oil, water = mass / rho_w_sc |
| `SWT` | tabulated krw(Sw), krow(Sw) | water-oil Pc must be zero (it is) |
| `VERTICAL BLOCK_CENTER WATER_OIL`, `REFDEPTH REFPRES DWOC` | hydrostatic pressure, Sw = connate | |
| `GEOMETRY K rw geofac wfrac skin` | Peaceman well index (same as OPM; DARTS computes the same value) | `--wi-scale` multiplies it |
| `SHUTIN '*'`, `TARGET BHP` | zero-rate control, then BHP control | |
| `INJECTOR MOBWEIGHT` | injector transmissibility rescaled by block total mobility | see below |
| `DATE` | one `run()` call per date | each `DATE` is a report time |
| `DTMAX` | `ts_control.dt_max` (`--dtmax` to override) | |

Two DARTS defaults differ from IMEX and OPM. Both are changed by default:

1. **Injector mobility.** DARTS takes the mobility of the injection flux from the well segment, which is pure water.
   IMEX `MOBWEIGHT` and OPM use the total mobility of the perforated block. Early on that is about 30 times smaller
   (krow(Swc)/mu_o against 1/mu_w). Without the fix, injection was 20% higher than OPM. The fix rescales the
   perforation transmissibility after every converged time step (one step lagged). Use `--inj-mobility water` to get
   the DARTS default back.
2. **Crossflow.** When a producer's new BHP target is above the block pressure, DARTS injects back through it. OPM
   stops the well until its next control keyword. The converter stops the well when its flow reverses (checked after
   every step) until the next `TARGET`, and lists it in `stopped_wells.csv`. Use `--allow-crossflow` to get the DARTS
   default back. It is not known what IMEX does here. Compare `stopped_wells.csv` with the IMEX `.out` file when you
   have one.

OBL resolution (`--n-points`, default 400 per axis) barely matters here: 100, 400 and 1000 points give final
cumulatives within 0.1%.

## Validation

### Against OPM Flow

OPM ran in the same environment from the `opm-simulators` wheel (`pip install opm
opm-simulators`, Linux, Python bindings to Flow):

```python
from opm.simulators import BlackOilSimulator
BlackOilSimulator("CASE.DATA").run()          # in the folder of build_decks.py --format opm
```

quick test (layer 31, two 32x32 patches; 2P1I and 5P3I with fixed and variable BHP, plus 1P0I), 5 years,
DARTS vs OPM:

| Case | Oil | Water | Injected water | Worst producer, oil |
|---|---|---|---|---|
| L31_32x32_00_1P0I_fixed | +0.26% | +1.51% | | 0.26% |
| L31_32x32_00_2P1I_fixed | +0.28% | +0.23% | +0.25% | 0.37% |
| L31_32x32_00_2P1I_variable | +0.17% | +0.24% | +0.22% | 0.24% |
| L31_32x32_00_5P3I_fixed | +0.19% | +0.20% | +0.20% | 0.26% |
| L31_32x32_00_5P3I_variable | +0.04% | +0.22% | +0.21% | 3.04% |
| L31_32x32_01_1P0I_fixed | +0.29% | +1.40% | | 0.29% |
| L31_32x32_01_2P1I_fixed | +0.12% | +1.46% | +0.06% | 0.14% |
| L31_32x32_01_2P1I_variable | +0.42% | +2.05% | -0.88% | 1.52% |
| L31_32x32_01_5P3I_fixed | +0.23% | +0.30% | +0.28% | 0.31% |
| L31_32x32_01_5P3I_variable | +0.25% | +0.33% | +0.30% | 0.31% |

Field totals, final cumulatives. The 1P0I water is small (about 130 sm3), so its percentage is large. Cell water
saturation at the last date (L31_32x32_00_2P1I_fixed) differs from OPM by at most 0.0009, and by 0.0002 on average.
In L31_32x32_00_2P1I_variable, DARTS stopped P2 on the same 5 dates on which OPM shut it.

### Against IMEX

Six dataset decks from the 1-day decks whose IMEX run ended with `Normal Termination` were run with
`run_darts_batch.py` (5 years, 1827 d; 3 s per 32x32 case, 18-21 s per 60x60 case, 26 s for the 120x60 one).
DARTS against IMEX, final cumulatives:

| Case | Wells | Oil | Water | Injected water | Worst producer, oil |
|---|---|---|---|---|---|
| L31_60x60_02_6P4I_fixed | 6P 4I | -0.11% | +0.07% | -0.00% | 0.13% |
| L32_60x60_02_6P4I_fixed | 6P 4I | -0.08% | +0.05% | +0.00% | 0.13% |
| L33_60x60_02_4P2I_fixed | 4P 2I | -0.11% | +0.04% | -0.00% | 0.22% |
| L31_120x60_00_1P0I_fixed | 1P 0I | -0.37% | +671% (94 vs 722 sm3) | | 0.37% |
| L01_32x32_00_2P1I_variable | 2P 1I | -0.14% | -0.38% | -0.32% | 0.21% |
| L01_32x32_03_5P3I_variable | 5P 3I | -0.77% | -6.07% | -1.09% | 5.41% (P4) |

- With fixed BHP, oil, water and injection agree within 0.4%. The 1P0I water is a few hundred sm3, so its
  percentage means nothing. The 60x60 cases are within 0.11% on oil.
- With variable BHP, oil agrees within 0.15% in the 2P1I case. In L01_32x32_03_5P3I_variable the field water is 6%
  low and producer P4 is 5.4% low in oil. DARTS stopped wells for crossflow 24 times on 11 dates in that case (P1-P5
  and I1, `stopped_wells.csv`). It is not known whether IMEX also keeps those wells closed until the next `TARGET`.
  This is the most likely cause, but `--allow-crossflow` was not tried on this case.

#### Cell maps at the last date

`compare_cells_imex_darts.py` plots the IMEX and DARTS pressure and water saturation of every cell at the last
date, with their difference (IMEX pressure is stored in kPa and converted to bar; Sw = 1 - So - Sg):

```bash
python cmg_to_darts.py $DATA/decks/decks/L32_60x60_02_6P4I_fixed.dat -o darts_out
python compare_cells_imex_darts.py $DATA/decks/decks/L32_60x60_02_6P4I_fixed.dat darts_out map.png
```

L32_60x60_02_6P4I_fixed (60x60 cells, 6 producers, 4 injectors), day 1827. Circles are producers, triangles
injectors:

(Figure not included in this repository.)

| | IMEX range | Mean abs. difference | Max abs. difference |
|---|---|---|---|
| Pressure | 210.2 - 514.1 bar | 0.05 bar | 0.39 bar |
| Sw | 0.18 - 0.54 | 0.0001 | 0.0038 |

The pressure difference is 0.1% of the range at most and is largest in the low-pressure corner near P3. The Sw
difference is concentrated on the waterfront (one-cell shifts of the front), as expected from different time stepping
and numerical dispersion. The two fields correlate at 0.99999 for both properties.
