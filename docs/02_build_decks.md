# 2. Build the simulation decks

This page explains how to build the CMG IMEX decks of the paper (2D SPE10 Tarbert waterflood, 1-day output), and the
variants used for the extensions (10-day decks, 3D decks, fluvial Upper Ness decks). The code is in `deck_builder/`
(Python 3.10+; it does not import anything from the rest of ResFlow). A deck is a text file: **this step runs no simulation**
(running them is in [03_run_simulators.md](03_run_simulators.md)).

```bash
cd deck_builder
pip install -r requirements.txt
# the relative-permeability file comes in a RAR archive: install `unrar` (or `unar` / `7z`) with your package manager
```

## How it works

```
0 download_spe10.py           (once) download SPE10 Kx, KR and PVT sources
  layer_correlation.py        (optional) layer correlations  ->  split.csv, a train / validation / test split of a layer range
1 generate_spe10_patches.py   SPE10 Kx  ->  permeability patches (32x32, 60x60, 120x60), split into train / validation / test by layer
2 generate_well_layouts.py    patch + well count  ->  optimised producer/injector positions
  bhp_schedule.py             case  ->  BHP values over time ("fixed" and "variable")
3 build_decks.py              patch + layout + schedule + template  ->  .dat deck (and OPM deck)
```

Words used below:
- **patch / family**: one permeability map cut from one layer, id like `L33_120x60_01` (layer 33, 120x60, 2nd patch).
- **case**: a patch with a well count, e.g. `L33_120x60_01_5P3I` (5 producers, 3 injectors).
- **deck**: a case with a BHP schedule, e.g. `L33_120x60_01_5P3I_variable`. This is what you simulate.
- **Which decks exist**: train and validation cases get the `variable` BHP schedule only (the models are trained on variable BHP). Test cases also get the `fixed` (constant BHP) schedule, as a generalisation test, and an extra single-producer case `1P0I` (constant BHP, no injector).
- Train, validation and test sets are assigned by whole layer; the set of a patch is in `manifest.csv`.
- **Reproducibility.** Every random choice uses `--seed`, default `0`. The same inputs and the same seed give the same
  results on any machine. The seed used is stored in `config_used.yaml` of steps 1 to 3.

Run the scripts from `deck_builder/`, but write all data to one separate folder, `$DATA`, so that the repository holds only
code. `$DATA` is only a shell variable used in the examples (any folder; inside ResFlow, `_data/` is git-ignored). Every `--out`
folder must not exist yet or be empty.

## The paper decks (2D, 1-day output)

```bash
cd deck_builder
DATA=$HOME/spe10_dataset          # all outputs and downloads go here
TEMPLATE=templates/spe10_2d_paper.tpl

# 0. download the inputs (once): SPE10 permeability, KR (UNISIM-IV) and PVT (UNISIM-II-D)
python download_spe10.py --cache-dir $DATA/inputs

# 1. permeability patches: built-in split by layer (Tarbert, layers 1-35), seed 0
python generate_spe10_patches.py --out $DATA/patches --perm-file $DATA/inputs/spe_perm.dat

# 2. well layouts: use the layouts of the paper (shipped), then the BHP schedules
mkdir -p $DATA/layouts && cp paper_layouts/well_locations.csv $DATA/layouts/
python bhp_schedule.py --well-locations $DATA/layouts/well_locations.csv --out $DATA/layouts/bhp_schedules.csv

# 3. decks: 185 report dates for train / validation, a DATE every day for test
python build_decks.py --patches $DATA/patches --layouts $DATA/layouts --out $DATA/decks \
    --template $TEMPLATE --cache-dir $DATA/inputs
```

This writes 3,142 decks (2,367 train/validation, 775 test) to `$DATA/decks/decks/`, with `inc/` (KR, PVT, CORNERS) and
`perm/` (the patches) next to them, plus `decks_index.csv`. All options are the defaults; the settings used in the paper
(`seed 0, n_train 14, n_eval 6, well counts 2/1,...,8/4, test 1/0, d_min 500, n_iter 2000, rate 0.1, x_base 15, w_bal 0.25,
dtmax 10, report dates example (test: daily)`) are the defaults of the scripts.

Two templates are provided in `deck_builder/templates/`:

| Template | Use |
|---|---|
| `spe10_2d_paper.tpl` | the template of the paper's 1-day decks and of the fluvial decks (`WSRF GRID 1`, `WSRF WELL 1`: results are written at every time step) |
| `spe10_2d.tpl` | the current example template (`WSRF ... TIME`: results are written only at the report dates; `1:{nk}` perforation range, needed for 3D). Used for the 10-day and the 3D decks |

> **Why the layouts are shipped.** The commands above rebuild the 3,142 decks of the paper byte for byte (checked).
> The well-placement code (`generate_well_layouts.py`) was improved after the paper layouts were made (flow regions,
> injectors in separate regions, an injector is not placed behind another one), so running it places the wells
> differently in 796 of the 2,782 cases (29%); the well counts and the BHP schedules are the same.
> `paper_layouts/well_locations.csv` holds the layouts of the paper. To make new layouts, run step 2 instead:
>
> ```bash
> python generate_well_layouts.py --patches $DATA/patches --out $DATA/layouts --jobs 8   # about 2 min with --jobs 24
> python bhp_schedule.py --well-locations $DATA/layouts/well_locations.csv --out $DATA/layouts/bhp_schedules.csv
> ```

A small test first (3 test patches, two well counts, about 30 s):

```bash
python generate_well_layouts.py --patches $DATA/patches --out $DATA/t_layouts --sets test --max-patches 3 --well-counts 2/1,8/4 --jobs 3
python bhp_schedule.py --well-locations $DATA/t_layouts/well_locations.csv --out $DATA/t_layouts/bhp_schedules.csv
python build_decks.py --patches $DATA/patches --layouts $DATA/t_layouts --out $DATA/t_decks --template $TEMPLATE --cache-dir $DATA/inputs
```

## Variants

The commands below reproduce the folders of the extensions. All start from the same `$DATA/inputs` (step 0).

### 10-day decks

A DATE every 10 days, BHP changes every 60 days (all sets, the same dates); wells open on day 10. The patches and layouts
are those of the 1-day decks; only step 3 changes (the BHP values are drawn in step 3 with `--seed`, and
`bhp_schedules.csv` is not read). The schedules are `variable`, plus `fixed` and `staged` (one well opened every 60 days)
for the test set.

```bash
python build_decks.py --patches $DATA/patches --layouts $DATA/layouts --out $DATA/decks_10d \
    --template templates/spe10_2d.tpl --cache-dir $DATA/inputs --date-step 10 --bhp-step 60
```

This writes 3,502 decks (2,727 variable, 415 fixed, 360 staged).

### 3D decks (5 layers)

Patches are blocks of 5 consecutive layers (`L01-05_32x32_00`); the split is by block: test 31-35, validation 26-30, train 1-25
(the default split with `--nk 5`; no `--layer-split`). Wells perforate all 5 layers and are placed on the layer-averaged map. The layouts
use `--k-min 10` (cells below 10 mD are barriers; layouts that isolate a well are re-annealed) and no `1P0I` test case.

```bash
python generate_spe10_patches.py --out $DATA/p3d --perm-file $DATA/inputs/spe_perm.dat --nk 5
python generate_well_layouts.py --patches $DATA/p3d --out $DATA/l3d --k-min 10 --test-well-counts "" --jobs 8
python bhp_schedule.py --well-locations $DATA/l3d/well_locations.csv --out $DATA/l3d/bhp_schedules.csv
python build_decks.py --patches $DATA/p3d --layouts $DATA/l3d --out $DATA/decks_3d --template templates/spe10_2d.tpl --cache-dir $DATA/inputs
# 10-day version of the same decks
python build_decks.py --patches $DATA/p3d --layouts $DATA/l3d --out $DATA/decks_3d_10d --template templates/spe10_2d.tpl \
    --cache-dir $DATA/inputs --date-step 10 --bhp-step 60
```

This writes 580 decks (1-day pattern; test decks with a DATE every day) and 648 decks (10-day).

### Fluvial decks (Upper Ness, layers 36-70)

The Upper Ness is a fluvial formation (layers 36-85 of SPE10; layers 36-70 are used). The layer split is not fixed: it is searched
by `layer_correlation.py` (random search, seed 0) so that no train layer correlates above 0.3 with a validation or test layer. The
result for layers 36-70: train 36-47 and 51-57, validation 65, 66, 67 and 69, test 49, 59, 60 and 61; the other layers are excluded.

```bash
python layer_correlation.py --out $DATA/fl_corr --perm-file $DATA/inputs/spe_perm.dat \
    --ni 60 --nj 220 --nk 85 --k-init 36 --k-end 70 --window 32
python generate_spe10_patches.py --out $DATA/fl_patches --perm-file $DATA/inputs/spe_perm.dat --layer-split $DATA/fl_corr/split.csv
python generate_well_layouts.py --patches $DATA/fl_patches --out $DATA/fl_layouts --k-min 10 --score-min-frac 0.2 --jobs 8
python bhp_schedule.py --well-locations $DATA/fl_layouts/well_locations.csv --out $DATA/fl_layouts/bhp_schedules.csv
python build_decks.py --patches $DATA/fl_patches --layouts $DATA/fl_layouts --out $DATA/fl_decks \
    --template templates/spe10_2d_paper.tpl --cache-dir $DATA/inputs
```

This writes 1,669 train/validation/test decks (test decks with a DATE every day).

## Run the decks

See [03_run_simulators.md](03_run_simulators.md): IMEX (paper path, needs a CMG licence), OPM Flow (`cmg_to_opm.py`; also
available with `build_decks.py --format opm`) and open-DARTS (`cmg_to_darts.py`).

> **Warning.** Only IMEX results (the path of the paper) and OPM Flow results can be converted to training data. **open-DARTS
> output cannot be converted to training data**: use it only to run and compare simulations.
<!-- TODO: OPM results can be converted with `resflow convert --format opm` (see docs/04_...) -->

## Reference: the steps in detail

### Step 1: permeability patches (`generate_spe10_patches.py`)

```bash
python generate_spe10_patches.py --out $DATA/patches --perm-file $PERM        # IMEX patch files (default)
python generate_spe10_patches.py --out $DATA/p --perm-file PERMI.inc --dry-run   # write no patch files
python generate_spe10_patches.py --out $DATA/patches --perm-file $PERM --layer-split $DATA/layer_corr/split.csv   # layers from layer_correlation.py
```

Reads Kx from `--perm-file` (the first of its three blocks; Ky = Kx in the patches, Kz is set in the deck), searches the validation layers (or reads the layer sets from `--layer-split`), cuts the patches and writes one
include file per patch to `<out>/perm/<family_id>.inc`, plus `manifest.csv` (one row per patch), `layer_assignment.csv`,
`validation_search.csv`, `colocation_corr.csv`, `overlap_by_layer.csv`, `patch_map_layers.png`, `summary.md` (all checks and
numbers), `config_used.yaml`, `README.md`. A failed check stops the run.

**3D patches** (`--nk`): `generate_well_layouts.py` and `build_decks.py` read `n_k` from the manifest, so nothing else needs an option. Wells are placed on one map per patch: the layer-averaged permeability on a single layer nk times thicker (lateral transmissibilities and oil volumes add over the layers; vertical flow is ignored), see `well_layout.collapse_layers`. Wells perforate all layers (`1:nk` in the template, which needs `{nk}`; the CORNERS file is `CORNERS_<ni>x<nj>x<nk>.inc`). `cmg_to_opm.py` expands the `k1:k2` perforation range.

Sets (layers 1-35 = Tarbert formation): test = layers 31-35 (32x32, 60x60 and 120x60 patches); validation = 5 layers chosen by
an exhaustive search over layers 1-30 (32x32 only); train = the other 25 layers (32x32 only). Patches are cropped, not rescaled.
With `--layer-split`, the sets come from a CSV (columns `layer,set`; `excluded` layers are not used) instead, e.g. the `split.csv` of `layer_correlation.py`; the patch types per set are the same (test layers also get 60x60 and 120x60), and the search and the Tarbert-only checks are not run. The CSV is copied to `layer_split_used.csv`. Run `layer_correlation.py` first to choose the layers.

| Option | Meaning (default) |
|---|---|
| `--out` | output folder, must not exist or be empty |
| `--perm-file` | required: `PERMI.inc` (one block of 1,122,000 values) or `spe_perm.dat` (three blocks); `download_spe10.py` fetches it |
| `--layer-split` | CSV `layer,set` (train / validation / test / excluded) replacing the built-in split (off); not with `--val-layers`; example for `--nk 3`: `layer_split_example.csv` |
| `--format imex\|opm` | patch file format (`imex`). `imex` = bare list of values, needed by step 3; `opm` = `PERMX` block + `COPY PERMX PERMY` |
| `--seed` | random seed (0) |
| `--dry-run` | do everything except writing the patch files |
| `--debug-npy` | also write `.npy` copies of the patches (off) |
| `--nk N` | layers per patch (1). Patches become 32x32xN etc. blocks of N consecutive layers; train / validation / test are made of whole blocks, so each set needs runs of at least N consecutive layers. With N > 1 and no `--layer-split` the split is test 31-35, validation 26-30 (or `--val-layers`), train = the rest of 1-30, and the validation search is not run (N <= 5). `--n-train` / `--n-eval` then count patches per block; `manifest.csv` has `layer` (first) and `layer_end` |
| `--train-stride S` | train blocks start every S layers inside a run of train layers (N; 1 = sliding blocks that share layers) |
| `--n-train` | 32x32 patches cut from each train layer (14) |
| `--n-eval` | 32x32 patches cut from each validation and test layer (6) |
| `--max-overlap` | largest overlap between two train patches of one layer, as a fraction of the patch area (0.5) |
| `--max-corr-val` | validation search: largest correlation of log10(Kx) between a train and a validation layer (0.5) |
| `--max-corr-test` | validation search: largest correlation between a train and a test layer (the value of train 1-25 / validation 26-30, 0.628) |
| `--std-tol` | validation search: the spread of log10(Kx) of the validation layers may differ from train by less than this fraction (0.2) |
| `--val-layers 26,27,...` | skip the search and use these validation layers |
| `--rotation-k` | the 120x60 patches are cut as 60x120 and rotated with `numpy.rot90(k)`; k=1 is counterclockwise (1) |
| `--avoid-colocated R` | move a validation/test 32x32 patch sideways (up to 50 tries) if its correlation with the same window in a train layer exceeds R (off; the correlations are always reported) |

### Step 2: well layouts and BHP schedules

```bash
python generate_well_layouts.py --patches $DATA/patches --out $DATA/layouts --jobs 8
python generate_well_layouts.py --patches $DATA/patches --out $DATA/l --sets test --well-counts 2/1,8/4 --max-patches 3   # a subset
python bhp_schedule.py --well-locations $DATA/layouts/well_locations.csv --out $DATA/layouts/bhp_schedules.csv
```

**Layouts** (`generate_well_layouts.py`; the logic is explained in [WELL_LAYOUT.md](WELL_LAYOUT.md)). For each patch and each well count (P producers / I injectors):
1. All wells are first placed as producers by simulated annealing, maximising the discounted volume each well drains (`Y_at`:
   volume far from a well counts less), with a balance penalty so that wells drain similar volumes. Wells are never closer
   than `--d-min`.
2. Roles. With `--k-min` the wells fall into flow regions (cells connected through I/J neighbours at or above `--k-min`).
   Every region gets one injector first, at its lowest-`Y_at` position (a region always keeps a producer). Each further
   injector goes to the region with the lowest #injectors/#producers ratio, at its lowest-`Y_at` position that is at least
   `--inj-factor` x `--d-min` from the other injectors and not behind an injector (an injector nearer to it and within 30
   degrees of the line to a producer of the region); the distance rule, then the shadow rule, is dropped if no position
   qualifies (flagged in `inj_spacing_relaxed` and `inj_behind_injector`). Without `--k-min` all wells form one region.
   A layout where some region cannot get both a producer and an injector is invalid (re-annealed, see `--max-attempts`).
   The other positions are producers (P1 has the highest `Y_at`).
3. If the wells do not fit with `--d-min`, wells are removed one at a time, a producer first, then an injector, and so on (at least
   1P and 1I); injectors only go below 2 when there are 2 producers (4P2I -> 3P2I -> 2P2I -> 2P1I -> 1P1I, never 4P1I). A revised count equal to a requested count (or to one already made for the patch) is skipped, so a patch never
   has two layouts with the same well count. The reasons are in `skipped_cases.csv`. "Room" means what a greedy placement fits.

Output: `well_locations.csv` (one row per case; unused wells are 0; `reduced` marks revised counts), `skipped_cases.csv`,
`config_used.yaml`, `grids/` (CORNERS include and base deck per grid size, shared by all patches).

| Option | Meaning (default) |
|---|---|
| `--patches`, `--out` | step 1 output folder; new output folder |
| `--well-counts` | `P/I` pairs, comma separated (`2/1,3/2,4/2,5/3,6/4,7/4,8/4`) |
| `--test-well-counts` | extra `P/I` pairs for the test set only (`1/0`: one producer, no injector; empty string = none) |
| `--sets` | sets to process (`train,validation,test`) |
| `--families` | only patches whose id matches this regex, e.g. `"L3[1-3]_32x32"` |
| `--max-patches` | only the first N selected patches (for tests) |
| `--seed` | random seed (0) |
| `--d-min` | minimum distance between any two wells, metres (500) |
| `--inj-factor` | preferred minimum injector-injector distance = factor x `--d-min` (1.5) |
| `--test-report-dates` | same as `--report-dates`, applied automatically to the decks of the test set (`daily`). Train and validation decks use `--report-dates` |
| `--sets` | write decks only for these sets (`train,validation,test`) |
| `--cell-size DX DY DZ` | cell size in metres (50 50 5); use the same in step 3 |
| `--n-iter` | annealing iterations per layout; more is better but slower (2000) |
| `--rate`, `--x-base` | discount of `Y_at`: a rate of `--rate` per `--x-base` equivalent metres of distance (0.10, 15.0) |
| `--k-min` | minimum cell permeability, mD; a well below it scores Y_at = 0 while annealing and makes the layout invalid (0 = off). With it on, cells below it are also barriers: a well with no I/J-connected path to another well scores 0 and invalidates the layout; wells are only placed on cells >= it |
| `--score-min-frac` | a layout is invalid if a well's Y_at is below this fraction of the mean Y_at of the other wells (0 = off) |
| `--max-attempts` | annealing runs (seed, seed+1, ...) tried for a valid layout before that well count goes to `skipped_cases.csv` (3) |
| `--w-bal` | weight of the penalty for wells that drain very different volumes (0.25) |
| `--jobs` | patches processed in parallel (1) |

**BHP schedules** (`bhp_schedule.py`). Values in kgf/cm2, uniform random, 1 decimal, reproducible for a seed: producers 180-260,
injectors 500-550. Two schedules for every case:
- `fixed`: one value per well, kept for the whole run (set on the first control date).
- `variable`: a new value per well on each of the 15 control dates (2020-01-05, then every 4 months on day 2 until
  2024-09-02). The run goes from 2020-01-01 to 2025-01-01.

Which cases get which schedule: every case gets `variable`; `fixed` is added only for the sets in `--fixed-sets` (default `test`). A case without injectors (`1P0I`) gets `fixed` only. So the training data uses variable BHP only, and the fixed-BHP and single-producer decks exist only for testing.

Output: long CSV with `case_id, schedule, well, date, bhp_kgfcm2`. Options: `--well-locations`, `--out`, `--seed`, `--fixed-sets` (sets that also get the fixed schedule, `test`).

### Step 3: decks (`build_decks.py`)

```bash
python build_decks.py --patches $DATA/patches --layouts $DATA/layouts --out $DATA/decks --template "$TEMPLATE"
python build_decks.py ... --out $DATA/decks_opm --template "$TEMPLATE" --format opm      # also OPM decks
python build_decks.py ... --report-dates daily                                      # a DATE every day
```

Writes one deck per case and schedule listed in `bhp_schedules.csv` (so train and validation get no fixed decks). Output: `decks/*.dat`, `inc/` (KR.inc from UNISIM-IV, PVT.inc from UNISIM-II-D,
CORNERS per grid size), `perm/` (the patches used), `decks_index.csv`. With `--format opm` each deck is also converted
(`cmg_to_opm.py`) into `opm/<deck>/`; the IMEX decks are kept. Needs the IMEX patch format from step 1 and the CSVs of step 2.

| Option | Meaning (default) |
|---|---|
| `--patches`, `--layouts`, `--out` | step 1 folder, step 2 folder (with `bhp_schedules.csv`), new output folder |
| `--template` | deck template, **required** (see below) |
| `--format imex\|opm` | `opm` also converts every deck for OPM Flow (`imex`) |
| `--schedules` | write decks only for these schedules, among those present in `bhp_schedules.csv` (`fixed,variable`) |
| `--dtmax` | `DTMAX` of the `NUMERICAL` section: the largest time step the simulator may take between two `DATE` lines, in days. Smaller is more accurate and slower; it only matters when dates are far apart (10) |
| `--report-dates` | extra `DATE` lines that carry no command but force results to be written that day (`example`). `example` = the pattern of the paper decks (185 dates), `daily` = every day (1828), `controls` = none (BHP dates only), or a file with one `YYYY-MM-DD` per line. The BHP control dates are always included, so the fixed and variable decks of a case share the same dates (`report_dates.py`) |
| `--cell-size DX DY DZ` | cell size for `CORNERS.inc`, metres (50 50 5); same as in step 2 |
| `--cache-dir` | where the UNISIM downloads (KR, PVT) are kept (`_inputs/` next to the scripts; the examples use `$DATA/inputs`) |
| `--date-step DAYS` | long-horizon decks (see below): a DATE line every DAYS days in every set; replaces `--report-dates`, `--test-report-dates` and `bhp_schedules.csv` (off) |
| `--bhp-step DAYS` | with `--date-step`, required: BHP changes every DAYS days, a multiple of `--date-step`; `0` = fixed BHP |
| `--open-step DAYS` | with `--date-step`: the `staged` schedule opens one well every DAYS days (`--bhp-step`, or `--date-step` if that is 0) |
| `--seed` | with `--date-step`: seed of the BHP values (0) |
| `--no-restart` | OPM only: skip grid results (pressure, saturation) at each date; faster, well and field results only |

#### Long-horizon decks (`--date-step`, `--bhp-step`; `time_grid.py`)

For models that predict long time spans. `--date-step 10 --bhp-step 60`: DATE lines every 10 days (all sets, same dates), wells
shut until the first DATE (day 10), where they open; the BHP of every well changes every 6th DATE line (days 10, 70, 130, ...),
never on the end date (2025-01-01). Step 2's `bhp_schedules.csv` is not read: the BHP values are drawn in step 3 (same ranges, `--seed`).
Schedules: train and validation = `variable`; test = `variable`, `fixed` (constant BHP) and `staged`; `1P0I` = `fixed` only.
`staged` opens one well per opening date (`--open-step`, default `--bhp-step`, or `--date-step` if `--bhp-step` is 0): P1, the
injector closest to the open producers, P2, the next closest injector, ... (when producers or injectors run out the other kind
follows) until all wells are open. A well gets its BHP on its opening date (IMEX opens a shut well when it gets a BHP) and, if
`--bhp-step` > 0, a new one on every later control date; with `--bhp-step 0` only the newly opened well gets a command.
It needs at least as many opening dates as wells (otherwise an error). With `--bhp-step 0` there is no `variable`: train and
validation get `fixed`, test gets `fixed` and `staged`. `--schedules` still filters (`fixed,variable,staged`).

```bash
python build_decks.py --patches $DATA/patches --layouts $DATA/layouts --out $DATA/decks --template "$TEMPLATE" --date-step 10 --bhp-step 60
```

### Template

The template is an input with no default: copy `templates/spe10_2d.tpl`, edit it and pass it with `--template`. It is IMEX
text with `{name}` placeholders and `#` directive lines (`template_engine.py`):
- `#for NAME` ... `#end` writes the block once per item of `NAME`; `#if NAME` ... `#end` writes it only if `NAME` is true.
- Lists: `producers`, `injectors` (items have `well i j bhp_limit`), `dates` (items have `date` and the list `controls`), and
  inside a date `controls` (items have `well bhp`). One producer block and one injector block serve any well count.
- Scalars: `case_id schedule n_prod n_inj ni nj corners_file perm_file pvt_file kr_file dtmax start_date end_date`.
- An undefined name is an error, so a typo never passes silently.


## Other scripts

Utilities that are not part of the build steps.

### Plot permeability and wells (`plot_decks.py`)

Plots Kx (log colour scale) and the well positions of cases, one PNG per `case_id`. The input is either `decks_index.csv` (step 3) or `well_locations.csv` (step 2); the script tells them apart by the header.

```bash
python plot_decks.py $DATA/decks/decks_index.csv L33_120x60_01_5P3I            # one case
python plot_decks.py $DATA/decks/decks_index.csv L36_32x32_01 --out plots       # all cases of a patch
python plot_decks.py $DATA/layouts/well_locations.csv L36_32x32_01 L01_32x32_00_2   # before any deck exists
```

- Each name is a `case_id` or the start of one: `L36_32x32_01` plots `L36_32x32_01_3P2I`, `L36_32x32_01_4P2I`, ... A name that matches nothing is an error.
- `case_id` is the key for both inputs: a case has the same map and wells under its `fixed` and `variable` schedules, so `decks_index.csv` also gives one PNG per case, not per deck.
- `--out`: output folder (`deck_plots`, created if needed). Files are `<out>/<case_id>.png`; existing ones are overwritten.
- `decks_index.csv`: wells and permeability are read from the first deck of the case (`decks/` or `decks_test/` next to the index), so the plot shows what the simulator uses.
- `well_locations.csv`: wells come from the table (zeros = unused) and the permeability from `<patches>/perm/<family_id>.inc`. `--patches` is the step 1 folder (default `<csv folder>/../patches`).
- Map: colour = Kx in md on a log scale (colour bar in md, not log10); red down-triangles = producers, white up-triangles = injectors, labelled with the well name; axes are 1-based cell indices (i, j).

### Layer correlation and split for any layer range (`layer_correlation.py`)

Run it before step 1 when the patches should come from other layers than the built-in Tarbert split; pass its `split.csv` to `generate_spe10_patches.py --layer-split`. It does not assume SPE10: you give the grid size and the layer range.

```bash
python layer_correlation.py --out $DATA/layer_corr --perm-file $DATA/inputs/spe_perm.dat \
    --ni 60 --nj 220 --nk 85 --k-init 36 --k-end 70 --window 32
```

`--ni --nj --nk --k-init --k-end` are required (layers are 1-based, `--k-end` included); `--window` (patch size) is optional and only switches on the co-location report. Other options: `--n-val`, `--n-test` (4, 4), `--max-corr` (0.3), `--min-train`, `--mean-tol`, `--std-tol`, `--n-samples`, `--seed`.

Writes `layer_corr.csv/.png` (layer x layer correlation of log10(Kx)), `corr_vs_lag.csv/.png`, `split.csv` (set per layer: train / validation / test / excluded), `split_candidates.csv` (best 50 splits), `colocation_corr.csv` (same window, eval layer vs train layers) and `summary.md`. The split is a seeded random search: no train layer may correlate above `--max-corr` with a validation or test layer (train layers that would are `excluded`), and the eval layers must have a similar log10(Kx) mean and spread as train.


## Files

| File | Purpose |
|---|---|
| `download_spe10.py`, `layer_correlation.py` | step 0: download SPE10, KR and PVT; layer correlations and layer split |
| `generate_spe10_patches.py`, `perm_io.py`, `layer_split.py`, `patch_layout.py` | step 1: CLI and checks; download, read, write; validation search; patch positions |
| `generate_well_layouts.py`, `well_layout.py`, `well_placement/`, `grid_geometry.py` | step 2: CLI; roles and well counts; placement code (annealing, distances); CORNERS writer |
| `bhp_schedule.py` | step 2: BHP schedules |
| `build_decks.py`, `template_engine.py`, `report_dates.py`, `time_grid.py`, `fluid_files.py`, `templates/` | step 3: decks; template syntax; report dates; regular DATE grid, BHP steps, staged schedule; KR and PVT files; example template |
| `cmg_to_opm.py`, `run_opm_batch.py`, `compare_cmg_opm.py` | IMEX -> OPM conversion; run and compare |
| `plot_decks.py` | plot permeability and wells of decks |
| `cmg_to_darts.py`, `run_darts_batch.py`, `DARTS.md` | IMEX deck -> open-DARTS model and run; batch run and compare; notes |
| `defaults.py` | default seed |

