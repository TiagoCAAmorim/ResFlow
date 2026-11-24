# 4. Training data from OPM Flow (no IMEX licence needed)

`resflow.io.opm` turns the results of an OPM Flow run into the same raw folders as the IMEX converter
(`model.json`, `res_static.pt`, `res_dynamic/{pres,so,sg,rs}.pth`, `well_*`/`con_*` files, `pvt01.pth`,
`krel01.{wo,go}.pth`; same layout, dtypes and units). From there `resflow build`, training and rollout are unchanged.

## 1. Run the decks with OPM

Build the IMEX decks (`docs/02_build_decks.md`; you do not need to run IMEX), install the simulator (`pip install opm
opm-simulators`: Linux x86_64 wheels only, CPython 3.8-3.14, Windows users use WSL; or the `flow` binary, 2026.04 or newer,
`--engine flow`), then convert the decks to OPM decks and run them:

    python deck_builder/run_opm_batch.py <decks folder> -j 8 [--timeout 3600] [--engine wheel|flow]   # writes <decks folder>_opm/<deck>/out/

or one deck by hand (see `docs/03_run_simulators.md`; the wheel writes next to the `.DATA`, move the results to `out/`):

    python deck_builder/cmg_to_opm.py <deck>.dat -o <case dir>
    cd <case dir> && flow <DECK>.DATA --output-dir=out

The OPM deck requests the per-connection vectors `CTFAC COPT CWPT CWIT CGPT CPR` and the restart file
(`RPTRST BASIC=2`), and forbids crossflow: a well whose flow would reverse is shut until its next control.
(`run_opm_batch.py` also tries to compare with an `.sr3`; without one it reports `NO_SR3`, the results are written anyway.)

## 2. Convert

    resflow convert --format opm --input <decks folder> --index <decks_index.csv> --output <dir> [--step-size 10]

The input is the folder with the IMEX `<deck>.dat` files (the PVT and KR tables, the well order and the BHP depths
are read from them, OPM does not output them); the OPM cases are expected in `<decks folder>_opm/<deck>/`. From Python:

    from resflow.io.opm import convert_opm_run, organize_opm_dataset
    convert_opm_run(case_dir, deck_dat, out_dir, step_size=1.0)         # one run -> raw folder
    organize_opm_dataset(decks, out, index_csv, opm_dir=cases_dir)      # whole folder -> train/val/test

Rules, as for IMEX: a run that ended at or before day 4 (the wells never opened) or in which no well produced or
injected is discarded and listed in `processing.log`; otherwise the run is kept up to the last day reached.
OPM completed every deck of our test subset that IMEX stopped early (at the first well opening, or later), so a dataset
built with OPM has a few more simulations than the IMEX one of the paper.
Samples are the restart (report) steps of OPM, grouped in series exactly `--step-size` days apart. For the paper
decks (185 `DATE` lines for train/validation, daily for test) this gives the same days as the IMEX results of the
same decks (checked on 37 decks): no extra report dates are needed. A custom `DATE` pattern works too, as long as the
steps you want to learn from are `--step-size` days apart.

## 3. Units and differences with the IMEX raw folders

| Quantity | Source in OPM | Conversion |
|---|---|---|
| pressure, BHP | `PRESSURE`, `WBHP`, `CPR` (bar) | / 0.980665 -> kgf/cm2 |
| volumes | `COPT CWPT CGPT CWIT` (sm3, cumulative) | none; well = sum over its connections |
| well index (`con_static.json`) | `CTFAC` | / 0.00852702 -> md.m |
| transmissibility | `TRANX/Y/Z` | / 0.00852702 -> md.m |
| depth, PORV, PORO, perms | `INIT` | none |

A well that OPM shut (no crossflow) shows zero rate (flat cumulatives), and its BHP is reported as the control
target, as IMEX does. IMEX writes the same quantities with crossflow allowed, so such wells keep producing or injecting
a little there.

Caveats: the shipped weights were trained on IMEX results. OPM and IMEX differ at simulator level (about 0.5-2 %
on cumulative volumes per well, larger relative differences for the tiny water production; pressures differ by up to
a few kgf/cm2 where a well changes control). IMEX writes well index with 3 significant digits, OPM gives the full
value (up to 0.3 % apart). Expect small metric changes when rolling the IMEX-trained models over OPM data, and
fine-tune on OPM data if you need the best match.

One difference matters for the imposed mode. In producers that make almost no water, OPM produces about 10 times
more water than IMEX in the first months (about 10 against 1 sm3/day for a 60x60 case; still about 0.01 % of the oil).
The PVT, relative permeability and initial saturation of the two decks are the same, and the converted volumes equal
the OPM well totals, so this is a simulator difference. The IMEX-trained state model is sensitive to these tiny
water volumes: on OPM data its imposed-mode rollouts can end at a shifted pressure level (detailed case: P MAE 2.4
against 0.4 MPa on IMEX data; with the shift removed, R2 above 0.99). The rate mode, which predicts the well
volumes itself, is not affected (P R2 at least 0.998 on the five test decks).
