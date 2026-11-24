"""Convert IMEX results (``.sr3`` + ``.out`` + ``.dat``) into raw folders, split in train/val/test.

Output layout (one folder per simulation, see `resflow.preprocess`)::

    <output_dir>/{train,val,test}/raw/<deck>/model.json, res_static.pt, ...

The split of each deck comes from a decks index CSV (columns ``deck`` and ``set`` with
train / validation / test).

Copyright (c) Tiago Amorim. MIT License.
"""
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from tqdm import tqdm

from ..io.cmg.sr3reader import Sr3Reader
from ..preprocess.model import (
    WELLS_OPEN_DAY, build_model_data, get_output_selection)
from ..preprocess.reservoir import build_static_grid_data, build_dynamic_grid_data
from ..preprocess.connection import build_static_con_data, build_dynamic_con_data
from ..preprocess.well import build_static_well_data, build_dynamic_well_data

SPLIT_DIRS = ('train', 'val', 'test')
# Values of the index CSV's split column -> split folder of the dataset root.
SPLIT_ALIASES = {'train': 'train', 'validation': 'val', 'val': 'val', 'test': 'test'}


def convert_sr3_to_files(
        sr3_path: Union[str, Path],
        output_dir: Union[str, Path],
        step_size: float = 1.0,
        verbose: Optional[bool] = False,
        wells_open_day: float = WELLS_OPEN_DAY,
        errors: Optional[list] = None,
    ) -> bool:
    """Convert one SR3 simulation to the raw folder `output_dir`.

    Returns False (and removes `output_dir`) when the run is discarded or cannot be
    converted; the reason is printed and appended to `errors` as (name, reason) if given.

    Failed-run rule: a run whose last output is at or before `wells_open_day` (the wells of
    the decks open at day 4) never opened its wells and is discarded; a run that ended
    later is kept up to the last output it reached.

    Args:
        sr3_path: the ``.sr3`` file (its ``.out`` and ``.dat`` files must be next to it).
        output_dir: folder to create for this simulation.
        step_size: days between the outputs used (1 for daily decks, 10 for 10-day decks).
    """
    sr3_path = Path(sr3_path)
    output_dir = Path(output_dir)

    def _fail(reason):
        print(f"Skipping {sr3_path.name}: {reason}")
        if errors is not None:
            errors.append((sr3_path.stem, reason))
        shutil.rmtree(output_dir, ignore_errors=True)
        print('Output directory removed.\n')
        return False

    try:
        _check_files(sr3_path)
    except (FileNotFoundError, ValueError) as e:
        return _fail(str(e))

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        sr3 = Sr3Reader(sr3_path)
        build_model_data(sr3, output_dir, step_size=step_size, verbose=verbose,
                         wells_open_day=wells_open_day)
        build_static_grid_data(sr3, output_dir, verbose=verbose)
        keep_indices, _ = get_output_selection(sr3, step_size)
        build_dynamic_grid_data(sr3, output_dir, verbose=verbose, keep_indices=keep_indices)
        build_static_con_data(sr3, output_dir, verbose=verbose)
        build_dynamic_con_data(sr3, output_dir, verbose=verbose, keep_indices=keep_indices)
        build_static_well_data(sr3, output_dir, verbose=verbose)
        build_dynamic_well_data(sr3, output_dir, verbose=verbose, keep_indices=keep_indices)
    except ValueError as e:
        return _fail(str(e))
    return True


def _check_files(sr3_path: Path) -> None:
    """Check that the SR3, OUT and DAT files exist."""
    for path in (sr3_path, sr3_path.with_suffix('.out'), sr3_path.with_suffix('.dat')):
        if not path.is_file():
            raise FileNotFoundError(f"file {path.name} does not exist in {path.parent}.")


def read_split_index(csv_path, split_column: str = 'set', name_column: str = 'deck') -> dict:
    """Read a decks index CSV into {deck name: split folder ('train'|'val'|'test')}.

    The CSV needs one row per simulation, with `name_column` holding the deck name (the
    file stem) and `split_column` holding train / validation (or val) / test. Shared by
    all converters (IMEX, OPM).
    """
    df = pd.read_csv(csv_path)
    for col in (name_column, split_column):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}. Columns: {list(df.columns)}")
    if df[name_column].duplicated().any():
        dups = df.loc[df[name_column].duplicated(), name_column].tolist()
        raise ValueError(f"Duplicated deck names in {csv_path}: {dups[:5]}...")
    unknown = sorted(set(df[split_column].astype(str).str.strip().str.lower()) - set(SPLIT_ALIASES))
    if unknown:
        raise ValueError(
            f"Unknown values in column '{split_column}' of {csv_path}: {unknown}. "
            f"Expected one of {sorted(SPLIT_ALIASES)}.")
    return {
        str(name): SPLIT_ALIASES[str(split).strip().lower()]
        for name, split in zip(df[name_column], df[split_column])
    }


def prepare_output_dir(output_dir: Union[str, Path], resume: bool = False,
                       overwrite: bool = False) -> Path:
    """Create the output directory of a conversion.

    An existing directory that already holds split folders is an error unless `resume`
    (keep it, skip simulations already converted) or `overwrite` (delete the split folders
    ``train/ val/ test/`` and ``processing.log`` inside it; nothing else is touched).
    """
    output_dir = Path(output_dir)
    existing = [output_dir / s for s in SPLIT_DIRS if (output_dir / s).exists()]
    if existing and not resume:
        if not overwrite:
            raise FileExistsError(
                f"{output_dir} already holds converted data ({', '.join(p.name for p in existing)}). "
                "Use resume to skip the simulations already converted, or overwrite to redo it.")
        for p in existing:
            shutil.rmtree(p)
        (output_dir / 'processing.log').unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def already_converted(output_dir: Path, stem: str) -> bool:
    """Whether a simulation already has a raw folder in any split of `output_dir`."""
    return any((output_dir / split / 'raw' / stem).exists() for split in SPLIT_DIRS)


def organize_sr3_dataset(
        folder: Union[str, Path],
        output_dir: Union[str, Path],
        split_csv: Union[str, Path],
        step_size: float = 1.0,
        resume: bool = False,
        overwrite: bool = False,
        split_column: str = 'set',
        name_column: str = 'deck',
        wells_open_day: float = WELLS_OPEN_DAY,
    ) -> List[str]:
    """Convert the IMEX simulations of `folder` into train/val/test raw folders.

    Each SR3 file listed in `split_csv` goes to ``<output_dir>/<split>/raw/<deck>``; SR3
    files not in the index are skipped. Progress and per-simulation details are written to
    ``<output_dir>/processing.log``. Discarded runs (see convert_sr3_to_files) are listed
    in the log and in the returned list of error messages.

    Args:
        folder: folder with the ``.sr3`` (+ ``.out``, ``.dat``) files.
        output_dir: where the raw folders are written.
        split_csv: decks index CSV (see read_split_index).
        step_size: days between the outputs used (1 daily decks, 10 for 10-day decks).
        resume: keep existing output and skip the simulations already converted.
        overwrite: delete the existing split folders of output_dir first.
        wells_open_day: runs ending at or before this day are discarded.

    Returns:
        Messages ``"<deck>: <reason>"`` of the simulations that were not converted.
    Raises:
        RuntimeError if no simulation could be converted (with the first reason).
    """
    output_dir = prepare_output_dir(output_dir, resume=resume, overwrite=overwrite)
    log_path = output_dir / 'processing.log'
    print(f"Processing log will be written to {log_path}")

    split_of = {f"{k}.sr3": v for k, v in read_split_index(split_csv, split_column, name_column).items()}
    folder = Path(folder)
    files = sorted(folder.glob('*.sr3'))
    present = {file.name for file in files}
    missing_ = sorted(n for n in split_of if n not in present)
    files = [file for file in files if file.name in split_of]

    skipped_ = []
    if resume:
        pending = []
        for file in files:
            (skipped_ if already_converted(output_dir, file.stem) else pending).append(file)
        skipped_ = [file.stem for file in skipped_]
        files = pending

    errors: list = []
    n_ok = 0
    with open(log_path, 'a' if resume else 'w', buffering=1, encoding='utf-8') as log_file, \
            redirect_stdout(log_file):
        print(f"Found {len(files) + len(skipped_):,} SR3 files in {folder.name}")
        counts = pd.Series([split_of[file.name] for file in files]).value_counts().to_dict()
        print(f"Split from {Path(split_csv).name}: {counts}; {len(missing_):,} indexed file(s) "
              "without an SR3 file.")
        print(f"Step size: {step_size:g} days; runs ending at day <= {wells_open_day:g} are discarded.")
        if resume:
            print(f"Skipping {len(skipped_):,} already-converted file(s); "
                  f"{len(files):,} remaining.")

        with tqdm(files, unit="model", file=sys.stderr) as pbar:
            for file in pbar:
                pbar.set_description(file.stem)
                print("\n" + 70 * "=" + f'\nProcessing {file.name}' + "\n" + 70 * "=")
                model_dir = output_dir / split_of[file.name] / 'raw' / file.stem
                if convert_sr3_to_files(
                        sr3_path=file, output_dir=model_dir, step_size=step_size,
                        verbose=True, wells_open_day=wells_open_day, errors=errors):
                    n_ok += 1

        if missing_:
            print(f"\n{len(missing_)} indexed deck(s) have no SR3 file in {folder.name}:")
            for m in missing_:
                print(f" - {m}")
        if errors:
            print(f"\n{len(errors)} file(s) not converted:")
            for stem, reason in errors:
                print(f" - {stem}: {reason}")

    messages = [f"{stem}: {reason}" for stem, reason in errors]
    print(f"Log written to {log_path}")
    print(f"Converted {n_ok} simulation(s).")
    if missing_:
        print(f"{len(missing_)} indexed deck(s) have no SR3 file (see log).")
    if skipped_:
        print(f"{len(skipped_)} file(s) skipped (already converted).")
    if messages:
        print(f"{len(messages)} file(s) not converted (see log): "
              f"{', '.join(stem for stem, _ in errors)}")
    if n_ok == 0 and messages:
        raise RuntimeError(f"No simulation could be converted. First failure: {messages[0]}")
    return messages
