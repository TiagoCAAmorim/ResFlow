"""Command line interface: ``resflow <command> ...``.

Commands are registered with `register_command`; each has its own module with an
``add_arguments(parser)`` and a ``run(args)``. Other modules (training, rollout, ...)
add theirs by calling `register_command` at import and being listed in
`OPTIONAL_COMMAND_MODULES`: a module that is not installed is skipped.

Commands defined here:

- ``convert``: simulation results -> raw folders split in train/val/test.
- ``build``: raw folders -> processed graph dataset.

Nothing is written outside the directories given on the command line.

Copyright (c) Tiago Amorim. MIT License.
"""
import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional

# Modules that register more commands when imported (skipped if missing).
OPTIONAL_COMMAND_MODULES = ["resflow.train.cli", "resflow.rollout.cli"]

# Converters by simulator: name -> module that holds `organize_<name>_dataset`.
FORMATS = ("imex", "opm")


class Command(NamedTuple):
    """A registered command."""
    help: str
    add_arguments: Callable[[argparse.ArgumentParser], None]
    run: Callable[[argparse.Namespace], Optional[int]]


COMMANDS: Dict[str, Command] = {}


def register_command(name: str, help: str, add_arguments, run) -> None:  # pylint: disable=redefined-builtin
    """Register the command `name`: `add_arguments(parser)` defines its options and
    `run(args)` runs it (returning an exit code or None)."""
    COMMANDS[name] = Command(help, add_arguments, run)


# MARK: convert
def _load_converter(fmt: str) -> Callable:
    """The ``organize_<fmt>_dataset`` function of a simulator format.

    A converter has the signature ``(folder, output_dir, split_csv, step_size, resume,
    overwrite, split_column, name_column, wells_open_day) -> list of failure messages`` and
    writes the raw folders as `resflow.data.convert.organize_sr3_dataset` does. The OPM
    converter lives in ``resflow.io.opm`` (function ``organize_opm_dataset``).
    """
    if fmt == "imex":
        from .data.convert import organize_sr3_dataset  # pylint: disable=import-outside-toplevel
        return organize_sr3_dataset
    if fmt == "opm":
        try:
            module = importlib.import_module("resflow.io.opm")
            return module.organize_opm_dataset
        except (ImportError, AttributeError) as exc:
            raise SystemExit(
                "The OPM converter is not available in this installation "
                f"(resflow.io.opm.organize_opm_dataset): {exc}") from exc
    raise SystemExit(f"Unknown format '{fmt}'. Choose from {FORMATS}.")


def _add_convert_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--format", choices=FORMATS, default="imex",
                   help="simulator that made the results (default: imex, the paper's data)")
    p.add_argument("--input", required=True, type=Path,
                   help="folder with the simulation results (IMEX: .sr3, .out and .dat files)")
    p.add_argument("--index", required=True, type=Path,
                   help="decks index CSV with the columns 'deck' and 'set' (train/validation/test)")
    p.add_argument("--output", required=True, type=Path,
                   help="folder to write the raw data to (train/, val/ and test/ inside)")
    p.add_argument("--step-size", type=float, default=1.0,
                   help="days between the outputs used: 1 for daily decks (default), "
                        "10 for the 10-day decks")
    p.add_argument("--wells-open-day", type=float, default=4.0,
                   help="a run that ends at or before this day never opened its wells and is "
                        "discarded (default: 4)")
    p.add_argument("--split-column", default="set", help="index column with the split")
    p.add_argument("--name-column", default="deck", help="index column with the deck name")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true",
                       help="keep the existing output and skip the simulations already converted")
    group.add_argument("--overwrite", action="store_true",
                       help="delete the existing train/ val/ test/ folders of --output first")


def _run_convert(args: argparse.Namespace) -> int:
    converter = _load_converter(args.format)
    messages = converter(
        folder=args.input, output_dir=args.output, split_csv=args.index,
        step_size=args.step_size, resume=args.resume, overwrite=args.overwrite,
        split_column=args.split_column, name_column=args.name_column,
        wells_open_day=args.wells_open_day)
    for msg in messages:
        print(f"not converted: {msg}", file=sys.stderr)
    return 0


# MARK: build
def _add_build_arguments(p: argparse.ArgumentParser) -> None:
    from .data.dataset import SAMPLES_RATIO  # pylint: disable=import-outside-toplevel
    p.add_argument("--root", required=True, type=Path,
                   help="dataset root with <split>/raw/<simulation>/ folders (output of convert); "
                        "the processed data is written next to them")
    p.add_argument("--step-size", type=float, default=1.0,
                   help="days between outputs, as used by convert: 1 (default) or 10")
    p.add_argument("--samples-ratio", type=float, default=SAMPLES_RATIO,
                   help="fraction of the cell-step samples drawn for training by the histogram "
                        f"resampling (default: {SAMPLES_RATIO}, the paper's value)")
    p.add_argument("--min-day", type=float, default=None,
                   help="only process steps from this day (default: all). Training windows "
                        "(0-1000 days in the paper) are chosen when the dataset is read, not here")
    p.add_argument("--max-day", type=float, default=None,
                   help="only process steps up to this day (default: all)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed of the random resampling (default: 0)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted build: simulations already processed are skipped")
    p.add_argument("--log", type=Path, default=None,
                   help="progress log (default: <root>/build.log)")


def _run_build(args: argparse.Namespace) -> int:
    from .data.dataset import build_dataset  # pylint: disable=import-outside-toplevel
    window = {}
    if args.min_day is not None:
        window["min_day"] = args.min_day
    if args.max_day is not None:
        window["max_day"] = args.max_day
    built = build_dataset(
        root=args.root, samples_ratio=args.samples_ratio, step_size=args.step_size,
        resume=args.resume, seed=args.seed, log_path=args.log, **window)
    for split, ds in built.items():
        print(f"{split}: {len(ds)} processed files, {ds.get_n_samples():,} one-step samples")
    return 0


register_command("convert", "simulation results -> raw folders (train/val/test)",
                 _add_convert_arguments, _run_convert)
register_command("build", "raw folders -> processed graph dataset",
                 _add_build_arguments, _run_build)


# MARK: main
def build_parser() -> argparse.ArgumentParser:
    """The argument parser with every registered command."""
    for module in OPTIONAL_COMMAND_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name is None or not module.startswith(exc.name):
                raise
    parser = argparse.ArgumentParser(prog="resflow", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, cmd in COMMANDS.items():
        p = sub.add_parser(name, help=cmd.help, description=cmd.help)
        cmd.add_arguments(p)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of the ``resflow`` command."""
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command].run(args) or 0


if __name__ == "__main__":
    # Run through the importable module: the optional command modules register into
    # resflow.cli.COMMANDS, which is not this __main__ copy.
    from resflow.cli import main as _main  # pylint: disable=import-self
    sys.exit(_main())
