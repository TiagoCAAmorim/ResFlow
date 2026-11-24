"""Training commands: ``fit-norm``, ``train-state``, ``train-rate``, ``finetune``.

Examples::

    resflow fit-norm --dataset data/spe10/datasets
    resflow train-state --config configs/state_2d_1d.yaml
    resflow train-state --resume runs/state_2d_1d          # continue an interrupted run
    resflow train-rate --config configs/rate_2d_1d.yaml
    resflow finetune --weights weights/state_2d_1d --config configs/finetune_fluvial.yaml

Config values can be overridden: ``--set dataset=/data/spe10 train.epochs=2``.

Copyright (c) Tiago Amorim. MIT License.
"""
import argparse
from pathlib import Path

from ..cli import register_command


def _common(p: argparse.ArgumentParser, config_required: bool) -> None:
    p.add_argument('--config', type=Path, required=config_required, help='training config (YAML)')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override config entries, e.g. dataset=data/spe10 train.epochs=1')
    p.add_argument('--run-dir', type=Path, default=None, help='output folder (overrides run_dir)')
    p.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')


def _load(args, required=True) -> dict:
    from .common import apply_overrides, load_config  # pylint: disable=import-outside-toplevel
    cfg = load_config(args.config, args.set) if args.config else apply_overrides({}, args.set)
    if args.run_dir is not None:
        cfg['run_dir'] = str(args.run_dir)
    if required:
        for key in ('dataset', 'run_dir'):
            if key not in cfg:
                raise SystemExit(f"The config needs '{key}' (or --set {key}=...).")
    return cfg


# fit-norm
def _add_fit_norm(p):
    p.add_argument('--dataset', required=True, type=Path, help='dataset root (with train/)')
    p.add_argument('--out', type=Path, default=None,
                   help='output file (default: <dataset>/analysis/stats/lin_scaler.pt)')
    p.add_argument('--max-samples', type=int, default=1000, help='stored series read (default 1000)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)


def _run_fit_norm(args):
    from .norm import fit_norm  # pylint: disable=import-outside-toplevel
    path = fit_norm(args.dataset, args.out, max_samples=args.max_samples, seed=args.seed,
                    device=args.device)
    print(f"Statistics written to {path}")
    return 0


# train-state
def _add_train_state(p):
    _common(p, config_required=False)
    p.add_argument('--resume', type=Path, default=None, help='run folder to continue')


def _run_train_state(args):
    from .state_trainer import train_state  # pylint: disable=import-outside-toplevel
    if args.resume is None and args.config is None:
        raise SystemExit("Give --config (new run) or --resume RUN_DIR.")
    cfg = _load(args, required=args.resume is None)
    train_state(cfg, resume=args.resume, device=args.device)
    return 0


# train-rate
def _add_train_rate(p):
    _common(p, config_required=False)
    p.add_argument('--resume', type=Path, default=None, help='run folder to continue')


def _run_train_rate(args):
    from .rate_trainer import train_rate  # pylint: disable=import-outside-toplevel
    if args.resume is None and args.config is None:
        raise SystemExit("Give --config (new run) or --resume RUN_DIR.")
    cfg = _load(args, required=args.resume is None)
    train_rate(cfg, resume=args.resume, device=args.device)
    return 0


# finetune
def _add_finetune(p):
    _common(p, config_required=True)
    p.add_argument('--weights', required=True, type=Path,
                   help='state model bundle to start from (its normalisation is kept)')


def _run_finetune(args):
    from .state_trainer import train_state  # pylint: disable=import-outside-toplevel
    cfg = _load(args)
    train_state(cfg, init_weights=args.weights, device=args.device)
    return 0


register_command('fit-norm', 'normalisation statistics of a training set (new trainings only)',
                 _add_fit_norm, _run_fit_norm)
register_command('train-state', 'train the state model (or --resume a run)', _add_train_state,
                 _run_train_state)
register_command('train-rate', 'train the rate model (or --resume a run)', _add_train_rate,
                 _run_train_rate)
register_command('finetune', 'fine-tune a state model bundle on another dataset', _add_finetune,
                 _run_finetune)
