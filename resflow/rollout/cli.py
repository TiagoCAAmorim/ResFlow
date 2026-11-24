"""``resflow rollout``: roll the test simulations out with a state model (and a rate model).

Example (the paper's test rollouts, both modes)::

    resflow rollout --weights weights/state_2d_1d --dataset data/spe10/datasets --split test \\
        --out results/rollouts/imposed --max-day 1827
    resflow rollout --weights weights/state_2d_1d --rate-weights weights/rate_2d_1d \\
        --dataset data/spe10/datasets --split test --out results/rollouts/rate --max-day 1827

Copyright (c) Tiago Amorim. MIT License.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from ..cli import register_command


def _add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument('--weights', required=True, type=Path,
                   help='state model: a weights bundle or a training run folder')
    p.add_argument('--rate-weights', type=Path, default=None,
                   help='rate model bundle: rate mode (well volumes predicted from BHP); '
                        'default: imposed mode (the simulation\'s well volumes)')
    p.add_argument('--dataset', required=True, type=Path, help='dataset root (output of build)')
    p.add_argument('--split', default='test', help='split folder to roll out (default: test)')
    p.add_argument('--sims', nargs='*', default=None,
                   help='simulation names: deck names, or the run names (.sr3 for IMEX, .DATA for OPM); default: all')
    p.add_argument('--out', required=True, type=Path, help='output folder')
    p.add_argument('--min-day', type=float, default=0.0, help='first day of the window (default 0)')
    p.add_argument('--max-day', type=float, default=1827.0,
                   help='end of the window (default 1827: the daily test decks are rolled out to '
                        'day 1825)')
    p.add_argument('--start-day', type=float, default=None,
                   help='start the rollout from the true state of this day (default: --min-day)')
    p.add_argument('--report-days', type=float, nargs='*', default=[1000.0],
                   help='days of rollout_summary.csv besides the last day (default: 1000)')
    p.add_argument('--fast', action='store_true',
                   help='GPU scatter sums and non-deterministic kernels (faster, not reproducible)')
    p.add_argument('--device', default=None, help='cuda or cpu (default: cuda if available)')
    p.add_argument('--n-cells', type=int, default=32,
                   help='cells written to rollout_<sim>.csv (well cells first; default 32)')


def run_rollouts(weights, dataset_root, split='test', out=None, rate_weights=None, sims=None,
                 min_day=0.0, max_day=1827.0, start_day=None, exact=True, device=None,
                 n_cells=32, report_days=(1000.0,), verbose=True) -> pd.DataFrame:
    """Roll out the simulations of a dataset split; writes the files of
    `resflow.rollout.outputs` into `out` and returns the summary table."""
    from ..data.dataset import SimDataset  # pylint: disable=import-outside-toplevel
    from ..models import load_bundle  # pylint: disable=import-outside-toplevel
    from .outputs import write_rollout  # pylint: disable=import-outside-toplevel
    from .rollout import rollout_simulation, set_deterministic  # pylint: disable=import-outside-toplevel

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_deterministic(exact)
    state_model = load_bundle(weights, device)
    rate_model = load_bundle(rate_weights, device) if rate_weights is not None else None
    if type(state_model).__name__ != 'StateModel':
        raise SystemExit(f"{weights} is not a state model bundle.")
    if rate_model is not None and type(rate_model).__name__ != 'RateModel':
        raise SystemExit(f"{rate_weights} is not a rate model bundle.")
    split_dir = Path(dataset_root) / split
    if not (split_dir / 'processed').is_dir():
        raise SystemExit(f"No built split at {split_dir} (run `resflow build` first).")
    ds = SimDataset(root=split_dir, min_day=min_day, max_day=max_day)
    if abs(float(state_model.config.get('step_size', ds.step_size)) - ds.step_size) > 1e-9:
        print(f"Warning: the model was trained with step {state_model.config.get('step_size')} "
              f"day(s); the dataset has {ds.step_size}.")
    names = ds.get_sim_names(unique=True)
    if sims:
        def _resolve(s):  # exact name, or the deck name of an IMEX (.sr3) or OPM (.DATA, upper case) run
            for c in (s, f'{s}.sr3', f'{s.upper()}.DATA'):
                if c in names:
                    return c
            return s
        wanted = [_resolve(s) for s in sims]
        missing = [s for s in wanted if s not in names]
        if missing:
            raise SystemExit(f"Not in {split_dir}: {missing}")
        names = wanted
    mode = 'rate' if rate_model is not None else 'imposed'
    rows = []
    for name in sorted(names):
        if verbose:
            print(f"Rollout ({mode}) {name} on {device}...")
        res = rollout_simulation(state_model, ds, name, device=device, rate_model=rate_model,
                                 start_day=start_day, exact=exact, verbose=verbose)
        if out is not None:
            rows.extend(write_rollout(out, res, ds, mode=mode, n_cells=n_cells,
                                      report_days=report_days))
        if verbose:
            status = 'done' if res.concluded else f'stopped ({res.stop_reason})'
            print(f"  {status}: {res.pred.shape[0] - 1} steps in {res.elapsed:.0f} s")
    summary = pd.DataFrame(rows)
    if out is not None and len(summary):
        path = Path(out) / 'rollout_summary.csv'
        if path.is_file():
            old = pd.read_csv(path)
            old = old[~(old['sim'].isin(summary['sim']) & (old['mode'] == mode))]
            summary = pd.concat([old, summary], ignore_index=True)
        summary.to_csv(path, index=False)
    return summary


def _run(args: argparse.Namespace) -> int:
    summary = run_rollouts(args.weights, args.dataset, args.split, args.out, args.rate_weights,
                           args.sims, args.min_day, args.max_day, args.start_day,
                           exact=not args.fast, device=args.device, n_cells=args.n_cells,
                           report_days=args.report_days)
    if len(summary):
        cols = [c for c in ('sim', 'mode', 'day', 'P_r2', 'P_shifted_r2', 'Sw_r2', 'P_mae_MPa',
                            'Sw_mae', 'MBe_oil_pct_abs', 'MBe_water_pct_abs') if c in summary]
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(summary[cols].to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    return 0


register_command('rollout', 'roll a state model (and a rate model) out on a dataset split',
                 _add_arguments, _run)
