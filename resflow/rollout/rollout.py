"""Autoregressive rollouts of the state model, with imposed well volumes or a rate model.

Imposed mode: at each step the state model reads the reference simulation's well volumes
of that step. Rate mode: the rate model predicts them from the current (predicted) state,
the BHP controls and the well indices, and the state model reads the prediction (BHPs are
controls, always the reference's).

Every step is rebuilt from the reference simulation (calendar, well controls); only the
state carries over. Rs and Sg are held at their initial values (constant in this data):
only P and So (hence Sw) are predicted, as in the paper. Saturations are clipped to
[0, 1] and rescaled where So + Sg > 1. No other correction is applied.

By default the rollout is deterministic: the edge flows are summed with an exact CPU sum
and torch's deterministic algorithms are on (see `set_deterministic`), so two runs give
identical results. The fast path (GPU scatter sums) is not run-to-run reproducible; small
differences grow over long rollouts.

Copyright (c) Tiago Amorim. MIT License.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from ..models.features import (STATE_NAMES, con_data_as_grid, net_well_volumes,
                               state_with_volumes)
from ..models.rate import RateModel

ROLLOUT_OUTPUTS = (1, 2)  # P and So are predicted; Rs (0) and Sg (3) are held


def set_deterministic(enabled: bool = True) -> None:
    """Turn torch's deterministic algorithms on/off (needs CUBLAS_WORKSPACE_CONFIG on GPU,
    set here if missing; it must be set before the first cuBLAS call)."""
    if enabled:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(enabled)


@dataclass
class RolloutResult:
    """Everything one rollout produced (CPU numpy arrays)."""
    sim_name: str
    days: np.ndarray                 # [n_days] days of the reference states
    true: np.ndarray                 # [n_days, N, 8] reference Rs, P, So, Sg, Sw, Vo, Vg, Vw
    pred: np.ndarray                 # [n_pred, N, 8] day 0 (true) + every predicted day
    concluded: bool
    stop_reason: Optional[str]
    stop_day: Optional[float]
    elapsed: float
    con_sum_true: np.ndarray         # [n_days, 3] net well volumes (O, G, W) of each step
    con_sum_used: np.ndarray         # [n_pred - 1, 3] the ones the state model was driven with
    step_size: float
    rate_days: List[float] = field(default_factory=list)   # rate mode: end day of each step
    rate_wells: Dict[str, Dict[int, Dict[str, list]]] = field(default_factory=dict)
    batch_all: object = None         # the reference sample (statics, well data)


def _collect_well_rates(batch, rates: torch.Tensor, bucket: dict) -> None:
    """Sum one step's per-cell predicted volumes per well."""
    for w in range(batch.wellp.shape[-2]):
        nodes = batch.conp_index[batch.wellp_index == w]
        vals = rates[nodes, :3].sum(dim=0).cpu().numpy() if nodes.numel() else np.zeros(3)
        entry = bucket['producer'].setdefault(w, {'Oil': [], 'Gas': [], 'Water': []})
        for phase, v in zip(('Oil', 'Gas', 'Water'), vals):
            entry[phase].append(float(v))
    for w in range(batch.wellw.shape[-2]):
        nodes = batch.conw_index[batch.wellw_index == w]
        val = rates[nodes, 3].sum().item() if nodes.numel() else 0.0
        bucket['water_injector'].setdefault(w, {'Water': []})['Water'].append(val)
    for w in range(batch.wellg.shape[-2]):
        bucket['gas_injector'].setdefault(w, {})


@torch.no_grad()
def rollout_simulation(state_model, dataset, sim_name: str, device='cpu',
                       rate_model: Optional[RateModel] = None, start_day: Optional[float] = None,
                       exact: bool = True, verbose: bool = False) -> RolloutResult:
    """Roll one simulation of `dataset` out from `start_day` (default: its first day).

    The days are the dataset's window (`SimDataset` min_day / max_day). As in the paper
    tool, the rollout predicts the states at days[1:-1] of the window (with max_day 1827 on
    the daily test decks: days 1 to 1825).
    """
    state_model.eval()
    state_model.exact_aggregation = exact
    if rate_model is not None:
        rate_model.eval()
        rate_model.exact_aggregation = exact
    days = dataset.get_days(sim_name=sim_name)
    if start_day is not None:
        days = [d for d in days if d >= start_day]
    if len(days) < 3:
        raise ValueError(f"{sim_name}: fewer than 3 days in the window from day {start_day}.")
    steps = np.diff(np.asarray(days, dtype=float))
    if not np.allclose(steps, steps[0]):
        raise ValueError(f"{sim_name}: days are not evenly spaced; cannot roll out.")
    step_size = float(dataset.step_size)
    batch_all = dataset.get_sim_days(sim_name=sim_name, days=days).to(device)
    transform = state_model.transform
    vpor = batch_all.stat[:, 1]
    true = state_with_volumes(transform, vpor, batch_all.x)
    con_delta, _ = con_data_as_grid(batch_all)
    con_sum_true = net_well_volumes(con_delta).sum(dim=1)

    rolling_x = batch_all.x[0].clone()
    preds = [true[0].cpu().numpy()]
    used = []
    bucket = {'producer': {}, 'water_injector': {}, 'gas_injector': {}}
    rate_days = []
    concluded, stop_reason, stop_day = True, None, None
    t0 = time.perf_counter()
    n_steps = len(days) - 2
    for i in range(n_steps):
        batch = dataset.to_single_step(batch_all, i).to(device)
        batch.x = rolling_x
        dv_con = None
        try:
            if rate_model is not None:
                rates = rate_model(batch)
                _collect_well_rates(batch, rates, bucket)
                rate_days.append(float(days[i]) + step_size)
                dv_con = RateModel.to_con_delta(rates)
                used.append(net_well_volumes(dv_con).sum(dim=0).cpu().numpy())
            else:
                used.append(con_sum_true[i].cpu().numpy())
            _, new_x = state_model.step(batch, dv_con=dv_con)
        except ValueError as exc:
            concluded, stop_reason, stop_day = False, f'model error: {exc}', float(days[i])
            used = used[:len(preds) - 1]
            rate_days = rate_days[:len(preds) - 1]
            break
        new_x = new_x.clone()
        new_x[:, 0] = batch.x[:, 0]
        new_x[:, 3] = batch.x[:, 3]
        if not torch.isfinite(new_x).all():
            concluded, stop_reason, stop_day = False, 'non-finite state (NaN/Inf)', float(days[i])
            used = used[:len(preds) - 1]
            rate_days = rate_days[:len(preds) - 1]
            break
        new_x[:, 2:] = new_x[:, 2:].clamp(min=0.0, max=1.0)
        s_sum = new_x[:, 2] + new_x[:, 3]
        over = s_sum > 1.0
        if over.any():
            new_x[over, 2:4] = new_x[over, 2:4] / s_sum[over].unsqueeze(-1)
        data_hat = state_with_volumes(transform, vpor, new_x)
        if not torch.isfinite(data_hat).all():
            concluded, stop_reason, stop_day = False, 'non-finite derived properties', float(days[i])
            used = used[:len(preds) - 1]
            rate_days = rate_days[:len(preds) - 1]
            break
        rolling_x = new_x
        preds.append(data_hat.cpu().numpy())
        if verbose and (i + 1) % 200 == 0:
            print(f"  {sim_name}: day {days[i + 1]:g} ({time.perf_counter() - t0:.0f} s)")
    if rate_model is not None:
        for kind in ('producer', 'water_injector'):
            for series in bucket[kind].values():
                for phase in series:
                    series[phase] = series[phase][:len(preds) - 1]
    return RolloutResult(
        sim_name=sim_name, days=np.asarray(days, dtype=float), true=true.cpu().numpy(),
        pred=np.stack(preds), concluded=concluded, stop_reason=stop_reason, stop_day=stop_day,
        elapsed=time.perf_counter() - t0, con_sum_true=con_sum_true.cpu().numpy(),
        con_sum_used=np.asarray(used, dtype=np.float64).reshape(-1, 3), step_size=step_size,
        rate_days=rate_days, rate_wells=bucket if rate_model is not None else {},
        batch_all=batch_all.cpu())


__all__ = ['rollout_simulation', 'RolloutResult', 'set_deterministic', 'STATE_NAMES']
