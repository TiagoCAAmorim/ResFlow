"""Rollout metrics and output files.

Files written per simulation ``<sim>`` (the SR3 name, e.g. ``L32_60x60_01_8P4I_variable.sr3``),
with the columns of the paper tool so the plotting code reads them unchanged:

- ``rollout_<sim>.npz``: days, pred, pred_days, true, var_names, ijk, stat, well_ij, wellp,
  wellw, wellg, conp/cong/conw_index, wellp/wellg/wellw_index, well_marker_ij/type/name,
  sim_model; rate mode also well_pred_days, pred_producer_rates [n, n_prod, 3] (dNp, dGp, dWp
  per step) and pred_water_injector_rates [n, n_winj].
- ``rollout_<sim>.csv``: model, day, cell, variable, True, Est for a few cells (the well cells
  and evenly spaced others) and the cumulative material-balance error rows MBe_<phase> and
  MBe_<phase>_pct (cell empty).
- ``rollout_errors_<sim>.csv``: per day and variable over all cells: mae, mse, min, p01 ... p99,
  max, mean, std (of the signed error pred - true), r2, shift, shifted_r2.
- ``rollout_thresholds_<sim>.csv``: % of cells whose |error| / max|true| (over the simulation,
  or over the day) is at or above alpha.
- rate mode: ``rollout_wells_<sim>.csv`` (per well: Oil, Gas, Water volumes per step and BHP, True
  and Est), ``rollout_cumulative_<sim>.csv`` and ``rollout_cumulative_errors_<sim>.csv``.
- ``rollout_performance.yaml``: one entry per simulation (days, steps, time, final metrics).
- ``rollout_summary.csv`` (new): one row per simulation, mode and report day with the paper
  metrics (P R2, shifted P R2, Sw R2, P MAE in MPa, Sw MAE, |MBe| oil and water %, field
  cumulative errors in rate mode).

Units: pressures in the files are kgf/cm2 (the data's unit); only rollout_summary.csv gives
the pressure MAE in MPa too.

Copyright (c) Tiago Amorim. MIT License.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from ..models.features import STATE_NAMES
from .rollout import RolloutResult

KGF_CM2_TO_MPA = 0.0980665
ERROR_THRESHOLDS = (0.1, 0.05, 0.01)
PHASES = ('Oil', 'Gas', 'Water')
_ERR_STATS = ('mse', 'min', 'p01', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'max', 'mean',
              'std', 'r2', 'shift', 'shifted_r2')
_CUMULATIVE_MIN_REL_TRUE = 1e-6
_CUMULATIVE_COMPONENTS = (
    ('Oil production', 'producer', 'wellp', 0, 'Oil'),
    ('Gas production', 'producer', 'wellp', 1, 'Gas'),
    ('Water production', 'producer', 'wellp', 2, 'Water'),
    ('Water injection', 'water_injector', 'wellw', 0, 'Water'),
)


# MARK: Metrics
def regression_metrics(true_vals, pred_vals) -> dict:
    """R2, MAE, RMSE, min/max signed error, best constant shift and the R2 after it."""
    true_vals = np.asarray(true_vals, dtype=float)
    errors = np.asarray(pred_vals, dtype=float) - true_vals
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
    shift = float(-errors.mean())
    nan = float('nan')
    return {
        'r2': float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else nan,
        'mae': float(np.abs(errors).mean()),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'min_error': float(errors.min()),
        'max_error': float(errors.max()),
        'shift': shift,
        'shifted_r2': float(1.0 - np.sum((errors + shift) ** 2) / ss_tot) if ss_tot > 1e-12 else nan,
    }


def daily_error_table(res: RolloutResult, var_names: Sequence[str] = STATE_NAMES,
                      thresholds: Sequence[float] = ERROR_THRESHOLDS):
    """Per-day error statistics (rollout_errors) and threshold exceedance (rollout_thresholds)."""
    n_pred = res.pred.shape[0]
    days = res.days[1:n_pred]
    max_abs_true = np.maximum(np.abs(res.true).max(axis=(0, 1)), 1e-12)
    err_rows, thr_rows = [], []
    for vi, vname in enumerate(var_names):
        stats = {k: [] for k in ('mae',) + _ERR_STATS}
        thr = {(d, a): [] for d in ('simulation', 'timestep') for a in thresholds}
        for k in range(1, n_pred):
            t = res.true[k, :, vi].astype(np.float64)
            col = res.pred[k, :, vi].astype(np.float64) - t
            abs_col = np.abs(col)
            ss_tot = float(np.sum((t - t.mean()) ** 2))
            ss_res = float(np.sum(col ** 2))
            mean = float(col.mean())
            pct = np.percentile(col, [1, 10, 25, 50, 75, 90, 99])
            vals = {'mae': float(abs_col.mean()), 'mse': ss_res / col.size, 'min': float(col.min()),
                    'max': float(col.max()), 'mean': mean, 'std': float(col.std()),
                    'r2': (1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float('nan'),
                    'shift': -mean,
                    'shifted_r2': ((1.0 - float(np.sum((col - mean) ** 2)) / ss_tot)
                                   if ss_tot > 1e-12 else float('nan'))}
            for name, p in zip(('p01', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99'), pct):
                vals[name] = float(p)
            for key, v in vals.items():
                stats[key].append(v)
            denoms = {'simulation': max_abs_true[vi], 'timestep': max(float(np.abs(t).max()), 1e-12)}
            for dname, den in denoms.items():
                norm = abs_col / den
                for a in thresholds:
                    thr[(dname, a)].append(100.0 * float(np.mean(norm >= a)))
        err_rows.append(pd.DataFrame({'model': res.sim_name, 'day': days, 'variable': vname,
                                      **{k: stats[k] for k in ('mae',) + _ERR_STATS}}))
        for (dname, a), series in thr.items():
            thr_rows.append(pd.DataFrame({'model': res.sim_name, 'day': days, 'variable': vname,
                                          'alpha': a, 'denom': dname, 'pct_exceed': series}))
    err = pd.concat(err_rows, ignore_index=True)
    thr_df = pd.concat(thr_rows, ignore_index=True)
    # same row order as the paper tool: denominator, variable, alpha
    thr_df['_d'] = thr_df['denom'].map({'simulation': 0, 'timestep': 1})
    thr_df['_v'] = thr_df['variable'].map({v: i for i, v in enumerate(var_names)})
    thr_df['_a'] = thr_df['alpha'].map({a: i for i, a in enumerate(thresholds)})
    thr_df = thr_df.sort_values(['_d', '_v', '_a'], kind='stable').drop(columns=['_d', '_v', '_a'])
    return err, thr_df.reset_index(drop=True)


def material_balance(res: RolloutResult):
    """Cumulative material-balance error per phase, true and predicted (raw and % of the
    initial in-place volume). Returns (days, true_raw, true_pct, est_raw, est_pct)."""
    vol_true = res.true[:, :, 5:8].astype(np.float64).sum(axis=1)       # [n_days, 3]
    vol_pred = res.pred[:, :, 5:8].astype(np.float64).sum(axis=1)       # [n_pred, 3]
    v0 = vol_true[0]
    true_days = res.days[1:]
    cum_con_true = np.cumsum(res.con_sum_true[:-1].astype(np.float64), axis=0)
    mbe_true = (vol_true[1:] - v0) - cum_con_true
    cum_con_used = np.cumsum(res.con_sum_used, axis=0)
    mbe_est = (vol_pred[1:] - v0) - cum_con_used[:vol_pred.shape[0] - 1]
    return true_days, mbe_true, 100.0 * mbe_true / v0, mbe_est, 100.0 * mbe_est / v0


def select_cells(batch_all, n_cells: int = 32) -> List[int]:
    """Cells written to rollout_<sim>.csv: the well cells, then evenly spaced others."""
    wells = []
    for attr in ('conp_index', 'conw_index', 'cong_index'):
        for c in getattr(batch_all, attr).tolist():
            if c not in wells:
                wells.append(c)
    n = batch_all.stat.shape[0]
    rest = [c for c in np.linspace(0, n - 1, max(n_cells, 1) + 2)[1:-1].round().astype(int).tolist()
            if c not in wells]
    return (wells + rest)[:max(n_cells, len(wells))]


# MARK: Wells (rate mode)
def _rel_error_pct(pred, true, ref):
    pred, true = np.asarray(pred, dtype=float), np.asarray(true, dtype=float)
    ok = true > _CUMULATIVE_MIN_REL_TRUE * max(float(ref), 1e-30)
    return np.where(ok, 100.0 * (pred - true) / np.where(ok, true, 1.0), np.nan)


def well_rows(res: RolloutResult) -> pd.DataFrame:
    """rollout_wells: per well, the per-step volumes (Oil, Gas, Water) and BHP, True vs Est."""
    b = res.batch_all
    days = res.days
    true_days = days[1:]
    pred_days = np.asarray(res.rate_days, dtype=float)
    n_pred = len(pred_days)
    rows = []

    def _add(label, wtype, series, day_arr, true_vals, est_vals):
        est = np.full(len(day_arr), np.nan)
        est[:len(est_vals)] = est_vals
        rows.append(pd.DataFrame({'model': res.sim_name, 'well': label, 'well_type': wtype,
                                  'series': series, 'day': day_arr, 'True': true_vals, 'Est': est}))

    wellp = b.wellp.numpy()
    for w in range(wellp.shape[1]):
        d = wellp[1:, w, :3] - wellp[:-1, w, :3]
        pred = res.rate_wells.get('producer', {}).get(w, {})
        label = f'Producer {w + 1}'
        for j, ph in enumerate(PHASES):
            _add(label, 'producer', ph, true_days, d[:, j], pred.get(ph, [])[:n_pred])
        _add(label, 'producer', 'BHP', days, wellp[:, w, 3], wellp[:1 + n_pred, w, 3])
    wellw = b.wellw.numpy()
    for w in range(wellw.shape[1]):
        d = wellw[1:, w, 0] - wellw[:-1, w, 0]
        pred = res.rate_wells.get('water_injector', {}).get(w, {})
        label = f'Water Injector {w + 1}'
        _add(label, 'water_injector', 'Water', true_days, d, pred.get('Water', [])[:n_pred])
        _add(label, 'water_injector', 'BHP', days, wellw[:, w, 1], wellw[:1 + n_pred, w, 1])
    wellg = b.wellg.numpy()
    for w in range(wellg.shape[1]):
        d = wellg[1:, w, 0] - wellg[:-1, w, 0]
        label = f'Gas Injector {w + 1}'
        _add(label, 'gas_injector', 'Gas', true_days, d, [])
        _add(label, 'gas_injector', 'BHP', days, wellg[:, w, 1], wellg[:1 + n_pred, w, 1])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def cumulative_tables(res: RolloutResult):
    """rollout_cumulative (series) and rollout_cumulative_errors (summary) tables."""
    b = res.batch_all
    pred_days = np.asarray(res.rate_days, dtype=float)
    n_reached = len(pred_days)
    comps = []
    for name, kind, tensor_name, col, phase in _CUMULATIVE_COMPONENTS:
        cum = getattr(b, tensor_name).numpy()[:, :, col]
        n_wells = cum.shape[1]
        if n_wells == 0:
            continue
        true = cum[1:] - cum[0]
        pred = np.zeros((n_reached, n_wells))
        for w in range(n_wells):
            s = np.asarray(res.rate_wells.get(kind, {}).get(w, {}).get(phase, []), dtype=float)
            m = min(len(s), n_reached)
            pred[:m, w] = s[:m]
        pred = np.cumsum(pred, axis=0)
        comps.append({'name': name, 'true': true, 'pred': pred, 'true_total': true.sum(axis=1),
                      'pred_total': pred.sum(axis=1),
                      'labels': [f'{"Producer" if kind == "producer" else "Water Injector"} {w + 1}'
                                 for w in range(n_wells)]})
    by_name = {c['name']: c for c in comps}
    oil, water = by_name.get('Oil production'), by_name.get('Water production')
    if oil is not None and water is not None and n_reached:
        err = water['pred'] - water['true'][:n_reached]
        oil_true = oil['true'][:n_reached]
        ok = oil_true > _CUMULATIVE_MIN_REL_TRUE * max(float(oil['true'][-1].max()), 1e-30)
        water['of_oil'] = np.where(ok, 100.0 * err / np.where(ok, oil_true, 1.0), np.nan)
        err_t = water['pred_total'] - water['true_total'][:n_reached]
        oil_t = oil['true_total'][:n_reached]
        ok = oil_t > _CUMULATIVE_MIN_REL_TRUE * max(float(oil['true_total'].max()), 1e-30)
        water['of_oil_total'] = np.where(ok, 100.0 * err_t / np.where(ok, oil_t, 1.0), np.nan)
    series, summary = [], []
    for c in comps:
        entries = [(lab, c['true'][:n_reached, w], c['pred'][:, w],
                    c['of_oil'][:, w] if 'of_oil' in c else None) for w, lab in enumerate(c['labels'])]
        entries.append(('Field', c['true_total'][:n_reached], c['pred_total'], c.get('of_oil_total')))
        for lab, tv, pv, of_oil in entries:
            ref = c['true_total'].max() if lab == 'Field' else c['true'][-1].max()
            rel = _rel_error_pct(pv, tv, ref)
            series.append(pd.DataFrame({'model': res.sim_name, 'component': c['name'], 'well': lab,
                                        'day': pred_days, 'True': tv, 'Est': pv, 'rel_err_pct': rel,
                                        'err_pct_of_oil': np.nan if of_oil is None else of_oil}))
            if n_reached:
                summary.append({'model': res.sim_name, 'component': c['name'], 'well': lab,
                                'last_day': float(pred_days[-1]), 'true_cum': float(tv[-1]),
                                'pred_cum': float(pv[-1]), 'rel_err_pct': float(rel[-1]),
                                'mean_abs_rel_err_pct': (float(np.nanmean(np.abs(rel)))
                                                         if np.isfinite(rel).any() else np.nan),
                                'err_pct_of_oil': np.nan if of_oil is None else float(of_oil[-1])})
    return (pd.concat(series, ignore_index=True) if series else None,
            pd.DataFrame(summary) if summary else None)


# MARK: Writing
def _well_markers(b, names: Optional[dict]):
    ijk = b.ijk.numpy()
    ij, types, out_names = [], [], []
    for wtype, con_attr, well_attr, short in (('producer', 'conp_index', 'wellp_index', 'P'),
                                              ('gas_injector', 'cong_index', 'wellg_index', 'GI'),
                                              ('water_injector', 'conw_index', 'wellw_index', 'WI')):
        con, wid = getattr(b, con_attr).numpy(), getattr(b, well_attr).numpy()
        if len(con) == 0:
            continue
        wnames = (names or {}).get(wtype) or []
        for w in np.unique(wid):
            ij.append(ijk[con[wid == w], :2].mean(axis=0))
            types.append(wtype)
            out_names.append(str(wnames[int(w)]) if int(w) < len(wnames) else f'{short}{int(w) + 1}')
    return {'well_marker_ij': np.array(ij, dtype=float).reshape(-1, 2),
            'well_marker_type': np.array(types, dtype=str),
            'well_marker_name': np.array(out_names, dtype=str)}


def well_names(dataset, sim_name: str) -> Optional[dict]:
    """Well names by type (lists indexed by well id) from the dataset details."""
    try:
        details = dataset.get_all_details(as_dataframe=True)
        row = details[details['name'] == sim_name].iloc[0]
        return {'producer': list(row['prod']), 'gas_injector': list(row['ginj']),
                'water_injector': list(row['winj'])}
    except (KeyError, IndexError):
        return None


def save_npz(path: Path, res: RolloutResult, names: Optional[dict] = None,
             var_names: Sequence[str] = STATE_NAMES) -> Path:
    """The raw rollout of one simulation (keys listed in the module docstring)."""
    b = res.batch_all
    ijk = b.ijk.numpy()
    idxs = np.concatenate([getattr(b, a).numpy() for a in ('conp_index', 'cong_index', 'conw_index')])
    arrays = {
        'sim_model': np.array(res.sim_name), 'var_names': np.array(list(var_names)),
        'days': res.days.astype(float), 'pred': res.pred.astype(np.float32),
        'pred_days': res.days[:res.pred.shape[0]].astype(float), 'true': res.true.astype(np.float32),
        'ijk': ijk, 'stat': b.stat.numpy(), 'well_ij': np.unique(ijk[idxs][:, :2], axis=0),
    }
    for attr in ('wellp', 'wellw', 'wellg', 'conp_index', 'conw_index', 'cong_index',
                 'wellp_index', 'wellw_index', 'wellg_index'):
        arrays[attr] = getattr(b, attr).numpy()
    arrays.update(_well_markers(b, names))
    if res.rate_wells:
        n = len(res.rate_days)
        prod, winj = res.rate_wells.get('producer', {}), res.rate_wells.get('water_injector', {})
        arrays['well_pred_days'] = np.asarray(res.rate_days, dtype=float)
        arrays['pred_producer_rates'] = (np.stack(
            [np.stack([prod[w]['Oil'], prod[w]['Gas'], prod[w]['Water']], axis=-1) for w in sorted(prod)],
            axis=1).astype(np.float32) if prod else np.zeros((n, 0, 3), dtype=np.float32))
        arrays['pred_water_injector_rates'] = (np.stack(
            [np.asarray(winj[w]['Water'], dtype=float) for w in sorted(winj)], axis=1)
            if winj else np.zeros((n, 0)))
    np.savez_compressed(path, **arrays)
    return path


def summary_rows(res: RolloutResult, err: pd.DataFrame, mbe, cum_summary: Optional[pd.DataFrame],
                 mode: str, report_days: Sequence[float] = (1000.0,)) -> List[dict]:
    """Paper metrics at the report days and at the last predicted day."""
    days_mbe, _, _, _, est_pct = mbe
    last = float(err['day'].max()) if len(err) else float('nan')
    rows = []
    for tag, day in [(f'{d:g}', float(d)) for d in report_days] + [('end', last)]:
        p = err[(err.variable == 'P') & (err.day == day)]
        sw = err[(err.variable == 'Sw') & (err.day == day)]
        if p.empty:
            continue
        r = {'sim': res.sim_name, 'mode': mode, 'day': tag, 'day_value': day,
             'P_r2': p.r2.iloc[0], 'P_shifted_r2': p.shifted_r2.iloc[0],
             'P_mae_kgf_cm2': p.mae.iloc[0], 'P_mae_MPa': p.mae.iloc[0] * KGF_CM2_TO_MPA,
             'Sw_r2': sw.r2.iloc[0], 'Sw_r2_clipped': max(sw.r2.iloc[0], -1.0) if np.isfinite(sw.r2.iloc[0]) else np.nan,
             'Sw_mae': sw.mae.iloc[0]}
        k = np.where(np.isclose(days_mbe[:len(est_pct)], day))[0]
        if len(k):
            r['MBe_oil_pct_abs'] = abs(est_pct[k[0], 0])
            r['MBe_water_pct_abs'] = abs(est_pct[k[0], 2])
        if cum_summary is not None and tag == 'end':
            for comp, col in (('Oil production', 'cum_oil_rel_err_pct'),
                              ('Water production', 'cum_water_rel_err_pct'),
                              ('Water injection', 'cum_winj_rel_err_pct')):
                f = cum_summary[(cum_summary.component == comp) & (cum_summary.well == 'Field')]
                if len(f):
                    r[col] = f.rel_err_pct.iloc[0]
            f = cum_summary[(cum_summary.component == 'Water production') & (cum_summary.well == 'Field')]
            if len(f):
                r['cum_water_err_pct_of_oil'] = f.err_pct_of_oil.iloc[0]
        rows.append(r)
    return rows


def write_rollout(out_dir: Path, res: RolloutResult, dataset=None, mode: str = 'imposed',
                  n_cells: int = 32, report_days: Sequence[float] = (1000.0,)) -> List[dict]:
    """Write every file of one rollout into `out_dir`; returns its summary rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = res.sim_name
    names = well_names(dataset, sim) if dataset is not None else None
    save_npz(out_dir / f'rollout_{sim}.npz', res, names)
    err, thr = daily_error_table(res)
    err.to_csv(out_dir / f'rollout_errors_{sim}.csv', index=False)
    thr.to_csv(out_dir / f'rollout_thresholds_{sim}.csv', index=False)

    mbe = material_balance(res)
    true_days, mbe_true, mbe_true_pct, mbe_est, mbe_est_pct = mbe
    n_pred = res.pred.shape[0]
    rows = []
    for i, vname in enumerate(STATE_NAMES):
        for c in select_cells(res.batch_all, n_cells):
            est = np.full(len(res.days), np.nan)
            est[:n_pred] = res.pred[:, c, i]
            rows.append(pd.DataFrame({'model': sim, 'day': res.days, 'cell': c, 'variable': vname,
                                      'True': res.true[:, c, i], 'Est': est}))
    for pos, p in enumerate(PHASES):
        for var, tv, ev in ((f'MBe_{p}', mbe_true[:, pos], mbe_est[:, pos]),
                            (f'MBe_{p}_pct', mbe_true_pct[:, pos], mbe_est_pct[:, pos])):
            est = np.full(len(true_days), np.nan)
            est[:len(ev)] = ev
            rows.append(pd.DataFrame({'model': sim, 'day': true_days, 'cell': np.nan,
                                      'variable': var, 'True': tv, 'Est': est}))
    pd.concat(rows, ignore_index=True).to_csv(out_dir / f'rollout_{sim}.csv', index=False)

    cum_summary = None
    if res.rate_wells:
        well_rows(res).to_csv(out_dir / f'rollout_wells_{sim}.csv', index=False)
        series, cum_summary = cumulative_tables(res)
        if series is not None:
            series.to_csv(out_dir / f'rollout_cumulative_{sim}.csv', index=False)
        if cum_summary is not None:
            cum_summary.to_csv(out_dir / f'rollout_cumulative_errors_{sim}.csv', index=False)

    final = None
    if n_pred > 1:
        final = {v: regression_metrics(res.true[n_pred - 1, :, i], res.pred[-1, :, i])
                 for i, v in enumerate(STATE_NAMES)}
    n_steps = n_pred - 1
    entry = {
        'n_nodes': int(res.true.shape[1]),
        'step_size': float(res.step_size),
        'mode': mode,
        'days_available': [float(res.days[0]), float(res.days[-1])],
        'days_simulated': [float(res.days[0]), float(res.days[n_steps])] if n_steps else None,
        'concluded': bool(res.concluded),
        'stop_reason': res.stop_reason,
        'stop_day': res.stop_day,
        'n_steps_requested': int(len(res.days) - 2),
        'n_steps_simulated': int(n_steps),
        'elapsed_seconds': float(res.elapsed),
        'elapsed_seconds_per_step': float(res.elapsed / n_steps) if n_steps else None,
        'final_step_metrics': final,
        'mass_balance_error_pct_final_step': (
            {p: float(mbe_est_pct[-1, i]) for i, p in enumerate(PHASES)} if len(mbe_est_pct) else None),
    }
    log = out_dir / 'rollout_performance.yaml'
    data = {}
    if log.is_file():
        with open(log, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
    data[sim] = entry
    with open(log, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return summary_rows(res, err, mbe, cum_summary, mode, report_days)
