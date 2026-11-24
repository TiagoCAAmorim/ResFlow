"""Short rollouts (CPU) in both modes: files and keys as the old tool, first steps vs the old code.

Skipped when weights/ or _data/ is absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / '_data' / 'port' / 'datasets' / 'spe10'
GOLD = ROOT / '_data' / 'golden' / 'rollouts'
SIM = 'L34_32x32_00_1P0I_fixed'
have = (DS / 'test' / 'processed').is_dir() and (ROOT / 'weights' / 'rate_2d_1d' / 'weights.pt').is_file()
NPZ_KEYS = {'days', 'pred', 'pred_days', 'true', 'var_names', 'ijk', 'stat', 'well_ij', 'wellp', 'wellw',
            'wellg', 'conp_index', 'conw_index', 'cong_index', 'wellp_index', 'wellw_index', 'wellg_index',
            'well_marker_ij', 'well_marker_type', 'well_marker_name', 'sim_model'}


@pytest.mark.skipif(not have, reason='weights/ or _data/ absent')
@pytest.mark.parametrize('mode', ['imposed', 'rate'])
def test_short_rollout(tmp_path, mode):
    from resflow.rollout.cli import run_rollouts
    rate = ROOT / 'weights' / 'rate_2d_1d' if mode == 'rate' else None
    summary = run_rollouts(ROOT / 'weights' / 'state_2d_1d', DS, 'test', tmp_path, rate_weights=rate,
                           sims=[SIM], max_day=12.0, device='cpu', report_days=(5.0,), verbose=False)
    sim = f'{SIM}.sr3'
    z = np.load(tmp_path / f'rollout_{sim}.npz')
    keys = NPZ_KEYS | ({'well_pred_days', 'pred_producer_rates', 'pred_water_injector_rates'}
                       if mode == 'rate' else set())
    assert keys <= set(z.files)
    assert z['pred'].shape[0] == len(z['days']) - 1 == 11          # days 0..10 predicted to 10
    err = pd.read_csv(tmp_path / f'rollout_errors_{sim}.csv')
    assert list(err.columns[:4]) == ['model', 'day', 'variable', 'mae']
    csv = pd.read_csv(tmp_path / f'rollout_{sim}.csv')
    assert {'MBe_Oil_pct', 'MBe_Water_pct', 'P', 'Sw'} <= set(csv.variable)
    assert (tmp_path / f'rollout_thresholds_{sim}.csv').is_file()
    assert (tmp_path / 'rollout_performance.yaml').is_file()
    if mode == 'rate':
        for f in ('rollout_wells', 'rollout_cumulative', 'rollout_cumulative_errors'):
            assert (tmp_path / f'{f}_{sim}.csv').is_file()
    assert set(summary['day']) == {'5', 'end'}
    # deterministic reference rollouts of the original code (folder names end in _exact)
    folders = sorted(GOLD.glob('*_imposed_exact' if mode == 'imposed' else '*_rate_*_exact'))
    gold = folders[0] / f'rollout_{sim}.npz' if folders else None
    if gold is not None and gold.is_file():
        g = np.load(gold)
        p_err = np.abs(z['pred'][:4, :, 1] - g['pred'][:4, :, 1]).max()
        assert p_err < 0.05, p_err      # kgf/cm2, first 3 days (CPU vs the GPU reference)
