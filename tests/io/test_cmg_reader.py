"""The vendored CMG reader must give exactly what the original reader gave on the subset decks.

Golden outputs (`_data/golden/cmg_reader/*.pkl`) were produced once with the original library
(rsimpy). Tests skip when `_data/` is absent.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _cmg_collect as cc  # noqa: E402

from resflow.io.cmg.sr3reader import Sr3Reader  # noqa: E402
from resflow.io.cmg.outreader import wi  # noqa: E402
from resflow.io.cmg.datreader import dat_parser, dat_run, dat_pvt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMEX = ROOT / '_data' / 'imex'
GOLDEN = ROOT / '_data' / 'golden' / 'cmg_reader'

DECKS = {
    'train_32x32_185dates': 'spe10/decks/L01_32x32_00_2P1I_variable',
    'test_daily_60x60': 'spe10/decks/L32_60x60_01_8P4I_variable',
    'deck_3d_10d': 'spe10_3d_10d/decks/L01-05_32x32_00_2P1I_variable',
}

pytestmark = pytest.mark.skipif(
    not (IMEX.is_dir() and GOLDEN.is_dir()), reason='_data/imex or _data/golden/cmg_reader absent')


@pytest.fixture(scope='module', params=list(DECKS))
def case(request):
    key = request.param
    golden_file = GOLDEN / f'{key}.pkl'
    deck = IMEX / DECKS[key]
    if not golden_file.is_file() or not deck.with_suffix('.sr3').is_file():
        pytest.skip(f'deck or golden missing for {key}')
    with open(golden_file, 'rb') as f:
        golden = pickle.load(f)
    new = cc.collect(deck, Sr3Reader, wi, dat_parser, dat_run, dat_pvt)
    return key, golden, new


def test_same_items(case):
    _, golden, new = case
    assert list(golden) == list(new)


@pytest.mark.parametrize('item', [
    'filepath_name', 'days_grid', 'days_well', 'n_active', 'ijk', 'grid_props', 'components',
    'grid_elements_n', 'grid_elements_head', 'wells', 'well_layers', 'layer_cell_act',
    'n_krel', 'krel', 'connections', 'transmissibilities',
    'static_grid', 'blockdepth_active', 'dyn_grid_all_days', 'well_cum_last_day',
    'well_dyn_all', 'well_dyn_every3', 'con_dyn_all', 'con_dyn_every3',
    'dat_pvt_parsed', 'pvt', 'dat_well_parsed', 'bhp_depth', 'wi_table',
])
def test_item_identical(case, item):
    key, golden, new = case
    assert item in golden, f'{item} missing in golden of {key}'
    diffs = cc.deep_diff(golden[item], new[item], item)
    assert not diffs, '\n'.join(diffs[:10])


def test_krsetn_identical(case):
    _, golden, new = case
    assert ('krsetn' in golden) == ('krsetn' in new)
    if 'krsetn' in golden:
        assert not cc.deep_diff(golden['krsetn'], new['krsetn'])


def test_sanity(case):
    """Guard against comparing empty things with empty things."""
    _, golden, new = case
    assert new['n_active'] > 0 and len(new['wells']) > 0 and len(new['wi_table']) > 0
    assert new['n_krel'] >= 1 and len(new['pvt']) >= 1
    assert isinstance(new['n_active'], int)
    assert np.asarray(new['days_grid']).ndim == 1
    assert new['dyn_grid_all_days']['vars']['PRES'][1].shape[1] == len(new['days_grid'])


def test_train_deck_has_185_dates():
    deck = IMEX / DECKS['train_32x32_185dates']
    if not deck.with_suffix('.dat').is_file():
        pytest.skip('deck absent')
    with open(GOLDEN / 'train_32x32_185dates.pkl', 'rb') as f:
        parsed = pickle.load(f)['dat_well_parsed']
    from resflow.io.cmg.datreader import dat_dates
    parser = dat_parser.DatParser()
    parser.process(deck.with_suffix('.dat'))
    assert len(dat_dates.get_from_dat_data(parser.get())) >= 185
    assert parsed
