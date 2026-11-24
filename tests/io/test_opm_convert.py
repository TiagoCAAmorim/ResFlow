"""OPM -> raw converter: unit tests on one OPM run (L33_32x32_01_8P4I_variable, daily test deck).

The run lives in ``_data/opm/spe10_opm/<deck>`` (made with deck_builder/cmg_to_opm.py + flow); tests that need it
or the IMEX golden raw folder skip when `_data/` is absent.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from resflow.io.opm import OpmRunDiscarded, convert_opm_run, organize_opm_dataset
from resflow.io.opm.convert import select_outputs
from resflow.io.opm.dat_tables import target_at

ROOT = Path(__file__).resolve().parents[2]
DECK = 'L33_32x32_01_8P4I_variable'
CASE = ROOT / '_data' / 'opm' / 'spe10_opm' / DECK
DAT = ROOT / '_data' / 'imex' / 'spe10' / 'decks' / f'{DECK}.dat'
GOLDEN = ROOT / '_data' / 'golden' / 'raw' / 'spe10' / 'test' / 'raw' / DECK
have_run = pytest.mark.skipif(not (CASE.is_dir() and DAT.is_file()), reason='_data OPM case absent')
have_golden = pytest.mark.skipif(not GOLDEN.is_dir(), reason='golden raw folder absent')


@pytest.fixture(scope='module')
def raw(tmp_path_factory):
    return convert_opm_run(CASE, DAT, tmp_path_factory.mktemp('raw') / DECK)


def test_target_at_applies_after_the_control_day():
    t = target_at([(4.0, 231.0), (10.0, 200.0)], [0, 4, 5, 10, 11])
    assert np.isnan(t[:2]).all() and t[2] == 231.0 and t[3] == 231.0 and t[4] == 200.0


def test_select_outputs_matches_date_lines():
    kept, groups = select_outputs([0., 1., 2., 5., 6., 7.])
    assert kept == [0, 1, 2, 3, 4, 5] and groups == [[0, 2], [3, 5]]


@have_run
def test_files_and_shapes(raw):
    m = json.loads((raw / 'model.json').read_text())
    assert list(m) == ['name', 'grid_type', 'model_type', 'n_cells', 'days', 'steps', 'prod', 'winj', 'ginj',
                       'n_krel', 'n_pvt']
    n = len(m['days'])
    assert m['days'][:3] == [0.0, 1.0, 2.0] and m['steps'] == [[0, n - 1]]
    for f in ('pres', 'so', 'sg', 'rs'):
        t = torch.load(raw / 'res_dynamic' / f'{f}.pth')
        assert t.shape == (n, m['n_cells']) and t.dtype == torch.float32
    assert torch.load(raw / 'con_dynamic' / 'np.pth').shape == (n, len(m['prod']))
    assert torch.load(raw / 'con_dynamic' / 'wi.pth').shape == (n, len(m['winj']))
    # cumulative volumes are monotonic and zero before the wells open (day 4)
    np_ = torch.load(raw / 'well_dynamic' / 'np.pth')
    assert (np_[:5] == 0).all() and (np_.diff(dim=0) >= -1e-3).all()


@have_run
def test_shut_wells_report_no_volume(raw):
    """A well shut by the no-crossflow rule (P3 from day ~610 in this deck) has flat cumulative volumes."""
    np_ = torch.load(raw / 'well_dynamic' / 'np.pth')[:, 2]
    assert np_[612] == np_[700] and np_[1000] > np_[700]  # shut, then flows again


@have_run
def test_discard_when_wells_never_open(tmp_path):
    with pytest.raises(OpmRunDiscarded):
        convert_opm_run(CASE, DAT, tmp_path / 'x', wells_open_day=1e9)
    assert not (tmp_path / 'x').exists()


@have_run
def test_organize_opm_dataset(tmp_path):
    (tmp_path / 'idx.csv').write_text(f'deck,set\n{DECK},test\nmissing_deck,train\n')
    msgs = organize_opm_dataset(DAT.parent, tmp_path / 'out', tmp_path / 'idx.csv', opm_dir=CASE.parent)
    assert (tmp_path / 'out' / 'test' / 'raw' / DECK / 'model.json').is_file()
    assert len(msgs) == 1 and msgs[0].startswith('missing_deck')


@have_run
@have_golden
def test_equals_golden_where_deterministic(raw):
    """Static data, PVT/KR and well index equal the IMEX ones."""
    for f in ('pvt01.pth',):
        a, b = torch.load(raw / f), torch.load(GOLDEN / f)
        assert all(torch.equal(a[k], b[k]) for k in b)
    for f in ('krel01.wo.pth', 'krel01.go.pth'):
        assert torch.equal(torch.load(raw / f), torch.load(GOLDEN / f))
    ga, gb = (torch.load(p / 'res_static.pt', weights_only=False) for p in (raw, GOLDEN))
    assert torch.equal(ga.edge_index, gb.edge_index) and torch.equal(ga.ijk, gb.ijk)
    for k in ('stat', 'edge_tran', 'edge_dist', 'edge_dist_mod'):
        assert torch.allclose(getattr(ga, k), getattr(gb, k), rtol=1e-5, atol=1e-3)
    ca, cb = (json.loads((p / 'con_static.json').read_text()) for p in (raw, GOLDEN))
    for t in cb:
        for w in cb[t]:
            for c, v in cb[t][w].items():
                assert ca[t][w][c]['cell'] == v['cell'] and ca[t][w][c]['wi'] == pytest.approx(v['wi'], rel=0.01)
    assert (raw / 'well_static.json').read_text() == (GOLDEN / 'well_static.json').read_text()


DECK_1P0I = 'L34_32x32_00_1P0I_fixed'
CASE_1P0I = ROOT / '_data' / 'opm' / 'spe10' / 'decks_opm' / DECK_1P0I
DAT_1P0I = ROOT / '_data' / 'imex' / 'spe10' / 'decks' / f'{DECK_1P0I}.dat'
GOLDEN_1P0I = ROOT / '_data' / 'golden' / 'raw' / 'spe10' / 'test' / 'raw' / DECK_1P0I


@pytest.mark.skipif(not (CASE_1P0I.is_dir() and DAT_1P0I.is_file() and GOLDEN_1P0I.is_dir()),
                    reason='_data 1P0I OPM case or golden absent')
def test_producer_only_deck(tmp_path):
    """A deck without injectors (1P0I) converts, writes no injector files, and has the IMEX file set."""
    raw = convert_opm_run(CASE_1P0I, DAT_1P0I, tmp_path / DECK_1P0I)
    m = json.loads((raw / 'model.json').read_text())
    assert len(m['prod']) == 1 and m['winj'] == [] and m['ginj'] == []
    files = lambda p: sorted(str(f.relative_to(p)) for f in p.rglob('*') if f.is_file())
    assert files(raw) == files(GOLDEN_1P0I)
    assert json.loads((raw / 'con_static.json').read_text())['winj'] == {}
