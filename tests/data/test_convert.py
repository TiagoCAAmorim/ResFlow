"""IMEX conversion must give what the original conversion gave (golden: `_data/golden/raw`).

Integers, strings, JSON and indices are compared exactly, floats with allclose(rtol=1e-6)
(the maximum differences are printed with `pytest -s`). Not written by the port:
`file_path` of model.json (absolute path) and the dense distance matrices of res_static.pt.
"""
from pathlib import Path

import pytest
import torch

from conftest import GOLDEN, SETS, needs_data
import _compare as cmp

pytestmark = needs_data

# Runs that ended at day 4 / never opened their wells (also discarded by the original code)
EXPECTED_FAILURES = {
    'spe10': {'L03_32x32_02_2P1I_variable'},
    'spe10_fluvial': set(),
    'spe10_10d': set(),
    'spe10_3d_10d': {'L31-35_120x60_00_2P1I_fixed'},
}


@pytest.mark.parametrize('name', list(SETS))
def test_raw_equals_golden(name, converted, capsys):
    root, messages, decks = converted[name]
    golden = GOLDEN / 'raw' / name
    sel = None if decks is None else set(decks)
    g = cmp.split_sim_folders(golden)
    p = cmp.split_sim_folders(root)
    # the golden folders of the simulations that were converted
    keys = sorted(k for k in g if sel is None or k[1] in sel)
    assert sorted(p) == keys, 'converted simulations differ from the golden ones'
    stats = cmp.Stats()
    for key in keys:
        cmp.compare_raw_folder(g[key], p[key], stats)
    with capsys.disabled():
        print(f"\n[convert {name}] {len(keys)} simulations: {stats.summary()}")
    assert stats.n_problems == 0, stats.problems[:5]


@pytest.mark.parametrize('name', list(SETS))
def test_discarded_runs_are_logged(name, converted):
    root, messages, decks = converted[name]
    expected = EXPECTED_FAILURES[name]
    if decks is not None:
        expected = expected & set(decks)
    assert {m.split(':')[0] for m in messages} == expected
    log = (root / 'processing.log').read_text()
    for m in messages:
        assert m in log or m.split(': ', 1)[1] in log
        assert 'wells never opened' in m
    for stem in expected:
        assert not any((root / s / 'raw' / stem).exists() for s in ('train', 'val', 'test'))


def test_no_machine_paths_in_raw_files(converted):
    root = converted['spe10_10d'][0]
    needle = str(Path(__file__).resolve().parents[2]).encode()
    for f in root.rglob('*.json'):
        assert needle not in f.read_bytes(), f
    model = next(root.rglob('model.json'))
    import json
    data = json.loads(model.read_text())
    assert 'file_path' not in data and data['name'].endswith('.sr3')


def test_static_graph_has_no_dense_matrices(converted):
    root = converted['spe10_10d'][0]
    data = torch.load(next(root.rglob('res_static.pt')), weights_only=False)
    keys = set(data.keys())
    assert not {'shortest_path_dist', 'shortest_path_dist_mod', 'paths'} & keys
    assert {'stat', 'edge_index', 'edge_tran', 'edge_dist', 'edge_dist_mod', 'ijk', 'mask'} <= keys


def test_comparison_detects_a_perturbed_file(converted, tmp_path):
    """Negative check: a modified float tensor and a modified integer are reported."""
    import shutil
    root = converted['spe10_10d'][0]
    sim = next(cmp.split_sim_folders(root).values().__iter__())
    copy = tmp_path / 'sim'
    shutil.copytree(sim, copy)
    golden = next(v for k, v in cmp.split_sim_folders(GOLDEN / 'raw' / 'spe10_10d').items()
                  if k[1] == sim.name)
    stats = cmp.Stats()
    cmp.compare_raw_folder(golden, copy, stats)
    assert stats.n_problems == 0
    pres = torch.load(copy / 'res_dynamic' / 'pres.pth')
    pres[3, 5] *= 1.001
    torch.save(pres, copy / 'res_dynamic' / 'pres.pth')
    idx = torch.load(copy / 'res_static.pt', weights_only=False)
    idx.edge_index[0, 0] += 1
    torch.save(idx, copy / 'res_static.pt')
    stats = cmp.Stats()
    cmp.compare_raw_folder(golden, copy, stats)
    assert stats.n_problems == 2


def test_step_size_mismatch_gives_a_clear_error(tmp_path):
    """10-day decks converted with step 1 keep no steps: clear message, not a crash."""
    import pandas as pd
    from resflow.data.convert import organize_sr3_dataset
    from conftest import IMEX
    index = pd.read_csv(IMEX / 'spe10_10d' / 'decks_index.csv').iloc[:1]
    index.to_csv(tmp_path / 'index.csv', index=False)
    with pytest.raises(RuntimeError, match='spaced exactly step_size=1 days apart'):
        organize_sr3_dataset(IMEX / 'spe10_10d' / 'decks', tmp_path / 'out',
                             tmp_path / 'index.csv', step_size=1.0)


def test_existing_output_is_not_overwritten_silently(converted, tmp_path):
    from resflow.data.convert import organize_sr3_dataset
    root = converted['spe10_10d'][0]
    with pytest.raises(FileExistsError):
        organize_sr3_dataset(Path('.'), root, tmp_path / 'x.csv')


def test_cli_convert_writes_only_where_told(tmp_path, monkeypatch):
    """`resflow convert` through the argparse entry point; nothing is written in the cwd."""
    import pandas as pd
    from resflow.cli import main
    from conftest import IMEX
    work = tmp_path / 'cwd'
    work.mkdir()
    monkeypatch.chdir(work)
    index = pd.read_csv(IMEX / 'spe10_10d' / 'decks_index.csv').iloc[:1]
    index.to_csv(tmp_path / 'index.csv', index=False)
    assert main(['convert', '--input', str(IMEX / 'spe10_10d' / 'decks'), '--index', str(tmp_path / 'index.csv'),
                 '--output', str(tmp_path / 'out'), '--step-size', '10']) == 0
    assert list(work.iterdir()) == []
    assert (tmp_path / 'out' / 'processing.log').is_file()
    with pytest.raises(RuntimeError, match='No simulation could be converted'):  # OPM: no deck in an empty folder
        main(['convert', '--format', 'opm', '--input', str(tmp_path), '--index', str(tmp_path / 'index.csv'),
              '--output', str(tmp_path / 'o2')])
