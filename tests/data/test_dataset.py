"""The processed dataset must equal the golden one (`_data/golden/datasets`, built by the original
code with samples_ratio=0.70, seed 0).

The random draw (resampling mask and weights) is reproduced exactly by building with the same
seed and the same processing order (`sim_order`, read from the golden data.json). Compared:
data.json (without the removed absolute `path`) and every tensor of every processed file
(without the never-read val_mask / test_mask). The spe10 set is built from its first 3 train
simulations only (same draw as the golden build up to there), to keep the test fast.
"""
import json
import shutil

import pytest
import torch

from conftest import GOLDEN, SETS, golden_order, needs_data
import _compare as cmp

pytestmark = needs_data


@pytest.fixture(scope='session')
def built(converted, tmp_path_factory):
    """{set: (dataset root, splits compared fully)} built with the golden parameters."""
    from resflow.data.dataset import build_dataset
    out = {}
    for name, step in SETS.items():
        raw_root = converted[name][0]
        root = tmp_path_factory.mktemp('ds') / name
        if name == 'spe10':  # train split only: the test split decks are daily and slow to process
            shutil.copytree(raw_root / 'train', root / 'train')
        else:
            shutil.copytree(raw_root, root)
        splits = [s for s in ('train', 'val', 'test')
                  if (root / s / 'raw').is_dir() and any((root / s / 'raw').iterdir())]
        order = {s: golden_order(name, s) for s in splits if (GOLDEN / 'datasets' / name / s).is_dir()}
        if name == 'spe10':
            splits = ['train']  # val/test draws depend on the whole train split
        build_dataset(root, samples_ratio=0.70, step_size=step, seed=0, sim_order=order)
        out[name] = (root, splits)
    return out


@pytest.mark.parametrize('name', list(SETS))
def test_processed_equals_golden(name, built, capsys):
    root, splits = built[name]
    stats = cmp.Stats()
    for split in splits:
        cmp.compare_dataset_split(GOLDEN / 'datasets' / name / split, root / split, stats,
                                  subset=(name == 'spe10'))
    with capsys.disabled():
        print(f"\n[build {name}] splits {splits}: {stats.summary()}")
    assert stats.n_tensors > 0
    assert stats.n_problems == 0, stats.problems[:5]


def test_step_size_is_stored_and_read(built):
    root = built['spe10_10d'][0]
    from resflow.data.dataset import SimDataset
    ds = SimDataset(root / 'train')  # no step_size given: read from data.json
    assert ds.step_size == 10.0
    assert all(v['step_size'] == 10.0 for v in ds.reload_details().values())


def test_dataset_needs_no_statistics(built):
    """Building and reading a dataset reads and writes no normalisation statistics."""
    from resflow.data.dataset import load_datasets
    root = built['spe10_10d'][0]
    out = load_datasets(root, load_test=True)
    assert out['train'] is not None and len(out['train']) > 0
    assert out['val'] is None  # this set has no val split
    assert not list(root.rglob('analysis'))
    assert not list(root.rglob('lin_scaler.pt')) and not list(root.rglob('optimal_g.pt'))
    sample = out['train'][0]
    assert sample.x.shape[-1] == 4


def test_ten_day_window_and_days(built):
    from resflow.data.dataset import SimDataset
    root = built['spe10_10d'][0]
    full = SimDataset(root / 'train')
    days = full.get_days()
    assert days[0] == 0 and days[-1] == 1810  # 183 outputs: step start days 0..1810 (max_day - step_size)
    assert all(b - a == 10 for a, b in zip(days, days[1:]))
    win = SimDataset(root / 'train', min_day=100, max_day=300)
    assert win.get_days()[0] == 100 and win.get_days()[-1] == 290
    for item in win:
        d0 = item.global_attr[1].item()
        assert 100 <= d0 < 300 and item.x.shape[0] == item.global_attr[4].item() + 1
    with pytest.raises(AssertionError, match='multiple of step_size'):
        SimDataset(root / 'train', min_day=15, step_size=10.0)


def test_sim_days_single_step_and_slices(built):
    from resflow.data.dataset import SimDataset
    root = built['spe10_10d'][0]
    ds = SimDataset(root / 'train')
    name = ds.get_sim_names(unique=True)[0]
    days = ds.get_days(name)
    batch = ds.get_sim_days(name, days[:6])
    assert batch.x.shape[0] == batch.y.shape[0] == 6
    assert batch.conp.shape[0] == batch.conp_y.shape[0] == 6 and batch.wellw_y.shape[0] == 6
    one = ds.to_single_step(batch, 2)
    assert torch.equal(one.x, batch.x[2]) and torch.equal(one.y, batch.y[2])
    assert one.global_attr[1].item() == batch.global_attr[1].item() + 2 * 10
    sl = ds.slice_data(batch, 1, 4)
    assert sl.x.shape[0] == 3 and torch.equal(sl.y, batch.y[1:4])
    single = ds.get_sim_day(name, days[3])
    assert torch.equal(single.x, batch.x[3])
    with pytest.raises(ValueError):
        ds.get_sim_days(name, [1.0e6])


def test_cli_build_default_order_is_valid(converted, tmp_path, monkeypatch):
    """`resflow build` with defaults (sorted order, seed 0): same files as the golden build and
    valid masks; only the random draw differs from the golden one."""
    from resflow.cli import main
    from resflow.data.dataset import SimDataset
    root = tmp_path / 'ds'
    shutil.copytree(converted['spe10_10d'][0], root)
    work = tmp_path / 'cwd'
    work.mkdir()
    monkeypatch.chdir(work)
    assert main(['build', '--root', str(root), '--step-size', '10']) == 0
    assert list(work.iterdir()) == []
    golden = json.loads((GOLDEN / 'datasets' / 'spe10_10d' / 'train' / 'data.json').read_text())
    mine = json.loads((root / 'train' / 'data.json').read_text())

    def key(v):
        return (v['name'], v['day_ini'], v['timestep_ini'], v['timestep_end'])
    assert sorted(map(key, golden.values())) == sorted(map(key, mine.values()))
    ds = SimDataset(root / 'train')
    ratios = []
    for item in ds:
        assert item.weights.shape == item.train_mask.shape
        assert bool(item.train_mask[:, item.conp_index].all())  # well cells always on
        assert bool((item.weights[~item.train_mask] == 0).all())  # weights only where drawn
        ratios.append(item.train_mask.float().mean().item())
    assert 0.3 < sum(ratios) / len(ratios) < 0.95
    # same seed, same default order -> reproducible
    root2 = tmp_path / 'ds2'
    shutil.copytree(converted['spe10_10d'][0], root2)
    main(['build', '--root', str(root2), '--step-size', '10'])
    a = torch.load(next((root / 'train' / 'processed').glob('data_0.pt')), weights_only=False)
    b = torch.load(next((root2 / 'train' / 'processed').glob('data_0.pt')), weights_only=False)
    assert torch.equal(a.train_mask, b.train_mask) and torch.equal(a.weights, b.weights)
