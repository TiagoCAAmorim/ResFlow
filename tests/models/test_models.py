"""State and rate models: feature registry, layers, bundles (CPU), one-step vs the old code.

The bundle tests need weights/; the golden tests need _data/ (skipped when absent).
"""
from pathlib import Path

import pytest
import torch

from resflow.models import load_bundle
from resflow.models.features import all_feature_names, grouped_names
from resflow.models.layers import GraphNet, LowRankConv

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / 'weights'
GOLDEN = ROOT / '_data' / 'golden' / 'onestep'
DATASETS = ROOT / '_data' / 'port' / 'datasets'
BUNDLES = ['state_2d_1d', 'rate_2d_1d', 'state_2d_1d_fluvial_ft']
have_weights = all((WEIGHTS / b / 'weights.pt').is_file() for b in BUNDLES)
have_golden = GOLDEN.is_dir() and (DATASETS / 'spe10' / 'test' / 'processed').is_dir()


def test_registry():
    names = all_feature_names()
    assert len(names) == 90 and len(set(names)) == 90
    assert grouped_names(['Sw', 'dVw_std', 'BHPw', 'G(f_w)']) == ['So', 'dVo_std', 'P', 'G(f_o)']


def test_lowrank_conv_batched_equals_single():
    torch.manual_seed(0)
    conv = LowRankConv((8, 4), [8], rank=2)
    x = torch.randn(3, 5, 8)
    ei = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
    e = torch.randn(3, 6, 4)
    out = conv(x, ei, e)
    for t in range(3):
        assert torch.allclose(out[t], conv(x[t], ei, e[t]), atol=1e-6)


def test_block_curriculum_and_dense():
    torch.manual_seed(0)
    net = GraphNet((6, 3), 2, (8, 4), [8], [4], [8], [8], [4], n_blocks=3, conv_rank=2,
                   dense_layers=True, init_scale=0.5)
    x, e = torch.randn(5, 6), torch.randn(4, 3)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    full = net(x, ei, e)
    net.processor.active_blocks = 1
    assert full.shape == (5, 2) and not torch.allclose(full, net(x, ei, e))


@pytest.mark.skipif(not have_weights, reason='weights/ absent')
@pytest.mark.parametrize('name', BUNDLES)
def test_bundle_loads_on_cpu(name):
    model = load_bundle(WEIGHTS / name, 'cpu')
    sd = torch.load(WEIGHTS / name / 'weights.pt', map_location='cpu', weights_only=True)
    assert set(model.state_dict()) == set(sd)
    assert all(v.dtype != torch.float64 for v in sd.values())
    assert float(model.transform.get_g()) == pytest.approx(1e-4)
    assert model.transform.get_z_mean() is None


@pytest.mark.skipif(not (have_weights and have_golden), reason='weights/ or _data/ absent')
@pytest.mark.parametrize('name', BUNDLES)
def test_onestep_matches_old_code(name):
    """First 3 golden samples (spe10 test decks) on CPU: inputs equal, outputs to float precision."""
    from resflow.data.dataset import SimDataset
    g = torch.load(GOLDEN / f'{name}.pt', map_location='cpu', weights_only=False)
    ds = SimDataset(root=DATASETS / 'spe10' / 'test', min_day=0.0, max_day=1827.0)
    model = load_bundle(WEIGHTS / name, 'cpu')
    cap = {}
    model.model.register_forward_hook(lambda m, a, o: cap.update(x=a[0], out=o))
    n = 0
    for s, r in zip(g['samples'], g['results']):
        if s['set'] != 'spe10' or s['n_nodes'] > 1024:
            continue
        b = ds.to_single_step(ds.get_sim_days(s['sim'], days=ds.get_days(s['sim'])), s['i_day'])
        with torch.no_grad():
            out = model(b) if name.startswith('state') else model(b)
        ref = r['dvol_hat'] if name.startswith('state') else r['rate_hat']
        assert torch.allclose(cap['x'], r['x_in'], rtol=1e-4, atol=1e-4 * r['x_in'].abs().max())
        assert (out - ref).abs().max() <= 1e-4 * ref.abs().max()
        if name.startswith('state'):
            _, t1 = model.step(b)
            assert (t1 - r['t1_hat']).abs().max() <= 1e-4 * r['t1_hat'].abs().max()
        n += 1
        if n == 3:
            break
    assert n == 3
