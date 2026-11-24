"""LinearScaler: fit equals the original fit; restore from a weights file needs no dataset."""
from pathlib import Path

import pytest
import torch

from resflow.data.normalization import LinearScaler, stats_path
from resflow.data.scalers import AsinhScaler, asinh_fit_transform

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / '_data' / 'golden' / 'physics' / 'scalers.pt'
CKPT = ROOT / '_data' / 'checkpoints' / 'state_2d_1d' / 'model.pth'


def _group(names):
    """Toy grouping: every 'fo*' feature borrows the statistics of 'fo'."""
    return ['fo' if n.startswith('fo') else n for n in names]


class _Samples(list):
    """A 'dataset' of two samples, each a [steps, nodes, features] tensor."""


@pytest.mark.skipif(not REF.is_file(), reason='golden scaler reference absent')
def test_fit_equals_original_fit():
    ref = torch.load(REF, weights_only=False)
    cols, names = ref['cols'], ref['names']
    data = _Samples([cols[:2500].reshape(5, 500, -1), cols[2500:].reshape(5, 500, -1)])
    scaler = LinearScaler()
    scaler.fit(data, names, get_values_fn=lambda d: d)
    for i, (scale, shift) in enumerate(ref['linear']):
        assert scaler.scales[i].item() == pytest.approx(scale, rel=1e-6), names[i]
        assert scaler.shifts[i].item() == pytest.approx(shift, rel=1e-6, abs=1e-12), names[i]
    assert scaler.coords.shape == (2,)
    # zero shift for flows/deltas/etc., mean for the others
    assert scaler.shifts[names.index('fo')].item() == 0.0
    assert scaler.shifts[names.index('P')].item() == pytest.approx(cols[:, 1].mean().item(), rel=1e-6)
    assert scaler.drop_degenerate_scale(names) == [n for n in names if n != 'const']


@pytest.mark.skipif(not REF.is_file(), reason='golden scaler reference absent')
def test_asinh_fit_equals_original_fit():
    ref = torch.load(REF, weights_only=False)
    x = ref['asinh_in']
    out = torch.cat([asinh_fit_transform(x[:, i:i + 1]) for i in range(x.shape[1])], dim=1)
    torch.testing.assert_close(out, ref['asinh_out'], rtol=1e-6, atol=0)
    assert isinstance(AsinhScaler().fit_transform(x[:, :1]), torch.Tensor)


def test_norm_denorm_roundtrip_and_groups():
    names = ['fo', 'fo_alias', 'P']
    s = LinearScaler(group_fn=_group)
    s.restore_from_state(
        {'n.scales': torch.tensor([2.0, 9.0, 4.0]), 'n.shifts': torch.tensor([0.0, 1.0, 10.0]),
         'n.coords': torch.zeros(2)}, 'n.', ['fo', 'fo_alias', 'P'])
    x = torch.randn(7, 3)
    assert torch.allclose(s.denorm(s.norm(x, names, grouped=False), names, grouped=False), x, atol=1e-6)
    # grouped: 'fo_alias' uses the statistics of 'fo'
    assert torch.allclose(s.norm(x, names, grouped=True)[:, 1], x[:, 1] / 2.0)
    assert s.norm(x).shape == x.shape  # all features, in order


def test_unfitted_scaler_refuses_to_run():
    with pytest.raises(RuntimeError, match='not been fitted'):
        LinearScaler().norm(torch.zeros(2, 3))


def test_save_and_load_stats(tmp_path):
    s = LinearScaler()
    s.restore_from_state({'a.scales': torch.ones(2), 'a.shifts': torch.zeros(2), 'a.coords': torch.zeros(2)},
                         'a.', ['x', 'y'])
    p = stats_path(tmp_path)
    assert p == tmp_path / 'analysis' / 'stats' / 'lin_scaler.pt'
    s.save_stats(p)
    t = LinearScaler()
    assert not t.load_stats(tmp_path / 'missing.pt')
    assert t.load_stats(p) and t.feature_names == ['x', 'y']


@pytest.mark.skipif(not CKPT.is_file(), reason='checkpoint absent')
def test_restore_from_the_shipped_weights_without_any_dataset():
    state = torch.load(CKPT, map_location='cpu', weights_only=True)
    n = len(state['data_norm_in.scales'])
    s = LinearScaler()
    s.restore_from_state(state, 'data_norm_in.', [f'f{i}' for i in range(n)])
    assert s.fitted and s.scales.shape == (n,) and s.coords.shape == (2,)
    with pytest.raises(ValueError, match='Refusing to guess'):
        LinearScaler().restore_from_state(state, 'data_norm_in.', ['only_one'])
