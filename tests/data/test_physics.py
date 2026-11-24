"""resflow.physics must give what the original code gave (golden reference files in _data/golden, made with the original code).

Skipped when `_data/golden/physics` or the checkpoints are absent.
"""
from pathlib import Path

import pytest
import torch

from resflow.physics import Transform, build_vols2p_net
from resflow.physics.transform import load_pvt_krel

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / '_data' / 'golden' / 'physics' / 'ref.pt'
CKPT = ROOT / '_data' / 'checkpoints' / 'state_2d_1d' / 'model.pth'
RAW = ROOT / '_data' / 'golden' / 'raw'

pytestmark = pytest.mark.skipif(
    not (REF.is_file() and CKPT.is_file() and RAW.is_dir()),
    reason='golden physics reference or checkpoint absent')

# float32 results of identical formulas: same operations, so (almost) bit-identical.
RTOL, ATOL = 1e-5, 1e-6
# Ridder iterations and the network make the state inversion sensitive; pressures in kgf/cm2.
ATOL_INVERSION = 2e-4


@pytest.fixture(scope='module')
def state():
    return torch.load(CKPT, map_location='cpu', weights_only=True)


@pytest.fixture(scope='module')
def ref():
    return torch.load(REF, weights_only=False)


def _transform(name, state, with_nn=True):
    folder = next((RAW / name / ('test' if name != 'spe10_3d_10d' else 'train') / 'raw').iterdir())
    net = build_vols2p_net(state, 'transform.vols2p_nn.') if with_nn else None
    return Transform.from_folder(folder, vols2p_nn=net)


@pytest.fixture(scope='module', params=['spe10', 'spe10_3d_10d', 'spe10_fluvial'])
def case(request, ref, state):
    name = request.param
    data = ref[name]
    graph = type('G', (), {})()
    graph.stat, graph.edge_index, graph.edge_tran = data['stat'], data['edge_index'], data['edge_tran']
    return name, data, graph, _transform(name, state), _transform(name, state, with_nn=False)


def close(a, b, atol=ATOL):
    assert a.shape == b.shape
    torch.testing.assert_close(a, b, rtol=RTOL, atol=atol)


def test_state_dict_keys_match_checkpoint(case, state):
    _, _, _, tr, _ = case
    sub = {k[len('transform.'):]: v for k, v in state.items() if k.startswith('transform.')}
    assert set(tr.state_dict()) == set(sub)
    tr.load_state_dict(sub, strict=True)  # shapes too


def test_node_transforms(case):
    _, d, _, tr, _ = case
    por0, t = d['stat'][:, 1], d['x']
    close(tr.t2vols(por0, t), d['t2vols'])
    close(tr.t2vres(por0, t), d['t2vres'])
    close(tr.get_vpor(por0, t[..., 1]), d['get_vpor'])
    close(tr.t2v(por0, t), d['t2v'])
    close(tr.pvt.get(t[0, :, 0], t[0, :, 1]), d['pvt_get'])


def test_state_inversion(case):
    _, d, _, tr, tr_nonn = case
    por0, pert = d['stat'][:, 1], d['vols_pert']
    close(tr.vols2p_quick(por0, pert), d['vols2p_quick'], atol=1e-4)
    close(tr.vols2t(por0, pert, quick=True), d['vols2t_quick'], atol=1e-4)
    out = tr.vols2t(por0, pert)
    assert out.shape == d['vols2t'].shape
    # pressure (column 1) is the solved unknown: compare it in absolute terms
    assert float((out[..., 1] - d['vols2t'][..., 1]).abs().max()) < ATOL_INVERSION
    close(out, d['vols2t'], atol=ATOL_INVERSION)
    close(tr_nonn.vols2t(por0, pert), d['vols2t_nonn'], atol=ATOL_INVERSION)


def test_potentials_mobilities_flows(case):
    _, d, g, tr, _ = case
    t = d['x']
    close(tr.get_phi(t, z=d['stat'][:, 0]), d['phi'])
    close(tr.get_kr(t), d['kr'])
    close(tr.get_mobility(t), d['mob'])
    close(tr.get_mobility(t, expanded=True), d['mob_exp'])
    close(tr.t2f(g, t, pvt_table=d['pvt_table'], inj_flow=True), d['t2f_inj'])
    close(tr.t2f(g, t), d['t2f'])


def test_gather(case):
    _, d, g, tr, _ = case
    close(tr.gather_f(g, d['t2f_inj']), d['gather_f'], atol=1e-4)
    close(tr.gather_f(g, d['t2f']), d['gather_f_4'], atol=1e-4)
    # exact variant: same CPU algorithm, bit-identical
    assert torch.equal(tr.gather_f_exact(g, d['t2f_inj']), d['gather_f_exact'])
    assert torch.equal(tr.gather_f_exact(g, d['t2f']), d['gather_f_exact_4'])


def test_gather_f_is_differentiable(case):
    _, d, g, tr, _ = case
    f = d['t2f_inj'].clone().requires_grad_(True)
    tr.gather_f(g, f).sum().backward()
    assert f.grad is not None and torch.isfinite(f.grad).all()


def test_pvt_file_naming(case, ref):
    name, d, _, _, _ = case
    folder = next((RAW / name / ('test' if name != 'spe10_3d_10d' else 'train') / 'raw').iterdir())
    pvt, wo, go = load_pvt_krel(folder, 1)
    assert wo.shape[1] == 3 and go.shape[1] == 3
    assert (folder / 'krel01.wo.pth').is_file() and not (folder / 'krel01.pth').exists()


def test_transform_runs_without_the_network(case):
    _, d, _, _, tr_nonn = case
    assert tr_nonn.vols2p_nn is None
    out = tr_nonn.vols2t(d['stat'][:, 1], d['vols_pert'][:1])
    assert torch.isfinite(out).all()
