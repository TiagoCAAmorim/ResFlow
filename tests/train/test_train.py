"""Tiny trainings on the subset (CPU): run, checkpoint, resume, load as a bundle, roll out.

Skipped when weights/ or _data/ is absent. The networks are shrunk to keep the test fast.
"""
from pathlib import Path

import pandas as pd
import pytest
import torch

from resflow.models.features import all_feature_names

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / '_data' / 'port' / 'datasets' / 'spe10'
W = ROOT / 'weights' / 'state_2d_1d' / 'weights.pt'
have = (DS / 'train' / 'processed').is_dir() and W.is_file()
TINY = {'hidden_dims': [16, 8], 'enc_nodes_dims': [16], 'enc_edges_dims': [8], 'dec_nodes_dims': [16],
        'kernel_dims': [16], 'edge_mlp_dims': [8], 'n_blocks': 2, 'conv_rank': 2}


@pytest.fixture(scope='module')
def stats(tmp_path_factory):
    """A statistics file (fit-norm format) made from the shipped weights' statistics."""
    sd = torch.load(W, map_location='cpu', weights_only=True)
    path = tmp_path_factory.mktemp('stats') / 'lin_scaler.pt'
    torch.save({'scales': sd['data_norm_in.scales'], 'shifts': sd['data_norm_in.shifts'],
                'coords': sd['data_norm_in.coords'], 'feature_names': all_feature_names()}, path)
    return path


@pytest.mark.skipif(not have, reason='weights/ or _data/ absent')
def test_state_train_resume_rollout(tmp_path, stats):
    from resflow.models import load_bundle
    from resflow.rollout.cli import run_rollouts
    from resflow.train.state_trainer import train_state
    run = tmp_path / 'run'
    cfg = {'model': 'state', 'dataset': str(DS), 'run_dir': str(run), 'seed': 0,
           'norm_stats': str(stats), 'vols2p_weights': str(W), 'data': {'min_day': 0, 'max_day': 12},
           'network': TINY,
           'train': {'epochs': 2, 'steps_per_epoch': 4, 'block_schedule': [1, 2],
                     'pushforward': {'schedule': [0, 2]}, 'clean_eval': {'every': 4, 'n_sims': 1},
                     'save_every': 4, 'log_every': 1}}
    train_state(cfg, device='cpu')
    log = pd.read_csv(run / 'train_log.csv')
    assert log.step.max() == 8 and (run / 'train_state.pt').is_file()
    assert log.clean_loss.notna().any()
    train_state({'train': {'epochs': 3}}, resume=run, device='cpu')
    log = pd.read_csv(run / 'train_log.csv')
    assert log.step.max() == 12 and list(log.step) == sorted(log.step)
    model = load_bundle(run, 'cpu')
    assert model.model.processor.n_blocks == 2
    out = run_rollouts(run, DS, 'test', tmp_path / 'ro', sims=['L34_32x32_00_1P0I_fixed'], max_day=5.0,
                       device='cpu', verbose=False, report_days=())
    assert len(out) == 1


@pytest.mark.skipif(not have, reason='weights/ or _data/ absent')
def test_rate_train(tmp_path, stats):
    from resflow.models import load_bundle
    from resflow.train.rate_trainer import train_rate
    run = tmp_path / 'rate'
    cfg = {'model': 'rate', 'dataset': str(DS), 'run_dir': str(run), 'norm_stats': str(stats),
           'data': {'min_day': 0, 'max_day': 12},
           'train': {'epochs': 2, 'steps_per_epoch': 5, 'clean_every': 2, 'eval_sims': 1, 'log_every': 1,
                     'save_every': 5}}
    train_rate(cfg, device='cpu')
    log = pd.read_csv(run / 'train_log.csv')
    assert log.step.max() == 10 and log.val_loss.notna().any()
    assert type(load_bundle(run, 'cpu')).__name__ == 'RateModel'


@pytest.mark.skipif(not have, reason='weights/ or _data/ absent')
def test_finetune_keeps_normalisation(tmp_path, stats):
    from resflow.models import load_bundle
    from resflow.train.state_trainer import train_state
    run = tmp_path / 'ft'
    cfg = {'model': 'state', 'dataset': str(DS), 'run_dir': str(run), 'loss_norm_stats': str(stats),
           'data': {'min_day': 0, 'max_day': 6},
           'train': {'epochs': 1, 'steps_per_epoch': 2, 'lr': 8.0e-6, 'block_schedule': [12],
                     'pushforward': {'schedule': [2]}, 'clean_eval': {'every': None}, 'save_every': 2}}
    train_state(cfg, init_weights=ROOT / 'weights' / 'state_2d_1d', device='cpu')
    new = torch.load(run / 'weights.pt', weights_only=True)
    old = torch.load(W, weights_only=True)
    for k in ('data_norm_in.scales', 'data_norm_out.shifts', 'transform.pvt.sat'):
        assert torch.equal(new[k], old[k])
    assert not torch.equal(new['model.decoder.layers.0.0.weight'], old['model.decoder.layers.0.0.weight'])
    load_bundle(run, 'cpu')
