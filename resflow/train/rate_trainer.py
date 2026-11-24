"""Training of the rate model (paper Table A-1, single run).

One optimizer step = one time step of one simulation. An epoch visits the training
simulations in random order (all days of the window) and their steps in random order,
for ``steps_per_epoch`` steps (default: one pass over every step of the training set).
Per step: Gaussian noise on the input state (``noise_std`` times the training standard
deviation of the one-step change of each state channel; an unphysical noisy state is
rejected and the true one used), loss = MSE of the normalised well volumes (dNp, dGp,
dWp, dWi, each with its own statistics) over the well cells, AdamW, gradient clipping,
EMA (saved weights = EMA). Learning rate: linear warm-up, then cosine decay to
``min_lr`` at the last step.

Copyright (c) Tiago Amorim. MIT License.
"""
import math
import time
from pathlib import Path
from typing import Optional

import torch

from ..data.dataset import load_datasets
from ..data.normalization import stats_path
from ..models.bundle import load_bundle, read_config
from ..models.features import TARGET_NAMES, all_feature_names, well_node_mask
from ..models.rate import NETWORK_DEFAULTS, RateModel, rate_targets, select_rate_features
from ..models.state import make_scaler
from ..physics.transform import Transform
from .common import (CSVLogger, WeightEMA, deep_update, load_train_state, optimizer_to,
                     read_run_config, resolve_path, rng_state, save_run, seed_everything,
                     set_rng_state)
from .state_trainer import is_physical_state

TRAIN_DEFAULTS = {
    'epochs': 7,
    'steps_per_epoch': None,      # None: every step of the training set (182,106 in the paper)
    'lr': 1.0e-4,
    'min_lr': 1.0e-6,
    'warmup_steps': 200,
    'warmup_start_factor': 0.01,
    'weight_decay': 1.0e-4,
    'grad_clip': 1.0,
    'ema_decay': 0.999,
    'noise_std': 0.05,
    'clean_every': 500,
    'eval_sims': 4,
    'save_every': 5000,
    'log_every': 50,
}
LOG_COLUMNS = ['step', 'epoch', 'epoch_step', 'loss', 'clean_loss', 'val_loss', 'lr',
               'noise_used', 'noise_rejected', 'minutes']


def warmup_cosine(step: int, warmup: int, total: int, start: float, min_factor: float) -> float:
    """LR multiplier: linear warm-up from `start` to 1, then cosine to `min_factor`."""
    if step < warmup:
        return start + (1.0 - start) * step / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))


def masked_mse(model: RateModel, batch, wells) -> torch.Tensor:
    y_hat = model.target_norm(model(batch))
    y_true = model.target_norm(rate_targets(batch))
    return ((y_hat - y_true)[..., wells, :] ** 2).mean()


@torch.no_grad()
def eval_loss(model: RateModel, dataset, device, n_sims: int, generator) -> float:
    """Masked MSE on `n_sims` random simulations (true states, all days)."""
    was = model.training
    model.eval()
    try:
        names = dataset.get_sim_names(unique=True)
        vals = []
        for k in torch.randperm(len(names), generator=generator).tolist()[:n_sims]:
            data = dataset.get_sim_days(names[k]).to(device)
            wells = well_node_mask(data)
            if wells.any():
                v = masked_mse(model, data, wells)
                if torch.isfinite(v):
                    vals.append(float(v))
        return sum(vals) / len(vals) if vals else float('nan')
    finally:
        model.train(was)


def build_new_model(cfg: dict, train_dataset, device) -> RateModel:
    """A new rate model: statistics from the training set, PVT/KR from its raw folders."""
    root = resolve_path(cfg['dataset'])
    norm_file = resolve_path(cfg.get('norm_stats')) or stats_path(root)
    if not Path(norm_file).is_file():
        raise SystemExit(f"No normalisation statistics at {norm_file}: run `resflow fit-norm "
                         f"--dataset {root}` first (or set norm_stats).")
    norm = make_scaler(device)
    if not norm.load_stats(norm_file) or norm.feature_names != all_feature_names():
        raise SystemExit(f"Invalid statistics file {norm_file}.")
    phys = cfg.get('physics', {})
    transform = Transform.from_folder(Path(train_dataset.raw_dir), g=float(phys.get('g', 1e-4)),
                                      z_mean=phys.get('z_mean'))
    feats = cfg.get('features', {})
    nodes, edges = select_rate_features(norm, feats.get('use_beta_ig', False),
                                        feats.get('use_beta_iw', True))
    model_cfg = {
        'model': 'rate',
        'network': {**NETWORK_DEFAULTS, **cfg.get('network', {})},
        'node_features': nodes,
        'edge_features': edges,
        'normalization': {'type': 'linear', 'grouped': False, 'feature_names': all_feature_names(),
                          'state_keys': ['data_norm.']},
        'well_type_embedding': {'num_types': 4, 'dim': int(cfg.get('well_type_embed_dim', 4))},
        'outputs': ['dNp', 'dGp', 'dWp', 'dWi'],
        'physics': {'g': float(phys.get('g', 1e-4)), 'z_mean': phys.get('z_mean')},
        'step_size': float(train_dataset.step_size),
    }
    return RateModel(model_cfg, transform, norm).to(device)


def train_rate(cfg: dict, resume: Optional[Path] = None, init_weights: Optional[Path] = None,
               device: Optional[str] = None) -> Path:
    """Train the rate model (from scratch, from a bundle, or resuming a run folder)."""
    device = device or cfg.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    resume_state = None
    if resume is not None:
        resume = Path(resume)
        saved = read_run_config(resume)
        cfg = deep_update(saved.get('training_config', {}), {k: v for k, v in cfg.items() if k == 'train'})
        resume_state = load_train_state(resume)
        cfg['run_dir'] = str(resume)
    tcfg = deep_update(TRAIN_DEFAULTS, cfg.get('train', {}))
    run_dir = Path(cfg['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg.get('seed', 0)
    seed_everything(seed)
    data = cfg.get('data', {})
    root = resolve_path(cfg['dataset'])
    splits = load_datasets(root, load_train=True, load_val=True, load_test=True,
                           min_day=data.get('min_day', 0.0), max_day=data.get('max_day', 1827.0))
    train_ds = splits['train']
    val_ds = splits['val'] if splits['val'] is not None else splits['test']
    if train_ds is None:
        raise SystemExit(f"No built train split under {root}.")

    if resume_state is not None:
        model = RateModel.from_state(saved, resume_state['live'], device)
        model_cfg = {k: v for k, v in saved.items() if k not in ('training_config', 'format_version')}
    elif init_weights is not None:
        model = load_bundle(init_weights, device)
        if not isinstance(model, RateModel):
            raise SystemExit(f"{init_weights} is not a rate model bundle.")
        model_cfg = {k: v for k, v in read_config(init_weights).items()
                     if k not in ('training', 'training_config', 'format_version')}
    else:
        model = build_new_model(cfg, train_ds, device)
        model_cfg = model.config
    model.step_size = float(train_ds.step_size)
    model_cfg['step_size'] = float(train_ds.step_size)
    model_cfg['training_config'] = cfg
    model.train()

    epoch_len = int(tcfg['steps_per_epoch'] or train_ds.get_n_samples())
    total = int(tcfg['epochs']) * epoch_len
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(tcfg['lr']), weight_decay=float(tcfg['weight_decay']))
    min_factor = float(tcfg['min_lr']) / float(tcfg['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine(
        s, int(tcfg['warmup_steps']), total, float(tcfg['warmup_start_factor']), min_factor))
    ema = WeightEMA(model, float(tcfg['ema_decay'])) if tcfg.get('ema_decay') else None

    step, start_epoch, start_pos = 0, 1, 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(resume_state['scheduler'])
        for g, base in zip(optimizer.param_groups, scheduler.base_lrs):
            g['lr'] = base * scheduler.lr_lambdas[0](scheduler.last_epoch)
        if ema is not None and resume_state.get('ema'):
            ema.load_shadow(resume_state['ema'])
        step, start_epoch, start_pos = (resume_state['step'], resume_state['epoch'],
                                        resume_state['epoch_step'])
        set_rng_state(resume_state['rng'])
        print(f"Resumed {run_dir} at step {step} (epoch {start_epoch}, step {start_pos} of it).")

    scale = model.data_norm.get_scale(TARGET_NAMES[:4]).to(device)
    active = (scale > 0).to(scale.dtype)
    noise_std = float(tcfg['noise_std'])
    logger = CSVLogger(run_dir / 'train_log.csv', LOG_COLUMNS, resume=resume_state is not None)
    eval_gen = torch.Generator().manual_seed(int(seed or 0) + 1)
    noise_used = noise_rejected = 0
    clean = val = float('nan')
    t0 = time.monotonic()
    ema_ctx = (lambda: ema.applied(model)) if ema is not None else None

    def checkpoint(epoch, epoch_step):
        save_run(run_dir, model, model_cfg, ema, {
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'step': step, 'epoch': epoch, 'epoch_step': epoch_step, 'rng': rng_state()})

    epoch_step = start_pos
    for epoch in range(start_epoch, int(tcfg['epochs']) + 1):
        g_epoch = torch.Generator().manual_seed(int(seed or 0) * 1000 + epoch)
        names = train_ds.get_sim_names(unique=True)
        sim_order = [names[k] for k in torch.randperm(len(names), generator=g_epoch).tolist()]
        skip = epoch_step if epoch == start_epoch else 0
        epoch_step = 0
        full = False
        for name in sim_order:
            sample = train_ds.get_sim_days(name)
            t_steps = int(sample.global_attr[4].item())
            order = torch.randperm(t_steps, generator=g_epoch).tolist()
            if skip >= len(order):
                skip -= len(order)
                epoch_step += len(order)
                continue
            if skip:
                epoch_step += skip
                order = order[skip:]
                skip = 0
            sample = sample.to(device)
            wells = well_node_mask(sample)
            if not wells.any():
                epoch_step += len(order)
                continue
            for ts in order:
                batch = train_ds.slice_data(sample, ts, ts + 1)
                x_clean = batch.x
                if noise_std > 0:
                    t_noisy = batch.x + torch.randn_like(batch.x) * (noise_std * scale) * active
                    if is_physical_state(t_noisy, model.transform):
                        batch.x = t_noisy
                        noise_used += 1
                    else:
                        noise_rejected += 1
                loss = masked_mse(model, batch, wells)
                step += 1
                epoch_step += 1
                if torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, float(tcfg['grad_clip']))
                    optimizer.step()
                    if ema is not None:
                        ema.update()
                scheduler.step()
                if step % int(tcfg['clean_every']) == 0:
                    batch.x = x_clean
                    with torch.no_grad():
                        model.eval()
                        if ema_ctx is not None:
                            with ema_ctx():
                                clean = float(masked_mse(model, batch, wells))
                        else:
                            clean = float(masked_mse(model, batch, wells))
                        model.train()
                if step % int(tcfg['log_every']) == 0:
                    logger.log({'step': step, 'epoch': epoch, 'epoch_step': epoch_step,
                                'loss': float(loss), 'clean_loss': clean, 'val_loss': val,
                                'lr': optimizer.param_groups[0]['lr'], 'noise_used': noise_used,
                                'noise_rejected': noise_rejected,
                                'minutes': (time.monotonic() - t0) / 60})
                if step % int(tcfg['save_every']) == 0:
                    checkpoint(epoch, epoch_step)
                if epoch_step >= epoch_len:
                    full = True
                    break
            if full:
                break
        if val_ds is not None:
            if ema_ctx is not None:
                with ema_ctx():
                    val = eval_loss(model, val_ds, device, int(tcfg['eval_sims']), eval_gen)
            else:
                val = eval_loss(model, val_ds, device, int(tcfg['eval_sims']), eval_gen)
            logger.log({'step': step, 'epoch': epoch, 'epoch_step': epoch_step, 'val_loss': val,
                        'lr': optimizer.param_groups[0]['lr'],
                        'minutes': (time.monotonic() - t0) / 60})
        checkpoint(epoch + 1, 0)
        print(f"Epoch {epoch} done: {epoch_step} steps (total {step}).", flush=True)
        epoch_step = 0
    print(f"Training finished: {run_dir}", flush=True)
    return run_dir
