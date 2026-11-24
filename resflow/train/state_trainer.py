"""Training of the state model (paper Section 3.9 and Table A-1).

One optimizer step = one time step of one simulation. An epoch visits the training
files in random order and each file's steps in random order, and ends after
``steps_per_epoch`` steps or when the files are exhausted. Per step:

1. Pushforward (from the epoch where the chain schedule is > 0): a chain length n is
   drawn uniformly from {0, ..., n_e}; the model is applied n times without gradient
   from the true state n steps before; the chain stops early at an unphysical state.
2. Loss: MSE of the normalised increments of P, Vo, Vg, Vw (channels ``loss_idxs`` of
   Rs, P, So, Sg, Sw, Vo, Vg, Vw; each divided by its training standard deviation; the
   statistics are the model's output statistics) over the cells of the dataset's training
   mask, of which non-well cells are dropped with probability ``train_mask.drop_frac`` at
   every step (seeded); well cells are always kept.
3. AdamW step with gradient clipping; EMA of the weights (the saved weights are the EMA).
4. Every ``clean_eval.every`` steps, a clean one-step loss (no pushforward, no noise) on
   ``clean_eval.n_sims`` random validation simulations steps ReduceLROnPlateau.

Optional (off in the paper recipe): Gaussian noise on the input state on the steps
without pushforward (``noise.std``, as a fraction of the per-cell one-step change).

The block curriculum runs only the first ``block_schedule[e]`` processor blocks in epoch e.

Not ported: the mass-balance and bias loss terms of the original training code (the
shipped weights were trained with them at weights 0.1 / 0.1).

Copyright (c) Tiago Amorim. MIT License.
"""
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader  # type: ignore  # noqa: F401  (documented dependency)

from ..data.dataset import load_datasets
from ..data.normalization import stats_path
from ..models.bundle import load_bundle, read_config
from ..models.features import (TARGET_NAMES, TARGET_NORM_NAMES, all_feature_names,
                               state_with_volumes, well_node_mask)
from ..models.state import StateModel, make_scaler, select_state_features
from ..physics.transform import Transform
from ..physics.vols2p import build_vols2p_net
from .common import (CSVLogger, WeightEMA, deep_update, epoch_value, load_train_state,
                     optimizer_to, read_run_config, resolve_path, rng_state, save_run,
                     seed_everything, set_rng_state)

TRAIN_DEFAULTS = {
    'epochs': 12,
    'steps_per_epoch': 108000,
    'lr': 5.0e-5,
    'weight_decay': 1.0e-4,
    'scheduler': {'factor': 0.98, 'patience': 5, 'min_lr': 1.0e-6},
    'clean_eval': {'every': 2000, 'n_sims': 4},
    'ema_decay': 0.999,
    'grad_clip': 1.0,
    'block_schedule': [8, 10, 12],
    'pushforward': {'schedule': [0, 1, 2, 3, 4, 4], 'randomize_depth': True},
    'loss_idxs': [1, 5, 6, 7],
    'train_mask': {'drop_frac': 0.5, 'seed': 42},
    'noise': {'std': 0.0, 'floor_frac': 0.0},
    'save_every': 2000,
    'log_every': 10,
    'max_consecutive_errors': 200,
}
LOG_COLUMNS = ['step', 'epoch', 'epoch_step', 'loss', 'clean_loss', 'lr', 'grad_norm',
               'pf_depth', 'blocks', 'minutes']


def is_physical_state(t: torch.Tensor, transform: Transform, sat_tol: float = 1e-3) -> bool:
    """Finite state with saturations in range and pressure inside the PVT table."""
    if not torch.isfinite(t).all():
        return False
    rs, p, so, sg = t[..., 0], t[..., 1], t[..., 2], t[..., 3]
    if (so < -sat_tol).any() or (sg < -sat_tol).any() or (so + sg > 1.0 + sat_tol).any():
        return False
    if (rs < -sat_tol).any():
        return False
    p_tab = transform.pvt.sat[:, 0]
    return bool((p >= p_tab.min()).all() and (p <= p_tab.max()).all())


def _split_or_fallback(splits: dict):
    """Validation split, or the test split for roots without one (fluvial set)."""
    return splits.get('val') if splits.get('val') is not None else splits.get('test')


@torch.no_grad()
def clean_step_loss(model: StateModel, dataset, device, loss_idxs, n_sims: int = 4,
                    generator: Optional[torch.Generator] = None, loss_norm=None) -> float:
    """Clean one-step loss (true input states, no pushforward, no noise, no mask drop)
    on `n_sims` random simulations of `dataset`; days with a non-finite prediction are
    left out."""
    was_training = model.training
    model.eval()
    try:
        names = dataset.get_sim_names(unique=True)
        perm = torch.randperm(len(names), generator=generator).tolist()
        sq, cnt = 0.0, 0
        for k in perm[:n_sims]:
            name = names[k]
            batch = dataset.get_sim_days(name, days=dataset.get_days(name)).to(device)
            vpor = batch.stat[:, 1]
            v0 = model.transform.t2vols(vpor, batch.x)
            y_true = state_with_volumes(model.transform, vpor, batch.y) - state_with_volumes(
                model.transform, vpor, batch.x, v0)
            v1 = v0 + model(batch)
            t1 = model.transform.vols2t(vpor, v1)
            y_hat = state_with_volumes(model.transform, vpor, t1, v1) - state_with_volumes(
                model.transform, vpor, batch.x, v0)
            norm = loss_norm or model.target_norm
            y_hat, y_true = norm(y_hat), norm(y_true)
            bad_day = (~torch.isfinite(y_hat).all(dim=-1)).flatten(1).any(dim=1)
            mask = batch.train_mask.clone()
            mask[bad_day] = False
            if not mask.any():
                continue
            err = (y_true[mask][..., loss_idxs] - y_hat[mask][..., loss_idxs]) ** 2
            sq += float(err.sum())
            cnt += err.numel()
        return sq / cnt if cnt else float('nan')
    finally:
        model.train(was_training)


def build_new_model(cfg: dict, train_dataset, device) -> StateModel:
    """A new state model: statistics from the training set (``norm_stats`` or the
    dataset's fit-norm file), physics tables from its raw folders, vols2p estimator from
    a weights file (``vols2p_weights``), features selected from the statistics."""
    root = resolve_path(cfg['dataset'])
    norm_file = resolve_path(cfg.get('norm_stats')) or stats_path(root)
    if not Path(norm_file).is_file():
        raise SystemExit(f"No normalisation statistics at {norm_file}: run `resflow fit-norm "
                         f"--dataset {root}` first (or set norm_stats).")
    norm_in, norm_out = make_scaler(device), make_scaler(device)
    for s in (norm_in, norm_out):
        if not s.load_stats(norm_file):
            raise SystemExit(f"Invalid statistics file {norm_file}.")
        if s.feature_names != all_feature_names():
            raise SystemExit(f"{norm_file} does not hold the feature registry of this version.")
    vols2p_file = resolve_path(cfg.get('vols2p_weights'))
    if vols2p_file is None or not Path(vols2p_file).is_file():
        raise SystemExit("vols2p_weights must point to a weights file holding the vols2p "
                         "estimator (e.g. weights/state_2d_1d/weights.pt).")
    vstate = torch.load(vols2p_file, map_location='cpu', weights_only=True)
    phys = cfg.get('physics', {})
    transform = Transform.from_folder(Path(train_dataset.raw_dir),
                                      vols2p_nn=build_vols2p_net(vstate, 'transform.vols2p_nn.'),
                                      g=float(phys.get('g', 1e-4)), z_mean=phys.get('z_mean'))
    feats = cfg.get('features', {})
    nodes, edges = select_state_features(norm_in, feats.get('use_beta_ig', False),
                                         feats.get('use_beta_iw', True))
    model_cfg = model_config_from(cfg, nodes, edges, step_size=train_dataset.step_size)
    return StateModel(model_cfg, transform, norm_in, norm_out).to(device)


def model_config_from(cfg: dict, nodes, edges, step_size: float) -> dict:
    """The bundle config of a new model from a training config."""
    from .. models.state import NETWORK_DEFAULTS  # pylint: disable=import-outside-toplevel
    return {
        'model': 'state',
        'network': {**NETWORK_DEFAULTS, **cfg.get('network', {})},
        'node_features': list(nodes),
        'edge_features': list(edges),
        'normalization': {'type': 'linear', 'grouped': True, 'feature_names': all_feature_names(),
                          'state_keys': ['data_norm_in.', 'data_norm_out.']},
        'input_asinh': {k: [float(v[0]), float(v[1])] for k, v in (cfg.get('input_asinh') or {}).items()},
        'well_type_embedding': cfg.get('well_type_embedding',
                                       {'num_types': 3, 'dim': 4, 'active_eps': 1.0e-4}),
        'outputs': ['dVo_std', 'dVg_std', 'dVw_std'],
        'physics': {'g': float(cfg.get('physics', {}).get('g', 1e-4)),
                    'z_mean': cfg.get('physics', {}).get('z_mean')},
        'model_batch_limit': cfg.get('model_batch_limit', 16),
        'step_size': float(step_size),
    }


def train_state(cfg: dict, resume: Optional[Path] = None, init_weights: Optional[Path] = None,
                device: Optional[str] = None) -> Path:
    """Train (from scratch, from a bundle with `init_weights`, or resuming a run folder).

    Returns the run folder. See the module docstring and configs/state_2d_1d.yaml.
    """
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
                           min_day=data.get('min_day', 0.0), max_day=data.get('max_day', 1000.0))
    train_ds, val_ds = splits['train'], _split_or_fallback(splits)
    if train_ds is None:
        raise SystemExit(f"No built train split under {root}.")
    step_size = float(train_ds.step_size)

    # Model
    if resume_state is not None:
        model = StateModel.from_state(saved, resume_state['live'], device)
        model_cfg = {k: v for k, v in saved.items() if k not in ('training_config', 'format_version')}
    elif init_weights is not None:
        model = load_bundle(init_weights, device)
        if not isinstance(model, StateModel):
            raise SystemExit(f"{init_weights} is not a state model bundle.")
        model_cfg = {k: v for k, v in read_config(init_weights).items()
                     if k not in ('training', 'training_config', 'format_version')}
        model_cfg['started_from'] = Path(init_weights).name
    else:
        model = build_new_model(cfg, train_ds, device)
        model_cfg = model.config
    model.train()
    model.model_batch_limit = cfg.get('model_batch_limit', model_cfg.get('model_batch_limit', 16))
    if abs(float(model_cfg.get('step_size', step_size)) - step_size) > 1e-9:
        print(f"Note: model step {model_cfg.get('step_size')} d, dataset step {step_size} d; "
              f"the trained model uses the dataset's.")
    model_cfg['step_size'] = step_size
    model_cfg['training_config'] = cfg

    # Loss statistics. From scratch: the model's own output statistics (fitted on this
    # dataset). Fine-tune (as the original trainer): the fine-tune dataset's statistics
    # (`fit-norm` on it), while the model's input/output scalers stay those of the weights.
    if init_weights is not None and not cfg.get('loss_norm_stats'):
        cfg['loss_norm_stats'] = str(stats_path(root))
    loss_norm = model.target_norm
    if cfg.get('loss_norm_stats'):
        lfile = resolve_path(cfg['loss_norm_stats'])
        if not Path(lfile).is_file():
            raise SystemExit(f"No loss statistics at {lfile}: run `resflow fit-norm --dataset {root}` "
                             f"(the fine-tune loss uses the fine-tune dataset's statistics) or set "
                             f"loss_norm_stats.")
        loss_scaler = make_scaler(device)
        if not loss_scaler.load_stats(lfile) or loss_scaler.feature_names != all_feature_names():
            raise SystemExit(f"Invalid statistics file {lfile}.")
        loss_norm = lambda y: loss_scaler.norm(y, TARGET_NORM_NAMES, grouped=True)  # noqa: E731
        print(f"Loss normalised with the statistics of {lfile}.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(tcfg['lr']), weight_decay=float(tcfg['weight_decay']))
    sch = tcfg['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=float(sch['factor']), patience=int(sch['patience']),
        min_lr=float(sch['min_lr']))
    ema = WeightEMA(model, float(tcfg['ema_decay'])) if tcfg.get('ema_decay') else None

    step, start_epoch, start_pos = 0, 1, 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state['optimizer'])
        optimizer_to(optimizer, device)
        scheduler.load_state_dict(resume_state['scheduler'])
        if ema is not None and resume_state.get('ema'):
            ema.load_shadow(resume_state['ema'])
        step, start_epoch, start_pos = (resume_state['step'], resume_state['epoch'],
                                        resume_state['epoch_step'])
        set_rng_state(resume_state['rng'])
        print(f"Resumed {run_dir} at step {step} (epoch {start_epoch}, step {start_pos} of it).")

    logger = CSVLogger(run_dir / 'train_log.csv', LOG_COLUMNS, resume=resume_state is not None)
    loss_idxs = list(tcfg['loss_idxs'])
    drop_frac = float(tcfg['train_mask']['drop_frac'])
    mask_seed = tcfg['train_mask'].get('seed')
    pf_cfg = tcfg['pushforward']
    noise_std = float(tcfg.get('noise', {}).get('std', 0.0))
    noise_floor = float(tcfg.get('noise', {}).get('floor_frac', 0.0))
    if noise_std > 0:
        base_scale = model.data_norm_out.get_scale(TARGET_NAMES[:4]).to(device)
        base_scale = torch.where(base_scale < 1e-7, torch.zeros_like(base_scale), base_scale)
    clean_gen = torch.Generator().manual_seed(int(seed or 0) + 1)
    if resume_state is not None and 'clean_gen' in resume_state:
        clean_gen.set_state(resume_state['clean_gen'])
    max_day = float(data.get('max_day', 1000.0))
    min_day = float(data.get('min_day', 0.0))
    t0 = time.monotonic()
    last_clean = float(resume_state.get('last_clean', float('nan'))) if resume_state else float('nan')
    consecutive_errors = 0
    epoch_step = start_pos

    # step counts at the last clean evaluation / log line / checkpoint
    marks = dict(resume_state.get('marks', {}) if resume_state is not None else {},)
    for key in ('clean', 'log', 'save'):
        marks.setdefault(key, step)

    def checkpoint(epoch, epoch_step):
        save_run(run_dir, model, model_cfg, ema, {
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'step': step, 'epoch': epoch, 'epoch_step': epoch_step, 'rng': rng_state(),
            'clean_gen': clean_gen.get_state(), 'marks': dict(marks), 'last_clean': last_clean})

    n_files = len(train_ds)
    for epoch in range(start_epoch, int(tcfg['epochs']) + 1):
        blocks = epoch_value(tcfg.get('block_schedule'), epoch)
        model.model.processor.active_blocks = blocks
        pf_target = epoch_value(pf_cfg.get('schedule'), epoch) or 0
        print(f"Epoch {epoch}: {blocks or model.model.processor.n_blocks} blocks, "
              f"pushforward chain up to {pf_target}", flush=True)
        g_epoch = torch.Generator().manual_seed(int(seed or 0) * 1000 + epoch)
        file_order = torch.randperm(n_files, generator=g_epoch).tolist()
        skip = epoch_step if epoch == start_epoch else 0
        epoch_step = 0
        done_epoch = False
        for fi in file_order:
            sample = train_ds[fi]
            if sample is None:
                continue
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
            idx = int(sample.global_attr[0].item())
            vpor = sample.stat[:, 1]
            with torch.no_grad():
                tv_all = state_with_volumes(model.transform, vpor, sample.x)
            wells = well_node_mask(sample)
            for ts in order:
                epoch_step += 1
                step += 1
                batch = train_ds.slice_data(sample, ts, ts + 1)
                day0 = batch.global_attr[1].item()
                if (batch.weights.sum().item() == 0 or day0 > max_day
                        or day0 + step_size < min_day + step_size):
                    continue
                t_x = batch.x
                v_x = tv_all[ts:ts + 1, :, 5:8]
                y_true = tv_all[ts + 1:ts + 2] - tv_all[ts:ts + 1]
                depth = 0
                pf_zero = False
                if pf_target > 0 and ts + 2 <= t_steps:
                    n = (int(torch.randint(0, pf_target + 1, (1,)).item())
                         if pf_cfg.get('randomize_depth', True) else pf_target)
                    pf_zero = n == 0
                    t_cur, v_cur, b_cur = t_x, v_x, batch
                    for k in range(n):
                        if ts + 1 + k + 1 > t_steps:
                            break
                        try:
                            with torch.no_grad():
                                t_next = model.transform.vols2t(vpor, v_cur + model(b_cur)).float()
                        except ValueError:
                            break
                        if not is_physical_state(t_next, model.transform):
                            break
                        nb = train_ds.slice_data(sample, ts + k + 1, ts + k + 2)
                        if nb.weights.sum().item() == 0:
                            break
                        nb.x = t_next
                        t_cur, v_cur, b_cur = t_next, model.transform.t2vols(vpor, t_next), nb
                        depth = k + 1
                    if depth:
                        batch, t_x, v_x = b_cur, t_cur, v_cur
                        y_true = (tv_all[ts + depth + 1:ts + depth + 2]
                                  - state_with_volumes(model.transform, vpor, t_cur, v_cur))
                if noise_std > 0 and depth == 0 and not pf_zero:
                    t_y = sample.x[ts + 1:ts + 2]
                    scale = (t_y - t_x).abs() + noise_floor * base_scale
                    t_noisy = t_x + torch.randn_like(t_x) * noise_std * scale * (base_scale > 0)
                    if is_physical_state(t_noisy, model.transform):
                        batch = train_ds.slice_data(sample, ts, ts + 1)
                        batch.x = t_noisy
                        t_x, v_x = t_noisy, model.transform.t2vols(vpor, t_noisy)
                        y_true = tv_all[ts + 1:ts + 2] - state_with_volumes(model.transform, vpor, t_x, v_x)
                try:
                    dv = model(batch)
                except ValueError as exc:
                    consecutive_errors += 1
                    if consecutive_errors > int(tcfg['max_consecutive_errors']):
                        raise RuntimeError(f"{consecutive_errors} consecutive forward errors") from exc
                    continue
                consecutive_errors = 0
                v_hat = v_x + dv
                t_hat = model.transform.vols2t(vpor, v_hat)
                y_hat = state_with_volumes(model.transform, vpor, t_hat, v_hat) - (
                    state_with_volumes(model.transform, vpor, t_x, v_x))
                y_hat_n, y_true_n = loss_norm(y_hat), loss_norm(y_true)
                nan_mask = torch.isnan(y_hat_n).any(dim=-1)
                mask = batch.train_mask.clone()
                if 0.0 < drop_frac < 1.0:
                    keep_wells = mask[..., wells].clone()
                    gen = None
                    if mask_seed is not None:
                        gen = torch.Generator(device=mask.device)
                        gen.manual_seed(int(mask_seed) + idx + epoch_step)
                    mask &= torch.rand(mask.shape, generator=gen, device=mask.device) >= drop_frac
                    mask[..., wells] = keep_wells
                mask[nan_mask] = False
                if not mask.any() or nan_mask.any():
                    continue
                loss = F.mse_loss(y_hat_n[mask][..., loss_idxs], y_true_n[mask][..., loss_idxs])
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float(tcfg['grad_clip']))
                if torch.isfinite(grad_norm):
                    optimizer.step()
                    if ema is not None:
                        ema.update()
                optimizer.zero_grad(set_to_none=True)

                every = tcfg['clean_eval'].get('every')
                evaluated = False
                if every and val_ds is not None and step - marks['clean'] >= int(every):
                    marks['clean'] = step
                    evaluated = True
                    last_clean = clean_step_loss(model, val_ds, device, loss_idxs,
                                                 int(tcfg['clean_eval']['n_sims']), clean_gen,
                                                 loss_norm)
                    if math.isfinite(last_clean):
                        scheduler.step(last_clean)
                if step - marks['log'] >= int(tcfg['log_every']) or evaluated:
                    marks['log'] = step
                    logger.log({'step': step, 'epoch': epoch, 'epoch_step': epoch_step,
                                'loss': float(loss), 'clean_loss': last_clean,
                                'lr': optimizer.param_groups[0]['lr'], 'grad_norm': float(grad_norm),
                                'pf_depth': depth, 'blocks': blocks or model.model.processor.n_blocks,
                                'minutes': (time.monotonic() - t0) / 60})
                if step - marks['save'] >= int(tcfg['save_every']):
                    marks['save'] = step
                    checkpoint(epoch, epoch_step)
                if epoch_step >= int(tcfg['steps_per_epoch']):
                    done_epoch = True
                    break
            if done_epoch:
                break
        checkpoint(epoch + 1, 0)
        print(f"Epoch {epoch} done: {epoch_step} steps (total {step}).", flush=True)
        epoch_step = 0
    print(f"Training finished: {run_dir}", flush=True)
    return run_dir
