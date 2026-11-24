"""Shared training utilities: config files, EMA, run folders, CSV log, resume state.

A run folder holds ``weights.pt`` (the EMA weights, or the live ones without EMA) and
``config.yaml`` (the model config plus the training config used), so it loads like a
weights bundle; ``train_state.pt`` (live weights, EMA, optimizer, scheduler, counters,
random states) lets a run resume; ``train_log.csv`` is the log.

Copyright (c) Tiago Amorim. MIT License.
"""
import copy
import csv
import random
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import torch
import yaml

from ..models.bundle import CONFIG_FILE, WEIGHTS_FILE, model_state_dict, save_bundle

TRAIN_STATE_FILE = "train_state.pt"
LOG_FILE = "train_log.csv"


def load_config(path: Union[str, Path], overrides: Optional[Iterable[str]] = None) -> dict:
    """Read a training config (YAML) and apply ``key.sub=value`` overrides (YAML values)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg: dict, overrides: Optional[Iterable[str]] = None) -> dict:
    """Apply ``key.sub=value`` overrides (values parsed as YAML) to a config dict."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' is not key=value.")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def resolve_path(value, base: Optional[Path] = None) -> Optional[Path]:
    """Path from a config value (relative paths are relative to the working directory)."""
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() or base is None else base / p


def epoch_value(schedule, epoch: int):
    """Entry of a per-epoch schedule for `epoch` (1-based); the last entry holds afterwards."""
    if not schedule:
        return None
    return schedule[min(epoch, len(schedule)) - 1]


def seed_everything(seed: Optional[int]) -> None:
    """Seed python, numpy and torch."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state() -> dict:
    """Random states of python, numpy and torch (CPU and CUDA)."""
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore the states saved by `rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class WeightEMA:
    """Exponential moving average of the trainable weights.

    `update()` after every optimizer step that ran; `state_dict(model)` is the model's
    state dict with the averaged weights in place of the live ones (buffers and frozen
    tensors from the live model)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {decay}.")
        self.decay = decay
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.shadow = [p.detach().clone() for _, p in self.named]

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_mul_(self.shadow, self.decay)  # pylint: disable=protected-access
        torch._foreach_add_(self.shadow, [p.detach() for _, p in self.named],  # pylint: disable=protected-access
                            alpha=1.0 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for (name, _), s in zip(self.named, self.shadow):
            sd[name] = s.clone()
        return sd

    def shadow_dict(self) -> Dict[str, torch.Tensor]:
        return {n: s.detach().cpu().clone() for (n, _), s in zip(self.named, self.shadow)}

    def load_shadow(self, shadow: Dict[str, torch.Tensor]) -> None:
        for i, (n, p) in enumerate(self.named):
            self.shadow[i] = shadow[n].to(p.device).clone()

    def applied(self, model):
        """Context manager: the averaged weights in the model inside the block."""
        ema = self

        class _Ctx:
            def __enter__(self):
                self.backup = [p.detach().clone() for _, p in ema.named]
                with torch.no_grad():
                    for (_, p), s in zip(ema.named, ema.shadow):
                        p.copy_(s)

            def __exit__(self, *exc):
                with torch.no_grad():
                    for (_, p), b in zip(ema.named, self.backup):
                        p.copy_(b)
                return False
        return _Ctx()


class CSVLogger:
    """Append rows (dicts) to a CSV file and echo a compact line to the console."""

    def __init__(self, path: Path, columns, resume: bool = False):
        self.path = Path(path)
        self.columns = list(columns)
        if not resume or not self.path.is_file():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.columns)
        self.t0 = time.monotonic()

    def log(self, row: dict, echo: bool = True) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(c, "") for c in self.columns])
        if echo:
            parts = []
            for c in self.columns:
                v = row.get(c, "")
                if v == "" or v is None:
                    continue
                parts.append(f"{c}={v:.4g}" if isinstance(v, float) else f"{c}={v}")
            print("  ".join(parts), flush=True)


def save_run(run_dir: Path, model, config: dict, ema: Optional[WeightEMA],
             train_state: Optional[dict] = None) -> None:
    """weights.pt (EMA weights when EMA is on), config.yaml and train_state.pt."""
    run_dir = Path(run_dir)
    weights = ema.state_dict(model) if ema is not None else model.state_dict()
    weights = {k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu()
               for k, v in weights.items()}
    save_bundle(run_dir, weights, config)
    if train_state is not None:
        state = {"live": model_state_dict(model), **train_state}
        if ema is not None:
            state["ema"] = ema.shadow_dict()
        tmp = run_dir / (TRAIN_STATE_FILE + ".tmp")
        torch.save(state, tmp)
        tmp.replace(run_dir / TRAIN_STATE_FILE)


def load_train_state(run_dir: Path) -> dict:
    """train_state.pt of a run folder (CPU tensors)."""
    path = Path(run_dir) / TRAIN_STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: not a run folder that can be resumed.")
    return torch.load(path, map_location="cpu", weights_only=False)


def read_run_config(run_dir: Path) -> dict:
    """config.yaml of a run folder."""
    with open(Path(run_dir) / CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, upd: dict) -> dict:
    """Recursive dict update (returns a new dict)."""
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def optimizer_to(optimizer: torch.optim.Optimizer, device) -> None:
    """Move an optimizer's state tensors to `device`."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v) and k != "step":
                state[k] = v.to(device)


__all__ = ["load_config", "epoch_value", "seed_everything", "rng_state", "set_rng_state",
           "WeightEMA", "CSVLogger", "save_run", "load_train_state", "read_run_config",
           "deep_update", "optimizer_to", "WEIGHTS_FILE", "CONFIG_FILE", "resolve_path"]
