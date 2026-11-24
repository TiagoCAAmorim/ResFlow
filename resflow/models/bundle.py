"""Weights bundles: a folder with ``weights.pt`` (state dict) and ``config.yaml``.

``weights.pt`` holds every tensor the model needs: network weights, normalisation
statistics, PVT and relative-permeability tables and (state model) the vols2p pressure
estimator. ``config.yaml`` holds the architecture, the ordered input feature names, the
normalisation feature list and grouping, the asinh input settings, G and z_mean, the
step size and units. Loading needs nothing else: no dataset, no statistics file.

The training run folders written by `resflow.train` use the same two files (plus
``train_state.pt`` to resume), so a run folder is also a bundle.

Copyright (c) Tiago Amorim. MIT License.
"""
import hashlib
from pathlib import Path
from typing import Dict, Union

import torch
import yaml

from .rate import RateModel
from .state import StateModel

WEIGHTS_FILE = "weights.pt"
CONFIG_FILE = "config.yaml"
FORMAT_VERSION = 1


def read_config(path: Union[str, Path]) -> dict:
    """The config.yaml of a bundle folder."""
    with open(Path(path) / CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_bundle(path: Union[str, Path], device: Union[str, torch.device] = "cpu"):
    """Model (StateModel or RateModel, eval mode) from a bundle or run folder.

    The tensors are mapped to `device`, so weights saved on a GPU load on a CPU-only
    machine.
    """
    path = Path(path)
    config = read_config(path)
    state = torch.load(path / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    kind = config.get("model")
    if kind == "state":
        model = StateModel.from_state(config, state)
    elif kind == "rate":
        model = RateModel.from_state(config, state)
    else:
        raise ValueError(f"{path / CONFIG_FILE}: unknown model type '{kind}'.")
    return model.to(device).eval()


def model_state_dict(model) -> Dict[str, torch.Tensor]:
    """float32 CPU copy of a model's state dict."""
    return {k: (v.detach().float() if v.is_floating_point() else v.detach()).cpu().clone()
            for k, v in model.state_dict().items()}


def save_bundle(path: Union[str, Path], state: Dict[str, torch.Tensor], config: dict,
                write_md5: bool = False) -> Path:
    """Write weights.pt and config.yaml (and MD5SUMS) into `path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / (WEIGHTS_FILE + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path / WEIGHTS_FILE)
    config = {"format_version": FORMAT_VERSION, **config}
    with open(path / CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None, width=100)
    if write_md5:
        write_md5sums(path)
    return path


def write_md5sums(path: Union[str, Path], names=(WEIGHTS_FILE, CONFIG_FILE)) -> Path:
    """MD5SUMS file (``md5sum -c`` format) for the given files of a folder."""
    path = Path(path)
    lines = []
    for name in names:
        digest = hashlib.md5((path / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}\n")
    (path / "MD5SUMS").write_text("".join(lines), encoding="utf-8")
    return path / "MD5SUMS"
