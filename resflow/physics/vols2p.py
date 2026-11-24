"""Pressure estimator network (vols2p): standard volumes per pore volume -> pressure.

The network gives the initial pressure of the equilibrium solver in
`Transform.vols2t` (a few Ridder iterations refine it). Its weights are not trained
here: they are stored inside the model weights (keys ``transform.vols2p_nn.*``).

Architecture: input normalisation (learnable affine, 3 inputs: vo, vg, vw per reference
pore volume), 11 engineered features, a mixture of 3 experts (MLP 11-32-64-128-64-32-1
with LayerNorm and GELU) mixed by a gating network, and an output affine layer giving
pressure in kgf/cm2.
Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Mapping, Optional

import torch
from torch import nn

FEAT_DIM = 11
DEFAULT_CONFIG = {
    "mlp_layers": [32, 64, 128, 64, 32],
    "activation": "gelu",
    "dropout": 0.0,
    "num_experts": 3,
}


class LearnableAffine(nn.Module):
    """y = scale * x + bias (per feature)."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


def _activation(name: str) -> nn.Module:
    return {"gelu": nn.GELU(), "silu": nn.SiLU(), "leakyrelu": nn.LeakyReLU(0.01),
            "tanh": nn.Tanh()}[name]


def make_features(x: torch.Tensor) -> torch.Tensor:
    """Expand (B, 3) into (B, 11) features: x, pairwise products, squares, sum, triple product."""
    x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([
        x1, x2, x3,
        x1 * x2, x1 * x3, x2 * x3,
        x1 ** 2, x2 ** 2, x3 ** 2,
        x1 + x2 + x3,
        x1 * x2 * x3,
    ], dim=1)


class _FeatureEngineering(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return make_features(x)


class _MLP(nn.Module):
    """Linear -> LayerNorm -> activation -> dropout blocks, last layer linear."""

    def __init__(self, dims, activation: str, dropout: float):
        super().__init__()
        layers = []
        n_linear = len(dims) - 1
        for i, (d_in, d_out) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(d_in, d_out))
            if i < n_linear - 1:
                layers.append(nn.LayerNorm(d_out))
                layers.append(_activation(activation))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Expert(nn.Module):
    def __init__(self, mlp_layers, activation, dropout):
        super().__init__()
        self.net = _MLP([FEAT_DIM] + list(mlp_layers) + [1], activation, dropout)

    def forward(self, x):
        return self.net(x)


class _Gating(nn.Module):
    def __init__(self, num_experts: int, activation: str):
        super().__init__()
        hidden = max(64, FEAT_DIM * 2)
        self.net = nn.Sequential(
            nn.Linear(FEAT_DIM, hidden), _activation(activation), nn.Linear(hidden, num_experts))

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class MixtureOfExpertsNet(nn.Module):
    """Input norm -> features -> gated experts -> output affine. Returns (B,)."""

    def __init__(self, mlp_layers, activation="gelu", dropout=0.0, num_experts=3):
        super().__init__()
        self.input_norm = LearnableAffine(3)
        self.experts = nn.ModuleList(
            [_Expert(mlp_layers, activation, dropout) for _ in range(num_experts)])
        self.gating = _Gating(num_experts, activation)
        self.features = _FeatureEngineering()
        self.output_denorm = LearnableAffine(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(self.input_norm(x))
        weights = self.gating(feat).unsqueeze(-1)  # (B, num_experts, 1)
        outputs = torch.stack([expert(feat) for expert in self.experts], dim=1)
        out = (weights * outputs).sum(dim=1)  # (B, 1)
        return self.output_denorm(out).squeeze(-1)


class Vols2PNet(nn.Module):
    """Wrapper that returns shape (B, 1); the network is `self.model` (state-dict prefix)."""

    def __init__(self, config: Optional[Mapping] = None):
        super().__init__()
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self.model = MixtureOfExpertsNet(
            cfg["mlp_layers"], cfg["activation"], cfg["dropout"], cfg["num_experts"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).unsqueeze(-1)


def build_vols2p_net(state: Optional[Mapping] = None, prefix: str = "", device="cpu",
                     config: Optional[Mapping] = None) -> Vols2PNet:
    """Build the network, optionally loading weights, frozen and in eval mode.

    Args:
        state: a state dict holding the weights under `prefix` (e.g. a model's state dict
            with prefix ``"transform.vols2p_nn."``) or None for random weights.
        prefix: key prefix of the network inside `state`.
        device: target device (the weights are mapped to it; works for CPU-only machines).
        config: overrides of DEFAULT_CONFIG.
    """
    net = Vols2PNet(config)
    if state is not None:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        net.load_state_dict(sub, strict=True)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net
