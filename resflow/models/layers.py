"""Graph network building blocks of the ResFlow models.

Only what the final architecture uses: MLPs, the low-rank edge-conditioned convolution,
the convolution block (edge MLP -> conv -> residual -> LayerNorm), the processor stack
(with the block curriculum switch and the optional dense aggregation of the rate model),
the encoder and the encode-process-decode network.

Module and attribute names follow the networks the shipped weights were trained with, so
their state-dict keys load unchanged.

Copyright (c) Tiago Amorim. MIT License.
"""
from typing import Optional, Sequence

import torch
from torch import nn

ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


class Activation(nn.Module):
    """Activation by name (``gelu`` in the paper models). Its own module, so the
    state-dict keys of the MLPs are those of the shipped weights."""

    def __init__(self, name: Optional[str]):
        super().__init__()
        if name is None:
            self.activation = nn.Identity()
        elif name.lower() in ACTIVATIONS:
            self.activation = ACTIVATIONS[name.lower()]()
        else:
            raise ValueError(f"Unknown activation '{name}'. Use one of {sorted(ACTIVATIONS)}.")

    def forward(self, x):
        return self.activation(x)


def _init_linear(linear: nn.Linear, init_type: str) -> None:
    """Weight initialisation: 'xavier_uniform' (paper models, zero bias) or 'default'
    (PyTorch's own)."""
    if init_type in (None, "default"):
        return
    if init_type != "xavier_uniform":
        raise ValueError(f"Unknown init_type '{init_type}'. Use 'xavier_uniform' or 'default'.")
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MLP(nn.Module):
    """Multi-layer perceptron: ``dims`` = input size followed by each layer's output size.

    Hidden layers are ``Sequential(Linear, activation)``; the last one is a plain Linear,
    optionally followed by the activation (``final_activation``).
    """

    def __init__(self, dims: Sequence[int], activation: Optional[str] = "gelu",
                 final_activation: bool = False, init_type: str = "xavier_uniform"):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("An MLP needs at least two dimensions.")
        self.activation = Activation(activation)
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.layers.append(nn.Sequential(nn.Linear(dims[i], dims[i + 1]), self.activation))
        self.layers.append(nn.Linear(dims[-2], dims[-1]))
        self.final_activation = self.activation if final_activation else None
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module, init_type)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.final_activation is not None:
            x = self.final_activation(x)
        return x


class LearnableAffine(nn.Module):
    """Per-feature ``a * x + b`` (input layer of the rate model's encoder)."""

    def __init__(self, num_features: int, init_scale: float = 1.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(num_features) * init_scale)
        self.b = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return self.a * x + self.b


def concat_edge_features(x: torch.Tensor, edge_attr: torch.Tensor,
                         edge_index: torch.Tensor) -> torch.Tensor:
    """[source node, target node, edge] features of every edge (x is [..., N, F])."""
    return torch.cat((x.index_select(-2, edge_index[0]), x.index_select(-2, edge_index[1]),
                      edge_attr), dim=-1)


class LowRankConv(nn.Module):
    """Edge-conditioned convolution with a rank-r kernel.

        W(e) = U(e) V(e)^T,  U, V in R^{d x r} (from an MLP of the edge features)
        x_i' = R x_i + b + sum_j U(e_ij) (V(e_ij)^T x_j)

    Messages flow from edge_index[0] to edge_index[1] and are summed. Works on [N, d] and
    on [T, N, d] inputs. Raises ValueError on non-finite inputs or a NaN output (the
    trainers skip such steps; the rollout stops).
    """

    def __init__(self, dims: Sequence[int], hidden_dims: Sequence[int] = (),
                 activation: str = "gelu", init_type: str = "xavier_uniform", rank: int = 8):
        super().__init__()
        d_node, d_edge = dims
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}.")
        self.d_node = d_node
        self.rank = rank
        self.check_finite = True
        self.kernel = MLP([d_edge] + list(hidden_dims) + [2 * d_node * rank],
                          activation=activation, init_type=init_type)
        self.lin_root = nn.Linear(d_node, d_node)

    def forward(self, x, edge_index, edge_attr):
        if self.check_finite and not (torch.isfinite(x).all() and torch.isfinite(edge_attr).all()):
            raise ValueError("NaN/Inf in the input of LowRankConv")
        uv = self.kernel(edge_attr).unflatten(-1, (2, self.d_node, self.rank))
        u, v = uv[..., 0, :, :], uv[..., 1, :, :]           # [..., E, d, r]
        x_src = x.index_select(-2, edge_index[0])           # [..., E, d]
        coef = torch.einsum('...dr,...d->...r', v, x_src)   # V^T x_j
        msg = torch.einsum('...dr,...r->...d', u, coef)     # U (V^T x_j)
        dst = edge_index[1].view(*([1] * (msg.ndim - 2)), -1, 1).expand_as(msg)
        agg = torch.zeros_like(x).scatter_add_(msg.ndim - 2, dst, msg)
        x_out = self.lin_root(x) + agg
        if self.check_finite and torch.isnan(x_out).any():
            raise ValueError("NaN after LowRankConv")
        return x_out


class ConvBlock(nn.Module):
    """One processor block: edge MLP on [x_src, x_dst, e] -> low-rank conv -> residual
    (``inner_skip``) -> LayerNorm (``use_bn``, applied after the residual: 'post')."""

    def __init__(self, hidden_dims, kernel_dims, edge_mlp_dims, activation="gelu",
                 use_bn=True, inner_skip=True, init_type="xavier_uniform", conv_rank=8):
        super().__init__()
        self.inner_skip = inner_skip
        self.use_bn = use_bn
        self.conv = LowRankConv(hidden_dims, kernel_dims, activation, init_type, rank=conv_rank)
        if use_bn:
            self.ln = nn.LayerNorm(hidden_dims[0])
        self.use_edge_mlp = edge_mlp_dims is not None
        if self.use_edge_mlp:
            self.edge_mlp = MLP([2 * hidden_dims[0] + hidden_dims[1]] + list(edge_mlp_dims)
                                + [hidden_dims[1]], activation=activation,
                                final_activation=True, init_type=init_type)

    def forward(self, x, edge_index, edge_attr):
        if self.use_edge_mlp:
            edge_attr = self.edge_mlp(concat_edge_features(x, edge_attr, edge_index))
        update = self.conv(x, edge_index, edge_attr)
        x = x + update if self.inner_skip else update
        if self.use_bn:
            x = self.ln(x)
        return x, edge_attr


class Processor(nn.Module):
    """Stack of `n_blocks` ConvBlocks.

    `active_blocks` (set by the trainer's block curriculum) runs only the first blocks;
    None runs all. With `dense_layers` (rate model) the outputs of every block (and the
    input) are concatenated and mixed by a linear layer; blocks not run contribute zeros.
    """

    def __init__(self, n_blocks, hidden_dims, kernel_dims, edge_mlp_dims, activation="gelu",
                 use_bn=True, inner_skip=True, init_type="xavier_uniform", conv_rank=8,
                 dense_layers=False):
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be at least 1, got {n_blocks}.")
        self.n_blocks = n_blocks
        self.active_blocks: Optional[int] = None
        self.dense_layers = dense_layers
        self.conv_blocks = nn.ModuleList([
            ConvBlock(hidden_dims, kernel_dims, edge_mlp_dims, activation, use_bn, inner_skip,
                      init_type, conv_rank) for _ in range(n_blocks)])
        if dense_layers:
            self.dense_nodes_mlp = MLP([(n_blocks + 1) * hidden_dims[0], hidden_dims[0]],
                                       activation=activation, init_type=init_type)
            self.dense_edges_mlp = MLP([(n_blocks + 1) * hidden_dims[1], hidden_dims[1]],
                                       activation=activation, init_type=init_type)

    def forward(self, x, edge_index, edge_attr):
        n_run = self.n_blocks if self.active_blocks is None else min(self.active_blocks,
                                                                     self.n_blocks)
        x_all, e_all = [x], [edge_attr]
        for block in self.conv_blocks[:n_run]:
            x, edge_attr = block(x, edge_index, edge_attr)
            if self.dense_layers:
                x_all.append(x)
                e_all.append(edge_attr)
        if self.dense_layers:
            for _ in range(self.n_blocks + 1 - len(x_all)):
                x_all.append(torch.zeros_like(x))
                e_all.append(torch.zeros_like(edge_attr))
            x = self.dense_nodes_mlp(torch.cat(x_all, dim=-1))
            edge_attr = self.dense_edges_mlp(torch.cat(e_all, dim=-1))
        return x, edge_attr


class Encoder(nn.Module):
    """Node MLP and edge MLP (on [x_src, x_dst, e]); optional learnable affine input
    layers (one per `init_scale` entry; the rate model has one, the state model none)."""

    def __init__(self, input_dims, hidden_dims, enc_nodes_dims, enc_edges_dims,
                 init_scale=(), activation="gelu", init_type="xavier_uniform"):
        super().__init__()
        if isinstance(init_scale, (int, float)):
            init_scale = [init_scale]
        self.norm_nodes = nn.ModuleList([LearnableAffine(input_dims[0], s) for s in init_scale])
        self.norm_edges = nn.ModuleList([LearnableAffine(input_dims[1], s) for s in init_scale])
        self.enc_nodes = MLP([input_dims[0]] + list(enc_nodes_dims) + [hidden_dims[0]],
                             activation=activation, init_type=init_type)
        self.enc_edges = MLP([2 * input_dims[0] + input_dims[1]] + list(enc_edges_dims)
                             + [hidden_dims[1]], activation=activation, init_type=init_type)

    def forward(self, x, edge_index, edge_attr):
        for norm in self.norm_nodes:
            x = norm(x)
        for norm in self.norm_edges:
            edge_attr = norm(edge_attr)
        edge_attr = self.enc_edges(concat_edge_features(x, edge_attr, edge_index))
        return self.enc_nodes(x), edge_attr


class GraphNet(nn.Module):
    """Encode-process-decode graph network (the paper's GNN).

    Args:
        input_dims: (node features, edge features).
        output_dim: node outputs.
        hidden_dims: latent (node, edge) sizes, e.g. (128, 64).
        enc_nodes_dims, enc_edges_dims, dec_nodes_dims, kernel_dims, edge_mlp_dims: hidden
            layer sizes of the encoder MLPs, decoder MLP, conv kernel MLP and edge MLPs.
        n_blocks, conv_rank: processor depth and kernel rank.
        decoder_concat_input: the decoder sees [processor output, network input].
        use_bn, add_inner_skip: LayerNorm after each block, residual around each conv.
        dense_layers: dense aggregation of all block outputs (rate model).
        init_scale: scales of the learnable affine input layers (empty: none).
    """

    def __init__(self, input_dims, output_dim, hidden_dims, enc_nodes_dims, enc_edges_dims,
                 dec_nodes_dims, kernel_dims, edge_mlp_dims, n_blocks, conv_rank=8,
                 activation="gelu", decoder_concat_input=True, use_bn=True,
                 add_inner_skip=True, dense_layers=False, init_scale=(),
                 init_type="xavier_uniform"):
        super().__init__()
        self.decoder_concat_input = decoder_concat_input
        self.encoder = Encoder(input_dims, hidden_dims, enc_nodes_dims, enc_edges_dims,
                               init_scale, activation, init_type)
        self.processor = Processor(n_blocks, hidden_dims, kernel_dims, edge_mlp_dims,
                                   activation, use_bn, add_inner_skip, init_type, conv_rank,
                                   dense_layers)
        dec_in = hidden_dims[0] + (input_dims[0] if decoder_concat_input else 0)
        self.decoder = MLP([dec_in] + list(dec_nodes_dims) + [output_dim],
                           activation=activation, init_type=init_type)

    def forward(self, x, edge_index, edge_attr):
        """x [..., N, F_node], edge_attr [..., E, F_edge] -> node outputs [..., N, output_dim]."""
        x_enc, e_enc = self.encoder(x, edge_index, edge_attr)
        h, _ = self.processor(x_enc, edge_index, e_enc)
        return self.decoder(torch.cat([h, x], dim=-1) if self.decoder_concat_input else h)
