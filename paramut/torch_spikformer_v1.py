"""Spikformer v1 CIFAR models with matched full-sequence normalization.

The two architectures follow ZK-Zhou/spikformer, ``cifar10/model.py`` and
``cifar10dvs/model.py``.

All methods receive [T,B,C,H,W] and return per-step [T,B,classes] logits.
Convolutions, projections and BN always process all T*B samples together;
only the temporal neuron changes in the block and PSN-style comparisons.

Adapted from https://github.com/ZK-Zhou/spikformer (MIT License).
Copyright (c) 2022 Zhaokun Zhou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import torch
from torch import nn
from spikingjelly.clock_driven import neuron as sj_neuron, surrogate

from .torch_models import TemporalBlockLIFNeuron


class SequenceTemporalNeuron(nn.Module):
    """Chunk just the neuron, keeping upstream BN independent of block size.

    Block neurons retain the project's soft boundary reset and boundary-state
    gradient convention. PSN-style uses its full mixing matrix over exactly one
    sequence, with no reset or carried state. Neither is the official LIF cell.
    """

    def __init__(self, *, stateless=False, **options):
        super().__init__()
        self.cell = TemporalBlockLIFNeuron(**options)
        self.stateless = bool(stateless)

    def forward(self, x):
        if self.stateless:
            self.cell.reset_state()
        # TemporalBlockLIFNeuron recognizes sequences by rank >= 5.
        tokens = x.ndim == 4
        seq = x.unsqueeze(-1) if tokens else x
        size = self.cell.block_size
        out = torch.cat([self.cell(chunk) for chunk in seq.split(size, dim=0)], dim=0)
        if self.stateless:
            self.cell.reset_state()
        return out.squeeze(-1) if tokens else out


def _apply_sequence(module, x):
    shape = x.shape
    y = module(x.flatten(0, 1))
    return y.reshape(shape[0], shape[1], *y.shape[1:]).contiguous()


class SpikformerV1Stem(nn.Module):
    def __init__(self, in_channels, embed_dim, dvs, spike):
        super().__init__()
        self.dvs = dvs
        current = in_channels
        for index, channels in enumerate([embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]):
            suffix = "" if index == 0 else str(index)
            setattr(self, "proj_conv" + suffix, nn.Conv2d(current, channels, 3, padding=1, bias=False))
            setattr(self, "proj_bn" + suffix, nn.BatchNorm2d(channels))
            setattr(self, "proj_lif" + suffix, spike())
            if dvs or index >= 2:
                setattr(self, "maxpool" + suffix, nn.MaxPool2d(3, stride=2, padding=1))
            current = channels
        self.rpe_conv = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dim)
        self.rpe_lif = spike()

    def forward(self, x):
        for index in range(4):
            suffix = "" if index == 0 else str(index)
            x = _apply_sequence(getattr(self, "proj_conv" + suffix), x)
            x = _apply_sequence(getattr(self, "proj_bn" + suffix), x)
            x = getattr(self, "proj_lif" + suffix)(x)
            if self.dvs or index >= 2:
                x = _apply_sequence(getattr(self, "maxpool" + suffix), x)
        residual = x
        x = self.rpe_lif(_apply_sequence(self.rpe_bn, _apply_sequence(self.rpe_conv, x)))
        x = (x + residual).flatten(-2)
        return x if self.dvs else x.transpose(-1, -2).contiguous()


class SpikformerV1Attention(nn.Module):
    def __init__(self, dim, heads, dvs, spike):
        super().__init__()
        self.dim, self.num_heads, self.dvs = dim, heads, dvs
        self.scale = 0.25 if dvs else 0.125
        for name in ["q", "k", "v", "proj"]:
            projection = (nn.Conv1d(dim, dim, 1, bias=name == "proj")
                          if dvs else nn.Linear(dim, dim))
            setattr(self, name + ("_conv" if dvs else "_linear"), projection)
            setattr(self, name + "_bn", nn.BatchNorm1d(dim))
            setattr(self, name + "_lif", spike())
        self.attn_lif = spike(0.5)

    def _project(self, x, name):
        projection = getattr(self, name + ("_conv" if self.dvs else "_linear"))
        x = _apply_sequence(projection, x)
        if not self.dvs:
            x = x.transpose(-1, -2)
        x = _apply_sequence(getattr(self, name + "_bn"), x)
        if not self.dvs:
            x = x.transpose(-1, -2).contiguous()
        return getattr(self, name + "_lif")(x)

    def _heads(self, x):
        if self.dvs:
            x = x.transpose(-1, -2)
        t, b, n, c = x.shape
        return x.reshape(t, b, n, self.num_heads, c // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

    def forward(self, x):
        q, k, v = (self._heads(self._project(x, name)) for name in ["q", "k", "v"])
        attn = q @ k.transpose(-2, -1)
        # Preserve the original operation ordering for each dataset.
        x = (attn @ v) * self.scale if self.dvs else (attn * self.scale) @ v
        t, b, _, n, _ = x.shape
        if self.dvs:
            x = x.transpose(3, 4).reshape(t, b, self.dim, n).contiguous()
        else:
            x = x.transpose(2, 3).reshape(t, b, n, self.dim).contiguous()
        return self._project(self.attn_lif(x), "proj")


class SpikformerV1MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dvs, spike):
        super().__init__()
        self.dvs = dvs
        for index, (in_dim, out_dim) in enumerate([(dim, hidden_dim), (hidden_dim, dim)], 1):
            name = f"fc{index}"
            projection = nn.Conv1d(in_dim, out_dim, 1) if dvs else nn.Linear(in_dim, out_dim)
            setattr(self, name + ("_conv" if dvs else "_linear"), projection)
            setattr(self, name + "_bn", nn.BatchNorm1d(out_dim))
            setattr(self, name + "_lif", spike())

    def forward(self, x):
        for index in [1, 2]:
            name = f"fc{index}"
            x = _apply_sequence(getattr(self, name + ("_conv" if self.dvs else "_linear")), x)
            if not self.dvs:
                x = x.transpose(-1, -2)
            x = _apply_sequence(getattr(self, name + "_bn"), x)
            if not self.dvs:
                x = x.transpose(-1, -2).contiguous()
            x = getattr(self, name + "_lif")(x)
        return x


class SpikformerV1Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dvs, spike):
        super().__init__()
        # Upstream registers these LayerNorms but does not call them. Keep them
        # for strict compatibility with official LIF state dictionaries.
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SpikformerV1Attention(dim, heads, dvs, spike)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = SpikformerV1MLP(dim, int(dim * mlp_ratio), dvs, spike)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class SpikformerV1(nn.Module):
    """Official v1 architecture with LIF, ParaMuT, or PSN-style neurons.

    ``block_options`` uses TemporalBlockLIFNeuron keyword names, for example
    temporal_mask_cap=.1 and temporal_mask_diagonal=.75. LIF uses the exact
    upstream tau=2, decay_input=True, hard reset=0, detach_reset=True, and
    Sigmoid(alpha=4). Block/PSN use the same charge scaling and surrogate but
    retain the repository's own temporal operator/reset definitions. A capped
    D+B operator starts with B's maximum constrained norm at most half its cap:
    tau=2 otherwise saturates the only off-diagonal coefficient at K=2, making
    its projection gradient zero before training can begin.
    """

    stateful_spikingjelly = True

    def __init__(self, dataset="cifar10", neuron="lif", timesteps=None,
                 block_size=None, embed_dim=None, depth=None, num_heads=None,
                 mlp_ratio=4., image_size=None, num_classes=10,
                 lif_backend="torch", block_options=None):
        super().__init__()
        if dataset not in {"cifar10", "cifar10dvs"}:
            raise ValueError("dataset must be cifar10 or cifar10dvs")
        if neuron not in {"lif", "block", "psn"}:
            raise ValueError("neuron must be lif, block, or psn")
        dvs = dataset == "cifar10dvs"
        self.dataset, self.neuron = dataset, neuron
        self.timesteps = int((16 if dvs else 4) if timesteps is None else timesteps)
        self.block_size = int((8 if dvs else 2) if block_size is None else block_size)
        self.embed_dim = int((256 if dvs else 384) if embed_dim is None else embed_dim)
        self.depth = int((2 if dvs else 4) if depth is None else depth)
        self.num_heads = int((16 if dvs else 12) if num_heads is None else num_heads)
        self.image_size = int((128 if dvs else 32) if image_size is None else image_size)
        self.num_classes = int(num_classes)
        self.patch_size = 16 if dvs else 4
        self.in_channels = 2 if dvs else 3
        if self.timesteps < 1 or self.block_size < 1 or self.depth < 1 or mlp_ratio <= 0:
            raise ValueError("timesteps, block_size, depth and mlp_ratio must be positive")
        if self.embed_dim < 8 or self.embed_dim % 8 or self.num_heads < 1 or self.embed_dim % self.num_heads:
            raise ValueError("embed_dim must be divisible by 8 and num_heads")
        if self.image_size < self.patch_size or self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by the dataset's SPS downsampling factor")
        if lif_backend not in {"torch", "cupy"}:
            raise ValueError("lif_backend must be torch or cupy")
        if neuron != "block" and block_options:
            raise ValueError("block_options only applies to neuron=block")

        def spike(threshold=1.):
            common = dict(tau=2., decay_input=True, v_threshold=threshold,
                          surrogate_function=surrogate.Sigmoid(alpha=4.), detach_reset=True)
            if neuron == "lif":
                return sj_neuron.MultiStepLIFNode(v_reset=0., backend=lif_backend, **common)
            if neuron == "psn":
                return SequenceTemporalNeuron(
                    stateless=True, v_reset=None, block_size=self.timesteps,
                    temporal_operator="learnable_full", block_reset="none", **common)
            options = dict(temporal_operator="learnable_causal", block_reset="soft",
                           temporal_mask_constraint="row_col_l1", temporal_mask_cap=.25,
                           temporal_mask_parameterization="diagonal_residual",
                           temporal_mask_diagonal=1.)
            options.update(block_options or {})
            node = SequenceTemporalNeuron(v_reset=None, block_size=self.block_size, **common, **options)
            cell = node.cell
            if (cell.temporal_mask_parameterization == "diagonal_residual"
                    and cell.temporal_mask_constraint != "none"):
                with torch.no_grad():
                    residual = torch.tril(cell.temporal_logits, diagonal=-1)
                    norm = residual.abs().sum(1).max()
                    if cell.temporal_mask_constraint == "row_col_l1":
                        norm = torch.maximum(norm, residual.abs().sum(0).max())
                    factor = min(1., .5 * cell.temporal_mask_cap / max(float(norm), 1e-12))
                    lower = torch.tril(torch.ones_like(cell.temporal_logits, dtype=torch.bool), diagonal=-1)
                    cell.temporal_logits[lower] *= factor
            return node

        self.patch_embed = SpikformerV1Stem(self.in_channels, self.embed_dim, dvs, spike)
        if dvs:
            # Also unused upstream; retained for strict LIF checkpoint loading.
            self.pos_embed = nn.Parameter(torch.zeros(1, (self.image_size // 16) ** 2, self.embed_dim))
        self.block = nn.ModuleList([
            SpikformerV1Block(self.embed_dim, self.num_heads, mlp_ratio, dvs, spike)
            for _ in range(self.depth)
        ])
        if dvs:
            self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        if dvs:
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # The original initializes only Linear/LayerNorm explicitly. Conv1d,
        # Conv2d and BN retain their PyTorch defaults, including DVS projections.
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def reset_state(self):
        for module in self.modules():
            if isinstance(module, sj_neuron.MultiStepLIFNode):
                module.reset()
            elif isinstance(module, TemporalBlockLIFNeuron):
                module.reset_state()

    def forward_features(self, x):
        x = self.patch_embed(x)
        for block in self.block:
            x = block(x)
        return x.mean(-1 if self.dataset == "cifar10dvs" else -2)

    def forward(self, x):
        if x.ndim != 5 or x.shape[0] != self.timesteps or x.shape[2] != self.in_channels:
            raise ValueError(f"Expected [T={self.timesteps},B,C={self.in_channels},H,W], got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(f"Expected {self.image_size}x{self.image_size} images, got {tuple(x.shape[-2:])}")
        return self.head(self.forward_features(x))
