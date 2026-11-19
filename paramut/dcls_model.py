"""DCLS SHD backbone and ParaMuT adapter with explicit temporal semantics.

The reference is SNN-delays d169b4e, 140 -> 256 -> 256 -> 20, 10 ms frames.
Delay currents are evaluated for the entire padded sequence, then neuron blocks
are evaluated with local state. Every synapse appends the upstream 12-step tail,
including the no-delay controls. Hidden LIF uses hard reset and detached reset
spikes; ParaMuT resets only at block boundaries using the last spike. K>1 is a
changed neuron, not an exact parallel implementation of the recurrent LIF.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from spikingjelly.activation_based import surrogate
from DCLS.construct.modules import Dcls1d
from .torch_models import TemporalBlockLIFNeuron

TAU = (10.05 + 1e-9) / 10.
KERNEL_SIZE = 25
TAIL = 12
VARIANTS = ('lif', 'dcls_lif', 'paramut', 'dcls_paramut')


class DelaySynapse(nn.Module):
    """B,C,T -> B,Cout,T+12, with causal delay range 0..24 time steps."""
    def __init__(self, in_features: int, out_features: int, delayed: bool):
        super().__init__()
        self.delayed = bool(delayed)
        self.kernel_size, self.tail = KERNEL_SIZE, TAIL
        if self.delayed:
            self.operator = Dcls1d(in_features, out_features, kernel_count=1,
                                  dilated_kernel_size=KERNEL_SIZE, bias=False, version='gauss')
            self.operator.SIG.requires_grad_(False)
        else:
            self.operator = nn.Conv1d(in_features, out_features, 1, bias=False)
        self._discrete_kernel = None
        self._cache_identity = None

    @property
    def weight(self):
        return self.operator.weight

    @property
    def P(self):
        return self.operator.P if self.delayed else None

    @property
    def SIG(self):
        return self.operator.SIG if self.delayed else None

    def invalidate_cache(self):
        self._discrete_kernel = self._cache_identity = None

    def _identity(self):
        return (self.weight._version, self.P._version, self.weight.device, self.weight.dtype)

    def kernel(self, discrete: bool, *, cached: bool = False):
        if not self.delayed:
            return self.weight
        if not discrete:
            return self.operator.DCK(self.weight, self.P, self.SIG)
        if cached:
            self.prepare_inference()
            return self._discrete_kernel
        # Exactly the official eval conversion: round/clamp positions, SIG=0,
        # max (triangular) interpolation including its normalization epsilon.
        p = self.P.round().clamp(-12., 12.)
        # DCLS 0.1.1 initializes index tensors lazily in forward().
        if hasattr(self.operator.DCK, '__init_tmp_variables__'):
            self.operator.DCK.__init_tmp_variables__(self.weight.device)
        return self.operator.DCK.forward_vmax(self.weight, p, torch.zeros_like(self.SIG))

    def prepare_inference(self):
        if self.delayed and self._cache_identity != self._identity():
            with torch.no_grad():
                self._discrete_kernel = self.kernel(True, cached=False).detach()
            self._cache_identity = self._identity()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.invalidate_cache()
        return self

    def _apply(self, fn, recurse=True):
        self.invalidate_cache()
        if self.delayed and 'IDX' not in self.operator.DCK._buffers:
            # Old DCLS keeps temporary index tensors outside registered buffers.
            self.operator.DCK.IDX = self.operator.DCK.lim = None
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.invalidate_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x):
        if self.delayed:
            kernel = self.kernel(not self.training, cached=not self.training)
            return F.conv1d(F.pad(x, (24, self.tail)), kernel)
        return F.pad(self.operator(x), (0, self.tail))


class LockedDropout(nn.Module):
    """A fresh B,H mask each forward, shared across all T, as in SpikingJelly."""
    def __init__(self, p=.4):
        super().__init__()
        self.p = p

    def forward(self, seq):
        if not self.training or self.p == 0:
            return seq
        mask = F.dropout(torch.ones_like(seq[0]), p=self.p, training=True)
        return seq * mask


class TemporalNeuron(nn.Module):
    """T,B,H -> T,B,H spikes; no persistent state and full state gradients."""
    def __init__(self, method='lif', K=4, constraint='none', cap=.75, diagonal=.5,
                 threshold=1., tau=TAU, surrogate_alpha=5., return_membrane=False):
        super().__init__()
        if method not in ('lif', 'paramut') or constraint not in ('none', 'full_cap', 'db'):
            raise ValueError((method, constraint))
        if K < 1:
            raise ValueError('K must be positive')
        self.method, self.K = method, int(K)
        self.threshold, self.tau = float(threshold), float(tau)
        self.surrogate_alpha = float(surrogate_alpha)
        self.return_membrane = bool(return_membrane)
        self.spike = surrogate.ATan(alpha=self.surrogate_alpha)
        self.constraint, self.cap, self.diagonal = constraint, float(cap), float(diagonal)
        if method == 'paramut':
            self.mixer = TemporalBlockLIFNeuron(
                tau=self.tau, block_size=self.K, decay_input=False,
                temporal_operator='learnable_causal', block_reset='none',
                temporal_mask_constraint='none' if constraint == 'none' else 'row_col_l1',
                temporal_mask_cap=self.cap,
                temporal_mask_parameterization='diagonal_residual' if constraint == 'db' else 'direct',
                temporal_mask_diagonal=self.diagonal, temporal_mask_diagonal_mode='fixed')
        else:
            self.mixer = None


    def trajectory(self, seq):
        if seq.ndim != 3 or len(seq) == 0:
            raise ValueError('Expected nonempty T,B,H currents')
        state = torch.zeros_like(seq[0])
        spikes, potentials = [], []
        if self.method == 'lif':
            for current in seq:
                # Upstream uses alpha*v in training, but subtraction in its
                # JIT eval path. Preserve both floating-point orders.
                carry = state * (1. - 1. / self.tau) if self.training else state - state / self.tau
                membrane = carry + current
                spike = self.spike(membrane - self.threshold)
                state = (1. - spike.detach()) * membrane
                spikes.append(spike)
                potentials.append(state)
            return torch.stack(spikes), torch.stack(potentials)
        matrix = self.mixer.effective_temporal_weights()
        alpha = 1. - 1. / self.tau
        for start in range(0, len(seq), self.K):
            current = seq[start:start+self.K]
            k = len(current)
            if self.K == 1:
                carry = state * (1. - 1. / self.tau) if self.training else state - state / self.tau
                membrane = carry[None] + matrix[0, 0] * current
            else:
                exponents = torch.arange(1, k+1, device=seq.device, dtype=seq.dtype)
                carry = torch.pow(seq.new_tensor(alpha), exponents)[:, None, None] * state[None]
                membrane = torch.einsum('ij,jbh->ibh', matrix[:k, :k], current) + carry
            spike = self.spike(membrane - self.threshold)
            # Only the final spike gates carried state. Do not detach membrane
            # state: detach_reset applies to spike, not to the whole block state.
            state = (1. - spike[-1].detach()) * membrane[-1]
            spikes.append(spike)
            potentials.append(torch.cat((membrane[:-1], state[None]), dim=0))
        return torch.cat(spikes), torch.cat(potentials)

    def forward(self, seq):
        return self.trajectory(seq)[1 if self.return_membrane else 0]


class DCLSBackbone(nn.Module):
    def __init__(self, variant='dcls_lif', *, width=256, K=4, constraint='none',
                 cap=.75, diagonal=.5, dropout=.4, surrogate_alpha=5.):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        if 'paramut' not in variant and constraint != 'none':
            raise ValueError('Matrix constraints require ParaMuT')
        self.variant, self.width = variant, int(width)
        self.K, self.constraint, self.cap, self.diagonal = K, constraint, cap, diagonal
        self.dropout_p, self.surrogate_alpha = dropout, surrogate_alpha
        delayed = variant.startswith('dcls_')
        method = 'paramut' if 'paramut' in variant else 'lif'
        self.synapses = nn.ModuleList([DelaySynapse(a, b, delayed)
                                      for a, b in ((140, width), (width, width), (width, 20))])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(2)])
        self.neurons = nn.ModuleList([TemporalNeuron(method, K, constraint, cap, diagonal,
                                                    surrogate_alpha=surrogate_alpha) for _ in range(2)])
        self.dropouts = nn.ModuleList([LockedDropout(dropout) for _ in range(2)])
        self.readout = TemporalNeuron('lif', threshold=1e9, surrogate_alpha=surrogate_alpha, return_membrane=True)

    def initialize(self, seed=0):
        torch.manual_seed(seed)
        for synapse in self.synapses:
            nn.init.kaiming_uniform_(synapse.weight, nonlinearity='relu')
        for synapse in self.synapses:
            if synapse.delayed:
                nn.init.uniform_(synapse.P, a=-KERNEL_SIZE//2, b=KERNEL_SIZE//2)
                synapse.operator.clamp_parameters()
                nn.init.constant_(synapse.SIG, 12.)
        return self

    def reset(self):
        """State is local to every forward; no spike/delay history is retained."""
        return self

    def sequence(self, x):
        if x.ndim != 3 or x.shape[-1] != 140 or x.shape[1] == 0:
            raise ValueError('Expected B,T,140 inputs')
        self.reset()
        seq = x.transpose(0, 1)
        for synapse, bn, neuron, dropout in zip(self.synapses[:2], self.bns, self.neurons, self.dropouts):
            seq = synapse(seq.permute(1, 2, 0)).permute(2, 0, 1)
            shape = seq.shape
            # Match SpikingJelly BatchNorm1d(step_mode='m') flatten order.
            seq = bn(seq.unsqueeze(-1).flatten(0, 1)).reshape(*shape, 1).squeeze(-1)
            seq = dropout(neuron(seq))
        current = self.synapses[-1](seq.permute(1, 2, 0)).permute(2, 0, 1)
        return self.readout(current)

    def forward(self, x):
        return self.sequence(x).softmax(dim=2).sum(dim=0)

    def prepare_inference(self):
        if self.training:
            raise RuntimeError('Call model.eval() before preparing inference')
        for synapse in self.synapses:
            synapse.prepare_inference()
        return self


    def optimizer_groups(self):
        return {'weights': [s.weight for s in self.synapses],
                'bn': [p for bn in self.bns for p in bn.parameters()],
                'positions': [s.P for s in self.synapses if s.delayed],
                'temporal': [p for n in self.neurons if n.mixer is not None for p in n.mixer.parameters()]}


    @torch.no_grad()
    def clamp_delays_(self):
        for synapse in self.synapses:
            if synapse.delayed:
                synapse.operator.clamp_parameters()
                synapse.invalidate_cache()

    @torch.no_grad()
    def step_sigma(self, epoch, epochs=150):
        final_epoch = epochs // 4
        for synapse in self.synapses:
            if synapse.delayed and epoch < final_epoch and float(synapse.SIG.flatten()[0]) > .23:
                synapse.SIG.mul_((.23 / 12.) ** (1. / final_epoch))
        return self


def build(variant='dcls_lif', seed=0, width=256, K=4, constraint='none', cap=.75,
          diagonal=.5, dropout=.4, surrogate_alpha=5.):
    with torch.random.fork_rng(devices=[]):
        return DCLSBackbone(variant, width=width, K=K, constraint=constraint, cap=cap,
                            diagonal=diagonal, dropout=dropout,
                            surrogate_alpha=surrogate_alpha).initialize(seed)
