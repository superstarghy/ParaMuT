"""SpikeCNN-style 1D backbone with BN and shared existing ParaMuT cells."""
import torch
from torch import nn
from torch.nn import functional as F
from .torch_models import TemporalBlockLIFNeuron

BASE_VARIANTS = ["lif", "k4", "k8", "k16", "k32", "psn"]
CAPS = {"c0p5": .5, "c0p75": .75, "c0p9": .9}
VARIANTS = BASE_VARIANTS + [f"k{size}_{suffix}" for size in (4, 8, 16, 32)
                           for suffix in (*CAPS, "db")]


class TriangleSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * (1-x.abs()).clamp_min(0)

class SequenceNeuron(nn.Module):
    def __init__(self, variant, timesteps):
        super().__init__()
        self.timesteps = timesteps
        if variant == 'lif':
            self.k = timesteps
            operator, reset = 'exact_recursive', 'soft'
        elif variant == 'psn':
            self.k = timesteps
            operator, reset = 'learnable_full', 'none'
        else:
            self.k = int(variant[1:])
            operator, reset = 'learnable_causal', 'soft_decay'
        self.cell = TemporalBlockLIFNeuron(tau=4., v_threshold=.5, v_reset=None,
            decay_input=False, detach_reset=False, surrogate_function=TriangleSpike.apply,
            block_size=self.k, temporal_operator=operator, block_reset=reset)

    def forward(self, x):
        if x.shape[-1] != self.timesteps: raise ValueError('Unexpected temporal length')
        # Reuse the core functional kernels, with local state. The public stateful
        # forward detaches between eval-mode calls; that would truncate attacks.
        seq = self.cell._charge_input(x.permute(2, 0, 1))
        state = torch.zeros_like(seq[0]); outputs = []
        kernel = getattr(self.cell, '_forward_'+self.cell.temporal_operator)
        for i in range(0, len(seq), self.k):
            spikes, state = kernel(seq[i:i+self.k], state)
            outputs.append(spikes)
        return torch.cat(outputs).permute(1, 2, 0)

class SpikeCNN(nn.Module):
    def __init__(self, variant='lif'):
        super().__init__()
        if variant not in BASE_VARIANTS: raise ValueError(variant)
        channels = [9, 32, 64, 64]
        self.convs = nn.ModuleList([nn.Conv1d(a, b, 8, bias=False) for a,b in zip(channels,channels[1:])])
        self.bns = nn.ModuleList([nn.BatchNorm1d(c) for c in channels[1:]])
        self.head = nn.Linear(64, 6)
        # Matrix initialization is deterministic, so all spatial weights share seed0 exactly.
        self.neurons = nn.ModuleList([SequenceNeuron(variant, t) for t in [128,64,32]])

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (128,9): raise ValueError('Expected B,128,9')
        x = x.transpose(1, 2)
        for conv, bn, neuron in zip(self.convs, self.bns, self.neurons):
            x = conv(F.pad(x, (7,0)))
            x = F.max_pool1d(neuron(bn(x)), 2)
        return self.head(x.mean(-1))

def build(variant, seed=0):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown HAR variant {variant!r}")
    base, _, suffix = variant.partition("_")
    # Keep the initialization of all convolution and readout weights paired
    # across variants without advancing the loader's random stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = SpikeCNN(base)
    if suffix in CAPS:
        for layer in model.neurons:
            layer.cell.temporal_mask_constraint = "row_col_l1"
            layer.cell.temporal_mask_cap = CAPS[suffix]
    elif suffix == "db":
        for layer in model.neurons:
            layer.cell = TemporalBlockLIFNeuron(
                tau=4., v_threshold=.5, v_reset=None, decay_input=False, detach_reset=False,
                surrogate_function=TriangleSpike.apply, block_size=layer.k,
                temporal_operator="learnable_causal", block_reset="soft_decay",
                temporal_mask_parameterization="diagonal_residual", temporal_mask_diagonal=.5,
                temporal_mask_diagonal_mode="fixed", temporal_mask_constraint="row_col_l1",
                temporal_mask_cap=.25)
    return model
