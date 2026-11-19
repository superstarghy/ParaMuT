# SPDX-FileCopyrightText: Copyright 2022 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Alexandre Bittar <abittar@idiap.ch>
# SPDX-License-Identifier: BSD-3-Clause
"""The fixed-length SpArch SHD comparison with PSN-style hidden neurons.

The input binning, boxcar spike, spatial layers and stochastic readout adapt
Idiap's SpArch (f8756254), copyright 2022 Idiap Research Institute. See
licenses/SpArch-BSD-3-Clause.txt for attribution and redistribution terms.
"""
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .torch_models import TemporalBlockLIFNeuron


class SpArchSHD(Dataset):
    def __init__(self, root, split):
        self.path = root / f"shd_{split}.h5"
        self.time_bins = np.linspace(0, 1.4, num=100)
        with h5py.File(self.path, "r") as source:
            self.labels = source["labels"][:].astype(np.int64)
        self.source = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.source is None:
            self.source = h5py.File(self.path, "r")
        bins = np.digitize(self.source["spikes/times"][index], self.time_bins)
        units = self.source["spikes/units"][index].astype(np.int64)
        counts = np.bincount(bins * 700 + units, minlength=100 * 700)
        return torch.from_numpy(counts.reshape(100, 700).astype(np.float32)), self.labels[index]


class BoxcarSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * ((x > -.5) & (x <= .5))


class PSNLayer(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.W = nn.Linear(features, 128, bias=False)
        # The original comparison initializes the full adLIF backbone before
        # replacing its hidden cells. Preserve the initialization of later layers.
        for low, high in [(np.exp(-1/5), np.exp(-1/25)), (np.exp(-1/30), np.exp(-1/120)),
                          (-1., 1.), (0., 2.)]:
            nn.init.uniform_(torch.empty(128), low, high)
        self.norm = nn.BatchNorm1d(128, momentum=.05)
        self.drop = nn.Dropout(.1)
        with torch.random.fork_rng(devices=[]):
            self.temporal_mixer = TemporalBlockLIFNeuron(
                tau=10., decay_input=False, v_threshold=1., block_size=100,
                temporal_operator="learnable_full", block_reset="none")

    def forward(self, x):
        current = self.W(x)
        current = self.norm(current.flatten(0, 1)).reshape_as(current)
        matrix = self.temporal_mixer.effective_temporal_weights(100).to(current)
        membrane = torch.einsum("ij,jbh->ibh", matrix, current.transpose(0, 1))
        return self.drop(BoxcarSpike.apply(membrane - 1.).transpose(0, 1))


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(128, 20, bias=False)
        self.bounds = (np.exp(-1 / 5), np.exp(-1 / 25))
        self.alpha = nn.Parameter(torch.empty(20))
        nn.init.uniform_(self.alpha, *self.bounds)
        self.norm = nn.BatchNorm1d(20, momentum=.05)

    def forward(self, x):
        x = self.W(x)
        x = self.norm(x.flatten(0, 1)).reshape_as(x)
        alpha = self.alpha.clamp(*self.bounds)
        state = torch.rand(len(x), 20).to(x.device)
        scores = x.new_zeros(len(x), 20)
        for current in x.unbind(1):
            state = alpha * state + (1 - alpha) * current
            scores = scores + F.softmax(state, dim=1)
        return scores


class SHDPSN(nn.Module):
    def __init__(self):
        super().__init__()
        self.snn = nn.Sequential(PSNLayer(700), PSNLayer(128), Readout())

    def forward(self, x):
        return self.snn(x)
