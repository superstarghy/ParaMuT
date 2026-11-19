"""The restricted augmentation policies used by the original Spikformer.

Adapted from ZK-Zhou/spikformer at commit
621268981827af47aa666c1b5f774de4281d90ad:
https://github.com/ZK-Zhou/spikformer/blob/621268981827af47aa666c1b5f774de4281d90ad/cifar10/aa_snn.py
https://github.com/ZK-Zhou/spikformer/blob/621268981827af47aa666c1b5f774de4281d90ad/cifar10dvs/autoaugment.py

Only the five CIFAR operations and six DVS operations actually used by the
released recipes are retained. CIFAR operations originally by Ross Wightman,
Copyright 2019. DVS operations use torchvision's tensor transformations.

MIT License
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

import math
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


class SpikformerCIFARRandAugment:
    """Released ``rand-m9-n1-mstd0.4-inc1`` restricted CIFAR policy.

The upstream policy chooses one of five operations uniformly, then applies
that operation with probability 0.5. This differs substantially from the
standard timm/torchvision policy, which also changes colors and contrast.
"""

    operations = ("Equalize", "Rotate", "ShearX", "TranslateXRel", "TranslateYRel")

    def __init__(self, mean=(0.4914, 0.4822, 0.4465)):
        self.fill = tuple(min(255, round(255 * x)) for x in mean)

    def __call__(self, img):
        operation = np.random.choice(self.operations, size=1)[0]
        if random.random() > 0.5:
            return img
        magnitude = max(0.0, min(random.gauss(9.0, 0.4), 10.0))
        if operation == "Equalize":
            return ImageOps.equalize(img)
        factor = magnitude / 10.0
        if random.random() > 0.5:
            factor = -factor
        kwargs = dict(resample=Image.Resampling.BICUBIC, fillcolor=self.fill)
        if operation == "Rotate":
            return img.rotate(factor * 30.0, **kwargs)
        if operation == "ShearX":
            matrix = (1, factor * 0.3, 0, 0, 1, 0)
        elif operation == "TranslateXRel":
            matrix = (1, 0, factor * 0.45 * img.size[0], 0, 1, 0)
        else:
            matrix = (1, 0, 0, 0, 1, factor * 0.45 * img.size[1])
        return img.transform(img.size, Image.Transform.AFFINE, matrix, **kwargs)


class SpikformerSNNAugmentWide:
    """The released six-operation DVS policy, shared across all frames.

The input is a float event-count tensor [T, 2, H, W]. Geometry and cutout
operate on every timestep/polarity together; counts are never normalized.
"""

    operations = ("Identity", "ShearX", "TranslateX", "TranslateY", "Rotate", "Cutout")

    def __init__(self, num_magnitude_bins=31):
        self.num_magnitude_bins = int(num_magnitude_bins)
        self.cutout = transforms.RandomErasing(p=1, scale=(0.001, 0.11), ratio=(1, 1))

    def __call__(self, x):
        operation = self.operations[int(torch.randint(len(self.operations), (1,)).item())]
        if operation == "Identity":
            return x
        bounds = {"ShearX": (-0.3, 0.3), "TranslateX": (-5.0, 5.0),
                  "TranslateY": (-5.0, 5.0), "Rotate": (-30.0, 30.0),
                  "Cutout": (1.0, 30.0)}
        magnitudes = torch.linspace(*bounds[operation], self.num_magnitude_bins)
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,))].item())
        if torch.randint(2, (1,)):
            magnitude *= -1.0
        if operation == "Cutout":
            return self.cutout(x)
        interpolation = transforms.InterpolationMode.NEAREST
        if operation == "Rotate":
            return TF.rotate(x, magnitude, interpolation=interpolation)
        translate = [0, 0]
        shear = [0.0, 0.0]
        if operation == "ShearX":
            shear[0] = math.degrees(magnitude)
        elif operation == "TranslateX":
            translate[0] = int(magnitude)
        else:
            translate[1] = int(magnitude)
        return TF.affine(x, angle=0.0, translate=translate, scale=1.0,
                         shear=shear, interpolation=interpolation)
