"""VGG11-BN used in the four image and event-frame comparisons."""
from __future__ import annotations

import torch
from torch import nn
from spikingjelly.clock_driven import functional, layer

from .torch_models import SLTTBPTTNeuron, TemporalBlockLIFNeuron, make_sj_surrogate


class VisionVGG(nn.Module):
    def __init__(self, in_channels, num_classes, dropout, timesteps, method, block_options):
        super().__init__()
        self.timesteps = timesteps
        self.block_size = 1 if method == "lif" else block_options["block_size"]
        self.surrogate_function = make_sj_surrogate("triangle", 1.0)
        layout = ((64,), (128, "M"), (256, 256, "M"), (512, 512, "M"), (512, 512))
        for i, stage in enumerate(layout, 1):
            layers = []
            for channels in stage:
                if channels == "M":
                    layers.append(nn.AvgPool2d(2, 2))
                    continue
                cell_args = dict(tau=1.1, v_threshold=1., decay_input=False, v_reset=None,
                                 surrogate_function=self.surrogate_function, detach_reset=False)
                conv = nn.Conv2d(in_channels, channels, 3, padding=1, bias=True)
                norm = nn.BatchNorm2d(channels)
                if method == "lif":
                    neuron = SLTTBPTTNeuron(**cell_args)
                else:
                    neuron = TemporalBlockLIFNeuron(**cell_args, **block_options)
                layers.extend((conv, norm, neuron, layer.Dropout(dropout)))
                in_channels = channels
            setattr(self, f"layer{i}", nn.Sequential(*layers))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(in_channels, num_classes))
        # These aliases are part of the reference VGG checkpoint format.
        convs = [m for m in self.modules() if isinstance(m, nn.Conv2d)]
        self.conv1, self.conv2 = convs[:2]
        self.fc = self.classifier[-1]
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, .01)
                nn.init.zeros_(module.bias)

    def reset_state(self):
        functional.reset_net(self)
        for module in self.modules():
            if isinstance(module, TemporalBlockLIFNeuron):
                module.reset_state()

    def _chunk(self, x):
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4, self.layer5):
            for operation in stage:
                if x.ndim == 4 or isinstance(operation, TemporalBlockLIFNeuron):
                    x = operation(x)
                else:
                    steps, batch = x.shape[:2]
                    out = operation(x.flatten(0, 1))
                    x = out.reshape(steps, batch, *out.shape[1:])
        if x.ndim == 4:
            return self.classifier(self.avgpool(x))
        steps, batch = x.shape[:2]
        return self.classifier(self.avgpool(x.flatten(0, 1))).reshape(steps, batch, -1)

    def forward(self, images):
        if images.ndim == 4:
            sequence = images.unsqueeze(0).expand(self.timesteps, *images.shape)
        else:
            if images.ndim != 5 or images.shape[1] != self.timesteps:
                raise ValueError("Expected static images or a batch with T event frames per sample")
            sequence = images.transpose(0, 1).contiguous()
        if self.block_size == 1:
            logits = [self._chunk(step) for step in sequence]
        else:
            logits = []
            for chunk in sequence.split(self.block_size):
                logits.extend(self._chunk(chunk).unbind(0))
        # The measured model accumulated each timestep's logits in this order.
        mean_logits = sum(logits[1:], logits[0]) / len(logits)
        return mean_logits, torch.stack(logits)
