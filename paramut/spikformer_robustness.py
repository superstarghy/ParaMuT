"""Raw-input, state-safe FGSM and PGD for the reproduced Spikformer v1.

CIFAR budgets use pixel values in [0, 1], before channel normalization. DVS
budgets use nonnegative, continuous frame counts, with no artificial upper
bound and no rounding. This is a frame-space threat, not a discrete event edit.

The hard spiking forward is unchanged. ``native`` uses the model's existing
surrogate backward; ``full_state`` additionally removes membrane/reset detach
operations during input differentiation. Batch normalization always stays in
evaluation mode. Neither surrogate attack certifies robustness.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F

from .torch_models import SJLIFNode, TemporalBlockLIFNeuron


# Match _spikformer_cifar_transforms in torch_data.py, not the legacy recipe.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
GRADIENT_MODES = ("native", "full_state")


def _check_mode(mode):
    if mode not in GRADIENT_MODES:
        raise ValueError(f"gradient_mode must be one of {GRADIENT_MODES}, got {mode!r}")
    return mode


class RawInputClassifier(nn.Module):
    """Expose [B, classes] logits for raw CIFAR pixels or DVS frame counts.

    A CIFAR perturbation is shared across all T repetitions of the same image.
    DVS inputs already contain T frames. State management is explicit through
    ``reset_state`` and is performed by the attack/evaluation helpers below.
    """

    def __init__(self, model, dataset, timesteps, gradient_mode="full_state"):
        super().__init__()
        if dataset not in {"cifar10", "cifar10dvs"}:
            raise ValueError("dataset must be cifar10 or cifar10dvs")
        if not isinstance(timesteps, Integral) or timesteps < 1:
            raise ValueError("timesteps must be a positive integer")
        self.model = model.float()
        self.dataset = dataset
        self.timesteps = int(timesteps)
        self.gradient_mode = _check_mode(gradient_mode)
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1), persistent=False)

    def reset_state(self):
        reset = getattr(self.model, "reset_state", None)
        if reset is not None:
            reset()

    def forward(self, raw):
        raw = raw.float()
        if self.dataset == "cifar10":
            if raw.ndim != 4 or raw.shape[1] != 3:
                raise ValueError("CIFAR input must have shape [B, 3, H, W]")
            images = (raw - self.mean) / self.std
            sequence = images.unsqueeze(0).expand(self.timesteps, -1, -1, -1, -1).contiguous()
        else:
            if raw.ndim != 5 or raw.shape[1] != self.timesteps or raw.shape[2] != 2:
                raise ValueError("DVS input must have shape [B, T, 2, H, W] with the configured T")
            sequence = raw.transpose(0, 1).contiguous()
        logits = self.model(sequence)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError("Spikformer must return [T, B, classes] logits")
        if logits.shape[:2] != (self.timesteps, raw.shape[0]):
            raise ValueError("Spikformer output T/B dimensions do not match the input")
        return logits.mean(0)


@contextmanager
def attack_gradient_mode(model, mode="full_state"):
    """Temporarily remove only state/reset detach operations, never train BN.

    Assigning the block cell's own ``training`` flag avoids its eval-only state
    detach without recursively changing any child or sibling module. All flags
    are restored even if forward/backward fails. Call with the model in eval.
    """
    mode = _check_mode(mode)
    saved = []
    try:
        if mode == "full_state":
            for module in model.modules():
                if isinstance(module, TemporalBlockLIFNeuron):
                    saved.append((module, "training", module.training))
                    saved.append((module, "detach_reset", module.detach_reset))
                    module.training = True
                    module.detach_reset = False
                elif SJLIFNode is not None and isinstance(module, SJLIFNode):
                    saved.append((module, "detach_reset", module.detach_reset))
                    module.detach_reset = False
        yield
    finally:
        for module, name, value in reversed(saved):
            setattr(module, name, value)


def _epsilon(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    return value


def _upper(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("upper_bound must be None or finite and positive")
    return value


def _check_input(clean, upper_bound=None):
    if not isinstance(clean, torch.Tensor) or not clean.is_floating_point() or clean.ndim < 2:
        raise ValueError("raw input must be a floating tensor with a batch dimension")
    if clean.shape[0] == 0 or not torch.isfinite(clean).all():
        raise ValueError("raw input must be nonempty and finite")
    if (clean < 0).any() or (upper_bound is not None and (clean > upper_bound).any()):
        raise ValueError("clean input lies outside the raw input domain")


def _project(candidate, clean, epsilon, upper_bound):
    if epsilon == 0:
        return clean.clone()
    projected = torch.maximum(torch.minimum(candidate, clean + epsilon), clean - epsilon)
    return projected.clamp(min=0., max=upper_bound)


def project_linf(candidate, clean, epsilon, upper_bound=None):
    """Project onto the raw L-infinity ball intersected with the valid domain."""
    epsilon, upper_bound = _epsilon(epsilon), _upper(upper_bound)
    _check_input(clean, upper_bound)
    if candidate.shape != clean.shape or candidate.device != clean.device:
        raise ValueError("candidate and clean must have matching shapes/devices")
    if not torch.isfinite(candidate).all():
        raise ValueError("candidate contains nonfinite values")
    return _project(candidate, clean, epsilon, upper_bound)


def _reset(model):
    reset = getattr(model, "reset_state", None)
    if reset is not None:
        reset()


def _check_logits(logits, batch):
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[0] != batch:
        raise ValueError("classifier must return [B, classes] logits")
    if not torch.isfinite(logits).all():
        raise RuntimeError("classifier produced nonfinite logits")


def predict_logits(model, x):
    """Evaluate the unchanged hard classifier in FP32, with fresh state."""
    model.eval()
    _reset(model)
    try:
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            logits = model(x.float())
            _check_logits(logits, x.shape[0])
            return logits.detach()
    finally:
        _reset(model)


def input_gradient(model, x, y, gradient_mode=None):
    """Return hard-label CE input gradients and corresponding detached logits.

    CE is summed, so per-example gradients do not acquire a batch-size scale.
    No parameter .grad buffers are populated. Disconnected/nonfinite gradients
    fail explicitly; zero gradients are valid and must be reported by callers.
    """
    mode = _check_mode(getattr(model, "gradient_mode", "full_state") if gradient_mode is None else gradient_mode)
    model.eval()
    _reset(model)
    try:
        with torch.enable_grad(), torch.autocast(device_type=x.device.type, enabled=False), attack_gradient_mode(model, mode):
            inputs = x.detach().float().requires_grad_(True)
            logits = model(inputs)
            _check_logits(logits, x.shape[0])
            loss = F.cross_entropy(logits, y, reduction="sum")
            if not loss.requires_grad:
                raise RuntimeError("attack loss is disconnected from the input")
            gradient = torch.autograd.grad(loss, inputs, allow_unused=True)[0]
            if gradient is None:
                raise RuntimeError("attack loss is disconnected from the input")
            if not torch.isfinite(gradient).all():
                raise RuntimeError("attack produced nonfinite input gradients")
            return gradient.detach(), logits.detach()
    finally:
        _reset(model)


def fgsm_attack(model, clean, y, epsilon, upper_bound=None, gradient_mode="full_state"):
    """One untargeted sign-gradient step in raw pixels or raw frame counts."""
    epsilon, upper_bound = _epsilon(epsilon), _upper(upper_bound)
    _check_mode(gradient_mode)
    _check_input(clean, upper_bound)
    if epsilon == 0:
        return clean.detach().clone()
    gradient, _ = input_gradient(model, clean, y, gradient_mode=gradient_mode)
    return _project(clean.detach() + epsilon * gradient.sign(), clean.detach(), epsilon, upper_bound).detach()


def _sample_ids(clean, sample_indices):
    if sample_indices is None:
        return list(range(clean.shape[0]))
    if isinstance(sample_indices, torch.Tensor):
        if sample_indices.ndim != 1 or sample_indices.dtype not in (
            torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64
        ):
            raise ValueError("sample_indices must be a one-dimensional integer tensor or sequence")
        sample_indices = sample_indices.detach().cpu().tolist()
    indices = list(sample_indices)
    if len(indices) != clean.shape[0] or any(not isinstance(i, Integral) or i < 0 for i in indices):
        raise ValueError("sample_indices must supply one nonnegative integer per example")
    return [int(i) for i in indices]


def random_noise(clean, epsilon, attack_seed=12345, sample_indices=None, restart=0):
    """Common deterministic uniform starts, independent of global RNG/batching.

    Seed derivation includes only the attack seed, dataset sample index, and
    restart. The underlying unit noise is shared across budgets and models.
    Pass global dataset indices for invariance to batch partitioning; omitted
    indices are batch-local. CPU generation also fixes noise across GPU types.
    """
    epsilon = _epsilon(epsilon)
    if not isinstance(attack_seed, Integral) or not isinstance(restart, Integral) or restart < 0:
        raise ValueError("attack_seed and restart must be integers, with restart nonnegative")
    indices = _sample_ids(clean, sample_indices)
    if epsilon == 0:
        return torch.zeros_like(clean)
    noise = []
    for sample in indices:
        key = f"spikformer_raw_pgd_v1:{int(attack_seed)}:{sample}:{int(restart)}".encode("ascii")
        seed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise.append(torch.empty(clean.shape[1:], dtype=torch.float32, device="cpu").uniform_(-1., 1., generator=generator))
    return torch.stack(noise).to(device=clean.device, dtype=clean.dtype).mul_(epsilon)


def _scores(logits, y):
    return logits.argmax(1).ne(y), F.cross_entropy(logits, y, reduction="none")


def _keep_candidate(best, best_success, best_loss, candidate, logits, y):
    success, loss = _scores(logits, y)
    replace = (success & ~best_success) | (success.eq(best_success) & loss.gt(best_loss))
    mask = replace.view((-1,) + (1,) * (candidate.ndim - 1))
    return (torch.where(mask, candidate.detach(), best),
            torch.where(replace, success, best_success),
            torch.where(replace, loss, best_loss))


def select_adversarial(model, clean, y, candidates):
    """Per example, keep any successful hard attack before comparing CE loss.

    Clean is always a candidate, so a clean error can never become a robust
    success. Ties keep the earlier candidate for deterministic selection.
    """
    best = clean.detach().clone()
    best_success, best_loss = _scores(predict_logits(model, best), y)
    if isinstance(candidates, torch.Tensor):
        candidates = [candidates]
    for candidate in candidates:
        if candidate.shape != clean.shape or candidate.device != clean.device:
            raise ValueError("candidate and clean must have matching shapes/devices")
        if not torch.isfinite(candidate).all():
            raise ValueError("candidate contains nonfinite values")
        logits = predict_logits(model, candidate)
        best, best_success, best_loss = _keep_candidate(best, best_success, best_loss, candidate, logits, y)
    return best.detach()


def pgd_attack(model, clean, y, epsilon, *, steps=20, restarts=2, alpha_ratio=.25,
               upper_bound=None, attack_seed=12345, sample_indices=None,
               fgsm_candidate=None, gradient_mode="full_state"):
    """Untargeted L-infinity PGD, retaining every evaluated candidate.

    The candidate union contains clean, FGSM, each random start, every iterate,
    and every restart's last point. Misclassification takes precedence over CE
    loss, so the returned attack is never weaker than its included candidates.
    All scoring uses hard evaluation predictions, not surrogate activations.
    """
    epsilon, upper_bound = _epsilon(epsilon), _upper(upper_bound)
    _check_mode(gradient_mode)
    _check_input(clean, upper_bound)
    if not isinstance(steps, Integral) or steps < 1 or not isinstance(restarts, Integral) or restarts < 1:
        raise ValueError("steps and restarts must be positive integers")
    alpha_ratio = float(alpha_ratio)
    if not math.isfinite(alpha_ratio) or alpha_ratio <= 0:
        raise ValueError("alpha_ratio must be finite and positive")
    indices = _sample_ids(clean, sample_indices)
    if epsilon == 0:
        return clean.detach().clone()
    clean = clean.detach()
    best = clean.clone()
    best_success, best_loss = _scores(predict_logits(model, clean), y)
    if fgsm_candidate is None:
        fgsm_candidate = fgsm_attack(model, clean, y, epsilon, upper_bound, gradient_mode)
    if fgsm_candidate.shape != clean.shape or fgsm_candidate.device != clean.device:
        raise ValueError("FGSM candidate and clean must have matching shapes/devices")
    if not torch.isfinite(fgsm_candidate).all() or not torch.equal(
        fgsm_candidate, _project(fgsm_candidate, clean, epsilon, upper_bound)
    ):
        raise ValueError("FGSM candidate violates the raw input domain or L-infinity budget")
    best, best_success, best_loss = _keep_candidate(
        best, best_success, best_loss, fgsm_candidate, predict_logits(model, fgsm_candidate), y)
    alpha = epsilon * alpha_ratio
    for restart in range(int(restarts)):
        noise = random_noise(clean, epsilon, attack_seed, indices, restart)
        adversarial = _project(clean + noise, clean, epsilon, upper_bound).detach()
        for _ in range(int(steps)):
            gradient, logits = input_gradient(model, adversarial, y, gradient_mode=gradient_mode)
            best, best_success, best_loss = _keep_candidate(
                best, best_success, best_loss, adversarial, logits, y)
            adversarial = _project(adversarial + alpha * gradient.sign(), clean, epsilon, upper_bound).detach()
        best, best_success, best_loss = _keep_candidate(
            best, best_success, best_loss, adversarial, predict_logits(model, adversarial), y)
    return best.detach()
