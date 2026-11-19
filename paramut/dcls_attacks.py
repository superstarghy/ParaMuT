"""Deterministic count-space attacks on the frozen integer-delay classifier."""
import math
import torch
from torch.nn import functional as F
from .spikformer_robustness import random_noise, _keep_candidate


def input_mask(x, lengths):
    return (torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]).unsqueeze(-1)


def gradient(model, x, y):
    if model.training:
        raise ValueError('BN and dropout must stay in evaluation mode')
    z = x.detach().requires_grad_(True)
    g, = torch.autograd.grad(F.cross_entropy(model(z), y), z)
    if not torch.isfinite(g).all():
        raise FloatingPointError('Nonfinite input gradient')
    return g.detach()


def attack(model, clean, labels, lengths, ids, epsilon, steps=7):
    if model.training or not math.isfinite(epsilon) or epsilon < 0 or steps < 1:
        raise ValueError('Invalid attack settings')
    clean = clean.detach()
    if not torch.isfinite(clean).all() or float(clean.min()) < 0:
        raise ValueError('Expected nonnegative finite raw counts')
    mask = input_mask(clean, lengths)
    if torch.count_nonzero(clean.masked_select(~mask.expand_as(clean))):
        raise ValueError('Artificial input padding must be zero')
    lo, hi = (clean-epsilon).clamp_min(0), clean+epsilon
    def project(z):
        return torch.where(mask, torch.maximum(lo, torch.minimum(hi, z)), clean).detach()
    with torch.no_grad():
        scores = model(clean)
        success = scores.argmax(1).ne(labels)
        loss = F.cross_entropy(scores, labels, reduction='none')
    best = clean.clone()
    if epsilon == 0:
        return clean.clone(), best
    def keep(z):
        nonlocal best, success, loss
        with torch.no_grad():
            best, success, loss = _keep_candidate(best, success, loss, z, model(z), labels)
    fgsm = project(clean + epsilon*gradient(model, clean, labels).sign())
    keep(fgsm)
    z = project(clean + random_noise(clean, epsilon, sample_indices=ids))
    keep(z)
    for _ in range(steps):
        z = project(z + epsilon/4*gradient(model, z, labels).sign())
        keep(z)
    return fgsm, best.detach()
