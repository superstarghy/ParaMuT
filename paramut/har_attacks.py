"""Signed standardized-sensor FGSM/PGD; reuse candidate and start utilities."""
import math
import torch
from torch.nn import functional as F
from .spikformer_robustness import random_noise, _keep_candidate

def gradient(model, x, y):
    if model.training: raise ValueError('Attack requires frozen eval-mode BN')
    x = x.detach().requires_grad_(True)
    logits = model(x)
    g, = torch.autograd.grad(F.cross_entropy(logits, y), x)
    if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite input gradient')
    return g.detach()

def attack(model, clean, y, epsilon, ids, steps=7):
    if model.training or not math.isfinite(epsilon) or epsilon < 0 or steps < 1:
        raise ValueError('Invalid attack configuration')
    if not torch.isfinite(clean).all(): raise ValueError('Nonfinite input')
    clean = clean.detach()
    with torch.no_grad():
        logits = model(clean)
        success = logits.argmax(1).ne(y)
        loss = F.cross_entropy(logits, y, reduction='none')
    best = clean.clone()
    if epsilon == 0: return clean.clone(), best
    lo, hi = clean-epsilon, clean+epsilon
    def project(x): return torch.maximum(lo, torch.minimum(hi, x)).detach()
    def keep(x):
        nonlocal best, success, loss
        with torch.no_grad():
            best, success, loss = _keep_candidate(best, success, loss, x, model(x), y)
    fgsm = project(clean + epsilon*gradient(model, clean, y).sign())
    keep(fgsm)
    x = project(clean + random_noise(clean, epsilon, sample_indices=ids))
    keep(x)
    for _ in range(steps):
        x = project(x + epsilon/4*gradient(model, x, y).sign())
        keep(x)
    return fgsm, best.detach()
