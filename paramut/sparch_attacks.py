"""Finite-bank EOT attacks for native random-state SpArch classifiers.

Optimize mean per-draw CE, not CE of averaged scores. Choose one input per
sample on a fixed search bank; report it on independent paired state draws.
The search-bank candidate envelope is not a stochastic robustness certificate.
"""
import math
from contextlib import contextmanager

import torch
from torch.nn import functional as F

from .spikformer_robustness import project_linf, random_noise


@contextmanager
def state_draw(seed, device):
    devices = [torch.device(device).index or torch.cuda.current_device()] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        yield


def scores(model, inputs):
    value = model(inputs)
    return value[0] if isinstance(value, tuple) else value


def bank_metrics(model, inputs, labels, seeds):
    predictions, losses = [], []
    with torch.no_grad():
        for seed in seeds:
            with state_draw(seed, inputs.device):
                output = scores(model, inputs)
            losses.append(F.cross_entropy(output, labels, reduction='none'))
            predictions.append(output.argmax(1))
    return torch.stack(predictions), torch.stack(losses).mean(0)


def bank_gradient(model, inputs, labels, seeds):
    if model.training or not seeds:
        raise ValueError('EOT requires eval mode and a nonempty seed bank')
    gradient = torch.zeros_like(inputs)
    # Free each draw graph separately: peak memory does not multiply by EOT.
    for seed in seeds:
        value = inputs.detach().requires_grad_(True)
        with state_draw(seed, inputs.device):
            loss = F.cross_entropy(scores(model, value), labels)
        current, = torch.autograd.grad(loss, value)
        if not torch.isfinite(current).all():
            raise FloatingPointError('Nonfinite EOT gradient')
        gradient.add_(current.detach() / len(seeds))
    return gradient


def attack_bank(model, clean, labels, epsilon, search_seeds, *, sample_ids,
                steps=7, alpha_ratio=.25, random_seed=12345):
    if not math.isfinite(epsilon) or epsilon < 0 or steps < 1 or alpha_ratio <= 0:
        raise ValueError('Invalid attack budget')
    if model.training or len(set(search_seeds)) != len(search_seeds) or not search_seeds:
        raise ValueError('Expected eval mode and distinct search-state seeds')
    clean = clean.detach().float()
    project_linf(clean, clean, epsilon)  # audited finite/nonnegative validation
    clean_predictions, best_loss = bank_metrics(model, clean, labels, search_seeds)
    best_correct = clean_predictions.eq(labels).float().mean(0)
    best = clean.clone()
    candidate_count = 1

    def keep(candidate):
        nonlocal best, best_correct, best_loss, candidate_count
        predictions, loss = bank_metrics(model, candidate, labels, search_seeds)
        correct = predictions.eq(labels).float().mean(0)
        replace = (correct < best_correct) | ((correct == best_correct) & (loss > best_loss))
        best = torch.where(replace.reshape((-1,) + (1,) * (clean.ndim - 1)), candidate, best)
        best_correct = torch.where(replace, correct, best_correct)
        best_loss = torch.where(replace, loss, best_loss)
        candidate_count += 1

    fgsm = clean.clone()
    if epsilon:
        gradient = bank_gradient(model, clean, labels, search_seeds)
        fgsm = project_linf(clean + epsilon * gradient.sign(), clean, epsilon).detach()
        keep(fgsm)
        lower, upper = (clean - epsilon).clamp_min(0), clean + epsilon
        uniform = (random_noise(clean, 1., random_seed, sample_ids, 0) + 1.) / 2
        candidate = project_linf(lower + uniform * (upper - lower), clean, epsilon).detach()
        keep(candidate)
        for _ in range(steps):
            gradient = bank_gradient(model, candidate, labels, search_seeds)
            candidate = project_linf(candidate + epsilon * alpha_ratio * gradient.sign(), clean, epsilon).detach()
            keep(candidate)
    actual, _ = bank_metrics(model, best, labels, search_seeds)
    if not torch.equal(actual.eq(labels).float().mean(0), best_correct):
        raise RuntimeError('Selected EOT candidate does not reproduce its search metric')
    return dict(fgsm=fgsm, adversarial=best.detach(), search_clean_predictions=clean_predictions,
                search_selected_predictions=actual, search_selected_loss=best_loss,
                candidate_count=candidate_count,
                gradient_draws=(steps + 1) * len(search_seeds) if epsilon else 0)
