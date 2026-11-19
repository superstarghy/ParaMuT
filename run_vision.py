#!/usr/bin/env python3
"""Train and evaluate the VGG11-BN vision comparisons."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from paramut.torch_data import build_loaders
from paramut.vision_corruptions import EVENT_CORRUPTION_MODES, corrupt_event_frames
from paramut.vision_vgg import VisionVGG


@dataclass(frozen=True)
class Recipe:
    timesteps: int
    block_size: int
    epochs: int
    batch_size: int
    seed: int
    eval_interval: int
    image_size: int
    diagonal: float
    cap: float


RECIPES = {
    "cifar10": Recipe(8, 4, 200, 128, 0, 50, 32, 1., .25),
    "cifar100": Recipe(8, 4, 200, 128, 0, 20, 32, .5, .25),
    "cifar10dvs": Recipe(10, 5, 100, 16, 1, 10, 48, .75, .1),
    "dvs128gesture": Recipe(10, 5, 100, 8, 1, 10, 128, .25, .125),
}
DVS = {"cifar10dvs", "dvs128gesture"}
CIFAR_BUDGETS = (.0039216, .0078431, .0156863, .0313725)
DVS_BUDGETS = (.0005, .001, .0015, .002, .0025, .003, .004, .005, .0075, .01)


def data_for(dataset, args, train):
    recipe = RECIPES[dataset]
    return build_loaders(
        dataset, args.data_dir, args.batch_size or recipe.batch_size,
        getattr(args, "train_limit", 0), args.test_limit, getattr(args, "download", False),
        num_workers=args.workers, augment=train, dvs_frames=recipe.timesteps, dvs_resize=recipe.image_size,
    )


def model_for(dataset, method, info):
    recipe = RECIPES[dataset]
    block = {}
    if method == "psn":
        block = dict(block_size=recipe.timesteps, temporal_operator="learnable_full", block_reset="none")
    elif method == "paramut":
        block = dict(block_size=recipe.block_size, temporal_operator="learnable_causal", block_reset="soft",
                     temporal_mask_parameterization="diagonal_residual", temporal_mask_diagonal=recipe.diagonal,
                     temporal_mask_constraint="row_col_l1", temporal_mask_cap=recipe.cap)
        if dataset == "cifar10dvs":
            block.update(temporal_mask_diagonal_mode="learnable", temporal_mask_diagonal_min=.5,
                         temporal_mask_diagonal_max=1.)
    elif method != "lif":
        raise ValueError(f"Unknown method {method}")
    return VisionVGG(info.in_channels, info.num_classes, .1 if dataset in DVS else 0.,
                     recipe.timesteps, method, block)


def accuracy(model, loader, device, perturb=None):
    model.eval()
    correct = samples = 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        if perturb is not None:
            x = perturb(x, y, i)
        with torch.no_grad():
            scores, _ = model(x)
            model.reset_state()
        correct += (scores.argmax(1) == y).sum().item()
        samples += len(y)
    return correct / samples, samples


def train(args):
    dataset, method = args.dataset, args.method
    recipe = RECIPES[dataset]
    seed = recipe.seed if args.seed is None else args.seed
    epochs = args.epochs or recipe.epochs
    torch.manual_seed(seed)
    train_loader, test_loader, info = data_for(dataset, args, train=True)
    model = model_for(dataset, method, info).to(args.device)
    event = dataset in DVS
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4 if event else 5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected = output / "model.pt"
    policy = "last" if dataset == "cifar10" else "best"
    best = -1.
    with (output / "history.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "training_loss", "training_accuracy", "test_accuracy", "learning_rate"])
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = correct = samples = 0
            for x, y in train_loader:
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                scores, steps = model(x)
                predictions = steps.flatten(0, 1)
                labels = y.repeat(steps.shape[0])
                loss = .95 * F.cross_entropy(predictions, labels, label_smoothing=.05 if event else 0.)
                loss += .05 * F.mse_loss(predictions, torch.ones_like(predictions))
                loss.backward()
                optimizer.step()
                model.reset_state()
                total_loss += loss.item() * len(y)
                correct += (scores.detach().argmax(1) == y).sum().item()
                samples += len(y)
            test_accuracy = None
            if epoch % recipe.eval_interval == 0 or epoch == epochs:
                test_accuracy, _ = accuracy(model, test_loader, args.device)
                if (policy == "best" and test_accuracy > best) or (policy == "last" and epoch == epochs):
                    tmp = selected.with_suffix(".tmp")
                    torch.save(dict(dataset=dataset, method=method, epoch=epoch, seed=seed,
                                    train_limit=args.train_limit, test_limit=args.test_limit,
                                    checkpoint_policy=policy, test_accuracy=test_accuracy,
                                    model_state_dict=model.state_dict()), tmp)
                    tmp.replace(selected)
                best = max(best, test_accuracy)
            writer.writerow([epoch, total_loss / samples, correct / samples, test_accuracy, scheduler.get_last_lr()[0]])
            stream.flush()
            print(f"Epoch {epoch}/{epochs}, training accuracy {correct / samples:.4f}, test accuracy {test_accuracy}", flush=True)
            scheduler.step()
    print(f"Selected checkpoint: {selected}")


def attack(model, x, y, info, epsilon, steps, pgd):
    # The VGG direct encoder passes the pixel/frame gradient through unchanged.
    # The original BPDA setting and native direct-input gradients coincide.
    mean = x.new_tensor(info.input_mean).view(1, -1, 1, 1) if info.input_mean else 0.
    std = x.new_tensor(info.input_std).view(1, -1, 1, 1) if info.input_std else 1.
    raw = (x * std + mean).clamp(0, 1).detach()
    if pgd:
        start = (raw + torch.empty_like(raw).uniform_(-epsilon, epsilon)).clamp(0, 1)
        adv = (start - mean) / std
    else:
        adv = x.detach()
    for _ in range(steps if pgd else 1):
        adv = adv.detach().requires_grad_(True)
        logits, _ = model(adv)
        grad, = torch.autograd.grad(F.cross_entropy(logits, y), adv)
        model.reset_state()
        raw_adv = (adv.detach() * std + mean).clamp(0, 1)
        raw_adv = raw_adv + (epsilon * .25 if pgd else epsilon) * grad.sign()
        raw_adv = torch.maximum(raw - epsilon, torch.minimum(raw + epsilon, raw_adv))
        adv = (raw_adv.clamp(0, 1) - mean) / std
    return adv.detach()


def evaluate(args):
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    dataset, method = saved["dataset"], saved["method"]
    if args.test_limit is None:
        args.test_limit = saved.get("test_limit", 0)
    _, test, info = data_for(dataset, args, train=False)
    model = model_for(dataset, method, info).to(args.device).eval()
    model.load_state_dict(saved["model_state_dict"])
    model.requires_grad_(False)
    torch.manual_seed(args.seed)
    clean, samples = accuracy(model, test, args.device)
    budgets = args.epsilons or (DVS_BUDGETS if dataset in DVS else CIFAR_BUDGETS)
    if args.event_corruptions and dataset not in DVS:
        raise ValueError("--event-corruptions requires an event dataset")
    rows = []
    for metric in ("fgsm", "pgd"):
        rows.append([metric, 0., clean])
        for epsilon in budgets:
            acc, _ = accuracy(model, test, args.device, lambda x, y, i: attack(
                model, x, y, info, epsilon, args.pgd_steps, pgd=metric == "pgd"))
            rows.append([metric, epsilon, acc])
    if args.event_corruptions:
        for mode_index, mode in enumerate(EVENT_CORRUPTION_MODES):
            rows.append([mode, 0., clean])
            for strength in (.01, .03, .05, .1):
                def corrupt(x, y, i):
                    rng = torch.Generator(device=x.device).manual_seed(12345 + 10007 * mode_index + i)
                    return corrupt_event_frames(x, strength, mode, generator=rng)
                acc, _ = accuracy(model, test, args.device, corrupt)
                rows.append([mode, strength, acc])
    path = args.output or Path(args.checkpoint).with_name("robustness.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "method", "evaluation", "strength", "accuracy", "samples"])
        writer.writerows([[dataset, method, *row, samples] for row in rows])
    print(f"Clean accuracy {clean:.4f} on {samples} samples. Wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train one dataset/method comparison.")
    training.add_argument("--dataset", required=True, choices=RECIPES)
    training.add_argument("--method", required=True, choices=["lif", "paramut", "psn"])
    training.add_argument("--output-dir", required=True, help="New run directory.")
    training.add_argument("--download", action="store_true")
    training.add_argument("--epochs", type=int)
    training.add_argument("--train-limit", type=int, default=0)
    training.add_argument("--seed", type=int, help="Override the dataset's training seed.")
    scoring = commands.add_parser("evaluate", help="Evaluate clean accuracy and FGSM/PGD curves.")
    scoring.add_argument("--checkpoint", required=True)
    scoring.add_argument("--output", type=Path, help="Defaults to robustness.csv beside the checkpoint.")
    scoring.add_argument("--seed", type=int, default=12345, help="PGD random start seed.")
    scoring.add_argument("--epsilons", type=float, nargs="+", help="Positive budgets in ascending order.")
    scoring.add_argument("--pgd-steps", type=int, default=7)
    scoring.add_argument("--event-corruptions", action="store_true", help="Include five event-frame corruption curves.")
    for sub in (training, scoring):
        sub.add_argument("--data-dir", required=True)
        sub.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        sub.add_argument("--workers", type=int, default=8)
        sub.add_argument("--batch-size", type=int, help="Override the dataset's training/evaluation batch size.")
        sub.add_argument("--test-limit", type=int, default=0 if sub is training else None,
                         help="Number of test samples. Zero evaluates the complete test set.")
    args = parser.parse_args()
    if args.workers < 0 or (args.test_limit or 0) < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error("workers and limits must be nonnegative and batch size positive")
    if args.command == "train":
        if args.train_limit < 0 or (args.epochs is not None and args.epochs < 1):
            parser.error("train limit must be nonnegative and epochs positive")
        train(args)
    else:
        if args.pgd_steps < 1 or (args.epsilons and (any(not 0 < x <= 1 for x in args.epsilons)
                                                      or sorted(set(args.epsilons)) != args.epsilons)):
            parser.error("PGD steps must be positive and epsilons strictly increasing within (0, 1]")
        evaluate(args)


if __name__ == "__main__":
    main()
