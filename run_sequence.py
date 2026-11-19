"""Train and evaluate the SHD and UCI HAR sequence classifiers."""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from paramut import har_spikecnn
from paramut.dcls_data import DCLSHDDataset, loader as frame_loader

DELAY_MODELS = {"lif": {"variant": "lif"}, "dcls_lif": {"variant": "dcls_lif"}}
for prefix, variant in (("pm", "paramut"), ("dcls_pm", "dcls_paramut")):
    for size in (4, 16, 32):
        for suffix, constraint, cap in (("none", "none", .75), ("cap075", "full_cap", .75),
                                        ("db", "db", .25)):
            DELAY_MODELS[f"{prefix}_k{size}_{suffix}"] = {
                "variant": variant, "K": size, "constraint": constraint, "cap": cap, "diagonal": .5}

GRIDS = {
    "dcls": [0., .005, .01, .02, .025, .05, .075, .1, .125, .15, .175, .2],
    "har": [0., .01, .025, .05, .1, .2],
    "sparch": [0., .0001, .00025, .0005, .001, .0025, .005, .01],
}


def protocol(dataset, variant):
    if dataset == "uci-har" and variant in har_spikecnn.VARIANTS:
        return "har"
    if dataset == "shd" and variant == "psn":
        return "sparch"
    if dataset == "shd" and variant in DELAY_MODELS:
        return "dcls"
    choices = har_spikecnn.VARIANTS if dataset == "uci-har" else [*DELAY_MODELS, "psn"]
    raise ValueError(f"Unknown {dataset} variant {variant!r}. Choose one of {', '.join(choices)}.")


def initialize(seed, recipe):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if recipe == "dcls":
        torch.backends.cudnn.enabled = False


def model_for(recipe, variant, seed, device):
    if recipe == "har":
        model = har_spikecnn.build(variant, seed)
    elif recipe == "sparch":
        from paramut.shd_psn import SHDPSN
        model = SHDPSN()
    else:
        from paramut.dcls_model import build
        model = build(**DELAY_MODELS[variant], seed=seed)
    return model.to(device)


def data_loader(recipe, root, split, batch_size):
    if recipe == "dcls":
        return frame_loader(DCLSHDDataset(root, split), batch_size, shuffle=split == "train")
    if recipe == "har":
        with np.load(root / "signals.npz") as signals:
            x = (signals[f"{split}_x"] - signals["mean"]) / signals["std"]
            y = signals[f"{split}_y"]
        data = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    else:
        from paramut.shd_psn import SpArchSHD
        data = SpArchSHD(root, split)
    return DataLoader(data, batch_size=batch_size, shuffle=split == "train")


def optimizers(model, recipe, epochs):
    if recipe == "har":
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        return [optimizer], [torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)]
    if recipe == "sparch":
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        return [optimizer], [torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=.7, patience=10, min_lr=1e-6)]
    groups = model.optimizer_groups()
    weights = [{"params": groups["weights"], "weight_decay": 1e-5},
               {"params": [], "weight_decay": 1e-5}, {"params": groups["bn"]}]
    if groups["temporal"]:
        weights.append({"params": groups["temporal"]})
    opts = [torch.optim.Adam(weights, lr=.001)]
    schedules = [torch.optim.lr_scheduler.OneCycleLR(opts[0], max_lr=.005, total_steps=epochs)]
    if groups["positions"]:
        opts.append(torch.optim.Adam(groups["positions"], lr=.1))
        schedules.append(torch.optim.lr_scheduler.CosineAnnealingLR(opts[1], T_max=epochs))
    return opts, schedules


@torch.no_grad()
def evaluate(model, loader, recipe, device):
    model.eval()
    if recipe == "dcls":
        model.prepare_inference()
    correct = count = 0
    per_batch = []
    for x, y, *_ in loader:
        predictions = model(x.to(device)).argmax(1)
        matches = predictions.eq(y.to(device))
        correct += int(matches.sum())
        count += len(y)
        per_batch.append(float(matches.float().mean()))
    return {"accuracy": correct / count, "batch_mean_accuracy": float(np.mean(per_batch))}


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def train(args, device):
    recipe = protocol(args.dataset, args.variant)
    initialize(args.seed, recipe)
    epochs = args.epochs if args.epochs is not None else {"dcls": 150, "har": 120, "sparch": 100}[recipe]
    if epochs < 1:
        raise ValueError("--epochs must be positive")
    batch_size = 256 if recipe == "dcls" else 128
    training = data_loader(recipe, args.data, "train", batch_size)
    testing = data_loader(recipe, args.data, "test", batch_size)
    model = model_for(recipe, args.variant, args.seed, device)
    opts, schedules = optimizers(model, recipe, epochs)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"dataset": args.dataset, "variant": args.variant, "seed": args.seed, "epochs": epochs}
    best = -1.
    for epoch in range(1, epochs + 1):
        model.train()
        correct = count = 0
        for x, y, *_ in training:
            x, y = x.to(device), y.to(device)
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            scores = model(x)
            targets = F.one_hot(y, 20).float() if recipe == "dcls" else y
            loss = F.cross_entropy(scores, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            if recipe == "har":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            for opt in opts:
                opt.step()
            if recipe == "dcls":
                model.clamp_delays_()
            correct += int(scores.detach().argmax(1).eq(y).sum())
            count += len(y)
        if recipe == "dcls":
            model.step_sigma(epoch - 1, epochs)
        metrics = evaluate(model, testing, recipe, device)
        selection = metrics["accuracy" if recipe == "har" else "batch_mean_accuracy"]
        for schedule in schedules:
            if recipe == "sparch":
                schedule.step(selection)
            else:
                schedule.step()
        if selection > best:
            best = selection
            checkpoint = {**meta, "epoch": epoch, "metrics": metrics, "model": model.state_dict()}
            temp = args.output / "best.tmp"
            torch.save(checkpoint, temp)
            temp.replace(args.output / "best.pt")
        row = {"epoch": epoch, "train_accuracy": correct / count, "selection_test": metrics}
        print(json.dumps(row), flush=True)
        with (args.output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    final = evaluate(model, testing, recipe, device)
    save_json(args.output / "training.json", {**meta, "best_epoch": checkpoint["epoch"],
              "selection_test": checkpoint["metrics"], "final_test": final, "test_selected": True})


def prediction_counts(model, recipe, x, y, lengths, ids, epsilon, batch_index):
    if recipe == "sparch":
        from paramut.sparch_attacks import attack_bank, bank_metrics
        search = [100000 + 100 * batch_index + i for i in range(4)]
        heldout = [10000000 + 100 * batch_index + i for i in range(8)]
        candidates = attack_bank(model, x, y, epsilon, search, sample_ids=ids)
        batches = (x, candidates["fgsm"], candidates["adversarial"])
        counts = [int(bank_metrics(model, z, y, heldout)[0].eq(y).sum()) for z in batches]
        return counts, len(y) * len(heldout)
    if recipe == "dcls":
        from paramut.dcls_attacks import attack
        fgsm, pgd = attack(model, x, y, lengths, ids, epsilon)
    else:
        from paramut.har_attacks import attack
        fgsm, pgd = attack(model, x, y, epsilon, ids)
    with torch.no_grad():
        counts = [int(model(z).argmax(1).eq(y).sum()) for z in (x, fgsm, pgd)]
    return counts, len(y)


def robustness(args, device):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    recipe = protocol(checkpoint["dataset"], checkpoint["variant"])
    initialize(checkpoint["seed"], recipe)
    model = model_for(recipe, checkpoint["variant"], checkpoint["seed"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    if recipe == "dcls":
        model.prepare_inference()
    grid = GRIDS[recipe] if args.epsilons is None else args.epsilons
    if sorted(set(grid)) != grid or any(not math.isfinite(e) or e < 0 for e in grid):
        raise ValueError("Epsilons must be distinct, finite, nonnegative and increasing.")
    parent_size = {"dcls": 256, "har": 64, "sparch": 32}[recipe]
    batches = data_loader(recipe, args.data, "test", parent_size)
    totals = np.zeros((len(grid), 3), dtype=np.int64)
    position = decisions = 0
    for index, (x, y, *metadata) in enumerate(batches):
        # Slicing the batch tensor preserves the parent group's temporal padding.
        size = 32 if recipe == "dcls" else parent_size
        for start in range(0, len(y), size):
            sl = slice(start, start + size)
            xx, yy = x[sl].to(device), y[sl].to(device)
            ids = list(range(position + start, position + start + len(yy)))
            lengths = None
            if recipe == "dcls":
                lengths, ids = metadata[0][sl].to(device), metadata[1][sl].tolist()
            for j, eps in enumerate(grid):
                counts, draws = prediction_counts(model, recipe, xx, yy, lengths, ids, eps, index)
                totals[j] += counts
                if j == 0:
                    decisions += draws
        position += len(y)
        print(f"Evaluated {position}/{len(batches.dataset)} test samples", flush=True)
    accuracies = totals / decisions
    names = ("clean", "fgsm", "pgd7")
    curves = [{"epsilon": eps, **dict(zip(names, row.tolist()))} for eps, row in zip(grid, accuracies)]
    result = {"dataset": checkpoint["dataset"], "variant": checkpoint["variant"], "seed": checkpoint["seed"],
              "epoch": checkpoint["epoch"], "samples": position, "curves": curves}
    if len(grid) > 1:
        area = (np.diff(grid)[:, None] * (accuracies[1:] + accuracies[:-1]) / 2).sum(axis=0)
        result["normalized_auc"] = dict(zip(names, (area / (grid[-1] - grid[0])).tolist()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    training = sub.add_parser("train", help="Train a classifier and save its best test-selected checkpoint.")
    training.add_argument("--dataset", choices=["shd", "uci-har"], required=True)
    training.add_argument("--variant", required=True, help="Model identifier (see docs/experiments/sequence.md).")
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--epochs", type=int, help="Override the study's epoch budget.")
    attack = sub.add_parser("robustness", help="Evaluate clean, FGSM and PGD7 accuracy.")
    attack.add_argument("--checkpoint", type=Path, required=True)
    attack.add_argument("--epsilons", type=float, nargs="+", help="Override the study's perturbation grid.")
    for command in (training, attack):
        command.add_argument("--data", type=Path, required=True, help="Prepared dataset directory.")
        command.add_argument("--output", type=Path, required=True, help="New training directory or results JSON file.")
        command.add_argument("--device", type=torch.device, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    (train if args.action == "train" else robustness)(args, args.device)


if __name__ == "__main__":
    main()
