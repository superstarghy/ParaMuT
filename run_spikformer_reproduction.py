#!/usr/bin/env python3
"""Train the dataset-specific Spikformer v1 recipes with matched neuron variants.

This intentionally does not reuse the historical compact Spikformer training
defaults. Checkpoints include optimizer, scheduler, scaler and RNG states and
can resume at an epoch boundary with ``--resume auto``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset
from timm.data import Mixup
from timm.data.random_erasing import RandomErasing
from timm.optim.optim_factory import param_groups_weight_decay
from timm.scheduler import CosineLRScheduler

from paramut.torch_data import build_loaders


UPSTREAM_COMMIT = "621268981827af47aa666c1b5f774de4281d90ad"
RECIPE_VERSION = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cifar10", "cifar10dvs"], required=True)
    parser.add_argument("--method", choices=["lif", "block", "psn"], default="lif")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", "--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None, help="Cosine cycle length, including warmup")
    parser.add_argument("--cooldown-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=.06)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--warmup-lr", type=float, default=1e-5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--mixup-off-epoch", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 AMP; default FP32 for all methods")
    parser.add_argument("--resume", default="auto", help="auto, none, or a last.pt path")
    parser.add_argument("--continue-from", default="", metavar="LAST_PT",
                        help="Branch a completed run into a new directory, extending only its cooldown")
    parser.add_argument("--evaluate-checkpoint", default="")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--synthetic-smoke-only", action="store_true")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-test-batches", type=int, default=0)
    args = parser.parse_args(argv)
    static = args.dataset == "cifar10"
    defaults = dict(seed=42 if static else 2021, epochs=400 if static else 96, batch_size=128 if static else 16,
                    timesteps=4 if static else 16, embed_dim=384 if static else 256,
                    depth=4 if static else 2, num_heads=12 if static else 16,
                    lr=.0005 if static else .001, warmup_epochs=20 if static else 10,
                    mixup_off_epoch=200 if static else 75)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.block_size is None:
        args.block_size = max(args.timesteps // 2, 1)
    if args.epochs < 1 or args.cooldown_epochs < 0 or args.timesteps < 1:
        parser.error("epochs/timesteps must be positive and cooldown nonnegative")
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("Mixup requires an even batch size >= 2")
    if args.block_size < 1 or args.timesteps % args.block_size:
        parser.error("block size must divide timesteps")
    if args.continue_from and (args.evaluate_checkpoint or args.prepare_data_only or args.synthetic_smoke_only):
        parser.error("--continue-from is for training, not evaluation/preparation/smoke modes")
    if args.continue_from and args.resume not in {"auto", "none"}:
        parser.error("Use --resume auto or none with --continue-from")
    return args


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def config_for(args):
    ignored = {"resume", "output_dir", "device", "workers", "download", "prepare_data_only",
               "synthetic_smoke_only", "evaluate_checkpoint", "continue_from"}
    config = {k: v for k, v in vars(args).items() if k not in ignored}
    config.update(recipe_version=RECIPE_VERSION, upstream_commit=UPSTREAM_COMMIT,
                  model="spikformer_v1", data_recipe="spikformer", bn_time_batch="full_sequence",
                  effective_neuron_block_size=(1 if args.method == "lif" else
                                               args.timesteps if args.method == "psn" else args.block_size),
                  optimizer="AdamW", weight_decay_excludes="bias and 1D parameters",
                  loss="cross_entropy_of_time_mean_logits", label_smoothing=.1,
                  mixup_alpha=.5, mixup_probability=1. if args.dataset == "cifar10" else .5,
                  checkpoint_selection="maximum test accuracy (upstream protocol)",
                  total_epochs=args.epochs + args.cooldown_epochs,
                  dvs_split="per-class sequential 90/10", dvs_split_by="number",
                  random_erasing=.25 if args.dataset == "cifar10" else 0.,
                  block_parameterization="D+B; D=1 fixed; B row/column L1 cap=.25; soft boundary reset",
                  block_initial_residual_max_norm_fraction=.5,
                  psn_definition="full T-by-T matrix, fixed threshold, no reset; PSN-style only")
    return config


def make_model(args):
    from paramut.torch_spikformer_v1 import SpikformerV1
    return SpikformerV1(dataset=args.dataset, neuron=args.method, timesteps=args.timesteps,
                        block_size=args.block_size, embed_dim=args.embed_dim, depth=args.depth,
                        num_heads=args.num_heads)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_continuation(source, output, config, loaders):
    """Validate a completed parent and carry its full training state forward.

    Extending cooldown preserves the entire learning-rate history. Changing
    t_initial, neurons, data or optimizer settings requires a separate experiment.
    The source directory is read-only; a child retains the parent's best even
    when no additional epoch improves its accuracy.
    """
    source = Path(source).resolve()
    if source.name != "last.pt":
        raise ValueError("--continue-from must reference last.pt, not a selected best checkpoint")
    if source.parent == Path(output).resolve():
        raise ValueError("Continuation requires a new output directory; preserve the source run")
    parent = torch.load(source, map_location="cpu", weights_only=False)
    original = parent["config"]
    allowed = {"cooldown_epochs", "total_epochs", "data_dir", "recipe_version"}
    legacy_root = original["data_dir"] if original.get("recipe_version", 1) == 1 else None
    split_manifest = data_manifest(*loaders, legacy_root=legacy_root)
    differences = [k for k in set(config) | set(original)
                   if k not in allowed and config.get(k) != original.get(k)]
    if differences:
        raise ValueError(f"Continuation configuration differs outside cooldown: {sorted(differences)}")
    if (parent["epoch"] != original["total_epochs"] or
            len(parent["history"]) != parent["epoch"] or
            [row["epoch"] for row in parent["history"]] != list(range(1, parent["epoch"] + 1))):
        raise ValueError("Continuation requires a completed parent with contiguous epoch history")
    if (config["total_epochs"] <= original["total_epochs"] or
            config["cooldown_epochs"] <= original["cooldown_epochs"]):
        raise ValueError("Continuation must extend total epochs by increasing cooldown")
    if parent["data_manifest"] != split_manifest:
        raise ValueError("Continuation dataset sample order/split differs from the parent")
    best_path = source.with_name("best.pt")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if (best["config"] != original or best["data_manifest"] != split_manifest or
            best["epoch"] != parent["best_epoch"] or best["best_epoch"] != parent["best_epoch"] or
            best["best_accuracy"] != parent["best_accuracy"] or
            best["history"] != parent["history"][:best["epoch"]]):
        raise ValueError("Parent best checkpoint does not match its last checkpoint")
    lineage = dict(source_last_checkpoint=str(source), source_last_sha256=file_sha256(source),
                   source_best_checkpoint=str(best_path), source_best_sha256=file_sha256(best_path),
                   source_epoch=parent["epoch"], source_total_epochs=original["total_epochs"],
                   target_total_epochs=config["total_epochs"], source_best_epoch=parent["best_epoch"],
                   source_best_accuracy=parent["best_accuracy"], source_config=original,
                   source_provenance=parent["provenance"], source_continuation=parent.get("continuation"),
                   policy="extend_cooldown_preserve_optimizer_scheduler_rng")
    return parent, best, lineage


def validate_continuation_source(source, lineage):
    """Resuming a child must not silently select a different parent."""
    if lineage is None:
        raise ValueError("--continue-from does not match this run's recorded source")
    if (file_sha256(source) != lineage["source_last_sha256"] or
            file_sha256(Path(source).with_name("best.pt")) != lineage["source_best_sha256"]):
        raise ValueError("Continuation source checkpoint content/hash changed")


def make_optimizer_scheduler(model, args):
    # The official timm factory excludes bias and BN affine parameters from decay.
    groups = param_groups_weight_decay(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, eps=1e-8, betas=(.9, .999))
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=args.min_lr,
                                 warmup_t=args.warmup_epochs, warmup_lr_init=args.warmup_lr,
                                 warmup_prefix=False, cycle_limit=1)
    return optimizer, scheduler


def encoded_inputs(images, args):
    if args.dataset == "cifar10":
        return images.unsqueeze(0).expand(args.timesteps, -1, -1, -1, -1).contiguous()
    if images.shape[1] != args.timesteps:
        raise ValueError(f"DVS frame count {images.shape[1]} != T={args.timesteps}")
    return images.transpose(0, 1).contiguous()


def mix_training_batch(images, targets, mixup, erasing, args):
    if args.dataset == "cifar10":
        mean = images.new_tensor((.4914, .4822, .4465)).view(1, 3, 1, 1)
        std = images.new_tensor((.2470, .2435, .2616)).view(1, 3, 1, 1)
        # Match upstream FastCollateMixup's uint8 rounding and ordering: augmentation,
        # pixel-space mixup, rounding, normalization, then random erasing.
        images = ((images * std + mean) * 255.).round().clamp_(0, 255)
        images, targets = mixup(images, targets)
        images = (images.round().div_(255.) - mean) / std
        images = erasing(images)
    else:
        images, targets = mixup(images, targets)
    return images, targets


def run_epoch(model, loader, args, device, optimizer=None, scaler=None, mixup=None, erasing=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, count = 0., 0, 0
    limit = args.max_train_batches if training else args.max_test_batches
    start = time.monotonic()
    for batch_index, (images, targets) in enumerate(loader):
        if limit and batch_index >= limit:
            break
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            images, targets = mix_training_batch(images, targets, mixup, erasing, args)
            optimizer.zero_grad(set_to_none=True)
        model.reset_state()
        with torch.set_grad_enabled(training):
            # Upstream DVS evaluates in FP32 even when AMP is enabled for training.
            with torch.autocast(device_type=device.type, enabled=args.amp and training):
                logits = model(encoded_inputs(images, args)).mean(0)
                loss = F.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at batch {batch_index}")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        n = images.shape[0]
        loss_sum += loss.detach().item() * n
        # During mixup this is agreement with the dominant soft target, not clean accuracy.
        labels = targets.argmax(1) if targets.ndim == 2 else targets
        correct += int((logits.detach().argmax(1) == labels).sum())
        count += n
        model.reset_state()
    if not count:
        raise ValueError("Empty loader; check dataset limits, batch size and drop_last")
    return dict(loss=loss_sum / count, accuracy=100. * correct / count, samples=count,
                seconds=time.monotonic() - start)


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def write_history(path, history):
    if history:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
        atomic_text(path, output.getvalue())


def dataset_indices(dataset):
    """Resolve nested limit/split Subsets to original dataset identities."""
    indices = list(range(len(dataset)))
    while isinstance(dataset, Subset):
        indices = [int(dataset.indices[i]) for i in indices]
        dataset = dataset.dataset
    return dataset, indices


def data_manifest(train_loader, test_loader, legacy_root=None):
    manifest = {}
    for name, loader in (("train", train_loader), ("test", test_loader)):
        dataset = loader.dataset
        origin, indices = dataset_indices(dataset)
        samples = getattr(origin, "samples", None)
        if samples:
            paths = [Path(samples[i][0]).relative_to(origin.root) for i in indices]
            if legacy_root is not None:
                # Older frame manifests included the data root in sample hashes.
                paths = [Path(legacy_root) / Path(origin.root).name / path for path in paths]
            identities = [p.as_posix() for p in paths]
        else:
            identities = [str(i) for i in indices]
        manifest[name] = dict(size=len(dataset), indices=list(indices),
                              sample_order_sha256=hashlib.sha256("\n".join(identities).encode()).hexdigest())
    return manifest


def provenance():
    import timm
    import torchvision
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root / "paramut/torch_spikformer_v1.py",
             root / "paramut/torch_models.py", root / "paramut/torch_data.py",
             root / "paramut/spikformer_augment.py"]
    return dict(source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in paths if p.exists()}, torch=torch.__version__, torchvision=torchvision.__version__,
                timm=timm.__version__, cuda=torch.version.cuda)


def synthetic_smoke(model, args, device, output):
    optimizer, _ = make_optimizer_scheduler(model, args)
    shape = ((args.batch_size, 3, 32, 32) if args.dataset == "cifar10" else
             (args.batch_size, args.timesteps, 2, 128, 128))
    images = torch.randn(shape, device=device) if args.dataset == "cifar10" else torch.poisson(torch.full(shape, .2, device=device))
    targets = torch.arange(args.batch_size, device=device) % 10
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    model.train()
    model.reset_state()
    with torch.autocast(device_type=device.type, enabled=args.amp):
        loss = F.cross_entropy(model(encoded_inputs(images, args)).mean(0), targets)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients) or not torch.isfinite(loss):
        raise FloatingPointError("Preflight loss/gradient check failed")
    if not any(g.abs().max() > 0 for g in gradients):
        raise FloatingPointError("Preflight has no nonzero gradient")
    scaler.step(optimizer)
    scaler.update()
    model.reset_state()
    model.eval()
    with torch.no_grad():
        first = model(encoded_inputs(images, args)).mean(0)
        model.reset_state()
        second = model(encoded_inputs(images, args)).mean(0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if not torch.isfinite(first).all():
        raise FloatingPointError("Preflight evaluation is non-finite")
    model.reset_state()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = dict(status="passed", dataset=args.dataset, method=args.method, batch_size=args.batch_size,
                   timesteps=args.timesteps, parameters=sum(p.numel() for p in model.parameters()),
                   train_loss=float(loss), seconds=time.monotonic() - start,
                   peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None)
    atomic_text(output / "smoke.json", json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config = config_for(args)
    # Freeze the startup source snapshot. Re-reading the working tree at each
    # checkpoint can record later edits that this process never executed.
    run_provenance = provenance()
    if not args.prepare_data_only:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; select --device cpu or use a CUDA-enabled machine")
        if args.amp and device.type != "cuda":
            raise ValueError("--amp requires CUDA")
        model = make_model(args).to(device)
        print(json.dumps(dict(config=config, parameters=sum(p.numel() for p in model.parameters()),
                              provenance=run_provenance)), flush=True)
        if args.synthetic_smoke_only:
            synthetic_smoke(model, args, device, output)
            return
    if not args.data_dir:
        raise ValueError("--data-dir is required outside synthetic smoke mode")
    loaders = build_loaders(args.dataset, args.data_dir, args.batch_size, args.train_limit or None,
                            args.test_limit or None, args.download, num_workers=args.workers,
                            augment=True,
                            dvs_frames=args.timesteps,
                            dvs_resize=128, data_recipe="spikformer",
                            train_drop_last=True)
    train_loader, test_loader, _ = loaders
    split_manifest = data_manifest(train_loader, test_loader)
    if args.prepare_data_only:
        # Touch every cached frame file once. Incomplete SpikingJelly caches must
        # fail here before training starts.
        if args.dataset == "cifar10dvs":
            for dataset in (train_loader.dataset, test_loader.dataset):
                origin, indices = dataset_indices(dataset)
                for index in indices:
                    sample_path = origin.samples[index][0]
                    with np.load(sample_path) as cached:
                        frames = cached["frames"]
                        if frames.shape != (args.timesteps, 2, 128, 128) or not np.isfinite(frames).all():
                            raise ValueError(f"Invalid cached event frames: {sample_path}")
            if not args.train_limit and not args.test_limit:
                if (len(train_loader.dataset), len(test_loader.dataset)) != (9000, 1000):
                    raise ValueError("CIFAR10-DVS cache must contain 9000 train and 1000 test samples")
        atomic_text(output / "data_manifest.json", json.dumps(split_manifest, indent=2) + "\n")
        print("Dataset cache verified: " + json.dumps({k: v["size"] for k, v in split_manifest.items()}), flush=True)
        return
    optimizer, scheduler = make_optimizer_scheduler(model, args)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    history, best_accuracy, best_epoch, start_epoch = [], -1., 0, 0
    last_path, best_path = output / "last.pt", output / "best.pt"
    resume_path = last_path if args.resume == "auto" else (Path(args.resume) if args.resume != "none" else None)
    if resume_path is not None and resume_path.parent.resolve() != output.resolve():
        raise ValueError("Resume into the original output directory, which also contains the selected best.pt")
    if args.evaluate_checkpoint:
        checkpoint = torch.load(args.evaluate_checkpoint, map_location="cpu", weights_only=False)
        for key in ("dataset", "method", "timesteps", "block_size", "embed_dim", "depth", "num_heads"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Evaluation checkpoint mismatch: {key}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        metrics = run_epoch(model, test_loader, args, device)
        atomic_text(output / "evaluation.json", json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
        return
    continuation, checkpoint, inherited_best = None, None, None
    initialize_branch = False
    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        original = checkpoint["config"]
        mismatches = [k for k in set(config) | set(original)
                      if k not in {"data_dir", "recipe_version"} and config.get(k) != original.get(k)]
        if mismatches:
            raise ValueError(f"Resume configuration differs: {mismatches}")
        # A resumed run retains its original metadata, including the best
        # checkpoint. The explicit --data-dir can point to a relocated copy.
        config = original
        legacy_root = original["data_dir"] if original.get("recipe_version", 1) == 1 else None
        split_manifest = data_manifest(train_loader, test_loader, legacy_root)
        continuation = checkpoint.get("continuation")
        if args.continue_from:
            validate_continuation_source(args.continue_from, continuation)
        if not best_path.exists():
            raise FileNotFoundError(f"Resume requires the run's selected best checkpoint: {best_path}")
    elif args.resume not in {"auto", "none"}:
        raise FileNotFoundError(resume_path)
    elif last_path.exists():
        raise FileExistsError("Output contains checkpoints; use --resume auto or a new directory")
    elif args.continue_from:
        checkpoint, inherited_best, continuation = load_continuation(
            args.continue_from, output, config, (train_loader, test_loader))
        marker = output / "continuation.json"
        if marker.exists() and json.loads(marker.read_text()) != continuation:
            raise ValueError("Output directory records a different continuation source")
        if best_path.exists() and not marker.exists():
            raise FileExistsError("Output contains an unrelated best checkpoint; use a new directory")
        # Validate both sets of weights before creating any child checkpoint.
        model.load_state_dict(inherited_best["model_state_dict"], strict=True)
        initialize_branch = True
    elif best_path.exists():
        raise FileExistsError("Output contains checkpoints; use --resume auto or a new directory")
    if checkpoint is not None:
        if not initialize_branch and checkpoint["data_manifest"] != split_manifest:
            raise ValueError("Dataset sample order/split changed since the checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, history = checkpoint["epoch"], checkpoint["history"]
        best_accuracy, best_epoch = checkpoint["best_accuracy"], checkpoint["best_epoch"]
        if initialize_branch:
            atomic_text(output / "continuation.json", json.dumps(continuation, indent=2) + "\n")
            # Only child metadata changes. All inherited weights and source
            # files remain unchanged; the selected best may precede the parent last.
            inherited_best.update(config=config, continuation=continuation, data_manifest=split_manifest)
            atomic_save(best_path, inherited_best)
            atomic_save(last_path, dict(checkpoint, config=config, continuation=continuation, data_manifest=split_manifest))
            write_history(output / "history.csv", history)
        restore_rng(checkpoint["rng"])
        print(f"Resuming at epoch {start_epoch + 1}; target={config['total_epochs']} "
              f"lr={optimizer.param_groups[0]['lr']:.8g}", flush=True)
    atomic_text(output / "config.json", json.dumps(dict(config=config, provenance=run_provenance,
                                                        continuation=continuation), indent=2) + "\n")
    atomic_text(output / "data_manifest.json", json.dumps(split_manifest, indent=2) + "\n")
    mixup = Mixup(mixup_alpha=.5, cutmix_alpha=0., prob=config["mixup_probability"],
                  mode="batch", label_smoothing=.1, num_classes=10)
    erasing = RandomErasing(probability=config["random_erasing"], mode="const", max_count=1, device=str(device))
    for epoch in range(start_epoch, args.epochs + args.cooldown_epochs):
        mixup.mixup_enabled = epoch < args.mixup_off_epoch
        current_lr = optimizer.param_groups[0]["lr"]
        train = run_epoch(model, train_loader, args, device, optimizer, scaler, mixup, erasing)
        scheduler.step(epoch + 1)
        test = run_epoch(model, test_loader, args, device)
        improved = test["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = test["accuracy"], epoch + 1
        row = dict(epoch=epoch + 1, lr=current_lr, train_loss=train["loss"],
                   train_soft_target_agreement=train["accuracy"], test_loss=test["loss"],
                   test_accuracy=test["accuracy"], best_accuracy=best_accuracy, best_epoch=best_epoch,
                   train_seconds=train["seconds"], test_seconds=test["seconds"],
                   train_samples=train["samples"], test_samples=test["samples"])
        history.append(row)
        payload = dict(config=config, provenance=run_provenance, continuation=continuation, epoch=epoch + 1,
                       model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                       scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=capture_rng(),
                       history=history, best_accuracy=best_accuracy, best_epoch=best_epoch,
                       data_manifest=split_manifest)
        if improved:
            atomic_save(best_path, payload)
        atomic_save(last_path, payload)
        write_history(output / "history.csv", history)
        print(json.dumps(row), flush=True)
    # A complete job independently reloads the selected best weights for a final
    # full test pass; last-epoch performance remains in the history and summary.
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    final = run_epoch(model, test_loader, args, device)
    summary = dict(dataset=args.dataset, method=args.method, status="completed", seed=args.seed,
                   timesteps=args.timesteps, epochs=args.epochs + args.cooldown_epochs,
                   best_epoch=best_epoch, best_test_accuracy=best_accuracy,
                   reloaded_best_test_accuracy=final["accuracy"], last_test_accuracy=history[-1]["test_accuracy"],
                   test_samples=final["samples"], best_checkpoint=str(best_path), last_checkpoint=str(last_path),
                   selection="test-selected best; single seed", config=config, continuation=continuation)
    atomic_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
