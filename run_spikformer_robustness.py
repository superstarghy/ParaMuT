#!/usr/bin/env python3
"""Full-test, raw-input FGSM/PGD evaluation of official Spikformer v1 best checkpoints.

Native and continuous-state surrogate attacks share the exact hard predictor.
Reports include each gradient mode and the per-sample union of their successes.
DVS budgets are fractional raw event-count units, not normalized-image epsilon.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
from importlib import metadata
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from run_spikformer_reproduction import (
    atomic_save, atomic_text, data_manifest, dataset_indices, encoded_inputs,
    file_sha256, make_model,
)
from paramut.torch_data import build_loaders
from paramut.spikformer_robustness import (
    RawInputClassifier, fgsm_attack, pgd_attack, predict_logits, input_gradient,
    select_adversarial, project_linf,
)

MODES = ("native", "full_state")
ROOT = Path(__file__).resolve().parent


class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, self.indices[index]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epsilons", default="")
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--pgd-restarts", type=int, default=2)
    parser.add_argument("--pgd-alpha-ratio", type=float, default=.25)
    parser.add_argument("--attack-seed", type=int, default=12345)
    parser.add_argument("--resume", choices=("auto", "none"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 0 or args.workers < 0 or args.pgd_steps < 1 or args.pgd_restarts < 1:
        parser.error("batch/workers must be nonnegative and PGD steps/restarts positive")
    if not math.isfinite(args.pgd_alpha_ratio) or args.pgd_alpha_ratio <= 0:
        parser.error("PGD alpha ratio must be finite and positive")
    return args


def budget_grid(dataset, requested):
    values = ([float(x.strip()) for x in requested.split(",")] if requested else
              [x / 255 for x in (0, .25, .5, 1, 2, 4, 8)] if dataset == "cifar10" else
              [0, .001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.])
    if (not values or values[0] != 0 or values != sorted(set(values)) or
            any(not math.isfinite(x) or x < 0 for x in values) or len(values) < 2):
        raise ValueError("epsilons must start at 0 and be strictly increasing finite nonnegative budgets")
    return values


def source_snapshot():
    names = ["run_spikformer_robustness.py", "run_spikformer_reproduction.py",
             "paramut/spikformer_robustness.py", "paramut/torch_spikformer_v1.py",
             "paramut/torch_models.py", "paramut/torch_data.py"]
    return {name: file_sha256(ROOT / name) for name in names}


def runtime_environment(device):
    return dict(torch=torch.__version__, torchvision=metadata.version("torchvision"),
                spikingjelly=metadata.version("spikingjelly"), numpy=np.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                gpu=torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def check_bounds(clean, adversarial, epsilon, upper):
    if not torch.isfinite(adversarial).all():
        raise FloatingPointError("Nonfinite adversarial input")
    tolerance = 2e-6 + 2 * torch.finfo(clean.dtype).eps * max(1., float(clean.abs().max()))
    maximum = float((adversarial - clean).abs().max())
    if maximum > epsilon + tolerance or float(adversarial.min()) < 0:
        raise ValueError(f"Attack left its threat set: max delta={maximum}, epsilon={epsilon}")
    if upper is not None and float(adversarial.max()) > upper:
        raise ValueError("Attack exceeded the raw image upper bound")
    if epsilon == 0 and not torch.equal(clean, adversarial):
        raise ValueError("Zero-budget attack changed the input")
    return maximum


def load_evaluation(args):
    path = Path(args.checkpoint).resolve()
    if path.name != "best.pt":
        raise ValueError("This workflow evaluates selected best.pt checkpoints only")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved["config"]
    if config.get("model") != "spikformer_v1":
        raise ValueError("Historical compact/expanded checkpoints require their own evaluation recipe")
    summary = json.loads(path.with_name("summary.json").read_text())
    if (summary["status"] != "completed" or summary["config"] != config or
            saved["epoch"] != summary["best_epoch"] or saved["best_accuracy"] != summary["best_test_accuracy"]):
        raise ValueError("Checkpoint and completed training summary disagree")
    dataset = config["dataset"]
    batch = args.batch_size or (128 if dataset == "cifar10" else 16)
    train, test, _ = build_loaders(
        dataset, args.data_dir or config["data_dir"], batch, config["train_limit"], config["test_limit"], False,
        num_workers=args.workers, augment=False,
        dvs_frames=config["timesteps"],
        dvs_resize=128, data_recipe="spikformer", train_drop_last=True)
    legacy_root = config["data_dir"] if config.get("recipe_version", 1) == 1 else None
    split = data_manifest(train, test, legacy_root)
    if split != saved["data_manifest"]:
        raise ValueError("Evaluation sample identities/order differ from the checkpoint")
    expected = 10000 if dataset == "cifar10" else 1000
    if config["test_limit"]:
        expected = min(expected, config["test_limit"])
    if len(test.dataset) != expected:
        raise ValueError("The test split size does not match the training recipe")
    count = expected
    if config["max_test_batches"]:
        count = min(expected, config["max_test_batches"] * config["batch_size"])
    if summary["test_samples"] != count:
        raise ValueError("The recorded test sample count does not match the checkpoint")
    # Preserve exact dataset/split, replacing only deterministic CIFAR
    # normalization with ToTensor. Normalization moves inside the predictor.
    if dataset == "cifar10":
        origin, _ = dataset_indices(test.dataset)
        origin.transform = transforms.ToTensor()
    test_dataset = test.dataset
    if count < expected:
        test_dataset = Subset(test_dataset, range(count))
        split = data_manifest(train, DataLoader(test_dataset), legacy_root)
    indexed = IndexedDataset(test_dataset, split["test"]["indices"])
    model = make_model(SimpleNamespace(**config))
    model.load_state_dict(saved["model_state_dict"], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise FloatingPointError("Nonfinite checkpoint weights")
    model.eval().requires_grad_(False)
    wrapped = RawInputClassifier(model, dataset, config["timesteps"]).to(args.device).eval()
    return path, saved, summary, config, split, indexed, wrapped, batch


def preflight(args, model, loader, config, epsilon, upper, output):
    images, labels, indices = next(iter(loader))
    images, labels = images.to(args.device), labels.to(args.device)
    before = {n: t.clone() for n, t in model.named_buffers() if "running_" in n or "num_batches" in n}
    clean = predict_logits(model, images)
    # Compare with the original CPU normalization/test-forward path.
    native = images.detach().cpu()
    if config["dataset"] == "cifar10":
        native = transforms.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616))(native)
    base = model.model
    base.reset_state()
    with torch.no_grad():
        reference = base(encoded_inputs(native.to(args.device), SimpleNamespace(**config))).mean(0)
    base.reset_state()
    torch.testing.assert_close(clean, reference, rtol=0, atol=0)
    diagnostics = {}
    for mode in MODES:
        grad, logits = input_gradient(model, images, labels, gradient_mode=mode)
        torch.testing.assert_close(clean, logits, rtol=0, atol=0)
        if not grad.abs().max() > 0:
            raise ValueError(f"{mode}: every input gradient is zero")
        fgsm = fgsm_attack(model, images, labels, epsilon, upper_bound=upper, gradient_mode=mode)
        pgd = pgd_attack(model, images, labels, epsilon, steps=2, restarts=1,
                         alpha_ratio=args.pgd_alpha_ratio, upper_bound=upper,
                         attack_seed=args.attack_seed, sample_indices=indices,
                         fgsm_candidate=fgsm, gradient_mode=mode)
        check_bounds(images, fgsm, epsilon, upper)
        check_bounds(images, pgd, epsilon, upper)
        torch.testing.assert_close(predict_logits(model, images), clean, rtol=0, atol=0)
        diagnostics[mode] = dict(gradient_max=float(grad.abs().max()),
                                 zero_gradient_samples=int((grad.flatten(1).abs().amax(1) == 0).sum()))
    buffers = dict(model.named_buffers())
    for name, value in before.items():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    result = dict(status="passed", checkpoint=str(Path(args.checkpoint).resolve()), dataset=config["dataset"],
                  method=config["method"], block_size=config["effective_neuron_block_size"],
                  batch_size=len(labels), epsilon=epsilon, raw_min=float(images.min()), raw_max=float(images.max()),
                  exact_original_forward=True, exact_gradient_mode_forward=True, batchnorm_unchanged=True,
                  gradients=diagnostics, source_sha256=source_snapshot(),
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if str(args.device).startswith("cuda") else None)
    atomic_text(output / "smoke.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def ensure_clean_reference(args, model, dataset, batch, request, summary, output):
    """Verify all clean predictions before spending time on adversarial sweeps."""
    path = output / "clean_reference.pt"
    if path.exists():
        reference = torch.load(path, map_location="cpu", weights_only=False)
        if reference["request"] != request:
            raise ValueError("Existing clean reference has a different checkpoint/attack request")
    else:
        started = time.monotonic()
        predictions, labels, indices, batch_input_sha256 = [], [], [], []
        raw_min, raw_max = math.inf, -math.inf
        input_digest = hashlib.sha256()
        loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=args.workers)
        for images, targets, ids in loader:
            raw_bytes = images.contiguous().numpy().tobytes()
            input_digest.update(raw_bytes)
            batch_input_sha256.append(hashlib.sha256(raw_bytes).hexdigest())
            raw_min, raw_max = min(raw_min, float(images.min())), max(raw_max, float(images.max()))
            predictions.append(predict_logits(model, images.to(args.device)).argmax(1).cpu())
            labels.append(targets); indices.append(ids)
        reference = dict(request=request, predictions=torch.cat(predictions), labels=torch.cat(labels),
                         sample_indices=torch.cat(indices), seconds=time.monotonic() - started,
                         raw_input_min=raw_min, raw_input_max=raw_max,
                         raw_inputs_sha256=input_digest.hexdigest(),
                         batch_input_sha256=batch_input_sha256,
                         labels_sha256=hashlib.sha256(torch.cat(labels).numpy().tobytes()).hexdigest())
    accuracy = 100. * float(reference["predictions"].eq(reference["labels"]).double().mean())
    if (len(reference["labels"]) != len(dataset) or len(dataset) != summary["test_samples"] or
            reference["sample_indices"].tolist() != request["test_manifest"]["indices"] or
            not math.isclose(accuracy, summary["reloaded_best_test_accuracy"], rel_tol=0, abs_tol=1e-9)):
        raise ValueError(f"Full-test clean precheck failed: accuracy={accuracy}, expected={summary['reloaded_best_test_accuracy']}")
    if not path.exists():
        atomic_save(path, reference)
    print(json.dumps(dict(status="clean_verified", samples=len(dataset), accuracy=accuracy,
                          raw_input_min=reference["raw_input_min"], raw_input_max=reference["raw_input_max"])), flush=True)
    return reference


def write_reports(output, progress, request, summary, config, checkpoint):
    labels = torch.cat(progress["labels"])
    clean = torch.cat(progress["clean_predictions"])
    clean_ok = clean.eq(labels)
    clean_accuracy = 100. * float(clean_ok.double().mean())
    expected = summary["reloaded_best_test_accuracy"]
    if not math.isclose(clean_accuracy, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Full-test clean score {clean_accuracy} != trainer's best reload {expected}")
    rows, arrays = [], dict(sample_indices=torch.cat(progress["sample_indices"]).numpy(),
                            labels=labels.numpy(), clean_predictions=clean.numpy())
    if len(labels) != summary["test_samples"] or arrays["sample_indices"].tolist() != request["test_manifest"]["indices"]:
        raise ValueError("Final attack report has missing/reordered test samples")
    for ei, epsilon in enumerate(request["epsilons"]):
        for attack in ("fgsm", "pgd"):
            for mode in (*MODES, "union"):
                key = f"{attack}_{mode}_eps{ei}"
                predicted = torch.cat(progress["predictions"][key])
                if predicted.shape != labels.shape:
                    raise ValueError("Incomplete attack predictions")
                arrays[key] = predicted.numpy()
                adversarial_ok = predicted.eq(labels)
                robust_ok = clean_ok & adversarial_ok
                rows.append(dict(dataset=config["dataset"], method=config["method"],
                                 effective_neuron_block_size=config["effective_neuron_block_size"],
                                 training_seed=config["seed"], best_epoch=checkpoint["epoch"], attack=attack,
                                 gradient_mode=mode, epsilon=epsilon, epsilon_units=request["epsilon_units"],
                                 samples=len(labels), clean_accuracy=clean_accuracy,
                                 adversarial_accuracy=100. * float(adversarial_ok.double().mean()),
                                 robust_accuracy=100. * float(robust_ok.double().mean()),
                                 successful_attacks=int((clean_ok & ~adversarial_ok).sum()),
                                 max_observed_linf=progress["max_linf"][key],
                                 attack_seed=request["attack_seed"], pgd_steps=request["pgd_steps"] if attack == "pgd" else 1,
                                 pgd_restarts_per_gradient=request["pgd_restarts"] if attack == "pgd" else 0))
        fgsm_union = arrays[f"fgsm_union_eps{ei}"] == arrays["labels"]
        pgd_union = arrays[f"pgd_union_eps{ei}"] == arrays["labels"]
        if np.any(pgd_union & ~fgsm_union):
            raise ValueError("PGD union lost a successful FGSM candidate")
    aucs = {}
    for attack in ("fgsm", "pgd"):
        for mode in (*MODES, "union"):
            selected = [r for r in rows if r["attack"] == attack and r["gradient_mode"] == mode]
            x, y = [r["epsilon"] for r in selected], [r["robust_accuracy"] for r in selected]
            aucs[f"{attack}_{mode}"] = sum((b-a)*(u+v)/2 for a,b,u,v in zip(x,x[1:],y,y[1:]))/(x[-1]-x[0])
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_text(output / "robustness.csv", text.getvalue())
    with (output / "predictions.npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    (output / "predictions.npz.tmp").replace(output / "predictions.npz")
    result = dict(status="completed", request=request, checkpoint_config=config,
                  checkpoint_provenance=checkpoint.get("provenance"), continuation=checkpoint.get("continuation"),
                  samples=len(labels), best_epoch=checkpoint["epoch"], clean_accuracy=clean_accuracy,
                  training_best_reload_accuracy=expected, clean_matches_training=True,
                  gradient_modes=list(MODES), attack_selection="per-sample success-first union over clean, FGSM, every PGD iterate/restart/mode and smaller-budget PGD candidates",
                  normalized_auc_percent=aucs, auc_interval=[request["epsilons"][0],request["epsilons"][-1]],
                  clean_input_gradient_diagnostics=progress["gradient_diagnostics"],
                  seconds=progress["seconds"], rows=len(rows), source_sha256=request["source_sha256"],
                  artifacts={name:file_sha256(output/name) for name in ("robustness.csv", "predictions.npz", "clean_reference.pt")})
    clean_reference = torch.load(output / "clean_reference.pt", map_location="cpu", weights_only=False)
    result["test_input_audit"] = {key:clean_reference[key] for key in
                                ("raw_inputs_sha256", "labels_sha256", "raw_input_min", "raw_input_max")}
    atomic_text(output / "summary.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ("status", "samples", "clean_accuracy", "normalized_auc_percent", "seconds")}), flush=True)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".python_eval.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.manual_seed(args.attack_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        path, checkpoint, summary, config, split, dataset, model, batch = load_evaluation(args)
        epsilons = budget_grid(config["dataset"], args.epsilons)
        upper = 1. if config["dataset"] == "cifar10" else None
        request = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path),
                       dataset=config["dataset"], batch_size=batch, test_manifest=split["test"],
                       epsilons=epsilons, epsilon_units="raw_pixel_0_1" if upper else "raw_event_count",
                       input_lower_bound=0., input_upper_bound=upper, fractional_perturbations=True,
                       pgd_steps=args.pgd_steps, pgd_restarts=args.pgd_restarts,
                       pgd_alpha_ratio=args.pgd_alpha_ratio, attack_seed=args.attack_seed,
                       gradient_modes=list(MODES), pgd_includes_lower_budget_candidates=True,
                       runtime_environment=runtime_environment(args.device), source_sha256=source_snapshot(), version=1)
        if args.preflight_only:
            loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=args.workers)
            preflight(args, model, loader, config, epsilons[1], upper, output)
            return
        summary_path = output / "summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            if previous.get("status") == "completed" and previous["request"] == request:
                for name, sha in previous["artifacts"].items():
                    if file_sha256(output / name) != sha:
                        raise ValueError(f"Completed evaluation artifact changed: {name}")
                print("Evaluation already completed and hashes verified", flush=True)
                return
            raise ValueError("Existing evaluation summary has a different request")
        progress_path = output / "progress.pt"
        if progress_path.exists():
            if args.resume != "auto":
                raise FileExistsError("Existing progress requires --resume auto or a new output directory")
            progress = torch.load(progress_path, map_location="cpu", weights_only=False)
            if progress["request"] != request:
                raise ValueError("Resume attack/checkpoint/source configuration differs")
        else:
            progress = dict(request=request, count=0, seconds=0., labels=[], sample_indices=[],
                            clean_predictions=[], predictions={}, max_linf={},
                            gradient_diagnostics={mode:dict(samples=0, zero_samples=0, max_abs=0.) for mode in MODES})
        if progress["count"] != sum(len(x) for x in progress["labels"]) or not 0 <= progress["count"] <= len(dataset):
            raise ValueError("Invalid saved progress sample count")
        atomic_text(output / "config.json", json.dumps(request, indent=2) + "\n")
        clean_reference = ensure_clean_reference(args, model, dataset, batch, request, summary, output)
        remaining = Subset(dataset, range(progress["count"], len(dataset)))
        loader = DataLoader(remaining, batch_size=batch, shuffle=False, num_workers=args.workers,
                            pin_memory=str(args.device).startswith("cuda"))
        print(json.dumps(dict(status="running", checkpoint=str(path), resume_samples=progress["count"],
                              total_samples=len(dataset), epsilons=epsilons, batch_size=batch)), flush=True)
        for images, labels, indices in loader:
            started = time.monotonic()
            batch_index = progress["count"] // batch
            if (progress["count"] % batch or
                    hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest() !=
                    clean_reference["batch_input_sha256"][batch_index]):
                raise ValueError("Attack input bytes differ from the verified clean reference")
            images, labels = images.to(args.device), labels.to(args.device)
            clean_logits = predict_logits(model, images)
            clean_pred = clean_logits.argmax(1)
            segment = slice(progress["count"], progress["count"] + len(labels))
            if (not torch.equal(clean_pred.cpu(), clean_reference["predictions"][segment]) or
                    not torch.equal(labels.cpu(), clean_reference["labels"][segment]) or
                    not torch.equal(indices, clean_reference["sample_indices"][segment])):
                raise ValueError("Attack batch differs from the verified clean reference")
            batch_predictions, batch_norms = {}, {}
            clean_gradients, batch_gradients = {}, {}
            for mode in MODES:
                gradient, gradient_logits = input_gradient(model, images, labels, gradient_mode=mode)
                torch.testing.assert_close(clean_logits, gradient_logits, rtol=0, atol=0)
                norms = gradient.flatten(1).abs().amax(1)
                clean_gradients[mode] = gradient
                batch_gradients[mode] = dict(samples=len(labels), zero_samples=int((norms == 0).sum()),
                                            max_abs=float(norms.max()))
            previous_pgd = {mode:images for mode in MODES}
            for ei, epsilon in enumerate(epsilons):
                candidates = {}
                if epsilon == 0:
                    candidates = {f"{attack}_{mode}":images for attack in ("fgsm","pgd") for mode in (*MODES,"union")}
                else:
                    for mode in MODES:
                        candidates[f"fgsm_{mode}"] = project_linf(
                            images + epsilon * clean_gradients[mode].sign(), images, epsilon, upper).detach()
                        candidates[f"pgd_{mode}"] = pgd_attack(
                            model, images, labels, epsilon, steps=args.pgd_steps, restarts=args.pgd_restarts,
                            alpha_ratio=args.pgd_alpha_ratio, upper_bound=upper, attack_seed=args.attack_seed,
                            sample_indices=indices, fgsm_candidate=candidates[f"fgsm_{mode}"], gradient_mode=mode)
                        candidates[f"pgd_{mode}"] = select_adversarial(
                            model, images, labels, [candidates[f"pgd_{mode}"], previous_pgd[mode]])
                        previous_pgd[mode] = candidates[f"pgd_{mode}"]
                    candidates["fgsm_union"] = select_adversarial(
                        model, images, labels, [candidates[f"fgsm_{m}"] for m in MODES])
                    candidates["pgd_union"] = select_adversarial(
                        model, images, labels, [candidates[f"{a}_{m}"] for a in ("fgsm","pgd") for m in MODES])
                for key, adversarial in candidates.items():
                    name = f"{key}_eps{ei}"
                    batch_norms[name] = check_bounds(images, adversarial, epsilon, upper)
                    batch_predictions[name] = (clean_pred if epsilon == 0 else predict_logits(model, adversarial).argmax(1)).cpu()
            progress["labels"].append(labels.cpu())
            progress["sample_indices"].append(indices)
            progress["clean_predictions"].append(clean_pred.cpu())
            for mode, diagnostics in batch_gradients.items():
                total = progress["gradient_diagnostics"][mode]
                total["samples"] += diagnostics["samples"]
                total["zero_samples"] += diagnostics["zero_samples"]
                total["max_abs"] = max(total["max_abs"], diagnostics["max_abs"])
            for key, value in batch_predictions.items():
                progress["predictions"].setdefault(key, []).append(value)
                progress["max_linf"][key] = max(progress["max_linf"].get(key, 0.), batch_norms[key])
            progress["count"] += len(labels)
            progress["seconds"] += time.monotonic() - started
            atomic_save(progress_path, progress)
            print(json.dumps(dict(samples=progress["count"], total=len(dataset), seconds=progress["seconds"])), flush=True)
        if progress["count"] != len(dataset):
            raise ValueError("Incomplete evaluation")
        write_reports(output, progress, request, summary, config, checkpoint)


if __name__ == "__main__":
    main()
