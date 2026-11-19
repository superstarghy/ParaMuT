"""CIFAR and event-frame vision dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetInfo:
    in_channels: int
    image_size: int
    num_classes: int
    input_mean: tuple[float, ...] | None = None
    input_std: tuple[float, ...] | None = None


class NormalizedDVSFrames:
    """Normalize an event sequence by its maximum count, then resize each frame."""

    def __init__(self, resize, augment=False):
        self.resize = resize
        self.augment = augment
        self.random_crop_padding = 8 if resize >= 96 else 4
        self.erase_prob = .1

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment or self.random_crop_padding <= 0:
            return x
        pad = self.random_crop_padding
        height, width = x.shape[-2:]
        padded = F.pad(x, (pad, pad, pad, pad))
        top = torch.randint(0, 2 * pad + 1, (1,)).item()
        left = torch.randint(0, 2 * pad + 1, (1,)).item()
        return padded[..., top:top + height, left:left + width]

    def _random_erase(self, x: torch.Tensor) -> torch.Tensor:
        if not self.augment or self.erase_prob <= 0 or torch.rand(1).item() >= self.erase_prob:
            return x
        height, width = x.shape[-2:]
        area = height * width
        erase_area = float(torch.empty(1).uniform_(0.01, 0.08).item()) * area
        aspect = float(torch.empty(1).uniform_(0.5, 2.0).item())
        erase_h = max(1, min(height, int(round((erase_area * aspect) ** 0.5))))
        erase_w = max(1, min(width, int(round((erase_area / aspect) ** 0.5))))
        top = torch.randint(0, height - erase_h + 1, (1,)).item()
        left = torch.randint(0, width - erase_w + 1, (1,)).item()
        out = x.clone()
        out[..., top:top + erase_h, left:left + erase_w] = 0.0
        return out

    def _augment_tensor(self, x: torch.Tensor) -> torch.Tensor:
        x = self._random_crop(x)
        x = self._random_erase(x)
        return x

    def __call__(self, frames):
        x = torch.as_tensor(frames, dtype=torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected DVS frames with shape [F, 2, H, W], got {tuple(x.shape)}")
        x = x / x.amax().clamp_min(1.0)
        if x.shape[-1] != self.resize:
            x = F.interpolate(x, size=(self.resize, self.resize), mode="bilinear", align_corners=False)
        return self._augment_tensor(x).clamp(0.0, 1.0)


class SpikformerDVSFrames:
    """Original Spikformer raw event counts, with optional sequence augmentation.

    This transform preserves counts above one and full sensor resolution. The same transform is applied to all frames.
    """

    def __init__(self, augment: bool = False):
        from .spikformer_augment import SpikformerSNNAugmentWide

        self.augment = bool(augment)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.policy = SpikformerSNNAugmentWide()

    def __call__(self, frames):
        x = torch.as_tensor(frames, dtype=torch.float32)
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected DVS frames [T, 2, H, W], got {tuple(x.shape)}")
        return self.policy(self.flip(x)) if self.augment else x


def spikformer_cifar_transforms(augment: bool):
    """CIFAR10 transforms from the released Spikformer recipe.

    Its random-resized crop uses scale=ratio=(1, 1), so there is no padded
    CIFAR crop. Standard deviations and the restricted RandAugment policy
    differ from this repository's historical CIFAR defaults.
    """
    from .spikformer_augment import SpikformerCIFARRandAugment
    from timm.data.transforms import RandomResizedCropAndInterpolation

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_steps = []
    if augment:
        train_steps.extend([
            RandomResizedCropAndInterpolation(32, scale=(1.0, 1.0), ratio=(1.0, 1.0),
                                             interpolation="bicubic"),
            transforms.RandomHorizontalFlip(),
            SpikformerCIFARRandAugment(mean),
        ])
    train_steps.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return transforms.Compose(train_steps), test_transform, mean, std


def prepare_dvs128gesture_root(data_dir: str) -> None:
    """Make a manually extracted DVS128 Gesture root usable by SpikingJelly.

    SpikingJelly checks resource files under ``root/download`` before it
    generates ``events_np`` and frame caches. Some manual installs place these
    files at ``root`` next to the extracted ``DvsGesture`` directory. Link them
    into ``download`` so the first preprocessing run can proceed without
    copying the archive.
    """

    root = Path(data_dir)
    events_np_root = root / "events_np"
    if events_np_root.exists():
        return
    download_root = root / "download"
    download_root.mkdir(parents=True, exist_ok=True)
    for file_name in ("DvsGesture.tar.gz", "gesture_mapping.csv", "LICENSE.txt", "README.txt"):
        expected = download_root / file_name
        if expected.exists():
            continue
        source = root / file_name
        if not source.exists():
            continue
        try:
            expected.symlink_to(Path("..") / file_name)
        except FileExistsError:
            pass
        except OSError:
            # Filesystems that disallow symlinks can still proceed with a small
            # manual fix; leave SpikingJelly's original integrity error intact.
            pass


def cifar_stats(name: str):
    if name == "cifar100":
        return (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


def cifar_transforms(name, augment):
    mean, std = cifar_stats(name)
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test = transforms.Compose(normalize)
    if not augment:
        return test, test, mean, std
    train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        *normalize,
        transforms.RandomErasing(p=.5, scale=(.02, .2), ratio=(.3, 3.3), value="random"),
    ])
    return train, test, mean, std


def stratified_split_indices(
    targets,
    train_ratio: float,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Create a reproducible class-balanced split without loading samples."""

    ratio = float(train_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"dvs_train_ratio must be in (0, 1), got {train_ratio}")
    labels = torch.as_tensor(targets, dtype=torch.long)
    generator = torch.Generator().manual_seed(int(seed))
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in labels.unique(sorted=True).tolist():
        class_indices = torch.nonzero(labels == int(label), as_tuple=False).flatten()
        order = torch.randperm(class_indices.numel(), generator=generator)
        class_indices = class_indices[order]
        split = int(round(class_indices.numel() * ratio))
        split = min(max(split, 1), max(class_indices.numel() - 1, 1))
        train_indices.extend(class_indices[:split].tolist())
        test_indices.extend(class_indices[split:].tolist())
    return train_indices, test_indices


def ordered_class_split_indices(targets, train_ratio: float = 0.9) -> tuple[list[int], list[int]]:
    """Match Spikformer's per-class split with ``random_split=False``.

    DatasetFolder's sample ordering is retained, and the first ceil(0.9*N)
    samples in each class form the train set. No frame loading is necessary.
    """
    ratio = float(train_ratio)
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"dvs_train_ratio must be in (0, 1), got {train_ratio}")
    labels = torch.as_tensor(targets, dtype=torch.long)
    train_indices, test_indices = [], []
    for label in labels.unique(sorted=True).tolist():
        indices = torch.nonzero(labels == int(label), as_tuple=False).flatten().tolist()
        split = math.ceil(len(indices) * ratio)
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    return train_indices, test_indices


def build_loaders(name: str, data_dir: str, batch_size: int, train_limit: int | None,
                  test_limit: int | None, download: bool, num_workers: int = 2, augment: bool = False,
                  dvs_frames: int = 10, dvs_resize: int = 128, data_recipe: str = "vgg",
                  train_drop_last: bool = False):
    if data_recipe not in {"vgg", "spikformer"}:
        raise ValueError(f"Unsupported vision recipe {data_recipe}")
    spikformer = data_recipe == "spikformer"
    if spikformer and name not in {"cifar10", "cifar10dvs"}:
        raise ValueError("The Spikformer recipe supports CIFAR10 and CIFAR10-DVS")
    if name in {"cifar10", "cifar100"}:
        if spikformer:
            train_transform, test_transform, mean, std = spikformer_cifar_transforms(augment)
        else:
            train_transform, test_transform, mean, std = cifar_transforms(name, augment)
        dataset = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        train = dataset(data_dir, train=True, transform=train_transform, download=download)
        test = dataset(data_dir, train=False, transform=test_transform, download=download)
        info = DatasetInfo(3, 32, 10 if name == "cifar10" else 100, mean, std)
    elif name in {"cifar10dvs", "dvs128gesture"}:
        from spikingjelly.datasets import load_npz_frames

        root = Path(data_dir)
        cache = root / f"frames_number_{dvs_frames}_split_by_number"
        frame_options = dict(data_type="frame", frames_number=dvs_frames, split_by="number")
        if spikformer:
            train_transform, test_transform = SpikformerDVSFrames(augment), SpikformerDVSFrames()
        else:
            train_transform = NormalizedDVSFrames(dvs_resize, augment)
            test_transform = NormalizedDVSFrames(dvs_resize)
        if name == "dvs128gesture":
            if download:
                raise ValueError("Download DVS128 Gesture manually and pass --data-dir")
            if (cache / "train").is_dir() and (cache / "test").is_dir():
                train = datasets.DatasetFolder(cache / "train", load_npz_frames, (".npz",), train_transform)
                test = datasets.DatasetFolder(cache / "test", load_npz_frames, (".npz",), test_transform)
            else:
                from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

                prepare_dvs128gesture_root(data_dir)
                train = DVS128Gesture(data_dir, train=True, transform=train_transform, **frame_options)
                test = DVS128Gesture(data_dir, train=False, transform=test_transform, **frame_options)
        else:
            if cache.is_dir():
                train_origin = datasets.DatasetFolder(cache, load_npz_frames, (".npz",), train_transform)
                test_origin = datasets.DatasetFolder(cache, load_npz_frames, (".npz",), test_transform)
            else:
                from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

                if not download and not (root / "events_np").is_dir():
                    raise FileNotFoundError("CIFAR10-DVS is absent. Supply its dataset root or pass --download")
                root.mkdir(parents=True, exist_ok=True)
                train_origin = CIFAR10DVS(data_dir, transform=train_transform, **frame_options)
                test_origin = CIFAR10DVS(data_dir, transform=test_transform, **frame_options)
            indices = (ordered_class_split_indices(train_origin.targets) if spikformer else
                       stratified_split_indices(train_origin.targets, .9, seed=0))
            train, test = Subset(train_origin, indices[0]), Subset(test_origin, indices[1])
        info = DatasetInfo(2, dvs_resize, 11 if name == "dvs128gesture" else 10)
    else:
        raise ValueError(f"Unsupported dataset {name}")

    if train_limit is not None and train_limit > 0:
        train = Subset(train, range(min(train_limit, len(train))))
    if test_limit is not None and test_limit > 0:
        test = Subset(test, range(min(test_limit, len(test))))

    # Recreate workers at every epoch so their augmentation RNGs derive from
    # the checkpointed main-process RNG. Persistent worker RNG states cannot
    # be restored by an epoch-boundary resume. Leave generator=None as well.
    loader_options = dict(num_workers=num_workers, pin_memory=data_recipe == "spikformer",
                          persistent_workers=False)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=train_drop_last, **loader_options)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, **loader_options)
    return train_loader, test_loader, info
