"""Corruptions on normalized DVS frame sequences."""

from __future__ import annotations

import torch


EVENT_CORRUPTION_MODES = (
    "event_drop",
    "event_insert",
    "polarity_flip",
    "frame_drop",
    "temporal_jitter",
)


def _event_random(shape, x: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    return torch.rand(shape, dtype=x.dtype, device=x.device, generator=generator)


def corrupt_event_frames(
    x: torch.Tensor,
    strength: float,
    mode: str,
    generator: torch.Generator | None = None,
    temporal_jitter_steps: int = 1,
) -> torch.Tensor:
    """Apply a DVS-style corruption to ``[B, T, 2, H, W]`` frame sequences.

    ``strength`` is a probability for drop/flip/jitter modes. For insertion it
    is the requested number of inserted non-zero bins relative to the original
    active-bin count. The function operates after frame normalization, so an
    inserted bin uses the sample's mean active-bin magnitude rather than a
    saturated value of one.
    """

    if x.ndim != 5 or int(x.shape[2]) != 2:
        raise ValueError(
            "Event corruptions require input shape [B, T, 2, H, W], "
            f"got {tuple(x.shape)}"
        )
    mode = str(mode).lower()
    if mode not in EVENT_CORRUPTION_MODES:
        raise ValueError(f"Unknown event corruption mode: {mode}")
    probability = min(max(float(strength), 0.0), 1.0)
    clean = x.detach()
    if probability <= 0:
        return clean

    if mode == "event_drop":
        keep = _event_random(clean.shape, clean, generator) >= probability
        return (clean * keep).detach()

    if mode == "event_insert":
        flat = clean.flatten(1)
        active = flat > 0
        active_count = active.sum(dim=1, keepdim=True)
        inactive_count = (~active).sum(dim=1, keepdim=True).clamp_min(1)
        insert_probability = (probability * active_count / inactive_count).clamp(0.0, 1.0)
        selected = (~active) & (
            _event_random(flat.shape, flat, generator) < insert_probability
        )
        active_mean = (
            flat.sum(dim=1, keepdim=True) / active_count.clamp_min(1)
        ).clamp(0.0, 1.0)
        out = flat + selected.to(flat.dtype) * active_mean
        return out.view_as(clean).clamp(0.0, 1.0).detach()

    if mode == "polarity_flip":
        active_location = clean.amax(dim=2, keepdim=True) > 0
        selected = active_location & (
            _event_random(active_location.shape, clean, generator) < probability
        )
        swapped = clean.flip(dims=(2,))
        return torch.where(selected.expand_as(clean), swapped, clean).detach()

    if mode == "frame_drop":
        drop = _event_random((clean.shape[0], clean.shape[1], 1, 1, 1), clean, generator) < probability
        return clean.masked_fill(drop, 0.0).detach()

    # Local temporal jitter: independently move selected event-bin mass to a
    # nearby frame. Boundary events stay in their original frame and collisions
    # are accumulated before returning to the normalized [0, 1] range.
    max_shift = max(int(temporal_jitter_steps), 1)
    selected = (clean > 0) & (_event_random(clean.shape, clean, generator) < probability)
    direction = _event_random(clean.shape, clean, generator) < 0.5
    moved_left = clean * (selected & direction)
    moved_right = clean * (selected & ~direction)
    stationary = clean * (~selected)
    shift = 1 if max_shift == 1 else int(
        torch.randint(1, max_shift + 1, (), device=clean.device, generator=generator).item()
    )
    # Do not wrap the beginning/end of the sequence.
    stationary[:, :shift] += moved_left[:, :shift]
    stationary[:, -shift:] += moved_right[:, -shift:]
    moved_left[:, :shift] = 0
    moved_right[:, -shift:] = 0
    out = stationary + torch.roll(moved_left, shifts=-shift, dims=1)
    out = out + torch.roll(moved_right, shifts=shift, dims=1)
    return out.clamp(0.0, 1.0).detach()
