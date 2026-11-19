"""10 ms SHD frames, variable-length batches, and evaluation padding."""
from __future__ import annotations

from functools import partial
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

CACHE_SCHEMA = 'paramut.dcls_shd_10ms.v1'
DEFAULT_CACHE = Path('data/shd/frames')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def frame_events(times, units):
    """Vectorized equivalent of the 2024 oracle plus upstream 5-channel sums.

    Subtract the first event time after converting seconds to milliseconds.
    The final frame includes all remaining events, including an exact right
    endpoint. No artificial block padding or fixed 1.4-second crop is applied.
    """
    times, units = np.asarray(times), np.asarray(units)
    if times.ndim != 1 or units.shape != times.shape or not len(times):
        raise ValueError('Expected nonempty matching one-dimensional event arrays')
    if not np.isfinite(times).all() or np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError('Event times must be finite, nonnegative, and ordered')
    if not np.issubdtype(units.dtype, np.integer) or np.any(units < 0) or np.any(units >= 700):
        raise ValueError('Expected integer SHD channel indices in [0,700)')
    t = 1000*times
    t = t-t[0]
    count = int(math.ceil(t[-1]/10))
    if count < 1:
        raise ValueError('Zero-duration sequence is undefined in the official oracle')
    bins = np.minimum((t//10).astype(np.int64), count-1)
    flat = bins*140 + units.astype(np.int64)//5
    result = np.bincount(flat, minlength=count*140).reshape(count, 140)
    if result.max(initial=0) > 2**24:
        raise OverflowError('Counts exceed exact float32 integer range')
    return result.astype(np.float32)


def collate(items, pad_to=None):
    """Pad to a fixed parent-batch T when splitting evaluation microbatches."""
    if not items:
        raise ValueError('Cannot collate an empty batch')
    xs, ys, ids = zip(*items)
    xs = [torch.as_tensor(x, dtype=torch.float32) for x in xs]
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    size = int(lengths.max()) if pad_to is None else int(pad_to)
    if size < int(lengths.max()) or any(x.ndim != 2 or x.shape[1] != 140 for x in xs):
        raise ValueError('Invalid shape or pad_to shorter than an input')
    x = torch.zeros(len(xs), size, 140, dtype=torch.float32)
    for i, seq in enumerate(xs):
        x[i, :len(seq)] = seq
    return x, torch.tensor(ys, dtype=torch.long), lengths, torch.tensor(ids, dtype=torch.long)


class DCLSHDDataset(Dataset):
    def __init__(self, cache_root=DEFAULT_CACHE, split='train', verify=True):
        if split not in ('train', 'test'):
            raise ValueError('Only official full train and test are supported')
        self.cache_root, self.split = Path(cache_root), split
        self.manifest = json.loads((self.cache_root/'manifest.json').read_text())
        if self.manifest['schema'] != CACHE_SCHEMA:
            raise ValueError('Unexpected DCLS cache schema')
        path = self.cache_root/f'{split}.npz'
        expected = self.manifest['splits'][split]
        if verify and sha256(path) != expected['cache_sha256']:
            raise RuntimeError('DCLS cache checksum mismatch')
        with np.load(path, allow_pickle=False) as z:
            self.frames = z['frames']
            self.offsets = z['offsets']
            self.labels = z['labels']
            self.ids = z['ids']
        self.lengths = np.diff(self.offsets)
        if self.frames.dtype != np.float32 or self.frames.shape != (int(self.offsets[-1]), 140):
            raise ValueError('Invalid cached frames')
        if len(self.offsets) != len(self.labels)+1 or len(self.ids) != len(self.labels):
            raise ValueError('Invalid ragged offsets')
        if self.offsets[0] != 0 or np.any(self.lengths <= 0) or len(np.unique(self.ids)) != len(self.ids):
            raise ValueError('Invalid lengths or duplicate sample IDs')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        start, end = self.offsets[index:index+2]
        return torch.from_numpy(self.frames[start:end]), int(self.labels[index]), int(self.ids[index])


def loader(dataset, batch_size=256, shuffle=False, num_workers=0, pad_to=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=partial(collate, pad_to=pad_to), pin_memory=True)
