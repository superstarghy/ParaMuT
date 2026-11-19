#!/usr/bin/env python3
"""Download original UCI archive; preserve official subject-disjoint windows."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile
import numpy as np

URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
CHANNELS = [f'{sensor}_{axis}' for sensor in ['body_acc', 'body_gyro', 'total_acc'] for axis in 'xyz']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    archive = a.root/'official.zip'
    if not archive.exists():
        tmp = archive.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=120) as src, tmp.open('wb') as dst:
            while True:
                chunk = src.read(1024*1024)
                if not chunk: break
                dst.write(chunk)
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as outer:
        nested = [n for n in outer.namelist() if n.endswith('.zip')]
        z = zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) if nested else outer
        if z.testzip() is not None: raise RuntimeError('Corrupt archive')
        names = z.namelist()
        def get(suffix):
            found = [n for n in names if n.endswith('/'+suffix)]
            if len(found) != 1: raise RuntimeError((suffix, found))
            return z.read(found[0])
        data = {}; splits = {}
        for split, count in [('train', 7352), ('test', 2947)]:
            x = np.stack([np.loadtxt(io.BytesIO(get(f'{split}/Inertial Signals/{c}_{split}.txt')),
                                      dtype=np.float32) for c in CHANNELS], axis=-1)
            y = np.loadtxt(io.BytesIO(get(f'{split}/y_{split}.txt')), dtype=np.int64)-1
            subject = np.loadtxt(io.BytesIO(get(f'{split}/subject_{split}.txt')), dtype=np.int64)
            assert x.shape == (count, 128, 9) and y.shape == subject.shape == (count,)
            assert np.isfinite(x).all() and set(y) == set(range(6))
            data.update({f'{split}_x': x, f'{split}_y': y, f'{split}_subject': subject})
            splits[split] = {'samples': count, 'subjects': sorted(set(subject.tolist())),
                             'classes': np.bincount(y, minlength=6).tolist()}
        assert not set(splits['train']['subjects']) & set(splits['test']['subjects'])
        assert len(splits['train']['subjects']) == 21 and len(splits['test']['subjects']) == 9
        mean = data['train_x'].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
        std = data['train_x'].std(axis=(0, 1), dtype=np.float64).astype(np.float32)
        assert (std > 0).all()
        data.update(mean=mean, std=std)
        # Contains unnormalized signals, so preprocessing and attack units stay explicit.
        np.savez_compressed(a.root/'signals.npz', **data)
        (a.root/'README.official.txt').write_bytes(get('README.txt'))
    manifest = {'source': URL, 'archive_sha256': sha(archive), 'signals_sha256': sha(a.root/'signals.npz'),
                'channels': CHANNELS, 'splits': splits, 'T': 128, 'Hz': 50,
                'normalization': 'per-channel mean/std fitted only to all official train windows',
                'mean': mean.tolist(), 'std': std.tolist(), 'raw_dtype': 'float32',
                'window_overlap': .5, 'subject_disjoint': True}
    (a.root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
