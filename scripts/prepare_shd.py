"""Acquire the official SHD HDF5 files and prepare variable-length count frames."""
import argparse
import gzip
import json
from pathlib import Path
from shutil import copyfileobj
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paramut.dcls_data import CACHE_SCHEMA, frame_events, sha256

HDF5_HASHES = {
    "train": "2bddb4bd46732f09982b7d1631b7c29c19853c73d3d240e3eb32bba909bdd6c1",
    "test": "47d8621a092cb483bea9448a93c0a3375a6726d370045090c2f7a77f70f95df3",
}


def acquire(root, split, download):
    target = root / f"shd_{split}.h5"
    if not target.exists():
        if not download:
            raise FileNotFoundError(f"Missing {target}. Supply the official HDF5 file or use --download.")
        root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=root) as directory:
            temp = Path(directory) / target.name
            url = f"https://zenkelab.org/datasets/shd/{target.name}.gz"
            print(f"Downloading {url}", flush=True)
            with urlopen(url, timeout=120) as response, gzip.GzipFile(fileobj=response) as source:
                with temp.open("wb") as destination:
                    copyfileobj(source, destination)
            if sha256(temp) != HDF5_HASHES[split]:
                raise ValueError(f"Unexpected downloaded content for {target.name}")
            temp.replace(target)
    elif sha256(target) != HDF5_HASHES[split]:
        raise ValueError(f"Unexpected content for {target}")
    return target


def convert(source_path, cache_path):
    frames = []
    offsets = [0]
    with h5py.File(source_path, "r") as source:
        labels = np.asarray(source["labels"], dtype=np.int64)
        ids = np.lexsort((np.arange(len(labels)), labels))
        times = source["spikes/times"]
        units = source["spikes/units"]
        for index in ids:
            event_times = times[index]
            counts = frame_events(event_times, units[index])
            if counts.sum(dtype=np.float64) != len(event_times):
                raise ValueError(f"Event count mismatch at index {index}")
            frames.append(counts)
            offsets.append(offsets[-1] + len(counts))
    np.savez_compressed(cache_path, frames=np.concatenate(frames), offsets=offsets,
                        labels=labels[ids], ids=ids)
    return {"samples": len(ids), "cache_sha256": sha256(cache_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/shd/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/shd/frames"))
    parser.add_argument("--download", action="store_true", help="Fetch missing raw HDF5 files.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = {split: acquire(args.source, split, args.download) for split in HDF5_HASHES}
    # Stage both splits before publishing the content-addressed manifest.
    manifest = {"schema": CACHE_SCHEMA, "source_hashes": HDF5_HASHES, "splits": {}}
    with TemporaryDirectory(dir=args.output) as directory:
        temp = Path(directory)
        for split, source in sources.items():
            print(f"Preparing {split} frames", flush=True)
            manifest["splits"][split] = convert(source, temp / f"{split}.npz")
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in ("train.npz", "test.npz", "manifest.json"):
            (temp / name).replace(args.output / name)


if __name__ == "__main__":
    main()
