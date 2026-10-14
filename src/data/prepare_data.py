"""Generate the synthetic training / validation / test tile sets.

    python -m src.data.prepare_data            # uses sizes from config.yaml
    python -m src.data.prepare_data --n-train 200 --n-val 50 --n-test 50
"""
from __future__ import annotations

import argparse

import numpy as np
from tqdm import tqdm

from src.config import load_config, project_path
from src.data.synthetic import make_training_tile


def generate_split(name, n, size, seed):
    out = project_path("data_dir") / "processed" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.npz"):
        f.unlink()
    rng = np.random.default_rng(seed)
    for i in tqdm(range(n), desc=f"{name:>5}", ncols=80):
        x, y = make_training_tile(rng, size)
        np.savez_compressed(out / f"{name}_{i:05d}.npz", x=x.astype(np.float16), y=y)
    return out


def main(argv=None):
    cfg = load_config()["dataset"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=cfg["n_train"])
    ap.add_argument("--n-val", type=int, default=cfg["n_val"])
    ap.add_argument("--n-test", type=int, default=cfg["n_test"])
    a = ap.parse_args(argv)
    seed, size = cfg["seed"], cfg["tile_size"]
    generate_split("train", a.n_train, size, seed)
    generate_split("val", a.n_val, size, seed + 1)
    generate_split("test", a.n_test, size, seed + 2)
    print("done ->", project_path("data_dir") / "processed")


if __name__ == "__main__":
    main()
