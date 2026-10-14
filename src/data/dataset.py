"""PyTorch dataset over .npz tiles produced by prepare_data / build_real_dataset.

Each tile file holds:
    x : (6, H, W) float16 surface reflectance
    y : (H, W)    uint8   1 = forest, 0 = not forest, 255 = ignore (cloud / no data)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

IGNORE = 255


def list_tiles(split_dir: Path) -> list[Path]:
    return sorted(Path(split_dir).glob("*.npz"))


def compute_stats(files, max_files=400):
    """Per-band mean/std over the training set (used to normalise inputs)."""
    s, s2, n = 0.0, 0.0, 0
    for f in files[:max_files]:
        x = np.load(f)["x"].astype(np.float64)
        s = s + x.sum(axis=(1, 2))
        s2 = s2 + (x ** 2).sum(axis=(1, 2))
        n += x.shape[1] * x.shape[2]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class TileDataset(Dataset):
    def __init__(self, files, mean, std, augment=False):
        self.files = list(files)
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x = (d["x"].astype(np.float32) - self.mean) / self.std
        y = d["y"].astype(np.int64)
        if self.augment:
            # satellite images have no "up": flips and 90° rotations are free data
            k = np.random.randint(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            # small brightness jitter imitates atmosphere / season differences
            x = x * np.random.uniform(0.95, 1.05) + np.random.normal(0, 0.03, (x.shape[0], 1, 1))
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), torch.from_numpy(np.ascontiguousarray(y))
