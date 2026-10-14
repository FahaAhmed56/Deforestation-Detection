"""Evaluate the trained model on the held-out test tiles.

    python -m src.evaluate

Writes models/test_metrics.json, docs/images/confusion_matrix.png and
docs/images/pr_curve.png.
"""
from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import plots
from src.config import project_path
from src.data.dataset import TileDataset, list_tiles
from src.models.losses import confusion, metrics_from_counts
from src.predict import ForestModel


@torch.no_grad()
def main():
    fm = ForestModel()
    files = list_tiles(project_path("data_dir") / "processed" / "test")
    loader = DataLoader(TileDataset(files, fm.mean[:, 0, 0], fm.std[:, 0, 0]), batch_size=16)
    counts = np.zeros(4)
    probs, labels = [], []
    for x, y in loader:
        logits = fm.model(x.to(fm.device)).cpu()
        counts += confusion(logits, y)
        v = y != 255
        probs.append(torch.sigmoid(logits[:, 0])[v].numpy()[::20])   # subsample for PR curve
        labels.append(y[v].numpy()[::20])
    m = metrics_from_counts(*counts)
    m["n_tiles"] = len(files)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, indent=2))
    (project_path("models_dir") / "test_metrics.json").write_text(json.dumps(m, indent=2))

    p, l = np.concatenate(probs), np.concatenate(labels)
    ths = np.linspace(0.02, 0.98, 49)
    prec = [((p >= t) & (l == 1)).sum() / max((p >= t).sum(), 1) for t in ths]
    rec = [((p >= t) & (l == 1)).sum() / max((l == 1).sum(), 1) for t in ths]
    img = project_path("images_dir")
    plots.confusion_matrix(counts, img / "confusion_matrix.png")
    plots.pr_curve(prec, rec, ths, 0.5, img / "pr_curve.png")


if __name__ == "__main__":
    main()
