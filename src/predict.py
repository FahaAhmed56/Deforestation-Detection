"""Inference: run the trained U-Net over a full scene of any size.

A scene (e.g. 1500×1500 px) is too large to feed in one go and the model was
trained on 128 px tiles, so we slide an overlapping window across it and
blend the predictions with a smooth (Hann) weight — this removes the visible
seams you get at tile edges with naive tiling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import load_config, project_path
from src.models.unet import UNet


class ForestModel:
    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = Path(checkpoint or project_path("models_dir") / "unet_forest.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"{checkpoint} not found — train first: python -m src.train")
        ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        mc = ck["model_cfg"]
        self.bands = ck["bands"]
        self.model = UNet(len(self.bands), 1, mc["base_channels"], mc["depth"], mc["dropout"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(self.device).eval()
        self.mean = np.asarray(ck["mean"], np.float32)[:, None, None]
        self.std = np.asarray(ck["std"], np.float32)[:, None, None]
        self.meta = {"epoch": ck.get("epoch"), "val": ck.get("val"), "path": str(checkpoint)}
        inf = load_config()["inference"]
        self.tile, self.overlap = inf["tile_size"], inf["overlap"]

    @torch.no_grad()
    def predict(self, image: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """image: (bands, H, W) reflectance → (H, W) forest probability 0..1"""
        c, h, w = image.shape
        t, step = self.tile, self.tile - self.overlap
        ph, pw = max(h, t), max(w, t)
        x = np.zeros((c, ph, pw), np.float32)
        x[:, :h, :w] = image
        x = (x - self.mean) / self.std

        ys = list(range(0, ph - t + 1, step)) or [0]
        xs = list(range(0, pw - t + 1, step)) or [0]
        if ys[-1] != ph - t: ys.append(ph - t)
        if xs[-1] != pw - t: xs.append(pw - t)
        win = np.outer(np.hanning(t), np.hanning(t)).astype(np.float32) + 1e-3

        acc = np.zeros((ph, pw), np.float32)
        wsum = np.zeros((ph, pw), np.float32)
        coords = [(yy, xx) for yy in ys for xx in xs]
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            batch = np.stack([x[:, yy:yy + t, xx:xx + t] for yy, xx in chunk])
            prob = torch.sigmoid(self.model(torch.from_numpy(batch).to(self.device)))[:, 0].cpu().numpy()
            for (yy, xx), p in zip(chunk, prob):
                acc[yy:yy + t, xx:xx + t] += p * win
                wsum[yy:yy + t, xx:xx + t] += win
        return (acc / wsum)[:h, :w]
