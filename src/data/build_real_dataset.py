"""Build a training set from REAL Sentinel-2 scenes (optional fine-tuning).

Labels come from one of two sources:

* ``--labels ndvi`` (default, zero setup): weak labels from spectral rules —
  forest = NDVI > 0.70 and SWIR1 < 0.20 and NIR > 0.22. Ambiguous pixels
  (NDVI between 0.45 and 0.70) and clouds are set to 255 = "ignore", so the
  model is only taught on confident pixels and learns to generalise to the rest.
* ``--labels <path.tif>``: any forest-mask GeoTIFF (e.g. Hansen Global Forest
  Change tree-cover, ESA WorldCover class 10, or your own hand-labelled mask).
  It is reprojected onto the Sentinel-2 grid automatically.

Example
    python -m src.data.build_real_dataset --bbox -63.2 -10.0 -63.0 -9.85 \
        --dates 2024-07-15 2025-07-15 --out data/real
    python -m src.train --data data/real --resume models/unet_forest.pt --epochs 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.data.sentinel import fetch
from src.geo import ndvi, transform_for


def ndvi_weak_labels(img, valid):
    v = ndvi(img)
    forest = (v > 0.70) & (img[4] < 0.20) & (img[3] > 0.22)
    non = v < 0.45
    y = np.full(v.shape, 255, np.uint8)
    y[forest], y[non] = 1, 0
    y[~valid] = 255
    return y


def raster_labels(path, bbox, shape):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    dst = np.zeros(shape, np.uint8)
    with rasterio.open(path) as src:
        reproject(rasterio.band(src, 1), dst, dst_transform=transform_for(bbox, shape),
                  dst_crs="EPSG:4326", resampling=Resampling.nearest)
    return (dst > 0).astype(np.uint8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--labels", default="ndvi")
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="data/real")
    a = ap.parse_args(argv)

    out = Path(a.out)
    for s in ("train", "val"):
        (out / s).mkdir(parents=True, exist_ok=True)
    rng, n = np.random.default_rng(0), 0
    for d in a.dates:
        img, valid, info = fetch(tuple(a.bbox), d)
        print("loaded", info)
        y = ndvi_weak_labels(img, valid) if a.labels == "ndvi" else raster_labels(a.labels, a.bbox, valid.shape)
        y[~valid] = 255
        t = a.tile
        for r in range(0, img.shape[1] - t + 1, t):
            for c in range(0, img.shape[2] - t + 1, t):
                yt = y[r:r + t, c:c + t]
                if (yt != 255).mean() < 0.3:      # mostly cloud / unlabelled
                    continue
                split = "val" if rng.random() < a.val_frac else "train"
                np.savez_compressed(out / split / f"{d}_{r}_{c}.npz",
                                    x=img[:, r:r + t, c:c + t].astype(np.float16), y=yt)
                n += 1
    print(f"wrote {n} tiles to {out}")


if __name__ == "__main__":
    main()
