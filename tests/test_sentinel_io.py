"""Offline test of the Sentinel-2 reader: builds fake L2A COG bands in UTM
(like the real product), then checks load_scene() reprojects them onto the
lon/lat grid, applies the +1000 DN offset and masks clouds from SCL."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_xy

from src.data.sentinel import load_scene

BBOX = (-63.12, -9.97, -63.08, -9.93)


def _fake_item(tmp_path):
    xs, ys = warp_xy("EPSG:4326", "EPSG:32720", [BBOX[0] - 0.02], [BBOX[3] + 0.02])
    assets = {}
    for band, res, val in [("B02", 10, 1300), ("B03", 10, 1500), ("B04", 10, 1300),
                           ("B08", 10, 4200), ("B11", 20, 2400), ("B12", 20, 1600), ("SCL", 20, 4)]:
        n = 800 // (res // 10)
        arr = np.full((n, n), val, np.uint16)
        if band == "SCL":
            arr[n // 2 - 40: n // 2, :] = 9           # a band of high cloud across the AOI
        path = tmp_path / f"{band}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="uint16",
                           crs="EPSG:32720", transform=from_origin(xs[0], ys[0], res, res), tiled=True) as dst:
            dst.write(arr, 1)
        assets[band] = SimpleNamespace(href=str(path))
    return SimpleNamespace(id="S2_FAKE", assets=assets, datetime=datetime(2024, 7, 20),
                           properties={"s2:processing_baseline": "05.10", "eo:cloud_cover": 3.0})


def test_load_scene_reprojects_offsets_and_masks(tmp_path):
    img, valid, info = load_scene(_fake_item(tmp_path), BBOX)
    assert img.shape[0] == 6 and img.shape[1:] == valid.shape
    assert abs(img.shape[1] - 443) < 5                       # 0.04° lat ≈ 4.4 km at 10 m
    assert np.allclose(img[3][valid].mean(), (4200 - 1000) / 10000, atol=1e-3)   # NIR = 0.32
    assert 0 < valid.mean() < 1                               # cloud rows masked, rest valid
    assert info["scene_id"] == "S2_FAKE" and info["acquired"] == "2024-07-20"
