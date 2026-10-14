"""Fast unit tests — run with:  python -m pytest -q"""
import numpy as np
import torch

from src.change_detection import clean_mask, detect_change, score_against_truth, vectorize
from src.data.synthetic import FOREST, make_landscape, make_training_tile, render
from src.geo import bbox_from_center, bbox_shape, ndvi, pixel_area_m2, transform_for
from src.models.losses import bce_dice_loss, confusion, metrics_from_counts
from src.models.unet import UNet


def test_unet_shapes_any_size():
    m = UNet(6, 1, base_channels=8, depth=3).eval()
    for h, w in [(64, 64), (128, 96), (70, 50)]:
        assert m(torch.randn(2, 6, h, w)).shape == (2, 1, h, w)


def test_loss_ignores_cloud_pixels():
    logits = torch.zeros(1, 1, 4, 4)
    y = torch.full((1, 4, 4), 255, dtype=torch.long)
    y[0, 0, 0] = 1
    assert torch.isfinite(bce_dice_loss(logits, y))
    tp, fp, fn, tn = confusion(torch.ones(1, 1, 4, 4) * 5, y)
    assert (tp, fp, fn, tn) == (1, 0, 0, 0)          # 15 ignored pixels not counted


def test_metrics_perfect():
    m = metrics_from_counts(10, 0, 0, 10)
    assert abs(m["iou"] - 1) < 1e-6 and abs(m["f1"] - 1) < 1e-6


def test_synthetic_spectra_are_plausible():
    rng = np.random.default_rng(0)
    cls = make_landscape(rng, 64, forest_frac=0.5)
    img, _ = render(rng, cls)
    v = ndvi(img)
    assert v[cls == FOREST].mean() > 0.7            # dense forest NDVI
    assert v[cls != FOREST].mean() < v[cls == FOREST].mean()
    x, y = make_training_tile(rng, 64)
    assert x.shape == (6, 64, 64) and set(np.unique(y)) <= {0, 1, 255}


def test_change_detection_finds_clearing():
    p1 = np.full((60, 60), 0.9, np.float32)
    p2 = p1.copy(); p2[10:30, 10:30] = 0.1           # 20x20 clearing
    n1 = np.full_like(p1, 0.85); n2 = n1.copy(); n2[10:30, 10:30] = 0.3
    out = detect_change(p1, p2, n1, n2, min_pixels=10)
    # opening with a round kernel trims the 4 square corners → 396 of 400 px
    assert 390 <= out["loss"].sum() <= 400 and out["gain"].sum() == 0
    truth = np.zeros_like(out["loss"]); truth[10:30, 10:30] = True
    assert score_against_truth(out["loss"], truth)["iou"] > 0.98


def test_ndvi_check_suppresses_false_alarm():
    p1 = np.full((40, 40), 0.9, np.float32); p2 = p1.copy(); p2[5:20, 5:20] = 0.2
    n = np.full_like(p1, 0.8)                         # NDVI unchanged → not real loss
    assert detect_change(p1, p2, n, n.copy())["loss"].sum() == 0


def test_min_area_removes_speckle():
    m = np.zeros((30, 30), bool); m[3, 3] = True; m[10:20, 10:20] = True
    assert clean_mask(m, open_radius=0, min_pixels=5).sum() == 100


def test_vectorize_area_and_geometry():
    bbox = bbox_from_center(-10, -63, 100, 100, 10)
    shape = bbox_shape(bbox, 10)
    assert abs(shape[0] - 100) <= 1 and abs(shape[1] - 100) <= 1
    px = pixel_area_m2(bbox, shape)
    assert 95 < px < 105                             # ≈ 10 m × 10 m
    m = np.zeros(shape, bool); m[20:40, 20:40] = True
    gj = vectorize(m, transform_for(bbox, shape), px)
    f = gj["features"][0]["properties"]
    assert len(gj["features"]) == 1 and abs(f["area_ha"] - 4.0) < 0.2   # 400 px × 100 m² = 4 ha
