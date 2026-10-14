"""Change detection between two dates ("post-classification comparison").

Steps
1. Threshold the U-Net probabilities → forest masks F1 (before) and F2 (after).
2. Candidate loss = F1 AND NOT F2   (was forest, is not any more).
   Candidate gain = NOT F1 AND F2   (regrowth / reforestation).
3. Confirm loss with an independent spectral check: NDVI must drop by at
   least `ndvi_drop`. This removes false alarms caused by one bad prediction.
4. Remove pixels that are cloud / shadow / no-data on *either* date.
5. Morphological opening + minimum mapping unit (min_area_ha) remove speckle.
6. Group pixels into connected patches and turn each patch into a GeoJSON
   polygon with its area, centroid and a confidence score.
"""
from __future__ import annotations

import numpy as np
from rasterio import features
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def clean_mask(mask, open_radius=1, min_pixels=10):
    if open_radius > 0:
        mask = ndimage.binary_opening(mask, structure=_disk(open_radius))
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_pixels
    return keep[lab]


def detect_change(prob1, prob2, ndvi1=None, ndvi2=None, valid=None, threshold=0.5,
                  ndvi_drop=0.15, min_pixels=10, open_radius=1):
    f1, f2 = prob1 >= threshold, prob2 >= threshold
    loss, gain = f1 & ~f2, ~f1 & f2
    if ndvi1 is not None and ndvi2 is not None:
        loss &= (ndvi1 - ndvi2) >= ndvi_drop
        gain &= (ndvi2 - ndvi1) >= ndvi_drop / 2
    if valid is not None:
        loss &= valid; gain &= valid
    loss = clean_mask(loss, open_radius, min_pixels)
    gain = clean_mask(gain, open_radius, min_pixels)
    return {"forest_t1": f1, "forest_t2": f2, "loss": loss, "gain": gain}


def vectorize(mask, transform, px_area_m2, props=None, score=None, kind="loss"):
    """Connected patches → GeoJSON FeatureCollection (EPSG:4326)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    feats = []
    if n == 0:
        return {"type": "FeatureCollection", "features": feats}
    counts = np.bincount(lab.ravel(), minlength=n + 1)
    conf = ndimage.mean(score, lab, np.arange(1, n + 1)) if score is not None else np.ones(n)
    polys: dict[int, list] = {}
    for geom, val in features.shapes(lab.astype(np.int32), mask=mask, transform=transform, connectivity=8):
        polys.setdefault(int(val), []).append(shape(geom))
    for i, (lid, parts) in enumerate(sorted(polys.items(), key=lambda kv: -counts[kv[0]])):
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        c = geom.centroid
        feats.append({
            "type": "Feature",
            "geometry": mapping(geom.simplify(transform.a * 0.5)),
            "properties": {
                "id": i + 1, "type": kind,
                "area_ha": round(counts[lid] * px_area_m2 / 10_000, 3),
                "pixels": int(counts[lid]),
                "confidence": round(float(np.clip(conf[lid - 1], 0, 1)), 3),
                "centroid": [round(c.y, 6), round(c.x, 6)],
                **(props or {}),
            },
        })
    return {"type": "FeatureCollection", "features": feats}


def score_against_truth(pred, truth, valid=None):
    """Pixel precision / recall / F1 / IoU of a change mask vs. a truth mask."""
    if valid is not None:
        pred, truth = pred & valid, truth & valid
    tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum()); fn = int((~pred & truth).sum())
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return {"precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r + 1e-9), 4), "iou": round(tp / (tp + fp + fn + 1e-9), 4)}
