"""Small geospatial helpers (bounding boxes, pixel areas, indices, PNG export)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from rasterio.transform import from_bounds

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


def bbox_shape(bbox, resolution_m=10.0):
    """Pixel (height, width) that covers a lon/lat bbox at a given ground resolution."""
    w, s, e, n = bbox
    lat = (s + n) / 2
    width = (e - w) * M_PER_DEG_LON_EQ * math.cos(math.radians(lat)) / resolution_m
    height = (n - s) * M_PER_DEG_LAT / resolution_m
    return max(1, round(height)), max(1, round(width))


def bbox_from_center(lat, lon, height_px, width_px, resolution_m=10.0):
    dlat = height_px * resolution_m / M_PER_DEG_LAT
    dlon = width_px * resolution_m / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat)))
    return (lon - dlon / 2, lat - dlat / 2, lon + dlon / 2, lat + dlat / 2)


def transform_for(bbox, shape):
    """Affine transform (pixel → lon/lat) for an EPSG:4326 grid."""
    return from_bounds(*bbox, width=shape[1], height=shape[0])


def pixel_area_m2(bbox, shape):
    w, s, e, n = bbox
    lat = (s + n) / 2
    px_w = (e - w) / shape[1] * M_PER_DEG_LON_EQ * math.cos(math.radians(lat))
    px_h = (n - s) / shape[0] * M_PER_DEG_LAT
    return px_w * px_h


def ndvi(img, red=2, nir=3):
    """Normalised Difference Vegetation Index: (NIR − Red) / (NIR + Red)."""
    r, n = img[red].astype(np.float32), img[nir].astype(np.float32)
    return (n - r) / (n + r + 1e-6)


def save_rgb_png(img, path, gain=3.2, gamma=0.9):
    rgb = np.stack([img[2], img[1], img[0]], -1)
    rgb = np.clip((rgb * gain) ** gamma, 0, 1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path, optimize=True)


def save_mask_png(mask, path, color, alpha=170):
    """Transparent PNG overlay: coloured where mask is True."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[mask] = (*color, alpha)
    Image.fromarray(out, "RGBA").save(path, optimize=True)


def save_prob_png(prob, path, valid=None):
    """Forest probability as a green transparent ramp."""
    h, w = prob.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = 20, 160, 60
    a = np.clip((prob - 0.3) / 0.5, 0, 1) * 190
    if valid is not None:
        a[~valid] = 0
    out[..., 3] = a.astype(np.uint8)
    Image.fromarray(out, "RGBA").save(path, optimize=True)
