"""Download real Sentinel-2 L2A imagery for any area — free, no account needed.

Source: Microsoft Planetary Computer STAC API (collection ``sentinel-2-l2a``).
STAC = SpatioTemporal Asset Catalog, a standard JSON API for searching
satellite scenes by place, date and cloud cover. Each scene ("item") links to
Cloud-Optimised GeoTIFFs (COGs), so we can stream *only the pixels inside our
area of interest* instead of downloading 800 MB tiles.

Pipeline for one date:
    search scenes ±N days → keep low-cloud ones → pick the clearest → for each
    band, open the COG, reproject on the fly to lon/lat (EPSG:4326) and read
    the AOI window → convert digital numbers to reflectance → build a validity
    mask from the Scene Classification Layer (SCL).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from src.config import load_config
from src.geo import bbox_shape, transform_for

# Scene Classification Layer classes that are NOT usable ground observations
SCL_INVALID = {0: "no data", 1: "saturated", 3: "cloud shadow", 8: "cloud (medium)",
               9: "cloud (high)", 10: "thin cirrus"}


def _parse(d):
    return d if isinstance(d, date) else date.fromisoformat(str(d))


def search_scene(bbox, target_date, window_days=None, max_cloud=None):
    """Return the least-cloudy STAC item near `target_date` covering `bbox`."""
    import planetary_computer
    import pystac_client

    cfg = load_config()["sentinel"]
    window_days = window_days or cfg["search_window_days"]
    max_cloud = cfg["max_cloud"] if max_cloud is None else max_cloud
    t = _parse(target_date)
    catalog = pystac_client.Client.open(cfg["stac_url"], modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[cfg["collection"]], bbox=bbox,
        datetime=f"{t - timedelta(days=window_days)}/{t + timedelta(days=window_days)}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        raise LookupError(f"No Sentinel-2 scene with <{max_cloud}% cloud within ±{window_days} days of {t}")

    def rank(it):  # prefer clear scenes, then closeness in time
        dt = abs((_parse(it.datetime.date()) - t).days)
        return it.properties.get("eo:cloud_cover", 100) + dt * 0.2
    return sorted(items, key=rank)[0]


def _read_band(href, bbox, shape, resampling):
    """Warp one band straight onto our lon/lat grid (exact bbox, exact shape).

    Defining the VRT with our own transform/width/height means GDAL only fetches
    the COG blocks that overlap the AOI and every band/date lands on the same grid.
    Pixels outside the scene footprint come back as 0 (= no data).
    """
    import rasterio
    from rasterio.vrt import WarpedVRT

    with rasterio.open(href) as src, WarpedVRT(
        src, crs="EPSG:4326", transform=transform_for(bbox, shape),
        width=shape[1], height=shape[0], resampling=resampling, nodata=0,
    ) as vrt:
        return vrt.read(1)


def load_scene(item, bbox, bands=None, resolution_m=None):
    """Read an AOI from a STAC item. Returns (image (B,H,W) float32, valid (H,W) bool, info)."""
    from rasterio.enums import Resampling

    cfg = load_config()
    bands = bands or cfg["bands"]
    shape = bbox_shape(bbox, resolution_m or cfg["sentinel"]["resolution_m"])
    cap = cfg["sentinel"]["max_pixels"]
    if max(shape) > cap:
        raise ValueError(f"AOI is {shape[1]}×{shape[0]} px; max is {cap} per side — draw a smaller box")

    # Since processing baseline 04.00 (Jan 2022) ESA adds +1000 to every DN.
    baseline = float(item.properties.get("s2:processing_baseline", "0") or 0)
    offset = 1000 if baseline >= 4.0 else 0

    img = np.zeros((len(bands),) + shape, np.float32)
    for i, b in enumerate(bands):
        dn = _read_band(item.assets[b].href, bbox, shape, Resampling.bilinear).astype(np.float32)
        img[i] = np.clip((dn - offset) / 10_000.0, 0, 1)
    scl = _read_band(item.assets["SCL"].href, bbox, shape, Resampling.nearest)
    valid = ~np.isin(scl, list(SCL_INVALID))
    info = {
        "scene_id": item.id,
        "acquired": item.datetime.date().isoformat(),
        "scene_cloud_pct": round(item.properties.get("eo:cloud_cover", -1), 1),
        "aoi_invalid_pct": round(100 * (1 - valid.mean()), 1),
        "platform": item.properties.get("platform"),
    }
    return img, valid, info


def fetch(bbox, target_date):
    """Convenience: search + load in one call."""
    item = search_scene(bbox, target_date)
    return load_scene(item, bbox)
