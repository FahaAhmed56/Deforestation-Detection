"""Physically-plausible synthetic Sentinel-2 scenes.

Why this exists
---------------
Real Sentinel-2 imagery is free, but *pixel-level forest labels* are not
something you can download in one click, and a full training set is many GB.
This module procedurally generates landscapes (forest, pasture, bare soil,
roads, rivers, regrowth, clouds) and renders them with realistic surface
reflectance values for the six Sentinel-2 bands the model uses. This lets the
whole system — training, inference, change detection and the map GUI — run
offline in minutes. For real data, see ``src/data/sentinel.py`` and
``src/data/build_real_dataset.py``.

The spectra below are typical L2A surface-reflectance values (0-1) taken from
the shape of published spectral signatures: healthy forest is dark in red and
SWIR but bright in near-infrared; bare soil is bright in SWIR.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# Land-cover class ids
PASTURE, FOREST, BARE, WATER, ROAD, REGROWTH = 0, 1, 2, 3, 4, 5
CLASS_NAMES = {PASTURE: "pasture", FOREST: "forest", BARE: "bare soil",
               WATER: "water", ROAD: "road", REGROWTH: "regrowth"}

#                 B02    B03    B04    B08    B11    B12
SPECTRA = np.array([
    [0.050, 0.085, 0.070, 0.270, 0.230, 0.130],   # pasture
    [0.030, 0.050, 0.030, 0.320, 0.140, 0.060],   # forest
    [0.085, 0.115, 0.150, 0.230, 0.320, 0.240],   # bare soil (fresh clearing)
    [0.050, 0.060, 0.040, 0.020, 0.010, 0.008],   # water
    [0.110, 0.135, 0.160, 0.210, 0.300, 0.250],   # unpaved road
    [0.036, 0.062, 0.042, 0.305, 0.170, 0.080],   # secondary regrowth (close to forest!)
], dtype=np.float32)
CLOUD_SPECTRUM = np.array([0.45, 0.45, 0.46, 0.48, 0.38, 0.30], dtype=np.float32)
TEXTURE = np.array([0.05, 0.14, 0.10, 0.03, 0.06, 0.08], dtype=np.float32)  # per class


# ---------------------------------------------------------------- helpers
def smooth_noise(rng, shape, sigma):
    """Zero-mean, unit-variance spatially-correlated noise."""
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return (n - n.mean()) / (n.std() + 1e-8)


def blob(rng, shape, cy, cx, ry, rx, roughness=0.35):
    """Irregular, organic-looking patch (an ellipse with a noisy edge)."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d = ((yy - cy) / max(ry, 1)) ** 2 + ((xx - cx) / max(rx, 1)) ** 2
    edge = smooth_noise(rng, shape, sigma=max(2, min(ry, rx) / 2)) * roughness
    return d < (1.0 + edge)


def draw_line(mask, p0, p1, width=1):
    """Rasterise a thick straight line segment into a boolean mask."""
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    ys = np.linspace(p0[0], p1[0], n).round().astype(int)
    xs = np.linspace(p0[1], p1[1], n).round().astype(int)
    h, w = mask.shape
    for dy in range(-(width // 2), width - width // 2):
        for dx in range(-(width // 2), width - width // 2):
            yy, xx = ys + dy, xs + dx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            mask[yy[ok], xx[ok]] = True
    return mask


def river(rng, shape):
    h, w = shape
    x = np.arange(w)
    y0 = rng.uniform(0.2, 0.8) * h
    amp, freq, phase = rng.uniform(5, h / 6), rng.uniform(1, 3) * np.pi / w, rng.uniform(0, 6)
    ys = y0 + amp * np.sin(freq * x + phase) + np.cumsum(rng.normal(0, 0.4, w))
    mask = np.zeros(shape, bool)
    width = int(rng.integers(2, 6))
    for xi, yi in zip(x, ys):
        lo, hi = int(yi - width / 2), int(yi + width / 2) + 1
        mask[max(lo, 0):max(min(hi, h), 0), xi] = True
    return mask


# ---------------------------------------------------------- landscape model
def make_landscape(rng, size, forest_frac=0.7):
    """Return an integer land-cover map of shape (size, size)."""
    shape = (size, size)
    cls = np.full(shape, PASTURE, np.uint8)
    f = smooth_noise(rng, shape, sigma=rng.uniform(5, 18))
    cls[f > np.quantile(f, 1 - forest_frac)] = FOREST

    # patches of bare soil and regrowth outside the forest
    b = smooth_noise(rng, shape, sigma=4)
    cls[(b > 1.5) & (cls != FOREST)] = BARE
    r = smooth_noise(rng, shape, sigma=5)
    cls[(r > 1.6) & (cls == PASTURE)] = REGROWTH
    if rng.random() < 0.35:
        cls[river(rng, shape)] = WATER
    return cls


def add_clearings(rng, cls, n, fresh=True, road_prob=0.6):
    """Cut `n` forest clearings (optionally connected by a road).

    Mimics the 'fishbone' pattern seen in the Amazon: a road is opened into
    the forest, then plots are cleared along it. Returns the mask of pixels
    that changed from forest to non-forest.
    """
    h, w = cls.shape
    before = cls == FOREST
    for _ in range(n):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        ry, rx = rng.uniform(3, h / 7), rng.uniform(3, w / 7)
        m = blob(rng, cls.shape, cy, cx, ry, rx)
        if rng.random() < 0.5:   # some clearings are rectangular plots
            y0, x0 = int(max(cy - ry, 0)), int(max(cx - rx, 0))
            m = np.zeros_like(m)
            m[y0:int(cy + ry), x0:int(cx + rx)] = True
        cls[m & (cls != WATER)] = BARE if fresh else PASTURE
        if rng.random() < road_prob:
            road = draw_line(np.zeros(cls.shape, bool), (cy, cx),
                             (rng.integers(0, h), rng.choice([0, w - 1])), width=2)
            cls[road & (cls != WATER)] = ROAD
    return before & (cls != FOREST)


# ------------------------------------------------------------- rendering
def render(rng, cls, atmosphere=None, cloud_cover=0.0):
    """Render a land-cover map into a (6, H, W) reflectance image.

    Returns (image, cloud_mask). `atmosphere` lets several dates of the same
    place share a look but differ slightly (sun angle, haze, season).
    """
    h, w = cls.shape
    img = SPECTRA[cls].transpose(2, 0, 1).copy()               # (6, H, W)
    # 1. within-class variation: species mix, soil moisture, grass greenness …
    #    each band drifts independently over large areas (±12 %)
    for b in range(img.shape[0]):
        img[b] *= 1.0 + 0.12 * smooth_noise(rng, (h, w), sigma=rng.uniform(8, 25))
    # 2. canopy / field texture
    tex = smooth_noise(rng, (h, w), sigma=1.2)
    fine = rng.standard_normal((h, w))
    img *= 1.0 + (0.6 * tex + 0.4 * fine)[None] * TEXTURE[cls][None] * 1.6
    # 3. terrain shading (hills make slopes facing away from the sun darker)
    img *= (1.0 + 0.12 * smooth_noise(rng, (h, w), sigma=10))[None]
    # 4. sensor point-spread: pixels on a boundary are a mix of both classes
    img = ndimage.gaussian_filter(img, sigma=(0, 0.7, 0.7))
    img += rng.normal(0, 0.007, img.shape)                     # sensor noise

    if atmosphere is None:
        atmosphere = random_atmosphere(rng)
    gain, haze = atmosphere
    img = img * gain[:, None, None] + haze[:, None, None]
    # 5. thin, un-masked haze patches (the cloud mask misses these in reality too)
    if rng.random() < 0.4:
        thin = np.clip(smooth_noise(rng, (h, w), sigma=15) * 0.08, 0, 0.12)
        img = img * (1 - thin[None]) + CLOUD_SPECTRUM[:, None, None] * thin[None]

    cloud = np.zeros((h, w), bool)
    if cloud_cover > 0:
        c = smooth_noise(rng, (h, w), sigma=rng.uniform(6, 14))
        alpha = np.clip((c - np.quantile(c, 1 - cloud_cover)) * 2.5, 0, 1)
        shadow = np.roll(alpha, (int(rng.integers(4, 10)), int(rng.integers(4, 10))), (0, 1))
        img *= (1 - 0.45 * shadow * (1 - alpha))[None]
        img = img * (1 - alpha[None]) + CLOUD_SPECTRUM[:, None, None] * alpha[None]
        cloud = (alpha > 0.35) | (shadow > 0.5)
    return np.clip(img, 0, 1).astype(np.float32), cloud


def random_atmosphere(rng):
    """Scene-wide differences: sun angle, season, residual atmospheric correction error."""
    gain = (rng.normal(1.0, 0.07) + rng.normal(0, 0.04, 6)).astype(np.float32)
    haze = np.abs(rng.normal(0, 0.015, 6)).astype(np.float32) * np.linspace(1.6, 0.3, 6)
    return gain, haze.astype(np.float32)


def make_training_tile(rng, size=128):
    """One (image, label) training pair. label: 1 forest, 0 not, 255 ignore."""
    cls = make_landscape(rng, size, forest_frac=rng.uniform(0.05, 0.95))
    add_clearings(rng, cls, int(rng.integers(0, 10)), fresh=rng.random() < 0.6)
    cloud_cover = rng.uniform(0.05, 0.35) if rng.random() < 0.3 else 0.0
    img, cloud = render(rng, cls, cloud_cover=cloud_cover)
    label = (cls == FOREST).astype(np.uint8)
    label[cloud] = 255                       # don't teach the model under clouds
    return img, label


# ------------------------------------------------------- demo time series
def make_demo_timeseries(seed=7, size=512, dates=None):
    """A forest frontier that is progressively cleared over several years.

    Returns a list of dicts {date, image, cloud, cls} plus the true loss mask
    between each consecutive pair (used to score the change detector).
    """
    rng = np.random.default_rng(seed)
    dates = dates or ["2023-07-15", "2024-07-12", "2025-07-17", "2026-07-14"]
    cls = make_landscape(rng, size, forest_frac=0.80)
    # a main highway with an initial settlement along it
    hw = draw_line(np.zeros(cls.shape, bool), (int(size * 0.85), 0), (int(size * 0.7), size - 1), 3)
    cls[hw] = ROAD
    for _ in range(6):
        cy = int(size * rng.uniform(0.70, 0.95)); cx = int(rng.integers(0, size))
        cls[blob(rng, cls.shape, cy, cx, rng.uniform(10, 30), rng.uniform(15, 40))
            & (cls != ROAD) & (cls != WATER)] = PASTURE

    series, true_loss = [], []
    branch_x = sorted(rng.choice(np.arange(40, size - 40), 7, replace=False))
    for t, date in enumerate(dates):
        if t > 0:
            before = cls == FOREST
            cls[cls == BARE] = PASTURE                    # last year's clearings age
            # open new side roads off the highway ("fishbone") and clear plots
            for bx in branch_x[(t - 1) * 2:(t - 1) * 2 + 3]:
                top = int(size * rng.uniform(0.15, 0.45))
                start = int(size * 0.85 - (size * 0.15) * bx / size)
                road = draw_line(np.zeros(cls.shape, bool), (start, bx), (top, bx + rng.integers(-30, 30)), 2)
                cls[road & (cls != WATER)] = ROAD
                for _ in range(int(rng.integers(3, 6))):
                    cy = int(rng.uniform(top, start)); side = rng.choice([-1, 1])
                    m = blob(rng, cls.shape, cy, bx + side * rng.uniform(10, 25),
                             rng.uniform(5, 14), rng.uniform(8, 20))
                    cls[m & (cls != ROAD) & (cls != WATER)] = BARE
            # one large illegal clearing
            cy, cx = int(rng.uniform(0.1, 0.6) * size), int(rng.uniform(0.1, 0.9) * size)
            cls[blob(rng, cls.shape, cy, cx, rng.uniform(15, 35), rng.uniform(20, 45))
                & (cls != WATER)] = BARE
            true_loss.append(before & (cls != FOREST))
        cover = [0.0, 0.12, 0.0, 0.08][t % 4]
        img, cloud = render(rng, cls, cloud_cover=cover)
        series.append({"date": date, "image": img, "cloud": cloud, "cls": cls.copy()})
    return series, true_loss
