# How ForestWatch works — every piece explained

This document walks through the whole system from satellite to map, and explains
**what each technology is, why it was chosen, and where it lives in the code**.

---

## 1. The big picture

```
 ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌──────────────┐
 │ Sentinel-2   │──►│ Pre-process  │──►│ U-Net         │──►│ Change detection │──►│ Web map GUI  │
 │ (2+ dates)   │   │ bands, clouds│   │ forest prob.  │   │ loss / gain      │   │ Leaflet      │
 └──────────────┘   └──────────────┘   └───────────────┘   └──────────────────┘   └──────────────┘
  src/data/          src/data/          src/predict.py      src/change_detection   app/server.py
  sentinel.py        sentinel.py        src/models/unet.py  .py                    app/static/*
```

1. **Get imagery** of the same area on two or more dates.
2. **Segment** each image into forest / non-forest with a U-Net (a neural network
   that outputs one label *per pixel*).
3. **Compare** the forest maps: pixels that were forest before and are not
   forest now are *loss* (deforestation).
4. **Clean and vectorise** the loss into polygons with areas in hectares.
5. **Show** everything on an interactive map, with statistics and downloads.
6. **Repeat** whenever a new satellite pass arrives (`src/monitor.py`).

---

## 2. Sentinel-2 — the satellite data

**What it is.** Two European Space Agency satellites (Sentinel-2A/2B, and now 2C)
that photograph all land on Earth roughly **every 5 days** at **10 m resolution**
(one pixel = 10 m × 10 m = 100 m²). The data is **free and open**.

**Bands.** A normal photo has 3 colours. Sentinel-2's MSI camera records 13 bands.
We use 6:

| Band | Wavelength | Resolution | Why it helps |
|------|-----------|-----------|--------------|
| B02 Blue | 490 nm | 10 m | true-colour image, haze |
| B03 Green | 560 nm | 10 m | true-colour image |
| B04 Red | 665 nm | 10 m | chlorophyll **absorbs** red → forest looks dark |
| B08 NIR | 842 nm | 10 m | leaf cell structure **reflects** NIR → forest is bright |
| B11 SWIR-1 | 1610 nm | 20 m | bare soil is bright, moist canopy is dark |
| B12 SWIR-2 | 2190 nm | 20 m | separates soil, burn scars, roads |

Infrared bands are what make this work: to the eye, a pasture and a forest are
both green, but in NIR + SWIR they look very different.

**Level-2A.** We use L2A products: "bottom-of-atmosphere" surface reflectance
(the atmosphere's effect has already been removed) plus a **Scene Classification
Layer (SCL)** that marks clouds, cloud shadows, cirrus and no-data. The digital
numbers are converted to reflectance with `(DN − 1000) / 10000` (ESA added the
+1000 offset from processing baseline 04.00 in January 2022 — handled
automatically in `load_scene`).

**How we download it — STAC + COG (`src/data/sentinel.py`).**
* **STAC** (SpatioTemporal Asset Catalog) is a standard JSON API for searching
  satellite data by area, date and cloud cover. We query Microsoft's free
  **Planetary Computer** catalogue with `pystac-client`; `planetary-computer`
  signs the URLs (no account needed).
* Each scene is stored as **Cloud-Optimised GeoTIFFs (COGs)** — GeoTIFFs laid out
  so a client can fetch just the bytes for one window. So instead of downloading a
  ~800 MB 110 × 110 km tile, `rasterio` streams *only the pixels inside your box*.
* `WarpedVRT` re-projects each band on the fly from the scene's UTM zone to
  plain longitude/latitude (EPSG:4326) and resamples the 20 m bands to 10 m, so all
  six bands and both dates line up pixel-for-pixel.
* We search ±45 days around each requested date and pick the clearest scene.

---

## 3. Why there is a synthetic data generator

Training a segmentation model needs **pixel-level labels** ("this pixel is
forest"). Those are expensive to make and gigabytes to download, which makes a
project hard to run on a laptop.

`src/data/synthetic.py` procedurally generates landscapes — forest patches,
pastures, bare soil, rivers, roads, regrowth, the Amazon-style "fishbone"
clearing pattern, clouds and cloud shadows — and renders them with **realistic
reflectance values for each of the 6 bands** (forest: red ≈ 0.03, NIR ≈ 0.32;
bare soil: SWIR-1 ≈ 0.32, etc.). Because we generate the land cover ourselves,
we know the exact truth, so we can:

* train the model offline in ~20 minutes on a CPU,
* **score** the change detector against the true deforestation (shown in the GUI).

Honest caveat: synthetic imagery is cleaner than reality, so the very high
scores on it are an upper bound. For real-world use, fine-tune on real scenes
(section 9).

Clouds get label **255 = ignore**: the loss function skips those pixels, so the
model is never punished for not seeing through clouds.

---

## 4. The U-Net model (`src/models/unet.py`)

**Semantic segmentation** = classify every pixel. A U-Net (Ronneberger et al.,
2015, originally for medical images) is the standard architecture for it.

```
 Encoder (what?)                          Decoder (where?)
 6×128×128 → conv → 16 ─────skip─────────► concat → conv → 16 → 1×1 conv → 1×128×128
              pool → 32 ───skip──────► concat → conv → 32 ↑
                     pool → 64 ─skip─► concat → conv → 64 ↑
                            pool → 128 ► concat → conv →128 ↑
                                   pool → 256 (bottleneck) ↑ up-conv
```

* **Encoder:** repeated `3×3 conv → BatchNorm → ReLU` blocks with max-pooling.
  Each level halves the resolution and doubles the channels, so the network sees
  larger context (is this green patch a forest or a field?).
* **Bottleneck:** the most abstract features, with **Dropout** for regularisation.
* **Decoder:** transposed convolutions up-sample back to full resolution.
* **Skip connections:** encoder features are concatenated into the decoder at the
  same level. This is the "U" — it restores fine detail, so clearing edges and
  10-m-wide roads stay sharp.
* **Output:** one logit per pixel → `sigmoid` → forest probability 0–1.

~1.9 million parameters (base width 16, depth 4) — small enough to train on a CPU.

**Building blocks explained**

| Piece | What it does |
|------|--------------|
| Convolution 3×3 | learns small spatial patterns (edges, textures) across all bands |
| BatchNorm | normalises activations → faster, more stable training |
| ReLU | non-linearity `max(0, x)` |
| MaxPool 2×2 | down-samples, enlarges receptive field |
| ConvTranspose 2×2 | learnable up-sampling |
| Dropout2d | randomly drops whole feature maps → less over-fitting |

---

## 5. Training (`src/train.py`)

| Choice | Why |
|-------|-----|
| **Input normalisation** — per-band mean/std from the training set, stored in the checkpoint | bands have very different ranges (NIR ≈ 0.3, blue ≈ 0.03) |
| **Augmentation** — random 90° rotations, flips, brightness jitter | satellite images have no "up"; jitter imitates haze/season |
| **Loss = 0.5·BCE + 0.5·Dice** | BCE gives well-calibrated per-pixel probabilities; Dice directly optimises overlap and copes with tiles that are 95 % forest |
| **AdamW** optimiser | Adam with decoupled weight decay |
| **One-Cycle learning-rate schedule** | warm up then anneal → fast convergence in few epochs |
| **Best-checkpoint saving by validation IoU** | keeps the model that generalises best, not the last one |

**Metrics.** With TP/FP/FN/TN counted over non-ignored pixels:
* **IoU** (Intersection over Union) = TP / (TP + FP + FN) — the main segmentation metric.
* **F1 / Dice** = 2·P·R / (P + R).
* **Precision** = of pixels called forest, how many are; **Recall** = of real forest, how much we found.

`src/evaluate.py` scores the held-out test set and draws the confusion matrix and
precision–recall curve.

---

## 6. Inference on whole scenes (`src/predict.py`)

A real area is e.g. 1500 × 1500 px. The model slides a 128 px window with
32 px overlap across it, and each window's prediction is multiplied by a
**Hann (cosine) weight** that is 1 in the centre and ~0 at the edges before being
averaged. Edge pixels have less context and are less reliable; blending removes the
visible seams of naive tiling.

---

## 7. Change detection (`src/change_detection.py`)

This is **post-classification comparison**:

1. `F1 = p_before ≥ 0.5`, `F2 = p_after ≥ 0.5`
2. **Loss** = `F1 AND NOT F2`; **regrowth** = `NOT F1 AND F2`
3. **NDVI confirmation:** NDVI = (NIR − Red)/(NIR + Red). A loss pixel must also
   have lost ≥ 0.15 NDVI. Two independent signals must agree → far fewer false alarms.
4. **Cloud masking:** pixels that are cloud/shadow on *either* date are excluded.
5. **Morphological opening** (erode then dilate with a disk) removes 1–2 px noise.
6. **Minimum mapping unit** — connected patches smaller than 0.1 ha are dropped.
7. **Vectorise:** `scipy.ndimage.label` groups pixels into patches; `rasterio.features.shapes`
   turns them into polygons in lon/lat; `shapely` merges and simplifies them. Each
   polygon gets **area (ha)**, centroid, dates and a **confidence** (mean drop in forest
   probability).

Area: each pixel's ground area is computed from the bbox and latitude
(a degree of longitude shrinks by cos(latitude)), so hectares are correct anywhere.

Output is **GeoJSON**, the standard web format for vector geodata — open it in QGIS,
Google Earth, geojson.io or any GIS.

---

## 8. The GUI — FastAPI + Leaflet

**Backend (`app/server.py`) — FastAPI**, a modern Python web framework that turns
type-annotated functions into a REST API, validates input with **Pydantic** and is
served by **Uvicorn** (an ASGI server). Analyses run in a background thread pool so
the browser stays responsive; it polls `/api/jobs/{id}` to stream the log.

**Frontend (`app/static/`) — vanilla JS + Leaflet.**
* **Leaflet** is the most widely used open-source web-map library. Satellite/OSM
  basemaps come as tiles; our results are added as `imageOverlay` PNGs positioned by
  the scene's bounds, and polygons as `L.geoJSON`.
* **Leaflet.draw** provides the rectangle tool for drawing an area of interest.
* Features: run picker, date-interval picker, layer toggles (before/after true colour,
  U-Net forest map, loss, regrowth, outlines), opacity, **swipe comparison** (drag a
  divider to wipe between before and after — implemented with CSS `clip-path`),
  an SVG forest-area/loss chart, a clickable table of the largest clearings that
  zooms the map, accuracy vs truth for the demo, and GeoJSON download.
* Leaflet is vendored (copied into the repo), so the app has no build step and
  doesn't depend on a CDN.

---

## 9. Using real data and fine-tuning

```bash
# 1. analyse any area (needs internet)
python -m src.pipeline --bbox -63.20 -10.00 -63.06 -9.88 --dates 2021-07-20 2024-07-20 --name Rondonia

# 2. build a real training set with automatic NDVI weak labels …
python -m src.data.build_real_dataset --bbox -63.2 -10.0 -63.06 -9.88 --dates 2023-07-15 2024-07-15 --out data/real
#    … or with a real forest mask GeoTIFF (Hansen GFC, ESA WorldCover, your own labels)
python -m src.data.build_real_dataset --bbox ... --dates ... --labels forest_mask.tif --out data/real

# 3. fine-tune the synthetic-pretrained model on it
python -m src.train --data data/real --resume models/unet_forest.pt --epochs 10
```

**Weak labels:** confident forest = NDVI > 0.70 & SWIR-1 < 0.20; confident
non-forest = NDVI < 0.45; everything in between is "ignore". The U-Net learns from
the confident pixels and its spatial context lets it generalise to the uncertain ones.

---

## 10. Continuous monitoring (`src/monitor.py`)

Because Sentinel-2 revisits every ~5 days, `monitor.py` can run on a schedule
(cron / Task Scheduler / GitHub Actions). For each area in `watchlist.yaml` it finds
the newest clear scene, compares it with the stored baseline, writes a normal run
(visible in the GUI), appends an alert to `outputs/alerts.jsonl` if the cleared area
exceeds the area's `alert_ha`, and moves the baseline forward.

---

## 11. Limitations (worth knowing)

* Synthetic training data is easier than reality; expect lower real-world scores until fine-tuned.
* Optical satellites can't see through clouds; in the wet tropics, pick dry-season dates.
  Radar (Sentinel-1) would solve this and is a natural extension.
* "Forest" here means dense canopy; plantations and secondary regrowth can be confused.
* Degradation (selective logging under an intact canopy) is much harder to see than clear-cutting.
