"""End-to-end pipeline: imagery → forest maps → change detection → map layers.

    # offline demo (synthetic Rondônia-style frontier, 4 yearly dates)
    python -m src.pipeline --demo

    # real Sentinel-2 imagery for any area (needs internet)
    python -m src.pipeline --bbox -63.20 -10.00 -63.05 -9.88 --dates 2023-07-15 2025-07-15 --name "Rondonia"

Each run writes outputs/runs/<run_id>/ containing:
    run.json                    metadata + statistics the GUI reads
    <date>_rgb.png              true-colour image
    <date>_forest.png           forest probability overlay
    <d1>_<d2>_loss.png/gain.png change overlays
    <d1>_<d2>_changes.geojson   cleared-area polygons (open in QGIS / geojson.io)
    summary_<d1>_<d2>.png       before/after figure
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src import plots
from src.change_detection import detect_change, score_against_truth, vectorize
from src.config import load_config, project_path
from src.geo import (bbox_from_center, ndvi, pixel_area_m2, save_mask_png, save_prob_png,
                     save_rgb_png, transform_for)

DEMO = {"name": "Demo · Rondônia-style forest frontier (synthetic)", "lat": -9.95, "lon": -63.10, "size": 512}


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "run"


def demo_scenes():
    from src.data.synthetic import make_demo_timeseries
    series, true_loss = make_demo_timeseries(size=DEMO["size"])
    bbox = bbox_from_center(DEMO["lat"], DEMO["lon"], DEMO["size"], DEMO["size"])
    scenes = [{"date": s["date"], "image": s["image"], "valid": ~s["cloud"],
               "info": {"scene_id": f"SYNTHETIC_{s['date']}", "acquired": s["date"],
                        "aoi_invalid_pct": round(100 * s["cloud"].mean(), 1)}} for s in series]
    return bbox, scenes, true_loss


def sentinel_scenes(bbox, dates, log=print):
    from src.data.sentinel import fetch
    scenes = []
    for d in dates:
        log(f"Searching Sentinel-2 near {d} …")
        img, valid, info = fetch(bbox, d)
        log(f"  {info['scene_id']} ({info['acquired']}, {info['aoi_invalid_pct']}% cloud in AOI)")
        scenes.append({"date": info["acquired"], "image": img, "valid": valid, "info": info})
    return scenes


def run_analysis(model, bbox, scenes, name, source, true_loss=None, log=print, run_id=None):
    cfg = load_config()["change"]
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{_slug(name)}"
    out = project_path("runs_dir") / run_id
    out.mkdir(parents=True, exist_ok=True)
    shape = scenes[0]["image"].shape[1:]
    tr, px_m2 = transform_for(bbox, shape), pixel_area_m2(bbox, shape)
    min_px = max(1, int(round(cfg["min_area_ha"] * 10_000 / px_m2)))

    meta = {"id": run_id, "name": name, "source": source, "bbox": list(bbox),
            "bounds": [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],    # Leaflet [[S,W],[N,E]]
            "shape": list(shape), "pixel_area_m2": round(px_m2, 2),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model.meta, "params": cfg, "scenes": [], "changes": []}

    probs, ndvis = [], []
    for s in scenes:
        t0 = time.time()
        p = model.predict(s["image"])
        probs.append(p); ndvis.append(ndvi(s["image"]))
        d = s["date"]
        save_rgb_png(s["image"], out / f"{d}_rgb.png")
        save_prob_png(p, out / f"{d}_forest.png", s["valid"])
        forest_px = int(((p >= cfg["forest_threshold"]) & s["valid"]).sum())
        forest_pct = 100 * forest_px / max(s["valid"].sum(), 1)
        meta["scenes"].append({
            "date": d, "rgb": f"{d}_rgb.png", "forest": f"{d}_forest.png",
            # forest share of the cloud-free pixels, scaled to the whole AOI, so a
            # cloudy date doesn't look like deforestation
            "forest_ha": round(forest_pct / 100 * shape[0] * shape[1] * px_m2 / 1e4, 1),
            "forest_ha_observed": round(forest_px * px_m2 / 1e4, 1),
            "forest_pct": round(forest_pct, 1),
            "mean_ndvi": round(float(ndvis[-1][s["valid"]].mean()), 3), **s["info"]})
        log(f"Forest map {d}: {meta['scenes'][-1]['forest_pct']}% forest  ({time.time() - t0:.1f}s)")

    for i in range(1, len(scenes)):
        a, b = scenes[i - 1], scenes[i]
        valid = a["valid"] & b["valid"]
        ch = detect_change(probs[i - 1], probs[i], ndvis[i - 1], ndvis[i], valid,
                           cfg["forest_threshold"], cfg["ndvi_drop"], min_px, cfg["open_radius"])
        key = f"{a['date']}_{b['date']}"
        save_mask_png(ch["loss"], out / f"{key}_loss.png", (229, 57, 53), 200)
        save_mask_png(ch["gain"], out / f"{key}_gain.png", (30, 136, 229), 180)
        props = {"from": a["date"], "to": b["date"]}
        gj = vectorize(ch["loss"], tr, px_m2, props, score=probs[i - 1] - probs[i], kind="loss")
        gj["features"] += vectorize(ch["gain"], tr, px_m2, props, score=probs[i] - probs[i - 1],
                                    kind="gain")["features"]
        (out / f"{key}_changes.geojson").write_text(json.dumps(gj))
        loss_ha = round(ch["loss"].sum() * px_m2 / 1e4, 2)
        entry = {"from": a["date"], "to": b["date"], "loss": f"{key}_loss.png", "gain": f"{key}_gain.png",
                 "geojson": f"{key}_changes.geojson", "loss_ha": loss_ha,
                 "gain_ha": round(ch["gain"].sum() * px_m2 / 1e4, 2),
                 "loss_patches": sum(f["properties"]["type"] == "loss" for f in gj["features"]),
                 "largest_patch_ha": max([f["properties"]["area_ha"] for f in gj["features"]
                                          if f["properties"]["type"] == "loss"], default=0),
                 "loss_pct_of_forest": round(100 * ch["loss"].sum() / max((ch["forest_t1"] & valid).sum(), 1), 2),
                 "summary": f"summary_{key}.png"}
        if true_loss is not None:
            entry["accuracy_vs_truth"] = score_against_truth(ch["loss"], true_loss[i - 1], valid)
        meta["changes"].append(entry)
        plots.change_summary(a["image"], b["image"], probs[i - 1], probs[i], ch["loss"], ch["gain"],
                             a["date"], b["date"], out / entry["summary"],
                             truth=true_loss[i - 1] if true_loss is not None else None)
        log(f"Change {a['date']} → {b['date']}: {loss_ha} ha lost in {entry['loss_patches']} patches")

    meta["totals"] = {
        "loss_ha": round(sum(c["loss_ha"] for c in meta["changes"]), 2),
        "gain_ha": round(sum(c["gain_ha"] for c in meta["changes"]), 2),
        "forest_start_ha": meta["scenes"][0]["forest_ha"],
        "forest_end_ha": meta["scenes"][-1]["forest_ha"],
    }
    if len(scenes) > 1:
        plots.timeline([s["date"] for s in meta["scenes"]], [s["forest_ha"] for s in meta["scenes"]],
                       [c["loss_ha"] for c in meta["changes"]], out / "timeline.png")
        meta["timeline"] = "timeline.png"
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    log(f"Done → {out}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run on the offline synthetic demo scene")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--dates", nargs="+", help="two or more dates YYYY-MM-DD")
    ap.add_argument("--name", default="Custom area")
    ap.add_argument("--checkpoint", default=None)
    a = ap.parse_args(argv)

    from src.predict import ForestModel
    model = ForestModel(a.checkpoint)
    if a.demo:
        bbox, scenes, truth = demo_scenes()
        run_analysis(model, bbox, scenes, DEMO["name"], "demo", truth, run_id="demo")
    else:
        if not a.bbox or not a.dates or len(a.dates) < 2:
            ap.error("give --demo, or --bbox W S E N and at least two --dates")
        bbox = tuple(a.bbox)
        run_analysis(model, bbox, sentinel_scenes(bbox, a.dates), a.name, "sentinel-2")


if __name__ == "__main__":
    main()
