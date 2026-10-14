"""Train the U-Net forest segmentation model.

    python -m src.train                      # synthetic tiles in data/processed
    python -m src.train --data data/real     # tiles from build_real_dataset
    python -m src.train --epochs 30 --resume models/unet_forest.pt

Outputs
    models/unet_forest.pt          best checkpoint (by validation IoU)
    models/history.json            per-epoch metrics
    docs/images/training_curves.png
    docs/images/sample_predictions.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ROOT, load_config, project_path
from src.data.dataset import TileDataset, compute_stats, list_tiles
from src.models.losses import bce_dice_loss, confusion, metrics_from_counts
from src.models.unet import UNet, count_parameters
from src import plots


def run_epoch(model, loader, optimizer=None, device="cpu", scheduler=None):
    train = optimizer is not None
    model.train(train)
    total, n, counts = 0.0, 0, np.zeros(4)
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:      # One-Cycle LR is stepped every batch
                    scheduler.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
            counts += confusion(logits, y)
    m = metrics_from_counts(*counts)
    m["loss"] = total / max(n, 1)
    return m


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(project_path("data_dir") / "processed"),
                    help="folder containing train/ val/ (test/) sub-folders of .npz tiles")
    ap.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
    ap.add_argument("--batch-size", type=int, default=cfg["train"]["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["train"]["lr"])
    ap.add_argument("--out", default=str(project_path("models_dir") / "unet_forest.pt"))
    ap.add_argument("--resume", default=None, help="fine-tune from an existing checkpoint")
    a = ap.parse_args(argv)

    torch.manual_seed(cfg["dataset"]["seed"])
    np.random.seed(cfg["dataset"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, torch.get_num_threads()))

    data = Path(a.data)
    train_files, val_files = list_tiles(data / "train"), list_tiles(data / "val")
    if not train_files:
        raise SystemExit(f"No tiles in {data/'train'} — run `python -m src.data.prepare_data` first.")

    ckpt = torch.load(a.resume, map_location=device, weights_only=False) if a.resume else None
    mean, std = (ckpt["mean"], ckpt["std"]) if ckpt else compute_stats(train_files)

    mc = cfg["model"]
    model = UNet(len(cfg["bands"]), 1, mc["base_channels"], mc["depth"], mc["dropout"]).to(device)
    if ckpt:
        model.load_state_dict(ckpt["state_dict"])
    print(f"device={device}  params={count_parameters(model):,}  train={len(train_files)}  val={len(val_files)}")

    tl = DataLoader(TileDataset(train_files, mean, std, augment=True), a.batch_size, shuffle=True,
                    num_workers=cfg["train"]["num_workers"])
    vl = DataLoader(TileDataset(val_files, mean, std), a.batch_size, num_workers=cfg["train"]["num_workers"])

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=cfg["train"]["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))

    history, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, tl, opt, device, sched)
        va = run_epoch(model, vl, None, device)
        row = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1),
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        flag = ""
        if va["iou"] > best:
            best = va["iou"]
            torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                        "bands": cfg["bands"], "model_cfg": mc, "epoch": ep, "val": va}, a.out)
            flag = "  ✓ saved"
        print(f"epoch {ep:2d}/{a.epochs}  loss {tr['loss']:.4f}/{va['loss']:.4f}  "
              f"val IoU {va['iou']:.4f}  F1 {va['f1']:.4f}  acc {va['accuracy']:.4f}  "
              f"({row['time_s']}s){flag}")

    out_dir = Path(a.out).parent
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    img = project_path("images_dir")
    plots.training_curves(history, img / "training_curves.png")

    best_ckpt = torch.load(a.out, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["state_dict"])
    plots.sample_predictions(model, val_files[:6], mean, std, img / "sample_predictions.png", device)
    print(f"best val IoU {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
