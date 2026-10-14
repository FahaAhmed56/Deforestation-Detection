"""All figures used in the README / docs (matplotlib, headless)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

FOREST = "#2e7d32"
LOSS = "#e53935"
GAIN = "#1e88e5"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


def true_color(x, gain=3.2, gamma=0.9):
    """(6,H,W) reflectance -> (H,W,3) displayable RGB (B04, B03, B02)."""
    rgb = np.stack([x[2], x[1], x[0]], -1).astype(np.float32)
    return np.clip((rgb * gain) ** gamma, 0, 1)


def training_curves(history, path):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    ax[0].plot(ep, [h["val_loss"] for h in history], "o-", label="validation")
    ax[0].set_title("Loss (BCE + Dice)"); ax[0].set_xlabel("epoch"); ax[0].legend()
    for k, c in [("iou", FOREST), ("f1", "#6a1b9a"), ("accuracy", "#ef6c00")]:
        ax[1].plot(ep, [h[f"val_{k}"] for h in history], "o-", color=c, label=f"val {k}")
    ax[1].plot(ep, [h["train_iou"] for h in history], "--", color=FOREST, alpha=.5, label="train iou")
    ax[1].set_title("Validation metrics"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_ylim(None, 1)
    ax[2].plot(ep, [h["lr"] for h in history], "o-", color="gray")
    ax[2].set_title("Learning rate (One-Cycle)"); ax[2].set_xlabel("epoch")
    fig.suptitle("U-Net forest segmentation — training", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


@torch.no_grad()
def sample_predictions(model, files, mean, std, path, device="cpu"):
    model.eval()
    n = len(files)
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 8))
    m, s = np.asarray(mean)[:, None, None], np.asarray(std)[:, None, None]
    for i, f in enumerate(files):
        d = np.load(f); x = d["x"].astype(np.float32); y = d["y"]
        p = torch.sigmoid(model(torch.from_numpy((x - m) / s).float()[None].to(device)))[0, 0].cpu().numpy()
        ax[0, i].imshow(true_color(x))
        lbl = np.ma.masked_where(y == 255, y)
        ax[1, i].imshow(lbl, cmap="Greens", vmin=0, vmax=1.3)
        ax[2, i].imshow(p, cmap="Greens", vmin=0, vmax=1)
        for r in range(3):
            ax[r, i].set_xticks([]); ax[r, i].set_yticks([])
    for r, t in enumerate(["Sentinel-2 true colour", "Ground truth (white = cloud, ignored)", "U-Net forest probability"]):
        ax[r, 0].set_ylabel(t, fontsize=9)
    fig.suptitle("Validation tiles: input → label → prediction", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def confusion_matrix(counts, path):
    tp, fp, fn, tn = counts
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    pct = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(pct, cmap="Greens", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{pct[i, j]:.1%}\n{int(cm[i, j]):,} px", ha="center", va="center",
                    color="white" if pct[i, j] > .5 else "black")
    ax.set_xticks([0, 1], ["non-forest", "forest"]); ax.set_yticks([0, 1], ["non-forest", "forest"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Test-set confusion matrix")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def pr_curve(precisions, recalls, thresholds, chosen, path):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.plot(recalls, precisions, color=FOREST)
    i = int(np.argmin(np.abs(np.asarray(thresholds) - chosen)))
    ax.scatter([recalls[i]], [precisions[i]], color=LOSS, zorder=3, label=f"threshold {chosen}")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("Precision–recall (test set)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def change_summary(before, after, p1, p2, loss, gain, d1, d2, path, truth=None):
    cols = 5 if truth is None else 6
    fig, ax = plt.subplots(1, cols, figsize=(3.6 * cols, 4))
    ax[0].imshow(true_color(before)); ax[0].set_title(f"Before · {d1}")
    ax[1].imshow(true_color(after)); ax[1].set_title(f"After · {d2}")
    ax[2].imshow(p1, cmap="Greens", vmin=0, vmax=1); ax[2].set_title("Forest prob. (before)")
    ax[3].imshow(p2, cmap="Greens", vmin=0, vmax=1); ax[3].set_title("Forest prob. (after)")
    ov = true_color(after).copy()
    ov[loss] = ov[loss] * .25 + np.array([.9, .1, .1]) * .75
    ov[gain] = ov[gain] * .4 + np.array([.1, .5, .95]) * .6
    ax[4].imshow(ov); ax[4].set_title("Detected loss (red) / gain (blue)")
    if truth is not None:
        t = np.zeros(truth.shape + (3,)); t[truth] = [.9, .1, .1]
        ax[5].imshow(t); ax[5].set_title("True loss (synthetic truth)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def timeline(dates, forest_ha, loss_ha, path):
    fig, ax1 = plt.subplots(figsize=(7, 3.6))
    ax1.plot(dates, forest_ha, "o-", color=FOREST, lw=2, label="forest area (ha)")
    ax1.set_ylabel("forest area (ha)", color=FOREST)
    ax2 = ax1.twinx()
    ax2.bar(dates[1:], loss_ha, color=LOSS, alpha=.6, width=.4, label="loss in interval (ha)")
    ax2.set_ylabel("loss since previous date (ha)", color=LOSS)
    ax2.spines["right"].set_visible(True)
    ax1.set_title("Forest cover over time")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
