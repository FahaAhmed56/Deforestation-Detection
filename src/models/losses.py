"""Loss and metrics that respect the 'ignore' label (255 = cloud / no-data)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE = 255


def bce_dice_loss(logits, target, dice_weight=0.5):
    """BCE handles per-pixel calibration; Dice handles class imbalance
    (a tile may be 95% forest) and directly optimises region overlap."""
    logits = logits.squeeze(1)
    valid = target != IGNORE
    t = target.clone().float()
    t[~valid] = 0
    bce = F.binary_cross_entropy_with_logits(logits[valid], t[valid])
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return (1 - dice_weight) * bce + dice_weight * dice


@torch.no_grad()
def confusion(logits, target, threshold=0.5):
    """Returns tp, fp, fn, tn counts over valid pixels."""
    pred = torch.sigmoid(logits.squeeze(1)) >= threshold
    valid = target != IGNORE
    t = target == 1
    tp = (pred & t & valid).sum().item()
    fp = (pred & ~t & valid).sum().item()
    fn = (~pred & t & valid).sum().item()
    tn = (~pred & ~t & valid).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn):
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    return {
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + eps),
        "iou": tp / (tp + fp + fn + eps),
    }
