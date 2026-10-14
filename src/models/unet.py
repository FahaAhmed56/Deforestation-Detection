"""U-Net for per-pixel forest / non-forest segmentation.

Architecture (depth=4, base=16):

  input 6×128×128
   ├─ enc0  16 ──────────────────────────────────────────┐ skip
   │   └─ pool → enc1  32 ───────────────────────────┐   │
   │         └─ pool → enc2  64 ────────────────┐    │   │
   │               └─ pool → enc3 128 ─────┐    │    │   │
   │                     └─ pool → bottleneck 256  │    │   │
   │                     up + concat ◄──────┘    │    │   │
   │               up + concat ◄─────────────────┘    │   │
   │         up + concat ◄────────────────────────────┘   │
   │   up + concat ◄──────────────────────────────────────┘
   └─ 1×1 conv → 1 logit per pixel (sigmoid → forest probability)

The encoder learns *what* is in the image; the skip connections carry the
fine spatial detail back to the decoder so clearing boundaries stay sharp.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """(3×3 conv → BatchNorm → ReLU) × 2"""

    def __init__(self, cin, cout, dropout=0.0):
        layers = [
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    def __init__(self, in_channels=6, n_classes=1, base_channels=16, depth=4, dropout=0.1):
        super().__init__()
        chs = [base_channels * 2 ** i for i in range(depth + 1)]
        self.down = nn.ModuleList()
        cin = in_channels
        for c in chs[:-1]:
            self.down.append(ConvBlock(cin, c))
            cin = c
        self.bottleneck = ConvBlock(chs[-2], chs[-1], dropout)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.dec.append(ConvBlock(c * 2, c))
        self.head = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for block in self.down:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:          # odd sizes
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
