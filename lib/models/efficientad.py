"""EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.

WACV 2024 — Batzner, Heckler, König
Reference: https://arxiv.org/abs/2303.14535
Official code: https://github.com/nelson1425/EfficientAd

Architectures
-------------
- **PDN-S / PDN-M**: Patch Description Networks (Table 1).  Lightweight CNNs
  that map a 256×256 RGB image to a 56×56 feature map with *out_channels*
  channels.  Used as both the teacher and the student (the student outputs
  2× *out_channels* — first half for the student-teacher path, second half
  for the student-autoencoder path).
- **Autoencoder**: Strided-conv encoder → bilinear-upsample decoder with
  dropout.  Hard-coded spatial sizes match a 256×256 input → 56×56 output.
"""

from __future__ import annotations

import torch.nn as nn


def get_pdn_small(out_channels: int = 384, padding: bool = False) -> nn.Sequential:
    """Patch Description Network — Small variant (Table 1)."""
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(3, 128, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(128, 256, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(256, 256, kernel_size=3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, out_channels, kernel_size=4),
    )


def get_pdn_medium(out_channels: int = 384, padding: bool = False) -> nn.Sequential:
    """Patch Description Network — Medium variant (Table 1)."""
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(3, 256, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(256, 512, kernel_size=4, padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(512, 512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
    )


def get_autoencoder(out_channels: int = 384) -> nn.Sequential:
    """EfficientAd autoencoder for logical anomaly detection (Section 3.3).

    Hard-coded spatial sizes in the decoder assume a 256×256 input image
    producing 56×56 feature maps, matching the PDN output resolution.
    """
    return nn.Sequential(
        # encoder
        nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode="bilinear"),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
    )
