"""Albumentations-based transform pipelines for anomaly detection."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """Training transforms with augmentation (applied to normal images only)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5
            ),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int = 256) -> A.Compose:
    """Deterministic transforms for validation / test."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_mask_transforms(image_size: int = 256) -> A.Compose:
    """Transforms for segmentation masks (resize only, no normalization)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=0),  # INTER_NEAREST
        ]
    )
