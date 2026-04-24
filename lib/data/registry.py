"""Dataset registry — factory for creating DataModules by name.

Usage::

    from lib.data import create_datamodule

    dm = create_datamodule("visa", dataset_root="datasets/visa", category="candle", ...)
    dm = create_datamodule("m2ad", dataset_root="datasets/m2ad", category="Bird", ...)
"""

from __future__ import annotations

from typing import Any

import albumentations as A

from .m2ad_datamodule import M2ADDataModule
from .visa_datamodule import VisADataModule

# Maps dataset name → (DataModule class, default root).
_REGISTRY: dict[str, tuple[type, str]] = {
    "visa": (VisADataModule, "datasets/visa"),
    "m2ad": (M2ADDataModule, "datasets/m2ad"),
}

DATASET_NAMES: list[str] = sorted(_REGISTRY.keys())


def create_datamodule(
    dataset: str,
    *,
    dataset_root: str | None = None,
    category: str | list[str] = "",
    image_size: int = 256,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform: A.Compose | None = None,
    eval_transform: A.Compose | None = None,
    mask_transform: A.Compose | None = None,
    **extra: Any,
) -> VisADataModule | M2ADDataModule:
    """Instantiate the DataModule for *dataset*.

    Parameters
    ----------
    dataset:
        One of :data:`DATASET_NAMES` (``"visa"`` or ``"m2ad"``).
    dataset_root:
        Override the default data directory.  Falls back to a sensible
        per-dataset default when *None*.
    """
    key = dataset.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose from {DATASET_NAMES}.")

    cls, default_root = _REGISTRY[key]
    root = dataset_root or default_root

    return cls(
        dataset_root=root,
        category=category,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        eval_transform=eval_transform,
        mask_transform=mask_transform,
    )
