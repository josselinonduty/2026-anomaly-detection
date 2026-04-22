"""Lightning DataModule for the M2AD anomaly detection dataset.

M2AD (Multi-View Multi-Illumination Anomaly Detection) uses this layout::

    M2AD/
    ├── Bird/
    │   ├── Good/       # Normal samples
    │   ├── GT/         # Ground-truth anomaly masks
    │   └── NG/         # Anomalous samples
    ├── Car/
    ... (10 categories total)

Each category contains images organised as
``{Good,NG}/{specimen}/{view}/{illumination}.png`` with corresponding
masks in ``GT/``.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import get_eval_transforms, get_mask_transforms, get_train_transforms

# Canonical M2AD categories (1024×1024 version).
M2AD_CATEGORIES = [
    "Bird",
    "Car",
    "Cube",
    "Dice",
    "Doll",
    "Holder",
    "Motor",
    "Ring",
    "Teapot",
    "Tube",
]


class M2ADDataset(Dataset):
    """Single-split dataset for M2AD (directory-based)."""

    def __init__(
        self,
        samples: list[dict],
        transform: A.Compose,
        mask_transform: A.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self.samples[idx]

        image = cv2.imread(str(entry["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Image not found: {entry['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = entry["label"]

        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        result: dict[str, torch.Tensor] = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }

        mask_path = entry.get("mask_path")
        if mask_path is not None and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if self.mask_transform:
                    mask = self.mask_transform(image=mask)["image"]
                mask = torch.from_numpy(np.asarray(mask)).float()
                if mask.max() > 1:
                    mask = mask / 255.0
            else:
                mask = torch.zeros(image_tensor.shape[1:], dtype=torch.float32)
        else:
            mask = torch.zeros(image_tensor.shape[1:], dtype=torch.float32)

        result["mask"] = mask
        return result


class M2ADDataModule:
    """Lightning-style DataModule for the M2AD dataset.

    Scans the directory tree for ``Good/`` (normal) and ``NG/`` (anomalous)
    images. Masks are loaded from ``GT/``.
    """

    def __init__(
        self,
        dataset_root: str | Path = "datasets/m2ad",
        category: str = "Bird",
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        train_transform: A.Compose | None = None,
        eval_transform: A.Compose | None = None,
        mask_transform: A.Compose | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._custom_train_transform = train_transform
        self._custom_eval_transform = eval_transform
        self._custom_mask_transform = mask_transform

        self._train_dataset: M2ADDataset | None = None
        self._val_dataset: M2ADDataset | None = None
        self._test_dataset: M2ADDataset | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scan_images(directory: Path) -> list[Path]:
        """Recursively collect image files under *directory*."""
        if not directory.exists():
            return []
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        return sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts)

    def _build_samples(self) -> tuple[list[dict], list[dict]]:
        """Return (normal_samples, anomalous_samples) for the category."""
        cat_dir = self.dataset_root / self.category

        good_dir = cat_dir / "Good"
        ng_dir = cat_dir / "NG"
        gt_dir = cat_dir / "GT"

        normal_samples: list[dict] = []
        for img_path in self._scan_images(good_dir):
            normal_samples.append(
                {"image_path": img_path, "label": 0, "mask_path": None}
            )

        anomalous_samples: list[dict] = []
        for img_path in self._scan_images(ng_dir):
            # Mask mirrors NG structure under GT/
            rel = img_path.relative_to(ng_dir)
            mask_path = gt_dir / rel
            anomalous_samples.append(
                {"image_path": img_path, "label": 1, "mask_path": mask_path}
            )

        return normal_samples, anomalous_samples

    # ------------------------------------------------------------------
    # Lightning DataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        normal_samples, anomalous_samples = self._build_samples()

        if not normal_samples:
            cat_dir = self.dataset_root / self.category
            raise FileNotFoundError(
                f"No images found for category '{self.category}' under "
                f"{cat_dir}. Download the M2AD dataset from "
                f"https://huggingface.co/datasets/ChengYuQi99/M2AD and "
                f"extract it into {self.dataset_root}."
            )

        # Validation: 10 % of normal + 10 % of anomalous
        rng = np.random.RandomState(42)

        val_n_count = max(1, int(len(normal_samples) * 0.1))
        val_n_idx = set(rng.choice(len(normal_samples), val_n_count, replace=False))
        train_normal = [s for i, s in enumerate(normal_samples) if i not in val_n_idx]
        val_normal = [s for i, s in enumerate(normal_samples) if i in val_n_idx]

        val_a_count = max(1, int(len(anomalous_samples) * 0.1)) if anomalous_samples else 0
        if val_a_count:
            val_a_idx = set(rng.choice(len(anomalous_samples), val_a_count, replace=False))
            test_anomalous = [s for i, s in enumerate(anomalous_samples) if i not in val_a_idx]
            val_anomalous = [s for i, s in enumerate(anomalous_samples) if i in val_a_idx]
        else:
            test_anomalous = anomalous_samples
            val_anomalous = []

        val_samples = val_normal + val_anomalous
        test_samples = train_normal + test_anomalous  # test includes normal + anomalous

        train_transform = self._custom_train_transform or get_train_transforms(
            self.image_size
        )
        eval_transform = self._custom_eval_transform or get_eval_transforms(
            self.image_size
        )
        mask_transform = self._custom_mask_transform or get_mask_transforms(
            self.image_size
        )

        if stage in ("fit", None):
            self._train_dataset = M2ADDataset(
                train_normal, train_transform, mask_transform
            )
            self._val_dataset = M2ADDataset(
                val_samples, eval_transform, mask_transform
            )
        if stage in ("test", None):
            self._test_dataset = M2ADDataset(
                test_samples, eval_transform, mask_transform
            )

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None, "Call setup('fit') first."
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None, "Call setup('test') first."
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def categories(self) -> list[str]:
        """List available categories by scanning subdirectories."""
        if self.dataset_root.exists():
            return sorted(
                d.name
                for d in self.dataset_root.iterdir()
                if d.is_dir() and (d / "Good").exists()
            )
        return list(M2AD_CATEGORIES)
