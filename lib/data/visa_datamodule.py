"""Lightning DataModule for the VisA anomaly detection dataset."""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import (
    get_eval_transforms,
    get_mask_transforms,
    get_train_transforms,
)


class VisADataset(Dataset):
    """Single-split dataset backed by the VisA split CSV."""

    def __init__(
        self,
        samples: pd.DataFrame,
        dataset_root: Path,
        transform: A.Compose,
        mask_transform: A.Compose | None = None,
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples.iloc[idx]

        image_path = self.dataset_root / row["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = 0 if row["label"] == "normal" else 1

        # Apply transforms to image
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        result: dict[str, torch.Tensor] = {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }

        # Load mask if available (anomalous samples)
        mask_value = row.get("mask", "")
        if pd.notna(mask_value) and str(mask_value).strip():
            mask_path = self.dataset_root / str(mask_value)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if self.mask_transform:
                    mask = self.mask_transform(image=mask)["image"]
                mask = torch.from_numpy(np.asarray(mask)).float()
                # Normalise to [0, 1] — VisA masks may be {0,1} or {0,255}.
                if mask.max() > 1:
                    mask = mask / 255.0
            else:
                mask = torch.zeros(image_tensor.shape[1:], dtype=torch.float32)
        else:
            mask = torch.zeros(image_tensor.shape[1:], dtype=torch.float32)

        result["mask"] = mask
        return result


class VisADataModule:
    """Lightning-style DataModule for the VisA dataset.

    Uses ``split_csv/1cls.csv`` which provides pre-defined train / test splits
    with columns: ``object, split, label, image, mask``.
    """

    def __init__(
        self,
        dataset_root: str | Path = "datasets/visa",
        category: str = "candle",
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        split_csv: str = "split_csv/1cls.csv",
        train_transform: A.Compose | None = None,
        eval_transform: A.Compose | None = None,
        mask_transform: A.Compose | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.category = category
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_csv = split_csv
        self._custom_train_transform = train_transform
        self._custom_eval_transform = eval_transform
        self._custom_mask_transform = mask_transform

        self._train_dataset: VisADataset | None = None
        self._val_dataset: VisADataset | None = None
        self._test_dataset: VisADataset | None = None

    # ------------------------------------------------------------------
    # Lightning DataModule interface
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        csv_path = self.dataset_root / self.split_csv
        df = pd.read_csv(csv_path)
        df = df[df["object"] == self.category]

        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        # Validation = 10 % of normal training images + 10 % of test anomalies.
        # This gives a balanced val set that can evaluate both image- and pixel-AUROC.
        val_normal = train_df.sample(frac=0.1, random_state=42)
        train_df = train_df.drop(val_normal.index)

        test_anomalous = test_df[test_df["label"] != "normal"]
        val_anomalous = test_anomalous.sample(frac=0.1, random_state=42)
        test_df = test_df.drop(val_anomalous.index)

        val_df = pd.concat([val_normal, val_anomalous], ignore_index=True)

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
            self._train_dataset = VisADataset(
                train_df, self.dataset_root, train_transform, mask_transform
            )
            self._val_dataset = VisADataset(
                val_df, self.dataset_root, eval_transform, mask_transform
            )
        if stage in ("test", None):
            self._test_dataset = VisADataset(
                test_df, self.dataset_root, eval_transform, mask_transform
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
        """List all available categories in the dataset."""
        csv_path = self.dataset_root / self.split_csv
        df = pd.read_csv(csv_path)
        return sorted(df["object"].unique().tolist())
