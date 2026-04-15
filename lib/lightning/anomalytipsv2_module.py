from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningModule

from lib.models.anomalytipsv2 import AnomalyTIPSv2
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class AnomalyTIPSv2Module(LightningModule):
    """Lightning wrapper around AnomalyTIPSv2.

    Parameters
    ----------
    model_name : str
        TIPSv2 variant, e.g. 'google/tipsv2-b14', 'google/tipsv2-l14'.
    smaller_edge_size : int
        Resize shorter edge to this size.
    masking : bool
        Apply PCA-based foreground masking.
    rotation : bool
        Augment references with rotations.
    masking_threshold : float
        PCA threshold for masking.
    gaussian_sigma : float
        Gaussian smoothing sigma for anomaly maps.
    top_percent : float
        Fraction of top patch distances for image-level scoring.
    image_size : int
        Output anomaly map size.
    imagenet_input_normalized : bool
        Whether incoming batch["image"] tensors are ImageNet-normalized.
        If True, they are de-normalized back to [0,1] for TIPSv2.
    """

    def __init__(
        self,
        model_name: str = "google/tipsv2-b14",
        smaller_edge_size: int = 448,
        masking: bool = True,
        rotation: bool = True,
        masking_threshold: float = 10.0,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        image_size: int = 448,
        imagenet_input_normalized: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.model = AnomalyTIPSv2(
            model_name=model_name,
            smaller_edge_size=smaller_edge_size,
            masking=masking,
            rotation=rotation,
            masking_threshold=masking_threshold,
            gaussian_sigma=gaussian_sigma,
            top_percent=top_percent,
        )

        self._image_size = image_size
        self._imagenet_input_normalized = imagenet_input_normalized

        self._train_features: list[np.ndarray] = []
        self._train_image_count: int = 0

        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _undo_imagenet_norm(images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

    def _to_tips_input(self, images: torch.Tensor) -> torch.Tensor:
        if self._imagenet_input_normalized:
            return self._undo_imagenet_norm(images)
        return images.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._to_tips_input(x)
        return self.model.predict_batch_tensor(
            x,
            output_size=(self._image_size, self._image_size),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if self.model._fitted:
            return

        images = self._to_tips_input(batch["image"])

        for i in range(images.shape[0]):
            img_t = images[i].detach().cpu()

            # Rotation augmentation operates on numpy uint8.
            img_np = (
                (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            )

            variants = (
                self.model.augment_reference(img_np)
                if self.model.rotation
                else [img_np]
            )

            for variant in variants:
                img_tensor, _ = self.model.prepare_image(variant)
                features = self.model.extract_features(img_tensor)
                self._train_features.append(features)

            self._train_image_count += 1

    def on_train_epoch_end(self) -> None:
        if self.model._fitted or not self._train_features:
            return

        all_features = np.concatenate(self._train_features, axis=0).astype(np.float32)
        norms = np.linalg.norm(all_features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self.model._memory_bank = all_features / norms
        self.model._fitted = True

        n_patches = self.model._memory_bank.shape[0]
        n_images = self._train_image_count

        self._train_features.clear()
        self._train_image_count = 0

        print(
            f"AnomalyTIPSv2: memory bank built — {n_patches} patches "
            f"from {n_images} reference images "
            f"(masking={self.model.masking}, rotation={self.model.rotation}, "
            f"backbone='{self.model.model_name}')"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_labels.clear()
        self._val_scores.clear()
        self._val_masks.clear()
        self._val_anomaly_maps.clear()

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if not self.model._fitted:
            return

        x = self._to_tips_input(batch["image"])
        scores, anomaly_maps = self.model.predict_batch_tensor(
            x,
            output_size=(self._image_size, self._image_size),
        )

        self._val_labels.append(batch["label"].cpu())
        self._val_scores.append(scores.cpu())
        self._val_masks.append(batch["mask"].cpu())
        self._val_anomaly_maps.append(anomaly_maps.squeeze(1).cpu())

    def _log_metrics(self, stage: str) -> None:
        if not self._val_labels:
            return

        labels = torch.cat(self._val_labels)
        scores = torch.cat(self._val_scores)
        masks = torch.cat(self._val_masks)
        anomaly_maps = torch.cat(self._val_anomaly_maps)

        image_auroc = compute_auroc(labels, scores)
        self.log(f"{stage}/image_auroc", image_auroc, prog_bar=True)

        if masks.sum() > 0:
            pixel_auroc = compute_pixel_auroc(masks, anomaly_maps)
            self.log(f"{stage}/pixel_auroc", pixel_auroc, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> None:  # type: ignore[override]
        return None

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_dir: str | Path) -> None:
        """Persist the memory bank and hparams to ckpt_dir."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.model._memory_bank is not None:
            np.save(ckpt_dir / "memory_bank.npy", self.model._memory_bank)

        state = {
            "hparams": dict(self.hparams),
            "fitted": self.model._fitted,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "AnomalyTIPSv2Module":
        """Restore an AnomalyTIPSv2 module from *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.ckpt", map_location=map_location)
        module = cls(**state["hparams"])
        if state["fitted"]:
            bank_path = ckpt_dir / "memory_bank.npy"
            module.model._memory_bank = np.load(bank_path)
            module.model._fitted = True
        return module
