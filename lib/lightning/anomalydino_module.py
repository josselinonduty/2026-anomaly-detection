"""PyTorch Lightning module for AnomalyDINO anomaly detection.

AnomalyDINO is a training-free, memory-bank method that uses DINOv2 features
for few-shot anomaly detection.  The "training" phase extracts patch features
from normal reference images and builds a memory bank.  Anomaly scoring is
performed via nearest-neighbour cosine distance lookup.

Only **one** training epoch is needed (same as PatchCore).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningModule

from lib.models.anomalydino import AnomalyDINO
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class AnomalyDINOModule(LightningModule):
    """Lightning wrapper around :class:`AnomalyDINO`.

    Parameters
    ----------
    model_name : str
        DINOv2 variant (default ``'dinov2_vits14'``).
    smaller_edge_size : int
        Resize shorter edge to this (default 448).
    masking : bool
        Apply PCA-based foreground masking (default True).
    rotation : bool
        Augment references with rotations (default True).
    masking_threshold : float
        PCA threshold for masking (default 10.0).
    gaussian_sigma : float
        Gaussian smoothing σ for anomaly maps (default 4.0).
    top_percent : float
        Fraction of top patch distances for image-level scoring (default 0.01).
    image_size : int
        Input image spatial resolution (for anomaly map output).
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        smaller_edge_size: int = 448,
        masking: bool = True,
        rotation: bool = True,
        masking_threshold: float = 10.0,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        image_size: int = 448,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # No gradient-based optimisation.
        self.automatic_optimization = False

        self.model = AnomalyDINO(
            model_name=model_name,
            smaller_edge_size=smaller_edge_size,
            masking=masking,
            rotation=rotation,
            masking_threshold=masking_threshold,
            gaussian_sigma=gaussian_sigma,
            top_percent=top_percent,
        )

        self._image_size = image_size

        # Feature accumulator for the single training epoch.
        self._train_images: list[np.ndarray] = []

        # Metric accumulators.
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.predict_batch_tensor(
            x, output_size=(self._image_size, self._image_size)
        )

    # ------------------------------------------------------------------
    # Training  (feature extraction + memory-bank construction)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if self.model._fitted:
            return

        # Collect raw image tensors — we feed them through the model's own
        # prepare_image pipeline in on_train_epoch_end to honour DINOv2's
        # exact preprocessing (resize by smaller edge, crop to patch multiple).
        images = batch["image"]  # (B, C, H, W) normalised tensors

        # Denormalise back to [0, 255] uint8 for the model's prepare_image.
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = (images_denorm.clamp(0, 1) * 255).byte()

        for i in range(images_denorm.shape[0]):
            img_np = images_denorm[i].permute(1, 2, 0).cpu().numpy()  # HWC RGB
            self._train_images.append(img_np)

    def on_train_epoch_end(self) -> None:
        if self.model._fitted or not self._train_images:
            return

        self.model.fit(self._train_images)
        n_patches = self.model._memory_bank.shape[0] if self.model._memory_bank is not None else 0

        self._train_images.clear()

        print(
            f"AnomalyDINO: memory bank built — {n_patches} patches "
            f"from {len(self._train_images) if self._train_images else 'all'} "
            f"reference images "
            f"(masking={self.model.masking}, rotation={self.model.rotation})",
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

        x = batch["image"]
        scores, anomaly_maps = self.model.predict_batch_tensor(
            x, output_size=(self._image_size, self._image_size)
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
    # Test  (same metric logic as validation)
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
    # Optimiser  (no-op — required by Lightning)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> None:  # type: ignore[override]
        return None

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_dir: str | Path) -> None:
        """Persist the memory bank and hparams to *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "hparams": dict(self.hparams),
            "memory_bank": self.model._memory_bank,
            "fitted": self.model._fitted,
        }
        torch.save(state, ckpt_dir / "model.ckpt")
