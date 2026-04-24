"""PyTorch Lightning module for classical feature-matching anomaly detection.

This is a memory-bank method — it does **not** learn parameters via gradient
descent.  The "training" phase stores normal reference images and precomputes
their keypoints/descriptors.  At inference, test images are aligned to the
best-matching reference via homography (SIFT/ORB + RANSAC) and anomaly maps
are produced from the pixel-level structural difference.

Only **one** training epoch is needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningModule

from lib.models.feature_match import FeatureMatch
from lib.utils.metrics import compute_auroc, compute_pixel_auroc

# ImageNet de-normalisation constants.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureMatchModule(LightningModule):
    """Lightning wrapper around :class:`FeatureMatch`.

    Parameters
    ----------
    descriptor : str
        ``"sift"`` or ``"orb"``.
    image_size : int
        Expected input spatial resolution.
    map_mode : str
        ``"dense"`` (Gaussian-blurred abs diff) or ``"ssim"`` (1 − SSIM).
    ratio_thresh : float
        Lowe's ratio-test threshold for descriptor matching.
    blur_sigma : float
        Gaussian sigma for difference map smoothing.
    """

    def __init__(
        self,
        descriptor: str = "sift",
        image_size: int = 256,
        map_mode: str = "dense",
        ratio_thresh: float = 0.75,
        blur_sigma: float = 7.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # No gradient-based optimisation.
        self.automatic_optimization = False

        self.model = FeatureMatch(
            descriptor=descriptor,
            image_size=image_size,
            map_mode=map_mode,
            ratio_thresh=ratio_thresh,
            blur_sigma=blur_sigma,
        )

        # Image accumulator for the single training epoch.
        self._train_images: list[np.ndarray] = []

        # Metric accumulators.
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_to_uint8(batch_tensor: torch.Tensor) -> list[np.ndarray]:
        """Convert an ImageNet-normalised (B,3,H,W) tensor to uint8 HWC."""
        imgs = batch_tensor.detach().cpu().float().permute(0, 2, 3, 1).numpy()
        imgs = imgs * _STD + _MEAN
        imgs = np.clip(imgs * 255, 0, 255).astype(np.uint8)
        return [imgs[i] for i in range(imgs.shape[0])]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    # ------------------------------------------------------------------
    # Training  (descriptor extraction + DB construction)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if self.model._fitted:
            return
        self._train_images.extend(self._batch_to_uint8(batch["image"]))

    def on_train_epoch_end(self) -> None:
        if self.model._fitted or not self._train_images:
            return
        self.model.fit(self._train_images)
        self._train_images.clear()

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
        scores, anomaly_maps = self.model.predict(x)

        self._val_labels.append(batch["label"].cpu())
        self._val_scores.append(scores.cpu())
        self._val_masks.append(batch["mask"].cpu())
        self._val_anomaly_maps.append(anomaly_maps.cpu())

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
        """Persist the reference images and hparams to *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "hparams": dict(self.hparams),
            "ref_colors": self.model._ref_colors,
            "fitted": self.model._fitted,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "FeatureMatchModule":
        """Restore a FeatureMatch module from *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(
            ckpt_dir / "model.ckpt", map_location=map_location, weights_only=False
        )
        module = cls(**state["hparams"])
        if state["fitted"]:
            # Re-fit from stored reference images to rebuild descriptors.
            module.model.fit(state["ref_colors"])
        return module
