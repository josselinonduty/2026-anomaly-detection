"""PyTorch Lightning module for SubspaceAD anomaly detection.

SubspaceAD is a training-free method that uses DINOv2 features with PCA
subspace modeling for few-shot anomaly detection. The "training" phase
extracts patch features from normal reference images, augments them with
random rotations, and fits a PCA model. Anomaly scoring is performed via
reconstruction residuals from this subspace.

Only **one** training epoch is needed (training-free fit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningModule

from lib.models.subspacead import SubspaceAD
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class SubspaceADModule(LightningModule):
    """Lightning wrapper around :class:`SubspaceAD`.

    Parameters
    ----------
    model_name : str
        DINOv2 variant (default ``'dinov2_vitg14'``).
    image_resolution : int
        Input resolution in pixels (default 672).
    layers : tuple[int, ...]
        Transformer block indices to average (default layers 22–28).
    pca_variance_threshold : float
        Explained variance threshold τ (default 0.99).
    aug_count : int
        Random rotations per normal image (default 30).
    gaussian_sigma : float
        Gaussian smoothing σ for anomaly maps (default 4.0).
    top_percent : float
        Fraction of top patch scores for image-level scoring (default 0.01).
    image_size : int
        Output anomaly map spatial resolution.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitg14",
        image_resolution: int = 672,
        layers: tuple[int, ...] | None = None,
        pca_variance_threshold: float = 0.99,
        aug_count: int = 30,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        image_size: int = 672,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # No gradient-based optimisation.
        self.automatic_optimization = False

        self.model = SubspaceAD(
            model_name=model_name,
            image_resolution=image_resolution,
            layers=layers,
            pca_variance_threshold=pca_variance_threshold,
            aug_count=aug_count,
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.predict_batch_tensor(
            x, output_size=(self._image_size, self._image_size)
        )

    # ------------------------------------------------------------------
    # Training  (feature extraction + PCA fitting)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if self.model._fitted:
            return

        # Denormalise back to [0, 255] uint8 for the model's prepare_image.
        images = batch["image"]  # (B, C, H, W) normalised tensors
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = (images_denorm.clamp(0, 1) * 255).byte()

        for i in range(images_denorm.shape[0]):
            img_np = images_denorm[i].permute(1, 2, 0).cpu().numpy()  # HWC RGB
            self._train_images.append(img_np)

    def on_train_epoch_end(self) -> None:
        if self.model._fitted or not self._train_images:
            return

        n_images = len(self._train_images)
        self.model.fit(self._train_images)

        self._train_images.clear()

        r = self.model._components.shape[1] if self.model._components is not None else 0
        print(
            f"SubspaceAD: PCA fitted — {r} components from {n_images} reference images "
            f"(aug_count={self.model.aug_count}, τ={self.model.pca_variance_threshold})",
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
        """Persist the PCA parameters and hparams to *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.model._mu is not None:
            np.save(ckpt_dir / "pca_mu.npy", self.model._mu)
        if self.model._components is not None:
            np.save(ckpt_dir / "pca_components.npy", self.model._components)

        state = {
            "hparams": dict(self.hparams),
            "fitted": self.model._fitted,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "SubspaceADModule":
        """Restore a SubspaceAD module from *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.ckpt", map_location=map_location)
        hparams = state["hparams"]
        # Convert layers list back to tuple if needed
        if "layers" in hparams and isinstance(hparams["layers"], list):
            hparams["layers"] = tuple(hparams["layers"])
        module = cls(**hparams)
        if state["fitted"]:
            module.model._mu = np.load(ckpt_dir / "pca_mu.npy")
            module.model._components = np.load(ckpt_dir / "pca_components.npy")
            module.model._fitted = True
        return module
