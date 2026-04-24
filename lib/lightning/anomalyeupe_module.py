"""PyTorch Lightning module for AnomalyEUPE anomaly detection.

AnomalyEUPE uses EUPE ONNX models with dual-level scoring (global CLS +
local patch NN).  The "training" phase extracts features from normal
reference images and builds two memory banks.  Only one training epoch is
needed.

The model itself runs on ONNX Runtime (CPU or CUDA), not PyTorch.  The
Lightning module bridges the dataloader / metrics / checkpoint interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningModule

from lib.models.anomalyeupe import AnomalyEUPE, _INPUT_SIZE
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class AnomalyEUPEModule(LightningModule):
    """Lightning wrapper around :class:`AnomalyEUPE`.

    Parameters
    ----------
    model_name : str
        EUPE ONNX model name (default ``'eupe_vitb16'``).
        Append ``_int8`` for quantised variant.
    masking : bool
        Apply PCA-based foreground masking (default True).
    rotation : bool
        Augment references with rotations (default True).
    global_weight : float
        Weight of global (CLS) score vs local (patch) score.
    image_size : int
        Output anomaly map spatial resolution.
    """

    def __init__(
        self,
        model_name: str = "eupe_vitb16",
        masking: bool = True,
        rotation: bool = True,
        masking_threshold: float = 10.0,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        global_weight: float = 0.3,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.model = AnomalyEUPE(
            model_name=model_name,
            masking=masking,
            rotation=rotation,
            masking_threshold=masking_threshold,
            gaussian_sigma=gaussian_sigma,
            top_percent=top_percent,
            global_weight=global_weight,
        )

        self._image_size = image_size

        # Feature accumulators for the single training epoch.
        self._train_cls: list[np.ndarray] = []
        self._train_patches: list[np.ndarray] = []
        self._train_image_count: int = 0

        # Metric accumulators.
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of normalised image tensors.

        Denormalises, converts to numpy, and runs ONNX inference.
        """
        images_np = self._tensor_to_numpy_batch(x)
        scores_np, maps_np = self.model.predict_batch_numpy(
            images_np, output_size=(self._image_size, self._image_size)
        )
        scores = torch.from_numpy(scores_np)
        # maps_np is (B, H, W) -> (B, 1, H, W)
        anomaly_maps = torch.from_numpy(maps_np).unsqueeze(1)
        return scores, anomaly_maps

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

        images = batch["image"]  # (B, C, H, W) normalised tensors
        images_np = self._tensor_to_numpy_batch(images)

        for img_np in images_np:
            variants = (
                self.model.augment_reference(img_np)
                if self.model.rotation
                else [img_np]
            )

            for variant in variants:
                cls_tok, patch_tok = self.model.extract(variant)
                self._train_cls.append(cls_tok)
                self._train_patches.append(patch_tok)

            self._train_image_count += 1

    def on_train_epoch_end(self) -> None:
        if self.model._fitted or not self._train_cls:
            return

        # CLS bank
        self.model._cls_bank = np.stack(self._train_cls, axis=0).astype(np.float32)
        norms = np.linalg.norm(self.model._cls_bank, axis=1, keepdims=True)
        self.model._cls_bank /= np.maximum(norms, 1e-12)
        self.model._cls_mean = self.model._cls_bank.mean(axis=0)
        self.model._cls_mean /= max(np.linalg.norm(self.model._cls_mean), 1e-12)

        # Patch bank — pre-allocated
        total_patches = sum(p.shape[0] for p in self._train_patches)
        dim = self._train_patches[0].shape[1]
        self.model._patch_bank = np.empty((total_patches, dim), dtype=np.float32)
        offset = 0
        for p in self._train_patches:
            n = p.shape[0]
            self.model._patch_bank[offset : offset + n] = p
            offset += n

        norms = np.linalg.norm(self.model._patch_bank, axis=1, keepdims=True)
        self.model._patch_bank /= np.maximum(norms, 1e-12)

        self.model._fitted = True

        n_cls = self.model._cls_bank.shape[0]
        n_patches = self.model._patch_bank.shape[0]
        n_images = self._train_image_count

        self._train_cls.clear()
        self._train_patches.clear()
        self._train_image_count = 0

        print(
            f"AnomalyEUPE: memory banks built — "
            f"{n_cls} CLS embeddings, {n_patches} patch tokens "
            f"from {n_images} reference images "
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
        scores, anomaly_maps = self.forward(x)

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
    # Optimiser  (no-op)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> None:  # type: ignore[override]
        return None

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_dir: str | Path) -> None:
        """Persist the memory banks and hparams to *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.model._cls_bank is not None:
            np.save(ckpt_dir / "cls_bank.npy", self.model._cls_bank)
        if self.model._cls_mean is not None:
            np.save(ckpt_dir / "cls_mean.npy", self.model._cls_mean)
        if self.model._patch_bank is not None:
            np.save(ckpt_dir / "patch_bank.npy", self.model._patch_bank)

        state = {
            "hparams": dict(self.hparams),
            "fitted": self.model._fitted,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "AnomalyEUPEModule":
        """Restore an AnomalyEUPE module from *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.ckpt", map_location=map_location)
        module = cls(**state["hparams"])
        if state["fitted"]:
            module.model._cls_bank = np.load(ckpt_dir / "cls_bank.npy")
            module.model._cls_mean = np.load(ckpt_dir / "cls_mean.npy")
            module.model._patch_bank = np.load(ckpt_dir / "patch_bank.npy")
            module.model._fitted = True
        return module

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy_batch(images: torch.Tensor) -> list[np.ndarray]:
        """Denormalise a batch of (B, C, H, W) tensors to HWC-RGB uint8."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = images.cpu() * std + mean
        images = (images.clamp(0, 1) * 255).byte()
        result: list[np.ndarray] = []
        for i in range(images.shape[0]):
            result.append(images[i].permute(1, 2, 0).numpy())
        return result
