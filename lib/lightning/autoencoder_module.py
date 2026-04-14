"""PyTorch Lightning module for anomaly detection training & evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from lib.models import AnomalyAutoencoder
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class AutoencoderModule(LightningModule):
    """Lightning wrapper around :class:`AnomalyAutoencoder`.

    Training minimises MSE reconstruction loss on *normal* images.
    Validation / test compute image-level and pixel-level AUROC using the
    per-pixel reconstruction error as anomaly score.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        latent_dim: int = 256,
        image_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = AnomalyAutoencoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            latent_dim=latent_dim,
            image_size=image_size,
        )
        self.criterion = nn.MSELoss()

        # Buffers for collecting epoch-level metrics
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch["image"]
        x_hat, _ = self.model(x)
        loss = self.criterion(x_hat, x)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_labels.clear()
        self._val_scores.clear()
        self._val_masks.clear()
        self._val_anomaly_maps.clear()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["image"]
        x_hat, _ = self.model(x)
        loss = self.criterion(x_hat, x)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        # Per-pixel anomaly map: mean squared error across channels
        anomaly_map = (x - x_hat).pow(2).mean(dim=1)  # (B, H, W)
        # Image-level score: mean of anomaly map
        score = anomaly_map.flatten(1).mean(dim=1)  # (B,)

        self._val_labels.append(batch["label"].cpu())
        self._val_scores.append(score.cpu())
        self._val_masks.append(batch["mask"].cpu())
        self._val_anomaly_maps.append(anomaly_map.cpu())

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
    # Test (same logic as validation)
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
