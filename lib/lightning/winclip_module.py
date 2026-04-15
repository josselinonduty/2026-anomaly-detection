"""PyTorch Lightning module for WinCLIP(+) anomaly detection.

WinCLIP is a zero-/few-shot method — it does **not** learn parameters via
gradient descent.  The "training" phase builds text features and optionally
a visual gallery from normal images (WinCLIP+).  Anomaly scoring is done
purely at inference time using CLIP features.

Only **one** training epoch is needed (for WinCLIP+ gallery construction).
"""

from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning import LightningModule

from lib.models.winclip import WinCLIP
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class WinCLIPModule(LightningModule):
    """Lightning wrapper around :class:`WinCLIP`.

    Parameters
    ----------
    category : str
        Object category name for prompt construction (e.g. "candle").
    backbone : str
        OpenCLIP model name. Default ``"ViT-B-16-plus-240"``.
    pretrained : str
        Pretrained dataset. Default ``"laion400m_e32"``.
    scales : tuple of int
        Window sizes in patches. Default ``(2, 3)``.
    image_size : int
        Expected input resolution. Default 240.
    k_shot : int
        Number of normal reference shots for WinCLIP+.
        0 means zero-shot (WinCLIP only).
    """

    def __init__(
        self,
        category: str = "candle",
        backbone: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
        scales: tuple[int, ...] = (2, 3),
        image_size: int = 240,
        k_shot: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # No gradient-based optimisation.
        self.automatic_optimization = False

        self.model = WinCLIP(
            backbone=backbone,
            pretrained=pretrained,
            scales=scales,
            image_size=image_size,
        )
        self.category = category
        self.k_shot = k_shot

        # Accumulator for normal images (WinCLIP+ gallery)
        self._train_images: list[torch.Tensor] = []
        self._gallery_built = False

        # Metric accumulators.
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
    # Training  (text features + optional visual gallery)
    # ------------------------------------------------------------------

    def on_train_epoch_start(self) -> None:
        if not self.model._text_ready:
            self.model.build_text_features(self.category)
            print(
                f"WinCLIP: text features built for category '{self.category}' "
                f"({len(TEMPLATES)}×{len(STATE_NORMAL)+len(STATE_ABNORMAL)} prompts)",
            )

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if self._gallery_built or self.k_shot == 0:
            return
        x = batch["image"]
        self._train_images.append(x.cpu())

    def on_train_epoch_end(self) -> None:
        if self._gallery_built or self.k_shot == 0:
            return

        if self._train_images:
            all_images = torch.cat(self._train_images, dim=0)
            # Limit to k_shot images if specified
            if self.k_shot > 0 and all_images.shape[0] > self.k_shot:
                all_images = all_images[: self.k_shot]

            all_images = all_images.to(self.device)
            self.model.build_visual_gallery(all_images)
            self._gallery_built = True
            self._train_images.clear()
            print(
                f"WinCLIP+: visual gallery built from {all_images.shape[0]} "
                f"normal images ({len(self.model.scales)} scales)",
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
        if not self.model._text_ready:
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
        """Persist the text features, visual gallery, and hparams."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "hparams": dict(self.hparams),
            "text_features": self.model.text_features,
            "text_ready": self.model._text_ready,
            "visual_gallery": self.model.visual_gallery,
        }
        torch.save(state, ckpt_dir / "model.ckpt")


# Re-export for the on_train_epoch_start print
from lib.models.winclip import STATE_ABNORMAL, STATE_NORMAL, TEMPLATES
