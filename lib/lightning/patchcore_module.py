"""PyTorch Lightning module for PatchCore anomaly detection.

PatchCore is a memory-bank method — it does **not** learn parameters via
gradient descent.  The "training" phase extracts patch features from normal
images, builds a coreset memory bank, and then anomaly scoring is performed
via nearest-neighbour lookup.

Only **one** training epoch is needed.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pytorch_lightning import LightningModule

from lib.models.patchcore import PatchCore
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


class PatchCoreModule(LightningModule):
    """Lightning wrapper around :class:`PatchCore`.

    Parameters
    ----------
    layers_to_extract : tuple of int
        Backbone layers to tap (default ``(2, 3)``).
    coreset_sampling_ratio : float
        Fraction of patches kept in the coreset (default 0.01 = 1 %).
    num_neighbors : int
        Neighbours for image-score re-weighting (Eq. 2, default 9).
    image_size : int
        Expected input spatial resolution.
    projection_dim : int
        Random-projection dimension for coreset selection.
    """

    def __init__(
        self,
        layers_to_extract: tuple[int, ...] = (2, 3),
        coreset_sampling_ratio: float = 0.01,
        num_neighbors: int = 9,
        image_size: int = 256,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # No gradient-based optimisation.
        self.automatic_optimization = False

        self.model = PatchCore(
            layers_to_extract=layers_to_extract,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            image_size=image_size,
            projection_dim=projection_dim,
        )

        # Feature accumulator for the single training epoch.
        self._train_features: list[torch.Tensor] = []

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
    # Training  (feature extraction + memory-bank construction)
    # ------------------------------------------------------------------

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        if self.model._fitted:
            return
        x = batch["image"]
        features, _ = self.model.extract_features(x)
        self._train_features.append(features.cpu())

    def on_train_epoch_end(self) -> None:
        if self.model._fitted or not self._train_features:
            return

        all_features = torch.cat(self._train_features, dim=0)
        n_total = all_features.shape[0]
        n_select = max(
            1,
            int(n_total * self.model.coreset_sampling_ratio),
        )

        indices = PatchCore._greedy_coreset_sampling(
            all_features,
            n_select,
            self.model.projection_dim,
        )
        self.model.memory_bank = all_features[indices].to(self.device)
        self.model._fitted = True
        self._train_features.clear()

        print(
            f"PatchCore: memory bank built — {n_select} coreset entries "
            f"from {n_total} patches "
            f"({self.model.coreset_sampling_ratio * 100:.1f} %)",
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
        """Persist the memory bank and hparams to *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "hparams": dict(self.hparams),
            "memory_bank": self.model.memory_bank,
            "fitted": self.model._fitted,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "PatchCoreModule":
        """Restore a PatchCore module from *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.ckpt", map_location=map_location)
        module = cls(**state["hparams"])
        if state["fitted"]:
            module.model.memory_bank = state["memory_bank"]
            module.model._fitted = True
        return module
