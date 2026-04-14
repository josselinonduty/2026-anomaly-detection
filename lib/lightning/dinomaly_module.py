"""PyTorch Lightning module for Dinomaly training & evaluation.

Implements the full Dinomaly training pipeline:
- Global Hard-Mining Cosine Loss with gradient shrinking
- StableAdamW optimiser with warm-up cosine LR schedule
- Anomaly scoring via per-pixel cosine distance with Gaussian smoothing
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import _LRScheduler

from lib.models.dinomaly import Dinomaly, build_dinomaly
from lib.utils.metrics import compute_auroc, compute_pixel_auroc
from lib.utils.stable_adamw import StableAdamW


# ── Loss helpers ─────────────────────────────────────────────────────


def _modify_grad(
    x: torch.Tensor,
    inds: torch.Tensor,
    factor: float = 0.0,
) -> torch.Tensor:
    """Hook function that shrinks gradients at well-reconstructed positions."""
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm_percent(
    en: list[torch.Tensor],
    de: list[torch.Tensor],
    p: float = 0.9,
    factor: float = 0.1,
) -> torch.Tensor:
    """Global Hard-Mining Cosine Loss.

    For each group pair *(en[i], de[i])*:
    1. Compute per-point cosine distance (encoder features detached).
    2. Find the threshold at the top ``(1 - p)`` percentile of distances.
    3. Compute global cosine loss = ``mean(1 - cos_sim(flatten(en), flatten(de)))``.
    4. Register a gradient hook on ``de`` that multiplies gradients by
       ``factor`` (default 0.1) wherever the distance is below the threshold.

    The hard-mining schedule is controlled externally by ramping ``p`` from 0
    to 0.9 over the first 1 000 iterations.
    """
    cos = nn.CosineSimilarity()
    loss = torch.tensor(0.0, device=en[0].device)

    for i in range(len(en)):
        a = en[i].detach()
        b = de[i]

        # Per-point cosine distance for the hard-mining mask
        with torch.no_grad():
            point_dist = 1.0 - cos(a, b).unsqueeze(1)  # (B, 1, H, W)

        # Top (1 - p) % threshold
        flat_dist = point_dist.view(-1)
        k = max(1, int(flat_dist.shape[0] * (1.0 - p)))
        thresh = flat_dist.kthvalue(flat_dist.shape[0] - k + 1)[0]

        # Global cosine loss (averaged over the flattened feature map)
        loss = loss + torch.mean(
            1.0
            - cos(
                a.reshape(a.shape[0], -1),
                b.reshape(b.shape[0], -1),
            )
        )

        # Gradient hook: shrink gradients for well-reconstructed points
        hook_fn = partial(_modify_grad, inds=point_dist < thresh, factor=factor)
        b.register_hook(hook_fn)

    return loss / len(en)


# ── Gaussian smoothing ───────────────────────────────────────────────


def _get_gaussian_kernel(kernel_size: int = 5, sigma: float = 4.0) -> nn.Conv2d:
    """Build a 1-channel Gaussian-blur convolutional filter."""
    coords = torch.arange(kernel_size, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1)
    mean = (kernel_size - 1) / 2.0
    variance = sigma**2
    kernel = (1.0 / (2.0 * math.pi * variance)) * torch.exp(
        -torch.sum((grid - mean) ** 2, dim=-1) / (2.0 * variance)
    )
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    conv = nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
    conv.weight.data = kernel
    conv.weight.requires_grad = False
    return conv


# ── LR scheduler ─────────────────────────────────────────────────────


class WarmCosineScheduler(_LRScheduler):
    """Linear warm-up followed by cosine annealing (step-level)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_value: float,
        final_value: float,
        total_iters: int,
        warmup_iters: int = 0,
        start_warmup_value: float = 0.0,
    ) -> None:
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
        iters = np.arange(total_iters - warmup_iters)
        cosine_schedule = final_value + 0.5 * (base_value - final_value) * (
            1.0 + np.cos(np.pi * iters / len(iters))
        )
        self.schedule = np.concatenate((warmup_schedule, cosine_schedule))
        super().__init__(optimizer)

    def get_lr(self):  # noqa: D102
        if self.last_epoch >= self.total_iters:
            return [self.final_value for _ in self.base_lrs]
        return [self.schedule[self.last_epoch] for _ in self.base_lrs]


# ── Lightning module ─────────────────────────────────────────────────


class DinomalyModule(LightningModule):
    """Lightning wrapper for the Dinomaly anomaly detection model.

    Parameters
    ----------
    backbone:
        DINOv2-Register backbone name (see :data:`_BACKBONE_CONFIGS`).
    dropout:
        Dropout rate for the noisy bottleneck (default 0.2).
    total_iters:
        Total training iterations (default 10 000).
    lr:
        Peak learning rate (default 2e-3).
    final_lr:
        Final learning rate after cosine annealing (default 2e-4).
    weight_decay:
        Weight decay for StableAdamW (default 1e-4).
    warmup_iters:
        Linear warm-up iterations (default 100).
    hm_ramp_iters:
        Iterations over which the hard-mining ``p`` ramps from 0 → 0.9.
    hm_p_final:
        Final hard-mining discard proportion (default 0.9).
    hm_factor:
        Gradient shrink factor for well-reconstructed points (default 0.1).
    max_ratio:
        Fraction of top anomaly-map pixels used for the image-level score
        (default 0.01 = top 1 %).
    eval_resize:
        Resize anomaly maps & GT masks to this size for pixel-level eval
        (default 256).
    """

    def __init__(
        self,
        backbone: str = "dinov2reg_vit_base_14",
        dropout: float = 0.2,
        total_iters: int = 10_000,
        lr: float = 2e-3,
        final_lr: float = 2e-4,
        weight_decay: float = 1e-4,
        warmup_iters: int = 100,
        hm_ramp_iters: int = 1000,
        hm_p_final: float = 0.9,
        hm_factor: float = 0.1,
        max_ratio: float = 0.01,
        eval_resize: int = 256,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model: Dinomaly = build_dinomaly(
            backbone=backbone,
            dropout=dropout,
        )

        # Gaussian smoothing filter (for inference anomaly maps)
        gaussian_kernel = _get_gaussian_kernel(kernel_size=5, sigma=4.0)
        self.register_buffer("_gaussian_weight", gaussian_kernel.weight.data)
        self._gaussian_padding = gaussian_kernel.padding[0]

        # Iteration counter (for p ramp-up and logging)
        self._global_iter = 0

        # Buffers for epoch-level metric collection
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):  # noqa: D102
        return self.model(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x = batch["image"]
        en, de = self.model(x)

        # Hard-mining p ramp-up: 0 → p_final over hm_ramp_iters
        p = min(
            self.hparams.hm_p_final * self._global_iter / self.hparams.hm_ramp_iters,
            self.hparams.hm_p_final,
        )
        loss = global_cosine_hm_percent(en, de, p=p, factor=self.hparams.hm_factor)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/hm_p", p, on_step=True, on_epoch=False)
        self._global_iter += 1
        return loss

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        # Clip gradients at max_norm=0.1 as in the reference implementation
        trainable = nn.ModuleList([self.model.bottleneck, self.model.decoder])
        nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_labels.clear()
        self._val_scores.clear()
        self._val_masks.clear()
        self._val_anomaly_maps.clear()

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x = batch["image"]
        en, de = self.model(x)

        # Anomaly map: mean cosine distance across groups, upsampled
        anomaly_map = self._compute_anomaly_map(en, de, out_size=x.shape[-1])
        # Gaussian smooth
        anomaly_map = F.conv2d(
            anomaly_map, self._gaussian_weight, padding=self._gaussian_padding
        )  # (B, 1, H, W)

        # Resize for pixel-level evaluation
        eval_size = self.hparams.eval_resize
        anomaly_map = F.interpolate(
            anomaly_map, size=eval_size, mode="bilinear", align_corners=False
        )
        anomaly_map = anomaly_map[:, 0]  # (B, H, W)

        # Image-level score: mean of top max_ratio pixels
        flat = anomaly_map.flatten(1)
        k = max(1, int(flat.shape[1] * self.hparams.max_ratio))
        scores = torch.topk(flat, k=k, dim=1)[0].mean(dim=1)  # (B,)

        # Resize GT masks to eval_size for pixel AUROC
        masks = batch["mask"]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = F.interpolate(masks.float(), size=eval_size, mode="nearest")[:, 0]

        self._val_labels.append(batch["label"].cpu())
        self._val_scores.append(scores.cpu())
        self._val_masks.append(masks.cpu())
        self._val_anomaly_maps.append(anomaly_map.cpu())

    def _compute_anomaly_map(
        self,
        en: list[torch.Tensor],
        de: list[torch.Tensor],
        out_size: int,
    ) -> torch.Tensor:
        """Per-group cosine distance → bilinear upsample → mean across groups."""
        maps = torch.stack(
            [1.0 - F.cosine_similarity(en[i], de[i]) for i in range(len(en))],
            dim=1,
        )  # (B, G, H, W)
        maps = F.interpolate(maps, size=out_size, mode="bilinear", align_corners=True)
        return maps.mean(dim=1, keepdim=True)

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
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_dir: str | Path) -> None:
        """Persist model weights to *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "hparams": dict(self.hparams),
            "model_state_dict": self.model.state_dict(),
            "global_iter": self._global_iter,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "DinomalyModule":
        """Restore a Dinomaly module from *ckpt_dir*."""
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.ckpt", map_location=map_location)
        module = cls(**state["hparams"])
        module.model.load_state_dict(state["model_state_dict"])
        module._global_iter = state["global_iter"]
        return module

    def configure_optimizers(self) -> dict:
        trainable = nn.ModuleList([self.model.bottleneck, self.model.decoder])
        optimizer = StableAdamW(
            trainable.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            weight_decay=self.hparams.weight_decay,
            amsgrad=True,
            eps=1e-10,
        )
        scheduler = WarmCosineScheduler(
            optimizer,
            base_value=self.hparams.lr,
            final_value=self.hparams.final_lr,
            total_iters=self.hparams.total_iters,
            warmup_iters=self.hparams.warmup_iters,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
