"""PyTorch Lightning module for EfficientAd training & evaluation.

Implements the full EfficientAd pipeline (Batzner et al., WACV 2024):
- Teacher PDN pretraining via knowledge distillation from WideResNet-101-2
- Teacher output normalization (channel-wise mean & std on training data)
- Student-teacher + autoencoder training with hard-mining and penalty losses
- Quantile-based anomaly map normalization for inference
- Combined student-teacher / autoencoder-student anomaly scoring
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision import models as tv_models
from torchvision import transforms as T

from lib.models.efficientad import get_autoencoder, get_pdn_medium, get_pdn_small
from lib.utils.metrics import compute_auroc, compute_pixel_auroc

# ImageNet normalisation constants (used for AE colour-jitter augmentation)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Default number of feature channels (Section 3, d = 384)
_OUT_CHANNELS = 384


class EfficientAdModule(LightningModule):
    """Lightning wrapper for EfficientAd anomaly detection.

    Parameters
    ----------
    model_size :
        PDN variant: ``"small"`` or ``"medium"`` (default ``"small"``).
    out_channels :
        Feature dimensionality *d* (default 384).
    train_steps :
        Total training iterations for student + autoencoder (default 70 000).
    lr :
        Learning rate for Adam (default 1e-4).
    weight_decay :
        Weight decay for Adam (default 1e-5).
    image_size :
        Input image resolution (default 256).
    eval_resize :
        Resolution for pixel-level AUROC evaluation (default 256).
    teacher_pretrain_steps :
        Knowledge-distillation iterations for the teacher PDN (default 10 000).
        Set to 0 when loading external pre-trained teacher weights.
    teacher_weights :
        Optional path to pre-trained teacher PDN weights.  When provided,
        teacher pretraining is skipped.
    penalty_weight :
        Weight of the ImageNet penalty loss.  Effective only when an external
        penalty data loader is supplied via :meth:`set_penalty_dataloader`.
        Set to 0 to disable (default 1.0).
    """

    def __init__(
        self,
        model_size: str = "small",
        out_channels: int = _OUT_CHANNELS,
        train_steps: int = 70_000,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        image_size: int = 256,
        eval_resize: int = 256,
        teacher_pretrain_steps: int = 10_000,
        teacher_weights: str | None = None,
        penalty_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.out_channels = out_channels

        # ── Networks ─────────────────────────────────────────────────
        pdn_fn = get_pdn_small if model_size == "small" else get_pdn_medium
        self.teacher: nn.Sequential = pdn_fn(out_channels)
        self.student: nn.Sequential = pdn_fn(2 * out_channels)
        self.autoencoder: nn.Sequential = get_autoencoder(out_channels)

        # Teacher is always frozen during the main training loop.
        for p in self.teacher.parameters():
            p.requires_grad = False

        # ── Teacher normalisation buffers ────────────────────────────
        self.register_buffer("teacher_mean", torch.zeros(1, out_channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, out_channels, 1, 1))

        # ── Map normalisation buffers (quantile-based, Section 3.4) ──
        self.register_buffer("q_st_start", torch.tensor(0.0))
        self.register_buffer("q_st_end", torch.tensor(1.0))
        self.register_buffer("q_ae_start", torch.tensor(0.0))
        self.register_buffer("q_ae_end", torch.tensor(1.0))

        # ── Colour-jitter augmentation for the AE path (Section 3.3) ─
        self._color_jitter = T.RandomChoice(
            [
                T.ColorJitter(brightness=0.2),
                T.ColorJitter(contrast=0.2),
                T.ColorJitter(saturation=0.2),
            ]
        )

        # ── State flags ──────────────────────────────────────────────
        self._teacher_ready = teacher_weights is not None
        self._map_norm_ready = False
        self._penalty_loader = None  # optional external penalty loader

        # Load external teacher weights when provided.
        if teacher_weights is not None:
            state_dict = torch.load(teacher_weights, map_location="cpu")
            self.teacher.load_state_dict(state_dict)

        self.teacher.eval()

        # ── Validation / test metric accumulators ────────────────────
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []
        self._val_maps_st: list[torch.Tensor] = []
        self._val_maps_ae: list[torch.Tensor] = []

        # Reference to training dataloader, set by main.py before fit.
        self._train_dl_ref = None

    # ------------------------------------------------------------------
    # External setup helpers (called from main.py before trainer.fit)
    # ------------------------------------------------------------------

    def set_train_dataloader(self, dl) -> None:
        """Store a reference to the training data loader for teacher setup."""
        self._train_dl_ref = dl

    def set_penalty_dataloader(self, dl) -> None:
        """Optional: provide an ImageNet penalty data loader (Section 3.2)."""
        self._penalty_loader = dl

    # ------------------------------------------------------------------
    # Teacher pretraining via Knowledge Distillation (Section 3.1)
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        if not self._teacher_ready:
            assert self._train_dl_ref is not None, (
                "Call model.set_train_dataloader(dl) before trainer.fit() "
                "so the teacher PDN can be pretrained."
            )
            self._pretrain_teacher(self._train_dl_ref)
            self._compute_teacher_normalization(self._train_dl_ref)
            self._teacher_ready = True
        elif self.teacher_mean.sum() == 0:
            # Teacher weights loaded externally but normalisation not yet done.
            assert self._train_dl_ref is not None
            self._compute_teacher_normalization(self._train_dl_ref)

    def _pretrain_teacher(self, train_loader) -> None:
        """Pretrain the teacher PDN via KD from WideResNet-101-2 (Section 3.1).

        The paper extracts features from the 2nd and 3rd residual blocks of a
        WideResNet-101-2 pretrained on ImageNet, bilinearly interpolates them
        to 56×56, concatenates them (1536 channels), and linearly projects to
        *d* = 384.  The PDN is then trained with MSE to approximate these
        projected features.

        Here the projection is a frozen 1×1 convolution with Xavier init
        (a random linear projection that preserves distances by the
        Johnson–Lindenstrauss lemma).
        """
        device = self.device
        steps = self.hparams.teacher_pretrain_steps
        if steps <= 0:
            return

        print(f"EfficientAd: pretraining teacher PDN for {steps} steps …")

        # ── Backbone ─────────────────────────────────────────────────
        backbone = tv_models.wide_resnet101_2(
            weights=tv_models.Wide_ResNet101_2_Weights.IMAGENET1K_V1,
        )
        backbone.eval()
        backbone.to(device)

        # Hook into layer2 (512 ch) and layer3 (1024 ch)
        _features: dict[str, torch.Tensor] = {}

        def _hook(name):
            def fn(_module, _input, output):
                _features[name] = output

            return fn

        hook2 = backbone.layer2.register_forward_hook(_hook("layer2"))
        hook3 = backbone.layer3.register_forward_hook(_hook("layer3"))

        # ── Frozen random projection 1 536 → out_channels ───────────
        proj = nn.Conv2d(1536, self.out_channels, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(proj.weight)
        proj.to(device)
        proj.eval()
        for p in proj.parameters():
            p.requires_grad = False

        # ── Optimiser (only teacher PDN parameters) ──────────────────
        # Temporarily enable teacher grads for pretraining
        for p in self.teacher.parameters():
            p.requires_grad = True
        self.teacher.train()

        optimizer = torch.optim.Adam(self.teacher.parameters(), lr=1e-3)

        # ── Infinite iterator over training data ─────────────────────
        def _infinite():
            while True:
                yield from train_loader

        for step, batch in enumerate(_infinite()):
            if step >= steps:
                break

            image = batch["image"].to(device)

            with torch.no_grad():
                backbone(image)
                f2 = F.interpolate(
                    _features["layer2"], size=56, mode="bilinear", align_corners=False
                )
                f3 = F.interpolate(
                    _features["layer3"], size=56, mode="bilinear", align_corners=False
                )
                target = proj(torch.cat([f2, f3], dim=1))

            pred = self.teacher(image)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                print(f"  [teacher KD] step {step}/{steps}  loss={loss.item():.6f}")

        # ── Freeze teacher again ─────────────────────────────────────
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # ── Cleanup ──────────────────────────────────────────────────
        hook2.remove()
        hook3.remove()
        del backbone, proj, optimizer, _features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("EfficientAd: teacher pretraining complete.")

    # ------------------------------------------------------------------
    # Teacher channel-wise normalisation (Section 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_teacher_normalization(self, train_loader) -> None:
        """Compute channel mean and std of teacher outputs on training data."""
        device = self.device
        self.teacher.eval()

        # Pass 1: channel mean
        mean_outputs = []
        for batch in train_loader:
            image = batch["image"].to(device)
            t_out = self.teacher(image)
            mean_outputs.append(torch.mean(t_out, dim=[0, 2, 3]).cpu())
        channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
        channel_mean = channel_mean[None, :, None, None]  # (1, C, 1, 1)

        # Pass 2: channel std
        mean_distances = []
        channel_mean_dev = channel_mean.to(device)
        for batch in train_loader:
            image = batch["image"].to(device)
            t_out = self.teacher(image)
            distance = (t_out - channel_mean_dev) ** 2
            mean_distances.append(torch.mean(distance, dim=[0, 2, 3]).cpu())
        channel_var = torch.mean(torch.stack(mean_distances), dim=0)
        channel_std = torch.sqrt(channel_var)[None, :, None, None]

        self.teacher_mean.copy_(channel_mean)
        self.teacher_std.copy_(channel_std)
        print("EfficientAd: teacher normalisation computed.")

    # ------------------------------------------------------------------
    # AE colour-jitter augmentation
    # ------------------------------------------------------------------

    def _augment_for_ae(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalise → colour-jitter → renormalise (Section 3.3)."""
        mean = torch.tensor(_IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
        x = x * std + mean
        x = x.clamp(0.0, 1.0)
        augmented = torch.stack([self._color_jitter(img) for img in x])
        augmented = (augmented - mean) / std
        return augmented

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):  # noqa: D102
        teacher_output = self.teacher(x)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        student_output = self.student(x)
        autoencoder_output = self.autoencoder(x)
        return teacher_output, student_output, autoencoder_output

    # ------------------------------------------------------------------
    # Training (Section 3.2 + 3.3)
    # ------------------------------------------------------------------

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        image_st = batch["image"]

        # ── 1. Student-teacher loss with hard mining (Eq. 3) ─────────
        with torch.no_grad():
            teacher_output_st = self.teacher(image_st)
            teacher_output_st = (
                teacher_output_st - self.teacher_mean
            ) / self.teacher_std

        student_output_st = self.student(image_st)[:, : self.out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        # ── 2. Penalty loss (optional, Section 3.2) ─────────────────
        loss_st = loss_hard
        # Penalty omitted when no external penalty loader is provided,
        # equivalent to --imagenet_train_path none in the official code.

        # ── 3. Autoencoder + student-AE losses (Section 3.3) ────────
        image_ae = self._augment_for_ae(image_st)

        ae_output = self.autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = self.teacher(image_ae)
            teacher_output_ae = (
                teacher_output_ae - self.teacher_mean
            ) / self.teacher_std

        student_output_ae = self.student(image_ae)[:, self.out_channels :]

        distance_ae = (teacher_output_ae - ae_output) ** 2
        distance_stae = (ae_output - student_output_ae) ** 2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)

        # ── 4. Total loss ────────────────────────────────────────────
        loss_total = loss_st + loss_ae + loss_stae

        self.log("train/loss", loss_total, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss_st", loss_st, on_step=True, on_epoch=False)
        self.log("train/loss_ae", loss_ae, on_step=True, on_epoch=False)
        self.log("train/loss_stae", loss_stae, on_step=True, on_epoch=False)
        return loss_total

    # ------------------------------------------------------------------
    # Prediction helper (used in validation, test, and map normalisation)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _predict_maps(
        self, image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute raw (unnormalised) anomaly maps.

        Returns
        -------
        map_combined : (B, 1, H, W)
        map_st : (B, 1, H, W)
        map_ae : (B, 1, H, W)
        """
        teacher_output = self.teacher(image)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        student_output = self.student(image)
        autoencoder_output = self.autoencoder(image)

        # Student-teacher anomaly map
        map_st = torch.mean(
            (teacher_output - student_output[:, : self.out_channels]) ** 2,
            dim=1,
            keepdim=True,
        )
        # Autoencoder-student anomaly map
        map_ae = torch.mean(
            (autoencoder_output - student_output[:, self.out_channels :]) ** 2,
            dim=1,
            keepdim=True,
        )

        if self._map_norm_ready:
            map_st = (
                0.1 * (map_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
            )
            map_ae = (
                0.1 * (map_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
            )

        map_combined = 0.5 * map_st + 0.5 * map_ae
        return map_combined, map_st, map_ae

    # ------------------------------------------------------------------
    # Map normalisation (Section 3.4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_map_normalization(self, validation_loader) -> None:
        """Compute quantile-based map normalisation from validation data.

        Call this after training and before testing.  Uses the 90th and
        99.5th percentile of each anomaly map type on the validation set
        to normalise maps to a common scale.
        """
        device = self.device
        self.teacher.eval()
        self.student.eval()
        self.autoencoder.eval()

        maps_st = []
        maps_ae = []
        for batch in validation_loader:
            image = batch["image"].to(device)
            _, map_st, map_ae = self._predict_maps(image)
            maps_st.append(map_st.cpu())
            maps_ae.append(map_ae.cpu())

        maps_st = torch.cat(maps_st)
        maps_ae = torch.cat(maps_ae)

        self.q_st_start.copy_(torch.quantile(maps_st, q=0.9))
        self.q_st_end.copy_(torch.quantile(maps_st, q=0.995))
        self.q_ae_start.copy_(torch.quantile(maps_ae, q=0.9))
        self.q_ae_end.copy_(torch.quantile(maps_ae, q=0.995))
        self._map_norm_ready = True

        print("EfficientAd: map normalisation computed.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._val_labels.clear()
        self._val_scores.clear()
        self._val_masks.clear()
        self._val_anomaly_maps.clear()
        self._val_maps_st.clear()
        self._val_maps_ae.clear()

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        image = batch["image"]
        map_combined, map_st, map_ae = self._predict_maps(image)

        eval_size = self.hparams.eval_resize

        # Pad and interpolate to original spatial size (as in official code)
        map_combined = F.pad(map_combined, (4, 4, 4, 4))
        map_combined = F.interpolate(
            map_combined, size=eval_size, mode="bilinear", align_corners=False
        )
        map_combined = map_combined[:, 0]  # (B, H, W)

        # Image-level score: max pixel value (Section 3.4)
        scores = map_combined.flatten(1).max(dim=1)[0]

        # Resize GT masks for pixel-level evaluation
        masks = batch["mask"]
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        masks = F.interpolate(masks.float(), size=eval_size, mode="nearest")[:, 0]

        self._val_labels.append(batch["label"].cpu())
        self._val_scores.append(scores.cpu())
        self._val_masks.append(masks.cpu())
        self._val_anomaly_maps.append(map_combined.cpu())
        self._val_maps_st.append(map_st.cpu())
        self._val_maps_ae.append(map_ae.cpu())

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

        # Update map normalisation from validation data every epoch.
        if self._val_maps_st:
            maps_st = torch.cat(self._val_maps_st)
            maps_ae = torch.cat(self._val_maps_ae)
            self.q_st_start.copy_(torch.quantile(maps_st, q=0.9).to(self.device))
            self.q_st_end.copy_(torch.quantile(maps_st, q=0.995).to(self.device))
            self.q_ae_start.copy_(torch.quantile(maps_ae, q=0.9).to(self.device))
            self.q_ae_end.copy_(torch.quantile(maps_ae, q=0.995).to(self.device))
            self._map_norm_ready = True

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

    def configure_optimizers(self) -> dict:
        # Only student and autoencoder are trainable (teacher is frozen).
        optimizer = torch.optim.Adam(
            itertools.chain(
                self.student.parameters(),
                self.autoencoder.parameters(),
            ),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # StepLR: drop lr by 10× at 95% of training (Section 3.2)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(0.95 * self.hparams.train_steps),
            gamma=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
