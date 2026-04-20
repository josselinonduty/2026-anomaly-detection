"""PyTorch Lightning module for DictAS (arXiv:2508.13560).

Implements the full DictAS training / evaluation pipeline:

- Online self-supervised construction of (query, reference) pairs from
  raw images (Appendix A.2, Listing 1, Algorithm 2).
- Query loss ``L_q`` (Eq. 7) + Contrastive Query Constraint ``L_CQC``
  (Eq. 8) + Text Alignment Constraint ``L_TAC`` (Eq. 9).
- Total loss ``L = L_q + λ_1 L_CQC + λ_2 L_TAC`` (Eq. 10) with
  ``λ_1 = λ_2 = 0.1`` by default.
- Few-shot inference: the dictionary is built from k normal reference
  images and the anomaly map is obtained via Eq. 11.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from lib.models.dictas import DictAS
from lib.utils.metrics import compute_auroc, compute_pixel_auroc


# ── Data transformation (reference image) — Appendix A.2 / Listing 1 ─
# We implement these on CPU tensors with torchvision-like semantics
# because we operate on (B, 3, H, W) tensors already normalised by the
# data module.  Keeping the probabilities identical to the paper.


def _reference_transform(image: torch.Tensor) -> torch.Tensor:
    """Apply a random geometric + occlusion transformation to a tensor image.

    Mirrors Listing 1 of the paper (Appendix A.2):

    - ``RandomRotate90`` (p=1)
    - ``Rotate`` 30°–270°           (p=1)
    - ``HorizontalFlip``            (p=0.5)
    - ``VerticalFlip``              (p=0.5)
    - ``GridDropout`` ratio=0.3     (p=0.5)
    - ``CoarseDropout`` 8 holes     (p=0.5)
    """
    x = image

    # RandomRotate90 — always
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k=k, dims=(-2, -1))

    # Rotate 30..270 — always, arbitrary angle by sampling
    angle = random.uniform(30.0, 270.0)
    x = _rotate(x, angle)

    if random.random() < 0.5:
        x = torch.flip(x, dims=(-1,))
    if random.random() < 0.5:
        x = torch.flip(x, dims=(-2,))
    if random.random() < 0.5:
        x = _grid_dropout(x, ratio=0.3)
    if random.random() < 0.5:
        x = _coarse_dropout(x, n_holes=8, max_h=32, max_w=32)

    return x


def _rotate(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate an image tensor around its centre (bilinear, reflection pad)."""
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    affine = torch.tensor(
        [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=x.dtype, device=x.device
    ).unsqueeze(0)
    grid = F.affine_grid(affine, size=[1, *x.shape[-3:]], align_corners=False)
    rotated = F.grid_sample(
        x.unsqueeze(0), grid, padding_mode="reflection", align_corners=False
    )
    return rotated.squeeze(0)


def _grid_dropout(
    x: torch.Tensor, ratio: float = 0.3, n_cells: int = 6
) -> torch.Tensor:
    """Zero out a regular grid of patches covering ``ratio`` of the image."""
    c, h, w = x.shape
    cell_h = max(1, h // n_cells)
    cell_w = max(1, w // n_cells)
    drop_h = max(1, int(cell_h * ratio))
    drop_w = max(1, int(cell_w * ratio))

    mask = torch.ones(1, h, w, device=x.device, dtype=x.dtype)
    for i in range(0, h, cell_h):
        for j in range(0, w, cell_w):
            mask[:, i : i + drop_h, j : j + drop_w] = 0
    return x * mask


def _coarse_dropout(
    x: torch.Tensor, n_holes: int, max_h: int, max_w: int
) -> torch.Tensor:
    """Randomly drop ``n_holes`` rectangles from the image."""
    _, h, w = x.shape
    out = x.clone()
    for _ in range(n_holes):
        hh = random.randint(1, max_h)
        ww = random.randint(1, max_w)
        y0 = random.randint(0, max(0, h - hh))
        x0 = random.randint(0, max(0, w - ww))
        out[:, y0 : y0 + hh, x0 : x0 + ww] = 0
    return out


# ── DRÆM-style anomaly synthesis (Algorithm 2) ───────────────────────
# We approximate the anomaly source A with another image sampled from
# the same batch — this avoids a hard dependency on DTD while keeping
# the statistics of Berlin-noise-modulated blending unchanged.


def _perlin_noise(h: int, w: int, device: torch.device) -> torch.Tensor:
    """2-D Perlin-style noise (value-noise octave sum).

    Returns a tensor of shape ``(h, w)`` in ``[0, 1]``.
    """
    octaves = 4
    persistence = 0.5
    noise = torch.zeros(h, w, device=device)
    amplitude = 1.0
    max_amp = 0.0
    for octave in range(octaves):
        freq = 2 ** (octave + 2)
        # Low-res value noise upsampled bilinearly.
        low = torch.rand(1, 1, freq, freq, device=device)
        up = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
        noise = noise + amplitude * up[0, 0]
        max_amp += amplitude
        amplitude *= persistence
    noise = noise / max_amp
    # Normalise to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def synthesize_anomaly(
    images: torch.Tensor,
    threshold: float = 0.5,
    beta: float = 0.5,
    max_tries: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Algorithm 2 of the paper — online DRÆM-style anomaly synthesis.

    Parameters
    ----------
    images : Tensor, shape (B, 3, H, W)
        Normal images on which to synthesise anomalies.
    threshold : float
        Perlin-noise binarisation threshold ``λ``.
    beta : float
        Blending coefficient ``γ`` between the anomaly source and the
        raw image (Algorithm 2, line 5).

    Returns
    -------
    x_q : Tensor, shape (B, 3, H, W)
        Query image with synthesised anomalies.
    G : Tensor, shape (B, H, W)
        Pixel-level pseudo-label ``G ∈ {0, 1}`` (1 = anomalous).
    y_q : Tensor, shape (B,)
        Image-level pseudo-label (0 = normal, 1 = contains anomaly).
    """
    b, c, h, w = images.shape
    device = images.device
    dtype = images.dtype

    # Anomaly source = another image shuffled from the batch.
    perm = torch.randperm(b, device=device)
    anomaly_src = images[perm]

    out = images.clone()
    masks = torch.zeros(b, h, w, device=device, dtype=dtype)

    for i in range(b):
        # Re-draw Perlin noise until the mask is non-empty (Algorithm 2).
        for _ in range(max_tries):
            noise = _perlin_noise(h, w, device)
            g = (noise > threshold).to(dtype)
            if g.sum() > 0:
                break
        gamma = random.uniform(0.2, 1.0) * beta + 0.3  # vary blend weight
        mask = g.unsqueeze(0)  # (1, H, W)
        # X_q = γ · (M_A ⊙ A) + (1 - γ) · (M_A ⊙ X) + M_A_bar ⊙ X
        blended = gamma * (mask * anomaly_src[i]) + (1.0 - gamma) * (mask * images[i])
        out[i] = blended + (1.0 - mask) * images[i]
        masks[i] = g

    y_q = (masks.flatten(1).sum(dim=1) > 0).long()
    return out, masks, y_q


# ── Lightning module ─────────────────────────────────────────────────


class DictASModule(LightningModule):
    """Lightning wrapper for the DictAS self-supervised FSAS model.

    Parameters
    ----------
    category : str
        Object category used for text prompt ensembling.
    backbone, pretrained, image_size, layer_indices, pool_kernel, lookup
        Forwarded to :class:`DictAS`.
    k_shot : int
        Number of normal reference images used at inference (``k ≥ 1``).
    lambda_cqc, lambda_tac : float
        Regularisation weights for ``L_CQC`` and ``L_TAC`` (Eq. 10).
        Default 0.1 / 0.1 (paper default).
    lr : float
        Adam learning rate (default 1e-4, Appendix A.4).
    """

    def __init__(
        self,
        category: str = "candle",
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        image_size: int = 336,
        layer_indices: tuple[int, ...] = (6, 12, 18, 24),
        pool_kernel: int = 3,
        lookup: str = "sparse",
        k_shot: int = 4,
        lambda_cqc: float = 0.1,
        lambda_tac: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = DictAS(
            backbone=backbone,
            pretrained=pretrained,
            image_size=image_size,
            layer_indices=layer_indices,
            pool_kernel=pool_kernel,
            lookup=lookup,  # type: ignore[arg-type]
        )

        self.category = category
        self.k_shot = k_shot

        # Reference gallery for inference: lazily filled from the training
        # loader, then reused during evaluation (no retraining required).
        self._reference_images: torch.Tensor | None = None

        # Metric accumulators.
        self._val_labels: list[torch.Tensor] = []
        self._val_scores: list[torch.Tensor] = []
        self._val_masks: list[torch.Tensor] = []
        self._val_anomaly_maps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Forward & text features
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        self.model.build_text_features(self.category)

    def on_test_start(self) -> None:
        if not self.model._text_ready:
            self.model.build_text_features(self.category)

    def on_validation_start(self) -> None:
        if not self.model._text_ready:
            self.model.build_text_features(self.category)

    # ------------------------------------------------------------------
    # Self-supervised training
    # ------------------------------------------------------------------

    def _make_reference_batch(self, images: torch.Tensor) -> torch.Tensor:
        """Generate one reference image per raw image via Appendix A.2 transforms.

        Returns a tensor of shape ``(B, 1, 3, H, W)`` — ``k = 1`` during
        training as prescribed by the paper (Sec. 3.5).
        """
        refs = torch.stack([_reference_transform(img) for img in images], dim=0)
        return refs.unsqueeze(1)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        raw = batch["image"]  # (B, 3, H, W) — treated as "normal" raw images

        # Cache the first k_shot normal images from training as the
        # inference-time reference gallery.  This guarantees the gallery is
        # populated even when ``trainer.datamodule`` is not set (e.g. when
        # ``trainer.fit`` is called with explicit dataloaders).
        if self._reference_images is None:
            refs = raw[: self.k_shot].detach().clone()
            if refs.shape[0] < self.k_shot:
                # Not enough samples in this batch — pad by repeating.
                pad = self.k_shot - refs.shape[0]
                refs = torch.cat([refs, refs[:1].expand(pad, -1, -1, -1)], dim=0)
            self._reference_images = refs

        # ── Online query / reference construction ────────────────────
        x_q, mask_g, y_q = synthesize_anomaly(raw)
        refs = self._make_reference_batch(raw)  # (B, 1, 3, H, W)

        # ── Forward through DictAS ───────────────────────────────────
        out = self.model(x_q, refs)
        F_q_list, F_r_list = out["F_q"], out["F_r"]
        x_q_emb, x_r_emb = out["x_q"], out["x_r"]

        # ── Build a per-token normal/abnormal mask (Eq. 7) ──────────
        g = self.model.grid_size
        # Down-sample pseudo-label to patch resolution.
        with torch.no_grad():
            mask_tok = F.interpolate(mask_g.unsqueeze(1), size=(g, g), mode="nearest")
            mask_tok = (
                (mask_tok > 0.5).float().reshape(mask_tok.shape[0], -1)
            )  # (B, HW)

        # ── Query loss L_q (Eq. 7) ───────────────────────────────────
        L_q = torch.tensor(0.0, device=raw.device)
        L_cqc = torch.tensor(0.0, device=raw.device)
        for F_q, F_r in zip(F_q_list, F_r_list):
            d = 1.0 - F.cosine_similarity(F_q, F_r, dim=-1)  # (B, HW)
            n_mask = 1.0 - mask_tok  # normal patches
            a_mask = mask_tok  # anomalous patches

            n_sum = n_mask.sum(dim=1).clamp(min=1.0)
            a_sum = a_mask.sum(dim=1)

            E_N = (d * n_mask).sum(dim=1) / n_sum
            L_q = L_q + E_N.mean()

            # Contrastive Query Constraint (Eq. 8) — only for samples that
            # actually contain synthesised anomalies.
            valid = (a_sum > 0).float()
            if valid.sum() > 0:
                E_A = (d * a_mask).sum(dim=1) / a_sum.clamp(min=1.0)
                diff = (E_N - E_A).clamp(min=0)
                L_cqc = L_cqc + (diff * valid).sum() / valid.sum().clamp(min=1.0)

        # ── Text Alignment Constraint (Eq. 9) ───────────────────────
        #  ℓ = CE( <x_r, F_text>, 0 ) + CE( <x_q, F_text>, y_q )
        text_feats = self.model.text_features  # (2, D)
        logits_r = 100.0 * x_r_emb @ text_feats.T  # (B, 2)
        logits_q = 100.0 * x_q_emb @ text_feats.T
        zeros = torch.zeros(raw.shape[0], dtype=torch.long, device=raw.device)
        L_tac = F.cross_entropy(logits_r, zeros) + F.cross_entropy(logits_q, y_q)

        # ── Total loss (Eq. 10) ──────────────────────────────────────
        loss = L_q + self.hparams.lambda_cqc * L_cqc + self.hparams.lambda_tac * L_tac

        self.log("train/L_q", L_q, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/L_cqc", L_cqc, on_step=True, on_epoch=True)
        self.log("train/L_tac", L_tac, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # Reference gallery construction (few-shot inference)
    # ------------------------------------------------------------------

    def _gather_reference_images(self) -> None:
        """Collect ``k_shot`` normal images from the training loader.

        Called lazily at the start of the first validation/test epoch.
        Uses the LightningModule ``trainer.datamodule`` when available.
        """
        if self._reference_images is not None:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None or trainer.datamodule is None:  # type: ignore[union-attr]
            return

        dm = trainer.datamodule  # type: ignore[union-attr]
        dm.setup("fit")
        loader = dm.train_dataloader()
        images: list[torch.Tensor] = []
        for batch in loader:
            images.append(batch["image"])
            if sum(img.shape[0] for img in images) >= self.k_shot:
                break
        refs = torch.cat(images, dim=0)[: self.k_shot].to(self.device)
        self._reference_images = refs

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._gather_reference_images()
        self._val_labels.clear()
        self._val_scores.clear()
        self._val_masks.clear()
        self._val_anomaly_maps.clear()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if self._reference_images is None:
            return
        x = batch["image"]
        # Expand references to the query batch.
        refs = self._reference_images.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
        scores, anomaly_map = self.model.predict(x, refs)

        self._val_labels.append(batch["label"].cpu())
        self._val_scores.append(scores.cpu())
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

    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    # ------------------------------------------------------------------
    # Optimiser  (Adam, lr=1e-4 — Appendix A.4)
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> torch.optim.Optimizer:
        trainable = nn.ModuleList(
            [self.model.g_Q, self.model.g_K, self.model.g_V, self.model.text_proj]
        )
        return torch.optim.Adam(
            trainable.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_checkpoint(self, ckpt_dir: str | Path) -> None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "hparams": dict(self.hparams),
            "g_Q": self.model.g_Q.state_dict(),
            "g_K": self.model.g_K.state_dict(),
            "g_V": self.model.g_V.state_dict(),
            "text_proj": self.model.text_proj.state_dict(),
            "text_features": self.model.text_features,
            "text_ready": self.model._text_ready,
            "reference_images": self._reference_images,
        }
        torch.save(state, ckpt_dir / "model.ckpt")

    @classmethod
    def load_checkpoint(
        cls, ckpt_dir: str | Path, map_location: str = "cpu"
    ) -> "DictASModule":
        ckpt_dir = Path(ckpt_dir)
        state = torch.load(ckpt_dir / "model.ckpt", map_location=map_location)
        module = cls(**state["hparams"])
        module.model.g_Q.load_state_dict(state["g_Q"])
        module.model.g_K.load_state_dict(state["g_K"])
        module.model.g_V.load_state_dict(state["g_V"])
        module.model.text_proj.load_state_dict(state["text_proj"])
        module.model.text_features = state["text_features"]
        module.model._text_ready = state["text_ready"]
        module._reference_images = state.get("reference_images")
        return module
