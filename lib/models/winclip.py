"""WinCLIP(+): Zero-/Few-Shot Anomaly Classification and Segmentation.

Reference
---------
Jeong, J., Zou, Y., Kim, T., Zhang, D., Ravichandran, A., & Dabeer, O. (2023).
"WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." CVPR 2023.

This implementation follows the paper and the anomalib reference:
- Backbone: ViT-B/16+ (ViT-B-16-plus-240) pre-trained via CLIP on LAION-400M.
- Compositional prompt ensemble on state words x template prompts (Sec. 3.1).
- Multi-scale window-based feature extraction at scales 2x2 and 3x3 (Sec. 3.2).
- Per-window: masked tokens are re-run through the ViT transformer to obtain
  CLS-pooled CLIP embeddings for each sliding window location.
- Zero-shot scoring: cosine similarity / tau softmax class scores, harmonic
  aggregation within and across scales including full-image CLS (Sec. 3.2).
- Few-shot (WinCLIP+): visual association score (Eq. 4), fused with zero-shot
  scores via arithmetic mean (Sec. 3.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

# Temperature hyperparameter from the CLIP paper (tau = 0.07).
TEMPERATURE = 0.07

# -- Prompt definitions (Section 3.1, Table 1) --
STATE_NORMAL = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage",
]

STATE_ABNORMAL = [
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage",
]

TEMPLATES = [
    "a cropped photo of the {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
    "a close-up photo of the {}",
    "a bright photo of a {}",
    "a bright photo of the {}",
    "a dark photo of the {}",
    "a dark photo of a {}",
    "a jpeg corrupted photo of a {}",
    "a jpeg corrupted photo of the {}",
    "a blurry photo of the {}",
    "a blurry photo of a {}",
    "a photo of a {}",
    "a photo of the {}",
    "a photo of a small {}",
    "a photo of the small {}",
    "a photo of a large {}",
    "a photo of the large {}",
    "a photo of the {} for visual inspection",
    "a photo of a {} for visual inspection",
    "a photo of the {} for anomaly detection",
    "a photo of a {} for anomaly detection",
]


# -- Utility functions (matching anomalib) --


def _class_scores(
    embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = TEMPERATURE,
    target_class: int | None = None,
) -> torch.Tensor:
    """Cosine-similarity softmax class scores (Eq. 1).

    Parameters
    ----------
    embeddings : (N, D) or (B, N, D)
    text_embeddings : (M, D) -- typically M=2 for [normal, abnormal].
    temperature : float
    target_class : optional int -- return only that class's probability.

    Returns
    -------
    (..., M) or (...,) if target_class is set.
    """
    e = F.normalize(embeddings, dim=-1)
    t = F.normalize(text_embeddings, dim=-1)
    if e.ndim == 2:
        sim = e @ t.T
    else:
        sim = torch.bmm(
            e,
            t.unsqueeze(0).expand(e.shape[0], -1, -1).transpose(-2, -1),
        )
    scores = (sim / temperature).softmax(dim=-1)
    if target_class is not None:
        return scores[..., target_class]
    return scores


def _harmonic_aggregation(
    window_scores: torch.Tensor,
    grid_size: int,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Aggregate per-window scores to a per-patch map via harmonic mean.

    Parameters
    ----------
    window_scores : (B, n_windows)
    grid_size : int -- spatial size G (grid is GxG).
    masks : (s*s, n_windows) -- patch indices per window.

    Returns
    -------
    (B, G, G) score map.
    """
    B = window_scores.shape[0]
    device = window_scores.device
    scores: list[torch.Tensor] = []
    for idx in range(grid_size * grid_size):
        patch_mask = torch.any(masks == idx, dim=0)  # (n_windows,) bool
        n_covering = patch_mask.sum()
        if n_covering == 0:
            scores.append(torch.zeros(B, device=device))
        else:
            inv_sum = (1.0 / window_scores[:, patch_mask].clamp(min=1e-8)).sum(dim=1)
            scores.append(n_covering.float() / inv_sum)
    return (
        torch.stack(scores, dim=1)
        .reshape(B, grid_size, grid_size)
        .nan_to_num(posinf=0.0)
    )


def _visual_association_score(
    embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Visual association score (Eq. 4): min cosine distance / 2.

    Parameters
    ----------
    embeddings : (B, P, D)
    reference_embeddings : (K, P, D)

    Returns
    -------
    (B, P) -- higher = more anomalous.
    """
    ref_flat = reference_embeddings.reshape(-1, embeddings.shape[-1])
    e = F.normalize(embeddings, dim=-1)
    r = F.normalize(ref_flat, dim=-1)
    sim = torch.bmm(
        e,
        r.unsqueeze(0).expand(e.shape[0], -1, -1).transpose(-2, -1),
    )
    return (1.0 - sim).min(dim=-1)[0] / 2.0


# -- Main model --


class WinCLIP(nn.Module):
    """WinCLIP(+) anomaly detection model.

    Parameters
    ----------
    backbone : str
        OpenCLIP model name. Default ``"ViT-B-16-plus-240"``.
    pretrained : str
        Pretrained dataset. Default ``"laion400m_e32"``.
    scales : tuple of int
        Window sizes (in patches). Default ``(2, 3)``.
    image_size : int
        Expected input resolution. Default 240.
    win_chunk_size : int
        Max windows per transformer batch (memory vs speed). Default 32.
    use_half : bool
        Run in float16. Default True.
    """

    def __init__(
        self,
        backbone: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
        scales: tuple[int, ...] = (2, 3),
        image_size: int = 240,
        win_chunk_size: int = 32,
        use_half: bool = True,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.image_size = image_size
        self.win_chunk_size = win_chunk_size
        self.use_half = use_half

        model, _, _ = open_clip.create_model_and_transforms(
            backbone,
            pretrained=pretrained,
        )
        model.eval()
        self.clip = model

        for param in self.clip.parameters():
            param.requires_grad = False
        if use_half:
            self.clip = self.clip.half()

        # Enable output_tokens for encode_image
        self.clip.visual.output_tokens = True

        self.tokenizer = open_clip.get_tokenizer(backbone)

        vit = self.clip.visual
        self.patch_size: int = vit.conv1.kernel_size[0]
        self.grid_size: int = image_size // self.patch_size
        self.embed_dim: int = vit.conv1.out_channels

        # Multi-scale masks: list of (s*s, n_windows) per scale
        self._masks_per_scale = self._generate_masks()

        # Text features (2, D) -- populated by build_text_features
        self.register_buffer("text_features", torch.empty(0))
        self._text_ready = False

        # Visual gallery for WinCLIP+
        self._visual_embeddings: list[torch.Tensor] | None = None
        self._patch_embeddings: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Mask generation
    # ------------------------------------------------------------------

    def _generate_masks(self) -> list[torch.Tensor]:
        """Sliding-window masks per scale, shape ``(s*s, n_windows)``."""
        G = self.grid_size
        grid = torch.arange(G * G).reshape(1, G, G).float()
        masks = []
        for s in self.scales:
            m = F.unfold(grid.unsqueeze(0), kernel_size=s, stride=1)
            masks.append(m.squeeze(0).long())
        return masks

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _get_feature_map(self, images: torch.Tensor) -> torch.Tensor:
        """Extract pre-transformer feature map (CLS + patches + pos embed).

        Returns ``(N, 1+G*G, width)``.
        """
        vit = self.clip.visual
        dtype = vit.conv1.weight.dtype
        x = images.to(dtype)

        x = vit.conv1(x)  # (N, W, G, G)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (N, W, G*G)
        x = x.permute(0, 2, 1)  # (N, G*G, W)

        cls = vit.class_embedding.to(dtype)
        cls = cls.unsqueeze(0).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)  # (N, 1+G*G, W)
        x = x + vit.positional_embedding.to(dtype)
        return x

    def _run_transformer(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run tokens through patch_dropout -> ln_pre -> transformer -> ln_post -> pool -> proj.

        Parameters
        ----------
        tokens : (B, L, W)

        Returns
        -------
        (B, D) -- projected CLS embeddings.
        """
        vit = self.clip.visual
        x = vit.patch_dropout(tokens)
        x = vit.ln_pre(x)
        x = vit.transformer(x)
        x = vit.ln_post(x)
        pooled, _ = vit._global_pool(x)
        if vit.proj is not None:
            pooled = pooled @ vit.proj
        return pooled

    def _get_window_embeddings(
        self,
        feature_map: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Compute embeddings for each window position at one scale.

        Masks and applies the transformer to each sliding-window subset,
        matching the anomalib reference implementation.

        Parameters
        ----------
        feature_map : (N, 1+G*G, W) -- pre-transformer features with CLS.
        masks : (s*s, n_windows) -- patch indices per window.

        Returns
        -------
        (N, n_windows, D) -- per-window embeddings.
        """
        N = feature_map.shape[0]
        n_windows = masks.shape[1]
        device = feature_map.device

        # Prepend CLS index (0) to each mask; shift patch indices by +1
        cls_idx = torch.zeros(1, n_windows, dtype=torch.long, device=device)
        full_masks = torch.cat(
            [cls_idx, masks.to(device) + 1], dim=0
        ).T  # (n_win, s*s+1)

        all_pooled: list[torch.Tensor] = []

        for chunk_start in range(0, n_windows, self.win_chunk_size):
            chunk_end = min(chunk_start + self.win_chunk_size, n_windows)
            chunk_masks = full_masks[chunk_start:chunk_end]  # (n_chunk, s*s+1)

            # Gather masked tokens -> (n_chunk * N, s*s+1, W)
            masked = torch.cat(
                [torch.index_select(feature_map, 1, m) for m in chunk_masks],
                dim=0,
            )

            pooled = self._run_transformer(masked)  # (n_chunk * N, D)
            n_chunk = chunk_masks.shape[0]
            pooled = pooled.reshape(n_chunk, N, -1)
            all_pooled.append(pooled)

        # (n_windows, N, D) -> (N, n_windows, D)
        return torch.cat(all_pooled, dim=0).permute(1, 0, 2)

    @torch.no_grad()
    def encode_image(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode images -> image, window, and patch embeddings.

        Returns
        -------
        image_embeddings : (N, D)
        window_embeddings : list of (N, n_windows, D), one per scale
        patch_embeddings : (N, G*G, D)
        """
        feature_map = self._get_feature_map(images)  # (N, 1+G*G, W)

        # Full-image forward -> image embeddings + patch embeddings
        vit = self.clip.visual
        x = vit.patch_dropout(feature_map)
        x = vit.ln_pre(x)
        x = vit.transformer(x)  # (N, 1+G*G, W)
        x = vit.ln_post(x)
        image_emb, _ = vit._global_pool(x)
        if vit.proj is not None:
            image_emb = image_emb @ vit.proj  # (N, D)
            patch_emb = x[:, 1:, :] @ vit.proj  # (N, G*G, D)
        else:
            patch_emb = x[:, 1:, :]

        # Window embeddings per scale (re-runs transformer on masked subsets)
        window_embs = [
            self._get_window_embeddings(feature_map, masks)
            for masks in self._masks_per_scale
        ]

        return image_emb, window_embs, patch_emb

    # ------------------------------------------------------------------
    # Text features (Sec. 3.1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_text_features(self, category: str) -> None:
        """Build compositional prompt ensemble text features."""
        device = next(self.clip.parameters()).device

        normal_phrases: list[str] = []
        abnormal_phrases: list[str] = []
        for template in TEMPLATES:
            for state in STATE_NORMAL:
                normal_phrases.append(template.format(state.format(category)))
            for state in STATE_ABNORMAL:
                abnormal_phrases.append(template.format(state.format(category)))

        normal_tokens = self.tokenizer(normal_phrases).to(device)
        abnormal_tokens = self.tokenizer(abnormal_phrases).to(device)

        normal_feats = self.clip.encode_text(normal_tokens)
        abnormal_feats = self.clip.encode_text(abnormal_tokens)

        avg_normal = normal_feats.mean(dim=0, keepdim=True)
        avg_abnormal = abnormal_feats.mean(dim=0, keepdim=True)

        # Store WITHOUT pre-normalisation; _class_scores normalises at score time
        self.text_features = torch.cat([avg_normal, avg_abnormal], dim=0)
        self._text_ready = True

    # ------------------------------------------------------------------
    # Visual gallery (WinCLIP+ -- Sec. 3.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_visual_gallery(self, normal_images: torch.Tensor) -> None:
        """Build per-scale visual + patch embeddings from normal images.

        Stores:
        - ``_visual_embeddings``: list of (K, n_windows, D) per scale.
        - ``_patch_embeddings``: (K, G*G, D).
        """
        K = normal_images.shape[0]
        all_window_embs: list[list[torch.Tensor]] = [[] for _ in self.scales]
        all_patch_embs: list[torch.Tensor] = []

        for i in range(K):
            img = normal_images[i : i + 1]
            _, win_embs, patch_emb = self.encode_image(img)
            for s_idx, we in enumerate(win_embs):
                all_window_embs[s_idx].append(we.cpu())
            all_patch_embs.append(patch_emb.cpu())

        device = normal_images.device
        self._visual_embeddings = [
            torch.cat(wes, dim=0).to(device) for wes in all_window_embs
        ]
        self._patch_embeddings = torch.cat(all_patch_embs, dim=0).to(device)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_zero_shot_scores(
        self,
        image_scores: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Zero-shot multi-scale anomaly map (Sec. 3.2).

        Full-image CLS score is an additional scale level. Harmonic aggregation
        within each scale; harmonic mean across scales.

        Returns (N, G, G).
        """
        G = self.grid_size
        multi_scale: list[torch.Tensor] = [
            image_scores.view(-1, 1, 1).expand(-1, G, G),
        ]

        for win_emb, masks in zip(window_embeddings, self._masks_per_scale):
            scores = _class_scores(
                win_emb,
                self.text_features,
                TEMPERATURE,
                target_class=1,
            )
            multi_scale.append(
                _harmonic_aggregation(scores, G, masks.to(win_emb.device)),
            )

        stacked = torch.stack(multi_scale, dim=0)
        n_scales = stacked.shape[0]
        return n_scales / (1.0 / stacked.clamp(min=1e-8)).sum(dim=0)

    def _compute_few_shot_scores(
        self,
        patch_embeddings: torch.Tensor,
        window_embeddings: list[torch.Tensor],
    ) -> torch.Tensor:
        """Few-shot multi-scale anomaly map (Sec. 3.3).

        Uses visual association score (Eq. 4) at each scale, aggregated via
        harmonic mean within scales, arithmetic mean across scales.

        Returns (N, G, G).
        """
        G = self.grid_size
        assert self._visual_embeddings is not None
        assert self._patch_embeddings is not None

        multi_scale: list[torch.Tensor] = [
            _visual_association_score(
                patch_embeddings,
                self._patch_embeddings,
            ).reshape(-1, G, G),
        ]

        for win_emb, ref_emb, masks in zip(
            window_embeddings,
            self._visual_embeddings,
            self._masks_per_scale,
        ):
            scores = _visual_association_score(win_emb, ref_emb)
            multi_scale.append(
                _harmonic_aggregation(scores, G, masks.to(win_emb.device)),
            )

        return torch.stack(multi_scale, dim=0).mean(dim=0)

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run anomaly detection on a batch of images.

        Processes one image at a time to bound peak memory.

        Parameters
        ----------
        images : (N, 3, H, W)

        Returns
        -------
        scores : (N,) image-level anomaly scores.
        anomaly_maps : (N, H, W) pixel-level maps upsampled to input resolution.
        """
        assert self._text_ready, "Call build_text_features(category) first."
        N, _, H, W = images.shape

        all_scores: list[torch.Tensor] = []
        all_maps: list[torch.Tensor] = []

        for i in range(N):
            img = images[i : i + 1]
            image_emb, win_embs, patch_emb = self.encode_image(img)

            # Full-image class score
            image_score = _class_scores(
                image_emb,
                self.text_features,
                TEMPERATURE,
                target_class=1,
            )

            # Zero-shot anomaly map
            zs_map = self._compute_zero_shot_scores(image_score, win_embs)

            if self._visual_embeddings is not None:
                fs_map = self._compute_few_shot_scores(patch_emb, win_embs)
                anomaly_map = (zs_map + fs_map) / 2.0
                image_score = (image_score + fs_map.amax(dim=(-2, -1))) / 2.0
            else:
                anomaly_map = zs_map

            anomaly_map = F.interpolate(
                anomaly_map.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

            all_scores.append(image_score.view(-1).cpu())
            all_maps.append(anomaly_map.squeeze(0).squeeze(0).cpu())

        scores = torch.cat(all_scores, dim=0).to(images.device)
        anomaly_maps = torch.stack(all_maps, dim=0).to(images.device)
        return scores, anomaly_maps

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predict(images)
