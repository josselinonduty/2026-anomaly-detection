"""WinCLIP(+): Zero-/Few-Shot Anomaly Classification and Segmentation.

Reference
---------
Jeong, J., Zou, Y., Kim, T., Zhang, D., Ravichandran, A., & Dabeer, O. (2023).
"WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." CVPR 2023.

This implementation follows the paper exactly:
- Backbone: ViT-B/16+ (ViT-B-16-plus-240) pre-trained via CLIP on LAION-400M.
- Compositional prompt ensemble on state words × template prompts (Sec. 3.1).
- Multi-scale window-based feature extraction at scales 2×2 and 3×3 (Sec. 3.2).
- Per-window CLIP visual features aligned with text via CLS token pooling.
- Zero-shot scoring: softmax over (normal, abnormal) text similarity, then
  harmonic aggregation across windows and scales (Sec. 3.2).
- Few-shot extension (WinCLIP+): builds a visual gallery from normal images
  and scores via max cosine similarity, fused with textual scores via
  harmonic mean (Sec. 3.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


# ── Prompt definitions (Section 3.1, Table 1) ───────────────────────
# State-level prompts: normal and abnormal descriptors composed with {object}
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

# Template-level prompts: photographic context wrappers
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


class WinCLIP(nn.Module):
    """WinCLIP(+) anomaly detection model.

    Parameters
    ----------
    backbone : str
        OpenCLIP model name. Default ``"ViT-B-16-plus-240"`` as in the paper.
    pretrained : str
        Pretrained dataset. Default ``"laion400m_e32"``.
    scales : tuple of int
        Window sizes (in patches) for multi-scale extraction.
        Default ``(2, 3)`` as in the paper.
    image_size : int
        Expected input resolution (after resize + crop). Default 240.
    """

    def __init__(
        self,
        backbone: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
        scales: tuple[int, ...] = (2, 3),
        image_size: int = 240,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.image_size = image_size

        # ── Load pre-trained CLIP model ──────────────────────────────
        model, _, _ = open_clip.create_model_and_transforms(
            backbone,
            pretrained=pretrained,
        )
        model.eval()
        self.clip = model

        # Freeze all parameters — WinCLIP never trains.
        for param in self.clip.parameters():
            param.requires_grad = False

        self.tokenizer = open_clip.get_tokenizer(backbone)

        # ── Extract ViT geometry from the visual encoder ─────────────
        vit = self.clip.visual
        self.patch_size: int = vit.conv1.kernel_size[0]  # type: ignore[index]
        self.grid_size: int = image_size // self.patch_size
        self.embed_dim: int = vit.conv1.out_channels  # transformer width

        # ── Pre-compute window masks ─────────────────────────────────
        self._build_window_masks()

        # ── Text features (populated by ``build_text_features``) ─────
        self.register_buffer("text_features", torch.empty(0))
        self._text_ready = False

        # ── Visual gallery (populated by WinCLIP+ ``build_visual_gallery``) ──
        self.visual_gallery: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Window mask generation
    # ------------------------------------------------------------------

    def _build_window_masks(self) -> None:
        """Pre-compute index masks for each window at each scale.

        For each scale s, windows of size s×s are slid over the grid_size×grid_size
        patch grid. Each mask stores the flattened patch indices belonging to
        that window.
        """
        G = self.grid_size
        index_grid = torch.arange(G * G, dtype=torch.long).reshape(G, G)

        masks: list[torch.Tensor] = []
        scale_begin: list[int] = []

        for s in self.scales:
            scale_begin.append(len(masks))
            for i in range(G - s + 1):
                for j in range(G - s + 1):
                    masks.append(index_grid[i : i + s, j : j + s].reshape(-1))

        self._masks = masks
        self._scale_begin = scale_begin
        self._n_windows_per_scale = [(self.grid_size - s + 1) ** 2 for s in self.scales]

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _encode_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings from CLIP's visual encoder (before transformer).

        Returns shape ``(N, grid_size**2, width)`` — no CLS token.
        """
        vit = self.clip.visual
        # conv1: (N, 3, H, W) -> (N, width, G, G)
        x = vit.conv1(images)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (N, width, G*G)
        x = x.permute(0, 2, 1)  # (N, G*G, width)
        # Add positional embedding (skip CLS token position at index 0)
        x = x + vit.positional_embedding.to(x.dtype)[1:, :]
        return x

    def _forward_windows(self, patch_tokens: torch.Tensor) -> list[torch.Tensor]:
        """Run windowed patch tokens through the CLIP ViT and return per-window CLS features.

        Parameters
        ----------
        patch_tokens : Tensor, shape (N, G*G, width)
            Patch embeddings (without CLS token).

        Returns
        -------
        list of Tensor
            One feature tensor per window, each of shape (N, embed_dim_out).
            Ordered: all windows for scale[0], then scale[1], etc.
        """
        vit = self.clip.visual
        device = patch_tokens.device
        dtype = patch_tokens.dtype
        N = patch_tokens.shape[0]

        all_features: list[torch.Tensor] = []

        for scale_idx, s in enumerate(self.scales):
            begin = self._scale_begin[scale_idx]
            end = (
                self._scale_begin[scale_idx + 1]
                if scale_idx + 1 < len(self._scale_begin)
                else len(self._masks)
            )
            scale_masks = self._masks[begin:end]
            n_win = len(scale_masks)

            if n_win == 0:
                continue

            # Gather window patches: (n_win, N, s*s, width)
            win_tokens = []
            for mask in scale_masks:
                mask_dev = mask.to(device)
                # (N, s*s, width)
                gathered = torch.gather(
                    patch_tokens,
                    dim=1,
                    index=mask_dev.unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(N, s * s, patch_tokens.shape[-1]),
                )
                win_tokens.append(gathered)

            # Stack and reshape for batch processing:
            # (n_win, N, s*s, W) -> (n_win * N, s*s, W)
            mx = torch.stack(win_tokens, dim=0)  # (n_win, N, s*s, W)
            mx = mx.reshape(n_win * N, s * s, mx.shape[-1])

            # Prepend CLS token with positional embedding[0]
            cls_token = (
                vit.class_embedding.to(dtype) + vit.positional_embedding.to(dtype)[0, :]
            )
            cls_tokens = cls_token.unsqueeze(0).expand(n_win * N, 1, -1)
            mx = torch.cat([cls_tokens, mx], dim=1)  # (n_win*N, s*s+1, W)

            # Pre-LN
            mx = vit.ln_pre(mx)

            # Transformer (expects LND format)
            mx = mx.permute(1, 0, 2)  # (s*s+1, n_win*N, W)
            mx = vit.transformer(mx)
            mx = mx.permute(1, 0, 2)  # (n_win*N, s*s+1, W)

            # CLS token pooling + post-LN + projection
            pooled = mx[:, 0, :]  # (n_win*N, W)
            pooled = vit.ln_post(pooled)
            if vit.proj is not None:
                pooled = pooled @ vit.proj  # (n_win*N, output_dim)

            # Reshape back: (n_win, N, output_dim)
            pooled = pooled.reshape(n_win, N, -1)

            # L2-normalize each feature
            pooled = F.normalize(pooled, dim=-1)

            # Append per-window features
            for w in range(n_win):
                all_features.append(pooled[w])  # (N, output_dim)

        return all_features

    # ------------------------------------------------------------------
    # Text feature construction (Section 3.1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_text_features(self, category: str) -> None:
        """Build compositional prompt ensemble text features.

        For each (template × state) combination, encode the text and average
        over all combinations to get a single normal and abnormal text feature.
        """
        device = next(self.clip.parameters()).device

        normal_phrases: list[str] = []
        abnormal_phrases: list[str] = []

        for template in TEMPLATES:
            for state in STATE_NORMAL:
                normal_phrases.append(template.format(state.format(category)))
            for state in STATE_ABNORMAL:
                abnormal_phrases.append(template.format(state.format(category)))

        # Tokenize and encode
        normal_tokens = self.tokenizer(normal_phrases).to(device)
        abnormal_tokens = self.tokenizer(abnormal_phrases).to(device)

        normal_feats = self.clip.encode_text(normal_tokens)  # (M, D)
        abnormal_feats = self.clip.encode_text(abnormal_tokens)  # (K, D)

        # Average over all prompt combinations
        avg_normal = normal_feats.mean(dim=0, keepdim=True)  # (1, D)
        avg_abnormal = abnormal_feats.mean(dim=0, keepdim=True)  # (1, D)

        # Concatenate and L2-normalize
        text_feats = torch.cat([avg_normal, avg_abnormal], dim=0)  # (2, D)
        text_feats = F.normalize(text_feats, dim=-1)
        self.text_features = text_feats
        self._text_ready = True

    # ------------------------------------------------------------------
    # Visual gallery construction (WinCLIP+ — Section 3.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_visual_gallery(self, normal_images: torch.Tensor) -> None:
        """Build per-scale visual feature galleries from normal reference images.

        Parameters
        ----------
        normal_images : Tensor, shape (K, 3, H, W)
            A batch of normal training images.
        """
        patch_tokens = self._encode_patches(normal_images)
        window_features = self._forward_windows(patch_tokens)

        # Group features by scale and concatenate across all normal images
        self.visual_gallery = []
        for scale_idx in range(len(self.scales)):
            begin = self._scale_begin[scale_idx]
            end = (
                self._scale_begin[scale_idx + 1]
                if scale_idx + 1 < len(self._scale_begin)
                else len(self._masks)
            )
            # Each window_features[i] is (K, D); stack and concatenate
            scale_feats = torch.cat(
                [window_features[i] for i in range(begin, end)],
                dim=0,
            )  # (n_windows * K, D)
            self.visual_gallery.append(scale_feats)

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def _textual_anomaly_map(
        self,
        window_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-patch anomaly scores from text-visual alignment.

        For each window, compute softmax over (normal, abnormal) text similarity.
        Anomaly score = 1/p_normal. Average across windows within a scale, then
        average across scales (Equation in Sec. 3.2).

        Returns shape (N, 1, grid_size, grid_size).
        """
        N = window_features[0].shape[0]
        G = self.grid_size

        scale_maps: list[torch.Tensor] = []

        for scale_idx, s in enumerate(self.scales):
            begin = self._scale_begin[scale_idx]
            end = (
                self._scale_begin[scale_idx + 1]
                if scale_idx + 1 < len(self._scale_begin)
                else len(self._masks)
            )

            token_scores = torch.zeros(N, G * G, device=window_features[0].device)
            token_weights = torch.zeros(N, G * G, device=window_features[0].device)

            for win_idx in range(begin, end):
                feats = window_features[win_idx]  # (N, D)
                # Similarity with text features: (N, 2)
                sim = 100.0 * feats @ self.text_features.T
                probs = sim.softmax(dim=-1)  # (N, 2)
                p_normal = probs[:, 0]  # (N,)

                # Anomaly score = 1 / p_normal
                score = 1.0 / p_normal  # (N,)

                # Assign to covered patches
                mask = self._masks[win_idx].to(feats.device)
                for patch_idx in mask:
                    token_scores[:, patch_idx] += score
                    token_weights[:, patch_idx] += 1.0

            # Average over overlapping windows
            token_scores = token_scores / token_weights.clamp(min=1.0)
            scale_maps.append(token_scores)

        # Average across scales
        scale_maps_t = torch.stack(scale_maps, dim=0)  # (n_scales, N, G*G)
        avg_map = scale_maps_t.mean(dim=0)  # (N, G*G)

        # Convert from 1/p_normal to anomaly probability: 1 - 1/(1/p_normal) = 1 - p_normal
        avg_map = 1.0 - 1.0 / avg_map

        return avg_map.reshape(N, 1, G, G)

    def _visual_anomaly_map(
        self,
        window_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-patch anomaly scores from visual gallery (WinCLIP+).

        For each window, find max cosine similarity to the corresponding
        gallery features. Score = 0.5 * (1 - max_sim).

        Returns shape (N, 1, grid_size, grid_size).
        """
        assert self.visual_gallery is not None
        N = window_features[0].shape[0]
        G = self.grid_size

        scale_maps: list[torch.Tensor] = []

        for scale_idx, s in enumerate(self.scales):
            begin = self._scale_begin[scale_idx]
            end = (
                self._scale_begin[scale_idx + 1]
                if scale_idx + 1 < len(self._scale_begin)
                else len(self._masks)
            )
            gallery = self.visual_gallery[scale_idx]  # (n_gallery, D)

            token_scores = torch.zeros(N, G * G, device=window_features[0].device)
            token_weights = torch.zeros(N, G * G, device=window_features[0].device)

            for win_idx in range(begin, end):
                feats = window_features[win_idx]  # (N, D)
                # Max cosine similarity to gallery
                sim = feats @ gallery.T  # (N, n_gallery)
                max_sim = sim.max(dim=1)[0]  # (N,)
                score = 0.5 * (1.0 - max_sim)  # (N,)

                mask = self._masks[win_idx].to(feats.device)
                for patch_idx in mask:
                    token_scores[:, patch_idx] += score
                    token_weights[:, patch_idx] += 1.0

            token_scores = token_scores / token_weights.clamp(min=1.0)
            scale_maps.append(token_scores)

        scale_maps_t = torch.stack(scale_maps, dim=0)
        avg_map = scale_maps_t.mean(dim=0)

        return avg_map.reshape(N, 1, G, G)

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run anomaly detection on a batch of images.

        Parameters
        ----------
        images : Tensor, shape (N, 3, H, W)

        Returns
        -------
        scores : Tensor, shape (N,)
            Image-level anomaly scores (max of anomaly map).
        anomaly_maps : Tensor, shape (N, 1, H, W)
            Pixel-level anomaly maps upsampled to input resolution.
        """
        assert self._text_ready, "Call build_text_features(category) first."

        H, W = images.shape[2], images.shape[3]

        patch_tokens = self._encode_patches(images)
        window_features = self._forward_windows(patch_tokens)

        # Textual anomaly map (always available — zero-shot)
        text_map = self._textual_anomaly_map(window_features)

        if self.visual_gallery is not None:
            # WinCLIP+: harmonic mean of textual and visual maps
            vis_map = self._visual_anomaly_map(window_features)
            # Harmonic mean: 1 / (1/a + 1/b) = ab / (a + b)
            # Clamp to avoid division by zero
            anomaly_map = 1.0 / (
                1.0 / text_map.clamp(min=1e-8) + 1.0 / vis_map.clamp(min=1e-8)
            )
        else:
            anomaly_map = text_map

        # Upsample to input resolution
        anomaly_map = F.interpolate(
            anomaly_map, size=(H, W), mode="bilinear", align_corners=False
        )

        # Image-level score = max of anomaly map
        scores = anomaly_map.reshape(images.shape[0], -1).max(dim=1)[0]

        return scores, anomaly_map.squeeze(1)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predict(images)
