"""DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation
via Dictionary Lookup.

Reference
---------
Qu, Z., Tao, X., Gong, X., Qu, S., Zhang, X., Wang, X., Shen, F., Zhang, Z.,
Prasad, M., & Ding, G. (2025).
"DictAS: A Framework for Class-Generalizable Few-Shot Anomaly Segmentation
via Dictionary Lookup."  arXiv:2508.13560

Implementation faithful to the paper:

- Backbone: CLIP ViT-L/14-336 (OpenAI), frozen.  Intermediate patch features
  are taken from the 6-th, 12-th, 18-th and 24-th transformer blocks
  (Appendix A.4).  An average-pooling with kernel 3 is applied on the
  patch grid to increase the receptive field (Appendix A.4).
- Dictionary Construction (Sec. 3.2): three learnable ``AttnBlock``s
  (Key, Query, Value generators) built from a multi-head self-attention
  followed by a two-layer MLP (Eq. 4).  The Value generator has an
  additional external residual connection (Eq. 3).
- Dictionary Lookup (Sec. 3.3): Query–Key matching ``z = F_Q F_K^T``
  followed by a Sparse Probability Module (sparsemax, Eq. 6).  Maximum
  and dense lookups are also provided.
- Text Alignment head (Sec. 3.4): multi-layer features are concatenated
  along the channel dimension, globally average-pooled, then linearly
  projected to the CLIP joint embedding space for alignment with the
  normal/abnormal text prototypes.
- Final anomaly map (Eq. 11): mean cosine distance between query and
  retrieved-result features over all ``L`` layers, upsampled to input
  resolution.
"""

from __future__ import annotations

from typing import Literal

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Prompt templates (Appendix A.3, Fig. 7) ──────────────────────────

_TEMPLATES: tuple[str, ...] = (
    "a photo of a {state}{cls}.",
    "a good photo of the {state}{cls}.",
    "a photo of my {state}{cls}.",
    "a photo of the {state}{cls}.",
    "a photo of a/the {state}{cls}.",
    "a photo of a/the small {state}{cls}.",
    "a bad photo of a/the {state}{cls}.",
    "a low resolution photo of a/the {state}{cls}.",
    "a cropped photo of a/the {state}{cls}.",
    "a bright photo of a/the {state}{cls}.",
    "a dark photo of a/the {state}{cls}.",
    "a black and white photo of a/the {state}{cls}.",
    "a jpeg corrupted photo of a/the {state}{cls}.",
    "a close-up photo of the {state}{cls}.",
    "There is a/the {state}{cls} in the scene.",
    "This is one {state}{cls} in the scene.",
)

_NORMAL_STATES: tuple[str, ...] = (
    "flawless {cls}",
    "perfect {cls}",
    "unblemished {cls}",
    "normal {cls}",
    "{cls} without flaw",
    "{cls} without defect",
    "{cls} without damage",
)

_ABNORMAL_STATES: tuple[str, ...] = (
    "damaged {cls}",
    "broken {cls}",
    "abnormal {cls}",
    "imperfect {cls}",
    "{cls} with flaw",
    "{cls} with defect",
    "{cls} with damage",
)


def _expand_prompts(category: str) -> tuple[list[str], list[str]]:
    """Build normal / abnormal prompt ensembles for a category."""
    normal, abnormal = [], []
    for template in _TEMPLATES:
        for state in _NORMAL_STATES:
            normal.append(
                template.format(state=state.format(cls=category) + " ", cls="")
            )
        for state in _ABNORMAL_STATES:
            abnormal.append(
                template.format(state=state.format(cls=category) + " ", cls="")
            )
    return normal, abnormal


# ── Sparse Probability Module (Eq. 6, Algorithm 1) ───────────────────


def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: projection onto the probability simplex (Martins & Astudillo 2016).

    Given a score vector ``z``, returns the sparse probability distribution
    ``w = max(z - tau, 0)`` with ``tau`` the unique threshold such that the
    output sums to 1 along ``dim``.
    """
    # Move target dim to the last position for convenience.
    z = z.transpose(dim, -1)
    orig_shape = z.shape
    z_flat = z.reshape(-1, orig_shape[-1])

    z_sorted, _ = torch.sort(z_flat, dim=-1, descending=True)
    cumsum = z_sorted.cumsum(dim=-1)
    k = torch.arange(1, orig_shape[-1] + 1, device=z.device, dtype=z.dtype).unsqueeze(0)

    # support = { k : 1 + k * z_sorted[k] > cumsum[k] }
    support = (1.0 + k * z_sorted) > cumsum
    k_z = support.to(z.dtype).sum(dim=-1, keepdim=True).clamp(min=1)

    # tau = (cumsum[k_z] - 1) / k_z  (Alg. 1, line 3-4)
    idx = (k_z.long() - 1).clamp(min=0)
    tau = (cumsum.gather(-1, idx) - 1.0) / k_z

    out = (z_flat - tau).clamp(min=0)
    out = out.view(orig_shape).transpose(dim, -1)
    return out


# ── Self-attention-based AttnBlock (Eq. 4) ───────────────────────────


class AttnBlock(nn.Module):
    """Single self-attention-based transformer block for feature adaptation.

    Implements ``F_out = TwoLayerMLP( softmax(Q K^T / sqrt(C)) V )``
    exactly as in Eq. (4) of the paper: multi-head self-attention with a
    projection-free output followed by a two-layer MLP.  No internal
    residual connections nor layer-norms are added here — the external
    residual around the Value generator is applied in the main module
    (Eq. 3).
    """

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, C)  →  (B, N, C)."""
        b, n, c = x.shape
        q = self.q_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.mlp(out)


# ── Main DictAS model ────────────────────────────────────────────────


class DictAS(nn.Module):
    """DictAS anomaly segmentation model.

    Parameters
    ----------
    backbone : str
        OpenCLIP model name.  Default ``"ViT-L-14-336"``.
    pretrained : str
        Pre-trained weights tag.  Default ``"openai"``.
    image_size : int
        Input resolution.  Default 336.
    layer_indices : tuple of int
        1-based indices of the transformer blocks whose patch features are
        extracted (Appendix A.4: 6, 12, 18, 24 for ViT-L/14).
    pool_kernel : int
        Average-pooling kernel applied on the patch grid to increase the
        receptive field (Appendix A.4).
    num_heads : int
        Multi-head count for every ``AttnBlock`` (Eq. 4).
    mlp_ratio : float
        Hidden size multiplier for the two-layer MLP.
    lookup : {'sparse', 'dense', 'max'}
        Dictionary lookup strategy (Eq. 5).  Default ``'sparse'`` (SPM).
    """

    def __init__(
        self,
        backbone: str = "ViT-L-14-336",
        pretrained: str = "openai",
        image_size: int = 336,
        layer_indices: tuple[int, ...] = (6, 12, 18, 24),
        pool_kernel: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        lookup: Literal["sparse", "dense", "max"] = "sparse",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.layer_indices = tuple(sorted(layer_indices))
        self.pool_kernel = pool_kernel
        self.lookup = lookup

        # ── CLIP backbone (frozen) ───────────────────────────────────
        model, _, _ = open_clip.create_model_and_transforms(
            backbone, pretrained=pretrained
        )
        model.eval()
        self.clip = model
        for p in self.clip.parameters():
            p.requires_grad = False

        self.tokenizer = open_clip.get_tokenizer(backbone)

        # ── ViT geometry ─────────────────────────────────────────────
        vit = self.clip.visual
        self.patch_size: int = vit.conv1.kernel_size[0]  # type: ignore[index]
        self.grid_size: int = image_size // self.patch_size
        self.feature_dim: int = vit.conv1.out_channels  # transformer width
        # Joint embedding dim (projection target)
        self.embed_dim: int = (
            vit.proj.shape[1] if vit.proj is not None else self.feature_dim
        )

        n_blocks = len(vit.transformer.resblocks)
        for idx in self.layer_indices:
            if idx < 1 or idx > n_blocks:
                raise ValueError(
                    f"layer_indices must be within [1, {n_blocks}], got {idx}"
                )

        # ── Learnable generators (shared across layers) ──────────────
        # Shared parameters keep the model compact and match the
        # paper's single-generator formulation applied per layer.
        self.g_Q = AttnBlock(self.feature_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.g_K = AttnBlock(self.feature_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
        self.g_V = AttnBlock(self.feature_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

        # ── Text-alignment projector (Sec. 3.4) ──────────────────────
        # Concatenate L layer features along channel dim → linear → joint emb.
        self.text_proj = nn.Linear(
            self.feature_dim * len(self.layer_indices), self.embed_dim
        )

        # Cached text features (built lazily by ``build_text_features``).
        self.register_buffer(
            "text_features",
            torch.zeros(2, self.embed_dim),
            persistent=False,
        )
        self._text_ready = False

    # ------------------------------------------------------------------
    # CLIP feature extraction
    # ------------------------------------------------------------------

    def _forward_visual_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run the CLIP ViT and return patch features at the selected layers.

        Parameters
        ----------
        x : Tensor, shape (B, 3, H, W)

        Returns
        -------
        list of Tensor, each (B, grid_size**2, feature_dim)
            Patch tokens (CLS excluded) after the requested blocks.
        """
        vit = self.clip.visual
        dtype = x.dtype

        # Patch embedding — mirrors ``open_clip.VisualTransformer.forward``.
        x = vit.conv1(x)  # (B, W, G, G)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, G*G, W)

        # Prepend CLS + positional embedding.
        cls_token = vit.class_embedding.to(dtype)
        cls_token = cls_token + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # (B, 1+G*G, W)
        x = x + vit.positional_embedding.to(dtype)
        x = vit.ln_pre(x)

        # OpenCLIP uses LND ordering inside the transformer.
        x = x.permute(1, 0, 2)  # (N, B, W)

        outputs: list[torch.Tensor] = []
        wanted = set(self.layer_indices)
        max_idx = max(self.layer_indices)
        for i, block in enumerate(vit.transformer.resblocks, start=1):
            x = block(x)
            if i in wanted:
                # back to NLD, drop CLS token, keep patch tokens only
                patch = x.permute(1, 0, 2)[:, 1:, :].contiguous()
                outputs.append(patch)
            if i >= max_idx:
                break
        return outputs

    def _avg_pool_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Apply an average-pooling of kernel ``pool_kernel`` on the patch grid.

        Expects ``feats`` of shape (B, H*W, C) and returns the same shape.
        """
        if self.pool_kernel is None or self.pool_kernel <= 1:
            return feats
        b, n, c = feats.shape
        g = self.grid_size
        assert n == g * g, f"expected {g * g} tokens, got {n}"
        x = feats.transpose(1, 2).reshape(b, c, g, g)
        pad = self.pool_kernel // 2
        x = F.avg_pool2d(
            x,
            kernel_size=self.pool_kernel,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        return x.reshape(b, c, g * g).transpose(1, 2).contiguous()

    @torch.no_grad()
    def extract_patch_features(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Extract (pooled) multi-layer patch features from frozen CLIP.

        Returned list has length ``L`` with each tensor of shape
        ``(B, H*W, C)``.  The backbone is always frozen; gradients only
        flow through the generators (which consume these features later).
        """
        feats = self._forward_visual_layers(images)
        return [self._avg_pool_features(f) for f in feats]

    # ------------------------------------------------------------------
    # Text features (Sec. 3.4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def build_text_features(self, category: str) -> None:
        """Compute normal/abnormal text prototypes for *category*.

        Templates × state words are encoded by the frozen CLIP text
        encoder and averaged to form two prompt-ensemble vectors
        ``F_text ∈ R^{2×D}`` (L2-normalised).
        """
        device = self.text_features.device
        normal, abnormal = _expand_prompts(category)

        def _encode(phrases: list[str]) -> torch.Tensor:
            tokens = self.tokenizer(phrases).to(device)
            feats = self.clip.encode_text(tokens).float()
            feats = F.normalize(feats, dim=-1)
            return feats.mean(dim=0)

        f_norm = F.normalize(_encode(normal), dim=-1)
        f_abn = F.normalize(_encode(abnormal), dim=-1)
        self.text_features = torch.stack([f_norm, f_abn], dim=0)
        self._text_ready = True

    # ------------------------------------------------------------------
    # Dictionary lookup
    # ------------------------------------------------------------------

    def _lookup(
        self, F_Q: torch.Tensor, F_K: torch.Tensor, F_V: torch.Tensor
    ) -> torch.Tensor:
        """Compute retrieved features ``F_r`` from a query/key/value triplet.

        Parameters
        ----------
        F_Q : Tensor, shape (B, Nq, C)    — Dictionary Query.
        F_K : Tensor, shape (B, Nk, C)    — Dictionary Key.
        F_V : Tensor, shape (B, Nk, C)    — Dictionary Value.
        """
        z = torch.einsum("bnc,bmc->bnm", F_Q, F_K)  # (B, Nq, Nk)
        if self.lookup == "sparse":
            w = sparsemax(z, dim=-1)
        elif self.lookup == "dense":
            w = z.softmax(dim=-1)
        elif self.lookup == "max":
            idx = z.argmax(dim=-1, keepdim=True)
            w = torch.zeros_like(z).scatter_(-1, idx, 1.0)
        else:
            raise ValueError(f"unknown lookup strategy: {self.lookup!r}")
        return torch.einsum("bnm,bmc->bnc", w, F_V)

    # ------------------------------------------------------------------
    # Main forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query_images: torch.Tensor,
        reference_images: torch.Tensor,
    ) -> dict[str, list[torch.Tensor] | torch.Tensor]:
        """Run a complete Dictionary Construction + Dictionary Lookup pass.

        The reference tensor can hold multiple shots per query; internally
        the reference features are flattened into ``kHW`` tokens per
        query (as in Sec. 3.3, Eq. 5 with ``Nk = k * H * W``).

        Parameters
        ----------
        query_images : Tensor, shape (B, 3, H, W)
        reference_images : Tensor, shape (B, k, 3, H, W)

        Returns
        -------
        dict with keys
            ``'F_q'``, ``'F_r'``   — lists of per-layer features (B, HW, C)
            ``'x_q'``, ``'x_r'``   — global embeddings in joint space (B, D)
        """
        if reference_images.dim() == 4:
            reference_images = reference_images.unsqueeze(1)  # (B, 1, 3, H, W)
        b, k = reference_images.shape[:2]
        h, w = reference_images.shape[3:]

        feats_q = self.extract_patch_features(query_images)
        ref_flat = reference_images.reshape(b * k, 3, h, w)
        feats_n = self.extract_patch_features(ref_flat)
        # Interleave k references per sample → (B, k*HW, C)
        feats_n = [
            f.view(b, k, f.shape[1], f.shape[2]).reshape(b, k * f.shape[1], f.shape[2])
            for f in feats_n
        ]

        F_q_list: list[torch.Tensor] = []
        F_r_list: list[torch.Tensor] = []
        for l, (F_q, F_n) in enumerate(zip(feats_q, feats_n)):
            F_Q = self.g_Q(F_q)
            F_K = self.g_K(F_n)
            F_V = F_n + self.g_V(F_n)  # Eq. 3 — external residual on V
            F_r = self._lookup(F_Q, F_K, F_V)
            F_q_list.append(F_q)
            F_r_list.append(F_r)

        # ── Global embeddings for TAC (Sec. 3.4) ────────────────────
        x_q = self._global_embedding(F_q_list)
        x_r = self._global_embedding(F_r_list)

        return {
            "F_q": F_q_list,
            "F_r": F_r_list,
            "x_q": x_q,
            "x_r": x_r,
        }

    def _global_embedding(self, feats: list[torch.Tensor]) -> torch.Tensor:
        """Concatenate multi-layer features → global avg pool → linear proj.

        Implements the reduction used by the Text Alignment Constraint
        (Sec. 3.4): ``x = L2Norm( Linear( GAP( Concat_c F^l ) ) )``.
        """
        concat = torch.cat(feats, dim=-1)  # (B, HW, L*C)
        pooled = concat.mean(dim=1)  # (B, L*C)
        x = self.text_proj(pooled)  # (B, D)
        return F.normalize(x, dim=-1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        query_images: torch.Tensor,
        reference_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce anomaly maps for a batch of queries given k-shot references.

        Parameters
        ----------
        query_images : Tensor, shape (B, 3, H, W)
        reference_images : Tensor, shape (B, k, 3, H, W)
            Normal reference images for each query (k ≥ 1).

        Returns
        -------
        scores : Tensor, shape (B,)
            Image-level anomaly score (max over the anomaly map).
        anomaly_map : Tensor, shape (B, H, W)
            Pixel-level anomaly map upsampled to the input resolution.
        """
        out = self.forward(query_images, reference_images)
        F_q_list, F_r_list = out["F_q"], out["F_r"]

        g = self.grid_size
        H, W = query_images.shape[-2:]

        maps = []
        for F_q, F_r in zip(F_q_list, F_r_list):
            # Cosine distance per patch (Eq. 11)
            d = 1.0 - F.cosine_similarity(F_q, F_r, dim=-1)  # (B, HW)
            m = d.reshape(-1, 1, g, g)
            maps.append(m)

        stacked = torch.stack(maps, dim=0)  # (L, B, 1, G, G)
        anomaly = (
            stacked.mean(dim=0) / 2.0
        )  # Eq. 11 scales by 1/(2L), mean already gives 1/L
        # The paper's Eq. 11 averages ``1 - cos`` over L and multiplies by 1/(2).
        # Keeping the explicit ½ factor to match the published formula.

        anomaly = F.interpolate(
            anomaly, size=(H, W), mode="bilinear", align_corners=False
        )
        anomaly = anomaly[:, 0]  # (B, H, W)

        scores = anomaly.flatten(1).max(dim=1)[0]
        return scores, anomaly
