"""Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection.

CVPR 2025 — Guo et al.
Reference: https://arxiv.org/abs/2405.14325
Official code: https://github.com/guojiajeremy/Dinomaly
"""

from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


# ── Building blocks ──────────────────────────────────────────────────


class Mlp(nn.Module):
    """Standard MLP used inside decoder Transformer blocks."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class bMlp(nn.Module):
    """Bottleneck MLP with dropout applied *before*, *between*, and *after* layers.

    The dropout acts as the "noisy bottleneck" — pseudo feature anomaly injection
    that forces the decoder to restore normal features rather than copy the input.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinearAttention2(nn.Module):
    """Efficient linear attention with ELU+1 feature map (O(N·d²) complexity).

    Unlike softmax attention, this *cannot* focus on specific regions — it
    naturally spreads attention across the full sequence.  This is key to
    preventing the decoder from passing identical information for unseen
    (anomalous) patterns.

    Computation via the kernel trick:
        φ(x) = elu(x) + 1
        kv  = φ(K)^T · V          (d × d)
        z   = 1 / (φ(Q) · Σφ(K))  (normalisation)
        out = kv · φ(Q) · z
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.eps = 1e-6

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        # φ(x) = elu(x) + 1  (non-negative feature map)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # Efficient linear attention via the kernel trick
        # kv = K^T V  -> (B, heads, d, d)
        kv = k.transpose(-2, -1) @ v
        # z = 1 / (Q · sum(K))  -> (B, heads, N)
        z = 1.0 / (
            q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1) + self.eps
        ).squeeze(-1)
        # out = Q · KV * z
        x = (q @ kv) * z.unsqueeze(-1)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Block(nn.Module):
    """Transformer block with configurable attention type.

    Used for the decoder layers. Default attention is :class:`LinearAttention2`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn: type[nn.Module] = LinearAttention2,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y, _attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ── Backbone configs ─────────────────────────────────────────────────

_BACKBONE_CONFIGS: dict[str, dict] = {
    "dinov2reg_vit_small_14": {
        "hub_name": "dinov2_vits14_reg",
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
    },
    "dinov2reg_vit_base_14": {
        "hub_name": "dinov2_vitb14_reg",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
    },
    "dinov2reg_vit_large_14": {
        "hub_name": "dinov2_vitl14_reg",
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers": [4, 6, 8, 10, 12, 14, 16, 18],
    },
}


def load_encoder(backbone: str) -> tuple[nn.Module, dict]:
    """Load a frozen DINOv2-Register encoder and return *(model, config)*."""
    if backbone not in _BACKBONE_CONFIGS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. "
            f"Choose from: {list(_BACKBONE_CONFIGS.keys())}"
        )
    cfg = _BACKBONE_CONFIGS[backbone]
    encoder = torch.hub.load("facebookresearch/dinov2", cfg["hub_name"])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, cfg


# ── Main model ───────────────────────────────────────────────────────


class Dinomaly(nn.Module):
    """Dinomaly reconstruction-based anomaly detection model.

    Parameters
    ----------
    encoder:
        Frozen DINOv2 ViT encoder.
    bottleneck:
        ``nn.ModuleList`` containing the bottleneck MLP(s).
    decoder:
        ``nn.ModuleList`` of Transformer decoder blocks.
    target_layers:
        Indices of encoder blocks whose outputs are collected.
    fuse_layer_encoder:
        Groups of *encoder feature indices* (relative to ``target_layers``)
        for the 2-group loose reconstruction constraint.
    fuse_layer_decoder:
        Corresponding groups for *decoder feature indices*.
    """

    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.ModuleList,
        decoder: nn.ModuleList,
        target_layers: list[int],
        fuse_layer_encoder: list[list[int]],
        fuse_layer_decoder: list[list[int]],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self._target_set = set(target_layers)

        if not hasattr(self.encoder, "num_register_tokens"):
            self.encoder.num_register_tokens = 0

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # ── Encoder (frozen) ──
        target_set = self._target_set
        last_target = self.target_layers[-1]
        with torch.no_grad():
            x = self.encoder.prepare_tokens_with_masks(x)
            en_list: list[torch.Tensor] = []
            for i, blk in enumerate(self.encoder.blocks):
                x = blk(x)
                if i in target_set:
                    en_list.append(x)
                if i >= last_target:
                    break

        side = int(
            math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens)
        )

        # ── Bottleneck (fuse + noisy MLP) ──
        x = self._fuse_features(en_list)
        for blk in self.bottleneck:
            x = blk(x)

        # ── Decoder ──
        de_list: list[torch.Tensor] = []
        for blk in self.decoder:
            x = blk(x)
            de_list.append(x)
        # Reverse so that de_list[0] corresponds to the deepest decoder layer
        de_list = de_list[::-1]

        # ── Group-wise feature fusion (loose reconstruction) ──
        en = [
            self._fuse_features([en_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_encoder
        ]
        de = [
            self._fuse_features([de_list[idx] for idx in idxs])
            for idxs in self.fuse_layer_decoder
        ]

        # Remove class + register tokens and reshape to (B, C, H, W)
        n_skip = 1 + self.encoder.num_register_tokens
        en = [e[:, n_skip:, :] for e in en]
        de = [d[:, n_skip:, :] for d in de]

        en = [
            e.permute(0, 2, 1).reshape(e.shape[0], -1, side, side).contiguous()
            for e in en
        ]
        de = [
            d.permute(0, 2, 1).reshape(d.shape[0], -1, side, side).contiguous()
            for d in de
        ]
        return en, de

    @staticmethod
    def _fuse_features(feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Average-pool a list of feature tensors (loose reconstruction)."""
        out = feat_list[0]
        for i in range(1, len(feat_list)):
            out = out + feat_list[i]
        return out * (1.0 / len(feat_list))


def build_dinomaly(
    backbone: str = "dinov2reg_vit_base_14",
    dropout: float = 0.2,
    num_decoder_layers: int = 8,
) -> Dinomaly:
    """Construct a :class:`Dinomaly` model from a named backbone.

    Parameters
    ----------
    backbone:
        One of ``dinov2reg_vit_small_14``, ``dinov2reg_vit_base_14``,
        ``dinov2reg_vit_large_14``.
    dropout:
        Dropout rate for the noisy bottleneck MLP.
    num_decoder_layers:
        Number of Transformer decoder blocks.
    """
    encoder, cfg = load_encoder(backbone)
    embed_dim = cfg["embed_dim"]
    num_heads = cfg["num_heads"]
    target_layers = cfg["target_layers"]

    # ── Bottleneck ──
    bottleneck = nn.ModuleList(
        [bMlp(embed_dim, embed_dim * 4, embed_dim, drop=dropout)]
    )

    # ── Decoder ──
    decoder = nn.ModuleList()
    for _ in range(num_decoder_layers):
        blk = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn_drop=0.0,
            attn=LinearAttention2,
        )
        decoder.append(blk)

    # 2-group loose reconstruction (default)
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    model = Dinomaly(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )

    # ── Weight initialisation (decoder + bottleneck only) ──
    trainable = nn.ModuleList([bottleneck, decoder])
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    return model
