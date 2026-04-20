"""Industrial Anomaly Detection — Gradio demo.

A modern, factory-floor-inspired UI for running few-shot / training-free
industrial anomaly detection methods on user-supplied images.

Launch:
    uv run python demo.py
    # or
    python demo.py --share
"""

from __future__ import annotations

import argparse
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")

# ── Model imports (lazy in UI, but modules are cheap to import) ─────
from lib.lightning import (
    AnomalyDINOModule,
    AnomalyTIPSv2Module,
    DictASModule,
    PatchCoreModule,
    WinCLIPModule,
)

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
#  Method registry
# ─────────────────────────────────────────────────────────────────────


@dataclass
class MethodSpec:
    key: str
    label: str
    tagline: str
    needs_nominal: bool  # nominal images mandatory?
    supports_nominal: bool  # model accepts nominal images?
    supports_zero_shot: bool
    image_size: int
    notes: str


METHODS: dict[str, MethodSpec] = {
    "patchcore": MethodSpec(
        key="patchcore",
        label="PatchCore",
        tagline="WideResNet-50-2 • memory-bank nearest-neighbour",
        needs_nominal=True,
        supports_nominal=True,
        supports_zero_shot=False,
        image_size=256,
        notes="Builds a coreset memory bank from ImageNet features. "
        "Few-shot friendly; fastest at inference.",
    ),
    "anomalydino": MethodSpec(
        key="anomalydino",
        label="AnomalyDINO",
        tagline="DINOv2 features • patch-wise cosine NN",
        needs_nominal=True,
        supports_nominal=True,
        supports_zero_shot=False,
        image_size=448,
        notes="Training-free, PCA foreground masking, rotation augmentation of references.",
    ),
    "anomalytipsv2": MethodSpec(
        key="anomalytipsv2",
        label="AnomalyTIPSv2",
        tagline="Google TIPSv2 ViT • patch-wise cosine NN",
        needs_nominal=True,
        supports_nominal=True,
        supports_zero_shot=False,
        image_size=448,
        notes="AnomalyDINO pipeline with a TIPSv2 backbone (stronger features).",
    ),
    "winclip": MethodSpec(
        key="winclip",
        label="WinCLIP / WinCLIP+",
        tagline="CLIP ViT-B/16+ • zero- or few-shot",
        needs_nominal=False,
        supports_nominal=True,
        supports_zero_shot=True,
        image_size=240,
        notes="Zero-shot by default using text prompts; add nominal shots for WinCLIP+.",
    ),
    "dictas": MethodSpec(
        key="dictas",
        label="DictAS",
        tagline="CLIP ViT-L/14-336 • dictionary lookup • sparsemax",
        needs_nominal=True,
        supports_nominal=True,
        supports_zero_shot=False,
        image_size=336,
        notes="Self-supervised class-generalizable FSAS (arXiv:2508.13560). "
        "Loads trained Key/Query/Value generators from checkpoints/dictas/.",
    ),
}


# ─────────────────────────────────────────────────────────────────────
#  Device resolution
# ─────────────────────────────────────────────────────────────────────


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu", 0)
    return torch.device("cpu")


DEVICE = resolve_device()


# ─────────────────────────────────────────────────────────────────────
#  Preprocessing
# ─────────────────────────────────────────────────────────────────────


def _to_rgb_np(img: Any) -> np.ndarray:
    """Coerce file path / PIL / ndarray to uint8 RGB HWC."""
    if img is None:
        return None
    if isinstance(img, (str, Path)):
        arr = np.asarray(Image.open(img).convert("RGB"))
    elif isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB"))
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            arr = np.stack([img] * 3, axis=-1)
        elif img.shape[-1] == 4:
            arr = img[..., :3]
        else:
            arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")
    return arr


def _resize_rgb(img_np: np.ndarray, size: int) -> np.ndarray:
    pil = Image.fromarray(img_np).resize((size, size), Image.BICUBIC)
    return np.asarray(pil)


def _to_imagenet_tensor(img_np: np.ndarray, size: int) -> torch.Tensor:
    """(H,W,3) uint8 → (1,3,size,size) ImageNet-normalised float tensor."""
    arr = _resize_rgb(img_np, size).astype(np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t


# ─────────────────────────────────────────────────────────────────────
#  Module builders
# ─────────────────────────────────────────────────────────────────────


def build_patchcore(params: dict) -> PatchCoreModule:
    return PatchCoreModule(
        coreset_sampling_ratio=float(params.get("coreset_sampling_ratio", 0.1)),
        num_neighbors=int(params.get("num_neighbors", 9)),
        image_size=METHODS["patchcore"].image_size,
    )


def build_anomalydino(params: dict) -> AnomalyDINOModule:
    return AnomalyDINOModule(
        model_name=params.get("dino_model", "dinov2_vits14"),
        smaller_edge_size=METHODS["anomalydino"].image_size,
        masking=bool(params.get("masking", True)),
        rotation=bool(params.get("rotation", True)),
        top_percent=float(params.get("top_percent", 0.01)),
        gaussian_sigma=float(params.get("gaussian_sigma", 4.0)),
        image_size=METHODS["anomalydino"].image_size,
    )


def build_anomalytipsv2(params: dict) -> AnomalyTIPSv2Module:
    return AnomalyTIPSv2Module(
        model_name=params.get("tips_model", "google/tipsv2-b14"),
        smaller_edge_size=METHODS["anomalytipsv2"].image_size,
        masking=bool(params.get("masking", True)),
        rotation=bool(params.get("rotation", True)),
        top_percent=float(params.get("top_percent", 0.01)),
        gaussian_sigma=float(params.get("gaussian_sigma", 4.0)),
        image_size=METHODS["anomalytipsv2"].image_size,
    )


def build_winclip(params: dict) -> WinCLIPModule:
    return WinCLIPModule(
        category=params.get("category", "object"),
        backbone=params.get("clip_backbone", "ViT-B-16-plus-240"),
        pretrained=params.get("clip_pretrained", "laion400m_e32"),
        scales=(2, 3),
        image_size=METHODS["winclip"].image_size,
        k_shot=int(params.get("k_shot", 0)),
    )


def build_dictas(params: dict) -> DictASModule:
    """Build a DictAS module, loading trained generators when available.

    If a previous checkpoint exists for any category, its learned Key /
    Query / Value generators are restored — DictAS is class-generalizable
    (self-supervised) so this is safe across categories.
    """
    from lib.utils.checkpoint import latest_checkpoint_dir

    category = params.get("category") or "object"
    ckpt_dir = latest_checkpoint_dir("checkpoints", "dictas", category)
    if ckpt_dir is None:
        # Fall back to any available DictAS checkpoint (class-generalizable).
        root = Path("checkpoints") / "dictas"
        if root.exists():
            candidates = [p for p in root.rglob("model.ckpt")]
            if candidates:
                ckpt_dir = sorted(candidates)[-1].parent

    if ckpt_dir is not None and (ckpt_dir / "model.ckpt").exists():
        module = DictASModule.load_checkpoint(ckpt_dir, map_location="cpu")
        # Rebuild text features for the requested category.
        module.category = category
        module.model.build_text_features(category)
        return module

    # No checkpoint available — use freshly initialised generators.
    return DictASModule(
        category=category,
        backbone=params.get("dictas_backbone", "ViT-L-14-336"),
        pretrained=params.get("dictas_pretrained", "openai"),
        image_size=METHODS["dictas"].image_size,
        lookup=params.get("dictas_lookup", "sparse"),
        k_shot=max(1, int(params.get("k_shot", 1))),
    )


# ─────────────────────────────────────────────────────────────────────
#  Fit-on-the-fly (few-shot)
# ─────────────────────────────────────────────────────────────────────


def _fit_patchcore(module: PatchCoreModule, nominals: list[np.ndarray]) -> None:
    size = METHODS["patchcore"].image_size
    tensors = torch.cat([_to_imagenet_tensor(img, size) for img in nominals], dim=0).to(
        DEVICE
    )
    module.to(DEVICE).eval()

    feats, _ = module.model.extract_features(tensors)
    feats = feats.cpu()
    n_select = max(1, int(feats.shape[0] * module.model.coreset_sampling_ratio))

    from lib.models.patchcore import PatchCore

    idx = PatchCore._greedy_coreset_sampling(
        feats, n_select, module.model.projection_dim
    )
    module.model.memory_bank = feats[idx].to(DEVICE)
    module.model._fitted = True


def _fit_anomalydino(module: AnomalyDINOModule, nominals: list[np.ndarray]) -> None:
    module.to(DEVICE).eval()
    module.model.fit(nominals)  # nominals: list[np.ndarray uint8 HWC]


def _fit_anomalytipsv2(module: AnomalyTIPSv2Module, nominals: list[np.ndarray]) -> None:
    module.to(DEVICE).eval()
    from lib.models.anomalytipsv2 import AnomalyTIPSv2

    all_features: list[np.ndarray] = []
    for img in nominals:
        variants = (
            module.model.augment_reference(img) if module.model.rotation else [img]
        )
        for variant in variants:
            img_tensor, _ = module.model.prepare_image(variant)
            feats = module.model.extract_features(img_tensor)
            all_features.append(feats)
    bank = np.concatenate(all_features, axis=0).astype(np.float32)
    bank = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
    module.model._memory_bank = bank
    module.model._fitted = True


def _fit_winclip(
    module: WinCLIPModule, nominals: list[np.ndarray], category: str
) -> None:
    module.to(DEVICE).eval()
    module.model.build_text_features(category)
    if nominals:
        size = METHODS["winclip"].image_size
        imgs = torch.cat(
            [_to_imagenet_tensor(img, size) for img in nominals], dim=0
        ).to(DEVICE)
        module.model.build_visual_gallery(imgs)


def _fit_dictas(
    module: DictASModule, nominals: list[np.ndarray], category: str
) -> None:
    """Cache nominal reference images in the module for later prediction."""
    module.to(DEVICE).eval()
    if not module.model._text_ready:
        module.model.build_text_features(category)
    size = METHODS["dictas"].image_size
    imgs = torch.cat([_to_imagenet_tensor(img, size) for img in nominals], dim=0).to(
        DEVICE
    )
    module._reference_images = imgs


# ─────────────────────────────────────────────────────────────────────
#  Predict helpers
# ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _predict_module(
    method_key: str,
    module: Any,
    test_rgb: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return (score, anomaly_map HxW float)."""
    spec = METHODS[method_key]
    size = spec.image_size
    x = _to_imagenet_tensor(test_rgb, size).to(DEVICE)

    if method_key == "patchcore":
        scores, maps = module.model.predict(x)  # (1,) and (1,H,W)
        return float(scores[0].cpu()), maps[0].cpu().numpy()

    if method_key == "anomalydino":
        scores, maps = module.model.predict_batch_tensor(x, output_size=(size, size))
        return float(scores[0].cpu()), maps[0, 0].cpu().numpy()

    if method_key == "anomalytipsv2":
        x_unnorm = (
            x * torch.tensor(_STD, device=x.device).view(1, 3, 1, 1)
            + torch.tensor(_MEAN, device=x.device).view(1, 3, 1, 1)
        ).clamp(0, 1)
        scores, maps = module.model.predict_batch_tensor(
            x_unnorm, output_size=(size, size)
        )
        return float(scores[0].cpu()), maps[0, 0].cpu().numpy()

    if method_key == "winclip":
        scores, maps = module.model.predict(x)
        return float(scores[0].cpu()), maps[0].cpu().numpy()

    if method_key == "dictas":
        refs = module._reference_images
        if refs is None:
            raise ValueError("DictAS requires nominal reference images.")
        refs = refs.to(x.device).unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)
        scores, maps = module.model.predict(x, refs)
        return float(scores[0].cpu()), maps[0].cpu().numpy()

    raise ValueError(f"Unknown method: {method_key}")


# ─────────────────────────────────────────────────────────────────────
#  Visualisation
# ─────────────────────────────────────────────────────────────────────


def _normalise_map(amap: np.ndarray) -> np.ndarray:
    amap = amap.astype(np.float32)
    lo, hi = float(amap.min()), float(amap.max())
    if hi - lo < 1e-8:
        return np.zeros_like(amap)
    return (amap - lo) / (hi - lo)


def _overlay_heatmap(
    rgb: np.ndarray, amap: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Blend a colormap-rendered anomaly map over the input image."""
    h, w = rgb.shape[:2]
    amap_r = (
        np.asarray(
            Image.fromarray((amap * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR
            )
        ).astype(np.float32)
        / 255.0
    )
    cmap = plt.get_cmap("inferno")
    colored = cmap(amap_r)[..., :3]  # (H,W,3) 0..1
    base = rgb.astype(np.float32) / 255.0
    blended = (1 - alpha * amap_r[..., None]) * base + alpha * amap_r[
        ..., None
    ] * colored
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended


def _render_heatmap(amap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return a pure colormap visualization of the anomaly map at given size."""
    w, h = size
    amap_r = (
        np.asarray(
            Image.fromarray((amap * 255).astype(np.uint8)).resize(
                (w, h), Image.BILINEAR
            )
        ).astype(np.float32)
        / 255.0
    )
    cmap = plt.get_cmap("inferno")
    colored = (cmap(amap_r)[..., :3] * 255).astype(np.uint8)
    return colored


# ─────────────────────────────────────────────────────────────────────
#  Decision function (score → NOMINAL / ANOMALY + confidence)
# ─────────────────────────────────────────────────────────────────────


def _map_statistics(amap_norm: np.ndarray) -> dict[str, float]:
    """Extract robust stats from a 0..1 normalised anomaly map."""
    flat = amap_norm.ravel()
    return {
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(flat.max()),
        "mean_top1": float(np.mean(np.sort(flat)[-max(1, len(flat) // 100) :])),
    }


def _decide(
    score: float, map_stats: dict[str, float], threshold: float
) -> tuple[str, float]:
    """Combine image-level score with map stats → (verdict, confidence ∈ [0,1]).

    ``threshold`` is the user-provided decision threshold on the map's
    normalised top-1 % mean. Confidence is a soft sigmoid distance from it.
    """
    # We primarily use the anomaly map's normalised top-percentile, because
    # raw scores have very different scales across methods.
    signal = map_stats["mean_top1"]
    verdict = "ANOMALY" if signal >= threshold else "NOMINAL"
    # Sigmoid sharpness: larger = sharper boundary.
    k = 10.0
    conf = 1.0 / (1.0 + np.exp(-k * (signal - threshold)))
    if verdict == "NOMINAL":
        conf = 1.0 - conf
    return verdict, float(np.clip(conf, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────
#  Main inference callback
# ─────────────────────────────────────────────────────────────────────


def run_inference(
    test_image,
    nominal_gallery,
    method_key: str,
    threshold: float,
    # PatchCore
    coreset_ratio: float,
    num_neighbors: int,
    # AnomalyDINO
    dino_model: str,
    adino_masking: bool,
    adino_rotation: bool,
    adino_top_percent: float,
    # AnomalyTIPSv2
    tips_model: str,
    # WinCLIP
    category: str,
    k_shot: int,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if test_image is None:
        raise gr.Error("Please upload a test image.")
    spec = METHODS[method_key]

    # Gather nominal images
    nominal_imgs: list[np.ndarray] = []
    if nominal_gallery:
        for item in nominal_gallery:
            if isinstance(item, (list, tuple)):
                item = item[0]
            nominal_imgs.append(_to_rgb_np(item))

    if spec.needs_nominal and not nominal_imgs:
        raise gr.Error(
            f"{spec.label} requires at least one nominal (defect-free) reference image."
        )

    progress(0.05, desc="Building model")
    t0 = time.perf_counter()

    if method_key == "patchcore":
        module = build_patchcore(
            {"coreset_sampling_ratio": coreset_ratio, "num_neighbors": num_neighbors}
        )
    elif method_key == "anomalydino":
        module = build_anomalydino(
            {
                "dino_model": dino_model,
                "masking": adino_masking,
                "rotation": adino_rotation,
                "top_percent": adino_top_percent,
            }
        )
    elif method_key == "anomalytipsv2":
        module = build_anomalytipsv2(
            {
                "tips_model": tips_model,
                "masking": adino_masking,
                "rotation": adino_rotation,
                "top_percent": adino_top_percent,
            }
        )
    elif method_key == "winclip":
        module = build_winclip(
            {
                "category": category or "object",
                "k_shot": k_shot if nominal_imgs else 0,
            }
        )
    elif method_key == "dictas":
        module = build_dictas(
            {
                "category": category or "object",
                "k_shot": max(1, int(k_shot)) if nominal_imgs else 1,
            }
        )
    else:
        raise gr.Error(f"Unknown method: {method_key}")

    t_build = time.perf_counter() - t0
    progress(0.3, desc="Fitting reference features")

    t1 = time.perf_counter()
    if method_key == "patchcore":
        _fit_patchcore(module, nominal_imgs)
    elif method_key == "anomalydino":
        _fit_anomalydino(module, nominal_imgs)
    elif method_key == "anomalytipsv2":
        _fit_anomalytipsv2(module, nominal_imgs)
    elif method_key == "winclip":
        _fit_winclip(module, nominal_imgs, category or "object")
    elif method_key == "dictas":
        _fit_dictas(module, nominal_imgs, category or "object")
    t_fit = time.perf_counter() - t1

    progress(0.75, desc="Running inference")
    t2 = time.perf_counter()
    test_rgb = _to_rgb_np(test_image)
    score, amap = _predict_module(method_key, module, test_rgb)
    t_pred = time.perf_counter() - t2

    amap_norm = _normalise_map(amap)
    stats = _map_statistics(amap_norm)
    verdict, confidence = _decide(score, stats, threshold)

    H, W = test_rgb.shape[:2]
    overlay = _overlay_heatmap(test_rgb, amap_norm, alpha=0.55)
    heatmap_only = _render_heatmap(amap_norm, (W, H))

    progress(1.0, desc="Done")

    # ── Status HTML badge ─────────────────────────────────────────
    if verdict == "ANOMALY":
        badge_html = f"""
        <div class="iad-verdict iad-verdict-anomaly">
            <span class="iad-badge-dot"></span>
            <span class="iad-verdict-label">ANOMALY DETECTED</span>
            <span class="iad-verdict-conf">{confidence * 100:.1f}% conf.</span>
        </div>
        """
    else:
        badge_html = f"""
        <div class="iad-verdict iad-verdict-ok">
            <span class="iad-badge-dot"></span>
            <span class="iad-verdict-label">PART NOMINAL</span>
            <span class="iad-verdict-conf">{confidence * 100:.1f}% conf.</span>
        </div>
        """

    # ── Metrics readout ───────────────────────────────────────────
    metrics_html = f"""
    <div class="iad-metrics">
      <div class="iad-metric">
        <div class="iad-metric-label">RAW SCORE</div>
        <div class="iad-metric-value">{score:.4f}</div>
      </div>
      <div class="iad-metric">
        <div class="iad-metric-label">MAP MEAN TOP-1%</div>
        <div class="iad-metric-value">{stats['mean_top1']:.3f}</div>
      </div>
      <div class="iad-metric">
        <div class="iad-metric-label">MAP P99</div>
        <div class="iad-metric-value">{stats['p99']:.3f}</div>
      </div>
      <div class="iad-metric">
        <div class="iad-metric-label">THRESHOLD</div>
        <div class="iad-metric-value">{threshold:.2f}</div>
      </div>
      <div class="iad-metric">
        <div class="iad-metric-label">DEVICE</div>
        <div class="iad-metric-value">{str(DEVICE).upper()}</div>
      </div>
      <div class="iad-metric">
        <div class="iad-metric-label">BUILD / FIT / INFER (s)</div>
        <div class="iad-metric-value">{t_build:.2f} · {t_fit:.2f} · {t_pred:.2f}</div>
      </div>
    </div>
    """

    log_text = (
        f"[{spec.label}] refs={len(nominal_imgs)}  "
        f"score={score:.4f}  top1%={stats['mean_top1']:.3f}  "
        f"verdict={verdict} ({confidence * 100:.1f}%)  "
        f"t_build={t_build:.2f}s  t_fit={t_fit:.2f}s  t_pred={t_pred:.2f}s"
    )

    return badge_html, metrics_html, overlay, heatmap_only, log_text


# ─────────────────────────────────────────────────────────────────────
#  UI: theme & custom CSS
# ─────────────────────────────────────────────────────────────────────

INDUSTRIAL_CSS = """
/* ---------- root tokens — DARK (default) ---------- */
:root, .dark {
  --iad-bg-0: #0a0c10;
  --iad-bg-1: #10141b;
  --iad-bg-2: #151b24;
  --iad-bg-3: #1d2430;
  --iad-line: #2a323f;
  --iad-line-bright: #3a4452;
  --iad-text: #e6eaf0;
  --iad-text-dim: #8c98a8;
  --iad-accent: #ffb020;       /* safety amber */
  --iad-accent-bright: #ffd166;
  --iad-accent-on: #1a1100;
  --iad-ok: #22c55e;
  --iad-ok-dim: #0f3d22;
  --iad-danger: #ef4444;
  --iad-danger-dim: #3b1414;
  --iad-info: #4aa3ff;
  --iad-grid-opacity: 0.08;
  --iad-stripe-dark: #000;
  --iad-log-bg: #05070a;
  --iad-log-fg: var(--iad-accent-bright);
  --iad-shadow: 0 6px 18px rgba(255, 176, 32, 0.15);
  --iad-shadow-hover: 0 10px 24px rgba(255, 176, 32, 0.28);
}

/* ---------- root tokens — LIGHT (engineering blueprint) ---------- */
html.iad-light, html.iad-light :root {
  --iad-bg-0: #f4f2ec;       /* paper */
  --iad-bg-1: #ffffff;
  --iad-bg-2: #f7f5ef;
  --iad-bg-3: #ecebe3;
  --iad-line: #c8c5ba;
  --iad-line-bright: #8a8676;
  --iad-text: #1a1f26;
  --iad-text-dim: #5a6472;
  --iad-accent: #c25200;     /* deep safety orange */
  --iad-accent-bright: #ea7317;
  --iad-accent-on: #ffffff;
  --iad-ok: #1f7a3a;
  --iad-ok-dim: #d5ead7;
  --iad-danger: #b91c1c;
  --iad-danger-dim: #f7dcdc;
  --iad-info: #1e6ab8;
  --iad-grid-opacity: 0.22;
  --iad-stripe-dark: #222222;
  --iad-log-bg: #1a1f26;
  --iad-log-fg: #ffd166;
  --iad-shadow: 0 6px 14px rgba(194, 82, 0, 0.18);
  --iad-shadow-hover: 0 10px 22px rgba(194, 82, 0, 0.28);
}


/* global background with subtle grid blueprint */
gradio-app, .gradio-container, body {
  background:
    linear-gradient(180deg, var(--iad-bg-0) 0%, var(--iad-bg-1) 100%) !important;
  color: var(--iad-text) !important;
  font-family: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, SFMono-Regular,
    Menlo, monospace !important;
}
.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(var(--iad-line) 1px, transparent 1px),
    linear-gradient(90deg, var(--iad-line) 1px, transparent 1px);
  background-size: 48px 48px;
  opacity: var(--iad-grid-opacity);
  z-index: 0;
}
.gradio-container > * { position: relative; z-index: 1; }

/* ---------- header ---------- */
.iad-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border: 1px solid var(--iad-line);
  background:
    repeating-linear-gradient(
      135deg,
      var(--iad-bg-2) 0 14px,
      var(--iad-bg-3) 14px 28px
    );
  border-radius: 8px;
  margin-bottom: 14px;
}
.iad-header-left {
  display: flex; align-items: center; gap: 14px;
}
.iad-logo {
  width: 44px; height: 44px;
  display: grid; place-items: center;
  border: 2px solid var(--iad-accent);
  color: var(--iad-accent);
  font-weight: 700;
  letter-spacing: 0.12em;
  background: var(--iad-bg-0);
  border-radius: 4px;
  box-shadow: inset 0 0 0 2px var(--iad-bg-1);
}
.iad-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: var(--iad-text);
  margin: 0;
}
.iad-subtitle {
  font-size: 11px;
  color: var(--iad-text-dim);
  letter-spacing: 0.22em;
  margin: 2px 0 0;
}
.iad-stripes {
  height: 18px; width: 160px;
  background: repeating-linear-gradient(
    135deg,
    var(--iad-accent) 0 10px,
    var(--iad-stripe-dark) 10px 20px
  );
  border-radius: 2px;
  opacity: 0.9;
}

/* ---------- theme toggle ---------- */
.iad-header-right {
  display: flex; align-items: center; gap: 14px;
}
.iad-theme-toggle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 7px 14px;
  font-family: inherit;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  background: #1a1f28;
  color: #ffb020;
  border: 1.5px solid #3a4452;
  border-radius: 5px;
  cursor: pointer;
  transition: border-color 0.15s ease, transform 0.08s ease,
             background 0.2s ease, color 0.2s ease;
}
.iad-theme-toggle:hover {
  border-color: #ffb020;
  transform: translateY(-1px);
  background: #222a36;
}
.iad-theme-toggle svg {
  width: 16px; height: 16px;
  flex-shrink: 0;
}
.iad-theme-toggle .iad-icon-sun { display: none; }
.iad-theme-toggle .iad-icon-moon { display: block; }
html.iad-light .iad-theme-toggle {
  background: #ffffff !important;
  color: #c25200 !important;
  border-color: #b0a898 !important;
}
html.iad-light .iad-theme-toggle:hover {
  border-color: #c25200 !important;
  background: #f7f5ef !important;
}
html.iad-light .iad-theme-toggle svg { color: #c25200; }
html.iad-light .iad-theme-toggle .iad-icon-sun { display: block; }
html.iad-light .iad-theme-toggle .iad-icon-moon { display: none; }

/* ---------- panels / cards ---------- */
.iad-panel {
  background: var(--iad-bg-2) !important;
  border: 1px solid var(--iad-line) !important;
  border-radius: 6px !important;
  padding: 12px 14px;
}

/*
 * Instead of applying borders to every .block/.form/.panel/.wrap,
 * we control Gradio's own CSS variables at the root so that Gradio
 * handles nesting correctly (no double borders, matching radii).
 */
:root, .dark {
  --block-border-width: 1px;
  --block-border-color: var(--iad-line);
  --block-radius: 6px;
  --block-background-fill: var(--iad-bg-2);
  --border-color-primary: var(--iad-line);
  --input-border-width: 1px;
  --input-border-color: var(--iad-line);
  --input-radius: 4px;
  --input-background-fill: var(--iad-bg-0);
  --panel-border-width: 0px;
  --layout-gap: 8px;
  --form-gap-width: 0px;
  --block-padding: 10px;
  --checkbox-border-width: 1px;
  --checkbox-border-color: var(--iad-line);
  --section-header-text-weight: 700;
}

.iad-section-title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.2em;
  color: var(--iad-accent);
  padding: 0 0 8px;
  border-bottom: 1px solid var(--iad-line);
  margin-bottom: 12px;
  display: flex; align-items: center; gap: 8px;
}
.iad-section-title::before {
  content: "";
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--iad-accent);
  box-shadow: 0 0 10px var(--iad-accent);
}

/* ---------- inputs ---------- */
label {
  color: var(--iad-text-dim) !important;
  text-transform: uppercase;
  font-size: 11px !important;
  letter-spacing: 0.14em;
  font-weight: 600 !important;
}

/* sliders */
input[type="range"] { accent-color: var(--iad-accent); }

/* ---------- buttons ---------- */
button.primary, .gr-button-primary, button[variant="primary"] {
  background: linear-gradient(
    180deg,
    var(--iad-accent-bright) 0%,
    var(--iad-accent) 100%
  ) !important;
  color: var(--iad-accent-on) !important;
  border: 1px solid var(--iad-accent) !important;
  border-radius: 4px !important;
  font-weight: 800 !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.35),
    var(--iad-shadow);
  transition: transform 0.08s ease, box-shadow 0.12s ease;
}
button.primary:hover, .gr-button-primary:hover, button[variant="primary"]:hover {
  transform: translateY(-1px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.35),
    var(--iad-shadow-hover);
}
button.secondary, .gr-button-secondary {
  background: var(--iad-bg-3) !important;
  color: var(--iad-text) !important;
  border: 1px solid var(--iad-line) !important;
  font-weight: 600 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 4px !important;
}
button.secondary:hover { border-color: var(--iad-accent) !important; }

/* ---------- gallery ---------- */
.iad-gallery > div {
  overflow: visible !important;
  max-height: none !important;
}

/* ---------- sliders — let Gradio handle borders via variables ---------- */
.iad-slider button {
  background: var(--iad-bg-1) !important;
  color: var(--iad-text) !important;
}

/* ---------- accordion ---------- */
.iad-accordion > button,
.iad-accordion summary {
  color: var(--iad-accent-bright) !important;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  font-size: 11px !important;
  font-weight: 700 !important;
}

/* ---------- groups inside accordion — no border (inner element) ---------- */
.iad-group {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 4px 0 !important;
}

/* ---------- verdict badge ---------- */
.iad-verdict {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 18px 22px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  border: 2px solid;
  position: relative;
  overflow: hidden;
}
.iad-verdict::before {
  content: "";
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    135deg,
    rgba(127, 127, 127, 0.08) 0 8px,
    transparent 8px 16px
  );
  pointer-events: none;
}
.iad-verdict-label {
  font-size: 20px;
  font-weight: 800;
  letter-spacing: 0.2em;
  flex: 1;
}
.iad-verdict-conf {
  font-size: 12px;
  letter-spacing: 0.18em;
  opacity: 0.8;
  padding: 4px 8px;
  border: 1px solid currentColor;
  border-radius: 3px;
}
.iad-badge-dot {
  width: 14px; height: 14px;
  border-radius: 50%;
  box-shadow: 0 0 0 3px rgba(0, 0, 0, 0.4), 0 0 18px currentColor;
  animation: iad-pulse 1.6s ease-in-out infinite;
}
@keyframes iad-pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.2); opacity: 0.7; }
}
.iad-verdict-ok {
  color: var(--iad-ok);
  border-color: var(--iad-ok);
  background: linear-gradient(90deg,
    var(--iad-ok-dim) 0%,
    color-mix(in srgb, var(--iad-ok) 6%, transparent) 100%);
}
.iad-verdict-ok .iad-badge-dot { background: var(--iad-ok); }
.iad-verdict-anomaly {
  color: var(--iad-danger);
  border-color: var(--iad-danger);
  background: linear-gradient(90deg,
    var(--iad-danger-dim) 0%,
    color-mix(in srgb, var(--iad-danger) 6%, transparent) 100%);
}
.iad-verdict-anomaly .iad-badge-dot { background: var(--iad-danger); }
.iad-verdict-standby {
  color: var(--iad-text-dim);
  border-color: var(--iad-line);
  background: var(--iad-bg-1);
}
.iad-verdict-standby .iad-badge-dot { background: var(--iad-text-dim); }

/* ---------- metrics grid ---------- */
.iad-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px;
  margin-top: 12px;
}
.iad-metric {
  border: 1px solid var(--iad-line);
  background: var(--iad-bg-1);
  padding: 10px 12px;
  border-radius: 4px;
  border-left: 3px solid var(--iad-accent);
}
.iad-metric-label {
  font-size: 10px;
  color: var(--iad-text-dim);
  letter-spacing: 0.22em;
  font-weight: 700;
}
.iad-metric-value {
  font-size: 18px;
  font-weight: 700;
  color: var(--iad-text);
  margin-top: 4px;
}

/* ---------- tagline under method selector ---------- */
.iad-method-tagline {
  font-size: 11px;
  color: var(--iad-text-dim);
  letter-spacing: 0.12em;
  padding: 4px 10px;
  border-left: 2px solid var(--iad-accent);
  margin: 6px 0 0;
  background: var(--iad-bg-1);
}
.iad-method-notes {
  font-size: 11px;
  color: var(--iad-text-dim);
  padding: 6px 10px;
  border: 1px dashed var(--iad-line-bright);
  border-radius: 4px;
  margin-top: 6px;
}

/* ---------- log console ---------- */
.iad-log textarea {
  background: var(--iad-log-bg) !important;
  color: var(--iad-log-fg) !important;
  font-family: 'JetBrains Mono', ui-monospace, monospace !important;
  font-size: 12px !important;
}

/* ---------- footer ---------- */
.iad-footer {
  margin-top: 10px;
  padding: 8px 12px;
  border-top: 1px solid var(--iad-line);
  color: var(--iad-text-dim);
  font-size: 10px;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  display: flex;
  justify-content: space-between;
}

/* hide gradio footer */
footer { display: none !important; }

/* ========== LIGHT MODE ========== */
/*
 * Light mode works by redefining --iad-* tokens (see top of file).
 * Gradio's own CSS vars reference those tokens, so blocks/inputs/etc.
 * automatically pick up the right colours — no need to re-target
 * every .block/.form/.panel individually.
 */
html.iad-light gradio-app,
html.iad-light .gradio-container,
html.iad-light body {
  background: linear-gradient(180deg, var(--iad-bg-0) 0%, var(--iad-bg-1) 100%) !important;
  color: var(--iad-text) !important;
}
html.iad-light label {
  color: var(--iad-text-dim) !important;
}
html.iad-light .iad-log textarea {
  background: var(--iad-log-bg) !important;
  color: var(--iad-log-fg) !important;
}
/* Gradio internal variables — force light palette */
html.iad-light {
  --body-text-color: var(--iad-text);
  --body-text-color-subdued: var(--iad-text-dim);
  --background-fill-primary: var(--iad-bg-1);
  --background-fill-secondary: var(--iad-bg-2);
  --block-background-fill: var(--iad-bg-2);
  --block-border-color: var(--iad-line);
  --block-label-background-fill: var(--iad-bg-1);
  --block-label-text-color: var(--iad-text-dim);
  --block-title-text-color: var(--iad-accent);
  --border-color-primary: var(--iad-line);
  --input-background-fill: var(--iad-bg-0);
  --input-border-color: var(--iad-line);
  --panel-background-fill: var(--iad-bg-2);
  --body-background-fill: var(--iad-bg-0);
  --checkbox-border-color: var(--iad-line);
  --neutral-50: var(--iad-bg-0);
  --neutral-100: var(--iad-bg-1);
  --neutral-200: var(--iad-bg-2);
  --neutral-300: var(--iad-line);
  --neutral-400: var(--iad-line-bright);
  --neutral-500: var(--iad-text-dim);
  --neutral-600: var(--iad-text-dim);
  --neutral-700: var(--iad-text);
  --neutral-800: var(--iad-text);
  --neutral-900: var(--iad-text);
  --neutral-950: var(--iad-text);
  --color-accent: var(--iad-accent);
  --color-accent-soft: var(--iad-bg-3);
  --slider-color: var(--iad-accent);
}
/* Ensure all text inside Gradio blocks uses theme-aware color */
html.iad-light span,
html.iad-light p,
html.iad-light div,
html.iad-light h1,
html.iad-light h2,
html.iad-light h3 {
  color: inherit;
}
/* Theme toggle — light overrides are in the main .iad-theme-toggle block */
"""


def _industrial_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.amber,
        secondary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
        font_mono=[
            gr.themes.GoogleFont("JetBrains Mono"),
            "ui-monospace",
            "monospace",
        ],
    ).set(
        body_background_fill="#0a0c10",
        body_background_fill_dark="#0a0c10",
        background_fill_primary="#10141b",
        background_fill_primary_dark="#10141b",
        background_fill_secondary="#151b24",
        background_fill_secondary_dark="#151b24",
        border_color_primary="#2a323f",
        border_color_primary_dark="#2a323f",
        body_text_color="#e6eaf0",
        body_text_color_dark="#e6eaf0",
        body_text_color_subdued="#8c98a8",
        body_text_color_subdued_dark="#8c98a8",
        block_background_fill="#151b24",
        block_background_fill_dark="#151b24",
        block_border_color="#2a323f",
        block_border_color_dark="#2a323f",
        block_label_background_fill="#10141b",
        block_label_background_fill_dark="#10141b",
        block_label_text_color="#8c98a8",
        block_label_text_color_dark="#8c98a8",
        block_title_text_color="#ffb020",
        block_title_text_color_dark="#ffb020",
        input_background_fill="#0a0c10",
        input_background_fill_dark="#0a0c10",
        input_border_color="#2a323f",
        input_border_color_dark="#2a323f",
        button_primary_background_fill="linear-gradient(180deg, #ffd166 0%, #ffb020 100%)",
        button_primary_background_fill_dark="linear-gradient(180deg, #ffd166 0%, #ffb020 100%)",
        button_primary_text_color="#1a1100",
        button_primary_text_color_dark="#1a1100",
        button_primary_border_color="#7a5000",
        button_primary_border_color_dark="#7a5000",
        button_secondary_background_fill="#1d2430",
        button_secondary_background_fill_dark="#1d2430",
        button_secondary_text_color="#e6eaf0",
        button_secondary_text_color_dark="#e6eaf0",
        slider_color="#ffb020",
        slider_color_dark="#ffb020",
    )
    theme.custom_css = INDUSTRIAL_CSS
    return theme


# ─────────────────────────────────────────────────────────────────────
#  UI construction
# ─────────────────────────────────────────────────────────────────────


HEADER_HTML = """
<div class="iad-header">
  <div class="iad-header-left">
    <div class="iad-logo">IAD</div>
    <div>
      <div class="iad-title">INDUSTRIAL ANOMALY DETECTION</div>
      <div class="iad-subtitle">VISION INSPECTION ▸ FEW-SHOT / ZERO-SHOT ▸ MULTI-MODEL</div>
    </div>
  </div>
  <div class="iad-header-right">
    <div class="iad-stripes" title="safety stripes"></div>
    <button id="iad-theme-toggle" class="iad-theme-toggle" type="button">
      <svg class="iad-icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      <svg class="iad-icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      <span class="iad-tg-label">THEME</span>
    </button>
  </div>
</div>
"""


# Injected into <head>; guaranteed to execute once the page loads.
THEME_TOGGLE_HEAD = """
<script>
(function() {
  function applySaved() {
    try {
      var saved = localStorage.getItem('iad-theme');
      if (saved === 'light') document.documentElement.classList.add('iad-light');
      else document.documentElement.classList.remove('iad-light');
    } catch (e) {}
  }
  applySaved();

  function bindToggle() {
    var btn = document.getElementById('iad-theme-toggle');
    if (!btn || btn.dataset.bound === '1') return false;
    btn.dataset.bound = '1';
    btn.addEventListener('click', function() {
      var h = document.documentElement;
      var isLight = h.classList.toggle('iad-light');
      try { localStorage.setItem('iad-theme', isLight ? 'light' : 'dark'); } catch (e) {}
    });
    return true;
  }
  // Gradio mounts components asynchronously — poll briefly until the button exists.
  var tries = 0;
  var t = setInterval(function() {
    if (bindToggle() || ++tries > 60) clearInterval(t);
  }, 100);
  // Also observe future re-mounts (e.g. during reloads).
  var mo = new MutationObserver(function() { bindToggle(); });
  mo.observe(document.body, { childList: true, subtree: true });
})();
</script>
"""


FOOTER_HTML = f"""
<div class="iad-footer">
  <span>▮ SYSTEM ONLINE — {str(DEVICE).upper()}</span>
  <span>QUALITY CONTROL • v0.1</span>
</div>
"""


def _method_info_html(method_key: str) -> str:
    spec = METHODS[method_key]
    req = (
        "REQUIRED"
        if spec.needs_nominal
        else ("OPTIONAL (few-shot)" if spec.supports_nominal else "NONE")
    )
    return (
        f'<div class="iad-method-tagline">▸ {spec.tagline} · '
        f"INPUT SIZE {spec.image_size}px · REFERENCES: {req}</div>"
        f'<div class="iad-method-notes">{spec.notes}</div>'
    )


def _on_method_change(method_key: str):
    spec = METHODS[method_key]
    gallery_label = (
        "NOMINAL REFERENCES  ▸  REQUIRED"
        if spec.needs_nominal
        else "NOMINAL REFERENCES  ▸  OPTIONAL"
    )
    return (
        gr.update(value=_method_info_html(method_key)),
        gr.update(label=gallery_label),
        gr.update(visible=method_key == "patchcore"),
        gr.update(visible=method_key == "anomalydino"),
        gr.update(visible=method_key == "anomalytipsv2"),
        gr.update(visible=method_key in {"winclip", "dictas"}),
        gr.update(visible=method_key in {"anomalydino", "anomalytipsv2"}),
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="IAD · Industrial Anomaly Detection",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row():
            # ── LEFT COLUMN: inputs ────────────────────────────────
            with gr.Column(scale=5):
                gr.HTML('<div class="iad-section-title">01 · SPECIMEN</div>')
                test_image = gr.Image(
                    label="TEST IMAGE",
                    type="numpy",
                    height=320,
                    sources=["upload", "clipboard"],
                    interactive=True,
                )

                gr.HTML('<div class="iad-section-title">02 · NOMINAL REFERENCES</div>')
                nominal_gallery = gr.Gallery(
                    label="NOMINAL REFERENCES  ▸  REQUIRED",
                    show_label=True,
                    columns=4,
                    rows=2,
                    height="auto",
                    allow_preview=True,
                    object_fit="cover",
                    interactive=True,
                    file_types=["image"],
                    elem_classes=["iad-gallery"],
                )

                gr.HTML('<div class="iad-section-title">03 · METHOD</div>')
                method_dropdown = gr.Dropdown(
                    label="INSPECTION METHOD",
                    choices=[(m.label, m.key) for m in METHODS.values()],
                    value="anomalydino",
                    interactive=True,
                )
                method_info = gr.HTML(_method_info_html("anomalydino"))

                threshold = gr.Slider(
                    label="DECISION THRESHOLD (top-1 % signal)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.35,
                    elem_classes=["iad-slider"],
                )

                # ── Advanced parameters ────────────────────────────
                with gr.Accordion(
                    "▸ ADVANCED PARAMETERS", open=False, elem_classes=["iad-accordion"]
                ):
                    # PatchCore-specific
                    with gr.Group(
                        visible=False, elem_classes=["iad-group"]
                    ) as grp_patchcore:
                        gr.Markdown("**PatchCore**")
                        coreset_ratio = gr.Slider(
                            label="CORESET SAMPLING RATIO",
                            minimum=0.01,
                            maximum=1.0,
                            step=0.01,
                            value=0.1,
                            elem_classes=["iad-slider"],
                        )
                        num_neighbors = gr.Slider(
                            label="NUM NEIGHBOURS (b)",
                            minimum=1,
                            maximum=32,
                            step=1,
                            value=9,
                            elem_classes=["iad-slider"],
                        )

                    # AnomalyDINO-specific
                    with gr.Group(
                        visible=True, elem_classes=["iad-group"]
                    ) as grp_adino:
                        gr.Markdown("**AnomalyDINO**")
                        dino_model = gr.Dropdown(
                            label="DINOV2 BACKBONE",
                            choices=[
                                "dinov2_vits14",
                                "dinov2_vitb14",
                                "dinov2_vitl14",
                            ],
                            value="dinov2_vits14",
                        )

                    # AnomalyTIPSv2-specific
                    with gr.Group(
                        visible=False, elem_classes=["iad-group"]
                    ) as grp_tips:
                        gr.Markdown("**AnomalyTIPSv2**")
                        tips_model = gr.Dropdown(
                            label="TIPSV2 BACKBONE",
                            choices=[
                                "google/tipsv2-b14",
                                "google/tipsv2-l14",
                            ],
                            value="google/tipsv2-b14",
                        )

                    # Shared DINO/TIPS options
                    with gr.Group(
                        visible=True, elem_classes=["iad-group"]
                    ) as grp_shared_dino:
                        gr.Markdown("**Feature-bank options**")
                        with gr.Row():
                            adino_masking = gr.Checkbox(
                                label="PCA FG MASKING",
                                value=True,
                            )
                            adino_rotation = gr.Checkbox(
                                label="ROTATE REFERENCES",
                                value=True,
                            )
                        adino_top_percent = gr.Slider(
                            label="TOP-K % FOR SCORE",
                            minimum=0.001,
                            maximum=0.1,
                            step=0.001,
                            value=0.01,
                            elem_classes=["iad-slider"],
                        )

                    # WinCLIP-specific
                    with gr.Group(
                        visible=False, elem_classes=["iad-group"]
                    ) as grp_winclip:
                        gr.Markdown("**WinCLIP**")
                        category = gr.Textbox(
                            label="OBJECT CATEGORY (for prompts)",
                            value="object",
                            placeholder="e.g. candle, capsule, pcb",
                        )
                        k_shot = gr.Slider(
                            label="K-SHOT (0 = zero-shot)",
                            minimum=0,
                            maximum=16,
                            step=1,
                            value=0,
                            elem_classes=["iad-slider"],
                        )

                with gr.Row():
                    run_btn = gr.Button("▶ RUN INSPECTION", variant="primary", scale=3)
                    clear_btn = gr.Button("CLEAR", variant="secondary", scale=1)

            # ── RIGHT COLUMN: outputs ──────────────────────────────
            with gr.Column(scale=7):
                gr.HTML('<div class="iad-section-title">04 · VERDICT</div>')
                verdict_html = gr.HTML(
                    """
                    <div class="iad-verdict iad-verdict-standby">
                      <span class="iad-badge-dot"></span>
                      <span class="iad-verdict-label">STANDBY</span>
                      <span class="iad-verdict-conf">AWAITING INPUT</span>
                    </div>
                    """
                )
                metrics_html = gr.HTML(
                    """
                    <div class="iad-metrics">
                      <div class="iad-metric">
                        <div class="iad-metric-label">STATUS</div>
                        <div class="iad-metric-value">—</div>
                      </div>
                    </div>
                    """
                )

                gr.HTML('<div class="iad-section-title">05 · ANOMALY MAP</div>')
                with gr.Row():
                    overlay_img = gr.Image(
                        label="OVERLAY (HEATMAP ON SPECIMEN)",
                        type="numpy",
                        height=360,
                        interactive=False,
                    )
                    heatmap_img = gr.Image(
                        label="HEATMAP",
                        type="numpy",
                        height=360,
                        interactive=False,
                    )

                gr.HTML('<div class="iad-section-title">06 · LOG</div>')
                log_box = gr.Textbox(
                    label="EVENT LOG",
                    lines=3,
                    interactive=False,
                    elem_classes=["iad-log"],
                )

        gr.HTML(FOOTER_HTML)

        # ── Wiring ──────────────────────────────────────────────────
        method_dropdown.change(
            _on_method_change,
            inputs=[method_dropdown],
            outputs=[
                method_info,
                nominal_gallery,
                grp_patchcore,
                grp_adino,
                grp_tips,
                grp_winclip,
                grp_shared_dino,
            ],
        )

        run_btn.click(
            run_inference,
            inputs=[
                test_image,
                nominal_gallery,
                method_dropdown,
                threshold,
                coreset_ratio,
                num_neighbors,
                dino_model,
                adino_masking,
                adino_rotation,
                adino_top_percent,
                tips_model,
                category,
                k_shot,
            ],
            outputs=[verdict_html, metrics_html, overlay_img, heatmap_img, log_box],
        )

        def _clear():
            return (
                None,
                None,
                """
                <div class="iad-verdict iad-verdict-standby">
                  <span class="iad-badge-dot"></span>
                  <span class="iad-verdict-label">STANDBY</span>
                  <span class="iad-verdict-conf">AWAITING INPUT</span>
                </div>
                """,
                """<div class="iad-metrics"><div class="iad-metric">
                    <div class="iad-metric-label">STATUS</div>
                    <div class="iad-metric-value">—</div></div></div>""",
                None,
                None,
                "",
            )

        clear_btn.click(
            _clear,
            inputs=None,
            outputs=[
                test_image,
                nominal_gallery,
                verdict_html,
                metrics_html,
                overlay_img,
                heatmap_img,
                log_box,
            ],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IAD Gradio demo")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    demo = build_ui()
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        debug=args.debug,
        theme=_industrial_theme(),
        css=INDUSTRIAL_CSS,
        head=THEME_TOGGLE_HEAD,
    )


if __name__ == "__main__":
    main()
