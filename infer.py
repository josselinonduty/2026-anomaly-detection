"""Run inference with a trained anomaly detection model.

Usage
-----
    python infer.py --model patchcore --category candle
    python infer.py --model efficientad --category capsules --checkpoint_dir checkpoints/efficientad/capsules/20260414_153012
    python infer.py --help

Loads the latest (or specified) checkpoint for the given model/category,
runs inference on the test split, and saves side-by-side image triplets
(input | GT mask | predicted anomaly map) to an output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lib.data import (
    DATASET_NAMES,
    create_datamodule,
)
from lib.data.transforms import get_eval_transforms, get_mask_transforms
from lib.lightning import (
    AnomalyDINOModule,
    AnomalyEUPEModule,
    AnomalyTIPSv2Module,
    AutoencoderModule,
    DictASModule,
    EfficientAdModule,
    FeatureMatchModule,
    PatchCoreModule,
    SubspaceADModule,
    WinCLIPModule,
)
from lib.utils.checkpoint import latest_checkpoint_dir

# ImageNet de-normalisation constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_LOADER_MAP = {
    "autoencoder": AutoencoderModule,
    "patchcore": PatchCoreModule,
    "efficientad": EfficientAdModule,
    "anomalydino": AnomalyDINOModule,
    "anomalyeupe": AnomalyEUPEModule,
    "anomalytipsv2": AnomalyTIPSv2Module,
    "winclip": WinCLIPModule,
    "dictas": DictASModule,
    "subspacead": SubspaceADModule,
    "feature_match": FeatureMatchModule,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly Detection — Inference")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="visa",
        choices=DATASET_NAMES,
        help="Dataset to use (default: visa).",
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--category", type=str, default="candle")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "autoencoder",
            "patchcore",
            "efficientad",
            "anomalydino",
            "anomalyeupe",
            "anomalytipsv2",
            "winclip",
            "dictas",
            "subspacead",
            "feature_match",
        ],
        help="Model architecture to use.",
    )

    # Checkpoint
    parser.add_argument(
        "--eupe_model_name",
        type=str,
        default="eupe_vitb16",
        choices=[
            "eupe_vitt16",
            "eupe_vitt16_int8",
            "eupe_vits16",
            "eupe_vits16_int8",
            "eupe_vitb16",
            "eupe_vitb16_int8",
        ],
        help="EUPE ONNX backbone variant (default: eupe_vitb16).",
    )

    # SubspaceAD-specific
    parser.add_argument(
        "--subspacead_resolution",
        type=int,
        default=672,
        help="Input resolution for SubspaceAD (default 672).",
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory. "
        "Defaults to the latest checkpoint under checkpoints/<model>/<category>/.",
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="checkpoints",
        help="Root directory for checkpoints (used when --checkpoint_dir is not set).",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save inference results. "
        "Defaults to output/<model>/<category>/.",
    )

    # Device
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


# ── Per-model anomaly map extraction ────────────────────────────────


@torch.no_grad()
def _predict_autoencoder(
    model: AutoencoderModule, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    x_hat, _ = model.model(images)
    return (images - x_hat).pow(2).mean(dim=1)


@torch.no_grad()
def _predict_patchcore(model: PatchCoreModule, images: torch.Tensor) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model.model.predict(images)
    return anomaly_maps


@torch.no_grad()
def _predict_efficientad(
    model: EfficientAdModule, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    map_combined, _, _ = model._predict_maps(images)
    map_combined = F.pad(map_combined, (4, 4, 4, 4))
    map_combined = F.interpolate(
        map_combined,
        size=images.shape[-1],
        mode="bilinear",
        align_corners=False,
    )
    return map_combined[:, 0]


@torch.no_grad()
def _predict_anomalydino(
    model: AnomalyDINOModule, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model(images)
    return anomaly_maps[:, 0]


@torch.no_grad()
def _predict_anomalyeupe(
    model: AnomalyEUPEModule, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model(images)
    return anomaly_maps[:, 0]


@torch.no_grad()
def _predict_anomalytipsv2(
    model: AnomalyTIPSv2Module, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model(images)
    return anomaly_maps[:, 0]


@torch.no_grad()
def _predict_winclip(
    model: WinCLIPModule, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model(images)
    return anomaly_maps


@torch.no_grad()
def _predict_dictas(model: DictASModule, images: torch.Tensor) -> torch.Tensor:
    """Return (B, H, W) anomaly maps.  Requires a reference gallery."""
    refs = model._reference_images
    if refs is None:
        raise RuntimeError(
            "DictAS checkpoint is missing the normal reference gallery; "
            "re-run main.py to build it."
        )
    refs = refs.to(images.device).unsqueeze(0).expand(images.shape[0], -1, -1, -1, -1)
    _, anomaly_maps = model.model.predict(images, refs)
    return anomaly_maps


@torch.no_grad()
def _predict_subspacead(model: SubspaceADModule, images: torch.Tensor) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model(images)
    return anomaly_maps[:, 0]


def _predict_feature_match(
    model: FeatureMatchModule, images: torch.Tensor
) -> torch.Tensor:
    """Return (B, H, W) anomaly maps."""
    _, anomaly_maps = model.model.predict(images)
    return anomaly_maps


_PREDICT_FN = {
    "autoencoder": _predict_autoencoder,
    "patchcore": _predict_patchcore,
    "efficientad": _predict_efficientad,
    "anomalydino": _predict_anomalydino,
    "anomalyeupe": _predict_anomalyeupe,
    "anomalytipsv2": _predict_anomalytipsv2,
    "winclip": _predict_winclip,
    "dictas": _predict_dictas,
    "subspacead": _predict_subspacead,
    "feature_match": _predict_feature_match,
}


# ── Helpers ──────────────────────────────────────────────────────────


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to a (H, W, 3) uint8 numpy array."""
    img = tensor.cpu() * _STD + _MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """Convert a (H, W) float mask to a uint8 numpy array."""
    m = mask.cpu().numpy()
    if m.max() > 0:
        m = m / m.max()
    return (m * 255).astype(np.uint8)


def _save_triplet(
    out_path: Path,
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> None:
    """Save an (input | GT mask | predicted mask) triplet as a single image."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_mask, cmap="inferno", vmin=0, vmax=255)
    axes[2].set_title("Predicted")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Resolve device ───────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", 0)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu", 0)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── Resolve checkpoint ───────────────────────────────────────────
    if args.checkpoint_dir is not None:
        ckpt_dir = Path(args.checkpoint_dir)
    else:
        ckpt_dir = latest_checkpoint_dir(
            args.checkpoint_root, args.model, args.category
        )
    if ckpt_dir is None or not ckpt_dir.exists():
        raise FileNotFoundError(
            f"No checkpoint found for {args.model}/{args.category}. "
            f"Train a model first or pass --checkpoint_dir explicitly."
        )

    # ── Load model ───────────────────────────────────────────────────
    loader_cls = _LOADER_MAP[args.model]
    model = loader_cls.load_checkpoint(ckpt_dir, map_location="cpu")
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt_dir}")

    # ── DataModule ───────────────────────────────────────────────────
    extra_dm_kwargs: dict = {}
    image_size = args.image_size
    if args.model == "anomalydino":
        image_size = 448
        extra_dm_kwargs.update(
            train_transform=get_eval_transforms(image_size),
            eval_transform=get_eval_transforms(image_size),
            mask_transform=get_mask_transforms(image_size),
        )
    elif args.model == "anomalyeupe":
        image_size = 224
        extra_dm_kwargs.update(
            train_transform=get_eval_transforms(image_size),
            eval_transform=get_eval_transforms(image_size),
            mask_transform=get_mask_transforms(image_size),
        )
    elif args.model == "anomalytipsv2":
        image_size = 448
        extra_dm_kwargs.update(
            train_transform=get_eval_transforms(image_size),
            eval_transform=get_eval_transforms(image_size),
            mask_transform=get_mask_transforms(image_size),
        )
    elif args.model == "winclip":
        image_size = 240
        extra_dm_kwargs.update(
            train_transform=get_eval_transforms(image_size),
            eval_transform=get_eval_transforms(image_size),
            mask_transform=get_mask_transforms(image_size),
        )
    elif args.model == "dictas":
        image_size = 336
        extra_dm_kwargs.update(
            train_transform=get_eval_transforms(image_size),
            eval_transform=get_eval_transforms(image_size),
            mask_transform=get_mask_transforms(image_size),
        )
    elif args.model == "subspacead":
        image_size = args.subspacead_resolution
        extra_dm_kwargs.update(
            train_transform=get_eval_transforms(image_size),
            eval_transform=get_eval_transforms(image_size),
            mask_transform=get_mask_transforms(image_size),
        )

    datamodule = create_datamodule(
        args.dataset,
        dataset_root=args.dataset_root,
        category=args.category,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **extra_dm_kwargs,
    )
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    # ── Output directory ─────────────────────────────────────────────
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("output") / args.model / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference loop ───────────────────────────────────────────────
    predict_fn = _PREDICT_FN[args.model]
    sample_idx = 0

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"]  # (B, H, W)

        anomaly_maps = predict_fn(model, images)  # (B, H, W)

        # Resize anomaly maps to match the mask spatial size if needed
        if anomaly_maps.shape[-2:] != masks.shape[-2:]:
            anomaly_maps = F.interpolate(
                anomaly_maps.unsqueeze(1).cpu(),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        else:
            anomaly_maps = anomaly_maps.cpu()

        for i in range(images.shape[0]):
            img_np = _denormalize(images[i])
            gt_np = _mask_to_numpy(masks[i])
            pred_np = _mask_to_numpy(anomaly_maps[i])

            _save_triplet(
                out_dir / f"{sample_idx:04d}.png",
                img_np,
                gt_np,
                pred_np,
            )
            sample_idx += 1

    print(f"\n✓ Inference complete — {sample_idx} images saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
