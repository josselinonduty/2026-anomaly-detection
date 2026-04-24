"""Anomaly detection training pipeline.

Usage
-----
    python main.py --model dictas --category candle --k_shot 4
    python main.py --model patchcore --category candle
    python main.py --model patchcore --category all
    python main.py --help
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import mlflow
import pytorch_lightning as pl
import torch
from codecarbon import EmissionsTracker
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.strategies import SingleDeviceStrategy

from lib.accelerators import XPUAccelerator
from lib.data import (
    DATASET_NAMES,
    create_datamodule,
)
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
from lib.lightning.callbacks import InferenceSpeedMonitor, MemoryMonitor
from lib.utils.checkpoint import make_checkpoint_dir, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly Detection")

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default="visa",
        choices=DATASET_NAMES,
        help="Dataset to use (default: visa).",
    )
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument(
        "--category",
        type=str,
        default="all",
        help=(
            "Category/categories to train on. Accepts a single category "
            "(e.g. 'candle'), a comma-separated list (e.g. 'candle,capsules'), "
            "or 'all' (default) for every category in the dataset. "
            "Per-class models iterate and train one model per category."
        ),
    )
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
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=256)

    # PatchCore-specific
    parser.add_argument(
        "--coreset_sampling_ratio",
        type=float,
        default=0.01,
        help="Fraction of patches kept in the coreset (PatchCore).",
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=9,
        help="Neighbours for image-score re-weighting (PatchCore).",
    )

    # EfficientAd-specific
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="PDN variant for EfficientAd (small or medium).",
    )

    # SubspaceAD-specific
    parser.add_argument(
        "--subspacead_backbone",
        type=str,
        default="dinov2_vitg14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
        help="DINOv2 backbone for SubspaceAD (default Giant).",
    )
    parser.add_argument(
        "--subspacead_resolution",
        type=int,
        default=672,
        help="Input resolution for SubspaceAD (default 672).",
    )
    parser.add_argument(
        "--subspacead_layers",
        type=int,
        nargs="+",
        default=None,
        help="DINOv2 layers to average for SubspaceAD (default: auto middle-7).",
    )
    parser.add_argument(
        "--subspacead_pca_ev",
        type=float,
        default=0.99,
        help="PCA explained variance threshold τ for SubspaceAD (default 0.99).",
    )
    parser.add_argument(
        "--subspacead_aug_count",
        type=int,
        default=30,
        help="Number of random rotations per normal image (SubspaceAD, default 30).",
    )
    parser.add_argument(
        "--subspacead_gaussian_sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma for anomaly map smoothing (SubspaceAD, default 4.0).",
    )
    parser.add_argument(
        "--subspacead_top_percent",
        type=float,
        default=0.01,
        help="Top patch fraction for image-level scoring (SubspaceAD, default 0.01).",
    )

    # AnomalyEUPE-specific
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
        help="EUPE ONNX backbone variant (default: eupe_vitb16). Append _int8 for quantised.",
    )
    parser.add_argument(
        "--eupe_global_weight",
        type=float,
        default=0.3,
        help="Weight of global (CLS) score vs local (patch) score (default 0.3).",
    )

    # AnomalyDINO-specific
    parser.add_argument(
        "--dino_model",
        type=str,
        default="dinov2_vits14",
        choices=[
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ],
        help="DINOv2 backbone for AnomalyDINO.",
    )
    parser.add_argument(
        "--smaller_edge_size",
        type=int,
        default=448,
        help="Resize shorter edge to this value (AnomalyDINO).",
    )
    parser.add_argument(
        "--no_masking",
        action="store_true",
        default=False,
        help="Disable PCA-based foreground masking (AnomalyDINO).",
    )
    parser.add_argument(
        "--no_rotation",
        action="store_true",
        default=False,
        help="Disable rotation augmentation of references (AnomalyDINO).",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=70000,
        help="Total training iterations for EfficientAd student + AE.",
    )
    parser.add_argument(
        "--teacher_pretrain_steps",
        type=int,
        default=10000,
        help="KD pretraining iterations for the EfficientAd teacher PDN.",
    )
    parser.add_argument(
        "--teacher_weights",
        type=str,
        default=None,
        help="Path to pre-trained teacher PDN weights (skips KD pretraining).",
    )

    # WinCLIP-specific
    parser.add_argument(
        "--clip_backbone",
        type=str,
        default="ViT-B-16-plus-240",
        help="OpenCLIP model name for WinCLIP.",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion400m_e32",
        help="Pretrained dataset for WinCLIP.",
    )
    parser.add_argument(
        "--k_shot",
        type=int,
        default=0,
        help="Normal reference shots (WinCLIP+: 0=zero-shot; DictAS: ≥1).",
    )

    # AnomalyTIPSv2-specific
    parser.add_argument(
        "--tips_model",
        type=str,
        default="google/tipsv2-b14",
        choices=[
            "google/tipsv2-b14",
            "google/tipsv2-l14",
            "google/tipsv2-so400m14",
            "google/tipsv2-g14",
        ],
        help="TIPSv2 backbone for AnomalyTIPSv2.",
    )

    # FeatureMatch-specific
    parser.add_argument(
        "--fm_descriptor",
        type=str,
        default="sift",
        choices=["sift", "orb"],
        help="OpenCV descriptor for FeatureMatch (default sift).",
    )
    parser.add_argument(
        "--fm_map_mode",
        type=str,
        default="dense",
        choices=["dense", "ssim"],
        help="Anomaly map: 'dense' (abs diff) or 'ssim' (1-SSIM).",
    )
    parser.add_argument(
        "--fm_ratio_thresh",
        type=float,
        default=0.75,
        help="Lowe's ratio test threshold (FeatureMatch).",
    )
    parser.add_argument(
        "--fm_blur_sigma",
        type=float,
        default=7.0,
        help="Gaussian sigma for difference map smoothing (FeatureMatch).",
    )

    # DictAS-specific
    parser.add_argument(
        "--dictas_backbone",
        type=str,
        default="ViT-L-14-336",
        help="OpenCLIP backbone for DictAS (default ViT-L-14-336).",
    )
    parser.add_argument(
        "--dictas_pretrained",
        type=str,
        default="openai",
        help="Pretrained weights tag for DictAS (default openai).",
    )
    parser.add_argument(
        "--dictas_layer_indices",
        type=int,
        nargs="+",
        default=[6, 12, 18, 24],
        help="1-based CLIP transformer blocks for feature extraction.",
    )
    parser.add_argument(
        "--dictas_pool_kernel",
        type=int,
        default=3,
        help="Avg-pool kernel on the patch grid (Appendix A.4).",
    )
    parser.add_argument(
        "--dictas_lookup",
        type=str,
        default="sparse",
        choices=["sparse", "dense", "max"],
        help="Dictionary lookup strategy (Eq. 5).",
    )
    parser.add_argument(
        "--dictas_lambda_cqc",
        type=float,
        default=0.1,
        help="Weight λ1 for the Contrastive Query Constraint (Eq. 10).",
    )
    parser.add_argument(
        "--dictas_lambda_tac",
        type=float,
        default=0.1,
        help="Weight λ2 for the Text Alignment Constraint (Eq. 10).",
    )

    # Training
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)

    # MLflow
    parser.add_argument("--experiment_name", type=str, default="visa-anomaly-detection")
    parser.add_argument("--tracking_uri", type=str, default="sqlite:///mlflow.db")

    return parser.parse_args()


# No models currently support unified multi-class training.
MULTI_CLASS_MODELS: set[str] = set()


def _resolve_categories(args: argparse.Namespace) -> list[str]:
    """Turn the raw ``--category`` string into a list of category names.

    * ``"all"`` or ``""`` → every category in the dataset
    * comma-separated → split into a list
    * single name → ``[name]``
    """
    raw = args.category.strip().lower()
    if raw in ("all", ""):
        # Need a temporary datamodule just to list available categories.
        dm = create_datamodule(args.dataset, dataset_root=args.dataset_root)
        return dm.categories
    return [c.strip() for c in args.category.split(",") if c.strip()]


def train_single(
    args: argparse.Namespace,
    category: str | list[str],
) -> None:
    """Run a full train + test cycle for *category* (str or list for multi-class)."""

    cat_label = "+".join(category) if isinstance(category, list) else category

    # ── MLflow setup ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=f"{args.model}-{cat_label}",
        log_model=True,
        save_dir="mlruns",
    )

    # ── CodeCarbon tracker ───────────────────────────────────────────
    Path("logs").mkdir(exist_ok=True)
    emissions_tracker = EmissionsTracker(
        project_name=f"{args.experiment_name}/{args.model}/{cat_label}",
        output_dir="logs",
        log_level="warning",
    )

    # ── DataModule ───────────────────────────────────────────────────
    extra_dm_kwargs: dict = {}

    datamodule = create_datamodule(
        args.dataset,
        dataset_root=args.dataset_root,
        category=category,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **extra_dm_kwargs,
    )

    # ── Model ────────────────────────────────────────────────────────
    model: (
        PatchCoreModule
        | EfficientAdModule
        | AutoencoderModule
        | AnomalyDINOModule
        | AnomalyEUPEModule
        | AnomalyTIPSv2Module
        | WinCLIPModule
        | DictASModule
        | SubspaceADModule
        | FeatureMatchModule
    )
    if args.model == "patchcore":
        model = PatchCoreModule(
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            num_neighbors=args.num_neighbors,
            image_size=args.image_size,
        )
        args.max_epochs = 1
    elif args.model == "efficientad":
        args.batch_size = 1
        datamodule.batch_size = args.batch_size
        model = EfficientAdModule(
            model_size=args.model_size,
            train_steps=args.train_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            teacher_pretrain_steps=args.teacher_pretrain_steps,
            teacher_weights=args.teacher_weights,
        )
        datamodule.setup("fit")
        steps_per_epoch = len(datamodule.train_dataloader())
        if args.max_epochs is None:
            args.max_epochs = int(math.ceil(args.train_steps / steps_per_epoch))
        model.set_train_dataloader(datamodule.train_dataloader())
    elif args.model == "autoencoder":
        model = AutoencoderModule(
            in_channels=3,
            base_channels=args.base_channels,
            depth=args.depth,
            latent_dim=args.latent_dim,
            image_size=args.image_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.model == "anomalydino":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        anomalydino_transform = get_eval_transforms(args.smaller_edge_size)
        anomalydino_mask_transform = get_mask_transforms(args.smaller_edge_size)
        extra_dm_kwargs.update(
            train_transform=anomalydino_transform,
            eval_transform=anomalydino_transform,
            mask_transform=anomalydino_mask_transform,
        )
        datamodule = create_datamodule(
            args.dataset,
            dataset_root=args.dataset_root,
            category=category,
            image_size=args.smaller_edge_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = args.smaller_edge_size
        model = AnomalyDINOModule(
            model_name=args.dino_model,
            smaller_edge_size=args.smaller_edge_size,
            masking=not args.no_masking,
            rotation=not args.no_rotation,
            image_size=args.image_size,
        )
        args.max_epochs = 1
    elif args.model == "anomalyeupe":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        eupe_image_size = 224
        eupe_transform = get_eval_transforms(eupe_image_size)
        eupe_mask_transform = get_mask_transforms(eupe_image_size)
        extra_dm_kwargs.update(
            train_transform=eupe_transform,
            eval_transform=eupe_transform,
            mask_transform=eupe_mask_transform,
        )
        datamodule = create_datamodule(
            args.dataset,
            dataset_root=args.dataset_root,
            category=category,
            image_size=eupe_image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = eupe_image_size
        model = AnomalyEUPEModule(
            model_name=args.eupe_model_name,
            masking=not args.no_masking,
            rotation=not args.no_rotation,
            global_weight=args.eupe_global_weight,
            image_size=eupe_image_size,
        )
        args.max_epochs = 1
    elif args.model == "winclip":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        assert isinstance(
            category, str
        ), "WinCLIP requires a single category for text prompts."
        winclip_image_size = 240
        winclip_transform = get_eval_transforms(winclip_image_size)
        winclip_mask_transform = get_mask_transforms(winclip_image_size)
        extra_dm_kwargs.update(
            train_transform=winclip_transform,
            eval_transform=winclip_transform,
            mask_transform=winclip_mask_transform,
        )
        datamodule = create_datamodule(
            args.dataset,
            dataset_root=args.dataset_root,
            category=category,
            image_size=winclip_image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = winclip_image_size
        model = WinCLIPModule(
            category=category,
            backbone=args.clip_backbone,
            pretrained=args.clip_pretrained,
            scales=(2, 3),
            image_size=winclip_image_size,
            k_shot=args.k_shot,
        )
        args.max_epochs = 1
    elif args.model == "dictas":
        # DictAS: CLIP ViT-L/14-336 at 336×336 (Appendix A.4).
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        assert isinstance(
            category, str
        ), "DictAS requires a single category for text prompts."
        dictas_image_size = 336
        dictas_transform = get_eval_transforms(dictas_image_size)
        dictas_mask_transform = get_mask_transforms(dictas_image_size)
        extra_dm_kwargs.update(
            train_transform=dictas_transform,
            eval_transform=dictas_transform,
            mask_transform=dictas_mask_transform,
        )
        datamodule = create_datamodule(
            args.dataset,
            dataset_root=args.dataset_root,
            category=category,
            image_size=dictas_image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = dictas_image_size
        model = DictASModule(
            category=category,
            backbone=args.dictas_backbone,
            pretrained=args.dictas_pretrained,
            image_size=dictas_image_size,
            layer_indices=tuple(args.dictas_layer_indices),
            pool_kernel=args.dictas_pool_kernel,
            lookup=args.dictas_lookup,
            k_shot=max(1, args.k_shot),
            lambda_cqc=args.dictas_lambda_cqc,
            lambda_tac=args.dictas_lambda_tac,
            lr=args.lr if args.lr != 1e-3 else 1e-4,  # paper default 1e-4
        )
        if args.max_epochs is None:
            args.max_epochs = 30  # Appendix A.4
    elif args.model == "subspacead":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        subspacead_resolution = args.subspacead_resolution
        subspacead_transform = get_eval_transforms(subspacead_resolution)
        subspacead_mask_transform = get_mask_transforms(subspacead_resolution)
        extra_dm_kwargs.update(
            train_transform=subspacead_transform,
            eval_transform=subspacead_transform,
            mask_transform=subspacead_mask_transform,
        )
        datamodule = create_datamodule(
            args.dataset,
            dataset_root=args.dataset_root,
            category=category,
            image_size=subspacead_resolution,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = subspacead_resolution
        model = SubspaceADModule(
            model_name=args.subspacead_backbone,
            image_resolution=subspacead_resolution,
            layers=tuple(args.subspacead_layers) if args.subspacead_layers else None,
            pca_variance_threshold=args.subspacead_pca_ev,
            aug_count=args.subspacead_aug_count,
            gaussian_sigma=args.subspacead_gaussian_sigma,
            top_percent=args.subspacead_top_percent,
            image_size=subspacead_resolution,
        )
        args.max_epochs = 1
    elif args.model == "anomalytipsv2":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        tipsv2_transform = get_eval_transforms(args.smaller_edge_size)
        tipsv2_mask_transform = get_mask_transforms(args.smaller_edge_size)
        extra_dm_kwargs.update(
            train_transform=tipsv2_transform,
            eval_transform=tipsv2_transform,
            mask_transform=tipsv2_mask_transform,
        )
        datamodule = create_datamodule(
            args.dataset,
            dataset_root=args.dataset_root,
            category=category,
            image_size=args.smaller_edge_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = args.smaller_edge_size
        model = AnomalyTIPSv2Module(
            model_name=args.tips_model,
            smaller_edge_size=args.smaller_edge_size,
            masking=not args.no_masking,
            rotation=not args.no_rotation,
            image_size=args.image_size,
        )
        args.max_epochs = 1
    elif args.model == "feature_match":
        model = FeatureMatchModule(
            descriptor=args.fm_descriptor,
            image_size=args.image_size,
            map_mode=args.fm_map_mode,
            ratio_thresh=args.fm_ratio_thresh,
            blur_sigma=args.fm_blur_sigma,
        )
        args.max_epochs = 1
    else:
        msg = f"Unknown model: {args.model!r}"
        raise ValueError(msg)

    # ── Callbacks ────────────────────────────────────────────────────
    callbacks = [InferenceSpeedMonitor(), MemoryMonitor()]
    if args.model == "autoencoder":
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                patience=args.patience,
                mode="min",
            )
        )

    # ── Trainer ──────────────────────────────────────────────────────
    if args.max_epochs is None:
        args.max_epochs = 100

    use_xpu = args.accelerator == "xpu" or (
        args.accelerator == "auto"
        and hasattr(torch, "xpu")
        and torch.xpu.is_available()
    )

    if use_xpu:
        print("Using XPU accelerator")
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=XPUAccelerator(),
            strategy=SingleDeviceStrategy(device=torch.device("xpu", 0)),
            devices=1,
            logger=mlflow_logger,
            callbacks=callbacks,
            log_every_n_steps=10,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            logger=mlflow_logger,
            callbacks=callbacks,
            log_every_n_steps=10,
        )

    for k, v in vars(args).items():
        mlflow_logger.experiment.set_tag(mlflow_logger.run_id, f"cli/{k}", str(v))

    emissions_tracker.start()
    try:
        datamodule.setup("fit")
        trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())

        ckpt_dir = make_checkpoint_dir("checkpoints", args.model, category)
        model.save_checkpoint(ckpt_dir)
        save_metadata(
            ckpt_dir,
            model_name=args.model,
            category=category,
            extra={k: str(v) for k, v in vars(args).items()},
        )

        datamodule.setup("test")
        trainer.test(model, datamodule.test_dataloader())
    finally:
        emissions = emissions_tracker.stop()
        if emissions is not None:
            mlflow_logger.log_metrics({"carbon/emissions_kg": emissions})

    print(f"\n✓ Training complete for category '{cat_label}'.")
    print(f"  Checkpoint saved to: {ckpt_dir.resolve()}")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {args.tracking_uri}")


def main() -> None:
    args = parse_args()
    categories = _resolve_categories(args)

    if args.model in MULTI_CLASS_MODELS and len(categories) > 1:
        # Multi-class models train a single unified model on all categories.
        print(f"[multi-class] Training unified {args.model} on {categories}")
        train_single(args, categories)
    else:
        # Per-class models iterate and train one model per category.
        for i, cat in enumerate(categories, 1):
            print(f"\n{'='*60}")
            print(f"[{i}/{len(categories)}] Training {args.model} on '{cat}'")
            print(f"{'='*60}")
            train_single(copy.deepcopy(args), cat)


if __name__ == "__main__":
    main()
