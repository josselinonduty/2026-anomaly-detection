"""Anomaly detection training pipeline.

Usage
-----
    python main.py --model dictas --category candle --k_shot 4
    python main.py --model patchcore --category candle
    python main.py --help
"""

from __future__ import annotations

import argparse
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
    VisADataModule,
    get_dinomaly_mask_transforms,
    get_dinomaly_transforms,
)
from lib.lightning import (
    AnomalyDINOModule,
    AnomalyTIPSv2Module,
    AutoencoderModule,
    DictASModule,
    DinomalyModule,
    EfficientAdModule,
    PatchCoreModule,
    WinCLIPModule,
)
from lib.lightning.callbacks import InferenceSpeedMonitor, MemoryMonitor
from lib.utils.checkpoint import make_checkpoint_dir, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisA Anomaly Detection")

    # Data
    parser.add_argument("--dataset_root", type=str, default="datasets/visa")
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
            "dinomaly",
            "efficientad",
            "anomalydino",
            "anomalytipsv2",
            "winclip",
            "dictas",
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

    # Dinomaly-specific
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov2reg_vit_base_14",
        choices=[
            "dinov2reg_vit_small_14",
            "dinov2reg_vit_base_14",
            "dinov2reg_vit_large_14",
        ],
        help="DINOv2 backbone for Dinomaly.",
    )
    parser.add_argument(
        "--total_iters",
        type=int,
        default=10000,
        help="Total training iterations (Dinomaly).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Bottleneck dropout rate (Dinomaly).",
    )

    # EfficientAd-specific
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "medium"],
        help="PDN variant for EfficientAd (small or medium).",
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


def main() -> None:
    args = parse_args()

    # ── MLflow setup ─────────────────────────────────────────────────
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow_logger = MLFlowLogger(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        run_name=f"{args.model}-{args.category}",
        log_model=True,
        save_dir="mlruns",
    )

    # ── CodeCarbon tracker ───────────────────────────────────────────
    Path("logs").mkdir(exist_ok=True)
    emissions_tracker = EmissionsTracker(
        project_name=f"{args.experiment_name}/{args.model}/{args.category}",
        output_dir="logs",
        log_level="warning",
    )

    # ── DataModule ───────────────────────────────────────────────────
    extra_dm_kwargs: dict = {}
    if args.model == "dinomaly":
        dinomaly_transform = get_dinomaly_transforms()
        dinomaly_mask_transform = get_dinomaly_mask_transforms()
        extra_dm_kwargs.update(
            train_transform=dinomaly_transform,
            eval_transform=dinomaly_transform,
            mask_transform=dinomaly_mask_transform,
        )
        args.batch_size = min(args.batch_size, 16)  # paper default

    datamodule = VisADataModule(
        dataset_root=args.dataset_root,
        category=args.category,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **extra_dm_kwargs,
    )

    # ── Model ────────────────────────────────────────────────────────
    model: (
        PatchCoreModule
        | DinomalyModule
        | EfficientAdModule
        | AutoencoderModule
        | AnomalyDINOModule
        | AnomalyTIPSv2Module
        | WinCLIPModule
        | DictASModule
    )
    if args.model == "patchcore":
        model = PatchCoreModule(
            coreset_sampling_ratio=args.coreset_sampling_ratio,
            num_neighbors=args.num_neighbors,
            image_size=args.image_size,
        )
        args.max_epochs = 1
    elif args.model == "dinomaly":
        args.image_size = 392
        model = DinomalyModule(
            backbone=args.backbone,
            dropout=args.dropout,
            total_iters=args.total_iters,
        )
        datamodule.image_size = args.image_size
        datamodule.batch_size = args.batch_size
        datamodule.setup("fit")
        steps_per_epoch = len(datamodule.train_dataloader())
        if args.max_epochs is None:
            args.max_epochs = int(math.ceil(args.total_iters / steps_per_epoch))
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
        datamodule = VisADataModule(
            dataset_root=args.dataset_root,
            category=args.category,
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
    elif args.model == "winclip":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        winclip_image_size = 240
        winclip_transform = get_eval_transforms(winclip_image_size)
        winclip_mask_transform = get_mask_transforms(winclip_image_size)
        extra_dm_kwargs.update(
            train_transform=winclip_transform,
            eval_transform=winclip_transform,
            mask_transform=winclip_mask_transform,
        )
        datamodule = VisADataModule(
            dataset_root=args.dataset_root,
            category=args.category,
            image_size=winclip_image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = winclip_image_size
        model = WinCLIPModule(
            category=args.category,
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

        dictas_image_size = 336
        dictas_transform = get_eval_transforms(dictas_image_size)
        dictas_mask_transform = get_mask_transforms(dictas_image_size)
        extra_dm_kwargs.update(
            train_transform=dictas_transform,
            eval_transform=dictas_transform,
            mask_transform=dictas_mask_transform,
        )
        datamodule = VisADataModule(
            dataset_root=args.dataset_root,
            category=args.category,
            image_size=dictas_image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            **extra_dm_kwargs,
        )
        args.image_size = dictas_image_size
        model = DictASModule(
            category=args.category,
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
    elif args.model == "anomalytipsv2":
        from lib.data.transforms import get_eval_transforms, get_mask_transforms

        tipsv2_transform = get_eval_transforms(args.smaller_edge_size)
        tipsv2_mask_transform = get_mask_transforms(args.smaller_edge_size)
        extra_dm_kwargs.update(
            train_transform=tipsv2_transform,
            eval_transform=tipsv2_transform,
            mask_transform=tipsv2_mask_transform,
        )
        datamodule = VisADataModule(
            dataset_root=args.dataset_root,
            category=args.category,
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

        datamodule.setup("test")
        trainer.test(model, datamodule.test_dataloader())
    finally:
        emissions = emissions_tracker.stop()
        if emissions is not None:
            mlflow_logger.log_metrics({"carbon/emissions_kg": emissions})

    ckpt_dir = make_checkpoint_dir("checkpoints", args.model, args.category)
    model.save_checkpoint(ckpt_dir)
    save_metadata(
        ckpt_dir,
        model_name=args.model,
        category=args.category,
        extra={k: str(v) for k, v in vars(args).items()},
    )

    print(f"\n✓ Training complete for category '{args.category}'.")
    print(f"  Checkpoint saved to: {ckpt_dir.resolve()}")
    print(f"  MLflow UI: mlflow ui --backend-store-uri {args.tracking_uri}")


if __name__ == "__main__":
    main()
