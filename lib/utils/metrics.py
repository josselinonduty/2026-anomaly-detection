"""Evaluation metrics for anomaly detection."""

from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score


def compute_auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """Image-level AUROC."""
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()
    if len(set(labels_np)) < 2:
        return 0.0
    return float(roc_auc_score(labels_np, scores_np))


def compute_pixel_auroc(masks: torch.Tensor, anomaly_maps: torch.Tensor) -> float:
    """Pixel-level AUROC (flattened)."""
    masks_np = masks.cpu().numpy().ravel()
    maps_np = anomaly_maps.cpu().numpy().ravel()
    binary_masks = (masks_np > 0.5).astype(int)
    if len(set(binary_masks)) < 2:
        return 0.0
    return float(roc_auc_score(binary_masks, maps_np))
