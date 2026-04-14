"""PatchCore: Towards Total Recall in Industrial Anomaly Detection.

Reference
---------
Roth, K., Pemula, L., Schiber, J., Lucchi, A., & Brox, T. (2022).
"Towards Total Recall in Industrial Anomaly Detection." CVPR 2022.

This implementation follows the paper exactly:
- Backbone: WideResNet-50-2 pre-trained on ImageNet (frozen).
- Intermediate feature extraction from layers 2 and 3.
- Locally aware patch features via average pooling (neighbourhood size p=3).
- Spatial alignment to the deepest extracted layer via adaptive average pooling.
- Coreset subsampling via greedy k-center approximation with random projection
  for efficiency (Section 3.2, Appendix).
- Anomaly scoring via nearest-neighbour distance in the coreset memory bank.
- Image-level re-weighting using softmax over local coreset density (Eq. 2).
- Gaussian-smoothed pixel-level anomaly maps upsampled to input resolution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class PatchCore(nn.Module):
    """PatchCore anomaly detection model.

    Parameters
    ----------
    layers_to_extract : tuple of int
        Which ResNet layer blocks to extract features from.
        Default ``(2, 3)`` as specified in the paper.
    neighbourhood_size : int
        Kernel size for local aware feature aggregation (parameter *p*).
        Default 3.
    coreset_sampling_ratio : float
        Fraction of patch features retained via coreset subsampling.
        The paper reports results for 1 % (``0.01``), 10 % and 25 %.
    num_neighbors : int
        Number of nearest coreset neighbours used in the re-weighting
        score (parameter *b* in Eq. 2).  Default 9.
    image_size : int
        Expected square input spatial resolution.
    projection_dim : int
        Target dimensionality for the random projection used to speed up the
        greedy coreset selection (Johnson–Lindenstrauss).  Set to 0 to disable.
    """

    def __init__(
        self,
        layers_to_extract: tuple[int, ...] = (2, 3),
        neighbourhood_size: int = 3,
        coreset_sampling_ratio: float = 0.01,
        num_neighbors: int = 9,
        image_size: int = 256,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.layers_to_extract = layers_to_extract
        self.neighbourhood_size = neighbourhood_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.image_size = image_size
        self.projection_dim = projection_dim

        # ── Backbone (WideResNet-50-2, ImageNet pre-trained) ─────────
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)

        self.stage0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stage1 = backbone.layer1  # 256 ch, stride  4
        self.stage2 = backbone.layer2  # 512 ch, stride  8
        self.stage3 = backbone.layer3  # 1024 ch, stride 16
        self.stage4 = backbone.layer4  # 2048 ch, stride 32

        # Freeze the entire backbone – PatchCore never trains it.
        for param in self.parameters():
            param.requires_grad = False

        # ── Local neighbourhood aggregation ──────────────────────────
        padding = neighbourhood_size // 2
        self.avg_pool = nn.AvgPool2d(
            kernel_size=neighbourhood_size,
            stride=1,
            padding=padding,
        )

        # ── Memory bank (populated by ``fit``) ───────────────────────
        self.register_buffer("memory_bank", torch.empty(0))
        self._fitted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _STAGES: dict[int, str] = {1: "stage1", 2: "stage2", 3: "stage3", 4: "stage4"}

    def _get_stage(self, idx: int) -> nn.Module:
        return getattr(self, self._STAGES[idx])

    # ------------------------------------------------------------------
    # Feature extraction  (Section 3.1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Extract locally-aware patch features.

        Returns
        -------
        features : (B·H'·W', C)
            Flattened patch-feature collection.
        shape : (B, H', W')
            Spatial dimensions of the feature map.
        """
        h = self.stage0(x)

        layer_features: dict[int, torch.Tensor] = {}
        max_layer = max(self.layers_to_extract)
        for i in range(1, max_layer + 1):
            h = self._get_stage(i)(h)
            if i in self.layers_to_extract:
                layer_features[i] = self.avg_pool(h)

        # Spatial alignment to the deepest extracted layer (paper §3.1).
        target_size = layer_features[max_layer].shape[2:]
        aligned: list[torch.Tensor] = []
        for i in sorted(self.layers_to_extract):
            feat = layer_features[i]
            if feat.shape[2:] != target_size:
                feat = F.adaptive_avg_pool2d(feat, target_size)
            aligned.append(feat)

        features = torch.cat(aligned, dim=1)  # (B, C_total, H', W')
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        return features, (B, H, W)

    # ------------------------------------------------------------------
    # Memory-bank construction  (Section 3.2)
    # ------------------------------------------------------------------

    def fit(self, dataloader: torch.utils.data.DataLoader) -> None:
        """Build the coreset memory bank from *normal* training data."""
        self.eval()
        device = next(self.buffers(), torch.tensor(0)).device

        features_list: list[torch.Tensor] = []
        for batch in dataloader:
            x = batch["image"].to(device)
            feats, _ = self.extract_features(x)
            features_list.append(feats.cpu())

        all_features = torch.cat(features_list, dim=0)  # (N, C)

        if self.coreset_sampling_ratio < 1.0:
            n_select = max(
                1,
                int(all_features.shape[0] * self.coreset_sampling_ratio),
            )
            indices = self._greedy_coreset_sampling(
                all_features,
                n_select,
                self.projection_dim,
            )
            self.memory_bank = all_features[indices].to(device)
        else:
            self.memory_bank = all_features.to(device)

        self._fitted = True

    @staticmethod
    def _greedy_coreset_sampling(
        features: torch.Tensor,
        n_select: int,
        projection_dim: int = 128,
    ) -> torch.Tensor:
        """Greedy k-center coreset selection (minimax facility location).

        Optionally projects features to a lower dimension via a random
        Gaussian matrix to speed up distance computations (Appendix A.3).

        Parameters
        ----------
        features : (N, C)   on CPU
        n_select : int
        projection_dim : int   set ≤ 0 to disable.

        Returns
        -------
        indices : (n_select,)  LongTensor
        """
        N, C = features.shape

        # Optional random projection for faster selection.
        if 0 < projection_dim < C:
            rng = torch.Generator().manual_seed(0)
            projector = torch.randn(C, projection_dim, generator=rng) / (
                projection_dim**0.5
            )
            projected = features @ projector
        else:
            projected = features

        # Start from a random point.
        first = torch.randint(0, N, (1,)).item()
        selected = [first]

        # Squared-L2 distances to the nearest selected point so far.
        diff = projected - projected[first]
        min_sq_dists = (diff * diff).sum(dim=1)  # (N,)

        for _ in range(1, n_select):
            idx = torch.argmax(min_sq_dists).item()
            selected.append(idx)
            diff = projected - projected[idx]
            new_sq_dists = (diff * diff).sum(dim=1)
            min_sq_dists = torch.minimum(min_sq_dists, new_sq_dists)

        return torch.tensor(selected, dtype=torch.long)

    # ------------------------------------------------------------------
    # Anomaly scoring  (Section 3.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute image-level scores and pixel-level anomaly maps.

        Returns
        -------
        scores : (B,)
        anomaly_maps : (B, image_size, image_size)
        """
        assert self._fitted, "Call fit() before predict()."

        features, (B, H, W) = self.extract_features(x)

        # L2 distances to every coreset member.
        distances = torch.cdist(features, self.memory_bank)  # (B·H·W, M)
        nn_dists, nn_indices = distances.min(dim=1)  # (B·H·W,)

        # ── Pixel-level anomaly map ──────────────────────────────────
        patch_scores = nn_dists.reshape(B, H, W)
        anomaly_map = F.interpolate(
            patch_scores.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        anomaly_map = self._apply_gaussian_smoothing(anomaly_map, sigma=4.0)

        # ── Image-level score with re-weighting (Eq. 2) ─────────────
        scores = self._compute_image_scores(
            patch_scores.reshape(B, -1),
            nn_indices.reshape(B, -1),
            nn_dists.reshape(B, -1),
        )

        return scores, anomaly_map

    def _compute_image_scores(
        self,
        patch_scores: torch.Tensor,  # (B, H*W)
        nn_indices: torch.Tensor,  # (B, H*W)
        nn_dists: torch.Tensor,  # (B, H*W)
    ) -> torch.Tensor:
        """Re-weighted image-level anomaly score (Eq. 2).

        For each image the most anomalous test patch p* is identified.  Its
        nearest coreset member m* is looked up and assessed for local density
        via its *b* nearest coreset neighbours.  A softmax over those distances
        produces a weight *w* that up-weights patches near sparse (hence more
        anomalous) coreset regions.
        """
        B = patch_scores.shape[0]
        max_idx = patch_scores.argmax(dim=1)  # (B,)

        scores: list[torch.Tensor] = []
        for b in range(B):
            s_star = patch_scores[b, max_idx[b]]
            m_star_idx = nn_indices[b, max_idx[b]]

            # Distances from m* to all coreset members.
            m_star = self.memory_bank[m_star_idx]
            m_dists = torch.cdist(
                m_star.unsqueeze(0),
                self.memory_bank,
            ).squeeze(
                0
            )  # (M,)

            # b nearest neighbours of m* (index 0 is m* itself with dist ≈ 0).
            nb = min(self.num_neighbors + 1, m_dists.shape[0])
            topk_dists, _ = m_dists.topk(nb, largest=False)
            nb_dists = topk_dists[1:]  # exclude self

            # Softmax re-weighting.
            w = 1.0 - (torch.exp(nb_dists[0]) / torch.exp(nb_dists).sum())
            scores.append(w * s_star)

        return torch.stack(scores)

    # ------------------------------------------------------------------
    # Gaussian smoothing of anomaly maps
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_gaussian_smoothing(
        anomaly_map: torch.Tensor,
        sigma: float = 4.0,
    ) -> torch.Tensor:
        """Apply 2-D Gaussian blur to each anomaly map in the batch."""
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)

        x = torch.arange(
            kernel_size,
            dtype=anomaly_map.dtype,
            device=anomaly_map.device,
        )
        x = x - kernel_size // 2
        gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d = gauss_1d / gauss_1d.sum()

        kernel_2d = gauss_1d[:, None] * gauss_1d[None, :]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

        padding = kernel_size // 2
        return F.conv2d(
            anomaly_map.unsqueeze(1),
            kernel_2d,
            padding=padding,
        ).squeeze(1)

    # ------------------------------------------------------------------
    # nn.Module interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Alias for :meth:`predict`."""
        return self.predict(x)
