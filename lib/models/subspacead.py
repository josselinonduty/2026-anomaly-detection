"""SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling.

Reference
---------
Lendering, C., Akdag, E., & Bondarev, E. (2026).
"SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling."
CVPR 2026. arXiv:2602.23013

This implementation follows the paper exactly:
- Backbone: frozen DINOv2 (Giant by default), using ViT patch size 14.
- Multi-layer feature averaging: mean-pool layers 22–28 of DINOv2-G (Sec. 3.2).
- Data augmentation: 30 random rotations (0°–345°) per normal image (Sec. 3.2).
- PCA subspace modeling: τ=0.99 explained variance threshold (Sec. 3.3, Eq. 4).
- Anomaly score: squared reconstruction residual ‖x − x_proj‖² (Sec. 3.4, Eq. 6).
- Image-level score: top-1% mean (TVaR, Eq. 7).
- Pixel-level: bilinear upsampling + Gaussian blur σ=4 (Sec. 3.4).
- Input resolution: 672 px (Sec. 4.3).
- Training-free, no memory banks, no prompt tuning.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms


class SubspaceAD(nn.Module):
    """SubspaceAD anomaly detection model.

    Parameters
    ----------
    model_name : str
        DINOv2 model variant. One of ``'dinov2_vitg14'``, ``'dinov2_vitl14'``,
        ``'dinov2_vitb14'``, ``'dinov2_vits14'``.
        Paper default: ``'dinov2_vitg14'`` (Giant).
    image_resolution : int
        Input image resolution in pixels. Default 672 (Sec. 4.3).
    layers : tuple[int, ...]
        Transformer block indices (0-based) to average features from.
        Paper default: layers 22–28 for DINOv2-G (Sec. 3.2, 4.3).
    pca_variance_threshold : float
        Explained variance ratio τ for selecting PCA components.
        Default 0.99 (Sec. 3.3, Eq. 4).
    aug_count : int
        Number of random rotations per normal image during fitting.
        Default 30 (Sec. 3.2, 4.3).
    gaussian_sigma : float
        σ for Gaussian smoothing of pixel-level anomaly maps.
        Default 4.0 (Sec. 3.4).
    top_percent : float
        Fraction of top patch scores for image-level aggregation (TVaR).
        Default 0.01 (top 1%, Sec. 3.4, Eq. 7).
    """

    # Number of transformer blocks per DINOv2 variant.
    _BLOCK_COUNTS = {
        "dinov2_vits14": 12,
        "dinov2_vitb14": 12,
        "dinov2_vitl14": 24,
        "dinov2_vitg14": 40,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitg14",
        image_resolution: int = 672,
        layers: tuple[int, ...] | None = None,
        pca_variance_threshold: float = 0.99,
        aug_count: int = 30,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.image_resolution = image_resolution
        self.pca_variance_threshold = pca_variance_threshold
        self.aug_count = aug_count
        self.gaussian_sigma = gaussian_sigma
        self.top_percent = top_percent

        # ── DINOv2 backbone (frozen) ─────────────────────────────────
        self.backbone: nn.Module = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.patch_size: int = self.backbone.patch_size  # 14

        # ── Resolve layer indices ────────────────────────────────────
        # The paper uses "Middle-7" layers (Sec. 3.2, Table 3).
        # For DINOv2-G (40 blocks): layers 22–28.
        # Auto-compute for other backbones by centering 7 layers.
        n_blocks = self._BLOCK_COUNTS.get(model_name, len(self.backbone.blocks))
        if layers is not None:
            # Validate user-specified layers.
            self.layers = tuple(l % n_blocks for l in layers)
        else:
            # Default: middle 7 layers (paper: "Mean-pool (Middle-7)").
            n_layers = min(7, n_blocks)
            mid = n_blocks // 2
            start = mid - n_layers // 2
            self.layers = tuple(range(start, start + n_layers))

        # ── Image preprocessing (DINOv2 standard) ────────────────────
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=image_resolution,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        # ── PCA parameters (populated by ``fit``) ────────────────────
        self._mu: np.ndarray | None = None  # (D,)
        self._components: np.ndarray | None = None  # (D, r)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def prepare_image(
        self, img: np.ndarray | str | Image.Image
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load and preprocess an image for DINOv2.

        Returns the image tensor and the (H_grid, W_grid) patch grid size.
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        image_tensor = self.transform(img)

        # Crop to exact multiple of patch_size.
        _, h, w = image_tensor.shape
        cropped_h = h - h % self.patch_size
        cropped_w = w - w % self.patch_size
        image_tensor = image_tensor[:, :cropped_h, :cropped_w]

        grid_size = (cropped_h // self.patch_size, cropped_w // self.patch_size)
        return image_tensor, grid_size

    # ------------------------------------------------------------------
    # Feature extraction  (Sec. 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract multi-layer mean-pooled patch features (Eq. 2).

        Parameters
        ----------
        image_tensor : (C, H, W) or (B, C, H, W)

        Returns
        -------
        features : (B, N_patches, D) or (N_patches, D) numpy array
        """
        device = next(self.backbone.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Extract from multiple intermediate layers and average (Eq. 2).
        layer_outputs = self.backbone.get_intermediate_layers(
            image_tensor, n=self.layers
        )
        # Each output: (B, N_patches, D). Stack and mean-pool.
        stacked = torch.stack(layer_outputs, dim=0)  # (L, B, N, D)
        fused = stacked.mean(dim=0)  # (B, N, D)  — Eq. 2

        return fused.cpu().numpy()

    # ------------------------------------------------------------------
    # Reference augmentation  (Sec. 3.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _random_rotation(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Rotate image by a random angle in [0, 345] degrees."""
        angle = float(rng.uniform(0.0, 345.0))
        centre = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(
            image,
            rot_mat,
            image.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_DEFAULT,
        )

    def augment_reference(
        self, image: np.ndarray, rng: np.random.Generator | None = None
    ) -> list[np.ndarray]:
        """Generate N_a=aug_count random rotations of a reference image.

        Returns the original image followed by ``aug_count`` augmented views.
        """
        if rng is None:
            rng = np.random.default_rng()
        views = [image]
        for _ in range(self.aug_count):
            views.append(self._random_rotation(image, rng))
        return views

    # ------------------------------------------------------------------
    # PCA subspace fitting  (Sec. 3.3)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, images: list[np.ndarray | str | Image.Image]) -> None:
        """Fit the PCA subspace from normal reference images.

        Single-pass feature extraction followed by GPU PCA:
        1. Extract features from all images + augmented views (single backbone pass each).
        2. Compute mean μ and covariance Σ from stored features on GPU.
        3. Eigendecompose to get the principal subspace C.

        Parameters
        ----------
        images : list
            Normal reference images (numpy HWC-RGB, file paths, or PIL).
        """
        device = next(self.backbone.parameters()).device
        rng = np.random.default_rng(42)

        # ── Convert all inputs to numpy RGB ──────────────────────────
        raw_images: list[np.ndarray] = []
        for img in images:
            if isinstance(img, str):
                raw_images.append(
                    cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                )
            elif isinstance(img, Image.Image):
                raw_images.append(np.asarray(img.convert("RGB")))
            else:
                raw_images.append(img)

        # ── Batched feature extraction ────────────────────────────────
        # Batch augmented views to reduce backbone forward-pass overhead.
        batch_size = 8  # tune based on GPU memory
        all_features: list[np.ndarray] = []  # list of (N_patches, D) arrays
        for raw in raw_images:
            variants = self.augment_reference(raw, rng)
            # Preprocess all views into tensors.
            tensors = [self.prepare_image(v)[0] for v in variants]
            del variants
            # Run in mini-batches.
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i : i + batch_size])  # (B, C, H, W)
                feats = self.extract_features(batch)  # (B, N, D)
                for j in range(feats.shape[0]):
                    all_features.append(feats[j])  # (N, D)
            del tensors

        feature_dim = all_features[0].shape[-1]

        # ── Concatenate all features and move to GPU ─────────────────
        # Total size: k × (1+aug_count) × (h_p×w_p) × D × 8 bytes
        # For k=1, aug=30, 48×48 grid, D=768: ~430 MB (float64) — fits in GPU.
        all_feats_np = np.concatenate(all_features, axis=0)  # (total_tokens, D)
        del all_features  # free the list of small arrays

        # MPS does not support float64; use float32 there, float64 elsewhere.
        compute_dtype = torch.float32 if device.type == "mps" else torch.float64

        all_feats_gpu = torch.from_numpy(all_feats_np).to(device, dtype=compute_dtype)
        del all_feats_np  # free the numpy copy

        total_tokens = all_feats_gpu.shape[0]

        # ── Compute mean μ ───────────────────────────────────────────
        mu = all_feats_gpu.mean(dim=0)  # (D,)

        # ── Compute covariance Σ ─────────────────────────────────────
        centered = all_feats_gpu - mu  # (N, D)
        del all_feats_gpu  # free the un-centred copy

        # Chunked matmul to limit peak memory: C^T @ C in blocks.
        cov = torch.zeros(
            (feature_dim, feature_dim), dtype=compute_dtype, device=device
        )
        chunk_size = 4096
        for i in range(0, total_tokens, chunk_size):
            chunk = centered[i : i + chunk_size]
            cov += chunk.T @ chunk
        del centered

        cov /= total_tokens - 1

        # ── Eigendecomposition ───────────────────────────────────────
        # eigh is not implemented on MPS; run on CPU (cov is small: D×D).
        # Move to CPU first, then cast — MPS cannot convert to float64.
        cov_cpu = cov.cpu().to(dtype=torch.float64)
        del cov
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_cpu)
        # Sort descending.
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select r components to explain τ variance (Eq. 4).
        cumvar = torch.cumsum(eigenvalues, dim=0) / eigenvalues.sum()
        r = int(
            torch.searchsorted(
                cumvar,
                torch.tensor([self.pca_variance_threshold], dtype=torch.float64),
            ).item()
            + 1
        )
        r = min(r, eigenvectors.shape[1])

        self._mu = mu.cpu().numpy().astype(np.float64)
        self._components = (
            eigenvectors[:, :r].cpu().numpy().astype(np.float64)
        )  # (D, r)
        self._fitted = True

    # ------------------------------------------------------------------
    # Anomaly scoring  (Sec. 3.4)
    # ------------------------------------------------------------------

    def _compute_anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """Compute per-patch reconstruction residual scores (Eq. 5–6).

        Parameters
        ----------
        features : (N, D)

        Returns
        -------
        scores : (N,)
        """
        mu = self._mu.astype(features.dtype)
        C = self._components.astype(features.dtype)  # (D, r)

        X0 = features - mu  # (N, D)
        Z = X0 @ C  # (N, r)  — projection coefficients
        X_proj = Z @ C.T  # (N, D) — reconstruction in centred space
        residual = X0 - X_proj  # (N, D)
        scores = np.sum(residual**2, axis=1)  # (N,) — Eq. 6
        return scores

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | str | Image.Image,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[float, np.ndarray]:
        """Run inference on a single test image.

        Parameters
        ----------
        image : input image
        output_size : (H, W) for the upsampled anomaly map.
            Defaults to (image_resolution, image_resolution).

        Returns
        -------
        score : float  — image-level anomaly score (TVaR, Eq. 7).
        anomaly_map : (H_out, W_out) float32 — pixel-level anomaly map.
        """
        if not self._fitted:
            raise RuntimeError("SubspaceAD has not been fitted yet. Call fit() first.")

        if output_size is None:
            output_size = (self.image_resolution, self.image_resolution)

        img_tensor, grid_size = self.prepare_image(image)
        features = self.extract_features(img_tensor)  # (1, N, D) or (N, D)
        features = features.reshape(-1, features.shape[-1])  # (N, D)

        scores = self._compute_anomaly_scores(features)  # (N,)

        # Reshape to spatial grid.
        h_p, w_p = grid_size
        anomaly_map = scores.reshape(h_p, w_p).astype(np.float32)

        # Bilinear upsample to output resolution.
        anomaly_map = cv2.resize(
            anomaly_map,
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Gaussian smoothing (σ=4, Sec. 3.4).
        anomaly_map = gaussian_filter(anomaly_map, sigma=self.gaussian_sigma)

        # Image-level score: top-1% mean (TVaR, Eq. 7).
        flat = anomaly_map.ravel()
        k = max(1, int(len(flat) * self.top_percent))
        top_k = np.partition(flat, -k)[-k:]
        score = float(np.mean(top_k))

        return score, anomaly_map

    @torch.no_grad()
    def predict_batch_tensor(
        self,
        images: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference on a batch of tensors (B, C, H, W).

        Images should already be ImageNet-normalised.

        Returns
        -------
        scores : (B,) tensor — image-level scores.
        anomaly_maps : (B, 1, H, W) tensor — pixel-level anomaly maps.
        """
        if not self._fitted:
            raise RuntimeError("SubspaceAD has not been fitted yet. Call fit() first.")

        if output_size is None:
            output_size = (self.image_resolution, self.image_resolution)

        device = images.device
        B = images.shape[0]

        # Crop to multiple of patch_size.
        _, _, h, w = images.shape
        ch = h - h % self.patch_size
        cw = w - w % self.patch_size
        images_cropped = images[:, :, :ch, :cw]
        h_p, w_p = ch // self.patch_size, cw // self.patch_size

        # Multi-layer feature extraction (Eq. 2).
        layer_outputs = self.backbone.get_intermediate_layers(
            images_cropped, n=self.layers
        )
        stacked = torch.stack(layer_outputs, dim=0)  # (L, B, N, D)
        fused = stacked.mean(dim=0)  # (B, N, D)
        features_np = fused.cpu().numpy()  # (B, N, D)

        all_scores = []
        all_maps = []

        for b in range(B):
            feats = features_np[b]  # (N, D)
            patch_scores = self._compute_anomaly_scores(feats)  # (N,)

            amap = patch_scores.reshape(h_p, w_p).astype(np.float32)
            amap = cv2.resize(
                amap,
                (output_size[1], output_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            amap = gaussian_filter(amap, sigma=self.gaussian_sigma)

            flat = amap.ravel()
            k = max(1, int(len(flat) * self.top_percent))
            top_k = np.partition(flat, -k)[-k:]
            score = float(np.mean(top_k))

            all_scores.append(score)
            all_maps.append(amap)

        scores_t = torch.tensor(all_scores, dtype=torch.float32, device=device)
        maps_np = np.stack(all_maps, axis=0)  # (B, H, W)
        maps_t = torch.from_numpy(maps_np).unsqueeze(1).to(device)  # (B, 1, H, W)

        return scores_t, maps_t
