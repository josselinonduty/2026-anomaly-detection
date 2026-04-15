"""AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2.

Reference
---------
Damm, S., Laszkiewicz, M., Lederer, J., & Fischer, A. (2025).
"AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2."
WACV 2025 (Oral).  arXiv:2405.14529

This implementation follows the paper exactly:
- Backbone: DINOv2 (frozen), using ViT-S/14 by default.
- Patch-level deep nearest-neighbour anomaly detection (Section 3.1).
- Cosine distance between test patches and the nominal memory bank M (Eq. 3).
- Image-level score: mean of the top-1 % patch distances (Eq. 4, Section 3.1).
- Pixel-level anomaly maps via bilinear upsampling + Gaussian smoothing (σ=4).
- Zero-shot PCA-based foreground masking using DINOv2 features (Section 3.2).
- Reference-image augmentation via 8-angle rotation (Section 3.2).
- Training-free: memory bank is built from normal reference images only.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from torchvision import transforms


class AnomalyDINO(nn.Module):
    """AnomalyDINO anomaly detection model.

    Parameters
    ----------
    model_name : str
        DINOv2 model variant. One of ``'dinov2_vits14'``, ``'dinov2_vitb14'``,
        ``'dinov2_vitl14'``, ``'dinov2_vitg14'``.
    smaller_edge_size : int
        Images are resized so the smaller edge equals this value.
        Default 448; the paper also evaluates 672.
    masking : bool
        Whether to apply PCA-based zero-shot foreground masking (Section 3.2).
    rotation : bool
        Whether to augment reference images with 8 rotations (Section 3.2).
    masking_threshold : float
        Threshold on the first PCA component for foreground/background
        separation.  Default 10 (as in the reference implementation).
    gaussian_sigma : float
        Standard deviation for Gaussian smoothing of the anomaly map.
        Default 4.0 (Section 3.1, following PatchCore).
    top_percent : float
        Fraction of highest patch distances used for image-level scoring.
        Default 0.01 (top 1 %, Section 3.1).
    rotation_angles : tuple of int
        Rotation angles for reference augmentation.
        Default ``(0, 45, 90, 135, 180, 225, 270, 315)``.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        smaller_edge_size: int = 448,
        masking: bool = True,
        rotation: bool = True,
        masking_threshold: float = 10.0,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        rotation_angles: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.masking = masking
        self.rotation = rotation
        self.masking_threshold = masking_threshold
        self.gaussian_sigma = gaussian_sigma
        self.top_percent = top_percent
        self.rotation_angles = rotation_angles

        # ── DINOv2 backbone (frozen) ─────────────────────────────────
        self.backbone: nn.Module = torch.hub.load(
            "facebookresearch/dinov2", model_name
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.patch_size: int = self.backbone.patch_size  # 14

        # ── Image preprocessing (DINOv2 standard) ────────────────────
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=smaller_edge_size,
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

        # ── Memory bank (populated by ``fit``) ───────────────────────
        self._memory_bank: np.ndarray | None = None
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
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract patch-level features from a preprocessed image tensor.

        Parameters
        ----------
        image_tensor : (C, H, W) or (1, C, H, W)

        Returns
        -------
        features : (N_patches, D)   numpy array
        """
        device = next(self.backbone.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        tokens = self.backbone.get_intermediate_layers(image_tensor)[0].squeeze()
        return tokens.cpu().numpy()

    # ------------------------------------------------------------------
    # PCA-based foreground masking  (Section 3.2)
    # ------------------------------------------------------------------

    def compute_foreground_mask(
        self,
        features: np.ndarray,
        grid_size: tuple[int, int],
        threshold: float | None = None,
        border: float = 0.2,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Compute a binary foreground mask via PCA thresholding.

        The first principal component of the DINOv2 patch features typically
        separates foreground from background.  A centre-crop heuristic (the
        "masking test" in Figure 2) determines whether the mask should be
        inverted.

        Parameters
        ----------
        features : (N, D)
        grid_size : (H_grid, W_grid)
        threshold : float or None
            PCA threshold; defaults to ``self.masking_threshold``.
        border : float
            Fraction of border to ignore in the centre-crop heuristic.
        kernel_size : int
            Kernel for dilation and morphological closing.

        Returns
        -------
        mask : (N,)  boolean array — True = foreground.
        """
        if threshold is None:
            threshold = self.masking_threshold

        pca = PCA(n_components=1, svd_solver="randomized")
        first_pc = pca.fit_transform(features.astype(np.float32))

        mask = first_pc.squeeze() > threshold

        # Centre-crop heuristic: if centre is mostly masked out, invert.
        mask_2d = mask.reshape(grid_size)
        r0 = int(grid_size[0] * border)
        r1 = int(grid_size[0] * (1 - border))
        c0 = int(grid_size[1] * border)
        c1 = int(grid_size[1] * (1 - border))
        centre = mask_2d[r0:r1, c0:c1]
        if centre.sum() <= centre.size * 0.35:
            mask = ~mask

        # Morphological post-processing: dilation + closing.
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        return mask_uint8.astype(bool).ravel()

    # ------------------------------------------------------------------
    # Reference augmentation  (Section 3.2)
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate an image by *angle* degrees around its centre."""
        centre = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(
            image,
            rot_mat,
            image.shape[1::-1],
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_DEFAULT,
        )

    def augment_reference(self, image: np.ndarray) -> list[np.ndarray]:
        """Augment a reference image with rotations.

        Returns a list of rotated images (including the original at 0°).
        """
        return [self.rotate_image(image, angle) for angle in self.rotation_angles]

    # ------------------------------------------------------------------
    # Memory-bank construction  (Section 3.1, Eq. 1)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, images: list[np.ndarray | str | Image.Image]) -> None:
        """Build the nominal memory bank M from reference images.

        Parameters
        ----------
        images : list
            Normal reference images (numpy HWC-RGB, file paths, or PIL).
        """
        all_features: list[np.ndarray] = []

        for img in images:
            # Convert to numpy RGB if needed.
            if isinstance(img, str):
                raw = cv2.cvtColor(
                    cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
            elif isinstance(img, Image.Image):
                raw = np.asarray(img.convert("RGB"))
            else:
                raw = img

            # Optionally augment with rotations.
            variants = self.augment_reference(raw) if self.rotation else [raw]

            for variant in variants:
                img_tensor, grid_size = self.prepare_image(variant)
                features = self.extract_features(img_tensor)

                # Optionally mask reference images (default: no masking on
                # reference, only on test — consistent with the paper).
                all_features.append(features)

        self._memory_bank = np.concatenate(all_features, axis=0).astype(np.float32)

        # Normalise for cosine distance (L2-normalised dot product).
        norms = np.linalg.norm(self._memory_bank, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self._memory_bank = self._memory_bank / norms

        self._fitted = True

    # ------------------------------------------------------------------
    # Anomaly scoring  (Section 3.1, Eqs. 2–4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | str | Image.Image,
    ) -> tuple[float, np.ndarray]:
        """Score a single test image.

        Returns
        -------
        score : float
            Image-level anomaly score (mean of top-1 % patch distances).
        anomaly_map : (H_img, W_img)  numpy array
            Pixel-level anomaly scores (bilinear-upsampled + Gaussian-smoothed).
        """
        assert self._fitted, "Call fit() before predict()."

        # Get original image dimensions.
        if isinstance(image, str):
            raw = cv2.cvtColor(
                cv2.imread(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        elif isinstance(image, Image.Image):
            raw = np.asarray(image.convert("RGB"))
        else:
            raw = image

        img_h, img_w = raw.shape[:2]

        img_tensor, grid_size = self.prepare_image(raw)
        features = self.extract_features(img_tensor)  # (N, D)

        # Foreground masking on test image.
        if self.masking:
            mask = self.compute_foreground_mask(features, grid_size)
        else:
            mask = np.ones(features.shape[0], dtype=bool)

        # L2-normalise test features for cosine distance.
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        features_normed = features / norms

        # Nearest-neighbour cosine distance (Eq. 2–3).
        # cosine_distance = 1 - cosine_similarity
        # With L2-normalised vectors: ||a - b||^2 / 2 = 1 - a·b = cosine_dist
        # FAISS L2 on normalised vectors gives ||a-b||^2, so divide by 2.
        masked_features = features_normed[mask]
        max_sim = self._chunked_nn_numpy(masked_features, self._memory_bank)
        distances = 1.0 - max_sim  # cosine distance

        # Place distances back into the full grid.
        output_distances = np.zeros(features.shape[0], dtype=np.float64)
        output_distances[mask] = distances

        # Image-level score: mean of top 1 % patch distances (Eq. 4).
        score = self._mean_top_percent(output_distances)

        # Pixel-level anomaly map: reshape → bilinear upsample → Gaussian smooth.
        distance_map = output_distances.reshape(grid_size)
        anomaly_map = cv2.resize(
            distance_map.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
        anomaly_map = gaussian_filter(anomaly_map, sigma=self.gaussian_sigma)

        return float(score), anomaly_map

    # ------------------------------------------------------------------
    # Chunked nearest-neighbour search
    # ------------------------------------------------------------------

    @staticmethod
    def _chunked_nn_numpy(
        queries: np.ndarray,
        bank: np.ndarray,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        """Chunked cosine NN on CPU (numpy).

        Splits the memory bank into chunks to keep intermediate matrices
        within reasonable size limits.
        """
        max_sim = np.full(queries.shape[0], -np.inf, dtype=np.float32)
        for start in range(0, bank.shape[0], chunk_size):
            chunk = bank[start : start + chunk_size]
            sims = queries @ chunk.T  # (M, chunk)
            chunk_max = sims.max(axis=1)
            np.maximum(max_sim, chunk_max, out=max_sim)
        return max_sim

    @staticmethod
    def _chunked_nn_torch(
        queries: torch.Tensor,
        bank: torch.Tensor,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """Chunked cosine NN on device (torch).

        Splits the memory bank into chunks to avoid exceeding device
        memory or integer-index limits (e.g. MPS INT_MAX).
        """
        max_sim = torch.full(
            (queries.shape[0],), -float("inf"), device=queries.device
        )
        for start in range(0, bank.shape[0], chunk_size):
            chunk = bank[start : start + chunk_size]
            sims = queries @ chunk.T
            chunk_max, _ = sims.max(dim=1)
            max_sim = torch.maximum(max_sim, chunk_max)
        return max_sim

    def _mean_top_percent(self, distances: np.ndarray) -> float:
        """Compute the mean of the top ``self.top_percent`` distances.

        This is the empirical tail value-at-risk for the 99 % quantile
        (Section 3.1).
        """
        flat = distances.ravel()
        k = max(1, int(len(flat) * self.top_percent))
        # Partial sort is faster than full sort for large arrays.
        top_k_indices = np.argpartition(flat, -k)[-k:]
        return float(np.mean(flat[top_k_indices]))

    # ------------------------------------------------------------------
    # Batch predict (convenience for Lightning module)
    # ------------------------------------------------------------------

    def predict_batch_tensor(
        self,
        images: torch.Tensor,
        original_sizes: list[tuple[int, int]] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of image tensors (already preprocessed by DINOv2 transform).

        Parameters
        ----------
        images : (B, C, H, W)  already normalised tensors.
        original_sizes : per-image (H, W) for anomaly-map upsampling.
            If None, anomaly maps are returned at ``output_size``.
        output_size : (H, W) fallback for anomaly-map spatial resolution.

        Returns
        -------
        scores : (B,) tensor — image-level anomaly scores.
        anomaly_maps : (B, 1, H_out, W_out) tensor — pixel-level scores.
        """
        assert self._fitted, "Call fit() before predict_batch_tensor()."

        device = images.device
        B = images.shape[0]

        # Ensure images have correct crop (multiple of patch_size).
        _, _, h, w = images.shape
        cropped_h = h - h % self.patch_size
        cropped_w = w - w % self.patch_size
        images = images[:, :, :cropped_h, :cropped_w]
        grid_h = cropped_h // self.patch_size
        grid_w = cropped_w // self.patch_size
        grid_size = (grid_h, grid_w)

        scores_list: list[float] = []
        maps_list: list[torch.Tensor] = []

        memory_bank_t = torch.from_numpy(self._memory_bank).to(device)

        for i in range(B):
            img_tensor = images[i].unsqueeze(0)
            tokens = self.backbone.get_intermediate_layers(img_tensor)[0].squeeze()
            features_np = tokens.cpu().numpy()

            # Masking.
            if self.masking:
                mask = self.compute_foreground_mask(features_np, grid_size)
            else:
                mask = np.ones(features_np.shape[0], dtype=bool)

            # L2-normalise.
            features_t = tokens  # (N, D) on device
            features_t = F.normalize(features_t, p=2, dim=1)

            masked_features = features_t[torch.from_numpy(mask).to(device)]

            # Cosine NN distances (chunked to avoid exceeding device limits).
            max_sim = self._chunked_nn_torch(masked_features, memory_bank_t)
            dists = 1.0 - max_sim

            # Place back.
            output_dists = torch.zeros(features_t.shape[0], device=device)
            mask_t = torch.from_numpy(mask).to(device)
            output_dists[mask_t] = dists.float()

            # Image-level score.
            flat = output_dists
            k = max(1, int(len(flat) * self.top_percent))
            topk_vals, _ = torch.topk(flat, k)
            score = topk_vals.mean().item()
            scores_list.append(score)

            # Pixel-level anomaly map.
            dist_map = output_dists.reshape(1, 1, grid_h, grid_w)
            if original_sizes is not None:
                oh, ow = original_sizes[i]
            elif output_size is not None:
                oh, ow = output_size
            else:
                oh, ow = cropped_h, cropped_w

            amap = F.interpolate(
                dist_map.float(), size=(oh, ow), mode="bilinear", align_corners=False
            )
            # Gaussian smoothing via conv2d approximation.
            amap_np = amap.squeeze().cpu().numpy()
            amap_np = gaussian_filter(amap_np, sigma=self.gaussian_sigma)
            amap = torch.from_numpy(amap_np).unsqueeze(0).unsqueeze(0)
            maps_list.append(amap)

        scores = torch.tensor(scores_list, device=device)
        anomaly_maps = torch.cat(maps_list, dim=0).to(device)

        return scores, anomaly_maps
