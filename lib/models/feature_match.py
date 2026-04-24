"""Classical feature-matching anomaly detection using OpenCV descriptors.

Pipeline (classical industrial inspection):

1. **fit** — store normal reference images and precompute their
   keypoints / descriptors (SIFT or ORB).
2. **predict** — for each test image:
   a. Match descriptors against every reference → pick the best
      reference (most RANSAC inliers).
   b. Compute a homography and warp the best reference onto the test
      image coordinate frame.
   c. Compute a per-pixel structural difference between the warped
      reference and the test image (Gaussian-blurred absolute diff,
      optionally combined with SSIM).
   d. The resulting difference map **is** the anomaly map.

Supports two anomaly-map post-processing strategies:

* **dense** — multi-scale Gaussian-blurred absolute difference (robust,
  smooth maps).
* **ssim** — structural-similarity index computed in sliding windows
  (1 − SSIM gives the anomaly signal).

No deep learning is involved — only OpenCV and NumPy.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet normalisation constants (used to undo dataloader preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FeatureMatch(nn.Module):
    """Classical feature-matching anomaly detector with image alignment.

    Parameters
    ----------
    descriptor : str
        ``"sift"`` or ``"orb"``.
    image_size : int
        Expected spatial resolution of input tensors.
    map_mode : str
        ``"dense"`` (Gaussian-blurred absolute diff) or ``"ssim"``
        (1 − SSIM window map).
    ratio_thresh : float
        Lowe's ratio-test threshold for descriptor matching (default 0.75).
    blur_sigma : float
        Gaussian sigma for smoothing the difference map (default 7.0).
    """

    def __init__(
        self,
        descriptor: str = "sift",
        image_size: int = 256,
        map_mode: str = "dense",
        ratio_thresh: float = 0.75,
        blur_sigma: float = 7.0,
    ) -> None:
        super().__init__()
        if descriptor not in ("sift", "orb"):
            raise ValueError(f"Unsupported descriptor: {descriptor!r}")
        if map_mode not in ("dense", "ssim"):
            raise ValueError(f"Unsupported map_mode: {map_mode!r}")

        self.descriptor = descriptor
        self.image_size = image_size
        self.map_mode = map_mode
        self.ratio_thresh = ratio_thresh
        self.blur_sigma = blur_sigma

        # Populated by fit(): list of (gray_image, keypoints, descriptors).
        self._references: list[tuple[np.ndarray, list, np.ndarray | None]] = []
        # Also keep the colour versions for colour-aware diff.
        self._ref_colors: list[np.ndarray] = []
        # Mean reference (for fallback when alignment fails).
        self._mean_ref: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Descriptor extraction
    # ------------------------------------------------------------------

    def _make_extractor(self) -> cv2.Feature2D:
        if self.descriptor == "sift":
            return cv2.SIFT_create(nfeatures=5000)  # type: ignore[attr-defined]
        else:
            return cv2.ORB_create(nfeatures=5000)  # type: ignore[attr-defined]

    @staticmethod
    def _tensor_to_uint8(tensor: torch.Tensor) -> list[np.ndarray]:
        """Undo ImageNet normalisation and convert (B,3,H,W) → list of uint8 HWC RGB."""
        arr = tensor.detach().cpu().float().permute(0, 2, 3, 1).numpy()
        arr = arr * _STD + _MEAN
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return [arr[i] for i in range(arr.shape[0])]

    def _extract(self, gray: np.ndarray) -> tuple[list, np.ndarray | None]:
        """Extract keypoints and descriptors from a grayscale image."""
        extractor = self._make_extractor()
        kps, des = extractor.detectAndCompute(gray, None)  # type: ignore[union-attr]
        return list(kps), des

    # ------------------------------------------------------------------
    # Fitting (store reference images + their descriptors)
    # ------------------------------------------------------------------

    def fit(self, images: list[np.ndarray]) -> None:
        """Store normal reference images and precompute descriptors.

        Parameters
        ----------
        images : list of ndarray
            uint8 HWC RGB images.
        """
        self._references.clear()
        self._ref_colors.clear()

        h, w = self.image_size, self.image_size
        for img in images:
            # Resize to expected size.
            img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
            kps, des = self._extract(gray)
            self._references.append((gray, kps, des))
            self._ref_colors.append(img_resized)

        # Precompute mean reference for fallback.
        if self._ref_colors:
            stack = np.stack([c.astype(np.float32) for c in self._ref_colors], axis=0)
            self._mean_ref = np.mean(stack, axis=0).astype(np.uint8)
        else:
            self._mean_ref = np.zeros((h, w, 3), dtype=np.uint8)

        self._fitted = True
        n_desc = sum(
            len(des) if des is not None else 0 for _, _, des in self._references
        )
        print(
            f"FeatureMatch: {len(images)} reference images stored, "
            f"{n_desc} total descriptors "
            f"(descriptor={self.descriptor})"
        )

    # ------------------------------------------------------------------
    # Alignment: find best reference and warp it onto the test image
    # ------------------------------------------------------------------

    def _make_matcher(self) -> cv2.BFMatcher:
        if self.descriptor == "sift":
            return cv2.BFMatcher(cv2.NORM_L2)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING)

    def _align_best_reference(
        self, test_gray: np.ndarray, test_color: np.ndarray
    ) -> np.ndarray:
        """Find the best-matching reference, align it to the test via homography.

        Returns the warped reference image (uint8 HWC RGB), same size as test.
        Falls back to the mean reference if alignment fails.
        """
        h, w = test_gray.shape[:2]
        test_kps, test_des = self._extract(test_gray)

        if test_des is None or len(test_des) == 0:
            return (
                self._mean_ref
                if self._mean_ref is not None
                else np.zeros_like(test_color)
            )

        matcher = self._make_matcher()
        best_inliers = -1
        best_H = None
        best_ref_idx = 0

        for idx, (ref_gray, ref_kps, ref_des) in enumerate(self._references):
            if ref_des is None or len(ref_des) < 4:
                continue

            # Match test → reference.
            try:
                raw_matches = matcher.knnMatch(test_des, ref_des, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test to filter good matches.
            good = []
            for m_pair in raw_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < self.ratio_thresh * n.distance:
                        good.append(m)

            if len(good) < 4:
                continue

            # Compute homography (ref → test).
            src_pts = np.float32([ref_kps[m.trainIdx].pt for m in good]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([test_kps[m.queryIdx].pt for m in good]).reshape(
                -1, 1, 2
            )

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None or mask is None:
                continue

            n_inliers = int(mask.sum())
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_H = H
                best_ref_idx = idx

        # Warp best reference onto test coordinate frame.
        if best_H is not None and best_inliers >= 8:
            ref_color = self._ref_colors[best_ref_idx]
            warped = cv2.warpPerspective(ref_color, best_H, (w, h))
            return warped
        else:
            # Fallback: use mean reference (no warp).
            if self._mean_ref is not None:
                return cv2.resize(self._mean_ref, (w, h))
            return np.zeros_like(test_color)

    # ------------------------------------------------------------------
    # Anomaly map computation (post-alignment)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur with auto kernel size."""
        ksize = int(6 * sigma + 1) | 1  # Ensure odd.
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _diff_anomaly_map(self, test: np.ndarray, ref_warped: np.ndarray) -> np.ndarray:
        """Multi-scale Gaussian-blurred absolute difference.

        Returns a (H, W) float32 anomaly map in [0, 1].
        """
        # Convert to float for precision.
        t = test.astype(np.float32)
        r = ref_warped.astype(np.float32)

        # Per-channel absolute diff, then take max across channels.
        diff = np.abs(t - r)  # (H, W, 3)
        diff_gray = np.max(diff, axis=2)  # (H, W)

        # Multi-scale smoothing for robust localisation.
        sigma = self.blur_sigma
        s1 = self._gaussian_blur(diff_gray, sigma)
        s2 = self._gaussian_blur(diff_gray, sigma * 2)
        s3 = self._gaussian_blur(diff_gray, sigma * 0.5)
        combined = np.maximum(np.maximum(s1, s2), s3)

        # Normalise to [0, 1].
        lo, hi = combined.min(), combined.max()
        if hi - lo < 1e-8:
            return np.zeros_like(combined)
        return (combined - lo) / (hi - lo)

    def _ssim_anomaly_map(self, test: np.ndarray, ref_warped: np.ndarray) -> np.ndarray:
        """Compute 1 − SSIM as anomaly map.

        Uses a sliding-window SSIM computation (luminance channel).
        Returns a (H, W) float32 anomaly map in [0, 1].
        """
        # Convert to grayscale float.
        if test.ndim == 3:
            t = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            t = test.astype(np.float32)
        if ref_warped.ndim == 3:
            r = cv2.cvtColor(ref_warped, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            r = ref_warped.astype(np.float32)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        ksize = int(6 * self.blur_sigma + 1) | 1

        mu_t = cv2.GaussianBlur(t, (ksize, ksize), self.blur_sigma)
        mu_r = cv2.GaussianBlur(r, (ksize, ksize), self.blur_sigma)

        mu_t_sq = mu_t * mu_t
        mu_r_sq = mu_r * mu_r
        mu_tr = mu_t * mu_r

        sigma_t_sq = cv2.GaussianBlur(t * t, (ksize, ksize), self.blur_sigma) - mu_t_sq
        sigma_r_sq = cv2.GaussianBlur(r * r, (ksize, ksize), self.blur_sigma) - mu_r_sq
        sigma_tr = cv2.GaussianBlur(t * r, (ksize, ksize), self.blur_sigma) - mu_tr

        ssim_map = ((2 * mu_tr + C1) * (2 * sigma_tr + C2)) / (
            (mu_t_sq + mu_r_sq + C1) * (sigma_t_sq + sigma_r_sq + C2)
        )

        # Anomaly = 1 − SSIM (clamp to [0, 1]).
        amap = np.clip(1.0 - ssim_map, 0.0, 1.0).astype(np.float32)

        # Light smoothing for cleaner output.
        amap = self._gaussian_blur(amap, self.blur_sigma * 0.5)

        # Normalise.
        lo, hi = amap.min(), amap.max()
        if hi - lo < 1e-8:
            return np.zeros_like(amap)
        return (amap - lo) / (hi - lo)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run anomaly detection on a batch of ImageNet-normalised tensors.

        Returns
        -------
        scores : Tensor, shape (B,)
            Image-level anomaly scores in [0, 1].
        anomaly_maps : Tensor, shape (B, H, W)
            Per-pixel anomaly scores.
        """
        uint8_imgs = self._tensor_to_uint8(images)
        H, W = images.shape[2], images.shape[3]

        scores = []
        amaps = []

        for img in uint8_imgs:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Step 1: align best reference to the test image.
            ref_warped = self._align_best_reference(gray, img)

            # Step 2: compute anomaly map from the pixel-level difference.
            if self.map_mode == "ssim":
                amap = self._ssim_anomaly_map(img, ref_warped)
            else:  # "dense"
                amap = self._diff_anomaly_map(img, ref_warped)

            # Image-level score: mean of top-1% pixels.
            flat = amap.ravel()
            top_k = max(1, len(flat) // 100)
            score = float(np.sort(flat)[-top_k:].mean())
            scores.append(score)
            amaps.append(torch.from_numpy(amap))

        scores_t = torch.tensor(scores, dtype=torch.float32)
        amaps_t = torch.stack(amaps)
        # Ensure spatial size matches input.
        if amaps_t.shape[-2:] != (H, W):
            amaps_t = F.interpolate(
                amaps_t.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            )[:, 0]

        return scores_t, amaps_t

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predict(x)
