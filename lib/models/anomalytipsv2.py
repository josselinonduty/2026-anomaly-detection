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
from transformers import AutoModel


class AnomalyTIPSv2(nn.Module):
    """AnomalyTIPSv2 anomaly detection model.

    This is a TIPSv2-backed adaptation of AnomalyDINO:
    - frozen TIPSv2 backbone
    - patch-level nearest-neighbour anomaly detection
    - cosine distance to nominal memory bank
    - image-level score = mean of top-k patch distances
    - pixel-level anomaly map = upsampled patch-distance map + Gaussian smoothing

    Parameters
    ----------
    model_name : str
        Hugging Face TIPSv2 checkpoint, e.g.
        'google/tipsv2-b14', 'google/tipsv2-l14',
        'google/tipsv2-so400m14', 'google/tipsv2-g14'.
    smaller_edge_size : int
        Images are resized so the smaller edge equals this value.
    masking : bool
        Whether to apply PCA-based foreground masking on test images.
    rotation : bool
        Whether to augment reference images with rotations.
    masking_threshold : float
        Threshold on the first PCA component for foreground/background separation.
    gaussian_sigma : float
        Standard deviation for Gaussian smoothing of the anomaly map.
    top_percent : float
        Fraction of highest patch distances used for image-level scoring.
    rotation_angles : tuple[int, ...]
        Rotation angles for reference augmentation.
    trust_remote_code : bool
        Passed to Hugging Face AutoModel.
    """

    def __init__(
        self,
        model_name: str = "google/tipsv2-b14",
        smaller_edge_size: int = 448,
        masking: bool = True,
        rotation: bool = True,
        masking_threshold: float = 10.0,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        rotation_angles: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
        trust_remote_code: bool = True,
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

        # TIPSv2 backbone (frozen).
        self.backbone: nn.Module = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # TIPSv2 models use patch size 14.
        self.patch_size: int = 14

        # TIPSv2 expects tensors in [0, 1], no ImageNet normalization.
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    size=smaller_edge_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )

        self._memory_bank: np.ndarray | None = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def prepare_image(
        self,
        img: np.ndarray | str | Image.Image | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Load and preprocess an image for TIPSv2.

        Returns the image tensor in [0,1] and the (H_grid, W_grid) patch grid.
        """
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
            image_tensor = self.transform(img)

        elif isinstance(img, Image.Image):
            image_tensor = self.transform(img.convert("RGB"))

        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img)
            image_tensor = self.transform(pil)

        elif isinstance(img, torch.Tensor):
            # Expect CHW tensor already in [0,1].
            if img.dim() != 3:
                raise ValueError(f"Expected CHW tensor, got shape={tuple(img.shape)}")
            image_tensor = img.detach().cpu().float().clamp(0.0, 1.0)
            image_tensor = transforms.Resize(
                size=self.smaller_edge_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )(image_tensor)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

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
        image_tensor : (C, H, W) or (1, C, H, W) in [0,1]

        Returns
        -------
        features : (N_patches, D) numpy array
        """
        device = next(self.backbone.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        out = self.backbone.encode_image(image_tensor)
        patch_tokens = out.patch_tokens.squeeze(0)  # (N, D)
        return patch_tokens.detach().cpu().numpy()

    @torch.no_grad()
    def extract_features_with_cls(
        self,
        image_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract patch tokens and global cls token."""
        device = next(self.backbone.parameters()).device
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        out = self.backbone.encode_image(image_tensor)
        patch_tokens = out.patch_tokens.squeeze(0)  # (N, D)
        cls_token = out.cls_token[:, 0, :].squeeze(0)  # (D,)

        return (
            patch_tokens.detach().cpu().numpy(),
            cls_token.detach().cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # PCA-based foreground masking
    # ------------------------------------------------------------------

    def compute_foreground_mask(
        self,
        features: np.ndarray,
        grid_size: tuple[int, int],
        threshold: float | None = None,
        border: float = 0.2,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Compute a binary foreground mask via PCA thresholding."""
        if threshold is None:
            threshold = self.masking_threshold

        pca = PCA(n_components=1, svd_solver="randomized")
        first_pc = pca.fit_transform(features.astype(np.float32))

        mask = first_pc.squeeze() > threshold

        mask_2d = mask.reshape(grid_size)
        r0 = int(grid_size[0] * border)
        r1 = int(grid_size[0] * (1 - border))
        c0 = int(grid_size[1] * border)
        c1 = int(grid_size[1] * (1 - border))
        centre = mask_2d[r0:r1, c0:c1]
        if centre.sum() <= centre.size * 0.35:
            mask = ~mask

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_uint8 = mask.astype(np.uint8).reshape(grid_size)
        mask_uint8 = cv2.dilate(mask_uint8, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        return mask_uint8.astype(bool).ravel()

    # ------------------------------------------------------------------
    # Reference augmentation
    # ------------------------------------------------------------------

    @staticmethod
    def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate an image by angle degrees around its centre."""
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
        """Augment a reference image with rotations."""
        return [self.rotate_image(image, angle) for angle in self.rotation_angles]

    # ------------------------------------------------------------------
    # Memory-bank construction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, images: list[np.ndarray | str | Image.Image | torch.Tensor]) -> None:
        """Build the nominal memory bank from reference images."""
        all_features: list[np.ndarray] = []

        for img in images:
            if isinstance(img, str):
                raw = cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            elif isinstance(img, Image.Image):
                raw = np.asarray(img.convert("RGB"))
            elif isinstance(img, torch.Tensor):
                # Tensor path: use directly, no rotation augmentation.
                img_tensor, _ = self.prepare_image(img)
                features = self.extract_features(img_tensor)
                all_features.append(features)
                continue
            else:
                raw = img

            variants = self.augment_reference(raw) if self.rotation else [raw]

            for variant in variants:
                img_tensor, _ = self.prepare_image(variant)
                features = self.extract_features(img_tensor)
                all_features.append(features)

        self._memory_bank = np.concatenate(all_features, axis=0).astype(np.float32)
        norms = np.linalg.norm(self._memory_bank, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self._memory_bank = self._memory_bank / norms

        self._fitted = True

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray | str | Image.Image | torch.Tensor,
    ) -> tuple[float, np.ndarray]:
        """Score a single test image.

        Returns
        -------
        score : float
            Image-level anomaly score.
        anomaly_map : (H_img, W_img) numpy array
            Pixel-level anomaly scores.
        """
        assert self._fitted, "Call fit() before predict()."

        if isinstance(image, str):
            raw = cv2.cvtColor(cv2.imread(image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img_h, img_w = raw.shape[:2]
            img_tensor, grid_size = self.prepare_image(raw)
        elif isinstance(image, Image.Image):
            raw = np.asarray(image.convert("RGB"))
            img_h, img_w = raw.shape[:2]
            img_tensor, grid_size = self.prepare_image(raw)
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[:2]
            img_tensor, grid_size = self.prepare_image(image)
        elif isinstance(image, torch.Tensor):
            _, h, w = image.shape
            img_h, img_w = h, w
            img_tensor, grid_size = self.prepare_image(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        features = self.extract_features(img_tensor)

        if self.masking:
            mask = self.compute_foreground_mask(features, grid_size)
        else:
            mask = np.ones(features.shape[0], dtype=bool)

        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        features_normed = features / norms

        masked_features = features_normed[mask]
        max_sim = self._chunked_nn_numpy(masked_features, self._memory_bank)
        distances = 1.0 - max_sim

        output_distances = np.zeros(features.shape[0], dtype=np.float64)
        output_distances[mask] = distances

        score = self._mean_top_percent(output_distances)

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
        """Chunked cosine nearest-neighbour search on CPU."""
        max_sim = np.full(queries.shape[0], -np.inf, dtype=np.float32)
        for start in range(0, bank.shape[0], chunk_size):
            chunk = bank[start : start + chunk_size]
            sims = queries @ chunk.T
            chunk_max = sims.max(axis=1)
            np.maximum(max_sim, chunk_max, out=max_sim)
        return max_sim

    @staticmethod
    def _chunked_nn_torch(
        queries: torch.Tensor,
        bank_np: np.ndarray,
        chunk_size: int = 65536,
    ) -> torch.Tensor:
        """Chunked cosine nearest-neighbour search on device."""
        device = queries.device
        max_sim = torch.full((queries.shape[0],), -float("inf"), device=device)
        for start in range(0, bank_np.shape[0], chunk_size):
            chunk = torch.from_numpy(bank_np[start : start + chunk_size]).to(device)
            sims = queries @ chunk.T
            chunk_max, _ = sims.max(dim=1)
            max_sim = torch.maximum(max_sim, chunk_max)
        return max_sim

    def _mean_top_percent(self, distances: np.ndarray) -> float:
        """Compute mean of top self.top_percent distances."""
        flat = distances.ravel()
        k = max(1, int(len(flat) * self.top_percent))
        top_k_indices = np.argpartition(flat, -k)[-k:]
        return float(np.mean(flat[top_k_indices]))

    # ------------------------------------------------------------------
    # Batch predict
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch_tensor(
        self,
        images: torch.Tensor,
        original_sizes: list[tuple[int, int]] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch of image tensors already in [0,1].

        Parameters
        ----------
        images : (B, C, H, W)
            Tensors in [0,1], no ImageNet normalization.
        original_sizes : list[(H, W)] | None
            Per-image output size.
        output_size : (H, W) | None
            Fallback output size.

        Returns
        -------
        scores : (B,)
        anomaly_maps : (B, 1, H_out, W_out)
        """
        assert self._fitted, "Call fit() before predict_batch_tensor()."

        device = images.device
        B = images.shape[0]

        _, _, h, w = images.shape
        cropped_h = h - h % self.patch_size
        cropped_w = w - w % self.patch_size
        images = images[:, :, :cropped_h, :cropped_w]

        grid_h = cropped_h // self.patch_size
        grid_w = cropped_w // self.patch_size
        grid_size = (grid_h, grid_w)

        scores_list: list[float] = []
        maps_list: list[torch.Tensor] = []

        for i in range(B):
            img_tensor = images[i].unsqueeze(0)
            out = self.backbone.encode_image(img_tensor)
            patch_tokens = out.patch_tokens.squeeze(0)  # (N, D)

            features_np = patch_tokens.detach().cpu().numpy()

            if self.masking:
                mask = self.compute_foreground_mask(features_np, grid_size)
            else:
                mask = np.ones(features_np.shape[0], dtype=bool)

            features_t = F.normalize(patch_tokens, p=2, dim=1)

            mask_t = torch.from_numpy(mask).to(device)
            masked_features = features_t[mask_t]

            max_sim = self._chunked_nn_torch(masked_features, self._memory_bank)
            dists = 1.0 - max_sim

            output_dists = torch.zeros(features_t.shape[0], device=device)
            output_dists[mask_t] = dists.float()

            flat = output_dists
            k = max(1, int(len(flat) * self.top_percent))
            topk_vals, _ = torch.topk(flat, k)
            score = topk_vals.mean().item()
            scores_list.append(score)

            dist_map = output_dists.reshape(1, 1, grid_h, grid_w)

            if original_sizes is not None:
                oh, ow = original_sizes[i]
            elif output_size is not None:
                oh, ow = output_size
            else:
                oh, ow = cropped_h, cropped_w

            amap = F.interpolate(
                dist_map.float(),
                size=(oh, ow),
                mode="bilinear",
                align_corners=False,
            )
            amap_np = amap.squeeze().detach().cpu().numpy()
            amap_np = gaussian_filter(amap_np, sigma=self.gaussian_sigma)
            amap = torch.from_numpy(amap_np).unsqueeze(0).unsqueeze(0).to(device)
            maps_list.append(amap)

        scores = torch.tensor(scores_list, device=device)
        anomaly_maps = torch.cat(maps_list, dim=0)
        return scores, anomaly_maps
