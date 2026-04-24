"""AnomalyEUPE: Dual-level anomaly detection with EUPE ONNX backbone.

This model uses EUPE (Efficient Universal Perception Encoder) via ONNX
Runtime for fast, lightweight anomaly detection.  Unlike the AnomalyDINO
pipeline (patch-only scoring), AnomalyEUPE combines **two** complementary
signals:

1. **Global score** — cosine distance between the test image's CLS token
   and the mean normal CLS embedding.  This captures holistic anomalies
   (wrong object, missing part, colour shift).
2. **Local score** — nearest-neighbour cosine distance of each test patch
   token to the patch-level memory bank.  This localises fine-grained
   defects (scratches, stains, cracks).

The final image-level score is a weighted combination of both.

Reference
---------
Zhu, C. et al. (2026). "Efficient Universal Perception Encoder."
arXiv:2603.22387

Key properties
--------------
- Backend: ONNX Runtime (CPU or CUDA).  No PyTorch dependency for inference.
- Fixed 224x224 input -> 196 patches (14x14 grid) always.
- INT8 quantised models available (~75 % smaller, negligible accuracy loss).
- Training-free: memory bank built from normal reference images only.
- ONNX models from ``rockerritesh/EUPE-ONNX`` on HuggingFace.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA

# ── Model catalogue ──────────────────────────────────────────────────
# Maps user-facing model name to (ONNX filename, embedding dim).
_ONNX_CATALOGUE: dict[str, tuple[str, int]] = {
    "eupe_vitt16": ("eupe_vitt16.onnx", 192),
    "eupe_vitt16_int8": ("eupe_vitt16_int8.onnx", 192),
    "eupe_vits16": ("eupe_vits16.onnx", 384),
    "eupe_vits16_int8": ("eupe_vits16_int8.onnx", 384),
    "eupe_vitb16": ("eupe_vitb16.onnx", 768),
    "eupe_vitb16_int8": ("eupe_vitb16_int8.onnx", 768),
}

_HF_REPO = "rockerritesh/EUPE-ONNX"

# Fixed spatial layout for all ViT-*/16 ONNX models at 224x224.
_INPUT_SIZE = 224
_GRID_H = 14
_GRID_W = 14
_N_PATCHES = _GRID_H * _GRID_W  # 196

# ImageNet normalisation constants.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class AnomalyEUPE:
    """Dual-level anomaly detection with EUPE ONNX backbone.

    Parameters
    ----------
    model_name : str
        One of the keys in ``_ONNX_CATALOGUE``.  Append ``_int8`` for
        the quantised variant.
    masking : bool
        PCA-based zero-shot foreground masking on patch tokens.
    rotation : bool
        Augment reference images with 8 rotations.
    masking_threshold : float
        Threshold for PCA foreground/background separation.
    gaussian_sigma : float
        Gaussian smoothing sigma for pixel-level anomaly maps.
    top_percent : float
        Fraction of top patch distances for local image-level score.
    global_weight : float
        Weight of the global (CLS) score in the final image score.
        ``1 - global_weight`` is applied to the local (patch) score.
    rotation_angles : tuple of int
        Rotation angles for reference augmentation.
    providers : list of str or None
        ONNX Runtime execution providers.  ``None`` -> auto-detect.
    """

    def __init__(
        self,
        model_name: str = "eupe_vitb16",
        masking: bool = True,
        rotation: bool = True,
        masking_threshold: float = 10.0,
        gaussian_sigma: float = 4.0,
        top_percent: float = 0.01,
        global_weight: float = 0.3,
        rotation_angles: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
        providers: list[str] | None = None,
    ) -> None:
        if model_name not in _ONNX_CATALOGUE:
            raise ValueError(
                f"Unknown model_name={model_name!r}. "
                f"Choose from: {sorted(_ONNX_CATALOGUE)}"
            )

        self.model_name = model_name
        self.masking = masking
        self.rotation = rotation
        self.masking_threshold = masking_threshold
        self.gaussian_sigma = gaussian_sigma
        self.top_percent = top_percent
        self.global_weight = global_weight
        self.rotation_angles = rotation_angles

        onnx_file, self.embed_dim = _ONNX_CATALOGUE[model_name]

        # Download ONNX weights from HuggingFace if not cached.
        model_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=onnx_file,
        )

        if providers is None:
            providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)

        # ── Memory banks (populated by ``fit``) ──────────────────────
        self._cls_bank: np.ndarray | None = None  # (N_ref, D)
        self._cls_mean: np.ndarray | None = None  # (D,)
        self._patch_bank: np.ndarray | None = None  # (N_ref * 196, D)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """Convert HWC-RGB uint8 -> (1, 3, 224, 224) float32 normalised."""
        pil = Image.fromarray(img).resize((_INPUT_SIZE, _INPUT_SIZE), Image.BICUBIC)
        x = np.array(pil, dtype=np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    @staticmethod
    def _to_rgb(img: np.ndarray | str | Image.Image) -> np.ndarray:
        """Normalise any input to HWC-RGB uint8 numpy array."""
        if isinstance(img, str):
            return cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if isinstance(img, Image.Image):
            return np.asarray(img.convert("RGB"))
        return np.ascontiguousarray(img)

    # ------------------------------------------------------------------
    # Feature extraction (ONNX)
    # ------------------------------------------------------------------

    def extract(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run ONNX inference on a single HWC-RGB image.

        Returns
        -------
        cls_token : (D,) float32
        patch_tokens : (196, D) float32
        """
        inp = self._preprocess(img)
        cls_token, patch_tokens = self.session.run(None, {"input": inp})
        return cls_token[0], patch_tokens[0]

    # ------------------------------------------------------------------
    # PCA-based foreground masking
    # ------------------------------------------------------------------

    def compute_foreground_mask(
        self,
        features: np.ndarray,
        threshold: float | None = None,
        border: float = 0.2,
        kernel_size: int = 3,
    ) -> np.ndarray:
        """Binary foreground mask via PCA thresholding on patch tokens.

        Returns
        -------
        mask : (196,) boolean array -- True = foreground.
        """
        if threshold is None:
            threshold = self.masking_threshold

        pca = PCA(n_components=1, svd_solver="randomized")
        first_pc = pca.fit_transform(features.astype(np.float32))
        mask = first_pc.squeeze() > threshold

        mask_2d = mask.reshape((_GRID_H, _GRID_W))
        r0 = int(_GRID_H * border)
        r1 = int(_GRID_H * (1 - border))
        c0 = int(_GRID_W * border)
        c1 = int(_GRID_W * (1 - border))
        centre = mask_2d[r0:r1, c0:c1]
        if centre.sum() <= centre.size * 0.35:
            mask = ~mask

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_uint8 = mask.astype(np.uint8)
        mask_uint8 = cv2.dilate(mask_uint8, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        return mask_uint8.astype(bool).ravel()

    # ------------------------------------------------------------------
    # Reference augmentation
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
        """Augment a reference image with rotations."""
        return [self.rotate_image(image, angle) for angle in self.rotation_angles]

    # ------------------------------------------------------------------
    # Memory-bank construction
    # ------------------------------------------------------------------

    def fit(self, images: list[np.ndarray | str | Image.Image]) -> None:
        """Build dual memory banks from normal reference images.

        Stores:
        - ``_cls_bank``   (N_variants, D)  -- CLS embeddings per variant
        - ``_cls_mean``   (D,)             -- mean normal CLS embedding
        - ``_patch_bank`` (N_variants*196, D) -- patch-level memory bank
        """
        cls_list: list[np.ndarray] = []
        patch_list: list[np.ndarray] = []

        for img in images:
            raw = self._to_rgb(img)
            variants = self.augment_reference(raw) if self.rotation else [raw]

            for variant in variants:
                cls_tok, patch_tok = self.extract(variant)
                cls_list.append(cls_tok)
                patch_list.append(patch_tok)

        # CLS bank
        self._cls_bank = np.stack(cls_list, axis=0).astype(np.float32)
        norms = np.linalg.norm(self._cls_bank, axis=1, keepdims=True)
        self._cls_bank /= np.maximum(norms, 1e-12)
        self._cls_mean = self._cls_bank.mean(axis=0)
        self._cls_mean /= max(np.linalg.norm(self._cls_mean), 1e-12)

        # Patch bank -- pre-allocated to avoid peak doubling
        total_patches = sum(p.shape[0] for p in patch_list)
        dim = patch_list[0].shape[1]
        self._patch_bank = np.empty((total_patches, dim), dtype=np.float32)
        offset = 0
        for p in patch_list:
            n = p.shape[0]
            self._patch_bank[offset : offset + n] = p
            offset += n
        del patch_list

        norms = np.linalg.norm(self._patch_bank, axis=1, keepdims=True)
        self._patch_bank /= np.maximum(norms, 1e-12)

        self._fitted = True

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray | str | Image.Image,
    ) -> tuple[float, np.ndarray]:
        """Score a single test image.

        Returns
        -------
        score : float
            Image-level anomaly score (weighted global + local).
        anomaly_map : (H_img, W_img)  numpy array
            Pixel-level anomaly heatmap.
        """
        assert self._fitted, "Call fit() before predict()."

        raw = self._to_rgb(image)
        img_h, img_w = raw.shape[:2]

        cls_tok, patch_tok = self.extract(raw)

        # ── Global score (CLS cosine distance) ───────────────────────
        cls_normed = cls_tok / max(np.linalg.norm(cls_tok), 1e-12)
        global_sim = float(np.dot(cls_normed, self._cls_mean))
        global_score = 1.0 - global_sim

        # ── Local score (patch NN cosine distance) ───────────────────
        if self.masking:
            mask = self.compute_foreground_mask(patch_tok)
        else:
            mask = np.ones(patch_tok.shape[0], dtype=bool)

        norms = np.linalg.norm(patch_tok, axis=1, keepdims=True)
        patch_normed = patch_tok / np.maximum(norms, 1e-12)

        masked_patches = patch_normed[mask]
        max_sim = self._chunked_nn(masked_patches, self._patch_bank)
        distances = 1.0 - max_sim

        output_distances = np.zeros(patch_tok.shape[0], dtype=np.float64)
        output_distances[mask] = distances

        local_score = self._mean_top_percent(output_distances)

        # ── Combined score ───────────────────────────────────────────
        score = (
            self.global_weight * global_score + (1.0 - self.global_weight) * local_score
        )

        # ── Pixel-level anomaly map ──────────────────────────────────
        distance_map = output_distances.reshape((_GRID_H, _GRID_W))
        anomaly_map = cv2.resize(
            distance_map.astype(np.float32),
            (img_w, img_h),
            interpolation=cv2.INTER_LINEAR,
        )
        anomaly_map = gaussian_filter(anomaly_map, sigma=self.gaussian_sigma)

        return float(score), anomaly_map

    # ------------------------------------------------------------------
    # Batch predict (convenience for Lightning module)
    # ------------------------------------------------------------------

    def predict_batch_numpy(
        self,
        images: list[np.ndarray],
        output_size: tuple[int, int] = (_INPUT_SIZE, _INPUT_SIZE),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score a batch of HWC-RGB uint8 images.

        Returns
        -------
        scores : (B,) float32 array
        anomaly_maps : (B, H_out, W_out) float32 array
        """
        assert self._fitted, "Call fit() before predict_batch_numpy()."

        scores_list: list[float] = []
        maps_list: list[np.ndarray] = []

        for raw in images:
            cls_tok, patch_tok = self.extract(raw)

            # Global score
            cls_normed = cls_tok / max(np.linalg.norm(cls_tok), 1e-12)
            global_sim = float(np.dot(cls_normed, self._cls_mean))
            global_score = 1.0 - global_sim

            # Local score
            if self.masking:
                mask = self.compute_foreground_mask(patch_tok)
            else:
                mask = np.ones(patch_tok.shape[0], dtype=bool)

            norms = np.linalg.norm(patch_tok, axis=1, keepdims=True)
            patch_normed = patch_tok / np.maximum(norms, 1e-12)
            masked = patch_normed[mask]
            max_sim = self._chunked_nn(masked, self._patch_bank)
            dists = 1.0 - max_sim

            output_dists = np.zeros(patch_tok.shape[0], dtype=np.float64)
            output_dists[mask] = dists

            local_score = self._mean_top_percent(output_dists)
            score = (
                self.global_weight * global_score
                + (1.0 - self.global_weight) * local_score
            )
            scores_list.append(score)

            dist_map = output_dists.reshape((_GRID_H, _GRID_W))
            amap = cv2.resize(
                dist_map.astype(np.float32),
                (output_size[1], output_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            amap = gaussian_filter(amap, sigma=self.gaussian_sigma)
            maps_list.append(amap)

        return np.array(scores_list, dtype=np.float32), np.stack(maps_list)

    def predict_batch_tensor(
        self,
        x: "torch.Tensor",
        output_size: tuple[int, int] | None = None,
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Convenience wrapper: accept an ImageNet-normalised (B,3,H,W) tensor.

        Returns torch tensors to match the AnomalyDINO API used by demo.py.
        scores : (B,) float32
        anomaly_maps : (B, 1, H_out, W_out) float32
        """
        import torch

        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        rgb = ((x * std + mean) * 255.0).clamp(0, 255).byte()

        images = [rgb[i].permute(1, 2, 0).cpu().numpy() for i in range(rgb.shape[0])]

        if output_size is None:
            output_size = (x.shape[2], x.shape[3])

        scores_np, maps_np = self.predict_batch_numpy(images, output_size=output_size)

        scores_t = torch.from_numpy(scores_np)
        maps_t = torch.from_numpy(maps_np).unsqueeze(1)  # (B, 1, H, W)
        return scores_t, maps_t

    # ------------------------------------------------------------------
    # Chunked nearest-neighbour
    # ------------------------------------------------------------------

    @staticmethod
    def _chunked_nn(
        queries: np.ndarray,
        bank: np.ndarray,
        chunk_size: int = 65536,
    ) -> np.ndarray:
        """Chunked cosine NN search (numpy, CPU)."""
        max_sim = np.full(queries.shape[0], -np.inf, dtype=np.float32)
        for start in range(0, bank.shape[0], chunk_size):
            chunk = bank[start : start + chunk_size]
            sims = queries @ chunk.T
            chunk_max = sims.max(axis=1)
            np.maximum(max_sim, chunk_max, out=max_sim)
        return max_sim

    def _mean_top_percent(self, distances: np.ndarray) -> float:
        """Mean of the top ``self.top_percent`` distances."""
        flat = distances.ravel()
        k = max(1, int(len(flat) * self.top_percent))
        top_k = np.argpartition(flat, -k)[-k:]
        return float(np.mean(flat[top_k]))
