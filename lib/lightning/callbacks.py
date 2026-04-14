"""Lightning callbacks."""

from __future__ import annotations

import os
import time
from typing import Any

import psutil
import torch
from pytorch_lightning import Callback, LightningModule, Trainer


class InferenceSpeedMonitor(Callback):
    """Measure mean per-batch and per-sample time during training and test.

    At the end of each train / test epoch the callback logs:

    * ``{stage}/batch_time_ms`` – mean wall-clock time per batch (ms).
    * ``{stage}/sample_time_ms`` – mean wall-clock time per sample (ms).
    * ``{stage}/throughput_samples_per_sec`` – samples / second.

    GPU synchronisation is performed automatically when the model lives on a
    CUDA or XPU device so that the timings are accurate.
    """

    def __init__(self) -> None:
        super().__init__()
        self._times: list[float] = []
        self._batch_sizes: list[int] = []
        self._t0: float = 0.0

    # -- train hooks ---------------------------------------------------

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._times.clear()
        self._batch_sizes.clear()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._sync(pl_module.device)
        self._t0 = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._sync(pl_module.device)
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        self._batch_sizes.append(self._infer_batch_size(batch))

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_speed("train", pl_module)

    # -- test hooks ----------------------------------------------------

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._times.clear()
        self._batch_sizes.clear()

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._sync(pl_module.device)
        self._t0 = time.perf_counter()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._sync(pl_module.device)
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        self._batch_sizes.append(self._infer_batch_size(batch))

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_speed("test", pl_module)

    # -- helpers -------------------------------------------------------

    def _log_speed(self, stage: str, pl_module: LightningModule) -> None:
        if not self._times:
            return
        total_time = sum(self._times)
        total_samples = sum(self._batch_sizes)
        n_batches = len(self._times)

        mean_batch_ms = (total_time / n_batches) * 1000.0
        mean_sample_ms = (total_time / max(total_samples, 1)) * 1000.0
        throughput = total_samples / max(total_time, 1e-9)

        pl_module.log(f"{stage}/batch_time_ms", mean_batch_ms)
        pl_module.log(f"{stage}/sample_time_ms", mean_sample_ms)
        pl_module.log(f"{stage}/throughput_samples_per_sec", throughput)

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _sync(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.synchronize(device)

    @staticmethod
    def _infer_batch_size(batch: Any) -> int:
        if isinstance(batch, dict):
            first = next(iter(batch.values()))
            return first.shape[0] if hasattr(first, "shape") else 1
        if isinstance(batch, (list, tuple)):
            first = batch[0]
            return first.shape[0] if hasattr(first, "shape") else 1
        if hasattr(batch, "shape"):
            return batch.shape[0]
        return 1


class MemoryMonitor(Callback):
    """Log RAM and VRAM usage at the end of each training and test epoch.

    Logged metrics:

    * ``{stage}/ram_usage_mb`` – process-resident RAM in MiB.
    * ``{stage}/vram_usage_mb`` – GPU memory allocated in MiB (CUDA / XPU).
    * ``{stage}/vram_reserved_mb`` – GPU memory reserved by the caching allocator.
    """

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _get_ram_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def _get_vram_mb(device: torch.device) -> tuple[float, float]:
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
            return allocated, reserved
        if device.type == "xpu" and hasattr(torch, "xpu"):
            allocated = torch.xpu.memory_allocated(device) / (1024 * 1024)
            reserved = torch.xpu.memory_reserved(device) / (1024 * 1024)
            return allocated, reserved
        return 0.0, 0.0

    def _log_memory(self, stage: str, pl_module: LightningModule) -> None:
        ram_mb = self._get_ram_mb()
        vram_alloc, vram_reserved = self._get_vram_mb(pl_module.device)

        pl_module.log(f"{stage}/ram_usage_mb", ram_mb)
        pl_module.log(f"{stage}/vram_usage_mb", vram_alloc)
        pl_module.log(f"{stage}/vram_reserved_mb", vram_reserved)

    # -- hooks ---------------------------------------------------------

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_memory("train", pl_module)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._log_memory("val", pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_memory("test", pl_module)
