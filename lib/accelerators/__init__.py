"""Custom XPU accelerator for PyTorch Lightning.

PyTorch supports Intel XPU natively (torch.xpu), but Lightning may not ship
a built-in XPU accelerator in all versions. This module bridges the gap.
"""

from __future__ import annotations

import torch
from pytorch_lightning.accelerators import Accelerator


class XPUAccelerator(Accelerator):
    """Lightning accelerator for Intel XPU devices."""

    @staticmethod
    def parse_devices(devices: int | list[int] | str) -> list[int]:
        if isinstance(devices, int):
            return list(range(devices))
        if isinstance(devices, str):
            return [int(d) for d in devices.split(",")]
        return list(devices)

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    def setup_device(self, device: torch.device) -> None:
        if device.type == "xpu":
            torch.xpu.set_device(device)

    def teardown(self) -> None:
        if hasattr(torch.xpu, "empty_cache"):
            torch.xpu.empty_cache()

    @staticmethod
    def name() -> str:
        return "xpu"

    def get_device_stats(self, device: torch.device | str) -> dict[str, float]:
        if hasattr(torch.xpu, "memory_allocated"):
            return {
                "xpu_memory_allocated": torch.xpu.memory_allocated(device),
                "xpu_memory_reserved": torch.xpu.memory_reserved(device),
            }
        return {}
