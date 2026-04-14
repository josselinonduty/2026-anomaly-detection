"""Unified checkpoint save / load for all model architectures.

Checkpoint layout::

    checkpoints/<model>/<category>/<YYYYMMDD_HHMMSS>/
        model.ckpt          # Lightning checkpoint (weights + hparams)
        memory_bank.pt      # PatchCore only — coreset memory bank
        metadata.json        # Human-readable run info

Every architecture exposes a ``save_checkpoint(path)`` /
``load_checkpoint(path)`` interface so the orchestrator can persist and
restore the *full* model state — including non-parameter artefacts like
PatchCore's memory bank or EfficientAd's normalisation buffers.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def make_checkpoint_dir(
    root: str | Path,
    model_name: str,
    category: str,
) -> Path:
    """Create and return a timestamped checkpoint directory.

    Returns
    -------
    Path
        ``<root>/<model_name>/<category>/<YYYYMMDD_HHMMSS>/``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(root) / model_name / category / timestamp
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def save_metadata(
    ckpt_dir: Path,
    *,
    model_name: str,
    category: str,
    extra: dict | None = None,
) -> None:
    """Write a small JSON sidecar with human-readable run info."""
    meta = {
        "model": model_name,
        "category": category,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        meta.update(extra)
    (ckpt_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def latest_checkpoint_dir(
    root: str | Path,
    model_name: str,
    category: str,
) -> Path | None:
    """Return the most-recent timestamped checkpoint directory, or *None*."""
    base = Path(root) / model_name / category
    if not base.exists():
        return None
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[0] if dirs else None
