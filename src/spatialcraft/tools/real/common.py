"""Shared lazy-loading, local-path, image, and array helpers for real tools."""

from __future__ import annotations

import io
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

ResourceT = TypeVar("ResourceT")
DEFAULT_TOOL_ROOT = Path("/l/users/xiwei.liu/tool")


class LazyResource(Generic[ResourceT]):
    """Load a heavyweight resource exactly once at first inference."""

    def __init__(self, factory: Callable[[], ResourceT]) -> None:
        self.factory = factory
        self._value: ResourceT | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._value is not None

    def get(self) -> ResourceT:
        if self._value is not None:
            return self._value
        with self._lock:
            if self._value is None:
                self._value = self.factory()
        return self._value

    def release(self) -> None:
        """Drop cached weights after a serial tool invocation has returned.

        The caller must ensure there is no in-flight use of the returned resource.
        """
        with self._lock:
            self._value = None


@dataclass(frozen=True, slots=True)
class SpatialToolPaths:
    root: Path = DEFAULT_TOOL_ROOT

    @classmethod
    def from_env(cls) -> SpatialToolPaths:
        return cls(
            Path(os.environ.get("SPATIALCRAFT_TOOL_ROOT", str(DEFAULT_TOOL_ROOT)))
        )

    @property
    def groundingdino_repo(self) -> Path:
        return self.root / "repos" / "GroundingDINO"

    @property
    def groundingdino_config(self) -> Path:
        return (
            self.groundingdino_repo
            / "groundingdino"
            / "config"
            / "GroundingDINO_SwinT_OGC.py"
        )

    @property
    def groundingdino_checkpoint(self) -> Path:
        return (
            self.root / "checkpoints" / "groundingdino" / "groundingdino_swint_ogc.pth"
        )

    @property
    def groundingdino_bert(self) -> Path:
        return self.root / "checkpoints" / "groundingdino" / "bert-base-uncased"

    @property
    def sam3_repo(self) -> Path:
        return self.root / "repos" / "sam3"

    @property
    def sam3_checkpoint(self) -> Path:
        return self.root / "checkpoints" / "sam3" / "facebook-sam3" / "sam3.pt"

    @property
    def moge_repo(self) -> Path:
        return self.root / "repos" / "MoGe"

    @property
    def moge_checkpoint(self) -> Path:
        return self.root / "checkpoints" / "moge2" / "moge-2-vits-normal" / "model.pt"

    @property
    def da3_repo(self) -> Path:
        return self.root / "repos" / "Depth-Anything-3"

    @property
    def da3_checkpoint(self) -> Path:
        return self.root / "checkpoints" / "depth_anything_3" / "DA3-BASE"

    @property
    def orient_repo(self) -> Path:
        return self.root / "repos" / "Orient-Anything"

    @property
    def orient_backbone(self) -> Path:
        return self.root / "checkpoints" / "orient_anything" / "dinov2-small"

    @property
    def orient_checkpoint(self) -> Path:
        return (
            self.root
            / "checkpoints"
            / "orient_anything"
            / "OriNet"
            / "cropsmallEx03"
            / "dino_weight.pt"
        )

    @property
    def easyocr_repo(self) -> Path:
        return self.root / "repos" / "EasyOCR"

    @property
    def easyocr_models(self) -> Path:
        return self.root / "checkpoints" / "easyocr"


def add_python_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"tool repository not found: {resolved}")
    value = str(resolved)
    if value not in sys.path:
        sys.path.insert(0, value)
    return resolved


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def numpy_bytes(array, *, compressed: bool = False, **arrays) -> bytes:
    import numpy as np

    stream = io.BytesIO()
    if arrays:
        values = {
            "array": np.asarray(array),
            **{k: np.asarray(v) for k, v in arrays.items()},
        }
        if compressed:
            np.savez_compressed(stream, **values)
        else:
            np.savez(stream, **values)
    else:
        np.save(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def png_bytes(array, *, mode: str | None = None) -> bytes:
    from PIL import Image

    stream = io.BytesIO()
    Image.fromarray(array, mode=mode).save(stream, format="PNG")
    return stream.getvalue()


def normalized_preview(array) -> bytes:
    import numpy as np

    value = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(value)
    if not finite.any():
        preview = np.zeros(value.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(value[finite], (2, 98))
        preview = np.clip((value - low) / max(1e-6, high - low), 0, 1)
        preview = (preview * 255).astype(np.uint8)
    return png_bytes(preview, mode="L")


__all__ = [
    "LazyResource",
    "SpatialToolPaths",
    "add_python_path",
    "normalized_preview",
    "numpy_bytes",
    "png_bytes",
    "require_file",
]
