"""Lazy adapters for production spatial models and algorithms."""

from .common import LazyResource, SpatialToolPaths
from .detect import GroundingDINOAdapter, GroundingDINOTool
from .graph import SceneGraphTool
from .mask import MaskTool
from .motion import FarnebackMotionTool
from .ocr import EasyOCRAdapter, OCRTool
from .pose import OrientAnythingAdapter, PoseTool
from .reconstruct import DepthAnything3Adapter, ReconstructionTool
from .registry import attach_remote_backends, create_real_tool_registry
from .scale import MoGe2Adapter, ScaleTool
from .segment import SAM3Adapter, SAM3Tool

__all__ = [
    "DepthAnything3Adapter",
    "EasyOCRAdapter",
    "FarnebackMotionTool",
    "GroundingDINOAdapter",
    "GroundingDINOTool",
    "LazyResource",
    "MaskTool",
    "MoGe2Adapter",
    "OCRTool",
    "OrientAnythingAdapter",
    "PoseTool",
    "ReconstructionTool",
    "SAM3Adapter",
    "SAM3Tool",
    "ScaleTool",
    "SceneGraphTool",
    "SpatialToolPaths",
    "attach_remote_backends",
    "create_real_tool_registry",
]
