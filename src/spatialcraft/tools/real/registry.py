"""Construction of the full lazy real-tool registry."""

from __future__ import annotations

from collections.abc import Mapping

from ..builtin import DrawTool, GeometryTool
from ..registry import ToolRegistry
from ..remote import RemoteSpatialTool
from .detect import GroundingDINOTool
from .graph import SceneGraphTool
from .mask import MaskTool
from .motion import FarnebackMotionTool
from .ocr import OCRTool
from .pose import PoseTool
from .reconstruct import ReconstructionTool
from .scale import ScaleTool
from .segment import SAM3Tool


def create_real_tool_registry() -> ToolRegistry:
    """Create all tools without loading any neural-network weights."""

    registry = ToolRegistry()
    for tool in (
        GroundingDINOTool(),
        SAM3Tool(),
        MaskTool(),
        GeometryTool(),
        ScaleTool(),
        ReconstructionTool(),
        PoseTool(),
        SceneGraphTool(),
        FarnebackMotionTool(),
        OCRTool(),
        DrawTool(),
    ):
        registry.register(tool)
    return registry


def attach_remote_backends(
    registry: ToolRegistry,
    endpoints: Mapping[str, str],
    *,
    headers: Mapping[str, str] | None = None,
) -> ToolRegistry:
    """Attach optional service clients while retaining identical public schemas."""

    for name, endpoint in endpoints.items():
        registry.register(
            RemoteSpatialTool(registry.spec(name), endpoint, headers=headers),
            backend="remote",
        )
    return registry


__all__ = ["attach_remote_backends", "create_real_tool_registry"]
