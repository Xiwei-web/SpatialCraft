"""SpatialCraft tool contracts, execution, persistence, and built-ins."""

from .artifact_store import ArtifactStore
from .base import (
    ArtifactPayload,
    FunctionTool,
    SpatialTool,
    ToolConfigurationError,
    ToolContext,
    ToolError,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
)
from .builtin import DrawTool, GeometryTool
from .executor import ToolExecutor
from .mock import MockDepthTool, MockDetectionTool
from .registry import ToolRegistry
from .remote import RemoteSpatialTool


def create_mock_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (MockDetectionTool(), MockDepthTool(), GeometryTool(), DrawTool()):
        registry.register(tool)
    return registry


__all__ = [
    "ArtifactPayload",
    "ArtifactStore",
    "DrawTool",
    "FunctionTool",
    "GeometryTool",
    "MockDepthTool",
    "MockDetectionTool",
    "RemoteSpatialTool",
    "SpatialTool",
    "ToolConfigurationError",
    "ToolContext",
    "ToolError",
    "ToolExecution",
    "ToolExecutionError",
    "ToolExecutor",
    "ToolRegistry",
    "ToolSpec",
    "create_mock_tool_registry",
]
