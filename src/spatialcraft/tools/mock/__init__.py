"""Deterministic mock tools for tests and CPU-only development."""

from .depth import MockDepthTool
from .detection import MockDetectionTool

__all__ = ["MockDepthTool", "MockDetectionTool"]
