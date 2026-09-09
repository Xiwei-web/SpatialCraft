"""HTTP implementation of the same :class:`SpatialTool` contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import (
    SpatialTool,
    ToolContext,
    ToolExecution,
    ToolExecutionError,
    ToolSpec,
)

RemoteTransport = Callable[[dict[str, Any]], Mapping[str, Any]]


class RemoteSpatialTool(SpatialTool):
    """Invoke a JSON/base64 spatial-tool service over HTTP or an injected transport."""

    def __init__(
        self,
        spec: ToolSpec,
        endpoint: str,
        *,
        transport: RemoteTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("remote tool endpoint cannot be empty")
        self.spec = spec
        self.endpoint = endpoint
        self.transport = transport
        self.headers = dict(headers or {})

    def execute(
        self, arguments: Mapping[str, Any], context: ToolContext
    ) -> ToolExecution:
        request = {
            "tool": self.spec.name,
            "version": self.spec.version,
            "arguments": dict(arguments),
            "context": context.to_wire(),
        }
        if self.transport is not None:
            response = self.transport(request)
        else:
            try:
                import httpx
            except ImportError as exc:
                raise ToolExecutionError("httpx is required for remote tools") from exc
            try:
                result = httpx.post(
                    self.endpoint,
                    json=request,
                    headers=self.headers,
                    timeout=context.timeout_s or self.spec.default_timeout_s,
                )
                result.raise_for_status()
                response = result.json()
            except Exception as exc:
                raise ToolExecutionError(
                    f"remote tool {self.spec.name} failed: {exc}"
                ) from exc
        if not isinstance(response, Mapping):
            raise ToolExecutionError("remote tool response must be a JSON object")
        try:
            return ToolExecution.from_wire(response)
        except Exception as exc:
            raise ToolExecutionError(f"invalid remote tool response: {exc}") from exc


__all__ = ["RemoteSpatialTool", "RemoteTransport"]
