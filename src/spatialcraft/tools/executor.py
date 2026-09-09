"""Validated tool dispatch, artifact persistence, and result auditing."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from spatialcraft.schemas import AgentAction, ToolCall, ToolResult, ToolStatus

from .artifact_store import ArtifactStore
from .base import ToolContext
from .registry import ToolRegistry
from .schema_builder import ToolSchemaError, validate_arguments


class ToolExecutor:
    """Execute model tool calls without exposing backend-specific behavior."""

    def __init__(self, registry: ToolRegistry, artifact_store: ArtifactStore) -> None:
        self.registry = registry
        self.artifact_store = artifact_store

    def execute(
        self,
        call: ToolCall,
        *,
        context: ToolContext | None = None,
        backend: str = "local",
    ) -> ToolResult:
        context = context or ToolContext(run_id=self.artifact_store.run_id)
        started_at = datetime.now(timezone.utc)
        started_clock = time.perf_counter()
        try:
            tool = self.registry.get(call.tool_name, backend=backend)
            arguments = validate_arguments(call.arguments, tool.spec.input_schema)
            # ArtifactRef URIs are relative to StorageLayout, not the process
            # working directory. Resolve URI-valued inputs without rewriting
            # the recorded model action or ordinary text/numeric arguments.
            resolved_uris = {}

            def resolve_uri(value):
                if isinstance(value, str) and value.startswith(("runs/", "objects/")):
                    path = self.artifact_store.layout.resolve_uri(value)
                    if not path.is_file():
                        raise ToolSchemaError(
                            f"Tool artifact URI does not exist: {value}"
                        )
                    resolved_uris[value] = str(path)
                    return str(path)
                return value

            arguments = {
                key: resolve_uri(value)
                if key.endswith("_uri")
                else [resolve_uri(item) for item in value]
                if key.endswith("_uris") and isinstance(value, (list, tuple))
                else value
                for key, value in arguments.items()
            }
            execution = tool.execute(arguments, context)
            artifacts = self.artifact_store.put_many(execution.artifacts)
            metadata = {
                "backend": backend,
                "tool_version": tool.spec.version,
                "invocation_id": context.invocation_id,
                **execution.metadata,
                "resolved_artifact_uris": resolved_uris,
            }
            if execution.confidence is not None:
                metadata["confidence"] = execution.confidence
            if execution.unit is not None:
                metadata["unit"] = execution.unit
            result = ToolResult(
                tool_call_id=call.call_id,
                tool_name=call.tool_name,
                status=execution.status,
                text=execution.text,
                structured_output=execution.structured_output,
                artifacts=artifacts,
                coordinate_frames=execution.coordinate_frames,
                error_type=execution.error_type,
                error_message=execution.error_message,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=(time.perf_counter() - started_clock) * 1000,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - failures become audited ToolResult
            error_type = (
                "argument_validation"
                if isinstance(exc, ToolSchemaError)
                else type(exc).__name__
            )
            result = ToolResult(
                tool_call_id=call.call_id,
                tool_name=call.tool_name,
                status=ToolStatus.FAILED,
                error_type=error_type,
                error_message=str(exc),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                latency_ms=(time.perf_counter() - started_clock) * 1000,
                metadata={"backend": backend, "invocation_id": context.invocation_id},
            )
        self.artifact_store.record_result(result)
        return result

    def execute_action(
        self,
        action: AgentAction,
        *,
        context: ToolContext | None = None,
        backend: str = "local",
        parallel: bool = False,
    ) -> tuple[ToolResult, ...]:
        if not action.tool_calls:
            return ()
        if not parallel or len(action.tool_calls) == 1:
            return tuple(
                self.execute(call, context=context, backend=backend)
                for call in action.tool_calls
            )
        with ThreadPoolExecutor(max_workers=len(action.tool_calls)) as pool:
            futures = [
                pool.submit(self.execute, call, context=context, backend=backend)
                for call in action.tool_calls
            ]
            return tuple(future.result() for future in futures)


__all__ = ["ToolExecutor"]
