"""Registry for interchangeable local and remote tool backends."""

from __future__ import annotations

from collections.abc import Iterable

from spatialcraft.models import ToolDefinition

from .base import SpatialTool, ToolConfigurationError, ToolSpec


class ToolRegistry:
    """Bind a public tool name to one or more execution backends."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, SpatialTool]] = {}
        self._specs: dict[str, ToolSpec] = {}

    @staticmethod
    def _key(value: str) -> str:
        key = value.strip().lower().replace("_", "-")
        if not key:
            raise ValueError("tool or backend name cannot be empty")
        return key

    def register(
        self,
        tool: SpatialTool,
        *,
        backend: str = "local",
        replace: bool = False,
    ) -> None:
        name = self._key(tool.spec.name)
        backend_key = self._key(backend)
        existing_spec = self._specs.get(name)
        if existing_spec is not None and (
            existing_spec.input_schema != tool.spec.input_schema
            or existing_spec.output_schema != tool.spec.output_schema
        ):
            raise ToolConfigurationError(
                f"backend {backend_key!r} changes the public schema for {name!r}"
            )
        backends = self._tools.setdefault(name, {})
        if backend_key in backends and not replace:
            raise ToolConfigurationError(
                f"tool {name!r} already has backend {backend_key!r}"
            )
        self._specs.setdefault(name, tool.spec)
        backends[backend_key] = tool

    def get(self, name: str, *, backend: str = "local") -> SpatialTool:
        name_key = self._key(name)
        backend_key = self._key(backend)
        try:
            return self._tools[name_key][backend_key]
        except KeyError as exc:
            available = sorted(self._tools.get(name_key, {}))
            raise ToolConfigurationError(
                f"tool {name_key!r} has no backend {backend_key!r}; available={available}"
            ) from exc

    def spec(self, name: str) -> ToolSpec:
        try:
            return self._specs[self._key(name)]
        except KeyError as exc:
            raise ToolConfigurationError(f"unknown tool: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def backends(self, name: str) -> tuple[str, ...]:
        return tuple(sorted(self._tools.get(self._key(name), {})))

    def definitions(
        self, names: Iterable[str] | None = None
    ) -> tuple[ToolDefinition, ...]:
        selected = self.names() if names is None else tuple(names)
        return tuple(self.spec(name).as_model_definition() for name in selected)


__all__ = ["ToolRegistry"]
