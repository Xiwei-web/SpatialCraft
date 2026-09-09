"""Small JSON-Schema builders and deterministic argument validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


class ToolSchemaError(ValueError):
    """Raised when a schema or invocation payload is invalid."""


def object_schema(
    properties: Mapping[str, Mapping[str, Any]],
    *,
    required: Sequence[str] = (),
    additional_properties: bool = False,
    description: str | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {key: dict(value) for key, value in properties.items()},
        "required": list(required),
        "additionalProperties": additional_properties,
    }
    if description:
        schema["description"] = description
    validate_schema(schema)
    return schema


def array_schema(
    items: Mapping[str, Any],
    *,
    min_items: int | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": dict(items)}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def enum_schema(*values: str, description: str | None = None) -> dict[str, Any]:
    if not values or any(not value for value in values):
        raise ValueError("enum schemas require non-empty string values")
    schema: dict[str, Any] = {"type": "string", "enum": list(values)}
    if description:
        schema["description"] = description
    return schema


def validate_schema(schema: Mapping[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ToolSchemaError("jsonschema is required for tool validation") from exc
    try:
        Draft202012Validator.check_schema(dict(schema))
    except Exception as exc:
        raise ToolSchemaError(f"invalid JSON Schema: {exc}") from exc


def validate_arguments(
    arguments: Mapping[str, Any], schema: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a detached JSON object after full Draft 2020-12 validation."""

    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ToolSchemaError("jsonschema is required for tool validation") from exc
    try:
        detached = json.loads(json.dumps(dict(arguments), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ToolSchemaError(f"tool arguments are not valid JSON: {exc}") from exc
    errors = sorted(
        Draft202012Validator(dict(schema)).iter_errors(detached),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise ToolSchemaError(f"invalid tool arguments at {path}: {error.message}")
    return detached


__all__ = [
    "ToolSchemaError",
    "array_schema",
    "enum_schema",
    "object_schema",
    "validate_arguments",
    "validate_schema",
]
