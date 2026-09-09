"""Shared, dependency-free helpers for SpatialCraft schema objects."""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints
from uuid import uuid4

from typing_extensions import Self

SchemaT = TypeVar("SchemaT", bound="SchemaMixin")


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(timezone.utc)


def normalize_datetime(value: datetime) -> datetime:
    """Normalize timestamps to aware UTC datetimes."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def new_id(prefix: str) -> str:
    """Create a stable opaque identifier with a human-readable prefix."""

    return f"{prefix}_{uuid4().hex}"


def require_non_empty(value: str, field_name: str) -> str:
    """Validate and normalize a required string."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_probability(value: float, field_name: str) -> float:
    """Validate a probability or confidence value."""

    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [0, 1], got {value}")
    return value


def _encode(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return normalize_datetime(value).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {
            field.name: _encode(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_encode(item) for item in value]
    return value


def _decode(annotation: Any, value: Any) -> Any:
    if annotation is Any or annotation is object:
        return value
    if value is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        non_none = [arg for arg in args if arg is not type(None)]
        last_error: Exception | None = None
        for option in non_none:
            try:
                return _decode(option, value)
            except (TypeError, ValueError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return value

    if origin is list:
        if not isinstance(value, list):
            raise TypeError("Expected a list")
        item_type = args[0] if args else Any
        return [_decode(item_type, item) for item in value]
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError("Expected a list or tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item) for item in value)
        if args and len(value) != len(args):
            raise ValueError(
                f"Expected a tuple with {len(args)} items, got {len(value)}"
            )
        return tuple(_decode(arg, item) for arg, item in zip(args, value))
    if origin is dict or origin is Mapping:
        if not isinstance(value, Mapping):
            raise TypeError("Expected a mapping")
        key_type, item_type = args if len(args) == 2 else (Any, Any)
        return {
            _decode(key_type, key): _decode(item_type, item)
            for key, item in value.items()
        }

    if annotation is datetime:
        if isinstance(value, datetime):
            return normalize_datetime(value)
        return normalize_datetime(
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    if annotation is Path:
        return Path(value)
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(value)
    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected a mapping for {annotation.__name__}")
        type_hints = get_type_hints(annotation)
        field_names = {field.name for field in fields(annotation)}
        unknown = set(value) - field_names
        if unknown:
            raise ValueError(
                f"Unknown fields for {annotation.__name__}: {sorted(unknown)}"
            )
        kwargs = {
            key: _decode(type_hints.get(key, Any), item) for key, item in value.items()
        }
        return annotation(**kwargs)
    return value


class SchemaMixin:
    """JSON serialization and strict reconstruction for schema dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        encoded = _encode(self)
        if not isinstance(encoded, dict):
            raise TypeError("Schema objects must serialize to a dictionary")
        return encoded

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=False, sort_keys=True
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        decoded = _decode(cls, data)
        if not isinstance(decoded, cls):
            raise TypeError(f"Could not decode {cls.__name__}")
        return decoded

    @classmethod
    def from_json(cls, payload: str) -> Self:
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise TypeError("Schema JSON payload must contain an object")
        return cls.from_dict(data)
