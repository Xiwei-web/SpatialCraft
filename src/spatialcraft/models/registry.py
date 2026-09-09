"""YAML-backed model and role registries with lazy provider construction."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from spatialcraft.schemas._base import require_non_empty

from .capabilities import Capability, ModelCapabilities
from .interfaces import (
    GenerationSettings,
    ModelConfigurationError,
    ModelProvider,
)


class ProviderKind(str, Enum):
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_RESPONSES = "openai_responses"
    GEMINI_NATIVE = "gemini_native"
    OPENAI_CHAT_COMPATIBLE = "openai_chat_compatible"
    VLLM_CLIENT = "vllm_client"
    TRANSFORMERS_LOCAL = "transformers_local"


@dataclass(frozen=True, slots=True, kw_only=True)
class APIConfig:
    """Remote API connection settings; secrets stay in environment variables."""

    api_key_env: str | None = None
    base_url: str | None = None
    base_url_env: str | None = None
    organization_env: str | None = None
    project_env: str | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 3
    default_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "api_key_env",
            "base_url",
            "base_url_env",
            "organization_env",
            "project_env",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, require_non_empty(value, name))
        object.__setattr__(self, "default_headers", dict(self.default_headers))
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    def api_key(self, *, required: bool = True) -> str | None:
        if self.api_key_env is None:
            if required:
                raise ModelConfigurationError("api_key_env is not configured")
            return None
        value = os.environ.get(self.api_key_env)
        if required and not value:
            raise ModelConfigurationError(
                f"required API key environment variable is unset: {self.api_key_env}"
            )
        return value

    def resolved_base_url(self) -> str | None:
        if self.base_url_env and os.environ.get(self.base_url_env):
            return os.environ[self.base_url_env]
        return self.base_url


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalModelConfig:
    """Local checkpoint and loading policy."""

    path: str
    tokenizer_path: str | None = None
    revision: str | None = None
    device_map: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    load_in_4bit: bool = False
    extra_load_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", require_non_empty(self.path, "path"))
        if self.tokenizer_path is not None:
            object.__setattr__(
                self,
                "tokenizer_path",
                require_non_empty(self.tokenizer_path, "tokenizer_path"),
            )
        object.__setattr__(self, "extra_load_kwargs", dict(self.extra_load_kwargs))


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelConfig:
    """One named model deployment independent of its consuming role."""

    alias: str
    provider: ProviderKind
    model_id: str
    capabilities: ModelCapabilities
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    api: APIConfig | None = None
    local: LocalModelConfig | None = None
    enabled: bool = True
    schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("alias", "model_id", "schema_version"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        if not isinstance(self.provider, ProviderKind):
            object.__setattr__(self, "provider", ProviderKind(self.provider))
        object.__setattr__(self, "metadata", dict(self.metadata))
        remote = {
            ProviderKind.OPENAI_EMBEDDINGS,
            ProviderKind.OPENAI_RESPONSES,
            ProviderKind.GEMINI_NATIVE,
            ProviderKind.OPENAI_CHAT_COMPATIBLE,
            ProviderKind.VLLM_CLIENT,
        }
        if self.provider in remote and self.api is None:
            raise ValueError(
                f"provider {self.provider.value} requires api configuration"
            )
        if self.provider is ProviderKind.TRANSFORMERS_LOCAL and self.local is None:
            raise ValueError("transformers_local requires local configuration")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        values = dict(data)
        try:
            values["provider"] = ProviderKind(values["provider"])
            values["capabilities"] = ModelCapabilities.from_dict(
                values.get("capabilities")
            )
            values["generation"] = GenerationSettings(
                **dict(values.get("generation") or {})
            )
            if values.get("api") is not None:
                values["api"] = APIConfig(**dict(values["api"]))
            if values.get("local") is not None:
                values["local"] = LocalModelConfig(**dict(values["local"]))
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelConfigurationError(
                f"invalid model configuration: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True, kw_only=True)
class RoleConfig:
    """Bind one pipeline role to a model and role-specific inference policy."""

    role: str
    model: str
    required_capabilities: frozenset[Capability] = field(default_factory=frozenset)
    system_prompt: str | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    fallback_models: tuple[str, ...] = ()
    enabled: bool = True
    schema_version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("role", "model", "schema_version"):
            object.__setattr__(self, name, require_non_empty(getattr(self, name), name))
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(Capability(item) for item in self.required_capabilities),
        )
        object.__setattr__(self, "generation", dict(self.generation))
        object.__setattr__(self, "fallback_models", tuple(self.fallback_models))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if len(self.fallback_models) != len(set(self.fallback_models)):
            raise ValueError("fallback_models cannot contain duplicates")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoleConfig:
        values = dict(data)
        try:
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelConfigurationError(f"invalid role configuration: {exc}") from exc


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_string(value: str, environment: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        resolved = environment.get(name, default)
        if resolved is None:
            raise ModelConfigurationError(
                f"environment variable {name} is required by configuration"
            )
        return resolved

    return _ENV_PATTERN.sub(replace, value)


def expand_environment(value: Any, environment: Mapping[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` and ``${VAR:-default}`` strings."""

    environment = os.environ if environment is None else environment
    if isinstance(value, str):
        return _expand_string(value, environment)
    if isinstance(value, list):
        return [expand_environment(item, environment) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_environment(item, environment) for item in value)
    if isinstance(value, dict):
        return {
            key: expand_environment(item, environment) for key, item in value.items()
        }
    return value


def load_yaml(path: str | Path, *, expand_env: bool = True) -> dict[str, Any]:
    """Load one YAML mapping without importing PyYAML until configuration time."""

    try:
        import yaml
    except ImportError as exc:
        raise ModelConfigurationError(
            "PyYAML is required to load model configs"
        ) from exc
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
    except OSError as exc:
        raise ModelConfigurationError(
            f"cannot read configuration {path}: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise ModelConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelConfigurationError(f"configuration must be a mapping: {path}")
    return expand_environment(value) if expand_env else value


class ModelRegistry:
    """Validated alias registry with lazy backend instantiation."""

    def __init__(self, configs: Iterable[ModelConfig] = ()) -> None:
        self._configs: dict[str, ModelConfig] = {}
        for config in configs:
            self.register(config)

    def register(self, config: ModelConfig, *, replace: bool = False) -> None:
        if config.alias in self._configs and not replace:
            raise ModelConfigurationError(f"duplicate model alias: {config.alias}")
        self._configs[config.alias] = config

    def get(self, alias: str, *, require_enabled: bool = True) -> ModelConfig:
        try:
            config = self._configs[alias]
        except KeyError as exc:
            raise ModelConfigurationError(f"unknown model alias: {alias}") from exc
        if require_enabled and not config.enabled:
            raise ModelConfigurationError(f"model is disabled: {alias}")
        return config

    def aliases(self, *, enabled_only: bool = True) -> tuple[str, ...]:
        return tuple(
            sorted(
                alias
                for alias, config in self._configs.items()
                if config.enabled or not enabled_only
            )
        )

    def load_file(
        self, path: str | Path, *, replace: bool = False, expand_env: bool = True
    ) -> ModelConfig:
        config = ModelConfig.from_dict(load_yaml(path, expand_env=expand_env))
        self.register(config, replace=replace)
        return config

    def load_directory(
        self,
        directory: str | Path,
        *,
        replace: bool = False,
        expand_env: bool = True,
    ) -> tuple[ModelConfig, ...]:
        root = Path(directory)
        if not root.is_dir():
            raise ModelConfigurationError(f"model config directory not found: {root}")
        return tuple(
            self.load_file(path, replace=replace, expand_env=expand_env)
            for path in sorted((*root.glob("*.yaml"), *root.glob("*.yml")))
        )

    def create_provider(self, alias: str, **kwargs: Any) -> ModelProvider:
        """Construct the configured backend without importing unused SDKs."""

        config = self.get(alias)
        if config.provider is ProviderKind.OPENAI_EMBEDDINGS:
            from .providers.openai_embeddings import OpenAIEmbeddingsProvider

            return OpenAIEmbeddingsProvider(config, **kwargs)
        if config.provider is ProviderKind.OPENAI_RESPONSES:
            from .providers.openai_responses import OpenAIResponsesProvider

            return OpenAIResponsesProvider(config, **kwargs)
        if config.provider is ProviderKind.GEMINI_NATIVE:
            from .providers.gemini_native import GeminiNativeProvider

            return GeminiNativeProvider(config, **kwargs)
        if config.provider is ProviderKind.OPENAI_CHAT_COMPATIBLE:
            from .providers.openai_chat_compatible import OpenAIChatCompatibleProvider

            return OpenAIChatCompatibleProvider(config, **kwargs)
        if config.provider is ProviderKind.VLLM_CLIENT:
            from .providers.vllm_client import VLLMClientProvider

            return VLLMClientProvider(config, **kwargs)
        if config.provider is ProviderKind.TRANSFORMERS_LOCAL:
            from .providers.transformers_local import TransformersLocalProvider

            return TransformersLocalProvider(config, **kwargs)
        raise ModelConfigurationError(f"unsupported provider: {config.provider.value}")


class RoleRegistry:
    """Validated role-to-model bindings."""

    def __init__(self, roles: Iterable[RoleConfig] = ()) -> None:
        self._roles: dict[str, RoleConfig] = {}
        for role in roles:
            self.register(role)

    def register(self, role: RoleConfig, *, replace: bool = False) -> None:
        if role.role in self._roles and not replace:
            raise ModelConfigurationError(f"duplicate role: {role.role}")
        self._roles[role.role] = role

    def get(self, role: str) -> RoleConfig:
        try:
            return self._roles[role]
        except KeyError as exc:
            raise ModelConfigurationError(f"unknown model role: {role}") from exc

    def load_directory(
        self, directory: str | Path, *, expand_env: bool = True
    ) -> tuple[RoleConfig, ...]:
        root = Path(directory)
        if not root.is_dir():
            raise ModelConfigurationError(f"role config directory not found: {root}")
        roles = []
        for path in sorted((*root.glob("*.yaml"), *root.glob("*.yml"))):
            role = RoleConfig.from_dict(load_yaml(path, expand_env=expand_env))
            self.register(role)
            roles.append(role)
        return tuple(roles)

    def validate(self, models: ModelRegistry) -> None:
        for role in self._roles.values():
            config = models.get(role.model)
            config.capabilities.require(*role.required_capabilities)
            for fallback in role.fallback_models:
                fallback_config = models.get(fallback)
                fallback_config.capabilities.require(*role.required_capabilities)


__all__ = [
    "APIConfig",
    "LocalModelConfig",
    "ModelConfig",
    "ModelRegistry",
    "ProviderKind",
    "RoleConfig",
    "RoleRegistry",
    "expand_environment",
    "load_yaml",
]
