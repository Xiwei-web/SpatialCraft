"""Client for an OpenAI-compatible vLLM serving process."""

from __future__ import annotations

from typing import Any

from ..interfaces import ModelConfigurationError, ModelRequestError
from ..registry import ModelConfig, ProviderKind
from .openai_chat_compatible import OpenAIChatCompatibleProvider


class VLLMClientProvider(OpenAIChatCompatibleProvider):
    """Use vLLM's HTTP server without loading model weights in the agent process."""

    provider_name = ProviderKind.VLLM_CLIENT.value
    accepted_provider_kinds = frozenset({ProviderKind.VLLM_CLIENT})

    def __init__(self, config: ModelConfig, *, client: Any | None = None) -> None:
        super().__init__(config, client=client)
        assert config.api is not None
        if client is None and not config.api.resolved_base_url():
            raise ModelConfigurationError("vllm_client requires api.base_url")

    def _api_key_required(self) -> bool:
        return False

    def _default_api_key(self) -> str:
        return "EMPTY"

    def test_connection(self) -> tuple[str, ...]:
        """Return model ids advertised by the configured vLLM endpoint."""

        try:
            response = self._make_client().models.list()
        except Exception as exc:
            raise ModelRequestError(
                f"cannot reach vLLM endpoint for {self.config.alias}: {exc}"
            ) from exc
        data = getattr(response, "data", [])
        return tuple(str(getattr(item, "id", item)) for item in data)


__all__ = ["VLLMClientProvider"]
