"""Lazy embeddings-only OpenAI client with validated, persistent vector caching.

No key is read and no client/network call is made at construction. Cache files
contain vectors and content hashes, never credentials or original input text.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from math import isfinite, sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4

from spatialcraft.storage.atomic_io import (
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_bytes,
)

from ..capabilities import Capability
from ..interfaces import (
    ModelCapabilityError,
    ModelConfigurationError,
    ModelProvider,
    ModelRequestError,
)
from ..registry import ModelConfig, ProviderKind


class OpenAIEmbeddingsProvider(ModelProvider):
    def __init__(
        self,
        config: ModelConfig,
        *,
        client: Any = None,
        cache_dir: str | Path | None = None,
        token_counter: Callable[[str], int] | None = None,
    ):
        if config.provider is not ProviderKind.OPENAI_EMBEDDINGS:
            raise ModelConfigurationError("Expected openai_embeddings configuration")
        config.capabilities.require(Capability.EMBEDDINGS)
        self.config, self._client = config, client
        self.dimensions = int(config.metadata.get("dimensions", 1536))
        if not 1 <= self.dimensions <= 1536:
            raise ModelConfigurationError(
                "text-embedding-3-small dimensions must be 1..1536"
            )
        if config.model_id != "text-embedding-3-small":
            raise ModelConfigurationError(
                "This experiment requires text-embedding-3-small"
            )
        assert config.api is not None
        self.identity = f"{config.model_id}:{self.dimensions}:l2:v1:{config.api.resolved_base_url() or 'https://api.openai.com/v1'}"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._memory: dict[str, tuple[float, ...]] = {}
        self._token_counter = token_counter
        self.usage = {"requests": 0, "input_tokens": 0, "cache_hits": 0}

    def generate(self, request):
        raise ModelCapabilityError("Embedding models cannot generate text")

    def _count(self, text: str) -> int:
        if self._token_counter is not None:
            return self._token_counter(text)
        import tiktoken

        return len(
            tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=())
        )

    def _make_client(self):
        if self._client is None:
            from openai import OpenAI

            assert self.config.api is not None
            self._client = OpenAI(
                api_key=self.config.api.api_key(),
                base_url=self.config.api.resolved_base_url(),
                timeout=self.config.api.timeout_seconds,
                max_retries=self.config.api.max_retries,
            )
        return self._client

    def _key(self, text: str) -> str:
        return sha256_bytes(
            canonical_json_bytes({"identity": self.identity, "text": text})
        )

    def _validate(self, vector) -> tuple[float, ...]:
        result = tuple(float(v) for v in vector)
        if len(result) != self.dimensions or not all(map(isfinite, result)):
            raise ModelRequestError(
                "Embedding response has invalid dimensions/nonfinite values"
            )
        norm = sqrt(sum(v * v for v in result))
        if norm <= 0:
            raise ModelRequestError("Embedding response is a zero vector")
        return result

    def _cached(self, key):
        if key in self._memory:
            return self._memory[key]
        if (
            self.cache_dir is not None
            and (path := self.cache_dir / f"{key}.json").exists()
        ):
            value = read_json(path)
            if (
                value.get("identity") != self.identity
                or value.get("input_sha256") != key
            ):
                raise ModelRequestError("Embedding cache identity mismatch")
            if (
                sha256_bytes(canonical_json_bytes(value["vector"]))
                != value["vector_sha256"]
            ):
                raise ModelRequestError("Embedding cache checksum mismatch")
            self._memory[key] = self._validate(value["vector"])
            return self._memory[key]
        return None

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input must be nonempty text")
        # Serialize cache fills across local workers; a crashed call can be retried.
        if self.cache_dir is None:
            return self._embed(texts)
        with file_lock(self.cache_dir / ".fill.lock", timeout=600):
            return self._embed(texts)

    def _embed(self, texts):
        missing: dict[str, str] = {}
        for text in texts:
            key = self._key(text)
            if self._cached(key) is None:
                missing[key] = text
            else:
                self.usage["cache_hits"] += 1
        pending = []
        for key, text in missing.items():
            tokens = self._count(text)
            if not 1 <= tokens <= 8192:
                raise ModelRequestError(
                    "Embedding input exceeds 8192 tokens; explicit decomposition required"
                )
            pending.append((key, text, tokens))
        while pending:
            batch, total = [], 0
            while pending and len(batch) < 64 and total + pending[0][2] <= 300000:
                item = pending.pop(0)
                batch.append(item)
                total += item[2]
            try:
                response = self._make_client().embeddings.create(
                    model=self.config.model_id,
                    input=[item[1] for item in batch],
                    dimensions=self.dimensions,
                    encoding_format="float",
                )
            except ModelConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001 - do not persist SDK errors containing request data
                raise ModelRequestError(
                    f"Embedding API call failed ({type(exc).__name__}); see account/network configuration"
                ) from None
            # Record successful HTTP usage separately from vector commits. A
            # crash/retry can charge twice; accounting must not pretend otherwise.
            if self.cache_dir is not None:
                atomic_write_json(
                    self.cache_dir / "usage" / f"{uuid4().hex}.json",
                    {
                        "identity": self.identity,
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                        "input_sha256": [item[0] for item in batch],
                        "input_tokens": int(response.usage.prompt_tokens),
                        "total_tokens": int(
                            getattr(
                                response.usage,
                                "total_tokens",
                                response.usage.prompt_tokens,
                            )
                        ),
                        "response_model": response.model,
                    },
                    overwrite=False,
                    mode=0o600,
                )
            if response.model != self.config.model_id:
                raise ModelRequestError("Embedding response model mismatch")
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ModelRequestError(
                    "Embedding response indices are incomplete or duplicated"
                )
            values = [self._validate(item.embedding) for item in ordered]
            # Validate the whole response before committing any vector.
            for (key, _, _), vector in zip(batch, values, strict=True):
                norm = sqrt(sum(v * v for v in vector))
                vector = tuple(v / norm for v in vector)
                if self.cache_dir is not None:
                    atomic_write_json(
                        self.cache_dir / f"{key}.json",
                        {
                            "identity": self.identity,
                            "input_sha256": key,
                            "vector": vector,
                            "vector_sha256": sha256_bytes(canonical_json_bytes(vector)),
                        },
                        overwrite=False,
                        mode=0o600,
                    )
                self._memory[key] = vector
            self.usage["requests"] += 1
            self.usage["input_tokens"] += int(response.usage.prompt_tokens)
        return tuple(self._memory[self._key(text)] for text in texts)
