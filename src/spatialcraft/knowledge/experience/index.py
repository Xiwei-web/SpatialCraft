"""Portable cosine index for Experience retrieval."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from spatialcraft.models import ModelProvider

from .bank import ExperienceBank


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic dependency-light fallback for tests and cold starts."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions < 8:
            raise ValueError("embedding dimensions must be at least 8")
        self.dimensions = dimensions

    @property
    def identity(self):
        return f"hashing:{self.dimensions}:v1"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.casefold()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, index] += sign
        return _normalize(matrix)


class ModelEmbedder:
    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider
        self.dimensions = int(getattr(provider, "dimensions", 0))
        self.identity = getattr(provider, "identity", None)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        values = np.asarray(self.provider.embed(tuple(texts)), dtype=np.float32)
        if (
            values.ndim != 2
            or len(values) != len(texts)
            or not np.isfinite(values).all()
        ):
            raise ValueError("Invalid embedding matrix")
        return _normalize(values)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class ExperienceIndex:
    def __init__(
        self,
        references: tuple[str, ...],
        vectors: np.ndarray,
        *,
        frozen: bool = False,
        embedding_identity: str | None = None,
    ) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(references) != len(vectors):
            raise ValueError("experience references and vectors must align")
        if len(references) != len(set(references)):
            raise ValueError("experience index references must be unique")
        self.references = tuple(references)
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite embedding index")
        self.embedding_identity = embedding_identity
        self.vectors = _normalize(vectors) if len(vectors) else vectors.copy()
        self.frozen = bool(frozen)
        if self.frozen:
            self.vectors.setflags(write=False)

    def freeze(self) -> ExperienceIndex:
        return ExperienceIndex(
            self.references,
            self.vectors.copy(),
            frozen=True,
            embedding_identity=self.embedding_identity,
        )

    @classmethod
    def build(cls, bank: ExperienceBank, embedder: Embedder) -> ExperienceIndex:
        items = bank.active()
        vectors = embedder.embed([item.prompt_text for item in items])
        return cls(
            tuple(item.reference for item in items),
            vectors,
            embedding_identity=getattr(embedder, "identity", None),
        )

    def search(
        self, query: str, embedder: Embedder, *, top_k: int = 5
    ) -> tuple[tuple[str, float], ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.references:
            return ()
        if self.embedding_identity != getattr(embedder, "identity", None):
            raise ValueError("Embedding model/space changed; rebuild the index")
        query_vector = _normalize(embedder.embed([query]))[0]
        scores = self.vectors @ query_vector
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return tuple(
            (self.references[index], float(scores[index])) for index in indices
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": "1.0",
            "embedding_identity": self.embedding_identity,
            "references": list(self.references),
            "vectors": self.vectors.tolist(),
            "dimensions": self.vectors.shape[1] if self.vectors.ndim == 2 else 0,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ExperienceIndex:
        if value.get("schema_version") != "1.0":
            raise ValueError("unsupported Experience Index schema version")
        references = tuple(value.get("references", ()))
        vectors = np.asarray(value.get("vectors", ()), dtype=np.float32)
        if not references and vectors.size == 0:
            vectors = np.empty((0, int(value.get("dimensions", 0))), dtype=np.float32)
        return cls(
            references, vectors, embedding_identity=value.get("embedding_identity")
        )


__all__ = ["Embedder", "ExperienceIndex", "HashingEmbedder", "ModelEmbedder"]
