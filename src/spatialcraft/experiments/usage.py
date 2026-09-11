"""Actual invocation accounting, separate from immutable logical result caches."""

from dataclasses import asdict
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from spatialcraft.models import ModelProvider, SequenceScore
from spatialcraft.models.serialization import (
    request_to_dict,
    response_from_dict,
    response_to_dict,
)
from spatialcraft.storage.atomic_io import atomic_write_json

from .journal import digest


def _elapsed(started):
    elapsed = (perf_counter() - started) * 1000
    return {"latency_ms": elapsed, "full_call_latency_ms": elapsed}


class UsageLedger:
    def __init__(self, root):
        self.root = root

    def record(self, event):
        atomic_write_json(
            self.root / "usage" / (uuid4().hex + ".json"),
            {
                "schema_version": 2,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **event,
            },
            overwrite=False,
            mode=0o600,
        )


class AuditedProviderV2(ModelProvider):
    def __init__(self, provider, journal, role):
        self.provider, self.journal, self.role = provider, journal, role
        self.ledger = UsageLedger(journal.root)

    def _event(self, request, kind):
        return (
            {
                key: request.metadata.get(key)
                for key in (
                    "phase",
                    "task_id",
                    "snapshot_id",
                    "step_index",
                    "operation",
                    "retry_index",
                    "requested_reasoning_mode",
                    "effective_reasoning_mode",
                )
            }
            | {
                key: request.metadata[key]
                for key in ("rollout_index", "rollout_prefix", "trajectory_id")
                if key in request.metadata
            }
            | {
                "kind": kind,
                "role": self.role,
                "model": request.model_alias,
                **(
                    {"phase": "accumulation_learning", "operation": "skill.ppo_score"}
                    if kind == "fixed_target_score"
                    else {}
                ),
                "request_sha256": digest(request_to_dict(request, identity=False)),
                "requested_max_output_tokens": request.settings.max_output_tokens
                if kind == "generation"
                else None,
                "requested_temperature": request.settings.temperature
                if kind == "generation"
                else None,
                "requested_top_p": request.settings.top_p
                if kind == "generation"
                else None,
                **(
                    self.provider.parameter_audit(request)
                    if kind == "generation"
                    and callable(getattr(self.provider, "parameter_audit", None))
                    else {}
                ),
                "template_sha256": request.metadata.get("template_sha256"),
                "client_retries": 0,
            }
        )

    def generate(self, request):
        inputs = {"request": request_to_dict(request, identity=False)}

        def invoke():
            started = perf_counter()
            event = self._event(request, "generation")
            try:
                response = self.provider.generate(request)
            except Exception as exc:
                self.ledger.record(
                    {
                        **event,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "actual_call": True,
                        "cache_reused": False,
                        "input_tokens": None,
                        "output_tokens": None,
                        "total_tokens": None,
                        **_elapsed(started),
                    }
                )
                raise
            self.ledger.record(
                {
                    **event,
                    "status": "completed",
                    "actual_call": True,
                    "cache_reused": False,
                    **asdict(response.usage),
                    "response_id": response.response_id,
                    "finish_reason": response.finish_reason,
                    "provider_latency_ms": response.latency_ms,
                    "server_model": response.raw.get("model")
                    if isinstance(response.raw, dict)
                    else None,
                    "server_model_version": (
                        response.raw.get(
                            "model_version", response.raw.get("modelVersion")
                        )
                        if isinstance(response.raw, dict)
                        else None
                    ),
                    "server_system_fingerprint": response.raw.get("system_fingerprint")
                    if isinstance(response.raw, dict)
                    else None,
                    **_elapsed(started),
                }
            )
            return response_to_dict(response)

        result, cached = self.journal.execute(
            "model_calls/" + digest(inputs), inputs, invoke
        )
        if cached:
            self.ledger.record(
                {
                    **self._event(request, "generation"),
                    "status": "reused",
                    "actual_call": False,
                    "cache_reused": True,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": 0,
                    "full_call_latency_ms": 0,
                    "source_response_id": result.get("response_id"),
                }
            )
        return response_from_dict(result)

    def score(self, request, target_text):
        inputs = {
            "request": request_to_dict(request, identity=False),
            "target": target_text,
        }

        def invoke():
            started = perf_counter()
            event = self._event(request, "fixed_target_score")
            try:
                response = self.provider.score(request, target_text)
            except Exception as exc:
                self.ledger.record(
                    {
                        **event,
                        "status": "failed",
                        "actual_call": True,
                        "cache_reused": False,
                        "error_type": type(exc).__name__,
                        "input_tokens": None,
                        "target_tokens": None,
                        "output_tokens": 0,
                        **_elapsed(started),
                    }
                )
                raise
            self.ledger.record(
                {
                    **event,
                    "status": "completed",
                    "actual_call": True,
                    "cache_reused": False,
                    "input_tokens": response.prompt_token_count,
                    "target_tokens": len(response.token_ids),
                    "output_tokens": 0,
                    **_elapsed(started),
                }
            )
            return asdict(response)

        result, cached = self.journal.execute(
            "ppo_calls/" + digest(inputs), inputs, invoke
        )
        if cached:
            self.ledger.record(
                {
                    **self._event(request, "fixed_target_score"),
                    "status": "reused",
                    "actual_call": False,
                    "cache_reused": True,
                    "input_tokens": 0,
                    "target_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "full_call_latency_ms": 0,
                }
            )
        return SequenceScore(**result)


class ScopedEmbedder:
    def __init__(self, embedder, journal, scope):
        self.embedder, self.scope = embedder, scope
        self.identity, self.dimensions = embedder.identity, embedder.dimensions
        self.ledger = UsageLedger(journal.root)

    def embed(self, texts):
        provider = self.embedder.provider
        before = dict(provider.usage)
        scope = dict(self.scope())
        scope["operation"] = scope.get(
            "embedding_operation", scope.get("operation", "embedding")
        )
        started = perf_counter()
        try:
            values = self.embedder.embed(texts)
        except Exception as exc:
            self.ledger.record(
                {
                    **scope,
                    "kind": "embedding",
                    "role": "embedding",
                    "model": self.identity,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "input_tokens": None,
                    "requests": None,
                    **_elapsed(started),
                }
            )
            raise
        after = provider.usage
        self.ledger.record(
            {
                **scope,
                "kind": "embedding",
                "role": "embedding",
                "model": self.identity,
                "status": "completed",
                "input_tokens": after["input_tokens"] - before["input_tokens"],
                "requests": after["requests"] - before["requests"],
                "cache_hits": after["cache_hits"] - before["cache_hits"],
                "texts": len(texts),
                **_elapsed(started),
            }
        )
        return values


def cost_report(root, deployment_count=None):
    """Build the versioned report from actual invocation records."""
    from spatialcraft.evaluation.cost_report_v2 import build_cost_report

    return build_cost_report(root, deployment_count)
