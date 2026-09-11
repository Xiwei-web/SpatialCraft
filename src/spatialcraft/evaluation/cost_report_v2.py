"""Aggregate actual invocation ledgers while retaining incomplete coverage."""

from collections import defaultdict
from math import isfinite
from pathlib import Path

from spatialcraft.storage.atomic_io import read_json

# Schema 3 totals describe arithmetic token work, not equivalent money or FLOPs.
KIND_TOKEN_FIELDS = {
    "generation_input_tokens": ("generation", "input_tokens"),
    "generation_output_tokens": ("generation", "output_tokens"),
    "generation_total_tokens": ("generation", "total_tokens"),
    "embedding_input_tokens": ("embedding", "input_tokens"),
    "scoring_prefix_tokens": ("fixed_target_score", "input_tokens"),
    "scoring_target_tokens": ("fixed_target_score", "target_tokens"),
}
FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_input_tokens",
    "target_tokens",
    *KIND_TOKEN_FIELDS,
    "latency_ms",
    "full_call_latency_ms",
    "provider_latency_ms",
)


def _applicable(event, field):
    kind = event.get("kind")
    if field in KIND_TOKEN_FIELDS:
        return kind == KIND_TOKEN_FIELDS[field][0]
    if field == "provider_latency_ms":
        return kind == "generation"
    if field in {"latency_ms", "full_call_latency_ms", "count"}:
        return True
    if str(kind).startswith("tool"):
        return False
    if field == "total_tokens":
        return kind in {"generation", "embedding", "fixed_target_score"}
    if kind == "embedding" and field in {
        "output_tokens",
        "reasoning_tokens",
        "cached_input_tokens",
        "target_tokens",
    }:
        return False
    if kind == "fixed_target_score" and field in {
        "reasoning_tokens",
        "cached_input_tokens",
    }:
        return False
    if field == "target_tokens":
        return kind == "fixed_target_score"
    return True


def _event_value(event, field):
    if field in KIND_TOKEN_FIELDS:
        field = KIND_TOKEN_FIELDS[field][1]
    if field != "total_tokens":
        return event.get(field)
    kind = event.get("kind")
    if kind == "embedding":
        return event.get("input_tokens")
    if kind == "generation" and event.get("total_tokens") is not None:
        return event["total_tokens"]
    parts = (
        "input_tokens",
        "target_tokens" if kind == "fixed_target_score" else "output_tokens",
    )
    values = [event.get(k) for k in parts]
    if any(v is None for v in values):
        return None
    if any(type(v) not in (int, float) or not isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid constituent token count")
    return sum(values)


def _measure(events, field):
    known, unknown, not_applicable = [], 0, 0
    for event in events:
        if not _applicable(event, field):
            not_applicable += 1
            continue
        # A logical cache reuse incurs zero incremental token cost; source
        # response usage must not be charged again, even if present in old logs.
        value = 0 if event.get("cache_reused") else _event_value(event, field)
        if value is None:
            unknown += 1
        elif not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
            raise ValueError(f"Invalid ledger {field}: {value!r}")
        else:
            known.append(value)
    observed = sum(known)
    return {
        "known_sum": observed,
        "known_records": len(known),
        "unknown_records": unknown,
        "not_applicable_records": not_applicable,
        "total": None if unknown else observed,
        "coverage_complete": unknown == 0,
    }


CALL_KINDS = {
    "all_calls": None,
    "model_calls": {"generation", "fixed_target_score"},
    "embedding_requests": {"embedding"},
    "tool_calls": {"tool"},
}


def _call_measure(events, kinds=None):
    normalized = []
    for event in events:
        kind = str(event.get("kind") or "unknown")
        if (
            kinds is not None
            and kind not in kinds
            and not ("tool" in kinds and kind.startswith("tool"))
        ):
            continue
        if kind == "embedding":
            value = event.get("requests")
        else:
            actual = event.get("actual_call")
            value = int(actual) if type(actual) is bool else None
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Invalid actual invocation count")
        normalized.append({"count": value, "cache_reused": event.get("cache_reused")})
    result = _measure(normalized, "count")
    result["not_applicable_records"] = len(events) - len(normalized)
    return result


def _summary(events):
    calls = _call_measure(events)
    return {
        "invocations": len(events),
        "actual_calls": calls["known_sum"],
        "actual_call_unknown_records": calls["unknown_records"],
        "actual_calls_total": calls["total"],
        "cache_reuses": sum(bool(e.get("cache_reused")) for e in events),
        "embedding_cache_hits": sum(e.get("cache_hits", 0) or 0 for e in events),
        "failures": sum(e.get("status") == "failed" for e in events),
        **{field: _measure(events, field) for field in FIELDS},
        "call_counts": {
            key: _call_measure(events, kinds) for key, kinds in CALL_KINDS.items()
        },
    }


def _per_task_measure(value, denominator, unattributed_records=0):
    complete = bool(
        denominator and value["coverage_complete"] and unattributed_records == 0
    )
    return {
        "known_sum_per_task": value["known_sum"] / denominator
        if denominator and unattributed_records == 0
        else None,
        "total": value["total"] / denominator if complete else None,
        "known_records": value["known_records"],
        "unknown_records": value["unknown_records"],
        "not_applicable_records": value["not_applicable_records"],
        "unattributed_records": unattributed_records,
        "coverage_complete": complete,
    }


def _normalize(summary, denominator, unattributed_records=0):
    return {
        "denominator": denominator,
        **{
            field: _per_task_measure(summary[field], denominator, unattributed_records)
            for field in FIELDS
        },
        "call_counts": {
            key: _per_task_measure(value, denominator, unattributed_records)
            for key, value in summary["call_counts"].items()
        },
    }


def _combined_measure(offline, online):
    known = (offline["known_sum_per_task"], online["known_sum_per_task"])
    complete = offline["coverage_complete"] and online["coverage_complete"]
    return {
        "known_sum_per_task": sum(known) if all(v is not None for v in known) else None,
        "total": offline["total"] + online["total"] if complete else None,
        "offline_unknown_records": offline["unknown_records"],
        "online_unknown_records": online["unknown_records"],
        "online_unattributed_records": online["unattributed_records"],
        "coverage_complete": complete,
    }


def _combined(offline, online):
    return {
        "offline_denominator": offline["denominator"],
        "online_denominator": online["denominator"],
        **{field: _combined_measure(offline[field], online[field]) for field in FIELDS},
        "call_counts": {
            key: _combined_measure(
                offline["call_counts"][key], online["call_counts"][key]
            )
            for key in CALL_KINDS
        },
    }


def _task_rows(events):
    groups = defaultdict(list)
    for event in events:
        groups[(event.get("phase") or "unknown", event.get("task_id"))].append(event)
    rows = []
    for (phase, task_id), values in sorted(
        groups.items(), key=lambda item: str(item[0])
    ):
        refs = {
            (e.get("rollout_prefix"), e.get("rollout_index"), e.get("trajectory_id"))
            for e in values
            if any(
                e.get(key) is not None
                for key in ("rollout_prefix", "rollout_index", "trajectory_id")
            )
        }
        rows.append(
            {
                "phase": phase,
                "task_id": task_id,
                "rollout_refs": [
                    {
                        "rollout_prefix": prefix,
                        "rollout_index": index,
                        "trajectory_id": trajectory,
                    }
                    for prefix, index, trajectory in sorted(refs, key=str)
                ],
                "records_without_rollout_identity": sum(
                    all(
                        e.get(key) is None
                        for key in ("rollout_prefix", "rollout_index", "trajectory_id")
                    )
                    for e in values
                ),
                **_summary(values),
            }
        )
    return rows


def build_cost_report(root, deployment_count=None):
    """Count physical events once; M amortizes offline cost, never observed online cost."""
    if deployment_count is not None and (
        type(deployment_count) is not int or deployment_count < 0
    ):
        raise ValueError("deployment_count must be a nonnegative integer or unknown")
    groups, all_events = defaultdict(list), []
    for path in sorted((Path(root) / "usage").glob("*.json")):
        event = read_json(path)
        if event.get("schema_version") != 2:
            raise ValueError(f"Unsupported usage ledger schema: {path}")
        key = (
            event.get("phase") or "unknown",
            event.get("kind"),
            event.get("role"),
            event.get("model"),
            event.get("operation"),
        )
        groups[key].append(event)
        all_events.append(event)
    rows = [
        {
            "phase": phase,
            "kind": kind,
            "role": role,
            "model": model,
            "operation": operation,
            **_summary(events),
        }
        for (phase, kind, role, model, operation), events in sorted(
            groups.items(), key=lambda item: str(item[0])
        )
    ]
    offline = [
        e for e in all_events if str(e.get("phase") or "").startswith("accumulation")
    ]
    online = [
        e for e in all_events if str(e.get("phase") or "").startswith("deployment")
    ]
    online_task_ids = {e["task_id"] for e in online if e.get("task_id") is not None}
    unattributed_online = sum(e.get("task_id") is None for e in online)
    offline_summary, online_summary = _summary(offline), _summary(online)
    amortized = _normalize(offline_summary, deployment_count)
    online_mean = _normalize(online_summary, len(online_task_ids), unattributed_online)
    return {
        "schema_version": 3,
        "groups": rows,
        "per_task": _task_rows(all_events),
        "deployment_count": deployment_count,
        "record_count": len(all_events),
        "offline_accumulation": offline_summary,
        "online_deployment": online_summary,
        "offline_amortized_per_deployment_task": {
            field: amortized[field]["total"] for field in FIELDS
        },
        "offline_amortized_coverage_per_deployment_task": amortized,
        "online_mean_per_deployment_task": online_mean,
        "amortized_total_per_deployment_task": _combined(amortized, online_mean),
        "deployment_task_attribution": {
            "observed_task_count": len(online_task_ids),
            "attributed_records": len(online) - unattributed_online,
            "unattributed_records": unattributed_online,
            "coverage_complete": bool(online_task_ids) and unattributed_online == 0,
        },
        "accumulation_latency_ms": offline_summary["latency_ms"]["total"],
        "offline_latency_ms_amortized_per_deployment_task": amortized["latency_ms"][
            "total"
        ],
        "token_source": "provider_or_executor_tokenizer_only",
        "prices_included": False,
        "gpu_resource_cost_included": False,
        "total_tokens_definition": "generation_total_tokens + embedding_input_tokens + scoring_prefix_tokens + scoring_target_tokens; reasoning tokens are already part of generation output",
        "token_work_is_equivalent_cost": False,
        "scoring_prefix_definition": "teacher-forced prompt including fixed generated prefix; target counted separately",
        "latency_definition": "sum of recorded invocation latency; historical records may have provider-only scope; not task or parallel wall time",
        "full_call_latency_definition": "wrapper elapsed time including preprocessing, tokenization and decoding, excluding journal bookkeeping; unavailable for historical records without the explicit field",
        "provider_latency_definition": "generation provider's own reported internal latency, when available",
        "online_mean_definition": "all_observed_online_cost / distinct_observed_deployment_task_ids; null if any event lacks task_id; zero-invocation tasks are not inferred",
        "amortization_definition": "offline_cost / deployment_count + observed_mean_online_cost",
        "rollout_allocation": "task events counted once; shared calls are not duplicated or allocated across rollouts",
        "unknown_usage_policy": "null total with known partial sum and coverage counts",
    }
