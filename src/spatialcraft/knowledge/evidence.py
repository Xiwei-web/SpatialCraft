"""Semantic evidence views for knowledge prompts, separate from replay/scoring logs.

These functions never modify schemas or journals. Numeric geometry, tool arguments,
coordinate conventions, validity and artifact dependencies are retained; provider
requests, raw generations, token-level scores and incidental identities are not.
This is evidence selection, not an automatic context-window compression policy.
"""

from __future__ import annotations

import json
from enum import Enum

_AUDIT_KEYS = {
    "raw_response",
    "raw",
    "audit",
    "model_response",
    "generated_text",
    "action_target",
    "input_ids",
    "output_ids",
    "attention_mask",
    "raw_text",
    "raw_generation",
    "raw_request",
    "request",
    "request_payload",
    "request_json",
    "model_request",
    "provider_request",
    "generation_request",
    "response_metadata",
    "usage",
    "usage_metadata",
    "token_logprobs",
    "logprobs",
    "action_likelihood",
    "fixed_action_scoring",
    "scoring_context",
    "generated_tokens",
    "token_count",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "action_id",
    "call_id",
    "tool_call_id",
    "result_id",
    "transition_id",
    "state_id",
    "message_id",
    "evidence_id",
    "request_id",
    "response_id",
    "created_at",
    "updated_at",
    "started_at",
    "finished_at",
    "timestamp",
    "latency_ms",
    "sha256",
    "content_hash",
    "random_seed",
    "task_fingerprint",
    "source_tool_result_id",
}


def strip_audit(value):
    """Return fresh JSON-compatible evidence, preserving spatial IDs and arrays."""
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            key: strip_audit(item)
            for key, item in value.items()
            if key not in _AUDIT_KEYS and not key.endswith("token_ids")
        }
    if isinstance(value, (list, tuple)):
        return [strip_audit(item) for item in value]
    return value


def render_action(action):
    """The executed action and meaningful arguments, never its raw model response."""
    result = {"action_type": action.action_type.value}
    if action.tool_calls:
        result["tool_calls"] = [
            {"tool_name": call.tool_name, "arguments": strip_audit(call.arguments)}
            for call in action.tool_calls
        ]
    if action.final_answer is not None:
        result["final_answer"] = action.final_answer
    if action.reasoning_summary:
        result["reasoning_summary"] = action.reasoning_summary
    return result


def render_tool_result(result):
    """Keep observations and their spatial/artifact contracts, without run audit."""
    value = {
        "tool_name": result.tool_name,
        "status": result.status.value,
        "text": result.text,
        "structured_output": strip_audit(result.structured_output),
        "artifacts": [strip_audit(item) for item in result.artifacts],
        "coordinate_frames": [strip_audit(item) for item in result.coordinate_frames],
    }
    if result.error_type or result.error_message:
        value["error_type"] = result.error_type
        value["error_message"] = result.error_message
    if result.metadata:
        metadata = strip_audit(result.metadata)
        if metadata:
            value["metadata"] = metadata
    return value


def render_transition(transition):
    return {
        "step_index": transition.step_index,
        "action": render_action(transition.action),
        "tool_results": [render_tool_result(item) for item in transition.tool_results],
    }


def render_public_task(task):
    """Only execution-visible task semantics; offline labels must be explicit."""
    public = task.without_reference_answer()
    return {
        "dataset": public.dataset,
        "question": public.question,
        "choices": list(public.choices),
        "answer_type": public.answer_type.value,
        "context": strip_audit(public.metadata),
    }


def render_state_evidence(state):
    """Existing tool observations and injected knowledge at a segment boundary.

    Callers select relevant prior results/media separately when resolving artifact
    dependencies; this entry-state view does not assign credit to earlier actions.
    """
    observations = []
    for message in state.messages:
        if message.role.value != "tool":
            continue
        try:
            content = strip_audit(json.loads(message.content))
        except (ValueError, TypeError):
            content = message.content
        observations.append(
            {
                "tool_name": message.metadata.get("tool_name"),
                "observation": content,
            }
        )
    spatial_keys = {
        "objects",
        "entities",
        "coordinate_frames",
        "frames",
        "scale_status",
        "length_unit",
        "unit",
        "validity",
        "valid",
        "warnings",
    }
    return {
        "step_index": state.step_index,
        "tool_observations": observations,
        "spatial_evidence": [strip_audit(item) for item in state.evidence],
        "injected_experiences": [
            {
                "reference": f"{item.experience_id}@{item.version}",
                "text": item.prompt_text,
            }
            for item in state.retrieved_experiences
        ],
        "spatial_metadata": strip_audit(
            {key: value for key, value in state.metadata.items() if key in spatial_keys}
        ),
    }
