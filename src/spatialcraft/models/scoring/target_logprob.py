"""Strict fixed-action scoring with explicit action-token audit traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256

from spatialcraft.models.interfaces import (
    MessageRole,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    SequenceScore,
)
from spatialcraft.schemas import ActionType, AgentAction, LogProbTrace, SkillItem


class ScoringMode(str, Enum):
    STRICT_TEACHER_FORCED = "strict_teacher_forced"
    SURROGATE = "surrogate"


def serialize_action(action: AgentAction) -> str:
    """Canonicalize the exact historical decision used as the scoring target."""

    if action.action_type is ActionType.TOOL:
        payload = {
            "type": "tool",
            "tool_calls": [
                {"name": call.tool_name, "arguments": call.arguments}
                for call in action.tool_calls
            ],
        }
    elif action.action_type is ActionType.FINAL:
        payload = {"type": "final", "answer": action.final_answer}
    else:
        payload = {"type": "noop"}
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def with_skill_prompt(request: ModelRequest, skill: SkillItem | None) -> ModelRequest:
    """Return the same context with exactly one marked skill slot changed."""

    messages = tuple(
        message
        for message in request.messages
        if not bool(message.metadata.get("ppo_skill_slot"))
    )
    content = skill.format_for_prompt() if skill is not None else "NONE"
    slot = ModelMessage.text(
        MessageRole.DEVELOPER,
        "Active procedural skill:\n" + content,
        metadata={
            "ppo_skill_slot": True,
            "skill_reference": skill.reference if skill else None,
        },
    )
    insertion = next(
        (
            index
            for index, message in enumerate(messages)
            if message.role is MessageRole.USER
        ),
        len(messages),
    )
    messages = (*messages[:insertion], slot, *messages[insertion:])
    return replace(
        request,
        messages=messages,
        metadata={
            **request.metadata,
            "ppo_scoring_mode": ScoringMode.STRICT_TEACHER_FORCED.value,
        },
    )


def request_fingerprint(request: ModelRequest) -> str:
    from spatialcraft.models.serialization import request_to_dict

    payload = request_to_dict(request, identity=False)
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class TargetLogprobScorer:
    """Adapter that refuses generated-text or prompt-token approximations."""

    mode = ScoringMode.STRICT_TEACHER_FORCED

    def __init__(self, provider: ModelProvider, *, model_alias: str) -> None:
        self.provider = provider
        self.model_alias = model_alias

    def score(self, request: ModelRequest, target_text: str) -> LogProbTrace:
        result: SequenceScore = self.provider.score(request, target_text)
        if result.target_text != target_text:
            raise ValueError("fixed-target scorer changed the target text")
        return LogProbTrace(
            model_alias=self.model_alias,
            target_text=target_text,
            token_ids=result.token_ids,
            token_logprobs=result.token_logprobs,
            prompt_token_count=result.prompt_token_count,
            tokenizer_id=result.model,
            metadata={
                "scoring_mode": self.mode.value,
                "prompt_tokens_excluded": True,
                "request_fingerprint": request_fingerprint(request),
                "thinking_score_mode": request.metadata.get(
                    "thinking_score_mode", "disabled"
                ),
                "fixed_thinking_prefix_sha256": sha256(
                    request.metadata["fixed_scoring_prefix"].encode("utf-8")
                ).hexdigest()
                if "fixed_scoring_prefix" in request.metadata
                else None,
                "thinking_tokens_excluded": "fixed_scoring_prefix" in request.metadata,
            },
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SurrogateScore:
    """Explicitly non-PPO score for providers without teacher forcing."""

    model_alias: str
    candidate_id: str
    score: float
    reason: str
    mode: ScoringMode = ScoringMode.SURROGATE


__all__ = [
    "ScoringMode",
    "SurrogateScore",
    "TargetLogprobScorer",
    "request_fingerprint",
    "serialize_action",
    "with_skill_prompt",
]
