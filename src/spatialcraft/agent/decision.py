"""One action per environment step, with bounded generation recovery."""

from __future__ import annotations

import json
import re
from dataclasses import replace

from spatialcraft.models import MessageRole, ModelMessage, ModelResponseError
from spatialcraft.models.response_parser import parse_generated_text
from spatialcraft.models.serialization import request_to_dict, response_to_dict
from spatialcraft.schemas import ActionType
from spatialcraft.tools.schema_builder import ToolSchemaError, validate_arguments

BUDGET_POLICY = "action_recovery_v2"
RECOVERY_TOKENS = 1024
FORCED_FINAL_TOKENS = 512
CONTROLLED_FAILURES = {
    "action_recovery_failed",
    "invalid_action",
    "forced_final_failed",
}


def recovery_request(request, *, forced_final=False):
    if forced_final:
        instruction = (
            "The trajectory's tool-interaction budget is exhausted. No more tools are allowed. "
            "Use all existing images, observations and interaction history to give your best "
            "final answer now. Do not repeat reasoning. Output only 'Final Answer: ...'."
        )
    else:
        instruction = (
            "The previous generation hit its output limit without a complete valid action. "
            "Recover the CURRENT step using the existing images, tool observations and current context. "
            "Do not restart or repeat reasoning. Output exactly ONE complete "
            "legal tool call, or 'Final Answer: ...' ONLY if current evidence is sufficient. "
            "You may gather more evidence; you are not required to finish the task now."
        )
    kind = "forced_final" if forced_final else "recovery"
    return replace(
        request,
        messages=(*request.messages, ModelMessage.text(MessageRole.USER, instruction)),
        tools=() if forced_final else request.tools,
        tool_choice=None if forced_final else request.tool_choice,
        settings=replace(
            request.settings,
            max_output_tokens=min(
                request.settings.max_output_tokens or 4096,
                FORCED_FINAL_TOKENS if forced_final else RECOVERY_TOKENS,
            ),
            temperature=0.0,
            top_p=1.0,
        ),
        metadata={
            **request.metadata,
            "chat_template_kwargs": {
                **request.metadata.get("chat_template_kwargs", {}),
                "enable_thinking": False,
            },
            "rollout_budget": {"policy": BUDGET_POLICY, "call_kind": kind},
        },
    )


def parse_complete_action(parser, response, request, *, final_only=False):
    """Salvage only delimited complete actions from a truncated generation.

    Unfinished prose, JSON, thinking and tool parameters never become answers.
    Text finals at a length boundary need a completed Final Answer line (newline)
    or a closed JSON final_answer object; an unterminated last line is ambiguous.
    """
    truncated = response.finish_reason == "length"
    raw = dict(response.raw) if isinstance(response.raw, dict) else {}
    text = str(raw.get("generated_text") or response.text or "")
    thinking = (
        request.metadata.get("chat_template_kwargs", {}).get("enable_thinking") is True
    )
    if "</think>" in text:
        before, marker, after = text.partition("</think>")
        leading = after[: len(after) - len(after.lstrip())]
        raw["sampled_thinking_prefix"] = before + marker + leading
        text = after
    elif "<think>" in text or (thinking and not raw.get("sampled_thinking_prefix")):
        raise ModelResponseError("No complete action outside unfinished thinking")

    candidate = replace(response, raw=raw or response.raw)
    if not response.tool_calls:
        selected = text
        if truncated:
            # A closed first tool block can precede a subsequently truncated draft.
            blocks = list(re.finditer(r"<tool_call>.*?</tool_call>", text, re.DOTALL))
            if blocks:
                selected = text[: blocks[0].end()]
            else:
                try:
                    envelope, _ = json.JSONDecoder().raw_decode(text.lstrip())
                except (ValueError, TypeError):
                    envelope = None
                if isinstance(envelope, dict) and envelope.get("tool_calls"):
                    selected = json.dumps(envelope)
                elif isinstance(envelope, dict) and isinstance(
                    envelope.get("final_answer"), str
                ):
                    selected = "Final Answer: " + envelope["final_answer"]
                else:
                    if "<tool_call" in text or text.lstrip().startswith(
                        ("{", "[", "<function", "<parameter")
                    ):
                        raise ModelResponseError(
                            "Incomplete structured action cannot contain a final answer"
                        )
                    final = re.search(
                        r"(?im)^\s*Final Answer:[ \t]*([^\r\n]+)\r?\n", text
                    )
                    if final is None:
                        raise ModelResponseError(
                            "Truncated output contains no delimited complete action"
                        )
                    selected = "Final Answer: " + final.group(1).strip()
        elif "<tool_call>" in text and "</tool_call>" in text:
            pass
        elif not text.strip() or text.lstrip().startswith(
            ("{", "[", "<function", "<parameter", "<tool_call")
        ):
            # Allow the supported complete JSON envelopes, not partial structures.
            try:
                envelope = json.loads(text)
            except (ValueError, TypeError) as exc:
                raise ModelResponseError("Incomplete action structure") from exc
            if isinstance(envelope, dict) and isinstance(
                envelope.get("final_answer"), str
            ):
                selected = "Final Answer: " + envelope["final_answer"]
            elif not isinstance(envelope, dict) or not envelope.get("tool_calls"):
                raise ModelResponseError("JSON output is not an action envelope")
        parsed = parse_generated_text(
            selected, provider=response.provider, model=response.model
        )
        candidate = replace(candidate, text=parsed.text, tool_calls=parsed.tool_calls)

    action = parser.parse(candidate)
    if action.action_type is ActionType.TOOL:
        if final_only or not request.tools:
            raise ModelResponseError(
                "Tool calls are not permitted in the final-answer call"
            )
        if len(action.tool_calls) != 1:
            raise ModelResponseError("Each step must contain exactly one tool call")
        call = action.tool_calls[0]
        if call.tool_name not in {tool.name for tool in request.tools}:
            raise ModelResponseError("Tool is not available in this request")
        try:
            validate_arguments(
                call.arguments, parser.tools.spec(call.tool_name).input_schema
            )
        except ToolSchemaError as exc:
            raise ModelResponseError(f"Invalid tool action: {exc}") from exc
    elif (
        action.action_type is not ActionType.FINAL
        or not (action.final_answer or "").strip()
    ):
        raise ModelResponseError("No valid tool action or final answer")
    if (
        action.action_type is ActionType.FINAL
        and request.metadata.get("rollout_budget", {}).get("call_kind") == "recovery"
        and not re.search(r"(?im)^\s*Final Answer:[ \t]*\S", candidate.text or "")
    ):
        raise ModelResponseError(
            "Action recovery requires an explicit Final Answer or tool call"
        )
    if action.action_type is ActionType.FINAL and re.fullmatch(
        r"\s*Final Answer:\s*", action.final_answer or "", re.IGNORECASE
    ):
        raise ModelResponseError("Final answer is empty")
    return action, candidate


def decide(request, parser, generate, *, forced_final=False):
    """Return a serializable decision. `generate(kind, request)` may be journaled."""
    events = []

    def call(kind, current):
        event = {
            "kind": kind,
            "step_index": current.metadata.get("step_index"),
            "state_id": current.metadata.get("state_id"),
            "max_output_tokens": current.settings.max_output_tokens,
        }
        events.append(event)
        try:
            response = generate(kind, current)
        except ModelResponseError as exc:
            event.update(parse_error=str(exc), token_truncated=False)
            return None, str(exc)
        event.update(
            finish_reason=response.finish_reason,
            token_truncated=response.finish_reason == "length",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            response_id=response.response_id,
        )
        return response, None

    def result(current, response, action=None, failure=None, error=None):
        value = {
            "policy": BUDGET_POLICY,
            "request": request_to_dict(current, identity=False),
            "response": response_to_dict(response) if response is not None else None,
            "action": action.to_dict() if action is not None else None,
            "events": events,
            "failure_kind": failure,
            "failure_reason": (error or "No valid model action") if failure else None,
        }
        # Keep non-journaled trajectories identical after JSON replay too.
        return json.loads(json.dumps(value, allow_nan=False))

    current = recovery_request(request, forced_final=True) if forced_final else request
    kind = "forced_final" if forced_final else "normal"
    response, error = call(kind, current)
    if response is not None:
        try:
            action, parsed = parse_complete_action(
                parser, response, current, final_only=forced_final
            )
            return result(current, parsed, action)
        except ModelResponseError as exc:
            error = str(exc)
    if forced_final:
        return result(current, response, failure="forced_final_failed", error=error)
    if response is None or response.finish_reason != "length":
        return result(current, response, failure="invalid_action", error=error)
    current = recovery_request(request)
    response, error = call("recovery", current)
    if response is not None:
        try:
            action, parsed = parse_complete_action(parser, response, current)
            return result(current, parsed, action)
        except ModelResponseError as exc:
            error = str(exc)
    return result(current, response, failure="action_recovery_failed", error=error)


def audit_metadata(events, transitions, failure_kind=None):
    tool_calls = sum(len(step.action.tool_calls) for step in transitions)
    return {
        "budget_policy": BUDGET_POLICY,
        "failure_kind": failure_kind,
        "environment_steps": tool_calls,
        "generation_events": events,
        "call_counts": {
            "normal_llm_calls": sum(e["kind"] == "normal" for e in events),
            "recovery_calls": sum(e["kind"] == "recovery" for e in events),
            "forced_final_answers": sum(e["kind"] == "forced_final" for e in events),
            "token_truncations": sum(e["token_truncated"] for e in events),
            "tool_calls": tool_calls,
        },
    }


def is_controlled_failure(trajectory):
    return (
        trajectory.reward == 0.0
        and trajectory.metadata.get("budget_policy") == BUDGET_POLICY
        and trajectory.metadata.get("failure_kind") in CONTROLLED_FAILURES
    )
