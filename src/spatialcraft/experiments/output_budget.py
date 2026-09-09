"""Bounded completion for auxiliary knowledge generation, not agent actions."""

import json
from dataclasses import replace
from functools import wraps
from inspect import signature

from spatialcraft.models import MessageRole, ModelMessage

from .journal import digest


class OutputBudgetExhausted(RuntimeError):
    """Only an unsuccessful budget-recovery call, never an infrastructure error."""

    def __init__(self, original, recovery):
        self.audit = {
            "policy": "force_completion_v1",
            "original_finish_reason": original.finish_reason,
            "recovery_finish_reason": recovery.finish_reason,
            "original_text_sha256": digest(original.text),
            "recovery_text_sha256": digest(recovery.text),
            "original_output_tokens": original.usage.output_tokens,
            "recovery_output_tokens": recovery.usage.output_tokens,
        }
        super().__init__("Output-limit recovery did not produce a complete result")


def completion_request(request, *, purpose):
    """One NEW call; retain original evidence, not the truncated assistant draft.

    Each call retains its own original token cap. Additional usage is recorded
    by the audited provider. Metadata separates recovery from cached originals.
    """
    if purpose != "knowledge":
        raise ValueError("Use the agent action-recovery policy for rollout decisions")
    instruction = (
        "The previous generation exhausted its output budget. Stop deliberating. "
        "Use only the existing question, images and tool evidence. "
        "Produce a fresh compact, complete result now, preferably under 512 tokens. "
        "Keep all fields required by the original requested schema, close every "
        "JSON string/bracket, and emit no code fence. For a summary, use at most "
        "six short sentences. Merge repeated observations; do not enumerate "
        "every step or artifact ID. Do not invent evidence."
    )
    return replace(
        request,
        messages=(*request.messages, ModelMessage.text(MessageRole.USER, instruction)),
        tools=(),
        tool_choice=None,
        settings=replace(request.settings, temperature=0.0),
        metadata={
            **request.metadata,
            "output_budget_recovery": {
                "policy": "force_completion_v1",
                "purpose": purpose,
            },
        },
    )


def budget_safe_learning(method):
    """Skip only a failed bounded recovery; preserve knowledge/queue atomically."""
    method_signature = signature(method)

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except OutputBudgetExhausted as exc:
            audit = {
                **exc.audit,
                "operation": method.__name__,
                "status": "skipped_output_limit",
            }
            print(
                "OUTPUT_BUDGET_FALLBACK " + json.dumps(audit, sort_keys=True),
                flush=True,
            )
            if method.__name__ == "retrieve":
                return ()
            if method.__name__ == "terminate":
                return True  # End this Skill, not the trajectory.
            bound = method_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            knowledge, rows = bound.arguments["knowledge"], bound.arguments["rows"]
            if method.__name__ == "update_experiences":
                return {
                    "experience_bank": knowledge.experiences.to_dict(),
                    "critique": None,
                    "updates": [],
                    "source_trajectory_ids": [r.trajectory_id for r in rows],
                    "output_budget_recovery": audit,
                }
            if method.__name__ == "evolve":
                from .evolution_queue import EvolutionQueue

                batch_index = bound.arguments["batch_index"]
                state = bound.arguments["evolution_state"]
                queue = EvolutionQueue(state)
                queue.enqueue(
                    rows,
                    task_index=batch_index,
                    active_references={s.reference for s in knowledge.skills.active()},
                )
                return {
                    "skill_pool": knowledge.skills.to_dict(),
                    "evolution_state": queue.state,
                    "round_index": batch_index,
                    "source_trajectory_ids": [r.trajectory_id for r in rows],
                    "pending_counts": {
                        ref: len(entries)
                        for ref, entries in queue.state["queues"].items()
                    },
                    "accepted_candidate_ids": [],
                    "output_budget_recovery": audit,
                }
            raise

    return wrapped
