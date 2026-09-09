"""Journaled real agent loop: durable selection, model, tools, beta, and verifier.

This is the experiment entry point; the older ExecutionLoop remains a lightweight
component/example. Infrastructure failures propagate, never become training zeros.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from spatialcraft.agent.decision import BUDGET_POLICY, audit_metadata, decide
from spatialcraft.models.serialization import (
    request_to_dict,
    response_from_dict,
    response_to_dict,
)
from spatialcraft.schemas import (
    ActionType,
    AgentAction,
    SpatialState,
    TaskSample,
    ToolResult,
    Trajectory,
    TrajectoryStatus,
    Transition,
    VerifierOutcome,
)
from spatialcraft.storage.atomic_io import sha256_file
from spatialcraft.tools import ToolContext

from .journal import RunJournal


class JournaledRollout:
    def __init__(self, loop, journal: RunJournal):
        self.loop, self.journal = loop, journal

    def run(
        self,
        task: TaskSample,
        *,
        prefix: str,
        initial_state: SpatialState,
        rollout_index: int,
        random_seed: int,
    ) -> Trajectory:
        loop = self.loop
        initial, _ = self.journal.execute(
            prefix + "/initial",
            {
                "task": task.to_dict(),
                "snapshot": loop.config.knowledge_snapshot_id,
                "rollout_index": rollout_index,
                "seed": random_seed,
                "experiences": [
                    ref.to_dict() for ref in initial_state.retrieved_experiences
                ],
            },
            lambda: {
                "state": initial_state.to_dict(),
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        state = SpatialState.from_dict(initial["state"])
        transitions = []
        status, final_answer, outcome = TrajectoryStatus.TRUNCATED, None, None
        failure_reason = "Agent action budget exhausted"
        failure_kind = None
        events = []
        safe_task = task.without_reference_answer()
        for step in range(loop.config.max_steps + 1):
            forced_final = step == loop.config.max_steps
            key = f"{prefix}/steps/{step:04d}"
            if loop.skill_controller is not None and not forced_final:
                selected, _ = self.journal.execute(
                    key + "/selection",
                    {"state": state.to_dict()},
                    lambda state=state: loop.skill_controller.before_step(
                        safe_task, state
                    ).to_dict(),
                )
                state = SpatialState.from_dict(selected)
            request = loop.composer.compose(safe_task, state)
            request = replace(
                request,
                settings=replace(
                    request.settings, seed=(random_seed * 100003 + step) % (2**63 - 1)
                ),
            )

            def generate(kind, current, key=key):
                call_key = (
                    key
                    + {
                        "normal": "/model",
                        "recovery": "/recovery/model",
                        "forced_final": "/forced_final/model",
                    }[kind]
                )
                wire, _ = self.journal.execute(
                    call_key,
                    {"request": request_to_dict(current, identity=False)},
                    lambda: response_to_dict(loop.provider.generate(current)),
                )
                return response_from_dict(wire)

            decision, _ = self.journal.execute(
                key + "/decision",
                {
                    "policy": BUDGET_POLICY,
                    "request": request_to_dict(request, identity=False),
                    "forced_final": forced_final,
                },
                lambda request=request, generate=generate, forced_final=forced_final: (
                    decide(
                        request, loop.action_parser, generate, forced_final=forced_final
                    )
                ),
            )
            events.extend(decision["events"])
            if decision["action"] is None:
                status = TrajectoryStatus.FAILED
                failure_kind = decision["failure_kind"]
                failure_reason = decision["failure_reason"]
                break
            request_wire = decision["request"]
            response_wire = decision["response"]
            response = response_from_dict(response_wire)
            action_wire, _ = self.journal.execute(
                key + "/action",
                {"decision": decision},
                lambda decision=decision: decision["action"],
            )
            action = AgentAction.from_dict(action_wire)
            scoring_prefix = (
                (response.raw or {}).get("sampled_thinking_prefix")
                if isinstance(response.raw, dict)
                else None
            )
            if request_wire.get("metadata", {}).get("chat_template_kwargs", {}).get(
                "enable_thinking"
            ) is True and (
                not isinstance(scoring_prefix, str) or "</think>" not in scoring_prefix
            ):
                raise ValueError(
                    "Thinking rollout is missing the fixed reasoning prefix audit"
                )
            used = tuple(ref.experience_id for ref in state.retrieved_experiences)
            results = []
            if action.action_type is ActionType.TOOL:
                for call_index, call in enumerate(action.tool_calls):

                    def execute(call=call, state=state):
                        result = loop.tool_executor.execute(
                            call,
                            context=ToolContext(
                                run_id=loop.tool_executor.artifact_store.run_id,
                                task_id=task.task_id,
                                state_id=state.state_id,
                            ),
                            backend=loop.config.tool_backend,
                        )
                        if not result.succeeded and result.error_type not in {
                            None,
                            "argument_validation",
                        }:
                            raise RuntimeError(
                                f"Tool {call.tool_name} failed: {result.error_type}; inspect artifact result audit"
                            )
                        return result.to_dict()

                    value, _ = self.journal.execute(
                        key + f"/tools/{call_index:03d}",
                        {"call": call.to_dict(), "state_id": state.state_id},
                        execute,
                    )
                    result = ToolResult.from_dict(value)
                    for artifact in result.artifacts:
                        path = loop.tool_executor.artifact_store.layout.resolve_uri(
                            artifact.uri
                        )
                        if artifact.sha256 and sha256_file(path) != artifact.sha256:
                            raise ValueError(
                                "Tool artifact changed; refusing corrupted resume"
                            )
                    results.append(result)
                next_wire, _ = self.journal.execute(
                    key + "/observation",
                    {
                        "state": state.to_dict(),
                        "action": action_wire,
                        "results": [r.to_dict() for r in results],
                    },
                    lambda state=state, action=action, results=tuple(results): (
                        loop.state_builder.after_tools(state, action, results).to_dict()
                    ),
                )
            elif action.action_type is ActionType.FINAL:
                verdict, _ = self.journal.execute(
                    key + "/verification",
                    {"task": task.to_dict(), "action": action_wire},
                    lambda action=action: loop.reward.compute(task, action).to_dict(),
                )
                outcome = VerifierOutcome.from_dict(verdict)
                final_answer, status = action.final_answer, TrajectoryStatus.COMPLETED
                next_wire, _ = self.journal.execute(
                    key + "/observation",
                    {"state": state.to_dict(), "action": action_wire},
                    lambda state=state, action=action: loop.state_builder.after_final(
                        state, action
                    ).to_dict(),
                )
            else:
                raise ValueError("No-op output is not a valid experimental action")
            next_state = SpatialState.from_dict(next_wire)
            if loop.skill_controller is not None:
                next_wire, _ = self.journal.execute(
                    key + "/termination",
                    {
                        "before": state.to_dict(),
                        "after": next_wire,
                        "action": action_wire,
                    },
                    lambda state=state, action=action, next_state=next_state: (
                        loop.skill_controller.after_step(
                            safe_task, state, action, next_state
                        ).to_dict()
                    ),
                )
                next_state = SpatialState.from_dict(next_wire)
            transition, _ = self.journal.execute(
                key + "/transition",
                {
                    "state": state.to_dict(),
                    "next_state": next_state.to_dict(),
                    "action": action_wire,
                    "sampled_thinking_prefix": scoring_prefix,
                },
                lambda step=step, state=state, next_state=next_state, action=action, results=tuple(results), used=used, request_wire=request_wire, outcome=outcome, scoring_prefix=scoring_prefix, decision=decision: (
                    Transition(
                        step_index=step,
                        state_before=state,
                        state_after=next_state,
                        action=action,
                        tool_results=results,
                        active_skill=state.active_skill,
                        used_experience_ids=used,
                        done=action.action_type is ActionType.FINAL,
                        reward=outcome.score if outcome is not None else None,
                        metadata={
                            "model_request": request_wire,
                            "sampled_thinking_prefix": scoring_prefix,
                            "call_kind": decision["events"][-1]["kind"],
                            "budget_policy": BUDGET_POLICY,
                            "consumes_environment_step": action.action_type
                            is ActionType.TOOL,
                        },
                    ).to_dict()
                ),
            )
            transitions.append(Transition.from_dict(transition))
            state = next_state
            if status is TrajectoryStatus.COMPLETED:
                break
        value, _ = self.journal.execute(
            prefix + "/trajectory",
            {
                "transitions": [t.to_dict() for t in transitions],
                "status": status.value,
                "outcome": outcome.to_dict() if outcome else None,
                "budget_audit": audit_metadata(events, transitions, failure_kind),
            },
            lambda: Trajectory(
                task=task,
                rollout_index=rollout_index,
                executor_model=loop.composer.model.alias,
                knowledge_snapshot_id=loop.config.knowledge_snapshot_id,
                status=status,
                transitions=tuple(transitions),
                final_answer=final_answer,
                verifier=outcome,
                reward=outcome.score if outcome else 0.0,
                random_seed=random_seed,
                started_at=datetime.fromisoformat(initial["started_at"]),
                finished_at=datetime.now(timezone.utc),
                failure_reason=None if outcome else failure_reason,
                metadata=audit_metadata(events, transitions, failure_kind),
            ).to_dict(),
        )
        return Trajectory.from_dict(value)
