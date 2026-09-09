"""Minimal multi-step MLLM → tool → observation → answer loop."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from spatialcraft.models import ModelProvider
from spatialcraft.schemas import (
    ActionType,
    AgentAction,
    TaskSample,
    Trajectory,
    TrajectoryStatus,
    Transition,
)
from spatialcraft.tools import ToolContext, ToolExecutor

from .action_parser import ActionParser
from .context_composer import ContextComposer
from .decision import audit_metadata, decide
from .state_builder import StateBuilder

if TYPE_CHECKING:
    from spatialcraft.rollout.reward import RewardComputer

    from .skill_controller import SkillController


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionConfig:
    max_steps: int = 50
    tool_backend: str = "local"
    parallel_tools: bool = False
    knowledge_snapshot_id: str = "vanilla"

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not self.knowledge_snapshot_id.strip():
            raise ValueError("knowledge_snapshot_id cannot be empty")


class ExecutionLoop:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        composer: ContextComposer,
        action_parser: ActionParser,
        tool_executor: ToolExecutor,
        reward: RewardComputer | None = None,
        state_builder: StateBuilder | None = None,
        config: ExecutionConfig | None = None,
        skill_controller: SkillController | None = None,
    ) -> None:
        self.provider = provider
        self.composer = composer
        self.action_parser = action_parser
        self.tool_executor = tool_executor
        if reward is None:
            from spatialcraft.rollout.reward import RewardComputer

            reward = RewardComputer()
        self.reward = reward
        self.state_builder = state_builder or StateBuilder()
        self.config = config or ExecutionConfig()
        self.skill_controller = skill_controller

    def run(
        self,
        task: TaskSample,
        *,
        rollout_index: int = 0,
        random_seed: int | None = None,
        initial_state=None,
    ) -> Trajectory:
        started_at = datetime.now(timezone.utc)
        state = initial_state or self.state_builder.initial(task)
        transitions: list[Transition] = []
        events = []
        failure_kind, failure_reason = None, None
        try:
            for step_index in range(self.config.max_steps + 1):
                forced_final = step_index == self.config.max_steps
                if self.skill_controller is not None and not forced_final:
                    state = self.skill_controller.before_step(
                        task.without_reference_answer(), state
                    )
                request = self.composer.compose(task.without_reference_answer(), state)
                if random_seed is not None:
                    request = replace(
                        request,
                        settings=replace(
                            request.settings,
                            seed=(random_seed * 100003 + step_index) % (2**63 - 1),
                        ),
                    )
                decision = decide(
                    request,
                    self.action_parser,
                    lambda kind, current: self.provider.generate(current),
                    forced_final=forced_final,
                )
                events.extend(decision["events"])
                if decision["action"] is None:
                    failure_kind, failure_reason = (
                        decision["failure_kind"],
                        decision["failure_reason"],
                    )
                    break
                action = AgentAction.from_dict(decision["action"])
                transition_metadata = {
                    "model_request": decision["request"],
                    "call_kind": decision["events"][-1]["kind"],
                    "consumes_environment_step": action.action_type is ActionType.TOOL,
                }
                used_experience_ids = tuple(
                    item.experience_id for item in state.retrieved_experiences
                )
                if action.action_type is ActionType.TOOL:
                    results = self.tool_executor.execute_action(
                        action,
                        context=ToolContext(
                            run_id=self.tool_executor.artifact_store.run_id,
                            task_id=task.task_id,
                            state_id=state.state_id,
                        ),
                        backend=self.config.tool_backend,
                        parallel=self.config.parallel_tools,
                    )
                    next_state = self.state_builder.after_tools(state, action, results)
                    if self.skill_controller is not None:
                        next_state = self.skill_controller.after_step(
                            task.without_reference_answer(), state, action, next_state
                        )
                    transitions.append(
                        Transition(
                            step_index=step_index,
                            state_before=state,
                            action=action,
                            tool_results=results,
                            state_after=next_state,
                            active_skill=state.active_skill,
                            used_experience_ids=used_experience_ids,
                            metadata=transition_metadata,
                        )
                    )
                    state = next_state
                    continue
                if action.action_type is ActionType.FINAL:
                    outcome = self.reward.compute(task, action)
                    next_state = self.state_builder.after_final(state, action)
                    if self.skill_controller is not None:
                        # Final termination is a pure lifecycle rule; no LLM call.
                        next_state = self.skill_controller.after_step(
                            task.without_reference_answer(), state, action, next_state
                        )
                    transitions.append(
                        Transition(
                            step_index=step_index,
                            state_before=state,
                            action=action,
                            state_after=next_state,
                            active_skill=state.active_skill,
                            used_experience_ids=used_experience_ids,
                            reward=outcome.score,
                            done=True,
                            metadata=transition_metadata,
                        )
                    )
                    return Trajectory(
                        task=task,
                        rollout_index=rollout_index,
                        executor_model=self.composer.model.alias,
                        knowledge_snapshot_id=self.config.knowledge_snapshot_id,
                        status=TrajectoryStatus.COMPLETED,
                        transitions=tuple(transitions),
                        final_answer=action.final_answer,
                        verifier=outcome,
                        reward=outcome.score,
                        random_seed=random_seed,
                        started_at=started_at,
                        finished_at=datetime.now(timezone.utc),
                        metadata=audit_metadata(events, transitions),
                    )
            return Trajectory(
                task=task,
                rollout_index=rollout_index,
                executor_model=self.composer.model.alias,
                knowledge_snapshot_id=self.config.knowledge_snapshot_id,
                status=TrajectoryStatus.FAILED,
                transitions=tuple(transitions),
                reward=0.0,
                random_seed=random_seed,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                failure_reason=failure_reason or "No valid action at the step limit",
                metadata=audit_metadata(events, transitions, failure_kind),
            )
        except Exception as exc:  # noqa: BLE001 - preserve partial rollout as data
            return Trajectory(
                task=task,
                rollout_index=rollout_index,
                executor_model=self.composer.model.alias,
                knowledge_snapshot_id=self.config.knowledge_snapshot_id,
                status=TrajectoryStatus.FAILED,
                transitions=tuple(transitions),
                reward=0.0,
                random_seed=random_seed,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                failure_reason=f"{type(exc).__name__}: {exc}",
                metadata=audit_metadata(events, transitions),
            )


__all__ = ["ExecutionConfig", "ExecutionLoop"]
