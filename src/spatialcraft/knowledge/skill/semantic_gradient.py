"""Post-rollout semantic diagnosis for skill initiation, policy, and termination."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

from spatialcraft.schemas import SemanticGradient, Trajectory

from .segments import SkillSegment, skill_segments

GradientGenerator = Callable[[SkillSegment, float, tuple[str, ...]], dict[str, str]]


def gradient_reference(gradient: SemanticGradient) -> str:
    stored = gradient.metadata.get("gradient_id")
    if stored:
        return str(stored)
    digest = sha256(gradient.to_json().encode("utf-8")).hexdigest()[:20]
    return f"semantic_gradient_{digest}"


class SemanticGradientEngine:
    def __init__(self, generator: GradientGenerator | None = None) -> None:
        self.generator = generator

    @staticmethod
    def sample(
        trajectories: tuple[Trajectory, ...], *, per_outcome: int = 4
    ) -> tuple[Trajectory, ...]:
        if per_outcome < 1:
            raise ValueError("per_outcome must be positive")
        ordered = sorted(
            trajectories,
            key=lambda item: (float(item.reward or 0.0), item.rollout_index),
        )
        if not ordered:
            return ()
        failures = ordered[:per_outcome]
        successes = list(reversed(ordered[-per_outcome:]))
        return tuple(
            {item.trajectory_id: item for item in (*failures, *successes)}.values()
        )

    @staticmethod
    def _fallback(
        trajectory: Trajectory, advantage: float, references: tuple[str, ...]
    ) -> dict[str, str]:
        tools = tuple(
            result.tool_name
            for transition in trajectory.transitions
            for result in transition.tool_results
        )
        if advantage >= 0:
            diagnosis = "Successful or above-baseline trajectory; preserve its evidence ordering."
            initiation = f"Activate for {trajectory.task.answer_type.value} questions matching this task pattern."
            policy = (
                "Preserve the observed tool sequence: " + " -> ".join(tools)
                if tools
                else "Preserve direct reasoning while explicitly checking spatial relations."
            )
            termination = (
                "Terminate once the answer-critical relation has independent support."
            )
        else:
            diagnosis = "Below-baseline trajectory; tighten evidence selection and stopping behavior."
            initiation = "Narrow activation to tasks where the skill can resolve a stated spatial uncertainty."
            policy = "Do not repeat uninformative evidence; validate each observation before the next action."
            termination = "Terminate or switch strategy when evidence remains unchanged or the step budget is near."
        if not references:
            diagnosis += (
                " No skill was active, so a new procedural skill may be required."
            )
        return {
            "diagnosis": diagnosis,
            "initiation": initiation,
            "policy": policy,
            "termination": termination,
        }

    def diagnose(
        self,
        trajectories: tuple[Trajectory, ...],
        advantages: dict[str, float],
        *,
        per_outcome: int | None = None,
        skill_references: frozenset[str] | None = None,
    ) -> tuple[SemanticGradient, ...]:
        output = []
        selected = (
            self.sample(trajectories, per_outcome=per_outcome)
            if per_outcome is not None
            else trajectories
        )
        for trajectory in selected:
            advantage = float(advantages.get(trajectory.trajectory_id, 0.0))
            for segment in skill_segments(trajectory):
                if (
                    skill_references is not None
                    and segment.skill_reference not in skill_references
                ):
                    continue
                refs = (segment.skill_reference,) if segment.skill_reference else ()
                values = (
                    self.generator(segment, advantage, refs)
                    if self.generator is not None
                    else self._fallback(segment, advantage, refs)
                )
                fingerprint = sha256(
                    f"{trajectory.trajectory_id}|{refs}|{segment.start_step}:{segment.end_step}|{advantage}".encode()
                ).hexdigest()[:20]
                output.append(
                    SemanticGradient(
                        trajectory_id=trajectory.trajectory_id,
                        diagnosis=values["diagnosis"],
                        is_related=bool(refs),
                        initiation=values.get("initiation", ""),
                        policy=values.get("policy", ""),
                        termination=values.get("termination", ""),
                        reward=trajectory.reward,
                        metadata={
                            "gradient_id": f"semantic_gradient_{fingerprint}",
                            "skill_references": refs,
                            "task_id": trajectory.task.task_id,
                            "advantage": advantage,
                            "post_rollout_only": True,
                            "segment_start": segment.start_step,
                            "segment_end": segment.end_step,
                            "transition_ids": [
                                t.transition_id for t in segment.transitions
                            ],
                        },
                    )
                )
        return tuple(output)


__all__ = ["GradientGenerator", "SemanticGradientEngine", "gradient_reference"]
