"""Compare rollouts after completion and extract reusable procedural lessons."""

from __future__ import annotations

from dataclasses import dataclass

from spatialcraft.schemas import Trajectory

from .task_decomposer import TextGenerator
from .visual_summarizer import VisualTrajectorySummarizer


@dataclass(frozen=True, slots=True, kw_only=True)
class RolloutCritique:
    task_id: str
    best_trajectory_ids: tuple[str, ...]
    failed_trajectory_ids: tuple[str, ...]
    reusable_condition: str
    reusable_action: str
    rationale: str
    summaries: tuple[str, ...]


class CrossRolloutCritic:
    def __init__(
        self,
        *,
        summarizer: VisualTrajectorySummarizer | None = None,
        generator: TextGenerator | None = None,
        strict: bool = False,
    ) -> None:
        self.summarizer = summarizer or VisualTrajectorySummarizer()
        self.generator = generator
        self.strict = strict
        if strict and generator is None:
            raise ValueError("Strict cross-rollout critique requires an LLM generator")

    def critique(self, trajectories: tuple[Trajectory, ...]) -> RolloutCritique:
        if not trajectories:
            raise ValueError("cross-rollout critique requires trajectories")
        task_ids = {item.task.task_id for item in trajectories}
        if len(task_ids) != 1:
            raise ValueError("cross-rollout critique requires one shared task")
        if len({t.knowledge_snapshot_id for t in trajectories}) != 1:
            raise ValueError(
                "Cross-rollout comparison requires one frozen knowledge snapshot"
            )
        if any(t.reward is None for t in trajectories):
            raise ValueError("Cross-rollout critique requires verified rewards")
        ordered = sorted(
            trajectories,
            key=lambda item: (-(item.reward or 0.0), item.rollout_index),
        )
        best_score = ordered[0].reward or 0.0
        best = tuple(
            item.trajectory_id
            for item in ordered
            if best_score > 0 and (item.reward or 0.0) == best_score
        )
        failed = tuple(
            item.trajectory_id
            for item in ordered
            if (item.reward or 0.0) < best_score or (item.reward or 0.0) <= 0
        )
        summaries = tuple(self.summarizer.summarize(item) for item in ordered)
        successful = ordered[0]
        tool_sequence = [
            result.tool_name
            for transition in successful.transitions
            for result in transition.tool_results
            if result.succeeded
        ]
        condition = (
            f"When solving {successful.task.dataset} tasks of type "
            f"{successful.task.metadata.get('question_type', successful.task.answer_type.value)}"
        )
        action = (
            "Use the evidence sequence "
            + (
                " → ".join(tool_sequence)
                if tool_sequence
                else "direct visual reasoning"
            )
            + ", preserve coordinate frames, then verify the final relation against the question."
        )
        rationale = (
            f"Compared {len(ordered)} rollouts; best reward={best_score:.3f}; "
            f"{len(failed)} lower-reward rollout(s)."
        )
        if self.generator is not None:
            prompt = (
                "Derive one reusable condition/action lesson from these post-rollout "
                "summaries. Return two lines prefixed CONDITION: and ACTION:.\n"
                "Do not treat tied all-failure trajectories as a successful strategy. "
                "Abstract evidence-grounded corrections; do not memorize this task's answer.\n"
                + "\n---\n".join(
                    f"Trajectory: {row.trajectory_id}; verified reward: {row.reward}\n{summary}"
                    for row, summary in zip(ordered, summaries, strict=True)
                )
            )
            generated = self.generator(prompt)
            extracted = {}
            for line in generated.splitlines():
                if line.upper().startswith("CONDITION:"):
                    condition = line.split(":", 1)[1].strip() or condition
                    extracted["condition"] = line.split(":", 1)[1].strip()
                elif line.upper().startswith("ACTION:"):
                    action = line.split(":", 1)[1].strip() or action
                    extracted["action"] = line.split(":", 1)[1].strip()
            if self.strict and not (
                extracted.get("condition") and extracted.get("action")
            ):
                raise ValueError(
                    "Critic must return nonempty CONDITION and ACTION; no heuristic fallback"
                )
        return RolloutCritique(
            task_id=next(iter(task_ids)),
            best_trajectory_ids=best,
            failed_trajectory_ids=failed,
            reusable_condition=condition,
            reusable_action=action,
            rationale=rationale,
            summaries=summaries,
        )


__all__ = ["CrossRolloutCritic", "RolloutCritique"]
