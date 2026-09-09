"""Post-rollout visual/tool evidence summaries for knowledge accumulation."""

from __future__ import annotations

from collections.abc import Callable

from spatialcraft.schemas import Trajectory


class VisualTrajectorySummarizer:
    """Runs only after a rollout and may therefore include verifier evidence."""

    def __init__(
        self,
        generator: Callable[[str], str] | None = None,
        *,
        visual_generator: Callable[[str, Trajectory], str] | None = None,
    ) -> None:
        self.generator = generator
        self.visual_generator = visual_generator

    def summarize(self, trajectory: Trajectory) -> str:
        tool_lines = []
        for transition in trajectory.transitions:
            for result in transition.tool_results:
                tool_lines.append(
                    f"step={transition.step_index} tool={result.tool_name} "
                    f"status={result.status.value} observation={result.text or ''} "
                    f"artifacts={','.join(item.artifact_id for item in result.artifacts)}"
                )
        deterministic = (
            f"Trajectory: {trajectory.trajectory_id}\nQuestion: {trajectory.task.question}\n"
            f"Task type: {trajectory.task.metadata.get('question_type', 'unknown')}\n"
            f"Tool evidence:\n" + ("\n".join(tool_lines) or "none") + "\n"
            f"Final answer: {trajectory.final_answer}\n"
            f"Reference answer: {trajectory.task.reference_answer}\n"
            f"Verifier score: {trajectory.reward}"
        )
        if self.generator is None and self.visual_generator is None:
            return deterministic
        prompt = (
            "Summarize only the observed images/tool evidence and post-rollout verifier "
            "feedback. This output is for memory construction, not agent execution.\n"
            + deterministic
        )
        generated = (
            self.visual_generator(prompt, trajectory)
            if self.visual_generator is not None
            else self.generator(prompt)
        ).strip()
        return generated or deterministic


__all__ = ["VisualTrajectorySummarizer"]
