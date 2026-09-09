"""Task-specific rewriting of retrieved experience without ground-truth access."""

from __future__ import annotations

from dataclasses import replace

from spatialcraft.schemas import RetrievedExperienceRef, TaskSample

from .task_decomposer import TextGenerator


class ContextualExperienceRewriter:
    def __init__(self, generator: TextGenerator | None = None) -> None:
        self.generator = generator

    def rewrite(
        self,
        task: TaskSample,
        retrieved: tuple[RetrievedExperienceRef, ...],
    ) -> tuple[RetrievedExperienceRef, ...]:
        safe_task = task.without_reference_answer()
        rewritten = []
        for item in retrieved:
            if self.generator is None:
                text = (
                    f"For the current {safe_task.dataset} task, apply this lesson only "
                    f"when its condition is observed: {item.original_text}"
                )
            else:
                prompt = (
                    "Filter then rewrite this experience as concise task-specific advice. "
                    "Return exactly SKIP if inapplicable, redundant, or contradicted by the task. Do not solve "
                    "the task and do not invent observations.\n"
                    f"Question: {safe_task.question}\nExperience: {item.original_text}"
                )
                text = self.generator(prompt).strip()
                if not text or text.upper() == "SKIP":
                    continue
            rewritten.append(replace(item, contextualized_text=text))
        return tuple(rewritten)


__all__ = ["ContextualExperienceRewriter"]
