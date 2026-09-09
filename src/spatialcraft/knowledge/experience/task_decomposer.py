"""Leak-safe task decomposition before Experience retrieval."""

from __future__ import annotations

import re
from collections.abc import Callable

from spatialcraft.schemas import TaskSample

TextGenerator = Callable[[str], str]


class TaskDecomposer:
    def __init__(
        self, generator: TextGenerator | None = None, *, max_subtasks: int = 6
    ) -> None:
        if max_subtasks < 1:
            raise ValueError("max_subtasks must be positive")
        self.generator = generator
        self.max_subtasks = max_subtasks

    def decompose(self, task: TaskSample) -> tuple[str, ...]:
        safe_task = task.without_reference_answer()
        if self.generator is not None:
            prompt = (
                "Decompose this spatial problem into short retrieval queries. "
                "Return one query per line and never infer an answer.\n"
                f"Dataset: {safe_task.dataset}\nQuestion: {safe_task.question}"
            )
            lines = [
                re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                for line in self.generator(prompt).splitlines()
            ]
            values = tuple(dict.fromkeys(line for line in lines if line))
            if values:
                return values[: self.max_subtasks]
        question_type = safe_task.metadata.get("question_type")
        queries = [safe_task.question]
        if question_type:
            queries.append(f"spatial task type: {question_type}")
        if len(safe_task.images) > 1:
            queries.append("compare object state and viewpoint across multiple images")
        if safe_task.choices:
            queries.append("eliminate inconsistent spatial answer choices")
        return tuple(queries[: self.max_subtasks])


__all__ = ["TaskDecomposer", "TextGenerator"]
