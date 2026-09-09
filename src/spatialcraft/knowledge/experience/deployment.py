"""Deployment-side Experience preparation and state injection."""

from __future__ import annotations

from spatialcraft.agent.state_builder import StateBuilder
from spatialcraft.schemas import SpatialState, TaskSample

from .contextual_rewriter import ContextualExperienceRewriter
from .retriever import ExperienceRetriever


class ExperienceDeployment:
    def __init__(
        self,
        retriever: ExperienceRetriever,
        *,
        rewriter: ContextualExperienceRewriter | None = None,
        state_builder: StateBuilder | None = None,
    ) -> None:
        self.retriever = retriever
        self.rewriter = rewriter or ContextualExperienceRewriter()
        self.state_builder = state_builder or StateBuilder()

    def prepare_state(
        self, task: TaskSample, *, top_k: int | None = None
    ) -> SpatialState:
        safe_task = task.without_reference_answer()
        retrieved = self.retriever.retrieve(safe_task, top_k=top_k)
        rewritten = self.rewriter.rewrite(safe_task, retrieved)
        return self.state_builder.initial(
            safe_task,
            retrieved_experiences=rewritten,
            metadata={"experience_enabled": True, "retrieved_count": len(rewritten)},
        )


__all__ = ["ExperienceDeployment"]
