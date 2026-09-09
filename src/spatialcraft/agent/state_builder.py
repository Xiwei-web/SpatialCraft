"""Immutable SpatialState construction for the agent execution loop."""

from __future__ import annotations

import json

from spatialcraft.schemas import (
    ActionType,
    ActiveSkillRef,
    AgentAction,
    ArtifactType,
    ConversationMessage,
    EvidenceKind,
    MessageRole,
    RetrievedExperienceRef,
    SpatialEvidence,
    SpatialState,
    TaskSample,
    ToolResult,
)

_EVIDENCE_KIND = {
    ArtifactType.IMAGE: EvidenceKind.VISUAL,
    ArtifactType.MASK: EvidenceKind.SEGMENTATION,
    ArtifactType.DEPTH: EvidenceKind.DEPTH,
    ArtifactType.BOUNDING_BOXES: EvidenceKind.DETECTION,
    ArtifactType.POSE: EvidenceKind.POSE,
    ArtifactType.OPTICAL_FLOW: EvidenceKind.MOTION,
    ArtifactType.SCENE_GRAPH: EvidenceKind.SCENE_GRAPH,
    ArtifactType.POINT_CLOUD: EvidenceKind.GEOMETRIC,
    ArtifactType.BEV: EvidenceKind.GEOMETRIC,
}


class StateBuilder:
    """Create state snapshots while preserving tool and knowledge provenance."""

    def initial(
        self,
        task: TaskSample,
        *,
        retrieved_experiences: tuple[RetrievedExperienceRef, ...] = (),
        active_skill: ActiveSkillRef | None = None,
        metadata: dict | None = None,
    ) -> SpatialState:
        return SpatialState(
            task_id=task.task_id,
            step_index=0,
            retrieved_experiences=retrieved_experiences,
            active_skill=active_skill,
            metadata={
                "dataset": task.dataset,
                "task_fingerprint": task.fingerprint,
                "reference_answer_exposed": False,
                **dict(metadata or {}),
            },
        )

    @staticmethod
    def _assistant_tool_message(action: AgentAction) -> ConversationMessage:
        calls = [
            {
                "call_id": call.call_id,
                "name": call.tool_name,
                "arguments": call.arguments,
            }
            for call in action.tool_calls
        ]
        return ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=action.reasoning_summary or "Calling spatial tools.",
            metadata={"tool_calls": calls},
        )

    @staticmethod
    def _tool_message(result: ToolResult) -> ConversationMessage:
        payload = {
            "status": result.status.value,
            "text": result.text,
            "structured_output": result.structured_output,
            "artifacts": [artifact.to_dict() for artifact in result.artifacts],
            "coordinate_frames": [
                frame.to_dict() for frame in result.coordinate_frames
            ],
            "error": result.error_message,
        }
        return ConversationMessage(
            role=MessageRole.TOOL,
            content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
            tool_call_id=result.tool_call_id,
            artifact_ids=tuple(item.artifact_id for item in result.artifacts),
            metadata={"tool_name": result.tool_name, "result_id": result.result_id},
        )

    @staticmethod
    def _evidence(result: ToolResult) -> SpatialEvidence:
        artifact_types = {artifact.artifact_type for artifact in result.artifacts}
        kind = next(
            (_EVIDENCE_KIND[item] for item in artifact_types if item in _EVIDENCE_KIND),
            EvidenceKind.OTHER,
        )
        frame_ids = {
            artifact.frame_id for artifact in result.artifacts if artifact.frame_id
        }
        return SpatialEvidence(
            kind=kind,
            summary=result.text or f"{result.tool_name} returned {result.status.value}",
            source_tool_result_id=result.result_id,
            artifact_ids=tuple(item.artifact_id for item in result.artifacts),
            frame_id=next(iter(frame_ids)) if len(frame_ids) == 1 else None,
            confidence=result.metadata.get("confidence"),
            metadata={"tool_name": result.tool_name, "status": result.status.value},
        )

    def after_tools(
        self,
        state: SpatialState,
        action: AgentAction,
        results: tuple[ToolResult, ...],
    ) -> SpatialState:
        if action.action_type is not ActionType.TOOL:
            raise ValueError("after_tools requires a tool action")
        messages = (
            *state.messages,
            self._assistant_tool_message(action),
            *(self._tool_message(result) for result in results),
        )
        evidence = (*state.evidence, *(self._evidence(result) for result in results))
        return state.next_step(
            messages=messages,
            tool_result_ids=(
                *state.tool_result_ids,
                *(result.result_id for result in results),
            ),
            evidence=evidence,
            token_count=state.token_count
            + sum(
                len(message.content.split())
                for message in messages[len(state.messages) :]
            ),
        )

    def after_final(self, state: SpatialState, action: AgentAction) -> SpatialState:
        if action.action_type is not ActionType.FINAL:
            raise ValueError("after_final requires a final action")
        message = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=action.final_answer or "",
            metadata={"final": True},
        )
        return state.next_step(
            messages=(*state.messages, message),
            token_count=state.token_count + len(message.content.split()),
        )


__all__ = ["StateBuilder"]
