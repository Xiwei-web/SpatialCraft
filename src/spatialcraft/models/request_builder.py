"""Validated construction of multimodal model requests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from spatialcraft.schemas import TaskSample

from .capabilities import validate_request_capabilities
from .interfaces import (
    ContentPart,
    GenerationSettings,
    MessageRole,
    ModelMessage,
    ModelRequest,
    ResponseToolCall,
    ToolDefinition,
)
from .registry import ModelConfig, RoleConfig


class RequestBuilder:
    """Fluent request builder that applies model and role defaults once."""

    def __init__(
        self,
        model: ModelConfig,
        *,
        role: RoleConfig | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if role is not None and model.alias not in {
            role.model,
            *role.fallback_models,
        }:
            raise ValueError(
                f"model {model.alias!r} is not configured for role {role.role!r}"
            )
        self.model = model
        self.role = role
        self._messages: list[ModelMessage] = []
        self._tools: list[ToolDefinition] = []
        self._settings = model.generation.overlay(role.generation if role else None)
        self._tool_choice: str | None = None
        self._parallel_tool_calls = True
        self._metadata = dict(metadata or {})
        if role is not None:
            self._metadata.setdefault("role", role.role)
            if role.system_prompt:
                self.system(role.system_prompt)

    def message(self, message: ModelMessage) -> RequestBuilder:
        self._messages.append(message)
        return self

    def system(self, text: str) -> RequestBuilder:
        return self.message(ModelMessage.text(MessageRole.SYSTEM, text))

    def developer(self, text: str) -> RequestBuilder:
        return self.message(ModelMessage.text(MessageRole.DEVELOPER, text))

    def user(self, text: str, *, media: Iterable[ContentPart] = ()) -> RequestBuilder:
        content = (ContentPart.text_part(text), *tuple(media))
        return self.message(ModelMessage(role=MessageRole.USER, content=content))

    def assistant(
        self,
        text: str | None = None,
        *,
        tool_calls: Iterable[ResponseToolCall] = (),
    ) -> RequestBuilder:
        content = (ContentPart.text_part(text),) if text else ()
        return self.message(
            ModelMessage(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tuple(tool_calls),
            )
        )

    def tool_result(
        self,
        *,
        call_id: str,
        output: str,
        tool_name: str | None = None,
    ) -> RequestBuilder:
        return self.message(
            ModelMessage.text(
                MessageRole.TOOL,
                output,
                tool_call_id=call_id,
                name=tool_name,
            )
        )

    def tools(self, definitions: Iterable[ToolDefinition]) -> RequestBuilder:
        self._tools.extend(definitions)
        return self

    def tool_choice(
        self, choice: str | None, *, parallel: bool = True
    ) -> RequestBuilder:
        self._tool_choice = choice
        self._parallel_tool_calls = parallel
        return self

    def settings(self, **overrides: Any) -> RequestBuilder:
        self._settings = self._settings.overlay(overrides)
        return self

    def metadata(self, **values: Any) -> RequestBuilder:
        self._metadata.update(values)
        return self

    def task(
        self,
        task: TaskSample,
        *,
        include_choices: bool = True,
        include_reference_answer: bool = False,
    ) -> RequestBuilder:
        """Append a leak-safe multimodal user message from a task schema."""

        text = task.question
        if include_choices and task.choices:
            choices = "\n".join(
                f"{index}. {choice}" for index, choice in enumerate(task.choices, 1)
            )
            text = f"{text}\n\nChoices:\n{choices}"
        if include_reference_answer and task.reference_answer is not None:
            text = f"{text}\n\nReference answer: {task.reference_answer}"
        media = tuple(
            ContentPart.image_uri(
                image.uri,
                mime_type=image.media_type,
            )
            for image in task.images
        )
        self._metadata.update(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "task_fingerprint": task.fingerprint,
            }
        )
        return self.user(text, media=media)

    def build(self) -> ModelRequest:
        request = ModelRequest(
            model_alias=self.model.alias,
            messages=tuple(self._messages),
            tools=tuple(self._tools),
            settings=self._settings,
            tool_choice=self._tool_choice,
            parallel_tool_calls=self._parallel_tool_calls,
            metadata=self._metadata,
        )
        validate_request_capabilities(request, self.model.capabilities)
        if self.role is not None:
            self.model.capabilities.require(*self.role.required_capabilities)
        return request


def build_request(
    model: ModelConfig,
    messages: Iterable[ModelMessage],
    *,
    role: RoleConfig | None = None,
    tools: Iterable[ToolDefinition] = (),
    generation: Mapping[str, Any] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> ModelRequest:
    """Functional request-construction helper."""

    builder = RequestBuilder(model, role=role, metadata=metadata)
    for message in messages:
        builder.message(message)
    builder.tools(tools).tool_choice(tool_choice, parallel=parallel_tool_calls)
    if generation:
        builder.settings(**dict(generation))
    return builder.build()


def with_generation(
    request: ModelRequest, settings: GenerationSettings
) -> ModelRequest:
    """Return a request with validated replacement generation settings."""

    return replace(request, settings=settings)


__all__ = ["RequestBuilder", "build_request", "with_generation"]
