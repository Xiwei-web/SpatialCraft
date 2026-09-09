"""Provider-neutral LLM-as-a-judge verifier."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from spatialcraft.models import (
    ContentPart,
    GenerationSettings,
    MessageRole,
    ModelMessage,
    ModelProvider,
    ModelRequest,
)
from spatialcraft.schemas import TaskSample, VerifierOutcome

from .base import AnswerVerifier, VerificationError, make_outcome

_SYSTEM_PROMPT = """You are a strict spatial-reasoning answer judge.
Compare the candidate only with the supplied reference and question. Return one JSON
object with keys: score (number in [0,1]), is_correct (boolean), and analysis (short
string). Do not add markdown or other text."""


def _parse_judgement(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            raise VerificationError("LLM judge did not return a JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise VerificationError("LLM judge returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise VerificationError("LLM judge response must be a JSON object")
    try:
        score = float(value["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("LLM judge response has no valid score") from exc
    if not 0.0 <= score <= 1.0:
        raise VerificationError("LLM judge score must be in [0, 1]")
    value["score"] = score
    value["is_correct"] = bool(value.get("is_correct", score >= 0.5))
    value["analysis"] = str(value.get("analysis", "")).strip() or None
    return value


class LLMJudgeVerifier(AnswerVerifier):
    """Judge ambiguous free-form answers through any configured model provider."""

    name = "llm_judge"

    def __init__(
        self,
        provider: ModelProvider,
        model_alias: str,
        *,
        include_images: bool = True,
        max_output_tokens: int = 256,
    ) -> None:
        if not model_alias.strip():
            raise ValueError("model_alias cannot be empty")
        self.provider = provider
        self.model_alias = model_alias
        self.include_images = include_images
        self.max_output_tokens = max_output_tokens

    def verify_answer(self, task: TaskSample, candidate: str) -> VerifierOutcome:
        payload = {
            "dataset": task.dataset,
            "question": task.question,
            "answer_type": task.answer_type.value,
            "choices": list(task.choices),
            "reference_answer": task.reference_answer,
            "candidate_answer": candidate,
        }
        content: list[ContentPart] = [
            ContentPart.text_part(json.dumps(payload, ensure_ascii=False, default=str))
        ]
        if self.include_images:
            for image in task.images:
                path = Path(image.uri)
                if path.is_file():
                    content.append(
                        ContentPart.image_bytes(
                            path.read_bytes(), mime_type=image.media_type or "image/png"
                        )
                    )
                else:
                    content.append(
                        ContentPart.image_uri(image.uri, mime_type=image.media_type)
                    )
        request = ModelRequest(
            model_alias=self.model_alias,
            messages=(
                ModelMessage.text(MessageRole.SYSTEM, _SYSTEM_PROMPT),
                ModelMessage(role=MessageRole.USER, content=tuple(content)),
            ),
            settings=GenerationSettings(
                max_output_tokens=self.max_output_tokens,
                temperature=0.0,
            ),
            metadata={"role": "verifier", "task_id": task.task_id},
        )
        response = self.provider.generate(request)
        if not response.text:
            raise VerificationError("LLM judge returned an empty response")
        judgement = _parse_judgement(response.text)
        return make_outcome(
            self.name,
            judgement["score"],
            is_correct=judgement["is_correct"],
            analysis=judgement["analysis"],
            details={
                "candidate": candidate,
                "judge_model": self.model_alias,
                "response_id": response.response_id,
            },
        )


__all__ = ["LLMJudgeVerifier"]
