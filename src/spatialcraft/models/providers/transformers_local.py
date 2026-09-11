"""Lazy local Hugging Face Transformers inference and scoring provider."""

from __future__ import annotations

import inspect
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

from ..capabilities import Capability, validate_request_capabilities
from ..interfaces import (
    ContentKind,
    ModelConfigurationError,
    ModelProvider,
    ModelRequest,
    ModelRequestError,
    ModelResponse,
    ModelResponseError,
    SequenceScore,
    TokenUsage,
)
from ..registry import ModelConfig, ProviderKind
from ..response_parser import parse_generated_text
from .offload import repair_qwen_cpu_offload


class TransformersLocalProvider(ModelProvider):
    """Run a local causal or multimodal model with lazy weight loading."""

    provider_name = ProviderKind.TRANSFORMERS_LOCAL.value
    _sampling_lock = threading.Lock()

    def __init__(
        self,
        config: ModelConfig,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        if config.provider is not ProviderKind.TRANSFORMERS_LOCAL:
            raise ModelConfigurationError(
                f"TransformersLocalProvider cannot serve {config.provider.value}"
            )
        self.config = config
        self._model = model
        self._processor = processor
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ModelConfigurationError(
                "install PyTorch for transformers_local"
            ) from exc
        return torch

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor
        with self._load_lock:
            if self._model is not None and self._processor is not None:
                return self._model, self._processor
            assert self.config.local is not None
            try:
                from transformers import (
                    AutoModelForCausalLM,
                    AutoModelForImageTextToText,
                    AutoProcessor,
                )
            except ImportError as exc:
                raise ModelConfigurationError(
                    "install 'transformers' for transformers_local"
                ) from exc
            path = Path(self.config.local.path).expanduser()
            if not path.exists():
                raise ModelConfigurationError(f"local model path not found: {path}")
            processor_path = self.config.local.tokenizer_path or str(path)
            common: dict[str, Any] = {
                "revision": self.config.local.revision,
                "trust_remote_code": self.config.local.trust_remote_code,
            }
            common = {key: value for key, value in common.items() if value is not None}
            self._processor = AutoProcessor.from_pretrained(processor_path, **common)

            torch = self._torch()
            model_kwargs: dict[str, Any] = {
                **common,
                "device_map": self.config.local.device_map,
                **self.config.local.extra_load_kwargs,
            }
            if self.config.local.dtype != "auto":
                try:
                    model_kwargs["torch_dtype"] = getattr(
                        torch, self.config.local.dtype
                    )
                except AttributeError as exc:
                    raise ModelConfigurationError(
                        f"unknown torch dtype: {self.config.local.dtype}"
                    ) from exc
            else:
                model_kwargs["torch_dtype"] = "auto"
            if self.config.local.load_in_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise ModelConfigurationError(
                        "4-bit loading requires transformers BitsAndBytes support"
                    ) from exc
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True
                )
            # AutoModel can return a feature-only backbone without generate().
            # Select a generation head explicitly for native vision-language models.
            loader = (
                AutoModelForImageTextToText
                if self.config.capabilities.supports(Capability.IMAGE_INPUT)
                else AutoModelForCausalLM
            )
            self._model = loader.from_pretrained(path, **model_kwargs)
            repair_qwen_cpu_offload(self._model)
            self._model.eval()
        return self._model, self._processor

    @staticmethod
    def _tools(request: ModelRequest) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]

    @staticmethod
    def _messages(request: ModelRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for message in request.messages:
            content: str | list[dict[str, Any]]
            if all(part.kind is ContentKind.TEXT for part in message.content):
                content = message.text_content
            else:
                parts: list[dict[str, Any]] = []
                for part in message.content:
                    if part.kind is ContentKind.TEXT:
                        parts.append({"type": "text", "text": part.text})
                    elif part.kind is ContentKind.IMAGE:
                        parts.append(
                            {
                                "type": "image",
                                "image": part.as_data_uri(),
                                **(
                                    {"mime_type": part.mime_type}
                                    if part.mime_type
                                    else {}
                                ),
                            }
                        )
                    elif part.kind is ContentKind.VIDEO:
                        parts.append({"type": "video", "video": part.as_data_uri()})
                    else:
                        raise ModelRequestError(
                            f"local adapter does not encode {part.kind.value} input"
                        )
                content = parts
            item: dict[str, Any] = {
                "role": message.role.value,
                "content": content,
            }
            if message.name:
                item["name"] = message.name
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments,
                        },
                    }
                    for call in message.tool_calls
                ]
            messages.append(item)
        # Native Qwen templates accept one leading system message, not developer
        # roles or multiple system messages. Preserve every instruction in order.
        instructions = []
        while messages and messages[0]["role"] in {"system", "developer"}:
            instructions.append(messages.pop(0)["content"])
        if any(m["role"] in {"system", "developer"} for m in messages):
            raise ModelRequestError(
                "Local system/developer instructions must precede user messages"
            )
        if instructions:
            messages.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
        return messages

    def _render(
        self, processor: Any, request: ModelRequest
    ) -> tuple[str, list[Any], list[Any]]:
        messages = self._messages(request)
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        overrides = dict(request.metadata.get("chat_template_kwargs") or {})
        if set(overrides) & {
            "tokenize",
            "add_generation_prompt",
            "tools",
            "chat_template",
        }:
            raise ModelRequestError(
                "chat_template_kwargs cannot replace managed fields"
            )
        template_kwargs.update(overrides)
        if request.tools:
            template_kwargs["tools"] = self._tools(request)
        try:
            rendered = processor.apply_chat_template(messages, **template_kwargs)
        except (AttributeError, TypeError) as exc:
            raise ModelRequestError(
                "local processor does not support the configured chat/tool template"
            ) from exc
        images: list[Any] = []
        videos: list[Any] = []
        if any(
            part.kind in {ContentKind.IMAGE, ContentKind.VIDEO}
            for message in request.messages
            for part in message.content
        ):
            try:
                from qwen_vl_utils import process_vision_info

                image_values, video_values = process_vision_info(messages)
                images = list(image_values or [])
                videos = list(video_values or [])
            except ImportError as exc:
                raise ModelConfigurationError(
                    "multimodal local requests require qwen-vl-utils"
                ) from exc
        return rendered, images, videos

    @staticmethod
    def _move_inputs(inputs: Any, model: Any) -> Any:
        device = getattr(model, "device", None)
        if device is None or str(device) == "meta":
            return inputs
        if isinstance(inputs, Mapping):
            return {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        if not hasattr(inputs, "to"):
            return inputs
        return inputs.to(device)

    def _prepare(self, processor: Any, model: Any, request: ModelRequest) -> Any:
        rendered, images, videos = self._render(processor, request)
        return self._move_inputs(
            self._encode(processor, rendered, images, videos), model
        )

    @staticmethod
    def _encode(processor, rendered, images, videos):
        kwargs: dict[str, Any] = {
            "text": [rendered],
            "padding": True,
            "return_tensors": "pt",
        }
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        try:
            inputs = processor(**kwargs)
        except TypeError:
            kwargs.pop("padding", None)
            inputs = processor(**kwargs)
        return inputs

    def generate(self, request: ModelRequest) -> ModelResponse:
        if request.model_alias != self.config.alias:
            raise ModelRequestError(
                f"request targets {request.model_alias}, provider serves {self.config.alias}"
            )
        validate_request_capabilities(request, self.config.capabilities)
        if request.tools and request.tool_choice not in {None, "auto"}:
            raise ModelRequestError(
                "transformers_local currently supports only automatic tool choice"
            )
        model, processor = self._load()
        inputs = self._prepare(processor, model, request)
        settings = request.settings
        if request.metadata.get("protocol_version") == "spatialcraft_v2":
            profile = request.metadata.get("operation_profile", {})
            input_count = int(inputs["input_ids"].shape[-1])
            maximum_input = profile.get("max_input_tokens")
            model_config = getattr(model.config, "text_config", model.config)
            window = self.config.capabilities.context_window or getattr(
                model_config, "max_position_embeddings", None
            )
            if maximum_input is not None and input_count > maximum_input:
                raise ModelRequestError(
                    "Operation input exceeds its explicit context budget"
                )
            if (
                window is not None
                and input_count + (settings.max_output_tokens or 512) > window
            ):
                raise ModelRequestError(
                    "Operation input and output budgets exceed the model context window"
                )
        generation: dict[str, Any] = {
            "max_new_tokens": settings.max_output_tokens or 512,
            "do_sample": bool(settings.temperature and settings.temperature > 0),
        }
        if settings.temperature is not None and settings.temperature > 0:
            generation["temperature"] = settings.temperature
        if settings.top_p is not None and generation["do_sample"]:
            generation["top_p"] = settings.top_p
        if settings.stop:
            raise ModelRequestError(
                "transformers_local requires tokenizer-specific stopping criteria; "
                "use generation.extra"
            )
        overlap = set(generation) & set(settings.extra)
        if overlap:
            raise ModelRequestError(
                f"generation.extra cannot replace managed fields: {sorted(overlap)}"
            )
        generation.update(settings.extra)
        # Some chat checkpoints omit generation_config.json and their base
        # config EOS differs from the tokenizer's end-of-turn token. Stop at
        # either, otherwise generation can continue into fictitious turns.
        tokenizer = getattr(processor, "tokenizer", processor)
        if "eos_token_id" not in generation:
            model_eos = getattr(
                getattr(model, "generation_config", None), "eos_token_id", None
            )
            eos_ids = (
                list(model_eos)
                if isinstance(model_eos, (tuple, list))
                else ([model_eos] if model_eos is not None else [])
            )
            chat_eos = getattr(tokenizer, "eos_token_id", None)
            if chat_eos is not None and chat_eos not in eos_ids:
                eos_ids.append(chat_eos)
            if eos_ids:
                generation["eos_token_id"] = eos_ids
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            generation.setdefault("pad_token_id", pad_id)
        torch = self._torch()
        started = perf_counter()
        try:
            # Save/restore the RNG so each rollout's seed actually controls
            # sampling without changing the next rollout's or caller's RNG.
            devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            )
            with (
                self._sampling_lock,
                self._inference_lock,
                torch.inference_mode(),
                torch.random.fork_rng(devices=devices),
            ):
                if settings.seed is not None:
                    torch.manual_seed(settings.seed)
                generated = model.generate(**inputs, **generation)
        except Exception as exc:
            raise ModelRequestError(
                f"local generation failed for {self.config.alias}: {exc}"
            ) from exc
        latency_ms = (perf_counter() - started) * 1000
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        new_tokens = generated[:, prompt_tokens:]
        text = processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # A length-limited native tool call can end mid-tag. Preserve it for
        # audit, but never parse or execute an incomplete action.
        eos_ids = generation.get("eos_token_id", [])
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        ended = bool(new_tokens.shape[-1]) and int(new_tokens[0, -1]) in eos_ids
        truncated = (
            int(new_tokens.shape[-1]) >= generation["max_new_tokens"] and not ended
        )
        raw = {"generated_token_ids": new_tokens[0].tolist(), "generated_text": text}
        thinking = (
            request.metadata.get("chat_template_kwargs", {}).get("enable_thinking")
            is True
        )
        if thinking and not truncated:
            if "</think>" not in text:
                if request.metadata.get("knowledge_output"):
                    # A malformed knowledge answer still consumed generation
                    # tokens. Preserve it for accounting and bounded JSON repair.
                    raw["thinking_prefix_incomplete"] = True
                else:
                    raise ModelResponseError(
                        "Thinking output ended without closing </think>; no action can be executed"
                    )
            else:
                before, marker, after = text.partition("</think>")
                # Qwen's generation prompt already supplies '<think>\n'. Keep
                # the exact generated continuation and boundary whitespace.
                leading = after[: len(after) - len(after.lstrip())]
                raw["sampled_thinking_prefix"] = before + marker + leading
        if request.metadata.get(
            "protocol_version"
        ) == "spatialcraft_v2" and not request.metadata.get("knowledge_output"):
            from spatialcraft.agent.action_target import capture_action_target

            raw["action_target"] = capture_action_target(
                text, raw["generated_token_ids"], tokenizer
            )
        response = (
            ModelResponse
            if truncated or request.metadata.get("knowledge_output")
            else parse_generated_text
        )(
            text=text,
            provider=self.provider_name,
            model=self.config.model_id,
            latency_ms=latency_ms,
            raw=raw,
        )
        return replace(
            response,
            finish_reason="length" if truncated else "stop",
            usage=TokenUsage(
                input_tokens=prompt_tokens,
                output_tokens=int(new_tokens.shape[-1]),
            ),
        )

    def score(self, request: ModelRequest, target_text: str) -> SequenceScore:
        self.config.capabilities.require(Capability.FIXED_TARGET_SCORING)
        if request.model_alias != self.config.alias:
            raise ModelRequestError("Scoring request targets another model")
        validate_request_capabilities(request, self.config.capabilities)
        if not target_text.strip():
            raise ValueError("Scoring requires a nonempty fixed target")
        model, processor = self._load()
        rendered, images, videos = self._render(processor, request)
        prefix = request.metadata.get("fixed_scoring_prefix")
        thinking = (
            request.metadata.get("chat_template_kwargs", {}).get("enable_thinking")
            is True
        )
        if thinking and (not isinstance(prefix, str) or "</think>" not in prefix):
            raise ModelRequestError(
                "Thinking action scoring requires a completed fixed sampled prefix"
            )
        fixed_ids = request.metadata.get("fixed_target_token_ids")
        fixed_prefix_ids = request.metadata.get("fixed_scoring_prefix_token_ids", [])
        torch = self._torch()
        if fixed_ids is not None:
            tokenizer = getattr(processor, "tokenizer", processor)
            if (
                not isinstance(fixed_ids, (list, tuple))
                or not fixed_ids
                or any(type(i) is not int or i < 0 for i in fixed_ids)
            ):
                raise ModelRequestError("Invalid recorded action token IDs")
            if not isinstance(fixed_prefix_ids, (list, tuple)) or any(
                type(i) is not int or i < 0 for i in fixed_prefix_ids
            ):
                raise ModelRequestError("Invalid recorded prefix token IDs")
            decode = lambda ids: tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            if decode(fixed_ids) != target_text or decode(fixed_prefix_ids) != (
                prefix or ""
            ):
                raise ModelRequestError(
                    "Recorded target/prefix text does not match exact token IDs"
                )
            combined = dict(self._encode(processor, rendered, images, videos))
            base = combined["input_ids"]
            continuation = torch.tensor(
                [list(fixed_prefix_ids) + list(fixed_ids)],
                dtype=base.dtype,
                device=base.device,
            )
            prompt_count = int(base.shape[-1]) + len(fixed_prefix_ids)
            combined["input_ids"] = torch.cat((base, continuation), dim=-1)
            # Qwen3.5 mRoPE indexes mm_token_type_ids with attention_mask.
            # Every appended prefix/action token is text (type 0); preserve all
            # original vision markers and grids so the model computes its own
            # multimodal positions for the complete teacher-forced sequence.
            for field, fill in (
                ("attention_mask", 1),
                ("token_type_ids", 0),
                ("mm_token_type_ids", 0),
            ):
                if field not in combined:
                    continue
                values = combined[field]
                if tuple(values.shape) != tuple(base.shape):
                    raise ModelRequestError(
                        f"Processor {field} is not aligned with input_ids"
                    )
                extension = torch.full(
                    tuple(continuation.shape),
                    fill,
                    dtype=values.dtype,
                    device=values.device,
                )
                combined[field] = torch.cat((values, extension), dim=-1)
            if "position_ids" in combined:
                raise ModelRequestError(
                    "Explicit processor position_ids need a model-specific fixed continuation adapter"
                )
            input_ids = combined["input_ids"]
            target_ids = input_ids[:, prompt_count:]
        else:
            if request.metadata.get("protocol_version") == "spatialcraft_v2":
                raise ModelRequestError("v2 scoring requires recorded action token IDs")
            if prefix is not None:
                if not isinstance(prefix, str) or not thinking:
                    raise ModelRequestError(
                        "Fixed thinking prefix requires explicit thinking mode"
                    )
                rendered += prefix
            prompt = self._encode(processor, rendered, images, videos)
            combined = self._encode(processor, rendered + target_text, images, videos)
            prompt_ids = prompt["input_ids"]
            input_ids = combined["input_ids"]
            prompt_count = int(prompt_ids.shape[-1])
            if input_ids.shape[-1] <= prompt_count:
                raise ValueError(
                    "target_text must produce at least one continuation token"
                )
            if not torch.equal(input_ids[:, :prompt_count], prompt_ids):
                raise ModelRequestError(
                    "tokenizer changed the prompt boundary; require a stable continuation boundary"
                )
            target_ids = input_ids[:, prompt_count:]
        batch = self._move_inputs(combined, model)
        target_count = int(target_ids.shape[-1])
        options = {"use_cache": False}
        limited = "logits_to_keep" in inspect.signature(model.forward).parameters
        if limited:
            # Avoid allocating logits for the (potentially long) visual prompt.
            options["logits_to_keep"] = target_count + 1
        try:
            with self._inference_lock, torch.inference_mode():
                output = model(**batch, **options)
        except Exception as exc:
            raise ModelRequestError(
                f"fixed-target scoring failed for {self.config.alias}: {exc}"
            ) from exc
        logits = (
            output.logits[:, :-1, :]
            if limited
            else output.logits[:, prompt_count - 1 : -1, :]
        )
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        selected = log_probs.gather(-1, target_ids.to(log_probs.device).unsqueeze(-1))
        values = selected.squeeze(0).squeeze(-1).detach().cpu().tolist()
        ids = target_ids.squeeze(0).tolist()
        return SequenceScore(
            model=self.config.model_id,
            target_text=target_text,
            token_ids=tuple(int(item) for item in ids),
            token_logprobs=tuple(float(item) for item in values),
            prompt_token_count=prompt_count,
        )

    def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        self.config.capabilities.require(Capability.EMBEDDINGS)
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty strings")
        model, processor = self._load()
        tokenizer = getattr(processor, "tokenizer", processor)
        inputs = tokenizer(
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        inputs = self._move_inputs(inputs, model)
        torch = self._torch()
        try:
            with self._inference_lock, torch.inference_mode():
                output = model(**inputs, output_hidden_states=True, return_dict=True)
        except Exception as exc:
            raise ModelRequestError(
                f"embedding inference failed for {self.config.alias}: {exc}"
            ) from exc
        hidden = output.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        return tuple(
            tuple(float(value) for value in row) for row in pooled.cpu().tolist()
        )


__all__ = ["TransformersLocalProvider"]
