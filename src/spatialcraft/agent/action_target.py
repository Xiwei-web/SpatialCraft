"""Locate a generated action at real tokenizer boundaries without re-tokenization."""

import json
import re
from hashlib import sha256


def capture_action_target(text, token_ids, tokenizer):
    ids = list(token_ids)

    def decode(values):
        return tokenizer.decode(
            values, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    if decode(ids) != text:
        return {"status": "unavailable", "reason": "decoder_mismatch"}
    offset = text.rfind("</think>")
    offset = offset + len("</think>") if offset >= 0 else 0
    visible = text[offset:]
    block = re.search(r"<tool_call>.*?</tool_call>", visible, re.DOTALL)
    if block:
        start, end, kind = offset + block.start(), offset + block.end(), "tool"
    else:
        final = re.search(r"(?im)^[ \t]*Final Answer:[ \t]*[^\r\n]+", visible)
        if final:
            start, end, kind = offset + final.start(), offset + final.end(), "final"
        else:
            leading = len(visible) - len(visible.lstrip())
            try:
                envelope, length = json.JSONDecoder().raw_decode(visible.lstrip())
            except (ValueError, TypeError):
                envelope, length = None, None
            if isinstance(envelope, dict) and (
                envelope.get("tool_calls")
                or isinstance(envelope.get("final_answer"), str)
            ):
                start, end = offset + leading, offset + leading + length
                kind = "tool" if envelope.get("tool_calls") else "final"
            elif visible.strip() and not visible.lstrip().startswith(
                ("{", "[", "<tool_call", "<function", "<parameter")
            ):
                # The executor accepts an ordinary textual final answer. Score
                # that exact continuation, including any visible explanation.
                start, end, kind = offset, offset + len(visible.rstrip()), "final"
            else:
                return {"status": "unavailable", "reason": "no_delimited_action"}
    # Boundary whitespace may be part of the same token as the opening marker.
    # Locate the nearest enclosing token boundaries and preserve that whitespace.
    boundaries = [(0, 0)]
    for index in range(1, len(ids) + 1):
        decoded = decode(ids[:index])
        if text.startswith(decoded):
            boundaries.append((len(decoded), index))
    left = max(
        (pair for pair in boundaries if pair[0] <= start), key=lambda x: (x[0], -x[1])
    )
    right = min(
        (pair for pair in boundaries if pair[0] >= end),
        key=lambda x: (x[0], x[1]),
        default=None,
    )
    if right is None or text[left[0] : start].strip() or text[end : right[0]].strip():
        return {"status": "unavailable", "reason": "action_boundary_not_token_aligned"}
    target_ids = ids[left[1] : right[1]]
    target = text[left[0] : right[0]]
    if not target_ids or decode(target_ids) != target:
        return {"status": "unavailable", "reason": "target_decoder_mismatch"}
    return {
        "status": "recorded",
        "action_type": kind,
        "text": target,
        "token_ids": target_ids,
        "prefix": text[: left[0]],
        "prefix_token_ids": ids[: left[1]],
        "token_start": left[1],
        "token_end": right[1],
        "generated_text_sha256": sha256(text.encode()).hexdigest(),
        "tokenizer_id": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
    }
