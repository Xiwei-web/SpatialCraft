"""Scored historical spans must identify the action actually executed."""

import pytest

from spatialcraft.agent.action_parser import ActionParser
from spatialcraft.agent.decision import parse_complete_action
from spatialcraft.models import (
    MessageRole,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ResponseToolCall,
)
from spatialcraft.tools import create_mock_tool_registry


def native(query):
    return (
        '<tool_call><function=detect><parameter=image_uri>/tmp/image.png</parameter><parameter=queries>["'
        + query
        + '"]</parameter></function></tool_call>'
    )


def run(text, target, *, truncated=False, tool_calls=()):
    tools = create_mock_tool_registry()
    request = ModelRequest(
        model_alias="test",
        messages=(ModelMessage.text(MessageRole.USER, "question"),),
        tools=tools.definitions(),
        metadata={"protocol_version": "spatialcraft_v2"},
    )
    response = ModelResponse(
        provider="test",
        model="test",
        text=text,
        tool_calls=tool_calls,
        finish_reason="length" if truncated else "stop",
        raw={
            "generated_text": text,
            "action_target": {
                "status": "recorded",
                "text": target,
                "token_ids": [1, 2],
                "prefix": "earlier generated prefix",
                "prefix_token_ids": [3],
            },
        },
    )
    return parse_complete_action(ActionParser(tools), response, request)


def assert_unscorable(action, response):
    for raw in (response.raw, action.raw_response):
        assert raw["action_target"]["status"] == "unavailable"
        assert raw["action_target"]["reason"] == "selected_action_mismatch"
        assert (
            "text" not in raw["action_target"]
            and "token_ids" not in raw["action_target"]
        )


def test_truncation_salvage_earlier_action_does_not_score_later_span():
    first, later = native("cup"), native("chair")
    action, response = run(
        "first analysis</think>\n"
        + first
        + "\nsecond analysis</think>\n"
        + later
        + "\n<tool_call>",
        later,
        truncated=True,
    )
    assert action.tool_calls[0].arguments["queries"] == ["cup"]
    assert_unscorable(action, response)


def test_matching_tool_span_survives_different_generated_call_ids():
    tool = native("cup")
    calls = (
        ResponseToolCall(
            name="detect",
            arguments={"image_uri": "/tmp/image.png", "queries": ["cup"]},
            call_id="actual-call",
        ),
    )
    action, response = run(tool, tool, tool_calls=calls)
    assert action.tool_calls[0].call_id == "actual-call"
    assert response.raw["action_target"]["status"] == "recorded"
    assert response.raw["action_target"]["token_ids"] == [1, 2]


def test_tool_argument_mismatch_preserves_execution_but_disables_scoring():
    action, response = run(native("cup"), native("chair"))
    assert action.tool_calls[0].arguments["queries"] == ["cup"]
    assert_unscorable(action, response)


@pytest.mark.parametrize(
    "text,target",
    [
        ("Final Answer: A", "Final Answer: B"),
        ("Final Answer: A", native("cup")),
        ("Final Answer: A", "<tool_call>broken"),
    ],
)
def test_final_type_text_and_invalid_span_mismatches_are_unscorable(text, target):
    action, response = run(text, target)
    assert action.final_answer == "Final Answer: A"
    assert_unscorable(action, response)


@pytest.mark.parametrize("text", ["Final Answer: A", "A", '{"final_answer":"A"}'])
def test_matching_final_and_json_envelope_remain_scoreable(text):
    action, response = run(text, text)
    assert response.raw["action_target"]["status"] == "recorded"
    assert action.raw_response["action_target"]["token_ids"] == [1, 2]
