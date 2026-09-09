import json

import pytest

from spatialcraft.experiments.learning import _json_object


def test_normal_objects_and_fences_unchanged(capsys):
    obj = {"policy": ['Use "detect"', "Check coordinates"], "terminate": False}
    for text in (json.dumps(obj), "```json\n" + json.dumps(obj) + "\n```"):
        assert _json_object(text) == obj
    assert capsys.readouterr().out == ""


def test_literal_controls_preserved_and_audited(capsys):
    assert _json_object('{"policy":"first\nsecond\tend"}') == {
        "policy": "first\nsecond\tend"
    }
    assert "literal_string_controls" in capsys.readouterr().out


def test_missing_terminal_quote_without_inventing_content(capsys):
    text = '{\n "diagnosis": "Good.",\n "termination": "Stop after independent verification.\n}'
    assert _json_object(text) == {
        "diagnosis": "Good.",
        "termination": "Stop after independent verification.",
    }
    assert "missing_terminal_value_quote" in capsys.readouterr().out


@pytest.mark.parametrize(
    "text",
    [
        '{"a":"unfinished',
        '{"a":1,}',
        '{"a":"missing quote, "b":2}',
        '{"a":NaN}',
        '{"a":1,"a":2}',
        "[]",
        'prefix {"a":1}',
        '{"a":1} suffix',
        "```json",
        '{"a":true',
        '{"a":["text\n}',
    ],
)
def test_ambiguous_or_incomplete_outputs_remain_errors(text):
    with pytest.raises((ValueError, TypeError)):
        _json_object(text)
