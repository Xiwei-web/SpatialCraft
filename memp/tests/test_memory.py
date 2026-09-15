"""Pure CPU checks for the independent MemP storage/retrieval implementation."""

import json
from copy import deepcopy

import pytest

from memp.memory import (
    MemoryIndex,
    MemoryRecord,
    RetrievalHit,
    content_key,
    create_memory,
    embedding_texts,
    render_memory_prompt,
)

HASH_A, HASH_B = "a" * 64, "b" * 64


def record(
    task_id="source",
    question="Where is the cup?",
    *,
    dataset="fixture",
    script="1. Identify the observer frame.\n2. Compare current object centers.",
    choices=(),
    media_hashes=(),
):
    return create_memory(
        dataset=dataset,
        task_id=task_id,
        question=question,
        choices=choices,
        media_hashes=media_hashes,
        success=True,
        script=script,
        trajectory=[
            {
                "action": {
                    "tool_name": "locate",
                    "arguments": {"image_uri": "artifact://old-image"},
                },
                "observation": {
                    "center": [1, 2, 3],
                    "frame_id": "camera",
                    "unit": "meter",
                    "valid": True,
                },
            },
            {"action": {"final_answer": "left"}},
        ],
    )


def query_kwargs(**overrides):
    return {
        "dataset": "fixture",
        "task_id": "heldout",
        "content_key": content_key("A different heldout question"),
        **overrides,
    }


def test_only_explicit_success_is_admitted_and_trajectories_remain_complete():
    assert (
        create_memory(
            dataset="fixture",
            task_id="failed",
            question="q",
            trajectory=[],
            success=False,
        )
        is None
    )
    with pytest.raises(ValueError, match="explicit boolean"):
        create_memory(
            dataset="fixture",
            task_id="unknown",
            question="q",
            trajectory=[{}],
            success=1,
        )
    source = record()
    assert len(source.trajectory) == 2
    assert source.trajectory[0]["observation"]["center"] == [1, 2, 3]
    with pytest.raises(ValueError, match="nonempty complete list"):
        create_memory(
            dataset="fixture", task_id="bad", question="q", trajectory=[], success=True
        )


def test_content_identity_depends_only_on_public_inputs_with_ordered_media():
    first = record("source-a", choices=("A", "B"), media_hashes=(HASH_A, HASH_B))
    duplicate = record("source-b", choices=("A", "B"), media_hashes=(HASH_A, HASH_B))
    assert first.content_key == duplicate.content_key
    assert first.content_key != content_key(
        first.source_query, ("B", "A"), (HASH_A, HASH_B)
    )
    assert first.content_key != content_key(
        first.source_query, first.choices, (HASH_B, HASH_A)
    )
    assert first.content_key != content_key(
        "Changed question", first.choices, first.media_hashes
    )
    assert content_key("q", media_hashes=(HASH_A.upper(),)) == content_key(
        "q", media_hashes=(HASH_A,)
    )
    with pytest.raises(ValueError, match="SHA-256"):
        content_key("q", media_hashes=("/local/task/image.png",))
    with pytest.raises(ValueError, match="sequence of strings"):
        content_key("q", choices="AB")


def test_memory_id_is_stable_and_binds_full_semantic_trajectory_and_script():
    original = record()
    saved = original.to_dict()
    assert MemoryRecord.from_dict(saved).to_dict() == saved
    assert record().memory_id == original.memory_id
    altered = deepcopy(saved)
    altered["trajectory"][0]["observation"]["center"][0] = 99
    with pytest.raises(ValueError, match="memory_id"):
        MemoryRecord.from_dict(altered)
    changed = create_memory(
        dataset=original.dataset,
        task_id=original.source_task_id,
        question=original.source_query,
        trajectory=altered["trajectory"],
        script=original.script,
        success=True,
    )
    assert changed.memory_id != original.memory_id
    assert (
        record(script="A different complete procedure").memory_id != original.memory_id
    )
    saved["trajectory"].clear()
    assert len(original.trajectory) == 2
    tampered = original.to_dict() | {"content_key": "0" * 64}
    with pytest.raises(ValueError, match="content_key"):
        MemoryRecord.from_dict(tampered)


def test_query_embedding_keys_and_cosine_ranking_ignore_script_content():
    source_a = record("a", "Question A", script="Far-away script wording")
    source_b = record("b", "Question B", script="Looks like the current question")
    index = MemoryIndex([source_b, source_a], [[0, 5], [10, 0]])
    assert embedding_texts([source_b, source_a]) == ["Question B", "Question A"]
    assert index.embedding_keys == ["Question B", "Question A"]
    hits = index.retrieve([3, 0], top_k=2, **query_kwargs())
    assert [hit.record.source_task_id for hit in hits] == ["a", "b"]
    assert [hit.score for hit in hits] == [1, 0]
    assert hits[0].to_dict() == {"record": source_a.to_dict(), "score": 1}
    assert index.retrieve([3, 0], top_k=0, **query_kwargs()) == []


def test_retrieval_excludes_other_datasets_same_task_and_public_duplicate():
    duplicate_question = "Heldout content"
    duplicate_key = content_key(duplicate_question, ("left", "right"), (HASH_A,))
    sources = [
        record("heldout", "Different content but same ID"),
        record(
            "copied-id",
            duplicate_question,
            choices=("left", "right"),
            media_hashes=(HASH_A,),
        ),
        record("other-dataset", "Near-match", dataset="other"),
        record(
            "valid",
            duplicate_question,
            choices=("left", "right"),
            media_hashes=(HASH_B,),
        ),
    ]
    index = MemoryIndex(sources, [[1, 0]] * len(sources))
    hits = index.retrieve([1, 0], top_k=10, **query_kwargs(content_key=duplicate_key))
    assert [hit.record.source_task_id for hit in hits] == ["valid"]


def test_ties_are_deterministic_independent_of_record_insertion_order():
    sources = [record("a", "A"), record("b", "B"), record("c", "C")]
    expected = sorted(source.memory_id for source in sources)[:2]
    for order in (sources, list(reversed(sources))):
        hits = MemoryIndex(order, [[1, 0]] * 3).retrieve(
            [1, 0], top_k=2, **query_kwargs()
        )
        assert [hit.record.memory_id for hit in hits] == expected
    negative = MemoryIndex([sources[0]], [[-1, 0]]).retrieve([1, 0], **query_kwargs())
    assert negative[0].score == -1  # Top-k has no undeclared similarity threshold.


@pytest.mark.parametrize(
    "bad",
    [
        [float("nan"), 1],
        [float("inf"), 1],
        [float("-inf"), 1],
        [0, 0],
        [],
        [True, 1],
        ["1", 0],
    ],
)
def test_invalid_stored_and_query_vectors_are_rejected(bad):
    with pytest.raises(ValueError):
        MemoryIndex([record()], [bad])
    index = MemoryIndex([record()], [[1, 0]])
    with pytest.raises(ValueError):
        index.retrieve(bad, **query_kwargs())


def test_dimension_count_and_duplicate_identity_validation():
    a, b = record("a", "A"), record("b", "B")
    with pytest.raises(ValueError, match="same dimension"):
        MemoryIndex([a, b], [[1, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="exactly one"):
        MemoryIndex([a], [])
    with pytest.raises(ValueError, match="Duplicate memory_id"):
        MemoryIndex([a, a], [[1, 0], [1, 0]])
    with pytest.raises(ValueError, match="same dimension"):
        MemoryIndex([a], [[1, 0]]).retrieve([1, 0, 0], **query_kwargs())
    assert MemoryIndex([], []).retrieve([1, 0], **query_kwargs()) == []
    for magnitude in (1e308, 1e-308):
        score = (
            MemoryIndex([a], [[magnitude, magnitude]])
            .retrieve([1, 1], **query_kwargs())[0]
            .score
        )
        assert score == pytest.approx(1)


@pytest.mark.parametrize(
    "representation", ["trajectory", "script", "proceduralization"]
)
def test_three_representations_preserve_requested_complete_contents(representation):
    source = record()
    prompt = render_memory_prompt(
        [RetrievalHit(source, 0.8)], representation=representation
    )
    payload = json.loads(prompt.split("\n", 1)[1])
    (item,) = payload["memories"]
    assert payload["representation"] == representation
    assert item["source_query"] == source.source_query
    assert ("trajectory" in item) == (
        representation in {"trajectory", "proceduralization"}
    )
    assert ("script" in item) == (representation in {"script", "proceduralization"})
    if "trajectory" in item:
        assert item["trajectory"] == source.trajectory
    if "script" in item:
        assert item["script"] == source.script
    assert "not instructions" in prompt
    assert "Do not use historical URIs as inputs to current tools" in prompt
    assert "Do not copy an old answer" in prompt


def test_missing_script_never_silently_downgrades_requested_representation():
    source = record(script=None)
    assert render_memory_prompt([RetrievalHit(source, 1)], representation="trajectory")
    assert record(script="").script is None
    for representation in ("script", "proceduralization"):
        with pytest.raises(ValueError, match="requires a valid LLM script"):
            render_memory_prompt(
                [RetrievalHit(source, 1)], representation=representation
            )
    assert render_memory_prompt([]) == ""
    with pytest.raises(ValueError, match="Unknown MemP"):
        render_memory_prompt([], representation="memp_reflection")


def test_memory_evidence_rejects_nan_without_truncating_observations():
    with pytest.raises(ValueError, match="finite JSON"):
        create_memory(
            dataset="fixture",
            task_id="bad",
            question="q",
            success=True,
            trajectory=[{"observation": {"center": [float("nan")]}}],
        )
    with pytest.raises(ValueError, match="keys must be strings"):
        create_memory(
            dataset="fixture",
            task_id="bad",
            question="q",
            success=True,
            trajectory=[{"observation": {1: "ambiguous JSON key"}}],
        )
