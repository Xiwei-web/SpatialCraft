"""Offline tests for MemRL selection, credit assignment and state integrity."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace

import pytest

from MemRL.memory import (
    MemoryIndex,
    MemoryRecord,
    calibrate_threshold,
    content_key,
    create_memory,
    render_memory_prompt,
    update_utilities,
)


def record(task="source", **kwargs):
    values = {
        "dataset": "robospatial",
        "source_task_id": task,
        "source_query": f"Where is the available surface for {task}?",
        "reflection": "Verify support and reference frame before choosing a location.",
        "source_key": f"environment/{task}/rollout_0",
        "media_hashes": ("a" * 64,),
    }
    values.update(kwargs)
    return MemoryRecord(**values)


def retrieve(index, query=(1.0, 0.0), **kwargs):
    values = {
        "dataset": "robospatial",
        "task_id": "current",
        "content_key": content_key("Current visual question", (), ("b" * 64,)),
    }
    values.update(kwargs)
    return index.retrieve(query, **values)


def test_record_round_trip_identity_ignores_mutable_utility():
    original = record(choices=("A. first", "B. second"))
    restored = MemoryRecord.from_dict(original.to_dict())
    updated = replace(original, q_value=0.7, visits=4)
    assert original == restored
    assert original.memory_id == updated.memory_id
    assert original.to_dict() != updated.to_dict()
    with pytest.raises(FrozenInstanceError):
        original.q_value = 0.5


def test_source_key_distinguishes_independent_rollouts():
    first = record()
    second = replace(first, source_key="environment/source/rollout_1", memory_id="")
    assert first.content_key == second.content_key
    assert first.memory_id != second.memory_id


@pytest.mark.parametrize("field,value", [
    ("source_query", "tampered"),
    ("reflection", "tampered"),
    ("source_key", "tampered"),
    ("memory_id", "memrl_wrong"),
    ("content_key", "0" * 64),
    ("schema_version", True),
    ("q_value", float("nan")),
    ("visits", -1),
])
def test_deserialization_rejects_invalid_or_inconsistent_payload(field, value):
    payload = record().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        MemoryRecord.from_dict(payload)


def test_deserialization_rejects_extra_or_missing_fields():
    payload = record().to_dict()
    with pytest.raises(ValueError):
        MemoryRecord.from_dict({**payload, "private_ground_truth": "A"})
    del payload["q_value"]
    with pytest.raises(ValueError):
        MemoryRecord.from_dict(payload)


def test_create_memory_is_not_success_filtered_or_initialized_from_source_reward():
    created = create_memory(
        dataset="sat", task_id="failed_trial", question="How did the camera move?",
        reflection="The failed attempt confused camera motion with object motion.",
        source_key="environment/failed_trial/0", q_init=0.0,
    )
    assert created.reflection.startswith("The failed")
    assert created.q_value == 0.0
    assert created.visits == 0


def test_semantic_filter_precedes_utility_reranking():
    closest = record("close", q_value=0.0)
    useful = record("useful", q_value=0.8)
    unrelated = record("unrelated", q_value=1.0)
    index = MemoryIndex(
        (closest, useful, unrelated), ((1, 0), (0.8, 0.6), (0, 1))
    )
    selected = retrieve(index, candidate_k=2, top_k=1, utility_weight=0.75)
    assert [hit.record.memory_id for hit in selected.hits] == [useful.memory_id]
    assert selected.audit["candidate_count"] == 2
    assert unrelated.memory_id not in {
        row["memory_id"] for row in selected.audit["candidates"]
    }
    winner = selected.hits[0]
    assert winner.similarity_z == pytest.approx(-1.0)
    assert winner.q_z == pytest.approx(1.0)
    assert winner.score == pytest.approx(0.5)


def test_threshold_is_strict_and_never_falls_back_to_unrelated_memory():
    index = MemoryIndex((record(),), ((1, 0),))
    assert not retrieve(index, threshold=1.0).hits
    assert not retrieve(index, query=(0, 1), threshold=0.0).hits
    assert retrieve(index, query=(0, 1), threshold=-0.1).hits


def test_self_and_exact_content_and_cross_dataset_are_excluded():
    task = "Which object is nearest?"
    opts = ("A. cup", "B. bowl")
    hashes = ("f" * 64,)
    blocked = (
        record("current"),
        record("different_id", source_query=task, choices=opts, media_hashes=hashes),
        record("other_dataset", dataset="erqa"),
    )
    valid = record("valid")
    selected = retrieve(
        MemoryIndex((*blocked, valid), [(1, 0)] * 4),
        content_key=content_key(task, opts, hashes),
    )
    assert [hit.record.memory_id for hit in selected.hits] == [valid.memory_id]
    assert selected.audit["eligible_count"] == 1


def test_constant_candidate_scores_normalize_to_zero_and_ties_are_stable():
    records = tuple(record(str(index), q_value=0.3) for index in range(10))
    forward = retrieve(MemoryIndex(records, [(1, 0)] * 10))
    reverse = retrieve(MemoryIndex(records[::-1], [(1, 0)] * 10))
    assert forward == reverse
    assert all(hit.similarity_z == hit.q_z == hit.score == 0 for hit in forward.hits)
    assert forward.audit["selected_ids"] == sorted(r.memory_id for r in records)[:3]


def test_retrieval_is_read_only_even_with_zero_or_empty_memory():
    bank = (record(q_value=0.7, visits=8),)
    before = [entry.to_dict() for entry in bank]
    index = MemoryIndex(bank, ((1, 0),))
    assert retrieve(index).hits
    assert not retrieve(index, top_k=0).hits
    assert [entry.to_dict() for entry in bank] == before
    assert not retrieve(MemoryIndex((), ())).hits


def test_credit_only_updates_actual_supplied_memories_once_and_preserves_id():
    a, b = record("a", q_value=0.2), record("b", q_value=0.7, visits=4)
    updated = update_utilities((a, b), (a.memory_id, a.memory_id), 1.0)
    assert updated[0].q_value == pytest.approx(0.44)
    assert updated[0].visits == 1
    assert updated[0].memory_id == a.memory_id
    assert updated[1] is b
    assert a.q_value == 0.2
    failed = update_utilities(updated, (a.memory_id,), 0.0)
    assert failed[0].q_value == pytest.approx(0.308)
    assert failed[0].visits == 2
    assert update_utilities((a, b), (), 0) == (a, b)


def test_unknown_credit_ids_and_duplicate_bank_records_fail():
    a = record()
    with pytest.raises(ValueError):
        update_utilities((a,), ("memrl_unknown",), 1)
    with pytest.raises(ValueError):
        update_utilities((a, a), (), 1)
    with pytest.raises(ValueError):
        MemoryIndex((a, a), ((1, 0), (1, 0)))


@pytest.mark.parametrize("reward", [-1, 1.1, True, "1", float("nan"), float("inf")])
def test_invalid_reward_rejected(reward):
    with pytest.raises(ValueError):
        update_utilities((record(),), (), reward)


@pytest.mark.parametrize("alpha", [-1, 1.1, True, "0.3", float("nan")])
def test_invalid_learning_rate_rejected(alpha):
    with pytest.raises(ValueError):
        update_utilities((record(),), (), 1.0, alpha)


def test_calibration_uses_unique_offdiagonal_pairs_and_linear_quantiles():
    vectors = ((1, 0), (1, 1), (0, 1))
    assert calibrate_threshold(vectors, 0.25) == pytest.approx(1 / (2 * math.sqrt(2)))
    assert calibrate_threshold(vectors, 0.0) == pytest.approx(0.0)
    assert calibrate_threshold(vectors, 1.0) == pytest.approx(1 / math.sqrt(2))
    assert calibrate_threshold(((1, 0),)) == -1.0
    with pytest.raises(ValueError):
        calibrate_threshold(())


@pytest.mark.parametrize("vectors", [
    ((0, 0),), ((float("nan"), 0),), ((float("inf"), 0),),
    ((True, 0),), (("1", 0),), ((1, 0), (1, 0, 0)), ((),),
])
def test_invalid_embedding_vectors_rejected_for_calibration_and_index(vectors):
    with pytest.raises(ValueError):
        calibrate_threshold(vectors)
    with pytest.raises(ValueError):
        MemoryIndex(tuple(record(str(i)) for i in range(len(vectors))), vectors)


def test_large_and_small_nonzero_vectors_normalize_without_overflow():
    index = MemoryIndex((record(),), ((1e308, 1e308),))
    selected = retrieve(index, query=(1e-300, 1e-300))
    assert selected.hits[0].similarity == pytest.approx(1.0)


def test_prompt_exposes_only_allowed_prior_fields():
    a = record(q_value=0.777, visits=93, source_key="private/journal/provenance")
    selected = retrieve(MemoryIndex((a,), ((1, 0),)))
    prompt = render_memory_prompt(selected.hits)
    assert a.source_query in prompt
    assert a.reflection in prompt
    for hidden in (a.source_key, a.memory_id, a.media_hashes[0], "q_value", "visits"):
        assert hidden not in prompt
    assert render_memory_prompt(()) == ""


def test_advance_reuses_old_vectors_and_accepts_utility_updates_without_mutation():
    original = record("first")
    appended = record("second")
    empty = MemoryIndex((), ())
    first = empty.advance((original,), (2, 0))
    changed = replace(original, q_value=0.6, visits=2)
    advanced = first.advance((changed, appended), (0, 3))
    assert empty.records == ()
    assert first.records == (original,)
    assert first.records[0].q_value == 0.0
    assert advanced.records == (changed, appended)
    assert advanced.vectors.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert first.vectors.tolist() == [[1.0, 0.0]]
    with pytest.raises(ValueError):
        advanced.vectors[0, 0] = 0.0
    assert retrieve(advanced, query=(0, 1)).hits


def test_advance_validates_only_new_vector(monkeypatch):
    from MemRL import memory

    first = record("first")
    appended = record("second")
    index = MemoryIndex((first,), ((1, 0),))
    normalize = memory._unit_vector
    calls = []

    def checked(values, dimension=None):
        calls.append(values)
        return normalize(values, dimension)

    monkeypatch.setattr(memory, "_unit_vector", checked)
    index.advance((first, appended), (0, 1))
    assert calls == [(0, 1)]


def test_advance_rejects_reordering_identity_changes_count_and_bad_vector():
    first, second, third = (record(name) for name in ("first", "second", "third"))
    index = MemoryIndex((first, second), ((1, 0), (0, 1)))
    for records, vector in (
        ((second, first, third), (1, 1)),
        ((first, third, second), (1, 1)),
        ((first, second), (1, 1)),
        ((first, second, third, record("fourth")), (1, 1)),
        ((first, second, third), (1, 0, 0)),
        ((first, second, third), (0, 0)),
    ):
        with pytest.raises(ValueError):
            index.advance(records, vector)
