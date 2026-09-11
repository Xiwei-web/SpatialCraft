"""Resumable environment collection and frozen deployment evaluation."""

from collections import Counter

from spatialcraft.experiments.journal import RunJournal, digest
from spatialcraft.experiments.run_api_baseline import (
    encode_media,
    immutable,
    read_tasks,
    score,
    summarize,
    validate_response_model,
)
from spatialcraft.models.providers.openai_responses import OpenAIResponsesProvider
from spatialcraft.models.serialization import (
    request_to_dict,
    response_from_dict,
    response_to_dict,
)
from spatialcraft.schemas import TaskSample
from spatialcraft.storage.atomic_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    file_lock,
    read_json,
    sha256_file,
)

from .core import (
    EMBEDDING_TEXT_POLICY,
    POLICY,
    ExampleIndex,
    embedding_text,
    make_record,
    make_request,
    representatives,
    response_status,
)


def sum_usage(usages):
    """Do not misreport absent reasoning-token accounting as zero tokens."""
    return {
        key: (
            sum(row[key] for row in usages)
            if all(row.get(key) is not None for row in usages)
            else None
        )
        for key in (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_input_tokens",
        )
    }


def _generate(provider, request, media, model_id):
    response = provider.generate(encode_media(request, media))
    validate_response_model(response.model, model_id)
    response_status(
        response
    )  # Failed/nonterminal calls must not become durable successes.
    return response_to_dict(response)


def _call(journal, key, request, media, provider, model_id):
    media_hashes = {
        part.uri: media[part.uri]
        for message in request.messages
        for part in message.content
        if part.uri
    }
    wire, reused = journal.execute(
        key,
        {"request": request_to_dict(request), "media_sha256": media_hashes},
        lambda: _generate(provider, request, media, model_id),
    )
    response = response_from_dict(wire)
    validate_response_model(response.model, model_id)
    return response, reused


def _progress(root, stage, report):
    report = {**report, "stage": stage, "policy": POLICY}
    atomic_write_json(root / stage / "progress.json", report)
    atomic_write_json(root / "progress.json", report)
    print(
        {
            key: report.get(key)
            for key in ("dataset", "stage", "status", "completed", "total", "accuracy")
        },
        flush=True,
    )


def load_memory(root, binding_sha256, embedding_identity):
    memory = root / "memory"
    snapshot = read_json(memory / "snapshot.json")
    if (
        snapshot["binding_sha256"] != binding_sha256
        or snapshot["embedding_identity"] != embedding_identity
        or snapshot["policy"] != POLICY
        or snapshot["embedding_text_policy"] != EMBEDDING_TEXT_POLICY
    ):
        raise ValueError("Frozen memory configuration mismatch")
    for name in ("records.jsonl", "index.json"):
        if sha256_file(memory / name) != snapshot["files"][name]:
            raise ValueError(f"Frozen memory changed: {name}")
    import json

    records = tuple(
        json.loads(line)
        for line in (memory / "records.jsonl").read_text().splitlines()
        if line
    )
    index = read_json(memory / "index.json")
    if (
        index["embedding_identity"] != embedding_identity
        or index["embedding_text_policy"] != EMBEDDING_TEXT_POLICY
        or index["records_sha256"] != snapshot["files"]["records.jsonl"]
        or len(records) != snapshot["record_count"]
    ):
        raise ValueError("Frozen example/index identity mismatch")
    return snapshot, ExampleIndex(records, index)


def collect_environment(journal, tasks, identity, config, options, embedder, provider):
    root = journal.root
    if (root / "memory/snapshot.json").exists():
        load_memory(root, journal.binding_digest, embedder.identity)
        return read_json(root / "results/environment.json")
    records, usages = [], []
    total = len(tasks) * options["environment_rollouts"]
    for task_index, task in enumerate(tasks):
        for rollout_index in range(options["environment_rollouts"]):
            request = make_request(
                task, config, stage="environment", rollout_index=rollout_index
            )
            response, _ = _call(
                journal,
                f"environment/tasks/{task_index:05d}/rollouts/{rollout_index:02d}/model",
                request,
                identity["media"],
                provider,
                config.model_id,
            )
            record = make_record(task, response, rollout_index)
            records.append(record)
            usages.append(response_to_dict(response)["usage"])
            # Make each lightweight record visible before the final corpus is frozen.
            immutable(
                root / "environment/records" / f"{record['record_id']}.json",
                canonical_json_bytes(record),
            )
            _progress(
                root,
                "environment",
                {
                    "dataset": task.dataset,
                    "status": "collecting",
                    "completed": len(records),
                    "total": total,
                },
            )
    selected = representatives(records)
    if not selected:
        raise ValueError(
            "No complete, nonempty environment outputs are available for RAG"
        )
    memory = root / "memory"
    immutable(
        memory / "records.jsonl", b"".join(canonical_json_bytes(r) for r in records)
    )
    records_hash = sha256_file(memory / "records.jsonl")
    texts = tuple(embedding_text(TaskSample.from_dict(r["task"])) for r in selected)
    index, _ = journal.execute(
        "environment/build_index",
        {
            "embedding_identity": embedder.identity,
            "records_sha256": records_hash,
            "text_policy": EMBEDDING_TEXT_POLICY,
            "texts": texts,
        },
        lambda: {
            "embedding_identity": embedder.identity,
            "embedding_text_policy": EMBEDDING_TEXT_POLICY,
            "records_sha256": records_hash,
            "record_ids": [r["record_id"] for r in selected],
            "dimensions": embedder.dimensions,
            "vectors": embedder.embed(texts),
        },
    )
    ExampleIndex(records, index)
    immutable(memory / "index.json", canonical_json_bytes(index))
    report = {
        "policy": POLICY,
        "dataset": tasks[0].dataset,
        "status": "completed",
        "tasks": len(tasks),
        "rollouts_per_task": options["environment_rollouts"],
        "model_calls": len(records),
        "record_count": len(records),
        "retrievable_tasks": len(selected),
        "response_status_counts": dict(Counter(r["status"] for r in records)),
        "usage": sum_usage(usages),
        "correctness_filter": False,
        "environment_labels_used": False,
    }
    immutable(root / "results/environment.json", canonical_json_bytes(report))
    # Snapshot is the final commit: deployment cannot use a partially built corpus.
    immutable(
        memory / "snapshot.json",
        canonical_json_bytes(
            {
                "policy": POLICY,
                "binding_sha256": journal.binding_digest,
                "model_id": config.model_id,
                "record_count": len(records),
                "retrievable_tasks": len(selected),
                "embedding_identity": embedder.identity,
                "embedding_text_policy": EMBEDDING_TEXT_POLICY,
                "files": {
                    name: sha256_file(memory / name)
                    for name in ("records.jsonl", "index.json")
                },
            }
        ),
    )
    _progress(root, "environment", {**report, "completed": total, "total": total})
    return report


def deploy(journal, public, private, identity, config, options, embedder, provider):
    root = journal.root
    snapshot, index = load_memory(root, journal.binding_digest, embedder.identity)
    snapshot_hash = sha256_file(root / "memory/snapshot.json")
    rows = []
    texts = tuple(embedding_text(task) for task in public)
    # Cache all queries in batches, then commit each retrieval and response separately.
    query_vectors, _ = journal.execute(
        "deployment/query_embeddings",
        {
            "embedding_identity": embedder.identity,
            "text_policy": EMBEDDING_TEXT_POLICY,
            "texts": texts,
            "memory_sha256": snapshot_hash,
        },
        lambda: {"vectors": embedder.embed(texts)},
    )
    if len(query_vectors["vectors"]) != len(public):
        raise ValueError("Deployment query vector count mismatch")
    for number, (task, reference, vector) in enumerate(
        zip(public, private, query_vectors["vectors"], strict=True)
    ):
        excluded = index.exact_matches(task, identity["media"])
        hits = index.retrieve(task, vector, options["top_k"], media=identity["media"])
        if not hits:
            raise ValueError(
                "No same-dataset prior task is available for this deployment task"
            )
        key = f"deployment/tasks/{number:05d}"
        retrieval, _ = journal.execute(
            key + "/retrieval",
            {
                "task_id": task.task_id,
                "memory_sha256": snapshot_hash,
                "query_sha256": digest(vector),
                "top_k": options["top_k"],
            },
            lambda task=task, hits=hits, excluded=excluded: {
                "task_id": task.task_id,
                "excluded_exact_match_record_ids": list(excluded),
                "examples": [
                    {
                        "record_id": hit["record"]["record_id"],
                        "task_id": hit["record"]["task_id"],
                        "cosine": hit["cosine"],
                    }
                    for hit in hits
                ],
            },
        )
        if [h["record"]["record_id"] for h in hits] != [
            r["record_id"] for r in retrieval["examples"]
        ]:
            raise ValueError("Retrieval changed on resume")
        request = make_request(task, config, stage="deployment", examples=hits)
        response, _ = _call(
            journal,
            key + "/model",
            request,
            identity["media"],
            provider,
            config.model_id,
        )
        result, _ = journal.execute(
            key + "/score",
            {
                "response_sha256": digest(response_to_dict(response)),
                "private_task_sha256": digest(reference.to_dict()),
            },
            lambda reference=reference, response=response, retrieval=retrieval: {
                **score(reference, response),
                "retrieval": retrieval["examples"],
                "excluded_exact_match_record_ids": retrieval[
                    "excluded_exact_match_record_ids"
                ],
                "memory_sha256": snapshot_hash,
            },
        )
        rows.append(result)
        report = {
            **summarize(task.dataset, rows, len(public)),
            "policy": POLICY,
            "completed": len(rows),
            "usage": sum_usage([row["usage"] for row in rows]),
        }
        _progress(root, "deployment", report)

    # Query caches/results are separate files; the memory must remain unchanged.
    if sha256_file(root / "memory/snapshot.json") != snapshot_hash:
        raise ValueError("Memory snapshot changed during deployment")
    load_memory(root, journal.binding_digest, embedder.identity)
    report = {
        **summarize(public[0].dataset, rows, len(public)),
        "policy": POLICY,
        "model": config.model_id,
        "reasoning_effort": config.generation.reasoning_effort,
        "max_output_tokens": config.generation.max_output_tokens,
        "model_calls": len(rows),
        "top_k": options["top_k"],
        "retrievable_tasks": snapshot["retrievable_tasks"],
        "memory_sha256": snapshot_hash,
        "deployment_memory_updates": 0,
        "usage": sum_usage([row["usage"] for row in rows]),
    }
    atomic_write_bytes(
        root / "results/predictions.jsonl",
        b"".join(canonical_json_bytes(row) for row in rows),
        mode=0o600,
    )
    atomic_write_json(root / "results/deployment.json", report)
    return report


def run_dataset(
    output,
    name,
    identity,
    base_binding,
    environment_config,
    deployment_config,
    options,
    embedder,
    stage,
    provider=None,
):
    root = output / name
    with file_lock(root / ".rag.lock", timeout=0):
        journal = RunJournal(
            root, {**base_binding, "dataset": name, "data_sha256": digest(identity)}
        )
        data = output / "data" / name
        public, private = (
            read_tasks(data / "public.jsonl"),
            read_tasks(data / "private.jsonl"),
        )
        environment = read_tasks(data / "environment.jsonl")
        provider = provider or OpenAIResponsesProvider(deployment_config)
        result = {"dataset": name}
        try:
            if stage in {"all", "environment"}:
                result["environment"] = collect_environment(
                    journal,
                    environment,
                    identity,
                    environment_config,
                    options,
                    embedder,
                    provider,
                )
            if stage in {"all", "deployment"}:
                result["deployment"] = deploy(
                    journal,
                    public,
                    private,
                    identity,
                    deployment_config,
                    options,
                    embedder,
                    provider,
                )
        except Exception as exc:
            previous = (
                read_json(root / "progress.json")
                if (root / "progress.json").exists()
                else {}
            )
            atomic_write_json(
                root / "progress.json",
                {
                    **previous,
                    "dataset": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                },
            )
            raise
        return result
