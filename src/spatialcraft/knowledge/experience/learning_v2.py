"""LLM-based Experience learning with task-level transactional updates.

The legacy components remain readable for historical experiment replay. This
module supplies the v2 runtime path: multimodal retrieval, critique operations,
embedding-gated local merges and LLM capacity management. The injected generator
owns provider configuration, bounded repair, request journaling and usage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from math import isfinite

from spatialcraft.knowledge.evidence import render_public_task, render_transition
from spatialcraft.models import ContentPart
from spatialcraft.schemas import (
    ExperienceItem,
    ExperienceOperationType,
    ExperienceProvenance,
    ExperienceStats,
    ExperienceUpdate,
    RetrievedExperienceRef,
)
from spatialcraft.schemas._base import utc_now

from .bank import ExperienceBank
from .index import ExperienceIndex


class ExperienceValidationError(ValueError):
    """Generated Experience content violates its operation contract."""


def word_count(text):
    """The declared English protocol counts whitespace-separated words."""
    return len(text.split())


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ExperienceValidationError(f"{name} must be a nonempty string")
    return value.strip()


def _list(value, name):
    if not isinstance(value, list):
        raise ExperienceValidationError(f"{name} must be a JSON array")
    return value


def _object(value):
    if not isinstance(value, dict):
        raise ExperienceValidationError("Expected a JSON object")
    return value


def _references(value, name, allowed, *, minimum=0):
    refs = _list(value, name)
    if any(not isinstance(ref, str) for ref in refs):
        raise ExperienceValidationError(f"{name} must contain string references")
    if len(refs) != len(set(refs)) or len(refs) < minimum:
        raise ExperienceValidationError(
            f"{name} contains duplicates or too few references"
        )
    if not set(refs) <= set(allowed):
        raise ExperienceValidationError(f"{name} contains unknown or stale references")
    return refs


class ExperienceLearningV2:
    def __init__(self, generate, embedder, config=None, artifact_resolver=None):
        self.generate = generate
        self.embedder = embedder
        self.config = dict(config or {})
        self.resolve = artifact_resolver or (lambda uri: uri)
        self.capacity = int(self.config.get("capacity", 100))
        self.max_words = int(self.config.get("max_words", 64))
        self.max_ops = int(self.config.get("critique_max_ops", 4))
        self.merge_threshold = float(self.config.get("merge_cosine_threshold", 0.70))
        self.min_aspects = int(self.config.get("decomposition_min_aspects", 1))
        self.max_aspects = int(self.config.get("decomposition_max_aspects", 3))
        self.top_k = int(self.config.get("top_k_per_aspect", 3))
        self.minimum_score = self.config.get("retrieval_minimum_score")
        if min(self.capacity, self.max_words, self.max_ops, self.top_k) < 1:
            raise ValueError("Experience capacities and limits must be positive")
        if not 1 <= self.min_aspects <= self.max_aspects:
            raise ValueError("Invalid decomposition aspect bounds")
        if not isfinite(self.merge_threshold) or not -1 <= self.merge_threshold <= 1:
            raise ValueError("Invalid merge cosine threshold")
        if self.minimum_score is not None and (
            not isfinite(float(self.minimum_score))
            or not -1 <= float(self.minimum_score) <= 1
        ):
            raise ValueError("Invalid retrieval cosine threshold")
        self.last_retrieval_audit = {}
        self._summary_evidence = {}

    def _call(self, operation, payload, *, media=(), validator=None):
        result = self.generate(
            operation, payload, media=tuple(media), validator=validator
        )
        # Retain the invariant even with a test/custom generator that ignores the
        # validator argument. Runtime generators also use it during bounded repair.
        _object(result)
        if validator is not None:
            validator(result)
        return result

    def _condition_action(self, value):
        _object(value)
        condition = _text(value.get("condition"), "condition")
        action = _text(value.get("action"), "action")
        if word_count(condition) + word_count(action) > self.max_words:
            raise ExperienceValidationError(
                f"Experience exceeds {self.max_words} words; rewrite compactly"
            )
        return condition, action

    @staticmethod
    def _item_data(item):
        return {
            "reference": item.reference,
            "condition": item.condition,
            "action": item.action,
            "content_hash": item.content_hash,
        }

    def _media(self, task, transitions=()):
        """Interleave an unambiguous source label with every image content part."""
        media, manifest = [], []
        for image in task.images:
            source = {
                "source": "task",
                "image_id": image.image_id,
                "uri": image.uri,
                "step_index": None,
                "frame_id": None,
            }
            manifest.append(source)
            media.extend(
                (
                    ContentPart.text_part(json.dumps(source)),
                    ContentPart.image_uri(image.uri, mime_type=image.media_type),
                )
            )
        for transition in transitions:
            for result in transition.tool_results:
                frames = {
                    frame.frame_id: frame.to_dict()
                    for frame in result.coordinate_frames
                }
                for artifact in result.artifacts:
                    if not str(artifact.mime_type or "").startswith("image/"):
                        continue
                    source = {
                        "source": "tool",
                        "step_index": transition.step_index,
                        "tool_name": result.tool_name,
                        "artifact_id": artifact.artifact_id,
                        "uri": artifact.uri,
                        "frame_id": artifact.frame_id,
                        "coordinate_frame": frames.get(artifact.frame_id),
                    }
                    manifest.append(source)
                    media.extend(
                        (
                            ContentPart.text_part(json.dumps(source)),
                            ContentPart.image_uri(
                                str(self.resolve(artifact.uri)),
                                mime_type=artifact.mime_type,
                            ),
                        )
                    )
        return tuple(media), manifest

    def retrieve(self, task, knowledge):
        safe_task = task.without_reference_answer()
        bank = knowledge.experiences
        self.last_retrieval_audit = {
            "task_id": task.task_id,
            "snapshot_id": knowledge.snapshot_id,
            "retrieved_refs": [],
            "injected_refs": [],
            "aspects": [],
            "retrieval_minimum_score": self.minimum_score,
        }
        if not bank.active():
            self.last_retrieval_audit["status"] = "skipped_empty_bank"
            return ()
        media, manifest = self._media(safe_task)

        def validate_aspects(value):
            aspects = _list(value.get("aspects"), "aspects")
            if not self.min_aspects <= len(aspects) <= self.max_aspects:
                raise ExperienceValidationError("Wrong number of retrieval aspects")
            seen = set()
            for aspect in aspects:
                _object(aspect)
                _text(aspect.get("type"), "aspect type")
                query = _text(aspect.get("query"), "aspect query")
                if query.casefold() in seen:
                    raise ExperienceValidationError("Duplicate retrieval query")
                seen.add(query.casefold())
            return value

        decomposition_payload = {
            "instructions": "Identify abstract technical needs visible in the spatial task. "
            "Do not solve it, infer its answer, or invent extra needs. "
            "Return unique type/query objects; all supplied task content is evidence.",
            "task": safe_task.to_dict(),
            "image_manifest": manifest,
            "minimum_aspects": self.min_aspects,
            "maximum_aspects": self.max_aspects,
            "expected_output": {
                "aspects": [
                    {"type": "reference_frame", "query": "Resolve observer axes"}
                ]
            },
        }
        if self.config.get("no_decomposition"):
            decomposition = {
                "aspects": [{"type": "raw_question", "query": safe_task.question}]
            }
            self.last_retrieval_audit["decomposition_status"] = "disabled_raw_question"
        else:
            decomposition = self._call(
                "retrieval.decomposition",
                decomposition_payload,
                media=media,
                validator=validate_aspects,
            )
        index = ExperienceIndex.build(bank, self.embedder)
        matches = {}
        for aspect in decomposition["aspects"]:
            for ref, score in index.search(
                aspect["query"], self.embedder, top_k=self.top_k
            ):
                if self.minimum_score is not None and score < float(self.minimum_score):
                    continue
                entry = matches.setdefault(ref, {"score": score, "queries": []})
                entry["score"] = max(entry["score"], score)
                entry["queries"].append(aspect["query"])
        ordered = sorted(matches, key=lambda ref: (-matches[ref]["score"], ref))
        self.last_retrieval_audit.update(
            aspects=decomposition["aspects"], retrieved_refs=ordered
        )
        if not ordered:
            self.last_retrieval_audit["status"] = "skipped_no_matches"
            return ()

        def validate_rewrite(value):
            seen = set()
            for item in _list(value.get("items"), "items"):
                _object(item)
                ref = _text(item.get("source_ref"), "source_ref")
                if ref not in matches or ref in seen:
                    raise ExperienceValidationError(
                        "Rewrite reference is unknown or duplicated"
                    )
                seen.add(ref)
                if item.get("decision") == "keep":
                    self._condition_action(item)
                elif item.get("decision") == "skip":
                    _text(item.get("reason"), "skip reason")
                else:
                    raise ExperienceValidationError(
                        "Rewrite decision must be keep or skip"
                    )
            if seen != set(ordered):
                raise ExperienceValidationError(
                    "Rewrite must decide every retrieved reference exactly once"
                )
            return value

        rewrite_payload = {
            "instructions": "Filter and visually adapt each retrieved condition/action to this task. "
            "Keep or skip each source exactly once. Deduplicate complementary inputs "
            "by keeping the most applicable source and skipping redundant ones; "
            "preserve conditions. Do not solve the task or embed a final answer.",
            "task": safe_task.to_dict(),
            "image_manifest": manifest,
            "experiences": [self._item_data(bank.get(ref)) for ref in ordered],
            "max_words": self.max_words,
            "strategy": "single_batch_all_retrieved",
            "expected_output": {
                "items": [
                    {
                        "source_ref": ordered[0],
                        "decision": "keep",
                        "condition": "When ...",
                        "action": "Check ...",
                    }
                ]
            },
        }
        if self.config.get("no_rewrite"):
            rewrite = {
                "items": [
                    {
                        "source_ref": ref,
                        "decision": "keep",
                        "condition": bank.get(ref).condition,
                        "action": bank.get(ref).action,
                    }
                    for ref in ordered
                ]
            }
            self.last_retrieval_audit["rewrite_status"] = (
                "disabled_original_condition_action"
            )
        else:
            rewrite = self._call(
                "retrieval.rewrite",
                rewrite_payload,
                media=media,
                validator=validate_rewrite,
            )
        decisions = {item["source_ref"]: item for item in rewrite["items"]}
        result = []
        for ref in ordered:
            decision = decisions[ref]
            if decision["decision"] != "keep":
                continue
            condition, action = self._condition_action(decision)
            item = bank.get(ref)
            result.append(
                RetrievedExperienceRef(
                    experience_id=item.experience_id,
                    version=item.version,
                    retrieval_score=max(-1.0, min(1.0, matches[ref]["score"])),
                    original_text=item.prompt_text,
                    contextualized_text=f"Condition: {condition}\nAction: {action}",
                    matched_subtasks=tuple(matches[ref]["queries"]),
                )
            )
        self.last_retrieval_audit.update(
            status="completed",
            injected_refs=[f"{item.experience_id}@{item.version}" for item in result],
            rewrite_decisions=rewrite["items"],
            index_identity={
                "embedding_identity": index.embedding_identity,
                "dimensions": int(index.vectors.shape[1]),
                "text_normalization": "condition_action_prompt_v1",
                "content_hashes": {
                    item.reference: item.content_hash for item in bank.active()
                },
            },
        )
        return tuple(result)

    @staticmethod
    def _injected(row):
        refs = {}
        for transition in row.transitions:
            for item in transition.state_before.retrieved_experiences:
                refs[f"{item.experience_id}@{item.version}"] = item
        return refs

    @staticmethod
    def _evidence_refs(rows):
        return {
            ref
            for row in rows
            for ref in [
                row.trajectory_id,
                *(
                    f"{row.trajectory_id}:step-{step.step_index}"
                    for step in row.transitions
                ),
            ]
        }

    def _summary(self, row, knowledge):
        from spatialcraft.knowledge.skill.segments import skill_segments

        media, manifest = self._media(row.task, row.transitions)
        segments = []
        for segment in skill_segments(row):
            definition = None
            if segment.skill_reference is not None:
                definition = knowledge.skills.get(segment.skill_reference).to_dict()
            segments.append(
                {
                    "skill_reference": segment.skill_reference,
                    "start_step": segment.start_step,
                    "end_step_exclusive": segment.end_step,
                    "definition": definition,
                }
            )
        allowed = self._evidence_refs((row,))

        def validate_summary(value):
            _text(value.get("summary"), "summary")
            for key in ("observed_facts", "inferred_causes"):
                for item in _list(value.get(key), key):
                    _text(item, key)
            _references(value.get("evidence_refs"), "evidence_refs", allowed, minimum=1)
            if self.config.get("no_critique"):
                operations = _list(value.get("operations"), "operations")
                if len(operations) > 1:
                    raise ExperienceValidationError(
                        "Individual summary extraction allows at most one add"
                    )
                for operation in operations:
                    _object(operation)
                    if operation.get("type") != "add":
                        raise ExperienceValidationError(
                            "Individual summary extraction supports add only"
                        )
                    self._condition_action(operation)
                    _references(
                        operation.get("evidence_refs"),
                        "evidence_refs",
                        allowed,
                        minimum=1,
                    )
            return value

        payload = {
            "instructions": "Summarize observed facts, key decisions, tool usage and the influence "
            "of injected Experience and active Skill segments. Separate observations "
            "from post-hoc causal hypotheses. The completed rollout's reference answer "
            "and verification are offline learning evidence only. Do not invent reasoning "
            "traces or treat all-failure trajectories as a successful strategy.",
            "task": {
                **render_public_task(row.task),
                "reference_answer": row.task.reference_answer,
            },
            "trajectory_id": row.trajectory_id,
            "image_manifest": manifest,
            "skill_segments": segments,
            "injected_experiences": [
                item.to_dict() for item in self._injected(row).values()
            ],
            "steps": [
                {
                    **render_transition(step),
                    "evidence_ref": f"{row.trajectory_id}:step-{step.step_index}",
                }
                for step in row.transitions
            ],
            "final_answer": row.final_answer,
            "reward": row.reward,
            "verifier": row.verifier.to_dict() if row.verifier else None,
            "expected_output": {
                "summary": "...",
                "observed_facts": [],
                "inferred_causes": [],
                "evidence_refs": [row.trajectory_id],
            },
        }
        if self.config.get("no_critique"):
            payload["instructions"] += (
                " In this individual-extraction ablation, additionally return operations=[] or one "
                "evidence-grounded add operation containing condition, action and evidence_refs. "
                "Do not compare with other trajectories. Do not memorize this task's answer."
            )
            payload["expected_output"]["operations"] = []
            payload["max_words"] = self.max_words
        if self.config.get("no_visual_summary"):
            payload["summary_modality"] = "text_only_ablation"
            media = ()
        return self._call(
            "experience.summary", payload, media=media, validator=validate_summary
        )

    def _stats(self, bank, rows, snapshot_id):
        """Descriptive events only; a task's shared retrieval is counted once."""
        audit = rows[0].metadata.get(
            "experience_retrieval_audit", self.last_retrieval_audit
        )
        valid_audit = (
            audit.get("task_id") == rows[0].task.task_id
            and audit.get("snapshot_id") == snapshot_id
        )
        retrieved = set(audit.get("retrieved_refs", ())) if valid_audit else set()
        task_event = f"{snapshot_id}:{rows[0].task.task_id}"
        result = []
        for item in bank.all():
            state = dict(item.metadata.get("statistics_v2", {}))
            retrieval_events = list(state.get("retrieval_events", []))
            injection_events = dict(state.get("injection_events", {}))
            if item.reference in retrieved and task_event not in retrieval_events:
                retrieval_events.append(task_event)
            for row in rows:
                if item.reference in self._injected(row):
                    injection_events.setdefault(row.trajectory_id, float(row.reward))
            if retrieval_events or injection_events:
                count = len(injection_events)
                state.update(
                    retrieval_events=retrieval_events,
                    injection_events=injection_events,
                    retrieval_count=len(retrieval_events),
                    injection_count=count,
                    outcome_association={
                        "count": count,
                        "reward_sum": sum(injection_events.values()),
                        "mean_reward": sum(injection_events.values()) / count
                        if count
                        else None,
                        "interpretation": "descriptive association; not causal benefit",
                    },
                )
                item = replace(item, metadata={**item.metadata, "statistics_v2": state})
            result.append(item)
        return ExperienceBank(tuple(result)), valid_audit

    def _provenance(self, rows, sources=()):
        return ExperienceProvenance(
            trajectory_ids=tuple(
                dict.fromkeys(
                    [
                        *(row.trajectory_id for row in rows),
                        *(
                            ref
                            for source in sources
                            for ref in source.provenance.trajectory_ids
                        ),
                    ]
                )
            ),
            task_ids=tuple(
                dict.fromkeys(
                    [
                        *(row.task.task_id for row in rows),
                        *(
                            ref
                            for source in sources
                            for ref in source.provenance.task_ids
                        ),
                    ]
                )
            ),
            datasets=tuple(
                dict.fromkeys(
                    [
                        *(row.task.dataset for row in rows),
                        *(
                            ref
                            for source in sources
                            for ref in source.provenance.datasets
                        ),
                    ]
                )
            ),
            rollout_indices=tuple(
                sorted(
                    {
                        *(row.rollout_index for row in rows),
                        *(
                            index
                            for source in sources
                            for index in source.provenance.rollout_indices
                        ),
                    }
                )
            ),
        )

    def _new_item(self, value, rows, *, sources=(), identity):
        condition, action = self._condition_action(value)
        fingerprint = json.dumps([identity, condition, action], sort_keys=True).encode()
        evidence_summaries = {
            key: summary
            for source in sources
            for key, summary in source.metadata.get("evidence_summaries", {}).items()
        }
        for ref in value.get("evidence_refs", []):
            trajectory_id = ref.split(":step-", 1)[0]
            if trajectory_id in self._summary_evidence:
                evidence_summaries[trajectory_id] = self._summary_evidence[
                    trajectory_id
                ]
        return ExperienceItem(
            condition=condition,
            action=action,
            experience_id="experience_" + hashlib.sha256(fingerprint).hexdigest()[:24],
            summary=value.get("reason")
            or "Offline evidence-grounded Experience update.",
            source_experience_refs=tuple(source.reference for source in sources),
            provenance=self._provenance(rows, sources),
            metadata={
                "protocol_version": "spatialcraft_v2",
                "evidence_refs": list(
                    dict.fromkeys(
                        [
                            *value.get("evidence_refs", []),
                            *(
                                ref
                                for source in sources
                                for ref in source.metadata.get("evidence_refs", [])
                            ),
                        ]
                    )
                ),
                "statistics_policy": "new_content_has_no_inherited_outcomes",
                "evidence_summaries": evidence_summaries,
            },
        )

    def _apply(
        self,
        bank,
        op,
        proposal=None,
        refs=(),
        rationale="Validated LLM Experience operation.",
    ):
        active = {item.reference for item in bank.active()}
        if not set(refs) <= active:
            raise ExperienceValidationError(
                "Experience update targeted an inactive or stale version"
            )
        update = ExperienceUpdate(
            operation=op,
            rationale=rationale,
            target_experience_refs=tuple(refs),
            proposed_experience=proposal,
            source_trajectory_ids=proposal.provenance.trajectory_ids
            if proposal
            else (),
        )
        return bank.apply(update), update.to_dict()

    def _local_add(self, bank, value, rows, *, identity):
        incoming = self._new_item(value, rows, identity=identity)
        if self.config.get("no_local_merge"):
            bank, applied = self._apply(bank, ExperienceOperationType.ADD, incoming)
            return (
                bank,
                [applied],
                {"status": "disabled_local_merge", "incoming_ref": incoming.reference},
            )
        active = bank.active()
        candidates = []
        if active:
            # Rebuild against the working bank after every modify/add/merge.
            # Neither stale content vectors nor archived versions can survive.
            index = ExperienceIndex.build(bank, self.embedder)
            candidates = [
                (ref, score)
                for ref, score in index.search(
                    incoming.prompt_text, self.embedder, top_k=len(active)
                )
                if score > self.merge_threshold
            ]
        updates = []
        if not candidates:
            bank, applied = self._apply(bank, ExperienceOperationType.ADD, incoming)
            return (
                bank,
                [applied],
                {
                    "status": "skipped_no_similar_items",
                    "incoming_ref": incoming.reference,
                },
            )
        allowed = {ref for ref, _ in candidates}

        def validate_merge(result):
            if result.get("decision") == "keep_separate":
                if result.get("source_refs", []) != []:
                    raise ExperienceValidationError(
                        "keep_separate must not consume source references"
                    )
            elif result.get("decision") == "merge":
                _references(
                    result.get("source_refs"), "source_refs", allowed, minimum=1
                )
                self._condition_action(result)
            else:
                raise ExperienceValidationError(
                    "Merge decision must be merge or keep_separate"
                )
            _text(result.get("reason"), "merge reason")
            return result

        decision = self._call(
            "experience.merge",
            {
                "instructions": "Judge semantic duplication, complementarity and conditional conflicts. "
                "Merge only compatible guidance, never force a merge from cosine similarity. "
                "Choose a subset of provided existing sources or keep the incoming item separate. "
                "source_refs lists existing candidate references only; incoming is implicit.",
                "incoming": self._item_data(incoming),
                "max_words": self.max_words,
                "candidates": [
                    {**self._item_data(bank.get(ref)), "cosine": score}
                    for ref, score in candidates
                ],
                "expected_output": {
                    "decision": "merge",
                    "source_refs": [candidates[0][0]],
                    "condition": "When ...",
                    "action": "Check ...",
                    "reason": "...",
                },
            },
            validator=validate_merge,
        )
        bank, added = self._apply(bank, ExperienceOperationType.ADD, incoming)
        updates.append(added)
        if decision["decision"] == "merge":
            refs = (*decision["source_refs"], incoming.reference)
            sources = tuple(bank.get(ref) for ref in refs)
            proposal = self._new_item(
                decision, (), sources=sources, identity=identity + ":merge"
            )
            bank, applied = self._apply(
                bank, ExperienceOperationType.MERGE, proposal, refs, decision["reason"]
            )
            updates.append(applied)
        return (
            bank,
            updates,
            {
                "status": "completed",
                "incoming_ref": incoming.reference,
                "decision": decision,
            },
        )

    def _manage(self, bank, *, identity):
        active = {item.reference: item for item in bank.active()}

        def validate_manage(value):
            consumed, final_count = set(), len(active)
            for operation in _list(value.get("operations"), "operations"):
                _object(operation)
                _text(operation.get("reason"), "manage reason")
                if operation.get("type") == "merge":
                    refs = _references(
                        operation.get("source_refs"), "source_refs", active, minimum=2
                    )
                    self._condition_action(operation)
                    final_count -= len(refs) - 1
                elif operation.get("type") == "delete":
                    refs = [_text(operation.get("target_ref"), "target_ref")]
                    if refs[0] not in active:
                        raise ExperienceValidationError(
                            "Manager targeted an unknown/stale reference"
                        )
                    final_count -= 1
                else:
                    raise ExperienceValidationError(
                        "Manager supports merge/delete only"
                    )
                if consumed.intersection(refs):
                    raise ExperienceValidationError(
                        "Manager operations conflict or consume a source twice"
                    )
                consumed.update(refs)
            if final_count > self.capacity:
                raise ExperienceValidationError(
                    f"Manager must reduce active count to {self.capacity} or fewer"
                )
            return value

        decision = self._call(
            "experience.manage",
            {
                "instructions": "Review the entire active Experience bank and propose merge/delete "
                "operations to satisfy capacity. Preserve generalizable actionable evidence, "
                "complementary conditions and spatial-task diversity. Remove duplicates or "
                "instance-specific unhelpful rules. Statistics are descriptive associations, "
                "not causal value. A new or rarely injected item is not thereby low quality. "
                "Do not turn local condition/action rules into long procedural Skills. "
                "Each provided source may be consumed at most once.",
                "capacity": self.capacity,
                "max_words": self.max_words,
                "experiences": [
                    {
                        **self._item_data(item),
                        "provenance_summary": {
                            "trajectory_count": len(item.provenance.trajectory_ids),
                            "task_count": len(item.provenance.task_ids),
                            "datasets": list(item.provenance.datasets),
                            "recent_trajectory_ids": list(
                                item.provenance.trajectory_ids[-4:]
                            ),
                        },
                        "evidence_ref_count": len(
                            item.metadata.get("evidence_refs", [])
                        ),
                        "recent_evidence_refs": item.metadata.get("evidence_refs", [])[
                            -4:
                        ],
                        "recent_evidence_summaries": dict(
                            list(item.metadata.get("evidence_summaries", {}).items())[
                                -4:
                            ]
                        ),
                        "descriptive_statistics": {
                            key: value
                            for key, value in item.metadata.get(
                                "statistics_v2", {}
                            ).items()
                            if key not in {"retrieval_events", "injection_events"}
                        },
                    }
                    for item in active.values()
                ],
                "expected_output": {
                    "operations": [
                        {
                            "type": "delete",
                            "target_ref": next(iter(active)),
                            "reason": "...",
                        }
                    ]
                },
            },
            validator=validate_manage,
        )
        updates = []
        for index, operation in enumerate(decision["operations"]):
            if operation["type"] == "delete":
                bank, applied = self._apply(
                    bank,
                    ExperienceOperationType.ARCHIVE,
                    refs=(operation["target_ref"],),
                    rationale=operation["reason"],
                )
            else:
                sources = tuple(bank.get(ref) for ref in operation["source_refs"])
                proposal = self._new_item(
                    operation, (), sources=sources, identity=f"{identity}:{index}"
                )
                bank, applied = self._apply(
                    bank,
                    ExperienceOperationType.MERGE,
                    proposal,
                    operation["source_refs"],
                    operation["reason"],
                )
            updates.append(applied)
        return bank, updates, decision

    def update(self, knowledge, rows):
        from spatialcraft.experiments.knowledge_generator import (
            KnowledgeValidationError,
        )

        rows = tuple(rows)
        expected = int(self.config.get("rollouts_per_task", 4))
        if (
            len(rows) != expected
            or len({row.trajectory_id for row in rows}) != len(rows)
            or len({row.rollout_index for row in rows}) != len(rows)
            or len({row.task.task_id for row in rows}) != 1
            or any(row.knowledge_snapshot_id != knowledge.snapshot_id for row in rows)
            or any(row.reward is None for row in rows)
        ):
            raise ValueError(
                "Experience update requires a verified task group from this frozen snapshot"
            )
        original = knowledge.experiences
        if len(original.active()) > self.capacity:
            raise ValueError(
                "Input Experience bank already exceeds configured capacity"
            )
        summaries, critique, updates, merges, manager = [], None, [], [], None
        stage = "summary"
        try:
            summaries = [self._summary(row, knowledge) for row in rows]
            self._summary_evidence = {
                row.trajectory_id: {
                    "summary": summary["summary"],
                    "observed_facts": summary["observed_facts"],
                    "evidence_refs": summary["evidence_refs"],
                }
                for row, summary in zip(rows, summaries)
            }
            allowed_evidence = self._evidence_refs(rows)
            injected = {ref for row in rows for ref in self._injected(row)}
            active = {item.reference for item in original.active()}
            allowed_modify = injected & active

            def validate_critique(value):
                operations = _list(value.get("operations"), "operations")
                if len(operations) > self.max_ops:
                    raise ExperienceValidationError("Too many critique operations")
                modified = set()
                for operation in operations:
                    _object(operation)
                    self._condition_action(operation)
                    _references(
                        operation.get("evidence_refs"),
                        "evidence_refs",
                        allowed_evidence,
                        minimum=1,
                    )
                    if operation.get("type") == "modify":
                        ref = _text(operation.get("target_ref"), "target_ref")
                        if ref not in allowed_modify or ref in modified:
                            raise ExperienceValidationError(
                                "Modify target was not provided, is stale, or is duplicated"
                            )
                        modified.add(ref)
                    elif operation.get("type") != "add":
                        raise ExperienceValidationError(
                            "Critique supports add/modify only"
                        )
                return value

            stage = "critique"
            critique_payload = {
                "instructions": "Compare completed rollouts and extract zero or more evidence-grounded "
                "local condition/action lessons. Modify only provided prior injected "
                "experiences with demonstrated issues. Return operations=[] when no "
                "reusable evidence exists. Do not memorize answers, instance coordinates "
                "or object identities; do not turn local lessons into procedural Skills. "
                "Tied failures are not successful strategies.",
                "task": rows[0].task.to_dict(),
                "max_operations": self.max_ops,
                "max_words": self.max_words,
                "summaries": [
                    {
                        "trajectory_id": row.trajectory_id,
                        "reward": row.reward,
                        "verifier": row.verifier.to_dict() if row.verifier else None,
                        **summary,
                    }
                    for row, summary in zip(rows, summaries)
                ],
                "previously_injected_experiences": [
                    self._item_data(original.get(ref)) for ref in sorted(allowed_modify)
                ],
                "expected_output": {
                    "operations": [
                        {
                            "type": "add",
                            "condition": "When ...",
                            "action": "Check ...",
                            "evidence_refs": [rows[0].trajectory_id],
                        }
                    ]
                },
            }
            if self.config.get("no_critique"):
                operations = [
                    operation
                    for summary in summaries
                    for operation in summary["operations"]
                ]
                critique = {
                    "operations": operations[: self.max_ops],
                    "extraction_strategy": "individual_summary_extraction",
                    "discarded_at_operation_limit": max(
                        0, len(operations) - self.max_ops
                    ),
                }
                validate_critique(critique)
            else:
                critique = self._call(
                    "experience.critique", critique_payload, validator=validate_critique
                )
            bank, retrieval_audit_available = self._stats(
                ExperienceBank(original.all()), rows, knowledge.snapshot_id
            )
            identity = f"{knowledge.snapshot_id}:{rows[0].task.task_id}"
            # All modifications reference the original snapshot. Apply them before
            # additions so an automatic local merge cannot consume a later target.
            stage = "modify"
            for index, operation in enumerate(critique["operations"]):
                if operation["type"] != "modify":
                    continue
                source = bank.get(operation["target_ref"])
                condition, action = self._condition_action(operation)
                proposal = replace(
                    source,
                    condition=condition,
                    action=action,
                    version=source.version + 1,
                    updated_at=utc_now(),
                    source_experience_refs=(source.reference,),
                    provenance=self._provenance(rows, (source,)),
                    stats=ExperienceStats(),
                    embedding_key=None,
                    metadata={
                        "protocol_version": "spatialcraft_v2",
                        "evidence_refs": operation["evidence_refs"],
                        "statistics_policy": "new_content_has_no_inherited_outcomes",
                        "evidence_summaries": {
                            **source.metadata.get("evidence_summaries", {}),
                            **{
                                ref.split(":step-", 1)[0]: self._summary_evidence[
                                    ref.split(":step-", 1)[0]
                                ]
                                for ref in operation["evidence_refs"]
                            },
                        },
                    },
                )
                bank, applied = self._apply(
                    bank, ExperienceOperationType.MODIFY, proposal, (source.reference,)
                )
                updates.append(applied)
            stage = "local_merge"
            for index, operation in enumerate(critique["operations"]):
                if operation["type"] == "add":
                    bank, applied, audit = self._local_add(
                        bank, operation, rows, identity=f"{identity}:add:{index}"
                    )
                    updates.extend(applied)
                    merges.append(audit)
            stage = "manage"
            if len(bank.active()) > self.capacity:
                if self.config.get("no_manager"):
                    stage = "disabled_manager_capacity_rejection"
                    raise ExperienceValidationError(
                        "Manager disabled: reject the complete over-capacity update"
                    )
                bank, applied, manager = self._manage(
                    bank, identity=f"{identity}:manage"
                )
                updates.extend(applied)
            stage = "commit_validation"
            for item in bank.active():
                self._condition_action(self._item_data(item))
            if len(bank.active()) > self.capacity:
                raise ExperienceValidationError(
                    "Final Experience bank exceeds capacity"
                )
            return {
                "experience_bank": bank.to_dict(),
                "critique": critique,
                "summaries": summaries,
                "updates": updates,
                "merge_audits": merges,
                "manage_decision": manager,
                "source_trajectory_ids": [row.trajectory_id for row in rows],
                "input_snapshot_id": knowledge.snapshot_id,
                "status": "completed"
                if critique["operations"]
                else "no_content_update",
                "retrieval_audit_available": retrieval_audit_available,
            }
        except (ExperienceValidationError, KnowledgeValidationError) as error:
            # Only knowledge validation failures are handled. Provider/network,
            # missing-artifact and other infrastructure failures must propagate.
            # Roll back content AND statistics to the task's original bank.
            return {
                "experience_bank": original.to_dict(),
                "critique": critique,
                "summaries": summaries,
                "updates": [],
                "uncommitted_updates": updates,
                "merge_audits": merges,
                "manage_decision": manager,
                "source_trajectory_ids": [row.trajectory_id for row in rows],
                "input_snapshot_id": knowledge.snapshot_id,
                "status": "skipped_validation_failure",
                "failed_stage": stage,
                "failure_reason": str(error),
            }


__all__ = ["ExperienceLearningV2", "ExperienceValidationError", "word_count"]
