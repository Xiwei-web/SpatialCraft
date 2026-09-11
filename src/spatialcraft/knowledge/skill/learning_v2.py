"""Operation-scoped, auditable Skill learning for the spatialcraft_v2 profile.

Legacy builders deliberately remain separate: old checkpoints keep their original
algorithm. This module requires semantic eligibility and recorded action spans.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256

from spatialcraft.experiments.evolution_queue import RelatedEvolutionQueue
from spatialcraft.experiments.journal import digest
from spatialcraft.knowledge.experience.index import _normalize
from spatialcraft.models import ContentPart
from spatialcraft.models.serialization import request_from_dict
from spatialcraft.schemas import (
    SkillCandidate,
    SkillEvolutionType,
    SkillItem,
    SkillStats,
)

from .credit_assignment import SkillCreditAssigner
from .pool import SkillPool
from .ppo_gate import SequenceLikelihoodExample, SequenceLikelihoodGate
from .segments import skill_segments


def _identity(value):
    return sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:24]


def _required_text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


class HistoricalActionUnavailable(ValueError):
    """Legacy/unavailable action spans cannot enter the v2 likelihood Gate."""


class SkillLearningV2:
    def __init__(
        self,
        generate,
        embedder,
        scorer,
        token_counter,
        trajectory_loader=None,
        config=None,
        resolver=None,
    ):
        self.generate, self.embedder, self.scorer = generate, embedder, scorer
        self.token_counter, self.load_trajectory = token_counter, trajectory_loader
        self.config, self.resolve = dict(config or {}), resolver or (lambda uri: uri)
        if token_counter is None:
            raise ValueError("Skill v2 needs the declared executor tokenizer")

    def _media(self, segment):
        media, seen = [], set()
        for image in segment.task.images:
            media.append(ContentPart.text_part(f"Original task image: {image.uri}"))
            media.append(ContentPart.image_uri(image.uri, mime_type=image.media_type))
            seen.add(image.uri)
        for transition in segment.transitions:
            for result in transition.tool_results:
                for artifact in result.artifacts:
                    if str(artifact.mime_type or "").startswith("image/"):
                        uri = str(self.resolve(artifact.uri))
                        if uri not in seen:
                            media.append(
                                ContentPart.text_part(
                                    f"Tool image from {transition.transition_id}, {result.tool_name}: {artifact.uri}"
                                )
                            )
                            media.append(
                                ContentPart.image_uri(uri, mime_type=artifact.mime_type)
                            )
                            seen.add(uri)
        return tuple(media)

    def _diagnose(self, row, segment, advantage, pool, queue):
        transitions = [t.transition_id for t in segment.transitions]
        key = "diagnosis_" + _identity(
            [digest(row.to_dict()), segment.skill_reference, transitions]
        )
        cached = queue.state["diagnosis_cache"].get(key)
        if cached is not None:
            return cached
        parent = pool.get(segment.skill_reference) if segment.skill_reference else None
        refs = set(transitions)

        def validate(value):
            if not isinstance(value, dict) or type(value.get("is_related")) is not bool:
                raise ValueError("Diagnosis must preserve a real is_related boolean")
            if type(value.get("no_change")) is not bool:
                raise ValueError("Diagnosis requires explicit no_change")
            _required_text(value.get("reason"), "reason")
            evidence = value.get("evidence_refs", [])
            if (
                not isinstance(evidence, list)
                or not all(isinstance(r, str) for r in evidence)
                or not set(evidence) <= refs
            ):
                raise ValueError(
                    "Diagnosis evidence must reference supplied transitions"
                )
            needs_new = value.get("needs_new_skill", False)
            if type(needs_new) is not bool or (parent is not None and needs_new):
                raise ValueError("Only real NONE segments may enter discovery")
            if (value["is_related"] or needs_new) and not evidence:
                raise ValueError(
                    "Related/discovery diagnosis needs evidence references"
                )
            if parent is None and needs_new:
                _required_text(value.get("discovery_key"), "discovery_key")
                _required_text(value.get("discovery_need"), "discovery_need")
            for field in ("initiation", "policy", "termination"):
                if value.get(field) is not None and not isinstance(value[field], str):
                    raise ValueError(f"Diagnosis {field} must be text or null")
            if (
                value["is_related"]
                and not value["no_change"]
                and not any(
                    value.get(f) for f in ("initiation", "policy", "termination")
                )
            ):
                raise ValueError("A related change requires a component update")
            return value

        payload = {
            "instructions": "Diagnose ONLY the supplied activation segment. Judge whether the Skill could affect the observed result; execution alone is not relevance, and association is not causal proof. Preserve successful components. Return no_change=true when no supported change exists. For actual NONE, compare the technical need with ALL existing Skills; set needs_new_skill only if uncovered. Reuse an existing discovery bucket key when the same need recurs. Never treat an unrelated active Skill as NONE.",
            "expected_output": {
                "is_related": "boolean",
                "no_change": "boolean",
                "reason": "text",
                "initiation": "text or null",
                "policy": "text or null",
                "termination": "text or null",
                "evidence_refs": "list of supplied transition IDs",
                "needs_new_skill": "boolean; true only for actual NONE",
                "discovery_key": "stable short technical-need key or null",
                "discovery_need": "abstract need or null",
            },
            "task_id": row.task.task_id,
            "trajectory_id": row.trajectory_id,
            "snapshot_id": row.knowledge_snapshot_id,
            "question": row.task.question,
            "choices": list(row.task.choices),
            "reward": row.reward,
            "advantage": advantage,
            "skill_ref": segment.skill_reference,
            "parent": parent.format_for_prompt() if parent else None,
            "existing_skills": [s.to_dict() for s in pool.active()]
            if parent is None
            else [],
            "discovery_buckets": queue.state["discovery_buckets"]
            if parent is None
            else {},
            "segment_start": segment.start_step,
            "segment_end": segment.end_step,
            "evidence": [
                {
                    "transition_id": t.transition_id,
                    "step": t.step_index,
                    "action": t.action.to_dict(),
                    "tool_results": [r.to_dict() for r in t.tool_results],
                }
                for t in segment.transitions
            ],
        }
        if self.config.get("no_semantic_gradient", False):
            return {
                "diagnosis_id": key,
                "trajectory_id": row.trajectory_id,
                "skill_ref": segment.skill_reference,
                "transition_ids": transitions,
                "segment_start": segment.start_step,
                "segment_end": segment.end_step,
                "reward": row.reward,
                "advantage": advantage,
                "snapshot_id": row.knowledge_snapshot_id,
                "is_related": parent is not None,
                "no_change": False,
                "reason": "Ablation: all executed segments are eligible without semantic judgment",
                "eligibility_mode": "executed_segment_ablation",
                "needs_new_skill": parent is None,
                "discovery_key": "unclassified_none_ablation"
                if parent is None
                else None,
                "discovery_need": "Unclassified real NONE evidence; aggregation must assess coverage",
                "initiation": "",
                "policy": "",
                "termination": "",
                "evidence_refs": transitions,
                "raw_evidence": payload["evidence"],
            }
        value = self.generate(
            "skill.semantic_gradient",
            payload,
            media=self._media(segment),
            validator=validate,
        )
        validate(value)
        return {
            **value,
            "diagnosis_id": key,
            "trajectory_id": row.trajectory_id,
            "skill_ref": segment.skill_reference,
            "transition_ids": transitions,
            "segment_start": segment.start_step,
            "segment_end": segment.end_step,
            "reward": row.reward,
            "advantage": advantage,
            "snapshot_id": row.knowledge_snapshot_id,
        }

    def _aggregate(self, target, entries, queue, parent, pool):
        diagnoses = [
            queue.state["diagnosis_cache"][key]
            for entry in entries
            for key in entry["diagnosis_ids"]
        ]
        evidence_ids = {t for entry in entries for t in entry["transition_ids"]}

        def validate(value):
            if not isinstance(value, dict) or type(value.get("no_change")) is not bool:
                raise ValueError("Aggregation requires explicit no_change")
            for field in ("initiation", "policy", "termination", "conflicts"):
                if not isinstance(value.get(field), str):
                    raise ValueError(f"Aggregation {field} must be text")  # noqa: TRY004 - JSON schema failures are repairable values
            refs = value.get("evidence_refs")
            if (
                not isinstance(refs, list)
                or not all(isinstance(r, str) for r in refs)
                or not set(refs) <= evidence_ids
            ):
                raise ValueError("Aggregation references unavailable evidence")
            if not value["no_change"] and (
                not refs
                or not any(
                    value[f].strip() for f in ("initiation", "policy", "termination")
                )
            ):
                raise ValueError(
                    "A proposed change requires supported component guidance"
                )
            return value

        payload = {
            "instructions": "Aggregate these semantically eligible diagnoses for ONE target. Analyze conflicts by conditions; do not concatenate incompatible unconditional procedures. Preserve no-change evidence. Return no_change when evidence cannot support an improvement. Discovery needs an explicit uncovered initiation condition; compare ALL current existing_skills and return no_change if the need is now covered.",
            "expected_output": {
                "no_change": "boolean",
                "initiation": "text",
                "policy": "text",
                "termination": "text",
                "conflicts": "text",
                "evidence_refs": "list of supplied transition IDs",
            },
            "target": target,
            "parent": parent.to_dict() if parent else None,
            "existing_skills": [s.to_dict() for s in pool.active()]
            if parent is None
            else [],
            "diagnoses": diagnoses,
            "batch": [
                {
                    "trajectory_id": e["trajectory_id"],
                    "reward": e["reward"],
                    "advantage": e["advantage"],
                }
                for e in entries
            ],
        }
        if self.config.get("no_gradient_aggregation", False):
            result = {
                "no_change": bool(parent) and all(d["no_change"] for d in diagnoses),
                **{
                    f: "\n".join(d.get(f) or "" for d in diagnoses)
                    for f in ("initiation", "policy", "termination")
                },
                "conflicts": "Ablation: no independent LLM aggregation; candidate must inspect raw diagnoses",
                "evidence_refs": sorted(evidence_ids),
                "diagnoses": diagnoses,
                "existing_skills": payload["existing_skills"],
                "aggregation_mode": "structured_diagnoses_ablation",
            }
        else:
            result = self.generate(
                "skill.gradient_aggregation", payload, validator=validate
            )
            validate(result)
        return {
            **result,
            "aggregation_id": "aggregate_" + _identity([target, entries]),
            "parent_skill_ref": parent.reference if parent else None,
            "source_gradient_ids": [d["diagnosis_id"] for d in diagnoses],
            "source_trajectory_ids": [e["trajectory_id"] for e in entries],
        }

    def _candidate(self, aggregate, parent, index, batch_index):
        evolution = SkillEvolutionType.REFINE if parent else SkillEvolutionType.NEW
        seed = int(self.config.get("seed", 0)) + batch_index * 100 + index
        candidate_id = "candidate_" + _identity(
            [aggregate["aggregation_id"], index, seed]
        )
        skill_id = (
            parent.skill_id
            if parent
            else "learned_" + _identity(aggregate["aggregation_id"])
        )

        def build(value):
            if not isinstance(value, dict):
                raise ValueError("Skill candidate must be an object")  # noqa: TRY004 - JSON schema failures are repairable values
            for field in ("name", "initiation", "termination"):
                _required_text(value.get(field), field)
            policy = value.get("policy")
            if (
                not isinstance(policy, list)
                or not policy
                or any(not isinstance(s, str) or not s.strip() for s in policy)
            ):
                raise ValueError("Skill policy must contain nonempty procedure steps")
            proposed = SkillItem(
                name=value["name"],
                initiation=value["initiation"],
                policy=tuple(policy),
                termination=value["termination"],
                skill_id=skill_id,
                version=parent.version + 1 if parent else 1,
                evolution_type=evolution,
                parent_skill_ref=parent.reference if parent else None,
                stats=SkillStats(last_evolved_iteration=batch_index),
                metadata={
                    "protocol_version": "spatialcraft_v2",
                    "source_aggregation_id": aggregate["aggregation_id"],
                    "tokenizer_id": self.config.get("tokenizer_id"),
                    "seed": seed,
                },
            )
            count = self.token_counter(proposed.format_for_prompt())
            if count > int(self.config.get("stored_skill_max_tokens", 1024)):
                raise ValueError(
                    "Stored Skill exceeds declared tokenizer budget; compress without removing conditions"
                )
            return proposed

        value = self.generate(
            "skill.candidate_generation",
            {
                "instructions": "Generate ONE reusable Skill from the original parent and the shared aggregation. This candidate is independent of the other candidates. Include precise initiation conditions, executable steps using real tools, and a verifiable termination rule. Do not store reasoning traces or current-task answers. Respect the stored token limit.",
                "expected_output": {
                    "name": "text",
                    "initiation": "text",
                    "policy": ["step text"],
                    "termination": "text",
                },
                "parent": parent.to_dict() if parent else None,
                "aggregation": aggregate,
                "candidate_index": index,
                "seed": seed,
                "stored_skill_max_tokens": int(
                    self.config.get("stored_skill_max_tokens", 1024)
                ),
                "available_tools": self.config.get("available_tools", []),
            },
            validator=build,
        )
        skill = build(value)
        return SkillCandidate(
            skill=skill,
            evolution_type=evolution,
            candidate_id=candidate_id,
            source_gradient_ids=tuple(aggregate["source_gradient_ids"]),
            source_trajectory_ids=tuple(aggregate["source_trajectory_ids"]),
            metadata={
                "seed": seed,
                "candidate_index": index,
                "aggregation_id": aggregate["aggregation_id"],
                "stored_token_count": self.token_counter(skill.format_for_prompt()),
            },
        )

    def _examples(self, entries, parent, recent):
        examples = []
        for entry in entries:
            row = recent.get(entry["trajectory_id"])
            if row is None:
                if self.load_trajectory is None:
                    raise ValueError(
                        "Skill evolution requires the committed trajectory loader"
                    )
                row = self.load_trajectory(entry["source_key"])
            if (
                row.trajectory_id != entry["trajectory_id"]
                or digest(row.to_dict()) != entry["trajectory_sha256"]
            ):
                raise ValueError("Queued trajectory identity/checksum mismatch")
            matching = [
                t for t in row.transitions if t.transition_id in entry["transition_ids"]
            ]
            if len(matching) != len(entry["transition_ids"]):
                raise ValueError(
                    "Queued evidence contains missing or duplicate transitions"
                )
            for transition in matching:
                active = transition.active_skill
                ref = f"{active.skill_id}@{active.version}" if active else None
                if ref != (parent.reference if parent else None):
                    raise ValueError("Historical action does not match parent/NONE")
                target = transition.metadata.get("action_target")
                if (
                    not isinstance(target, dict)
                    or not target.get("text")
                    or not target.get("token_ids")
                ):
                    raise HistoricalActionUnavailable(
                        "Skill v2 requires recorded original action text and token IDs"
                    )
                if "model_request" not in transition.metadata:
                    raise ValueError("Missing original historical model request")
                request = request_from_dict(transition.metadata["model_request"])
                prefix = target.get("prefix", "")
                if not isinstance(prefix, str):
                    raise ValueError("Action prefix must be recorded text")  # noqa: TRY004 - JSON schema failures are repairable values
                if prefix:
                    request = replace(
                        request,
                        metadata={**request.metadata, "fixed_scoring_prefix": prefix},
                    )
                request = replace(
                    request,
                    metadata={
                        **request.metadata,
                        "fixed_target_token_ids": list(target["token_ids"]),
                        "fixed_scoring_prefix_token_ids": list(
                            target.get("prefix_token_ids", [])
                        ),
                        "likelihood_distribution": "raw_model_softmax",
                    },
                )
                examples.append(
                    SequenceLikelihoodExample(
                        trajectory_id=row.trajectory_id,
                        transition_id=transition.transition_id,
                        base_request=request,
                        target_text=target["text"],
                        target_token_ids=tuple(target["token_ids"]),
                        advantage=entry["advantage"],
                    )
                )
        return tuple(examples)

    def _statistics(self, pool, rows, diagnoses, advantages):
        credits = []
        for row in rows:
            references = {
                d["skill_ref"]
                for d in diagnoses[row.trajectory_id]
                if d["skill_ref"] is not None and d["is_related"]
            }
            for reference in sorted(references):
                skill = pool.get(reference)
                metadata = dict(skill.metadata)
                seen = list(metadata.get("quality_trajectory_ids", []))
                if row.trajectory_id in seen:
                    continue
                seen.append(row.trajectory_id)
                informative = list(metadata.get("quality_nonzero_trajectory_ids", []))
                advantage = advantages[row.trajectory_id]
                if advantage != 0:
                    informative.append(row.trajectory_id)
                metadata.update(
                    quality_trajectory_ids=seen,
                    quality_nonzero_trajectory_ids=informative,
                    gain_semantics="task_group_outcome_association_not_causal",
                )
                stats = skill.stats.record_usage(
                    advantage=advantage, successful=row.reward == 1
                )
                stats = replace(stats, maturity=len(seen))
                pool = pool.replace_item(replace(skill, stats=stats, metadata=metadata))
                credits.append(
                    {
                        "trajectory_id": row.trajectory_id,
                        "skill_reference": reference,
                        "advantage": advantage,
                        "successful": row.reward == 1,
                    }
                )
        return pool, credits

    def _maintain(self, pool):
        removed, audit = [], []
        minimum = int(self.config.get("minimum_quality_trajectories", 3))
        for skill in pool.active():
            informative = len(
                set(skill.metadata.get("quality_nonzero_trajectory_ids", []))
            )
            if (
                not self.config.get("no_score_pruning", False)
                and informative >= minimum
                and skill.stats.average_gain < 0
            ):
                pool = pool.archive(skill.reference)
                removed.append(skill.reference)
                audit.append(
                    {
                        "type": "quality_archive",
                        "target": skill.reference,
                        "reason": "negative_associated_gain_with_nonzero_evidence",
                        "evidence_count": informative,
                    }
                )
        ranked = sorted(
            pool.active(),
            key=lambda s: (-s.stats.average_gain, -s.stats.frequency, s.reference),
        )
        retained = []
        for skill in ranked:
            duplicate = next(
                (
                    other
                    for other in retained
                    if (skill.initiation, skill.policy, skill.termination)
                    == (other.initiation, other.policy, other.termination)
                ),
                None,
            )
            if duplicate is not None:
                pool = pool.archive(skill.reference)
                removed.append(skill.reference)
                audit.append(
                    {
                        "type": "exact_duplicate",
                        "target": skill.reference,
                        "retained": duplicate.reference,
                    }
                )
            else:
                retained.append(skill)
        if self.config.get("deduplication_mode", "llm") == "llm" and len(retained) > 1:
            vectors = _normalize(
                self.embedder.embed([s.format_for_prompt() for s in retained])
            )
            threshold = float(self.config.get("deduplication_cosine_threshold", 0.90))
            pairs = [
                (a, b)
                for i, a in enumerate(retained)
                for j, b in enumerate(retained)
                if j > i and float(vectors[i] @ vectors[j]) >= threshold
            ]
            if pairs:
                available_pairs = {
                    frozenset((a.reference, b.reference)) for a, b in pairs
                }

                def validate(value):
                    operations = (
                        value.get("duplicates") if isinstance(value, dict) else None
                    )
                    if not isinstance(operations, list):
                        raise ValueError(  # noqa: TRY004 - JSON schema failure
                            "Skill deduplication requires a duplicates list"
                        )
                    targets, kept = set(), set()
                    for op in operations:
                        if not isinstance(op, dict):
                            raise ValueError("Invalid deduplication operation")  # noqa: TRY004 - JSON schema failures are repairable values
                        keep, remove = op.get("keep_ref"), op.get("archive_ref")
                        if (
                            frozenset((keep, remove)) not in available_pairs
                            or keep == remove
                            or remove in targets
                        ):
                            raise ValueError(
                                "Deduplication requires a unique supplied pair"
                            )
                        _required_text(op.get("reason"), "reason")
                        targets.add(remove)
                        kept.add(keep)
                    if targets & kept:
                        raise ValueError("Conflicting deduplication chain")
                    return value

                result = self.generate(
                    "skill.deduplication",
                    {
                        "instructions": "Compare only these candidate pairs. Archive a Skill only if BOTH its initiation conditions and full procedure/termination function are redundant with the retained Skill. Similar language or embedding similarity is not enough. Return no operation when functions are complementary.",
                        "expected_output": {
                            "duplicates": [
                                {
                                    "keep_ref": "reference",
                                    "archive_ref": "reference",
                                    "reason": "text",
                                }
                            ]
                        },
                        "pairs": [[a.to_dict(), b.to_dict()] for a, b in pairs],
                    },
                    validator=validate,
                )
                validate(result)
                for op in result["duplicates"]:
                    pool = pool.archive(op["archive_ref"])
                    removed.append(op["archive_ref"])
                    audit.append({"type": "llm_semantic_duplicate", **op})
        capacity = int(self.config.get("capacity", 20))
        if capacity < 1:
            raise ValueError("Skill capacity must be positive")
        # Protection against unsupported quality deletion does not grant infinite
        # capacity: overflow ranks all survivors by gain, use count, then identity.
        ranked = sorted(
            pool.active(),
            key=lambda s: (-s.stats.average_gain, -s.stats.frequency, s.reference),
        )
        for skill in ranked[capacity:]:
            pool = pool.archive(skill.reference)
            removed.append(skill.reference)
            audit.append({"type": "capacity_archive", "target": skill.reference})
        return pool, removed, audit

    @staticmethod
    def _validation_error():
        from spatialcraft.experiments.knowledge_generator import (
            KnowledgeValidationError,
        )

        return KnowledgeValidationError

    def evolve(self, knowledge, rows, batch_index, evolution_state=None):
        if (
            not rows
            or len({r.task.task_id for r in rows}) != 1
            or len({r.knowledge_snapshot_id for r in rows}) != 1
        ):
            raise ValueError("Skill learning requires one complete frozen task group")
        expected_n = int(self.config.get("rollouts_per_task", 4))
        if len(rows) != expected_n:
            raise ValueError("Skill learning must wait for every configured rollout")
        queue = RelatedEvolutionQueue(evolution_state)
        pool = SkillPool(knowledge.skills.all())
        advantages = {
            a.trajectory_id: a.advantage for a in SkillCreditAssigner().advantages(rows)
        }
        diagnoses = {}
        for row in rows:
            diagnoses[row.trajectory_id] = [
                self._diagnose(row, segment, advantages[row.trajectory_id], pool, queue)
                for segment in skill_segments(row)
            ]
        queue.enqueue(
            rows,
            task_index=batch_index,
            diagnoses=diagnoses,
            active_references={s.reference for s in pool.active()},
        )
        pool, credits = self._statistics(pool, rows, diagnoses, advantages)
        batches = queue.take_round(
            batch_size=int(self.config.get("batch_trajectories", 6)),
            preferred_low=int(self.config.get("preferred_low_reward", 3)),
            preferred_high=int(self.config.get("preferred_high_reward", 3)),
            maximum_parents=int(self.config.get("max_parent_skills_per_round", 2)),
            maximum_targets=int(self.config.get("maximum_targets_per_round", 2)),
            maximum_discovery=int(self.config.get("max_discovery_per_round", 1)),
        )
        gate = (
            None
            if self.config.get("no_ppo_gate", False)
            else SequenceLikelihoodGate(
                self.scorer,
                epsilon=float(self.config.get("ppo_clip_epsilon", 0.2)),
                acceptance_margin=float(self.config.get("acceptance_gain_margin", 0)),
            )
        )
        records, proposals, accepted, aggregates, batch_audits, failures = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        recent = {r.trajectory_id: r for r in rows}
        for target, entries in batches.items():
            parent = None if target.startswith("DISCOVERY:") else pool.get(target)
            audit = {
                "target": target,
                "parent_skill_ref": parent.reference if parent else None,
                "entries": entries,
                "source_trajectory_ids": [e["trajectory_id"] for e in entries],
                "original_task_advantages": {
                    e["trajectory_id"]: e["advantage"] for e in entries
                },
                "reward_counts": {
                    str(r): sum(e["reward"] == r for e in entries) for r in (0, 1)
                },
                "status": "pending",
            }
            batch_audits.append(audit)
            aggregate = self._aggregate(target, entries, queue, parent, pool)
            aggregates.append(aggregate)
            if aggregate["no_change"]:
                audit["status"] = "no_change"
                continue
            if not self.config.get("no_ppo_gate", False) and all(
                e["advantage"] == 0 for e in entries
            ):
                audit["status"] = "no_relative_scoring_signal"
                continue
            examples = None
            if not self.config.get("no_ppo_gate", False):
                try:
                    examples = self._examples(entries, parent, recent)
                except HistoricalActionUnavailable as exc:
                    audit["status"] = "batch_rejected_missing_span"
                    failures.append(
                        {
                            "target": target,
                            "reason": str(exc),
                            "status": "batch_rejected_missing_span",
                        }
                    )
                    continue
            candidates = []
            for index in range(int(self.config.get("candidates_per_target", 3))):
                try:
                    candidates.append(
                        self._candidate(aggregate, parent, index, batch_index)
                    )
                except self._validation_error() as exc:
                    failures.append(
                        {"target": target, "candidate_index": index, "reason": str(exc)}
                    )
            if not candidates:
                audit["status"] = "candidate_generation_failed"
                queue.state["queues"][target] = sorted(
                    entries + queue.state["queues"][target], key=lambda e: e["sequence"]
                )
                continue
            proposals.extend(c.to_dict() for c in candidates)
            if self.config.get("no_ppo_gate", False):
                from .ppo_gate import SequenceLikelihoodResult

                result = SequenceLikelihoodResult(
                    accepted_candidate_id=candidates[0].candidate_id,
                    records=tuple(
                        {
                            "candidate_id": c.candidate_id,
                            "parent_skill_ref": parent.reference if parent else "NONE",
                            "accepted": index == 0,
                            "candidate_objective": None,
                            "reason": "first_valid_candidate_ungated_ablation"
                            if index == 0
                            else "not_first_valid_candidate",
                            "trajectory_scores": [],
                            "metadata": {
                                "scoring_mode": "ungated_ablation",
                                "formal_np_ppo": False,
                            },
                        }
                        for index, c in enumerate(candidates)
                    ),
                )
            else:
                result = gate.select(
                    tuple(candidates), parent=parent, examples=examples
                )
            records.extend(result.records)
            audit["status"] = "accepted" if result.accepted_candidate_id else "rejected"
            if result.accepted_candidate_id:
                chosen = next(
                    c
                    for c in candidates
                    if c.candidate_id == result.accepted_candidate_id
                )
                pool = pool.refine(chosen.skill) if parent else pool.add(chosen.skill)
                accepted.append(chosen.candidate_id)
        pool, pruned, maintenance = self._maintain(pool)
        discarded = queue.discard_inactive({s.reference for s in pool.active()})
        return {
            "skill_pool": pool.to_dict(),
            "credits": credits,
            "semantic_gradients": [d for values in diagnoses.values() for d in values],
            "aggregates": aggregates,
            "candidates": proposals,
            "candidate_failures": failures,
            "ppo_records": records,
            "accepted_candidate_ids": accepted,
            "pruned_skill_references": pruned,
            "maintenance_operations": maintenance,
            "source_trajectory_ids": [r.trajectory_id for r in rows],
            "evolution_batches": batch_audits,
            "evolution_state": queue.state,
            "pending_counts": {k: len(v) for k, v in queue.state["queues"].items()},
            "discarded_inactive_version_entries": discarded,
            "round_index": batch_index,
            "scoring_mode": "sequence_raw_model_likelihood_surrogate",
            "protocol_version": "spatialcraft_v2",
            "ablations": {
                name: bool(self.config.get(name, False))
                for name in (
                    "no_semantic_gradient",
                    "no_gradient_aggregation",
                    "no_ppo_gate",
                    "no_score_pruning",
                )
            },
        }
