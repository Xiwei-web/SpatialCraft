import json
from dataclasses import asdict

import pytest

from spatialcraft.experiments.fast_qwen import fast_local_config
from spatialcraft.experiments.journal import RunJournal, _execution_revision, digest
from spatialcraft.models.registry import LocalModelConfig
from spatialcraft.storage.atomic_io import atomic_write_json


def test_new_kernel_does_not_reuse_old_ppo_probabilities(tmp_path, monkeypatch):
    import spatialcraft.experiments.fast_qwen as fast
    from spatialcraft.models import SequenceScore

    monkeypatch.setattr(
        fast, "request_to_dict", lambda request, identity=False: {"prompt": request}
    )
    journal = RunJournal(tmp_path, {"kernel_runtime": {"profile": "new"}})
    old_inputs = {"request": {"prompt": "fixed"}, "target": "yes"}
    old = SequenceScore(
        model="q",
        target_text="yes",
        token_ids=(1,),
        token_logprobs=(-99.0,),
        prompt_token_count=1,
    )
    journal.execute("ppo_calls/" + digest(old_inputs), old_inputs, lambda: asdict(old))

    class Scorer:
        calls = 0

        def score(self, request, target):
            self.calls += 1
            return SequenceScore(
                model="q",
                target_text=target,
                token_ids=(1,),
                token_logprobs=(-1.0,),
                prompt_token_count=1,
            )

    scorer = Scorer()
    provider = fast.FastAuditedProvider(scorer, journal)
    assert provider.score("fixed", "yes").token_logprobs == (-1.0,)
    assert provider.score("fixed", "yes").token_logprobs == (-1.0,)
    assert scorer.calls == 1


def configurations():
    local = LocalModelConfig(
        path="/model",
        device_map="cuda:0",
        dtype="bfloat16",
        extra_load_kwargs={"local_files_only": True, "attn_implementation": "sdpa"},
    )
    fast = fast_local_config(local)
    return json.loads(json.dumps(asdict(local))), json.loads(json.dumps(asdict(fast)))


def test_fast_config_keeps_precision_and_model():
    old, new = configurations()
    assert old["device_map"] == "cuda:0"
    assert new["device_map"] == "balanced"
    assert new["dtype"] == old["dtype"] == "bfloat16"
    assert new["path"] == old["path"]
    assert new["load_in_4bit"] is False


def test_backend_revision_reuses_commits_and_rejects_semantic_changes(tmp_path):
    old_local, new_local = configurations()
    old = {"code_sha256": "old", "model_local": old_local, "settings": {"tokens": 4096}}
    journal = RunJournal(tmp_path, old)
    journal.execute("done", {"x": 1}, lambda: {"reward": 1})
    before = (tmp_path / "stages/done/result.json").read_bytes()
    overrides = {
        "model_local": new_local,
        "kernel_runtime": {
            "profile": "qwen35_9b_fla_two_gpu_v1",
            "expected_gpus": 2,
            "installation_sha256": "abc",
        },
    }
    current = {**old, **overrides, "code_sha256": "new"}
    atomic_write_json(
        tmp_path / "code_patch.json",
        {
            "schema_version": 1,
            "old_binding_sha256": digest(old),
            "new_code_sha256": "new",
            "reason": "User approved FLA and two GPUs",
            "source_changes": {"fast_qwen.py": "added"},
            "execution_overrides": overrides,
        },
    )
    resumed = RunJournal(tmp_path, current)
    assert resumed.read_committed("done") == {"reward": 1}
    assert resumed.execute("done", {"x": 1}, lambda: pytest.fail("replayed"))[1]
    assert (tmp_path / "stages/done/result.json").read_bytes() == before
    for changed in (
        {**overrides, "settings": {}},
        {**overrides, "model_local": {**new_local, "dtype": "float16"}},
        {**overrides, "model_local": {**new_local, "path": "/other"}},
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            _execution_revision(old, changed)
    with pytest.raises(ValueError, match="Invalid code-patch"):
        RunJournal(tmp_path, {**current, "settings": {"tokens": 1024}})
