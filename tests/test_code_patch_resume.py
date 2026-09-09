import pytest

from spatialcraft.experiments.journal import RunJournal, digest
from spatialcraft.storage.atomic_io import atomic_write_json, read_json


def test_two_patch_lineage_and_no_backward_commits(tmp_path):
    old = {"code_sha256": "original", "settings": {"tokens": 4096}}
    RunJournal(tmp_path, old).execute("original", {}, lambda: {"ok": 0})
    patch1 = {
        "schema_version": 1,
        "old_binding_sha256": digest(old),
        "new_code_sha256": "detector",
        "reason": "detector bounds",
        "source_changes": {"detect.py": {"before": "a", "after": "b"}},
    }
    atomic_write_json(tmp_path / "code_patch.json", patch1)
    intermediate = RunJournal(tmp_path, {**old, "code_sha256": "detector"})
    intermediate.execute("middle", {}, lambda: {"ok": 1})
    with pytest.raises(RuntimeError):
        intermediate.execute("retry", {}, lambda: (_ for _ in ()).throw(RuntimeError()))
    before = {p: p.read_bytes() for p in (tmp_path / "stages").rglob("*.json")}
    patch2 = {
        **patch1,
        "new_code_sha256": "jsonfix",
        "reason": "JSON syntax",
        "parent_code_sha256": "detector",
        "prior_patches": [patch1],
    }
    atomic_write_json(tmp_path / "code_patch.json", patch2)
    current_binding = {**old, "code_sha256": "jsonfix"}
    current = RunJournal(tmp_path, current_binding)
    assert current.read_committed("original") == {"ok": 0}
    assert current.execute("middle", {}, lambda: pytest.fail("reexecuted"))[1]
    current.execute("retry", {}, lambda: {"ok": 2})
    assert current.read_committed("retry") == {"ok": 2}
    assert all(p.read_bytes() == value for p, value in before.items())
    for change in (
        {"parent_code_sha256": "unknown"},
        {"prior_patches": []},
        {"new_code_sha256": "original"},
    ):
        atomic_write_json(tmp_path / "code_patch.json", {**patch2, **change})
        with pytest.raises(ValueError, match="Invalid code-patch"):
            RunJournal(tmp_path, current_binding)
    atomic_write_json(tmp_path / "code_patch.json", patch2)
    path = tmp_path / "stages/middle/result.json"
    commit = read_json(path)
    atomic_write_json(path, {**commit, "binding_sha256": digest(old)})
    with pytest.raises(ValueError, match="checksum/binding"):
        current.read_committed("middle")


def test_explicit_patch_reuses_commits_without_rewriting(tmp_path):
    old = {"code_sha256": "old", "settings": {"tokens": 4096}}
    journal = RunJournal(tmp_path, old)
    journal.execute("done", {"x": 1}, lambda: {"reward": 1})
    with pytest.raises(RuntimeError):
        journal.execute(
            "failed", {"x": 2}, lambda: (_ for _ in ()).throw(RuntimeError())
        )
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    new = {**old, "code_sha256": "new"}
    with pytest.raises(ValueError, match="Resume binding changed"):
        RunJournal(tmp_path, new)
    atomic_write_json(
        tmp_path / "code_patch.json",
        {
            "schema_version": 1,
            "old_binding_sha256": digest(old),
            "new_code_sha256": "new",
            "reason": "detector output bounds",
            "source_changes": {"detect.py": {"before": "a", "after": "b"}},
        },
    )
    resumed = RunJournal(tmp_path, new)
    assert resumed.execute("done", {"x": 1}, lambda: pytest.fail("reexecuted")) == (
        {"reward": 1},
        True,
    )
    assert resumed.read_committed("done") == {"reward": 1}
    assert resumed.execute("failed", {"x": 2}, lambda: {"ok": True}) == (
        {"ok": True},
        False,
    )
    assert resumed.read_committed("failed") == {"ok": True}
    resumed.execute("new", {}, lambda: {"ok": True})
    assert RunJournal(tmp_path, new).read_committed("new") == {"ok": True}
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(ValueError, match="Stage inputs changed"):
        resumed.execute("done", {"x": 3}, dict)
    with pytest.raises(ValueError, match="Invalid code-patch"):
        RunJournal(tmp_path, {**new, "settings": {"tokens": 1024}})
    with pytest.raises(ValueError, match="Invalid code-patch"):
        RunJournal(tmp_path, old)
