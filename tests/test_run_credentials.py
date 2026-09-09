import os

import pytest

from spatialcraft.experiments.run import load_api_key_file


def test_private_explicit_credential(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "key"
    path.write_text("synthetic-test-key\n")
    path.chmod(0o600)
    load_api_key_file(path)
    assert os.environ["OPENAI_API_KEY"] == "synthetic-test-key"


def test_reject_exposed_empty_symlink(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = tmp_path / "key"
    path.write_text("synthetic-test-key")
    path.chmod(0o644)
    with pytest.raises(ValueError):
        load_api_key_file(path)
    path.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_api_key_file(link)
    path.write_text("")
    with pytest.raises(ValueError):
        load_api_key_file(path)
    assert "OPENAI_API_KEY" not in os.environ
