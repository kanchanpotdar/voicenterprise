"""Reject invalid throttling before connecting to the database."""
import pytest
from retrieval.enrich_canonical_payloads import run
from retrieval.enrich_canonical_payloads import memory_percent
from types import SimpleNamespace


@pytest.mark.parametrize("kwargs", [dict(batch_size=0), dict(pause_seconds=-1), dict(max_batches=0)])
def test_invalid_limits_do_not_open_client(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run("http://invalid", "unused", tmp_path / "state.json", **kwargs)


def test_memory_guard_reads_percentage_without_shell(monkeypatch):
    def fake_run(args, **kwargs):
        assert args[-1] == "voiceenterprise-qdrant"
        assert "shell" not in kwargs
        return SimpleNamespace(stdout="74.05%\n")
    monkeypatch.setattr("retrieval.enrich_canonical_payloads.subprocess.run", fake_run)
    assert memory_percent("voiceenterprise-qdrant") == 74.05
