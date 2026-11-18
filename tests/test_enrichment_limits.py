"""Reject invalid throttling before connecting to the database."""
import pytest
from retrieval.enrich_canonical_payloads import run


@pytest.mark.parametrize("kwargs", [dict(batch_size=0), dict(pause_seconds=-1), dict(max_batches=0)])
def test_invalid_limits_do_not_open_client(tmp_path, kwargs):
    with pytest.raises(ValueError):
        run("http://invalid", "unused", tmp_path / "state.json", **kwargs)
