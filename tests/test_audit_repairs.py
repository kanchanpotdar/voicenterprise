"""Synthetic regression cases for independently audited integrity defects."""
from ingestion.chunker import _split_long_text
from ingestion.metadata import extract_metadata_for_record
from evaluation.audit_scoring import score_answer
from evaluation.run_controlled_generation_benchmark import generation_cache_key


def test_paragraph_overlap_never_exceeds_body_limit():
    chunks = _split_long_text("a" * 90 + "\n\n" + "b" * 10 + "\n\n" + "c" * 95, 100, 20)
    assert all(len(c) <= 100 for c in chunks)
    assert any("c" * 95 in c for c in chunks)


def test_truncation_and_fabrication_are_rejected():
    assert not score_answer("q", "basic", "The limit is 10. [TRUNCATION DETECTED: output cut off]", ["The limit is 10."]).answer_correct_strict
    assert not score_answer("q", "info_not_found", "The secret is 123. The owner is not specified.", []).answer_correct_strict


def test_cache_distinguishes_token_budget():
    kwargs = dict(model="m", system_prompt="s", user_prompt="u", temperature=0)
    assert generation_cache_key(**kwargs, max_tokens=20) != generation_cache_key(**kwargs, max_tokens=40)


def test_real_dates_and_2030_are_supported():
    result = extract_metadata_for_record("r", "d", "2030-01-01", "2025-02-31")
    assert result.extracted_dates == ["2030-01-01"]


def test_global_version_ids_survive_batch_boundaries(tmp_path):
    import pandas as pd
    from data.batch_ingest_full import run
    source = tmp_path / "source.parquet"
    pd.DataFrame([
        dict(doc_id="same", source_type="jira", title="One", content="First version"),
        dict(doc_id="same", source_type="jira", title="Two", content="Second version"),
    ]).to_parquet(source)
    out = tmp_path / "out"
    run(source, out, batch_rows=1)
    frames = [pd.read_parquet(p) for p in (out / "batches").glob("*_chunks.parquet")]
    result = pd.concat(frames)
    assert result.chunk_id.nunique() == 2
    assert result.has_duplicate_doc_id_conflict.all()


def test_all_offline_review_probes_are_regressions(capsys):
    import json
    from evaluation.review_checks_2026_10_27 import main
    main()
    result = json.loads(capsys.readouterr().out)
    for key in ("cache_key_ignores_max_tokens", "fabrication_plus_unrelated_caveat_marked_correct",
                "wrong_number_marked_supported", "empty_answer_marked_verified"):
        assert result[key] is False
    assert not result["truncated_answer"]["marked_correct"]
    assert result["chunk_citation"]["cited_sentence_count"] == 1
    assert result["versioned_chunk_citation_count"] == 1
    for mode in ("cached_tokenizer", "fallback"):
        assert result[f"budget_{mode}"]["actual"] <= 50
    order = result["multiquery_rank_lost"]
    assert order["fused_order"] == order["context_order"]
    assert result["voice_timing"]["reported_ttfa_ms"] >= 100


def test_failed_tts_never_writes_fake_audio(tmp_path, monkeypatch):
    from voice.voice_pipeline import StreamingTTS
    import pytest
    def fail(*args, **kwargs): raise OSError("synthetic failure")
    monkeypatch.setattr("voice.voice_pipeline.subprocess.run", fail)
    target = tmp_path / "output.aiff"
    with pytest.raises(RuntimeError):
        StreamingTTS().synthesize_to_file("text", str(target))
    assert not target.exists()
