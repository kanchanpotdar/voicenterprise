"""Unit tests for Query Expansion and Question Decomposition (retrieval/query_expansion.py)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rag.llm import MockLLM
from retrieval.indexer import RetrievedChunk
from retrieval.query_expansion import (
    MultiQueryRetriever,
    QueryExpander,
    QueryExpansionResult,
    SubQuestion,
)


def _create_mock_chunk(doc_id: str, chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        record_id=doc_id,
        doc_id=doc_id,
        score=score,
        source_type="confluence",
        normalized_title=f"Doc {doc_id}",
        section_title="Overview",
        chunk_text=f"Content for {doc_id}",
        latest_date=None,
        date_confidence="none",
        slack_channels=[],
        ticket_ids=[],
    )


def test_query_expander_disabled():
    expander = QueryExpander(enabled=False)
    res = expander.expand("What is the p99 latency in us-west-2?")
    assert res.mode == "none"
    assert res.all_search_queries == ["What is the p99 latency in us-west-2?"]
    assert res.expanded_queries == []


def test_query_expander_alternative_queries():
    canned_json = json.dumps({
        "expanded_queries": [
            "us-west-2 p99 text generation latency jump",
            "hosted text generation endpoint mitigation us-west-2"
        ],
        "key_entities": ["us-west-2", "hosted text generation"],
        "exact_identifiers": ["us-west-2"],
        "constraints": ["p99"],
        "negations": [],
        "uncertainties": []
    })
    mock_llm = MockLLM(default_response=canned_json)
    expander = QueryExpander(llm_client=mock_llm, default_mode="alternative_queries", max_expansions=2)

    res = expander.expand("What caused the brief p99 latency jump in us-west-2?")
    assert res.mode == "alternative_queries"
    assert len(res.expanded_queries) == 2
    assert "us-west-2" in res.exact_identifiers
    assert res.all_search_queries[0] == "What caused the brief p99 latency jump in us-west-2?"
    assert len(res.all_search_queries) == 3


def test_query_expander_terminology_mode():
    canned_json = json.dumps({
        "expanded_queries": [
            "FP8 INT8 quantization safe mode inference precision",
            "low bit math quantization kernel changes"
        ],
        "key_entities": ["low bit math", "FP8", "INT8"],
        "exact_identifiers": [],
        "constraints": ["safe mode"],
        "negations": [],
        "uncertainties": []
    })
    mock_llm = MockLLM(default_response=canned_json)
    expander = QueryExpander(llm_client=mock_llm, default_mode="terminology")

    res = expander.expand("In the recent change that made low bit math safer for inference?")
    assert res.mode == "terminology"
    assert "FP8 INT8" in res.expanded_queries[0]
    assert len(res.all_search_queries) == 3


def test_query_expander_decomposition_mode():
    canned_json = json.dumps({
        "expanded_queries": [],
        "key_entities": ["Redwood Inference", "hotfix"],
        "exact_identifiers": [],
        "constraints": ["emergency"],
        "negations": [],
        "uncertainties": [],
        "subquestions": [
            {
                "subquestion": "emergency serving-runtime hotfix preconditions and required approvals",
                "requested_information": "Preconditions and approvals",
                "entities_identifiers": [],
                "expected_evidence_type": "confluence",
                "dependencies": [],
                "constraints": []
            },
            {
                "subquestion": "hosted dedicated rollout rollback steps hotfix",
                "requested_information": "Rollout and rollback procedure",
                "entities_identifiers": [],
                "expected_evidence_type": "confluence",
                "dependencies": [],
                "constraints": []
            }
        ]
    })
    mock_llm = MockLLM(default_response=canned_json)
    expander = QueryExpander(llm_client=mock_llm, default_mode="decomposition")

    res = expander.expand("What is Redwood's process for shipping an emergency hotfix?")
    assert res.mode == "decomposition"
    assert len(res.subquestions) == 2
    assert "preconditions" in res.subquestions[0].subquestion
    assert res.all_search_queries[0] == "What is Redwood's process for shipping an emergency hotfix?"
    assert len(res.all_search_queries) == 3


def test_query_expander_safe_fallback_on_malformed_json():
    # If LLM returns non-JSON text
    mock_llm = MockLLM(default_response="I don't know how to format JSON.")
    expander = QueryExpander(llm_client=mock_llm)

    res = expander.expand("Some user query")
    assert res.original_query == "Some user query"
    assert res.expanded_queries == []
    assert res.all_search_queries == ["Some user query"]
    assert res.provenance_log.get("status") == "fallback_no_json"


def test_multi_query_retriever_rrf_fusion():
    mock_retriever = MagicMock()
    # Branch 1 (original): returns doc_A (rank 1), doc_B (rank 2)
    # Branch 2 (expansion): returns doc_B (rank 1), doc_C (rank 2)
    def mock_retrieve_docs(query, mode, top_k_docs, diversity_filter):
        if "original" in query.lower():
            return (
                ["doc_A", "doc_B"],
                [_create_mock_chunk("doc_A", "doc_A__c0", 0.9), _create_mock_chunk("doc_B", "doc_B__c0", 0.8)]
            )
        else:
            return (
                ["doc_B", "doc_C"],
                [_create_mock_chunk("doc_B", "doc_B__c0", 0.85), _create_mock_chunk("doc_C", "doc_C__c0", 0.7)]
            )

    mock_retriever.retrieve_documents.side_effect = mock_retrieve_docs

    canned_json = json.dumps({
        "expanded_queries": ["expanded query 1"],
        "key_entities": [],
        "exact_identifiers": [],
        "constraints": [],
        "negations": [],
        "uncertainties": []
    })
    mock_llm = MockLLM(default_response=canned_json)
    expander = QueryExpander(llm_client=mock_llm)
    mq_retriever = MultiQueryRetriever(retriever=mock_retriever, expander=expander)

    doc_ids, chunks, exp_res, provenance = mq_retriever.retrieve(
        query="original query",
        mode="bm25",
        expansion_mode="alternative_queries",
        top_k_docs=5,
    )

    # doc_B appeared in both branches:
    # RRF(doc_A) = 1.0 / (60 + 1) = 0.01639
    # RRF(doc_B) = 1.0 / (60 + 2) + 0.8 / (60 + 1) = 0.01613 + 0.01311 = 0.02924
    # doc_B has higher combined score than doc_A, so doc_B should rank first!
    assert doc_ids[0] == "doc_B"
    assert doc_ids[1] == "doc_A"
    assert doc_ids[2] == "doc_C"
    assert len(chunks) == 3
    assert len(provenance) == 2


def test_expansion_cache_write_and_hit_without_second_llm_call(tmp_path):
    from retrieval.query_expansion import ExpansionCache, expansion_cache_key

    cache_file = tmp_path / "exp_cache.json"
    cache = ExpansionCache(path=cache_file)

    calls = []

    def fake_fn(prompt, sys_prompt):
        calls.append(prompt)
        return json.dumps({
            "expanded_queries": ["FP8 INT8 safe mode pass rate", "numeric precision step down threshold"],
            "key_entities": ["FP8"],
            "exact_identifiers": [],
            "constraints": [],
            "negations": [],
            "uncertainties": [],
        })

    llm = MockLLM(response_fn=fake_fn)
    expander = QueryExpander(llm_client=llm, cache=cache)

    q = "What is the default pass rate for stepping down numeric mode?"
    r1 = expander.expand(q, mode="alternative_queries", max_expansions=2)
    assert r1.status == "success"
    assert r1.failed is False
    assert r1.fell_back_to_original is False
    assert r1.cache_status == "write"
    assert len(calls) == 1
    assert cache.writes == 1
    assert cache_file.exists()

    # Whitespace-variant of the same query must hit the persisted cache without calling LLM
    cache_reloaded = ExpansionCache(path=cache_file)
    expander2 = QueryExpander(llm_client=llm, cache=cache_reloaded)
    r2 = expander2.expand("  What is the   default pass rate for stepping down numeric mode? \n", mode="alternative_queries", max_expansions=2)
    assert r2.status == "cache_hit"
    assert r2.cache_status == "hit"
    assert r2.failed is False
    assert r2.expanded_queries == ["FP8 INT8 safe mode pass rate", "numeric precision step down threshold"]
    assert len(calls) == 1  # No second LLM call!
    assert cache_reloaded.hits == 1

    # Different mode or limit produces a distinct cache key
    k_alt = expansion_cache_key(q, "alternative_queries", "mock-llm-v1", "sys", 2)
    k_term = expansion_cache_key(q, "terminology", "mock-llm-v1", "sys", 2)
    k_lim = expansion_cache_key(q, "alternative_queries", "mock-llm-v1", "sys", 3)
    assert len({k_alt, k_term, k_lim}) == 3


def test_expansion_failures_are_logged_reported_and_never_cached(tmp_path, caplog):
    import logging
    from rag.llm import LLMResponse
    from retrieval.query_expansion import ExpansionCache, QueryExpansionError

    cache_file = tmp_path / "exp_cache.json"
    cache = ExpansionCache(path=cache_file)

    # 1. LLM raises runtime error (e.g. HTTP 429)
    bad_llm = MagicMock()
    bad_llm.model = "mock-fail"
    bad_llm.generate.side_effect = RuntimeError("429 Too Many Requests: TPD exhausted")

    expander = QueryExpander(llm_client=bad_llm, cache=cache)
    with caplog.at_level(logging.WARNING):
        res_err = expander.expand("query 1")
    assert res_err.status == "fallback_error"
    assert res_err.failed is True
    assert res_err.fell_back_to_original is True
    assert len(cache) == 0
    assert any("Query expansion FAILED" in rec.message for rec in caplog.records)

    # 2. Truncated output (finish_reason == "length")
    trunc_llm = MagicMock()
    trunc_llm.model = "mock-trunc"
    trunc_llm.generate.return_value = LLMResponse(
        text='{"expanded_queries": ["partial',
        model="mock-trunc",
        finish_reason="length",
    )
    expander_trunc = QueryExpander(llm_client=trunc_llm, cache=cache)
    res_trunc = expander_trunc.expand("query 2")
    assert res_trunc.status == "fallback_truncated"
    assert res_trunc.failed is True
    assert res_trunc.fell_back_to_original is True
    assert len(cache) == 0

    # 3. Valid JSON with empty expansions -> success_empty, falls back to original, not cached
    empty_llm = MockLLM(default_response=json.dumps({"expanded_queries": []}))
    expander_empty = QueryExpander(llm_client=empty_llm, cache=cache)
    res_empty = expander_empty.expand("query 3")
    assert res_empty.status == "success_empty"
    assert res_empty.failed is False
    assert res_empty.fell_back_to_original is True
    assert len(cache) == 0

    # 4. strict=True raises QueryExpansionError instead of silent fallback
    strict_expander = QueryExpander(llm_client=bad_llm, cache=cache, strict=True)
    with pytest.raises(QueryExpansionError, match="429 Too Many Requests"):
        strict_expander.expand("query 4")


def test_corrupt_and_invalid_cache_entries_handled_safely(tmp_path, caplog):
    import logging
    from retrieval.query_expansion import ExpansionCache

    # Corrupt JSON file on disk
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        c1 = ExpansionCache(path=bad_file)
    assert len(c1) == 0
    assert bad_file.read_text(encoding="utf-8") == "{not valid json"  # untouched

    # Invalid schema entry inside an otherwise valid JSON cache
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text(
        json.dumps({"bad_key": {"v": 1, "expanded_queries": []}}),
        encoding="utf-8",
    )
    c2 = ExpansionCache(path=tampered_file)
    assert c2.get("bad_key") is None
    assert c2.rejected == 1
