"""Unit tests for CoverageAwareContextBuilder (rag/coverage_context_builder.py)."""

from __future__ import annotations

import pytest

from rag.context_builder import ContextBuilder, estimate_tokens
from rag.coverage_context_builder import CoverageAwareContextBuilder
from retrieval.indexer import RetrievedChunk, fuse_bm25_and_dense_chunks


def _create_test_chunk(
    doc_id: str,
    chunk_id: str,
    text: str,
    score: float = 1.0,
    source: str = "confluence",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        record_id=doc_id,
        doc_id=doc_id,
        score=score,
        source_type=source,
        normalized_title=f"Doc {doc_id}",
        section_title="Policy",
        chunk_text=text,
        latest_date=None,
        date_confidence="none",
        slack_channels=[],
        ticket_ids=[],
    )


def test_coverage_context_builder_novelty_over_redundancy():
    builder = CoverageAwareContextBuilder(max_context_tokens=1000, max_chunks_per_doc=3)

    # Chunk 1: introduces core policy
    ch1 = _create_test_chunk(
        "doc_alpha", "doc_alpha__c0",
        "The production serving runtime deploys to us-east-1 and us-west-2 clusters.",
        score=10.0
    )
    # Chunk 2: near duplicate of chunk 1 from same doc
    ch2 = _create_test_chunk(
        "doc_alpha", "doc_alpha__c1",
        "The production serving runtime deploys to us-east-1 and us-west-2 cluster nodes.",
        score=9.9
    )
    # Chunk 3: introduces completely novel facts from another doc
    ch3 = _create_test_chunk(
        "doc_beta", "doc_beta__c0",
        "Rollback procedures require explicit approval from on-call release engineers and QA.",
        score=9.2
    )

    ctx, audits = builder.build_context_with_audit([ch1, ch2, ch3])

    # Both distinct docs should be included
    assert "doc_alpha" in ctx.unique_doc_ids
    assert "doc_beta" in ctx.unique_doc_ids
    # ch3 should have high novelty ratio
    beta_audit = next(a for a in audits if a.chunk_id == "doc_beta__c0")
    assert beta_audit.selected is True
    assert beta_audit.novelty_ratio > 0.5


def test_coverage_context_builder_authority_boost():
    builder = CoverageAwareContextBuilder(max_context_tokens=1000)

    # Older informal draft with slightly higher search score
    ch_draft = _create_test_chunk(
        "doc_gmail", "doc_gmail__c0",
        "We might send invoices on Monday with raw PO numbers.",
        score=10.0,
        source="gmail"
    )
    # Final governing specification with supersession notice
    ch_final = _create_test_chunk(
        "doc_conf", "doc_conf__c0",
        "This specification supersedes all earlier drafts. The approved standard requires lowercase SHA-256.",
        score=9.8,
        source="confluence"
    )

    ctx, audits = builder.build_context_with_audit([ch_draft, ch_final])

    conf_audit = next(a for a in audits if a.chunk_id == "doc_conf__c0")
    assert conf_audit.has_authority_signal is True
    assert "doc_conf" in ctx.unique_doc_ids


def test_coverage_context_builder_token_budget_enforced():
    # Long text exceeding 100 tokens
    long_text = "This is a detailed operational instruction containing numerous parameters. " * 30
    builder = CoverageAwareContextBuilder(max_context_tokens=150)

    ch1 = _create_test_chunk("doc_1", "doc_1__c0", "Short overview instruction.", score=10.0)
    ch2 = _create_test_chunk("doc_2", "doc_2__c0", long_text, score=9.0)

    ctx, audits = builder.build_context_with_audit([ch1, ch2])

    assert ctx.total_tokens <= 150
    assert ctx.total_tokens == estimate_tokens(ctx.formatted_context)
    # ch2 should be excluded or audited with token_budget_exhausted
    doc2_audit = next(a for a in audits if a.chunk_id == "doc_2__c0")
    assert doc2_audit.selected is False
    assert "budget" in doc2_audit.reason


def test_corroborated_hybrid_fusion_preserves_multi_doc_and_corroborated_semantic():
    """Verify fuse_bm25_and_dense_chunks prevents uncorroborated dense distractors from displacing top-6 BM25 docs."""
    # Simulate qst_0447: 6 gold docs at BM25 ranks 1, 2, 3, 5, 6, 7 (plus a sibling chunk at rank 4)
    bm25_chunks = [
        _create_test_chunk("gold_1", "gold_1__c0", "SOP Phase 1 preconditions", score=43.4),
        _create_test_chunk("gold_2", "gold_2__c0", "SOP Phase 2 approvals", score=42.7),
        _create_test_chunk("gold_3", "gold_3__c0", "SOP Phase 3 hosted rollout", score=39.4),
        _create_test_chunk("gold_1", "gold_1__c1", "SOP Phase 1 sibling detail", score=35.4),
        _create_test_chunk("gold_4", "gold_4__c0", "SOP Phase 4 dedicated rollout", score=35.0),
        _create_test_chunk("gold_5", "gold_5__c0", "SOP Phase 5 verification", score=33.8),
        _create_test_chunk("gold_6", "gold_6__c0", "SOP Phase 6 postmortem SLA", score=32.5),
    ] + [
        _create_test_chunk(f"tail_{i}", f"tail_{i}__c0", f"Tail doc {i}", score=25.0 - i * 0.2)
        for i in range(8, 45)
    ]

    # Dense stream has uncorroborated distractors at ranks 1..6 (only tail_38 at bm25 rank 38)
    # plus corroborated gold_4 at dense rank 8
    dense_chunks = [
        _create_test_chunk("tail_38", "tail_38__c0", "Uncorroborated dense distractor 1", score=0.88),
        _create_test_chunk("pure_dense_2", "pure_dense_2__c0", "Uncorroborated dense distractor 2", score=0.86),
        _create_test_chunk("pure_dense_3", "pure_dense_3__c0", "Uncorroborated dense distractor 3", score=0.85),
        _create_test_chunk("pure_dense_4", "pure_dense_4__c0", "Uncorroborated dense distractor 4", score=0.84),
        _create_test_chunk("pure_dense_5", "pure_dense_5__c0", "Uncorroborated dense distractor 5", score=0.83),
        _create_test_chunk("pure_dense_6", "pure_dense_6__c0", "Uncorroborated dense distractor 6", score=0.82),
        _create_test_chunk("semantic_only_7", "semantic_only_7__c0", "Pure semantic hit at rank 7", score=0.81),
        _create_test_chunk("gold_4", "gold_4__c0", "SOP Phase 4 dedicated rollout", score=0.80),
    ]

    fused = fuse_bm25_and_dense_chunks(bm25_chunks, dense_chunks, top_k_chunks=40)
    top_docs = list(dict.fromkeys(c.doc_id for c in fused))

    # All 6 gold documents must be the top 6 unique documents (ahead of uncorroborated dense distractors!)
    assert set(top_docs[:6]) == {"gold_1", "gold_2", "gold_3", "gold_4", "gold_5", "gold_6"}
    # Corroborated gold_4 (BM25 rank 5 + Dense rank 8) is boosted into the top 4
    assert top_docs.index("gold_4") < 4
    # Pure semantic hit at dense rank 7 still remains within top-20 unique documents
    assert "semantic_only_7" in top_docs[:20]
    # Lexical scores are attached for downstream context builders
    assert all(c.lexical_score is not None for c in fused)


def test_context_builder_rrf_multi_document_coverage_qst_0447_pattern():
    """Verify ContextBuilder packs all 6 primary documents under RRF scores instead of starving docs 3..6."""
    builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3)

    # Simulate 6 multi-doc SOP documents with ~420-token chunks and RRF scores + lexical_score >= 0.55
    chunks = []
    for i in range(1, 7):
        doc_id = f"dsid_sop_doc_{i}"
        rrf_score = 1.0 / (1.0 + i)
        para_a = f"Phase {i} primary requirement and mandatory approval workflow. " * 18
        para_b = f"Phase {i} secondary verification and telemetry step details. " * 18
        ch0 = _create_test_chunk(doc_id, f"{doc_id}__c000", f"{para_a}\n\n{para_b}", score=rrf_score)
        ch0.lexical_score = 0.80
        chunks.append(ch0)
        # Add high-scoring sibling chunks on docs 1 and 2 that previously crowded out docs 3..6
        if i <= 2:
            sib = _create_test_chunk(doc_id, f"{doc_id}__c001", f"Sibling appendix for doc {i}. " * 30, score=rrf_score * 0.85)
            sib.lexical_score = 0.80
            chunks.append(sib)

    chunks.sort(key=lambda c: c.score, reverse=True)
    ctx = builder.build_context(chunks)

    assert set(ctx.unique_doc_ids) == {f"dsid_sop_doc_{i}" for i in range(1, 7)}
    assert ctx.total_tokens <= 2500
    assert ctx.total_tokens == estimate_tokens(ctx.formatted_context)


def test_context_builder_rrf_two_doc_conflict_concentration_qst_0427_pattern():
    """Verify ContextBuilder preserves all 3 sibling spec chunks on 2-doc conflict queries under RRF."""
    builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3)
    conf_id = "dsid_d93aed75845e4a0b8126a80d8a7a5a41"
    gmail_id = "dsid_86f033ed92b84f3a8b381d09b0b95357"

    # Simulate raw RRF qst_0427 where top 7 chunks belong to conf_id (3 chunks) and gmail_id (4 chunks)
    chunks = [
        _create_test_chunk(conf_id, f"{conf_id}__c000", "Confluence header: supersedes June 2027 email.", score=1.000),
        _create_test_chunk(conf_id, f"{conf_id}__c002", "Confluence SLA: Tuesday by 07:00 PT.", score=0.667),
        _create_test_chunk(conf_id, f"{conf_id}__c001", "Confluence spec: po_fingerprint lowercase SHA-256 hex of PO_NUMBER|PO_LINE_UUID.", score=0.417),
        _create_test_chunk(gmail_id, f"{gmail_id}__c003", "Earlier Gmail draft: Monday delivery unhashed PO#|LINEUUID.", score=0.400),
        _create_test_chunk(gmail_id, f"{gmail_id}__c001", "Gmail thread context part 1.", score=0.375),
        _create_test_chunk(gmail_id, f"{gmail_id}__c002", "Gmail thread context part 2.", score=0.286),
        _create_test_chunk(gmail_id, f"{gmail_id}__c000", "Gmail thread context part 0.", score=0.229),
        _create_test_chunk("dsid_noise_1", "dsid_noise_1__c000", "Unrelated billing doc. " * 25, score=0.170),
        _create_test_chunk("dsid_noise_2", "dsid_noise_2__c000", "Unrelated vendor doc. " * 25, score=0.150),
    ]

    ctx = builder.build_context(chunks)
    chunk_ids = [p.chunk_id for p in ctx.passages]
    assert f"{conf_id}__c000" in chunk_ids
    assert f"{conf_id}__c001" in chunk_ids
    assert f"{conf_id}__c002" in chunk_ids
    assert f"{gmail_id}__c003" in chunk_ids
    assert ctx.total_tokens <= 2500
    assert ctx.total_tokens == estimate_tokens(ctx.formatted_context)


def test_coverage_context_builder_rejects_low_rank_generic_policy_distractor():
    """Verify generic words like 'policy'/'standard'/'final' on low-rank distractors do not leapfrog gold docs."""
    builder = CoverageAwareContextBuilder(max_context_tokens=450, max_chunks_per_doc=2)

    ch_gold1 = _create_test_chunk(
        "doc_gold_1", "doc_gold_1__c0",
        "Primary architecture details for the regional failover router.",
        score=0.80,
    )
    ch_gold2 = _create_test_chunk(
        "doc_gold_2", "doc_gold_2__c0",
        "Secondary replication quorum settings across availability zones.",
        score=0.45,
    )
    # Low-score distractor packed with generic authority words that used to get +0.30 boost
    ch_distractor = _create_test_chunk(
        "doc_distractor", "doc_distractor__c0",
        "Official standard policy specification approved final rfc changelog for cafeteria menus.",
        score=0.05,
    )

    ctx, audits = builder.build_context_with_audit([ch_gold1, ch_gold2, ch_distractor])
    assert ctx.unique_doc_ids[:2] == ["doc_gold_1", "doc_gold_2"]
    dist_audit = next(a for a in audits if a.chunk_id == "doc_distractor__c0")
    assert dist_audit.has_authority_signal is False
    assert ctx.total_tokens == estimate_tokens(ctx.formatted_context)
    assert ctx.total_tokens <= 450
