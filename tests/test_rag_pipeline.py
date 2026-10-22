"""Unit and integration tests for Phase 2 Baseline Text RAG System.

Covers:
- Retrieval-to-generation integration
- Evidence deduplication and token-budget enforcement
- Citation validation (valid, invalid, hallucinated, chunk-level, passage-level)
- Missing-evidence abstention
- Conflicting evidence detection
- Empty retrieval handling
- Malformed model responses and provider failures
- Environment and configuration validation
- Evaluation metric correctness
- End-to-end operation using deterministic MockLLM (zero API costs)
"""

from __future__ import annotations

import pytest

from evaluation.eval_rag import DeterministicJudge, overall_answer_score
from rag.citation_validator import CitationValidator
from rag.context_builder import AssembledContext, ContextBuilder, EvidencePassage
from rag.generator import AnswerGenerator, detect_abstention, detect_contradiction
from rag.llm import MockLLM, OpenAICompatibleLLM
from rag.pipeline import RAGPipeline
from retrieval.indexer import RetrievedChunk


def _create_dummy_chunk(
    doc_id: str,
    chunk_id: str,
    text: str = "Standard operating procedure for inference.",
    score: float = 0.9,
    source: str = "confluence",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        record_id=doc_id,
        doc_id=doc_id,
        score=score,
        source_type=source,
        normalized_title=f"Title for {doc_id}",
        section_title="Overview",
        chunk_text=text,
        latest_date=None,
        date_confidence="none",
        slack_channels=[],
        ticket_ids=[],
    )


def _create_dummy_passage(idx: int, doc_id: str) -> EvidencePassage:
    return EvidencePassage(
        passage_index=idx,
        doc_id=doc_id,
        chunk_id=f"{doc_id}__c000",
        source_type="github",
        normalized_title=f"Doc {doc_id}",
        section_title="Overview",
        chunk_text="Fact details.",
        score=1.0,
    )


# -----------------------------------------------------------------------------
# 1. Evidence Deduplication & Token Budget Tests
# -----------------------------------------------------------------------------


def test_context_builder_deduplication_and_budget():
    builder = ContextBuilder(max_context_tokens=300, max_chunks_per_doc=2)

    chunks = [
        _create_dummy_chunk("dsid_1", "dsid_1__c000", "Chunk 0 of doc 1. " * 10),
        _create_dummy_chunk("dsid_1", "dsid_1__c001", "Chunk 1 of doc 1. " * 10),
        _create_dummy_chunk("dsid_1", "dsid_1__c002", "Chunk 2 of doc 1. " * 10),  # Should be skipped (max 2)
        _create_dummy_chunk("dsid_2", "dsid_2__c000", "Chunk 0 of doc 2. " * 10),
        _create_dummy_chunk("dsid_3", "dsid_3__c000", "Chunk 0 of doc 3. " * 10),
    ]

    ctx = builder.build_context(chunks, max_tokens_override=200)

    # Token budget must be strictly respected
    assert ctx.total_tokens <= 200
    assert len(ctx.passages) >= 1
    # Document 1 should not have more than 2 chunks
    doc1_passages = [p for p in ctx.passages if p.doc_id == "dsid_1"]
    assert len(doc1_passages) <= 2


def test_context_builder_multi_doc_diversity():
    builder = ContextBuilder(max_context_tokens=1000, max_chunks_per_doc=2)
    # Ensure distinct documents are balanced while sibling chunks remain contiguous
    chunks = [
        _create_dummy_chunk("dsid_a", "dsid_a__c0", "Doc A content"),
        _create_dummy_chunk("dsid_a", "dsid_a__c1", "Doc A content second"),
        _create_dummy_chunk("dsid_b", "dsid_b__c0", "Doc B content"),
        _create_dummy_chunk("dsid_c", "dsid_c__c0", "Doc C content"),
    ]
    ctx = builder.build_context(chunks)
    # All distinct documents should be represented
    passage_docs = [p.doc_id for p in ctx.passages]
    assert set(passage_docs) == {"dsid_a", "dsid_b", "dsid_c"}
    assert len(ctx.unique_doc_ids) == 3
    # Sibling chunks of dsid_a should appear contiguously in order
    a_passages = [p for p in ctx.passages if p.doc_id == "dsid_a"]
    assert len(a_passages) == 2
    assert a_passages[0].chunk_id == "dsid_a__c0"
    assert a_passages[1].chunk_id == "dsid_a__c1"
    # Chunk concentration metrics should be populated
    assert ctx.max_chunk_concentration == 0.5  # 2 of 4 passages from dsid_a
    assert ctx.distinct_docs_count == 3


# -----------------------------------------------------------------------------
# 2. Citation Validation Tests
# -----------------------------------------------------------------------------


def test_citation_validation_valid():
    validator = CitationValidator()
    ctx = AssembledContext(
        formatted_context="",
        unique_doc_ids=["dsid_11111111111111111111111111111111", "dsid_22222222222222222222222222222222"],
        passages=[
            EvidencePassage(
                passage_index=1,
                doc_id="dsid_11111111111111111111111111111111",
                chunk_id="dsid_11111111111111111111111111111111__c000",
                source_type="slack",
                normalized_title="Title",
                section_title="Overview",
                chunk_text="Text",
                score=1.0,
            )
        ],
    )

    answer = "The quota limit is 50 MiB [dsid_11111111111111111111111111111111]."
    result = validator.validate(answer, ctx)

    assert result.has_hallucinated_citations is False
    assert result.citation_precision == 1.0
    assert "dsid_11111111111111111111111111111111" in result.valid_doc_ids
    assert len(result.invalid_doc_ids) == 0


def test_citation_validation_hallucinated():
    validator = CitationValidator()
    ctx = AssembledContext(
        formatted_context="",
        unique_doc_ids=["dsid_11111111111111111111111111111111"],
    )

    # Model hallucinates a doc_id that was never provided
    answer = "The secret is rotated [dsid_99999999999999999999999999999999]."
    result = validator.validate(answer, ctx)

    assert result.has_hallucinated_citations is True
    assert result.citation_precision == 0.0
    assert "dsid_99999999999999999999999999999999" in result.invalid_doc_ids


def test_citation_validation_passage_references():
    validator = CitationValidator()
    doc_id = "dsid_11111111111111111111111111111111"
    ctx = AssembledContext(
        formatted_context="",
        unique_doc_ids=[doc_id],
        passages=[
            EvidencePassage(
                passage_index=1,
                doc_id=doc_id,
                chunk_id=f"{doc_id}__c000",
                source_type="confluence",
                normalized_title="Doc 1",
                section_title="Overview",
                chunk_text="Content",
                score=1.0,
            )
        ],
    )

    # Model cites [Passage 1]
    answer = "The rollback procedure requires SRE approval [Passage 1]."
    result = validator.validate(answer, ctx)

    assert result.has_hallucinated_citations is False
    assert doc_id in result.valid_doc_ids
    assert result.citation_precision == 1.0


# -----------------------------------------------------------------------------
# 3. Missing Evidence & Contradiction Detection Tests
# -----------------------------------------------------------------------------


def test_detect_abstention():
    assert detect_abstention("I cannot answer this question based on the provided evidence.") is True
    assert detect_abstention("The information is not available in the documents.") is True
    assert detect_abstention("Insufficient documentation was found regarding this account.") is True
    assert detect_abstention("The default size limit is 50 MiB [dsid_123].") is False


def test_detect_contradiction():
    assert detect_contradiction("The documents present conflicting instructions regarding the port.") is True
    assert detect_contradiction("The ADR specifies 30%, whereas an earlier Jira ticket recommended 20%.") is True
    assert detect_contradiction("The service is running on port 8080.") is False


# -----------------------------------------------------------------------------
# 4. Empty Retrieval & Model Error Handling Tests
# -----------------------------------------------------------------------------


def test_generator_empty_retrieval_handling():
    mock_llm = MockLLM(default_response="Should not be called")
    generator = AnswerGenerator(llm_client=mock_llm)

    empty_ctx = AssembledContext(formatted_context="", passages=[])
    ans = generator.generate_answer("What is the secret?", empty_ctx)

    assert ans.is_abstention is True
    assert "cannot answer" in ans.answer_text.lower()
    assert ans.citation_validation.has_hallucinated_citations is False


def test_openai_compatible_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key provided"):
        OpenAICompatibleLLM(api_key=None)


# -----------------------------------------------------------------------------
# 5. Evaluation Metric Correctness Tests
# -----------------------------------------------------------------------------


def test_overall_answer_score_calculation():
    assert overall_answer_score(True, 100.0) == 100.0
    assert overall_answer_score(True, 50.0) == 50.0
    assert overall_answer_score(False, 100.0) == 0.0
    assert overall_answer_score(False, 0.0) == 0.0
    with pytest.raises(ValueError):
        overall_answer_score(True, 105.0)


def test_deterministic_judge_abstention_on_info_not_found():
    judge = DeterministicJudge()
    correct, completeness, expl = judge.evaluate(
        question="Where is the confidential audit log?",
        question_type="info_not_found",
        generated_answer="I cannot answer this query based on the provided documents.",
        gold_answer="Must state that the query cannot be answered from available documents.",
        answer_facts=["Must state that the query cannot be answered from available documents."],
        is_abstention=True,
        has_contradiction=False,
    )
    assert correct is True
    assert completeness == 100.0


def test_split_facts_and_constraints_separation():
    from evaluation.eval_rag import split_facts_and_constraints

    raw = [
        "The primary cache is Redis.",
        "The answer must not attribute the failure to autoscaler cooldown.",
        "The failover duration is 15 seconds.",
        "Do not claim the emergency kill switch is kv.compaction.enabled=false.",
    ]
    pos, neg = split_facts_and_constraints(raw)
    assert len(pos) == 2
    assert "The primary cache is Redis." in pos
    assert "The failover duration is 15 seconds." in pos
    assert len(neg) == 2
    assert "The answer must not attribute the failure to autoscaler cooldown." in neg
    assert "Do not claim the emergency kill switch is kv.compaction.enabled=false." in neg


def test_deterministic_judge_constrained_negative_constraints_compliant():
    from evaluation.eval_rag import DeterministicJudge

    judge = DeterministicJudge()
    answer = "The primary cache is Redis and the failover duration is 15 seconds."
    facts = [
        "The primary cache is Redis.",
        "The failover duration is 15 seconds.",
        "The answer must not attribute the failure to autoscaler cooldown or quota limits.",
        "Do not claim that the emergency kill switch is disabled.",
    ]
    correct, completeness, expl = judge.evaluate(
        question="How does failover work?",
        question_type="constrained",
        generated_answer=answer,
        gold_answer="Redis primary cache with 15 second failover.",
        answer_facts=facts,
        is_abstention=False,
        has_contradiction=False,
    )
    assert correct is True
    assert completeness == 100.0  # 2/2 positive facts covered


def test_deterministic_judge_constrained_negative_constraint_violation():
    from evaluation.eval_rag import DeterministicJudge

    judge = DeterministicJudge()
    # Model incorrectly attributes failure to autoscaler cooldown (prohibited)
    answer = "The primary cache is Redis with 15s failover, caused by an autoscaler cooldown failure."
    facts = [
        "The primary cache is Redis.",
        "The failover duration is 15 seconds.",
        "The answer must not attribute the failure to autoscaler cooldown or quota limits.",
    ]
    correct, completeness, expl = judge.evaluate(
        question="How does failover work?",
        question_type="constrained",
        generated_answer=answer,
        gold_answer="Redis primary cache with 15 second failover.",
        answer_facts=facts,
        is_abstention=False,
        has_contradiction=False,
    )
    assert correct is False
    assert "Failed negative constraint" in expl


def test_citation_validator_zero_citations_non_abstention_precision():
    validator = CitationValidator()
    ctx = AssembledContext(
        formatted_context="[Evidence Passage 1 | Doc ID: dsid_11111111111111111111111111111111]\nFact.",
        passages=[_create_dummy_passage(1, "dsid_11111111111111111111111111111111")],
        unique_doc_ids=["dsid_11111111111111111111111111111111"],
    )
    # Substantive statement without any citation
    ans = "The server timeout limit is exactly 300 seconds."
    res = validator.validate(ans, ctx, is_abstention=False)

    assert res.total_citations_found == 0
    assert res.citation_precision == 0.0
    assert res.has_unsupported_claims is True


def test_citation_validator_zero_citations_abstention_precision():
    validator = CitationValidator()
    ctx = AssembledContext(
        formatted_context="[Evidence Passage 1 | Doc ID: dsid_11111111111111111111111111111111]\nFact.",
        passages=[_create_dummy_passage(1, "dsid_11111111111111111111111111111111")],
        unique_doc_ids=["dsid_11111111111111111111111111111111"],
    )
    ans = "I cannot answer this question because the information is not provided in the documents."
    res = validator.validate(ans, ctx, is_abstention=True)

    assert res.total_citations_found == 0
    assert res.citation_precision == 1.0
    assert res.has_unsupported_claims is False


def test_deterministic_judge_failure_to_abstain_on_info_not_found():
    judge = DeterministicJudge()
    correct, completeness, expl = judge.evaluate(
        question="Where is the confidential audit log?",
        question_type="info_not_found",
        generated_answer="The confidential audit log is stored in /var/log/audit.log.",
        gold_answer="Must state that the query cannot be answered from available documents.",
        answer_facts=["Must state that the query cannot be answered from available documents."],
        is_abstention=False,
        has_contradiction=False,
    )
    assert correct is False
    assert completeness == 0.0


# -----------------------------------------------------------------------------
# 6. End-to-End Mocked Pipeline Test
# -----------------------------------------------------------------------------


class DummyRetriever:
    """Mock retriever satisfying EnterpriseRetriever interface."""

    def __init__(self, doc_ids: list[str]) -> None:
        self.doc_ids = doc_ids

    def retrieve_documents(self, query: str, mode: str = "bm25", top_k_docs: int = 20, **kwargs):
        chunks = [
            _create_dummy_chunk(did, f"{did}__c000", f"Verified fact for {did}.", score=1.0)
            for did in self.doc_ids
        ]
        return self.doc_ids[:top_k_docs], chunks


def test_end_to_end_mocked_rag_pipeline():
    gold_id = "dsid_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    retriever = DummyRetriever([gold_id])
    mock_llm = MockLLM(
        default_response=f"According to the official runbook [{gold_id}], the system is nominal."
    )
    generator = AnswerGenerator(llm_client=mock_llm)
    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
        context_builder=ContextBuilder(),
    )

    ans = pipeline.query("What is the status?", mode="bm25")

    assert ans.question == "What is the status?"
    assert gold_id in ans.cited_doc_ids
    assert ans.citation_validation.citation_precision == 1.0
    assert ans.citation_validation.has_hallucinated_citations is False
    assert ans.is_abstention is False
    assert ans.context_passages_count == 1


# -----------------------------------------------------------------------------
# 7. Milestone 1.1 Grounding, Precedence & Sibling Assembly Regression Tests
# -----------------------------------------------------------------------------


def test_claim_level_support_lexical_entailment():
    """Verify CitationValidator distinguishes valid doc ID citing with actual claim support vs mismatch."""
    validator = CitationValidator()
    doc_id = "dsid_11111111111111111111111111111111"
    ctx = AssembledContext(
        formatted_context="",
        unique_doc_ids=[doc_id],
        passages=[
            EvidencePassage(
                passage_index=1,
                doc_id=doc_id,
                chunk_id=f"{doc_id}__c000",
                source_type="confluence",
                normalized_title="ClearEdge Invoicing",
                section_title="Security",
                chunk_text="The po_fingerprint must be a SHA-256 hexadecimal digest of the PO_NUMBER and PO_LINE_UUID.",
                score=1.0,
            )
        ],
    )

    # 1. Supported sentence: cites doc_id and uses facts from chunk_text
    supported_ans = "The po_fingerprint is a SHA-256 hexadecimal digest of the PO_NUMBER [dsid_11111111111111111111111111111111]."
    res_sup = validator.validate(supported_ans, ctx)
    assert res_sup.citation_precision == 1.0
    assert res_sup.supported_sentences_count >= 1
    assert res_sup.sentences[0].is_supported_by_context is True
    assert res_sup.sentences[0].lexical_overlap_pct >= 0.35

    # 2. Unsupported sentence: cites doc_id but asserts completely unrelated/fabricated claims
    unsupported_ans = "The quantum encryption orbital satellite telemetry was activated yesterday [dsid_11111111111111111111111111111111]."
    res_unsup = validator.validate(unsupported_ans, ctx)
    assert res_unsup.citation_id_validity == 1.0  # ID is valid
    assert res_unsup.sentences[0].is_supported_by_context is False  # claim is NOT supported
    assert res_unsup.unsupported_sentences_count >= 1


def test_unsourced_substantive_claims_flagged():
    """Verify CitationValidator flags sentences asserting substantive facts without any citations."""
    validator = CitationValidator()
    doc_id = "dsid_11111111111111111111111111111111"
    ctx = AssembledContext(
        formatted_context="",
        unique_doc_ids=[doc_id],
        passages=[_create_dummy_passage(1, doc_id)],
    )

    # Sentence with zero citations asserting facts
    ans = "All invoices are delivered on Tuesdays at 07:00 PT without fail."
    res = validator.validate(ans, ctx, is_abstention=False)

    assert res.has_unsourced_claims is True
    assert res.unsourced_sentences_count == 1
    assert res.sentence_citation_rate == 0.0


def test_chunk_concentration_audit():
    """Verify AssembledContext measures chunk concentration and distribution across parent documents."""
    builder = ContextBuilder(max_context_tokens=3000, max_chunks_per_doc=3)
    chunks = [
        _create_dummy_chunk("doc_a", "doc_a__c0", "Doc A part 1", score=10.0),
        _create_dummy_chunk("doc_a", "doc_a__c1", "Doc A part 2", score=9.5),
        _create_dummy_chunk("doc_a", "doc_a__c2", "Doc A part 3", score=9.0),
        _create_dummy_chunk("doc_b", "doc_b__c0", "Doc B part 1", score=8.0),
    ]
    ctx = builder.build_context(chunks)

    assert ctx.distinct_docs_count == 2
    assert ctx.doc_chunk_distribution["doc_a"] == 3
    assert ctx.doc_chunk_distribution["doc_b"] == 1
    assert ctx.max_chunk_concentration == 0.75  # 3 of 4 passages from doc_a
    # Check to_dict() serialization
    d = ctx.to_dict()
    assert d["max_chunk_concentration"] == 0.75
    assert d["distinct_docs_count"] == 2


def test_conflict_precedence_qst_0427_assembly():
    """Regression test: Ensure sibling spec chunks are not starved by lower-scoring noise documents."""
    builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3)
    # Simulate qst_0427 retrieval ranking:
    # Confluence chunks have rank 1, 2, 3 (scores 50.3, 49.6, 41.9)
    # Gmail thread has rank 4 (score 39.5)
    # Irrelevant docs have rank 5, 6 (scores 20.5, 18.5)
    confluence_id = "dsid_d93aed75845e4a0b8126a80d8a7a5a41"
    gmail_id = "dsid_86f033ed92b84f3a8b381d09b0b95357"
    noise_id_1 = "dsid_0516792a3e1a4d0082c33a21e609c895"
    noise_id_2 = "dsid_d3bc26457ef54c2881c5320c62a1da22"

    chunks = [
        _create_dummy_chunk(confluence_id, f"{confluence_id}__c000", "Overview and supersession notice", score=50.3),
        _create_dummy_chunk(confluence_id, f"{confluence_id}__c002", "Invoice XML snippet", score=49.6),
        _create_dummy_chunk(confluence_id, f"{confluence_id}__c001", "po_fingerprint convention: SHA-256 lowercase hex", score=41.9),
        _create_dummy_chunk(gmail_id, f"{gmail_id}__c003", "Earlier proposal: PO#|LINEUUID no hashing delivered Monday", score=39.5),
        _create_dummy_chunk(noise_id_1, f"{noise_id_1}__c000", "Irrelevant billing terms", score=20.5),
        _create_dummy_chunk(noise_id_2, f"{noise_id_2}__c001", "Irrelevant AP window", score=18.5),
    ]

    ctx = builder.build_context(chunks)

    # Confluence chunk 1 (SHA-256 specification) MUST be included
    chunk_ids = [p.chunk_id for p in ctx.passages]
    assert f"{confluence_id}__c001" in chunk_ids, "Crucial spec chunk was starved out!"
    # Gmail chunk 3 (superseded proposal) MUST also be included
    assert f"{gmail_id}__c003" in chunk_ids, "Superseded draft chunk was missing!"
    # Sibling chunks of confluence must be in sequential order: c000, c001, c002
    confluence_chunks = [p.chunk_id for p in ctx.passages if p.doc_id == confluence_id]
    assert confluence_chunks == [
        f"{confluence_id}__c000",
        f"{confluence_id}__c001",
        f"{confluence_id}__c002",
    ]


# -----------------------------------------------------------------------------
# 8. Milestone 1.2 Regressions & Grounding Audit Tests
# -----------------------------------------------------------------------------


def test_qst_0427_policy_selection_and_citation_alignment():
    """Verify ClearEdge qst_0427 citation-to-claim alignment.

    Requirements:
    1. Final policy: poFingerprint is lowercase SHA-256 hex digest of PO_NUMBER|PO_LINE_UUID.
    2. Weekly delivery: Tuesday by 07:00 PT.
    3. Cites confluence doc for final policy.
    4. Obsolete Monday / unhashed draft is described as superseded and cited using Gmail doc_id.
    5. Final Confluence doc is NOT cited as evidence for obsolete Monday delivery.
    """
    confluence_id = "dsid_d93aed75845e4a0b8126a80d8a7a5a41"
    gmail_id = "dsid_86f033ed92b84f3a8b381d09b0b95357"

    validator = CitationValidator()
    ctx = AssembledContext(
        formatted_context="",
        unique_doc_ids=[confluence_id, gmail_id],
        passages=[
            _create_dummy_passage(1, confluence_id),
            _create_dummy_passage(2, gmail_id),
        ],
    )

    # Compliant answer:
    compliant_answer = (
        f"The final po_fingerprint format is a lowercase SHA-256 hexadecimal digest of "
        f"`PO_NUMBER|PO_LINE_UUID` [{confluence_id}]. "
        f"The weekly invoice delivery time is Tuesdays by 07:00 PT [{confluence_id}]. "
        f"An earlier draft instructed to deliver invoices on Mondays and use unhashed `PO#|LINEUUID`, "
        f"but this guidance was explicitly superseded [{gmail_id}]."
    )

    res = validator.validate(compliant_answer, ctx)
    assert res.citation_precision == 1.0
    assert confluence_id in res.valid_doc_ids
    assert gmail_id in res.valid_doc_ids

    # Check sentence citation alignment
    for s in res.sentences:
        if "Mondays" in s.sentence or "unhashed" in s.sentence:
            # Must cite Gmail, MUST NOT cite Confluence
            assert gmail_id in s.cited_doc_ids, "Superseded claim failed to cite Gmail draft!"
            assert confluence_id not in s.cited_doc_ids, "Confluence falsely cited for obsolete Monday rule!"
        if "Tuesdays" in s.sentence or "SHA-256" in s.sentence:
            assert confluence_id in s.cited_doc_ids


def test_grounding_metrics_separation():
    """Verify separation of citation ID validity, sentence citation rate, and claim support.

    Distinguishes:
    - Valid ID pointing to an unsupported claim (hallucination with valid ID)
    - Unsourced substantive factual claim
    - Numerical, date, and negation sensitivity
    """
    doc_id = "dsid_11111111111111111111111111111111"
    validator = CitationValidator()
    judge = DeterministicJudge()

    # Document text defines: limit is 10 MiB, effective 2027-01-01, cannot deploy on weekends
    doc_text = "The per-file upload limit is 10 MiB, effective 2027-01-01. Deployments cannot be made on weekends."
    passage = EvidencePassage(
        passage_index=1,
        doc_id=doc_id,
        chunk_id=f"{doc_id}__c000",
        source_type="confluence",
        normalized_title="Limits",
        section_title="Policies",
        chunk_text=doc_text,
        score=1.0,
    )
    ctx = AssembledContext(
        formatted_context=doc_text,
        unique_doc_ids=[doc_id],
        passages=[passage],
    )

    # Case A: Correct grounded answer
    ans_correct = f"The upload limit is 10 MiB effective 2027-01-01 [{doc_id}]."
    val_a = validator.validate(ans_correct, ctx)
    assert val_a.citation_id_validity == 1.0
    assert val_a.sentence_citation_rate == 1.0
    assert val_a.claim_support_rate == 1.0

    # Case B: Fabricated number with a valid doc_id citation
    ans_wrong_number = f"The upload limit is 500 GiB for all files [{doc_id}]."
    val_b = validator.validate(ans_wrong_number, ctx)
    assert val_b.citation_id_validity == 1.0  # ID is valid in the database
    # But lexical claim-support detects low overlap on the fabricated claim
    assert val_b.sentences[0].is_supported_by_context is False

    # Case C: Prohibited constraint violation (asserting prohibited negation)
    prohibited_claims = ["The answer must not claim deployments are allowed on weekends."]
    ans_violating_negation = f"Deployments are allowed on weekends without restrictions [{doc_id}]."
    correct, completeness, expl = judge.evaluate(
        question="Can we deploy on weekends?",
        question_type="constrained",
        generated_answer=ans_violating_negation,
        gold_answer="",
        answer_facts=prohibited_claims,
        is_abstention=False,
        has_contradiction=False,
    )
    assert correct is False
    assert "asserted prohibited claim" in expl


