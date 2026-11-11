"""Unit tests for AnswerVerifier and Self-Correction (rag/verifier.py)."""

from __future__ import annotations

import pytest

from rag.context_builder import AssembledContext, EvidencePassage
from rag.llm import MockLLM
from rag.verifier import AnswerVerifier, VerificationCheckResult


def _create_mock_passage(doc_id: str, text: str) -> EvidencePassage:
    return EvidencePassage(
        passage_index=1,
        doc_id=doc_id,
        chunk_id=f"{doc_id}__c000",
        source_type="confluence",
        normalized_title=f"Doc {doc_id}",
        section_title="Overview",
        chunk_text=text,
        score=10.0,
        latest_date=None,
        date_confidence="none",
        token_count=len(text) // 4,
    )


def test_verifier_supported_answer():
    doc_id = "dsid_11111111111111111111111111111111"
    pass_text = "The maximum allowed upload size is 250MB for batch processing."
    ctx = AssembledContext(
        formatted_context=pass_text,
        passages=[_create_mock_passage(doc_id, pass_text)],
        unique_doc_ids=[doc_id],
    )

    verifier = AnswerVerifier()
    ans = f"The maximum allowed upload size is 250MB for batch processing [{doc_id}]."
    res = verifier.verify("What is the upload size limit?", ans, ctx)

    assert res.is_verified is True
    assert res.citation_id_validity == 1.0
    assert res.has_hallucinated_citations is False
    assert res.has_numerical_inconsistencies is False


def test_verifier_numerical_inconsistency_detected():
    doc_id = "dsid_11111111111111111111111111111111"
    pass_text = "The system enforces a timeout of 150ms per token."
    ctx = AssembledContext(
        formatted_context=pass_text,
        passages=[_create_mock_passage(doc_id, pass_text)],
        unique_doc_ids=[doc_id],
    )

    verifier = AnswerVerifier()
    # Hallucinated number 999ms instead of 150ms
    ans = f"The system enforces a timeout of 999ms per token [{doc_id}]."
    res = verifier.verify("What is the timeout?", ans, ctx)

    assert res.has_numerical_inconsistencies is True
    assert len(res.numerical_mismatches) > 0
    assert "999" in res.numerical_mismatches[0]
    assert res.is_verified is False


def test_verifier_unsupported_claim_detected():
    doc_id = "dsid_11111111111111111111111111111111"
    pass_text = "Standard deployment steps are documented in wiki."
    ctx = AssembledContext(
        formatted_context=pass_text,
        passages=[_create_mock_passage(doc_id, pass_text)],
        unique_doc_ids=[doc_id],
    )

    verifier = AnswerVerifier()
    # Sentence with zero citations asserting substantive claims
    ans = "We deploy directly to production without any code reviews or canary tests."
    res = verifier.verify("How do we deploy?", ans, ctx)

    assert res.has_unsupported_claims is True
    assert res.is_verified is False
    assert res.revision_needed is True


def test_verifier_conflict_handling_omission():
    doc_final = "dsid_d93a61f5c6a1b853874c728e124803d1"
    doc_draft = "dsid_86f0c3d9a105c317be9ef282542a1b9c"
    ctx = AssembledContext(
        formatted_context="Final spec supersedes earlier draft.",
        passages=[
            _create_mock_passage(doc_final, "Final specification supersedes the June email draft. The hash is SHA-256."),
            _create_mock_passage(doc_draft, "Earlier draft proposed un-hashed string."),
        ],
        unique_doc_ids=[doc_final, doc_draft],
    )

    verifier = AnswerVerifier()
    # Answer only mentions the final spec, completely omitting the supersession/contradiction description
    ans = f"The required format is SHA-256 [{doc_final}]."
    res = verifier.verify("ClearEdge pilot po_fingerprint format?", ans, ctx)

    assert res.has_conflict_handling_error is True
    assert "supersession" in res.conflict_error_message.lower()
    assert res.is_verified is False


def test_verifier_self_correction_execution():
    doc_id = "dsid_11111111111111111111111111111111"
    pass_text = "Verified invoice delivery is Tuesday 07:00 PT."
    ctx = AssembledContext(
        formatted_context=pass_text,
        passages=[_create_mock_passage(doc_id, pass_text)],
        unique_doc_ids=[doc_id],
    )

    # Initial answer lacks citations
    initial_ans = "Invoice delivery is Tuesday 07:00 PT without fail."
    corrected_ans = f"Verified invoice delivery is Tuesday 07:00 PT [{doc_id}]."

    mock_llm = MockLLM(default_response=corrected_ans)
    verifier = AnswerVerifier(llm_client=mock_llm, max_revisions=1)

    final_text, check_res, rev_count = verifier.verify_and_correct(
        "When are invoices delivered?", initial_ans, ctx
    )

    assert rev_count == 1
    assert f"[{doc_id}]" in final_text
    assert check_res.is_verified is True
