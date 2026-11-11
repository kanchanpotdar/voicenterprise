"""Tests for the audit scorer. Includes manually labelled real Phase 6 outputs."""

import json
from pathlib import Path

import pytest

from evaluation.audit_scoring import (
    abstains,
    caveats_missing_detail,
    constraint_violation_suspected,
    fact_covered,
    is_empty_or_truncated,
    score_answer,
)

REPORT = Path(__file__).resolve().parent.parent / "evaluation" / "reports" / "phase6_end_to_end_report.json"


def test_number_anchor_required():
    assert not fact_covered("The default pass-rate threshold is 0.92.", "The default pass-rate threshold is not stated.")
    assert fact_covered("The default pass-rate threshold is 0.92.", "The default pass-rate threshold is 0.92.")


def test_abstention_never_scores_complete_on_answerable():
    s = score_answer(
        "q", "semantic",
        "The supplied evidence does not contain any statement about the default pass rate threshold.",
        ["The default pass-rate threshold is 0.92."],
    )
    assert s.abstains and not s.answer_correct_strict and s.fact_completeness_pct == 0.0


def test_info_not_found_abstention_is_correct():
    s = score_answer("q", "info_not_found", "The documents do not contain this information.", ["The answer must state that it is not answerable."])
    assert s.answer_correct_strict


def test_info_not_found_hallucination_is_wrong():
    s = score_answer("q", "info_not_found", "The topic is named orders-v2 [dsid_" + "a" * 32 + "].", ["The answer must state that it is not answerable."])
    assert not s.answer_correct_strict


def test_info_not_found_with_component_description_and_missing_detail_caveat():
    # qst_0485 pattern: documents describe component (routing token fields) but omit specific wire format
    ans = (
        "Queue-token format in admission response:\n"
        "1. Encoding - a compact token carried by edge router. The exact wire-format (e.g., protobuf, base-64) "
        "is not spelled out, but the token is defined as a structured object [dsid_" + "a" * 32 + "].\n"
        "2. Required fields: token_id, tenant_id, policy_version."
    )
    facts = [
        "The answer must state at some point that the query is not fully answerable from available documents "
        "or caveat the provided information with why it does not fully address the query."
    ]
    s = score_answer("qst_0485", "info_not_found", ans, facts)
    assert caveats_missing_detail(ans)
    assert s.answer_correct_strict
    assert s.fact_completeness_pct == 100.0


def test_info_not_found_hallucinating_omitted_detail_fails():
    # Fabricating wire-format without caveating that it is omitted from evidence
    ans = (
        "Queue-token format in admission response:\n"
        "1. Encoding - encoded as proto3 binary wire format with snappy compression [dsid_" + "a" * 32 + "].\n"
        "2. Required fields: token_id, tenant_id, policy_version."
    )
    facts = [
        "The answer must state at some point that the query is not fully answerable from available documents "
        "or caveat the provided information with why it does not fully address the query."
    ]
    s = score_answer("qst_0485", "info_not_found", ans, facts)
    assert not caveats_missing_detail(ans)
    assert not s.abstains
    assert not s.answer_correct_strict
    assert s.fact_completeness_pct == 0.0


def test_multi_part_answerable_with_partial_caveat_not_rejected():
    # qst_0312 pattern: answerable question answered in paragraph 1 with citation;
    # paragraph 2 notes that secondary detail (path) is not in evidence.
    ans = (
        "The updated on-call runbook must be pushed to Confluence by 2026-08-16 [dsid_" + "b" * 32 + "].\n\n"
        "The evidence does not contain a Confluence URL or path for the on-call runbook, so that information "
        "cannot be provided from the supplied documents."
    )
    facts = [
        "The updated on-call runbook is due to be pushed to Confluence by 2026-08-16.",
        "The Confluence path for the on-call runbook is internal/runbooks/oncall.",
    ]
    s = score_answer("qst_0312", "intra_document_reasoning", ans, facts)
    # Must NOT be rejected as a full abstention!
    assert not s.abstains
    # Covered the first fact (2026-08-16)
    assert "The updated on-call runbook is due to be pushed to Confluence by 2026-08-16." in s.covered_facts
    # Missing the second fact (internal/runbooks/oncall)
    assert "The Confluence path for the on-call runbook is internal/runbooks/oncall." in s.missing_facts
    assert s.fact_completeness_pct == 50.0



def test_empty_and_truncated_detected():
    assert is_empty_or_truncated("\n\n[TRUNCATION DETECTED: x]")["empty"]
    assert not is_empty_or_truncated("A real answer that is long enough.\n[TRUNCATION DETECTED: x]")["empty"]


def test_negated_mention_is_not_a_violation():
    c = "The answer must not claim the final po_fingerprint uses SHA-1."
    assert not constraint_violation_suspected(c, "Do not use SHA-1; the final value is SHA-256.")
    assert constraint_violation_suspected(c, "The final po_fingerprint uses SHA-1.")


def test_superseded_label_is_not_a_violation():
    c = "The answer must not claim the final delivery time is Monday 08:00 PT."
    assert not constraint_violation_suspected(c, "The previous SLA was Monday 08:00 PT.")


def test_fact_with_only_stopwords_overlap_fails():
    assert not fact_covered("Northstar Bank was granted an exception with payload logging disabled and metadata retained for 30 days.", "The bank was granted nothing relevant.")


@pytest.mark.skipif(not REPORT.exists(), reason="historical report missing")
def test_manual_labels_on_real_outputs():
    import pandas as pd
    df = pd.read_parquet(Path(__file__).resolve().parent.parent / "data" / "splits" / "questions_dev.parquet")
    d = json.loads(REPORT.read_text())["results_by_config"]
    facts = {r.question_id: list(r.answer_facts) for r in df.itertuples()}
    types = {r.question_id: r.question_type for r in df.itertuples()}
    # Manually read: these answers abstain / are empty and are NOT correct.
    wrong = [
        ("qst_0178", c) for c in ["E0_Baseline", "E1_BestRetrieval", "E2_RetrievalExpansion", "E3_Decomposition", "E4_CoverageContext"]
    ] + [("qst_0214", "E0_Baseline"), ("qst_0214", "E4_CoverageContext"), ("qst_0437", "E1_BestRetrieval"), ("qst_0437", "E4_CoverageContext")]
    for q, c in wrong:
        s = score_answer(q, types[q], d[c][q]["answer_text"], facts[q])
        assert not s.answer_correct_strict, (q, c)
    # Manually read: qst_0427 E4 states SHA-256, no whitespace, raw unhashed forbidden, Tuesdays 07:00, superseded older guidance.
    s = score_answer("qst_0427", types["qst_0427"], d["E4_CoverageContext"]["qst_0427"]["answer_text"], facts["qst_0427"])
    # Known, documented false negative: the answer states `PO_NUMBER|PO_LINE_UUID` and "no whitespace"
    # but never uses the words "pipe delimiter", so the lexical scorer (recall >= 0.6) rejects it.
    # A human reviewer would accept it. The scorer is intentionally NOT tuned to hide this.
    assert len(s.missing_facts) == 1 and "pipe delimiter" in s.missing_facts[0]
    assert s.fact_completeness_pct == pytest.approx(80.0)
    assert not s.suspected_constraint_violations
