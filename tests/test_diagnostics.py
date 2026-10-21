"""Unit tests for P1 diagnostic inspection tools (evaluation/inspect_p1_checks.py)."""

import inspect
import numpy as np
import pandas as pd

from evaluation.inspect_p1_checks import (
    _has_expected_docs,
    check_2_read_failed_queries,
    classify_retrieval_failure,
    get_stratified_sample,
)


def test_stratified_round_robin_sampling_and_determinism():
    df = pd.DataFrame(
        [
            {"question_id": "q1", "question_type": "basic"},
            {"question_id": "q2", "question_type": "basic"},
            {"question_id": "q3", "question_type": "basic"},
            {"question_id": "q4", "question_type": "semantic"},
            {"question_id": "q5", "question_type": "semantic"},
            {"question_id": "q6", "question_type": "completeness"},
        ]
    )

    s1 = get_stratified_sample(df, num_samples=4)
    s2 = get_stratified_sample(df, num_samples=4)

    # Deterministic across repeated calls (used by dense vs hybrid comparison)
    assert list(s1["question_id"]) == list(s2["question_id"])
    # Round-robin across sorted categories ('basic', 'completeness', 'semantic'), then 2nd round 'basic'
    assert list(s1["question_id"]) == ["q1", "q6", "q4", "q2"]
    assert "question_type" in s1.columns


def test_stratified_sample_handles_missing_question_type_column_and_nulls():
    df_no_col = pd.DataFrame([{"question_id": "q1"}, {"question_id": "q2"}, {"question_id": "q3"}])
    sampled = get_stratified_sample(df_no_col, num_samples=2)
    assert list(sampled["question_id"]) == ["q1", "q2"]

    df_with_nulls = pd.DataFrame(
        [
            {"question_id": "q1", "question_type": None},
            {"question_id": "q2", "question_type": "basic"},
        ]
    )
    sampled_nulls = get_stratified_sample(df_with_nulls, num_samples=2)
    assert len(sampled_nulls) == 2


def test_has_expected_docs_handles_numpy_arrays_without_ambiguity():
    arr_non_empty = np.array(["dsid_1", "dsid_2"])
    arr_empty = np.array([])

    assert _has_expected_docs(arr_non_empty) is True
    assert _has_expected_docs(arr_empty) is False
    assert _has_expected_docs(None) is False
    assert _has_expected_docs(["dsid_1"]) is True
    assert _has_expected_docs([]) is False


def test_classify_retrieval_failure_labels():
    indexed = {f"d{i}" for i in range(1, 20)}

    # 1. Missing from corpus
    c_missing = classify_retrieval_failure(
        expected=["d_not_indexed"],
        retrieved_docs=["d1", "d2", "d3", "d4", "d5"],
        indexed_doc_ids=indexed,
    )
    assert c_missing == "missing_from_corpus"

    # 2. Multi-doc budget (>5 gold docs)
    c_budget = classify_retrieval_failure(
        expected=["d1", "d2", "d3", "d4", "d5", "d6", "d7"],
        retrieved_docs=["d1", "d2", "d3", "d4", "d5", "d6", "d8"],
        indexed_doc_ids=indexed,
    )
    assert c_budget.startswith("multi_doc_budget")

    # 3. Ranking suboptimal (missed at top-5, found in ranks 6..20)
    c_rank = classify_retrieval_failure(
        expected=["d10"],
        retrieved_docs=["d1", "d2", "d3", "d4", "d5", "d10", "d11"],
        indexed_doc_ids=indexed,
    )
    assert c_rank.startswith("ranking_suboptimal")

    # 4. Lexical/semantic gap (not found in top-20)
    c_gap = classify_retrieval_failure(
        expected=["d19"],
        retrieved_docs=["d1", "d2", "d3", "d4", "d5", "d6", "d7"],
        indexed_doc_ids=indexed,
    )
    assert c_gap.startswith("lexical_semantic_gap")


def test_default_diagnostic_sample_size_is_30():
    sig = inspect.signature(check_2_read_failed_queries)
    assert sig.parameters["sample_size"].default == 30
