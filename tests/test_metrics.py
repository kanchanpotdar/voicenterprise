"""Unit tests for P0.4 evaluation metrics (evaluation/metrics.py)."""

import math
import pytest

from evaluation.metrics import (
    evaluate_retrieval_predictions,
    hit_at_k,
    invalid_extra_docs_count,
    overall_answer_score,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_recall_at_k_basic_and_cutoff():
    expected = ["doc_1", "doc_2", "doc_3", "doc_4"]
    retrieved = ["doc_9", "doc_2", "doc_8", "doc_4", "doc_1"]

    assert math.isclose(recall_at_k(retrieved, expected, k=2), 0.25)
    assert math.isclose(recall_at_k(retrieved, expected, k=4), 0.50)
    assert math.isclose(recall_at_k(retrieved, expected, k=None), 0.75)


def test_metrics_with_empty_expected_docs():
    # info_not_found and high_level have 0 expected_doc_ids
    retrieved = ["doc_a", "doc_b"]
    expected = []

    assert recall_at_k(retrieved, expected, k=5) is None
    assert hit_at_k(retrieved, expected, k=5) is None
    assert reciprocal_rank_at_k(retrieved, expected, k=5) is None
    assert invalid_extra_docs_count(retrieved, expected, k=5) == 2
    assert invalid_extra_docs_count([], expected, k=5) == 0


def test_deduplication_in_retrieved_docs():
    expected = ["doc_1", "doc_2"]
    retrieved = ["doc_1", "doc_1", "doc_9", "doc_2"]

    # After order-preserving deduplication: ["doc_1", "doc_9", "doc_2"]
    assert math.isclose(recall_at_k(retrieved, expected, k=2), 0.5)
    assert math.isclose(recall_at_k(retrieved, expected, k=3), 1.0)
    assert invalid_extra_docs_count(retrieved, expected, k=None) == 1


def test_hit_and_mrr_at_k():
    expected = ["doc_target"]
    retrieved = ["doc_x", "doc_y", "doc_target", "doc_z"]

    assert hit_at_k(retrieved, expected, k=2) == 0.0
    assert hit_at_k(retrieved, expected, k=3) == 1.0
    assert reciprocal_rank_at_k(retrieved, expected, k=2) == 0.0
    assert math.isclose(reciprocal_rank_at_k(retrieved, expected, k=3), 1.0 / 3.0)


def test_overall_answer_score():
    assert overall_answer_score(True, 85.0) == 85.0
    assert overall_answer_score(False, 100.0) == 0.0
    with pytest.raises(ValueError):
        overall_answer_score(True, 105.0)


def test_evaluate_retrieval_predictions_breakdown():
    questions = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "expected_doc_ids": ["d1"],
        },
        {
            "question_id": "q2",
            "question_type": "completeness",
            "expected_doc_ids": ["d2", "d3"],
        },
        {
            "question_id": "q3",
            "question_type": "info_not_found",
            "expected_doc_ids": [],
        },
    ]
    preds = {
        "q1": ["d1", "d_extra"],
        "q2": ["d_wrong", "d2"],
        "q3": ["d_hallucinated"],
    }

    report = evaluate_retrieval_predictions(questions, preds, ks=(1, 2), low_sample_threshold=10)
    overall = report["overall"]

    assert overall["num_questions"] == 3
    assert overall["num_questions_with_gold_docs"] == 2
    # Recall@2 over q1 (1.0) and q2 (0.5) -> 0.75; q3 is None (ignored in recall)
    assert math.isclose(overall["recall_at_k"][2], 0.75)
    # Invalid extra docs at k=2: q1 has 1, q2 has 1, q3 has 1 -> avg 1.0
    assert math.isclose(overall["avg_invalid_extra_docs_at_k"][2], 1.0)
    assert report["by_category"]["info_not_found"]["recall_all"] is None
    assert report["by_category"]["info_not_found"]["low_sample_warning"] is True
