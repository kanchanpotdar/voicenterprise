"""Deterministic retrieval and benchmark evaluation metrics (P0.4).

Implements:
- Document Recall@K (matching EnterpriseRAG-Bench official Document Recall when K=None)
- Hit@K (binary indicator of whether at least one expected document appears in top-K)
- Mean Reciprocal Rank (MRR@K)
- Invalid Extra Docs@K (number of irrelevant documents returned, lower is better)
- Per-category breakdown with explicit sample size (n) and gold-doc count reporting
- Placeholder interface for P2 LLM-as-a-judge (Answer Correctness & Answer Completeness)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: Optional[int] = None,
) -> Optional[float]:
    """Compute fraction of expected_doc_ids found in top-k retrieved_doc_ids.

    Returns None when expected_doc_ids is empty (e.g., 'info_not_found' or 'high_level'
    questions without specific ground-truth document IDs), so categories without gold
    documents do not distort document recall averages.
    """
    expected_set = set(expected_doc_ids)
    if not expected_set:
        return None
    deduped = _dedupe_preserve_order(retrieved_doc_ids)
    sliced = deduped[:k] if k is not None else deduped
    hits = len(expected_set.intersection(sliced))
    return hits / len(expected_set)


def hit_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: Optional[int] = None,
) -> Optional[float]:
    """Return 1.0 if at least one expected_doc_id is in top-k retrieved_doc_ids, else 0.0.

    Returns None when expected_doc_ids is empty.
    """
    expected_set = set(expected_doc_ids)
    if not expected_set:
        return None
    deduped = _dedupe_preserve_order(retrieved_doc_ids)
    sliced = deduped[:k] if k is not None else deduped
    return 1.0 if expected_set.intersection(sliced) else 0.0


def reciprocal_rank_at_k(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: Optional[int] = None,
) -> Optional[float]:
    """Compute Reciprocal Rank (1 / rank_of_first_relevant_doc) within top-k.

    Returns None when expected_doc_ids is empty.
    """
    expected_set = set(expected_doc_ids)
    if not expected_set:
        return None
    deduped = _dedupe_preserve_order(retrieved_doc_ids)
    sliced = deduped[:k] if k is not None else deduped
    for idx, doc_id in enumerate(sliced, start=1):
        if doc_id in expected_set:
            return 1.0 / idx
    return 0.0


def invalid_extra_docs_count(
    retrieved_doc_ids: Sequence[str],
    expected_doc_ids: Sequence[str],
    k: Optional[int] = None,
) -> int:
    """Count retrieved documents in top-k that are NOT in expected_doc_ids.

    Matches the EnterpriseRAG-Bench 'Invalid Extra Docs' metric (lower is better).
    Note: For 'info_not_found' questions (expected_doc_ids is empty), every returned
    document counts as an invalid extra doc.
    """
    expected_set = set(expected_doc_ids)
    deduped = _dedupe_preserve_order(retrieved_doc_ids)
    sliced = deduped[:k] if k is not None else deduped
    return sum(1 for doc_id in sliced if doc_id not in expected_set)


def overall_answer_score(answer_correct: bool, completeness_pct: float) -> float:
    """Compute official EnterpriseRAG-Bench Overall Score for a single question.

    Overall Score = answer_correct (0 or 1) * completeness (0.0 to 100.0).
    """
    if not (0.0 <= completeness_pct <= 100.0):
        raise ValueError(f"completeness_pct must be in [0, 100], got {completeness_pct}")
    return float(completeness_pct) if answer_correct else 0.0


@dataclass
class CategoryMetrics:
    """Aggregated retrieval metrics for a question category or overall split."""

    category: str
    num_questions: int
    num_questions_with_gold_docs: int
    recall_all: Optional[float]
    recall_at_k: Dict[int, Optional[float]]
    hit_at_k: Dict[int, Optional[float]]
    mrr_at_k: Dict[int, Optional[float]]
    avg_invalid_extra_docs_all: float
    avg_invalid_extra_docs_at_k: Dict[int, float]
    low_sample_warning: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean_ignore_none(values: Iterable[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def evaluate_retrieval_predictions(
    questions: Sequence[Dict[str, Any]],
    predictions_by_qid: Dict[str, Sequence[str]],
    ks: Sequence[int] = (5, 10, 20),
    low_sample_threshold: int = 15,
) -> Dict[str, Any]:
    """Evaluate retrieval predictions across all questions and break down by question_type.

    Args:
        questions: Sequence of question dicts containing at least:
            - 'question_id': str
            - 'question_type': str
            - 'expected_doc_ids': list[str]
        predictions_by_qid: Mapping from question_id -> ordered sequence of retrieved doc_ids.
        ks: Cutoff ranks K for Recall@K, Hit@K, MRR@K, and InvalidExtraDocs@K.
        low_sample_threshold: Flag categories with fewer than this many questions.

    Returns:
        Dictionary with 'overall', 'by_category', and 'per_question' results.
    """
    per_question: List[Dict[str, Any]] = []
    by_cat_records: Dict[str, List[Dict[str, Any]]] = {}

    for q in questions:
        qid = q["question_id"]
        qtype = q["question_type"]
        expected = list(q.get("expected_doc_ids") or [])
        retrieved = list(predictions_by_qid.get(qid, []))

        q_res: Dict[str, Any] = {
            "question_id": qid,
            "question_type": qtype,
            "num_expected_docs": len(expected),
            "num_retrieved_docs": len(_dedupe_preserve_order(retrieved)),
            "recall_all": recall_at_k(retrieved, expected, k=None),
            "invalid_extra_docs_all": invalid_extra_docs_count(retrieved, expected, k=None),
            "recall_at_k": {k: recall_at_k(retrieved, expected, k=k) for k in ks},
            "hit_at_k": {k: hit_at_k(retrieved, expected, k=k) for k in ks},
            "mrr_at_k": {k: reciprocal_rank_at_k(retrieved, expected, k=k) for k in ks},
            "invalid_extra_docs_at_k": {
                k: invalid_extra_docs_count(retrieved, expected, k=k) for k in ks
            },
        }
        per_question.append(q_res)
        by_cat_records.setdefault(qtype, []).append(q_res)

    def _aggregate(cat_name: str, records: List[Dict[str, Any]]) -> CategoryMetrics:
        n_total = len(records)
        n_with_gold = sum(1 for r in records if r["num_expected_docs"] > 0)
        return CategoryMetrics(
            category=cat_name,
            num_questions=n_total,
            num_questions_with_gold_docs=n_with_gold,
            recall_all=_mean_ignore_none(r["recall_all"] for r in records),
            recall_at_k={
                k: _mean_ignore_none(r["recall_at_k"][k] for r in records) for k in ks
            },
            hit_at_k={
                k: _mean_ignore_none(r["hit_at_k"][k] for r in records) for k in ks
            },
            mrr_at_k={
                k: _mean_ignore_none(r["mrr_at_k"][k] for r in records) for k in ks
            },
            avg_invalid_extra_docs_all=(
                sum(r["invalid_extra_docs_all"] for r in records) / n_total
                if n_total > 0
                else 0.0
            ),
            avg_invalid_extra_docs_at_k={
                k: (
                    sum(r["invalid_extra_docs_at_k"][k] for r in records) / n_total
                    if n_total > 0
                    else 0.0
                )
                for k in ks
            },
            low_sample_warning=(n_total < low_sample_threshold),
        )

    overall_metrics = _aggregate("ALL", per_question)
    category_metrics = {
        cat: _aggregate(cat, recs).to_dict()
        for cat, recs in sorted(by_cat_records.items())
    }

    return {
        "overall": overall_metrics.to_dict(),
        "by_category": category_metrics,
        "per_question": per_question,
    }
