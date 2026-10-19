"""Deterministic question splitting and evaluation protocol (P0.2).

Addresses two critical methodological requirements:
1. Prevents gold-document leakage between dev and held-out test sets by grouping
   questions that share any expected_doc_id into connected components before splitting.
2. Protects small question categories (10-20 total examples) from tiny dev sample sizes
   by applying category-aware allocation (50% dev on small categories <= 30 examples,
   35% dev on large categories >= 40 examples) AND assigning 5-fold cross-validation
   indices within the dev split for repeated development evaluation.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq


def _find_connected_components(df: pd.DataFrame) -> List[List[int]]:
    """Group row indices of questions that share any expected_doc_id."""
    doc_to_rows: Dict[str, List[int]] = defaultdict(list)
    for idx, doc_ids in enumerate(df["expected_doc_ids"]):
        for doc_id in doc_ids:
            doc_to_rows[doc_id].append(idx)

    adj: Dict[int, Set[int]] = defaultdict(set)
    for rows in doc_to_rows.values():
        if len(rows) > 1:
            for r1 in rows:
                for r2 in rows:
                    if r1 != r2:
                        adj[r1].add(r2)

    visited: Set[int] = set()
    components: List[List[int]] = []
    for idx in range(len(df)):
        if idx in visited:
            continue
        comp: List[int] = []
        stack = [idx]
        visited.add(idx)
        while stack:
            curr = stack.pop()
            comp.append(curr)
            for nxt in sorted(adj[curr]):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))
    return components


def create_question_splits(
    questions_parquet_path: Path,
    output_dir: Path,
    seed: int = 42,
    num_dev_folds: int = 5,
) -> Dict[str, Any]:
    """Create leakage-free dev and held-out test splits from questions/test.parquet."""
    table = pq.read_table(questions_parquet_path)
    df = table.to_pandas()

    components = _find_connected_components(df)
    multi_question_components = [c for c in components if len(c) > 1]

    # Target dev ratio per category:
    # Small categories (<= 30 total questions) get 50% in dev so dev tuning has at least
    # 5-15 examples per category; larger categories (>= 40 questions) get 35% in dev.
    category_counts = df["question_type"].value_counts().to_dict()
    target_dev_counts: Dict[str, int] = {}
    for cat, total in category_counts.items():
        if total <= 30:
            target_dev_counts[cat] = total // 2
        else:
            target_dev_counts[cat] = int(round(total * 0.35))

    rng = random.Random(seed)

    # Group components by their primary question_type
    comps_by_cat: Dict[str, List[List[int]]] = defaultdict(list)
    for comp in components:
        primary_cat = df.iloc[comp[0]]["question_type"]
        comps_by_cat[primary_cat].append(comp)

    dev_indices: Set[int] = set()
    test_indices: Set[int] = set()

    # First assign multi-question components deterministically so shared gold docs stay together
    current_dev_by_cat: Dict[str, int] = defaultdict(int)
    for cat in sorted(comps_by_cat.keys()):
        cat_comps = list(comps_by_cat[cat])
        rng.shuffle(cat_comps)
        # Sort multi-node components first to place them cleanly without exceeding target
        cat_comps.sort(key=lambda c: len(c), reverse=True)
        for comp in cat_comps:
            if any(idx in dev_indices or idx in test_indices for idx in comp):
                continue
            # Check if placing in dev keeps us at or closer to target_dev_counts
            comp_cats = [df.iloc[i]["question_type"] for i in comp]
            can_fit_dev = all(
                current_dev_by_cat[c] + comp_cats.count(c) <= target_dev_counts[c]
                for c in set(comp_cats)
            )
            if can_fit_dev:
                for i in comp:
                    dev_indices.add(i)
                    current_dev_by_cat[df.iloc[i]["question_type"]] += 1
            else:
                for i in comp:
                    test_indices.add(i)

    # Verify zero overlap in expected_doc_ids between dev and test
    dev_docs = set(df.iloc[sorted(dev_indices)]["expected_doc_ids"].explode().dropna())
    test_docs = set(df.iloc[sorted(test_indices)]["expected_doc_ids"].explode().dropna())
    shared_docs = dev_docs.intersection(test_docs)
    if shared_docs:
        raise RuntimeError(
            f"Leakage detected! {len(shared_docs)} expected_doc_ids shared between dev and test."
        )

    df["split"] = [
        "dev" if i in dev_indices else "held_out_test" for i in range(len(df))
    ]

    # Assign stratified K-fold indices within dev split for repeated dev experiments
    df["dev_fold"] = -1
    for cat in sorted(category_counts.keys()):
        cat_dev_rows = df[(df["split"] == "dev") & (df["question_type"] == cat)].index.tolist()
        rng.shuffle(cat_dev_rows)
        for rank, row_idx in enumerate(cat_dev_rows):
            df.loc[row_idx, "dev_fold"] = rank % num_dev_folds

    output_dir.mkdir(parents=True, exist_ok=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    test_df = df[df["split"] == "held_out_test"].reset_index(drop=True)

    dev_path = output_dir / "questions_dev.parquet"
    test_path = output_dir / "questions_held_out_test.parquet"
    all_splits_path = output_dir / "questions_with_splits.parquet"

    dev_df.to_parquet(dev_path, index=False)
    test_df.to_parquet(test_path, index=False)
    df.to_parquet(all_splits_path, index=False)

    # Build summary statistics per category
    category_breakdown: Dict[str, Any] = {}
    for cat in sorted(category_counts.keys()):
        n_total = int(category_counts[cat])
        n_dev = int((dev_df["question_type"] == cat).sum())
        n_test = int((test_df["question_type"] == cat).sum())
        category_breakdown[cat] = {
            "total": n_total,
            "dev": n_dev,
            "held_out_test": n_test,
            "dev_ratio": round(n_dev / n_total, 4),
            "low_sample_warning": n_dev < 15 or n_test < 15,
        }

    summary = {
        "seed": seed,
        "num_dev_folds": num_dev_folds,
        "total_questions": len(df),
        "dev_questions": len(dev_df),
        "held_out_test_questions": len(test_df),
        "unique_dev_expected_docs": len(dev_docs),
        "unique_held_out_test_expected_docs": len(test_docs),
        "shared_expected_docs_between_splits": len(shared_docs),
        "num_multi_question_components": len(multi_question_components),
        "category_breakdown": category_breakdown,
        "protocol_rules": {
            "holdout_policy": (
                "questions_held_out_test.parquet must NEVER be inspected for prompt engineering, "
                "chunk size tuning, threshold selection, or entity-extraction rule design."
            ),
            "small_category_policy": (
                "Categories with low_sample_warning=True have <=15 examples per split; "
                "use 5-fold cross-validation on dev (dev_fold 0..4) and report sample sizes (n) "
                "alongside any category-level claims."
            ),
        },
    }

    summary_path = output_dir / "split_protocol.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    q_path = base / "EnterpriseRAG-Bench" / "data" / "questions" / "test.parquet"
    out_dir = base / "data" / "splits"
    result = create_question_splits(q_path, out_dir, seed=42, num_dev_folds=5)
    print(json.dumps(result, indent=2))
