"""Phase 2 Controlled Retrieval Experiments: Baselines and Diversity Controls.

Evaluates 6 configurations across all 178 answerable development questions (Group A):
1. BM25, K=20
2. BM25, K=40
3. Dense retrieval, K=20
4. Dense retrieval, K=40
5. Hybrid RRF, K=20
6. Hybrid RRF, K=40

Also tests Experiment B:
- BM25 K=40 without diversity vs BM25 K=40 with diversity (max_chunks_per_doc=3, Jaccard=0.70)
- Hybrid K=40 without diversity vs Hybrid K=40 with diversity

Detailed tracking for Group B (14 Diagnostic questions):
- qst_0178, qst_0242, qst_0282, qst_0214, qst_0203
- qst_0043, qst_0139, qst_0436, qst_0437, qst_0447, qst_0448, qst_0427, qst_0499, qst_0481
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from evaluation.evaluation_harness import (
    DIAGNOSTIC_QIDS,
    HARD_CASE_13_QIDS,
    load_development_benchmark,
    score_retrieval_predictions,
)
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent


def run_phase2_experiments():
    print("=" * 80)
    print("PHASE 2: CONTROLLED RETRIEVAL EXPERIMENTS (BASELINES & DIVERSITY)")
    print("=" * 80)

    df_all, df_group_a, df_group_b = load_development_benchmark()
    print(f"Loaded {len(df_all)} total dev questions.")
    print(f"Group A (Answerable with gold docs): {len(df_group_a)} questions.")
    print(f"Group B (Diagnostic questions): {len(df_group_b)} questions.")

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )

    # Check if we already have cached predictions from prior verified runs
    raw_cache_path = BASE_DIR / "evaluation" / "reports" / "retrieval_experiment_raw_cache.json"
    hybrid_cache_path = BASE_DIR / "evaluation" / "reports" / "hybrid_retrieval_cache.json"

    cached_bm25_k20 = {}
    cached_bm25_k40 = {}
    cached_bm25_k40_div = {}
    cached_hybrid_k20 = {}
    cached_hybrid_k40_div = {}

    if raw_cache_path.exists():
        with open(raw_cache_path) as f:
            c = json.load(f)
            cached_bm25_k20 = c.get("preds_baseline", {})
            cached_bm25_k40 = c.get("preds_exp_a", {})
            cached_bm25_k40_div = c.get("preds_exp_b", {})

    if hybrid_cache_path.exists():
        with open(hybrid_cache_path) as f:
            hc = json.load(f)
            cached_hybrid_k20 = hc.get("preds_hybrid_20", {})
            cached_hybrid_k40_div = hc.get("preds_hybrid_40", {})

    # Compute Dense K=20 and Dense K=40 predictions across all answerable questions
    dense_k20_preds: Dict[str, List[str]] = {}
    dense_k40_preds: Dict[str, List[str]] = {}
    hybrid_k40_nodiv_preds: Dict[str, List[str]] = {}

    print("\nComputing Dense retrieval predictions across Group A...", flush=True)
    t_dense0 = time.perf_counter()
    for idx, row in df_group_a.iterrows():
        qid = str(row["question_id"])
        query = str(row["question"])
        docs_40, _ = retriever.retrieve_documents(query, mode="dense", top_k_docs=40, diversity_filter=False)
        dense_k40_preds[qid] = docs_40
        dense_k20_preds[qid] = docs_40[:20]
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_group_a):
            print(f"  Processed {idx + 1}/{len(df_group_a)} dense queries ({time.perf_counter() - t_dense0:.1f}s elapsed)...", flush=True)

    # Compute Hybrid K=40 without diversity if not already cached
    print("\nComputing Hybrid K=40 (No Diversity) predictions...", flush=True)
    t_hyb0 = time.perf_counter()
    for idx, row in df_group_a.iterrows():
        qid = str(row["question_id"])
        query = str(row["question"])
        docs_40, _ = retriever.retrieve_documents(query, mode="hybrid", top_k_docs=40, diversity_filter=False)
        hybrid_k40_nodiv_preds[qid] = docs_40
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df_group_a):
            print(f"  Processed {idx + 1}/{len(df_group_a)} hybrid queries ({time.perf_counter() - t_hyb0:.1f}s elapsed)...", flush=True)

    # Score all 6 primary configurations
    configs = [
        ("BM25 (K=20)", cached_bm25_k20),
        ("BM25 (K=40)", cached_bm25_k40),
        ("Dense (K=20)", dense_k20_preds),
        ("Dense (K=40)", dense_k40_preds),
        ("Hybrid (K=20)", cached_hybrid_k20),
        ("Hybrid (K=40)", hybrid_k40_nodiv_preds),
    ]

    metrics_by_config = {}
    for name, preds in configs:
        m = score_retrieval_predictions(name, preds, df_group_a)
        metrics_by_config[name] = m.to_dict()

    # Experiment B: Diversity filtering comparison
    m_bm25_div = score_retrieval_predictions("BM25 (K=40 + Div)", cached_bm25_k40_div, df_group_a)
    m_hybrid_div = score_retrieval_predictions("Hybrid (K=40 + Div)", cached_hybrid_k40_div, df_group_a)
    metrics_by_config["BM25 (K=40 + Div)"] = m_bm25_div.to_dict()
    metrics_by_config["Hybrid (K=40 + Div)"] = m_hybrid_div.to_dict()

    # Diagnostic question ranks breakdown
    diag_summary = {}
    for row in df_group_b.to_dict(orient="records"):
        qid = row["question_id"]
        qtype = row["question_type"]
        raw_g = row.get("expected_doc_ids")
        gold_docs = [str(x) for x in raw_g] if raw_g is not None and len(raw_g) > 0 else []
        diag_summary[qid] = {
            "question_type": qtype,
            "num_gold_docs": len(gold_docs),
            "ranks_by_config": {},
        }
        for cfg_name, preds in [
            ("BM25_K20", cached_bm25_k20),
            ("BM25_K40", cached_bm25_k40),
            ("Dense_K20", dense_k20_preds),
            ("Dense_K40", dense_k40_preds),
            ("Hybrid_K20", cached_hybrid_k20),
            ("Hybrid_K40", hybrid_k40_nodiv_preds),
            ("Hybrid_K40_Div", cached_hybrid_k40_div),
        ]:
            retrieved = preds.get(qid, [])
            ranks = {}
            for gdoc in gold_docs:
                try:
                    rank = retrieved.index(gdoc) + 1
                except ValueError:
                    rank = f">{len(retrieved)}"
                ranks[gdoc] = rank
            diag_summary[qid]["ranks_by_config"][cfg_name] = ranks

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_answerable_questions": len(df_group_a),
        "configurations": metrics_by_config,
        "diagnostic_questions_ranks": diag_summary,
    }

    out_path = BASE_DIR / "evaluation" / "reports" / "phase2_retrieval_experiments_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Phase 2 Retrieval Report to: {out_path}", flush=True)

    # Print summary table
    print("\n" + "=" * 110)
    print("PHASE 2 SUMMARY TABLE (Group A: 178 Answerable Development Questions)")
    print("=" * 110)
    header = f"{'Configuration':<24} | {'R@5':<8} | {'R@10':<8} | {'R@20':<8} | {'R@40':<8} | {'Hit@10':<8} | {'MRR@10':<8} | {'Complete@40':<12} | {'Hard-Case Mean':<14}"
    print(header)
    print("-" * 110)
    for name in [
        "BM25 (K=20)", "BM25 (K=40)", "BM25 (K=40 + Div)",
        "Dense (K=20)", "Dense (K=40)",
        "Hybrid (K=20)", "Hybrid (K=40)", "Hybrid (K=40 + Div)"
    ]:
        d = metrics_by_config[name]
        print(
            f"{name:<24} | "
            f"{d['recall_at_5']*100.0:>6.2f}% | "
            f"{d['recall_at_10']*100.0:>6.2f}% | "
            f"{d['recall_at_20']*100.0:>6.2f}% | "
            f"{d['recall_at_40']*100.0:>6.2f}% | "
            f"{d['hit_at_10']*100.0:>6.2f}% | "
            f"{d['mrr_at_10']:>8.4f} | "
            f"{d['complete_gold_coverage_at_40']*100.0:>10.2f}% | "
            f"{d['hard_case_13_mean_recall']*100.0:>12.2f}%"
        )
    print("=" * 110)
    retriever.client.close()


if __name__ == "__main__":
    run_phase2_experiments()
