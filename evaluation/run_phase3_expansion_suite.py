"""Phase 3: Controlled Query Expansion and Question Decomposition Experiments.

This script executes controlled experiments comparing:
- C0: Baseline (No expansion / original query only)
- C1: Alternative Query Reformulation (2-3 complementary queries)
- C2: Domain Terminology Expansion (bridging colloquial vocabulary gaps)
- D1: Sub-Question Decomposition (decomposing multi-part/multi-evidence questions)

Evaluated across:
1. Group B: 14 Diagnostic Questions (terminology gaps, multi-doc, conflicts, candidate cutoff)
2. The 13 BM25 Recall@20 Audit Misses
3. Comparative analysis of gold document ranks, Recall@K, Complete Gold Coverage, and branch provenance.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from evaluation.evaluation_harness import (
    DIAGNOSTIC_QIDS,
    HARD_CASE_13_QIDS,
    load_development_benchmark,
)
from rag.llm import OpenAICompatibleLLM
from retrieval.indexer import EnterpriseRetriever
from retrieval.query_expansion import MultiQueryRetriever, QueryExpander

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase3_experiments():
    print("=" * 80)
    print("PHASE 3: QUERY EXPANSION & QUESTION DECOMPOSITION EXPERIMENTS")
    print("=" * 80)

    df_all, df_group_a, df_group_b = load_development_benchmark()

    # Load 13 hard cases subset
    df_13_misses = df_group_a[df_group_a["question_id"].isin(HARD_CASE_13_QIDS)]
    print(f"Loaded Group B: {len(df_group_b)} diagnostic questions.")
    print(f"Loaded 13-Failure subset: {len(df_13_misses)} questions.")

    # Initialize Retriever & LLM
    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )
    llm = OpenAICompatibleLLM(model="openai/gpt-oss-20b", timeout_seconds=30.0)
    expander = QueryExpander(llm_client=llm)
    mq_retriever = MultiQueryRetriever(retriever=retriever, expander=expander)

    # Evaluation targets: Union of Group B (14) and 13 misses (total 21 unique questions)
    target_qids = sorted(list(set(DIAGNOSTIC_QIDS + HARD_CASE_13_QIDS)))
    df_targets = df_all[df_all["question_id"].isin(target_qids)].copy()
    print(f"Total targeted evaluation set: {len(df_targets)} questions.\n")

    modes_to_test = [
        ("C0_baseline", "none"),
        ("C1_alternatives", "alternative_queries"),
        ("C2_terminology", "terminology_expansion"),
        ("D1_decomposition", "decomposition"),
    ]

    # Store detailed per-question results
    results_by_mode: Dict[str, Dict[str, Any]] = {name: {} for name, _ in modes_to_test}

    for mode_name, exp_mode in modes_to_test:
        print(f"--- Running Mode: {mode_name} (expansion_mode={exp_mode}) ---")
        t0 = time.perf_counter()
        
        for idx, row in df_targets.iterrows():
            qid = str(row["question_id"])
            query = str(row["question"])
            raw_g = row.get("expected_doc_ids")
            gold_docs = [str(x) for x in raw_g] if raw_g is not None and len(raw_g) > 0 else []

            # Retrieve with BM25 base
            t_q0 = time.perf_counter()
            doc_ids, chunks, exp_res, provenance = mq_retriever.retrieve(
                query=query,
                mode="bm25",
                expansion_mode=exp_mode,
                top_k_docs=40,
                diversity_filter=True,
            )
            elapsed_ms = (time.perf_counter() - t_q0) * 1000

            # Rank of each gold document
            gold_ranks = {}
            for gdoc in gold_docs:
                try:
                    rank = doc_ids.index(gdoc) + 1
                except ValueError:
                    rank = f">{len(doc_ids)}"
                gold_ranks[gdoc] = rank

            # Compute hit rates
            hits_5 = sum(1 for g in gold_docs if g in doc_ids[:5])
            hits_10 = sum(1 for g in gold_docs if g in doc_ids[:10])
            hits_20 = sum(1 for g in gold_docs if g in doc_ids[:20])
            hits_40 = sum(1 for g in gold_docs if g in doc_ids[:40])
            total_g = len(gold_docs) if gold_docs else 1

            results_by_mode[mode_name][qid] = {
                "question": query,
                "question_type": row.get("question_type"),
                "num_gold_docs": len(gold_docs),
                "gold_docs": gold_docs,
                "gold_ranks": gold_ranks,
                "recall_at_5": hits_5 / total_g,
                "recall_at_10": hits_10 / total_g,
                "recall_at_20": hits_20 / total_g,
                "recall_at_40": hits_40 / total_g,
                "complete_coverage_20": hits_20 == len(gold_docs) if gold_docs else False,
                "complete_coverage_40": hits_40 == len(gold_docs) if gold_docs else False,
                "num_expanded_queries": len(exp_res.expanded_queries),
                "expanded_queries": exp_res.expanded_queries,
                "subquestions": [sq.subquestion for sq in exp_res.subquestions],
                "retrieval_latency_ms": elapsed_ms,
            }

        elapsed_total = time.perf_counter() - t0
        print(f"Finished {mode_name} in {elapsed_total:.1f}s\n")

    # Aggregate summaries
    summary_by_mode = {}
    for mode_name, _ in modes_to_test:
        recs_all = list(results_by_mode[mode_name].values())
        recs_diag = [results_by_mode[mode_name][qid] for qid in DIAGNOSTIC_QIDS if qid in results_by_mode[mode_name]]
        recs_13 = [results_by_mode[mode_name][qid] for qid in HARD_CASE_13_QIDS if qid in results_by_mode[mode_name]]

        def calc_mean(records, key):
            vals = [r[key] for r in records if r["num_gold_docs"] > 0]
            return sum(vals) / len(vals) if vals else 0.0

        summary_by_mode[mode_name] = {
            "all_targets_mean_recall_20": calc_mean(recs_all, "recall_at_20"),
            "all_targets_mean_recall_40": calc_mean(recs_all, "recall_at_40"),
            "diagnostic_14_mean_recall_20": calc_mean(recs_diag, "recall_at_20"),
            "diagnostic_14_mean_recall_40": calc_mean(recs_diag, "recall_at_40"),
            "diagnostic_14_complete_cov_20": sum(1 for r in recs_diag if r["complete_coverage_20"]) / len(recs_diag) if recs_diag else 0.0,
            "diagnostic_14_complete_cov_40": sum(1 for r in recs_diag if r["complete_coverage_40"]) / len(recs_diag) if recs_diag else 0.0,
            "audit_13_misses_mean_recall_20": calc_mean(recs_13, "recall_at_20"),
            "audit_13_misses_mean_recall_40": calc_mean(recs_13, "recall_at_40"),
            "audit_13_misses_complete_cov_20": sum(1 for r in recs_13 if r["complete_coverage_20"]) / len(recs_13) if recs_13 else 0.0,
            "audit_13_misses_complete_cov_40": sum(1 for r in recs_13 if r["complete_coverage_40"]) / len(recs_13) if recs_13 else 0.0,
        }

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Configuration':<20} | {'14-Diag R@20':<12} | {'14-Diag R@40':<12} | {'14-Diag Cov@40':<14} | {'13-Miss R@20':<12} | {'13-Miss R@40':<12}")
    print("-" * 95)
    for mode_name, sm in summary_by_mode.items():
        print(
            f"{mode_name:<20} | "
            f"{sm['diagnostic_14_mean_recall_20']*100:>10.2f}% | "
            f"{sm['diagnostic_14_mean_recall_40']*100:>10.2f}% | "
            f"{sm['diagnostic_14_complete_cov_40']*100:>12.2f}% | "
            f"{sm['audit_13_misses_mean_recall_20']*100:>10.2f}% | "
            f"{sm['audit_13_misses_mean_recall_40']*100:>10.2f}%"
        )
    print("=" * 95)

    # Detailed ranks on key diagnostic questions
    print("\n--- Diagnostic Question Ranks Across Expansion Modes ---")
    key_diag = ["qst_0178", "qst_0242", "qst_0282", "qst_0436", "qst_0437", "qst_0447", "qst_0448", "qst_0427"]
    for qid in key_diag:
        if qid not in df_targets["question_id"].values:
            continue
        print(f"\nQuestion {qid}:")
        for mode_name, _ in modes_to_test:
            res = results_by_mode[mode_name].get(qid, {})
            ranks = res.get("gold_ranks", {})
            exp = res.get("expanded_queries", [])
            sqs = res.get("subquestions", [])
            print(f"  {mode_name:<18} -> Ranks: {ranks}")
            if exp:
                print(f"    Expansions: {exp[:2]}")
            if sqs:
                print(f"    Sub-questions: {sqs[:2]}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary_by_mode,
        "results_by_mode": results_by_mode,
    }

    report_path = REPORTS_DIR / "phase3_query_expansion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPhase 3 Report saved to {report_path}")


if __name__ == "__main__":
    run_phase3_experiments()
