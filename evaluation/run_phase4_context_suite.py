"""Phase 4: Coverage-Aware Context Construction Experiments.

This script executes controlled experiments comparing:
- Baseline ContextBuilder: Rank-ordered, deduplicated chunk packing.
- CoverageAwareContextBuilder: Greedy evidence coverage with novelty bonuses,
  redundancy penalties, parent document diversity, and authority/supersession boosting.

Evaluated across:
1. Group B: 14 Diagnostic Questions
2. Multi-document evidence targets:
   - qst_0436: Multi-language SDK reports
   - qst_0437: Multi-incident postmortems and team assignments
   - qst_0447: Multi-phase emergency hotfix SOP across surfaces
   - qst_0448: Multi-account customer retention exceptions
   - qst_0427: ClearEdge PO fingerprint specification superseding draft guidance
3. Metrics:
   - Gold Document Coverage in Context (fraction of gold docs present in assembled passages)
   - Complete Gold Coverage in Context (100% of gold docs present in assembled passages)
   - Number of distinct parent documents in context
   - Dropped sibling chunks (avoiding redundant duplicates of already-covered documents)
   - Strict budget compliance (token count <= 2500)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from evaluation.evaluation_harness import (
    DIAGNOSTIC_QIDS,
    load_development_benchmark,
)
from rag.context_builder import ContextBuilder
from rag.coverage_context_builder import CoverageAwareContextBuilder
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_phase4_experiments():
    print("=" * 80)
    print("PHASE 4: COVERAGE-AWARE CONTEXT CONSTRUCTION EXPERIMENTS")
    print("=" * 80)

    df_all, df_group_a, df_group_b = load_development_benchmark()

    # Diagnostic set (14 questions)
    print(f"Evaluating across Group B: {len(df_group_b)} diagnostic questions.\n")

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )

    # Builders
    baseline_builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3)
    coverage_builder = CoverageAwareContextBuilder(
        max_context_tokens=2500,
        max_chunks_per_doc=3,
        novelty_weight=0.40,
        redundancy_penalty_weight=0.50,
        authority_boost=0.30,
    )

    builders = [
        ("Baseline_ContextBuilder", baseline_builder),
        ("CoverageAware_ContextBuilder", coverage_builder),
    ]

    results_by_builder: Dict[str, Dict[str, Any]] = {name: {} for name, _ in builders}

    for b_name, builder in builders:
        print(f"--- Evaluating {b_name} ---")
        for idx, row in df_group_b.iterrows():
            qid = str(row["question_id"])
            query = str(row["question"])
            raw_g = row.get("expected_doc_ids")
            gold_docs = [str(x) for x in raw_g] if raw_g is not None and len(raw_g) > 0 else []

            # Retrieve candidate pool (BM25 K=40)
            chunks = retriever.retrieve_chunks(query, mode="bm25", top_k_chunks=40)

            # Build context
            if isinstance(builder, CoverageAwareContextBuilder):
                assembled, audits = builder.build_context_with_audit(chunks, query=query)
            else:
                assembled = builder.build_context(chunks)
                audits = []

            # Inspect included doc IDs
            context_doc_ids = {p.doc_id for p in assembled.passages}

            # Coverage metrics
            gold_in_context = [g for g in gold_docs if g in context_doc_ids]
            num_gold_in_context = len(gold_in_context)
            total_gold = len(gold_docs) if gold_docs else 1
            cov_ratio = num_gold_in_context / total_gold
            is_complete = (num_gold_in_context == len(gold_docs)) if gold_docs else True

            # Sibling chunk deduplication:
            # how many chunks from same doc were included vs how many candidate chunks existed
            num_passages = len(assembled.passages)
            num_distinct_docs = len(context_doc_ids)

            # Check for ClearEdge supersession signal in qst_0427
            superseded_labeled = False
            authoritative_present = False
            if qid == "qst_0427":
                for p in assembled.passages:
                    txt = p.chunk_text.lower()
                    if "supersede" in txt or "final" in txt:
                        authoritative_present = True
                    if "superseded" in txt or "[superseded" in txt:
                        superseded_labeled = True

            results_by_builder[b_name][qid] = {
                "question": query,
                "question_type": row.get("question_type"),
                "total_gold_docs": len(gold_docs),
                "gold_docs": gold_docs,
                "gold_in_context": gold_in_context,
                "missing_gold_from_context": [g for g in gold_docs if g not in context_doc_ids],
                "gold_coverage_ratio": cov_ratio,
                "is_complete_coverage": is_complete,
                "num_passages_in_context": num_passages,
                "num_distinct_docs_in_context": num_distinct_docs,
                "total_tokens_used": assembled.total_tokens,
                "budget_limit": 2500,
                "budget_exceeded": assembled.total_tokens > 2500,
                "audits_count": len(audits),
                "authoritative_signal_present": authoritative_present,
                "superseded_labeled": superseded_labeled,
            }

    # Summarize Group B
    summary = {}
    for b_name, _ in builders:
        recs = list(results_by_builder[b_name].values())
        gold_recs = [r for r in recs if r["total_gold_docs"] > 0]
        mean_cov = sum(r["gold_coverage_ratio"] for r in gold_recs) / len(gold_recs)
        complete_cov = sum(1 for r in gold_recs if r["is_complete_coverage"]) / len(gold_recs)
        mean_docs = sum(r["num_distinct_docs_in_context"] for r in recs) / len(recs)
        mean_passages = sum(r["num_passages_in_context"] for r in recs) / len(recs)
        mean_tokens = sum(r["total_tokens_used"] for r in recs) / len(recs)

        # Multi-doc subset specifically: qst_0436, qst_0437, qst_0447, qst_0448, qst_0427
        multi_qids = ["qst_0436", "qst_0437", "qst_0447", "qst_0448", "qst_0427"]
        multi_recs = [results_by_builder[b_name][qid] for qid in multi_qids if qid in results_by_builder[b_name]]
        multi_cov = sum(r["gold_coverage_ratio"] for r in multi_recs) / len(multi_recs) if multi_recs else 0.0
        multi_complete = sum(1 for r in multi_recs if r["is_complete_coverage"]) / len(multi_recs) if multi_recs else 0.0

        summary[b_name] = {
            "group_b_mean_gold_coverage": mean_cov,
            "group_b_complete_gold_coverage": complete_cov,
            "group_b_mean_distinct_docs_in_context": mean_docs,
            "group_b_mean_passages_in_context": mean_passages,
            "group_b_mean_tokens_used": mean_tokens,
            "multidoc_subset_mean_coverage": multi_cov,
            "multidoc_subset_complete_coverage": multi_complete,
        }

    print("\n" + "=" * 90)
    print(f"{'Builder':<30} | {'Mean Cov':<10} | {'Complete Cov':<14} | {'Multi-Doc Cov':<14} | {'Distinct Docs':<14}")
    print("-" * 90)
    for b_name, sm in summary.items():
        print(
            f"{b_name:<30} | "
            f"{sm['group_b_mean_gold_coverage']*100:>8.2f}% | "
            f"{sm['group_b_complete_gold_coverage']*100:>12.2f}% | "
            f"{sm['multidoc_subset_mean_coverage']*100:>12.2f}% | "
            f"{sm['group_b_mean_distinct_docs_in_context']:>12.2f}"
        )
    print("=" * 90)

    # Detailed inspection of multi-document targets
    print("\n--- Multi-Document Target Question Details ---")
    multi_targets = ["qst_0436", "qst_0437", "qst_0447", "qst_0448", "qst_0427"]
    for qid in multi_targets:
        print(f"\nQuestion: {qid}")
        for b_name, _ in builders:
            r = results_by_builder[b_name].get(qid, {})
            print(f"  {b_name:<30}:")
            print(f"    Gold Docs ({r.get('total_gold_docs')}): {r.get('gold_docs')}")
            print(f"    In Context ({len(r.get('gold_in_context', []))}): {r.get('gold_in_context')}")
            print(f"    Missing ({len(r.get('missing_gold_from_context', []))}): {r.get('missing_gold_from_context')}")
            print(f"    Coverage: {r.get('gold_coverage_ratio', 0)*100:.1f}%, Complete: {r.get('is_complete_coverage')}, Tokens: {r.get('total_tokens_used')}/2500")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "results_by_builder": results_by_builder,
    }

    report_path = REPORTS_DIR / "phase4_context_coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPhase 4 Report saved to {report_path}")


if __name__ == "__main__":
    run_phase4_experiments()
