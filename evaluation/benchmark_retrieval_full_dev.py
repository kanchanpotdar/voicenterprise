"""Comprehensive Full Development Retrieval Evaluation (BM25 vs Hybrid, K=20 vs K=40).

Compares:
1. BM25 Baseline (K=20)
2. BM25 Deeper Pool (K=40 + Diversity)
3. Hybrid Baseline (K=20)
4. Hybrid Deeper Pool (K=40 + Diversity)

Evaluates on all 193 development questions (178 questions with gold expected_doc_ids).
Reports Recall@5, @10, @20, @40, Hit@10, MRR@10, per-category R@20, 13-failure gold coverage,
and retrieval latencies.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="Local mode is not recommended for collections with more than 20,000 points.*",
    category=UserWarning,
)

from evaluation.metrics import evaluate_retrieval_predictions, recall_at_k
from retrieval.indexer import EnterpriseRetriever, RetrievedChunk, _jaccard_similarity

BASE_DIR = Path(__file__).resolve().parent.parent

FAILURES_13_QIDS = [
    "qst_0043", "qst_0139", "qst_0178", "qst_0203", "qst_0214",
    "qst_0229", "qst_0242", "qst_0282", "qst_0356", "qst_0413",
    "qst_0436", "qst_0437", "qst_0448",
]


def run_full_dev_retrieval():
    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    dev_df = pd.read_parquet(dev_path)
    questions_list = dev_df.to_dict(orient="records")

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )

    # 1. Load BM25 predictions from cache
    bm25_cache_file = BASE_DIR / "evaluation" / "reports" / "retrieval_experiment_raw_cache.json"
    if not bm25_cache_file.exists():
        raise RuntimeError(f"BM25 cache not found at {bm25_cache_file}")

    bm25_cached = json.loads(bm25_cache_file.read_text(encoding="utf-8"))
    preds_bm25_20 = bm25_cached["preds_baseline"]
    preds_bm25_40 = bm25_cached["preds_exp_b"]
    bm25_latencies = bm25_cached["latencies_ms"]
    avg_bm25_latency = sum(bm25_latencies) / len(bm25_latencies)

    # 2. Check or run Hybrid predictions
    hybrid_cache_file = BASE_DIR / "evaluation" / "reports" / "hybrid_retrieval_cache.json"
    if hybrid_cache_file.exists():
        print(f"Loading cached Hybrid predictions from {hybrid_cache_file} ...")
        hybrid_cached = json.loads(hybrid_cache_file.read_text(encoding="utf-8"))
        preds_hybrid_20 = hybrid_cached["preds_hybrid_20"]
        preds_hybrid_40 = hybrid_cached["preds_hybrid_40"]
        hybrid_latencies = hybrid_cached["latencies_ms"]
        avg_hybrid_latency = sum(hybrid_latencies) / len(hybrid_latencies)
    else:
        print(f"Computing Hybrid retrieval over all {len(questions_list)} development queries ...")
        preds_hybrid_20 = {}
        preds_hybrid_40 = {}
        hybrid_latencies = []
        t0_hy = time.perf_counter()

        for idx, q in enumerate(questions_list, start=1):
            qid = str(q["question_id"])
            qtext = str(q["question"])

            t_q0 = time.perf_counter()
            # Retrieve 160 chunks via Hybrid (RRF)
            raw_chunks = retriever.retrieve_chunks(
                query=qtext,
                mode="hybrid",
                top_k_chunks=160,
            )
            t_q1 = time.perf_counter()
            hybrid_latencies.append((t_q1 - t_q0) * 1000.0)

            # Hybrid K=20 (first 80 chunks, up to 20 unique docs)
            seen_20 = set()
            docs_20 = []
            for ch in raw_chunks[:80]:
                if ch.doc_id not in seen_20:
                    seen_20.add(ch.doc_id)
                    docs_20.append(ch.doc_id)
                    if len(docs_20) >= 20:
                        break
            preds_hybrid_20[qid] = docs_20

            # Hybrid K=40 with diversity filter
            filtered_chunks: List[RetrievedChunk] = []
            doc_counts: Dict[str, int] = {}
            for ch in raw_chunks:
                cnt = doc_counts.get(ch.doc_id, 0)
                if cnt >= 3:
                    continue
                is_dup = False
                for ex in [c for c in filtered_chunks if c.doc_id == ch.doc_id]:
                    if _jaccard_similarity(ch.chunk_text, ex.chunk_text) >= 0.70:
                        is_dup = True
                        break
                if not is_dup:
                    filtered_chunks.append(ch)
                    doc_counts[ch.doc_id] = cnt + 1

            seen_40 = set()
            docs_40 = []
            for ch in filtered_chunks:
                if ch.doc_id not in seen_40:
                    seen_40.add(ch.doc_id)
                    docs_40.append(ch.doc_id)
                    if len(docs_40) >= 40:
                        break
            preds_hybrid_40[qid] = docs_40

            if idx % 25 == 0 or idx == len(questions_list):
                print(f"  Processed {idx}/{len(questions_list)} hybrid queries ({time.perf_counter() - t0_hy:.1f}s elapsed)...", flush=True)

        avg_hybrid_latency = sum(hybrid_latencies) / len(hybrid_latencies)
        cache_data = {
            "preds_hybrid_20": preds_hybrid_20,
            "preds_hybrid_40": preds_hybrid_40,
            "latencies_ms": hybrid_latencies,
            "total_time": time.perf_counter() - t0_hy,
        }
        hybrid_cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
        print(f"Cached Hybrid predictions to {hybrid_cache_file}")

    # 3. Evaluate metrics
    eval_bm25_20 = evaluate_retrieval_predictions(questions_list, preds_bm25_20, ks=(5, 10, 20, 40))
    eval_bm25_40 = evaluate_retrieval_predictions(questions_list, preds_bm25_40, ks=(5, 10, 20, 40))
    eval_hybrid_20 = evaluate_retrieval_predictions(questions_list, preds_hybrid_20, ks=(5, 10, 20, 40))
    eval_hybrid_40 = evaluate_retrieval_predictions(questions_list, preds_hybrid_40, ks=(5, 10, 20, 40))

    # 4. Evaluate 13 failures specifically
    failures_q_map = {str(q["question_id"]): q for q in questions_list if str(q["question_id"]) in FAILURES_13_QIDS}
    failures_comparison = {}
    for fid in FAILURES_13_QIDS:
        q_row = failures_q_map.get(fid)
        if not q_row:
            continue
        raw_exp = q_row.get("expected_doc_ids")
        exp_docs = [str(x) for x in raw_exp] if raw_exp is not None and len(raw_exp) > 0 else []
        r_bm25_20 = recall_at_k(preds_bm25_20.get(fid, []), exp_docs, k=20) or 0.0
        r_bm25_40 = recall_at_k(preds_bm25_40.get(fid, []), exp_docs, k=40) or 0.0
        r_hy_20 = recall_at_k(preds_hybrid_20.get(fid, []), exp_docs, k=20) or 0.0
        r_hy_40 = recall_at_k(preds_hybrid_40.get(fid, []), exp_docs, k=40) or 0.0
        failures_comparison[fid] = {
            "question_type": q_row.get("question_type"),
            "num_expected": len(exp_docs),
            "recall_bm25_k20": r_bm25_20,
            "recall_bm25_k40": r_bm25_40,
            "recall_hybrid_k20": r_hy_20,
            "recall_hybrid_k40": r_hy_40,
        }

    # Summary table
    def _get_cat(ev, cat, k=20):
        return ev["by_category"].get(cat, {}).get("recall_at_k", {}).get(k)

    summary = {
        "BM25 (K=20)": {
            "Recall@5": eval_bm25_20["overall"]["recall_at_k"][5],
            "Recall@10": eval_bm25_20["overall"]["recall_at_k"][10],
            "Recall@20": eval_bm25_20["overall"]["recall_at_k"][20],
            "Recall@40": eval_bm25_20["overall"]["recall_at_k"][40],
            "Hit@10": eval_bm25_20["overall"]["hit_at_k"][10],
            "MRR@10": eval_bm25_20["overall"]["mrr_at_k"][10],
            "R@20_basic": _get_cat(eval_bm25_20, "basic"),
            "R@20_semantic": _get_cat(eval_bm25_20, "semantic"),
            "R@20_project_related": _get_cat(eval_bm25_20, "project_related"),
            "R@20_conflicting_info": _get_cat(eval_bm25_20, "conflicting_info"),
            "R@20_completeness": _get_cat(eval_bm25_20, "completeness"),
            "13_failures_mean_recall": sum(f["recall_bm25_k20"] for f in failures_comparison.values()) / len(failures_comparison),
            "latency_ms": avg_bm25_latency,
        },
        "BM25 (K=40 + Div)": {
            "Recall@5": eval_bm25_40["overall"]["recall_at_k"][5],
            "Recall@10": eval_bm25_40["overall"]["recall_at_k"][10],
            "Recall@20": eval_bm25_40["overall"]["recall_at_k"][20],
            "Recall@40": eval_bm25_40["overall"]["recall_at_k"][40],
            "Hit@10": eval_bm25_40["overall"]["hit_at_k"][10],
            "MRR@10": eval_bm25_40["overall"]["mrr_at_k"][10],
            "R@20_basic": _get_cat(eval_bm25_40, "basic"),
            "R@20_semantic": _get_cat(eval_bm25_40, "semantic"),
            "R@20_project_related": _get_cat(eval_bm25_40, "project_related"),
            "R@20_conflicting_info": _get_cat(eval_bm25_40, "conflicting_info"),
            "R@20_completeness": _get_cat(eval_bm25_40, "completeness"),
            "13_failures_mean_recall": sum(f["recall_bm25_k40"] for f in failures_comparison.values()) / len(failures_comparison),
            "latency_ms": avg_bm25_latency,
        },
        "Hybrid (K=20)": {
            "Recall@5": eval_hybrid_20["overall"]["recall_at_k"][5],
            "Recall@10": eval_hybrid_20["overall"]["recall_at_k"][10],
            "Recall@20": eval_hybrid_20["overall"]["recall_at_k"][20],
            "Recall@40": eval_hybrid_20["overall"]["recall_at_k"][40],
            "Hit@10": eval_hybrid_20["overall"]["hit_at_k"][10],
            "MRR@10": eval_hybrid_20["overall"]["mrr_at_k"][10],
            "R@20_basic": _get_cat(eval_hybrid_20, "basic"),
            "R@20_semantic": _get_cat(eval_hybrid_20, "semantic"),
            "R@20_project_related": _get_cat(eval_hybrid_20, "project_related"),
            "R@20_conflicting_info": _get_cat(eval_hybrid_20, "conflicting_info"),
            "R@20_completeness": _get_cat(eval_hybrid_20, "completeness"),
            "13_failures_mean_recall": sum(f["recall_hybrid_k20"] for f in failures_comparison.values()) / len(failures_comparison),
            "latency_ms": avg_hybrid_latency,
        },
        "Hybrid (K=40 + Div)": {
            "Recall@5": eval_hybrid_40["overall"]["recall_at_k"][5],
            "Recall@10": eval_hybrid_40["overall"]["recall_at_k"][10],
            "Recall@20": eval_hybrid_40["overall"]["recall_at_k"][20],
            "Recall@40": eval_hybrid_40["overall"]["recall_at_k"][40],
            "Hit@10": eval_hybrid_40["overall"]["hit_at_k"][10],
            "MRR@10": eval_hybrid_40["overall"]["mrr_at_k"][10],
            "R@20_basic": _get_cat(eval_hybrid_40, "basic"),
            "R@20_semantic": _get_cat(eval_hybrid_40, "semantic"),
            "R@20_project_related": _get_cat(eval_hybrid_40, "project_related"),
            "R@20_conflicting_info": _get_cat(eval_hybrid_40, "conflicting_info"),
            "R@20_completeness": _get_cat(eval_hybrid_40, "completeness"),
            "13_failures_mean_recall": sum(f["recall_hybrid_k40"] for f in failures_comparison.values()) / len(failures_comparison),
            "latency_ms": avg_hybrid_latency,
        },
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": len(questions_list),
        "total_questions_with_gold": eval_bm25_20["overall"]["num_questions_with_gold_docs"],
        "summary_table": summary,
        "failures_13_breakdown": failures_comparison,
        "eval_bm25_20": eval_bm25_20["overall"],
        "eval_bm25_40": eval_bm25_40["overall"],
        "eval_hybrid_20": eval_hybrid_20["overall"],
        "eval_hybrid_40": eval_hybrid_40["overall"],
    }

    out_file = BASE_DIR / "evaluation" / "reports" / "full_dev_retrieval_comparison.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full report to: {out_file}")

    print("\n" + "=" * 105)
    print("FULL DEVELOPMENT SET RETRIEVAL BENCHMARK: BM25 vs. HYBRID (K=20 vs. K=40)")
    print("=" * 105)
    header = f"{'Metric':<28} | {'BM25 (K=20)':<16} | {'BM25 (K=40+Div)':<16} | {'Hybrid (K=20)':<16} | {'Hybrid (K=40+Div)':<16}"
    print(header)
    print("-" * len(header))
    metrics_display = [
        ("Recall@5", "Recall@5", "{:.2%}"),
        ("Recall@10", "Recall@10", "{:.2%}"),
        ("Recall@20", "Recall@20", "{:.2%}"),
        ("Recall@40", "Recall@40", "{:.2%}"),
        ("Hit@10", "Hit@10", "{:.2%}"),
        ("MRR@10", "MRR@10", "{:.4f}"),
        ("R@20 (basic)", "R@20_basic", "{:.2%}"),
        ("R@20 (semantic)", "R@20_semantic", "{:.2%}"),
        ("R@20 (project_related)", "R@20_project_related", "{:.2%}"),
        ("R@20 (conflicting_info)", "R@20_conflicting_info", "{:.2%}"),
        ("R@20 (completeness)", "R@20_completeness", "{:.2%}"),
        ("13-Failure Mean Recall", "13_failures_mean_recall", "{:.2%}"),
        ("Retrieval Latency", "latency_ms", "{:.1f} ms"),
    ]
    for label, key, fmt in metrics_display:
        v_b20 = summary["BM25 (K=20)"][key]
        v_b40 = summary["BM25 (K=40 + Div)"][key]
        v_h20 = summary["Hybrid (K=20)"][key]
        v_h40 = summary["Hybrid (K=40 + Div)"][key]
        s_b20 = fmt.format(v_b20) if v_b20 is not None else "N/A"
        s_b40 = fmt.format(v_b40) if v_b40 is not None else "N/A"
        s_h20 = fmt.format(v_h20) if v_h20 is not None else "N/A"
        s_h40 = fmt.format(v_h40) if v_h40 is not None else "N/A"
        print(f"{label:<28} | {s_b20:<16} | {s_b40:<16} | {s_h20:<16} | {s_h40:<16}")
    print("=" * 105)


if __name__ == "__main__":
    run_full_dev_retrieval()
