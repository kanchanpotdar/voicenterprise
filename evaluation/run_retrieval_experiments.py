"""VoiceEnterprise - Retrieval Optimization Experiments (Step 2 & Step 3).

Compares:
1. Baseline: Top-20 parent docs from 80 candidate chunks (no diversity filter).
2. Experiment A: Top-40 parent docs from 160 candidate chunks (no diversity filter).
3. Experiment B: Top-40 parent docs from 160 candidate chunks with parent-document diversity
   (cap max 3 chunks per parent doc, near-duplicate Jaccard >= 0.70 suppression).

Evaluates across all 193 development questions in data/splits/questions_dev.parquet
(178 questions with gold expected_doc_ids).
Tracks Recall@5/10/20/40, Hit@10, MRR@10, per-category R@20, 13-failure coverage,
parent doc count, duplicate chunk rate, and retrieval latency.
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

# The 13 BM25 Recall@20 failure queries audited in Step 1
FAILURES_13_QIDS = [
    "qst_0043",
    "qst_0139",
    "qst_0178",
    "qst_0203",
    "qst_0214",
    "qst_0229",
    "qst_0242",
    "qst_0282",
    "qst_0356",
    "qst_0413",
    "qst_0436",
    "qst_0437",
    "qst_0448",
]


def count_duplicate_chunks(chunks: List[RetrievedChunk], threshold: float = 0.70) -> Tuple[int, float]:
    """Count chunks that are near-duplicates of an earlier chunk with the same doc_id."""
    if not chunks:
        return 0, 0.0
    duplicates = 0
    seen_by_doc: Dict[str, List[str]] = {}
    for ch in chunks:
        doc_texts = seen_by_doc.setdefault(ch.doc_id, [])
        is_dup = False
        for prev_text in doc_texts:
            if _jaccard_similarity(ch.chunk_text, prev_text) >= threshold:
                is_dup = True
                break
        if is_dup:
            duplicates += 1
        else:
            doc_texts.append(ch.chunk_text)
    rate = duplicates / len(chunks)
    return duplicates, rate


def run_experiments():
    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    dev_df = pd.read_parquet(dev_path)
    questions_list = dev_df.to_dict(orient="records")

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )

    print(f"Loaded {len(questions_list)} development questions.")
    print("Executing retrieval over dev set (candidate pool = 160 chunks per query) ...")

    # Storage for predictions and metrics
    preds_baseline: Dict[str, List[str]] = {}
    preds_exp_a: Dict[str, List[str]] = {}
    preds_exp_b: Dict[str, List[str]] = {}

    parent_docs_count_baseline = []
    parent_docs_count_exp_a = []
    parent_docs_count_exp_b = []

    dup_rates_baseline = []
    dup_rates_exp_a = []
    dup_rates_exp_b = []

    latencies_ms = []

    t0_all = time.perf_counter()

    cache_file = BASE_DIR / "evaluation" / "reports" / "retrieval_experiment_raw_cache.json"

    if cache_file.exists():
        print(f"Loading cached raw predictions from {cache_file} ...")
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        preds_baseline = cached["preds_baseline"]
        preds_exp_a = cached["preds_exp_a"]
        preds_exp_b = cached["preds_exp_b"]
        parent_docs_count_baseline = cached["parent_docs_count_baseline"]
        parent_docs_count_exp_a = cached["parent_docs_count_exp_a"]
        parent_docs_count_exp_b = cached["parent_docs_count_exp_b"]
        dup_rates_baseline = cached["dup_rates_baseline"]
        dup_rates_exp_a = cached["dup_rates_exp_a"]
        dup_rates_exp_b = cached["dup_rates_exp_b"]
        latencies_ms = cached["latencies_ms"]
        total_time = cached.get("total_time", 319.10)
    else:
        for idx, q in enumerate(questions_list, start=1):
            qid = str(q["question_id"])
            qtext = str(q["question"])

            # Time retrieval of candidate pool (160 chunks for BM25)
            t_q0 = time.perf_counter()
            raw_chunks = retriever.retrieve_chunks(
                query=qtext,
                mode="bm25",
                top_k_chunks=160,
            )
            t_q1 = time.perf_counter()
            latencies_ms.append((t_q1 - t_q0) * 1000.0)

            # 1. Baseline: First 80 chunks, no diversity filter, up to 20 unique docs
            base_chunks = raw_chunks[:80]
            base_doc_ids = []
            seen_b = set()
            for ch in base_chunks:
                if ch.doc_id not in seen_b:
                    seen_b.add(ch.doc_id)
                    base_doc_ids.append(ch.doc_id)
                    if len(base_doc_ids) >= 20:
                        break
            preds_baseline[qid] = base_doc_ids
            parent_docs_count_baseline.append(len(seen_b))
            _, d_rate_b = count_duplicate_chunks(base_chunks)
            dup_rates_baseline.append(d_rate_b)

            # 2. Experiment A: All 160 chunks, no diversity filter, up to 40 unique docs
            exp_a_doc_ids = []
            seen_a = set()
            for ch in raw_chunks:
                if ch.doc_id not in seen_a:
                    seen_a.add(ch.doc_id)
                    exp_a_doc_ids.append(ch.doc_id)
                    if len(exp_a_doc_ids) >= 40:
                        break
            preds_exp_a[qid] = exp_a_doc_ids
            parent_docs_count_exp_a.append(len(seen_a))
            _, d_rate_a = count_duplicate_chunks(raw_chunks)
            dup_rates_exp_a.append(d_rate_a)

            # 3. Experiment B: 160 chunks with diversity filter (max 3 chunks per doc, Jaccard >= 0.70 suppression)
            filtered_chunks_b: List[RetrievedChunk] = []
            doc_counts_b: Dict[str, int] = {}
            for ch in raw_chunks:
                cnt = doc_counts_b.get(ch.doc_id, 0)
                if cnt >= 3:
                    continue
                is_dup = False
                for ex in [c for c in filtered_chunks_b if c.doc_id == ch.doc_id]:
                    if _jaccard_similarity(ch.chunk_text, ex.chunk_text) >= 0.70:
                        is_dup = True
                        break
                if not is_dup:
                    filtered_chunks_b.append(ch)
                    doc_counts_b[ch.doc_id] = cnt + 1

            exp_b_doc_ids = []
            seen_exp_b = set()
            for ch in filtered_chunks_b:
                if ch.doc_id not in seen_exp_b:
                    seen_exp_b.add(ch.doc_id)
                    exp_b_doc_ids.append(ch.doc_id)
                    if len(exp_b_doc_ids) >= 40:
                        break
            preds_exp_b[qid] = exp_b_doc_ids
            parent_docs_count_exp_b.append(len(seen_exp_b))
            _, d_rate_exp_b = count_duplicate_chunks(filtered_chunks_b)
            dup_rates_exp_b.append(d_rate_exp_b)

            if idx % 25 == 0 or idx == len(questions_list):
                print(f"  Processed {idx}/{len(questions_list)} queries ({time.perf_counter() - t0_all:.1f}s elapsed)...", flush=True)

        total_time = time.perf_counter() - t0_all
        cache_data = {
            "preds_baseline": preds_baseline,
            "preds_exp_a": preds_exp_a,
            "preds_exp_b": preds_exp_b,
            "parent_docs_count_baseline": parent_docs_count_baseline,
            "parent_docs_count_exp_a": parent_docs_count_exp_a,
            "parent_docs_count_exp_b": parent_docs_count_exp_b,
            "dup_rates_baseline": dup_rates_baseline,
            "dup_rates_exp_a": dup_rates_exp_a,
            "dup_rates_exp_b": dup_rates_exp_b,
            "latencies_ms": latencies_ms,
            "total_time": total_time,
        }
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    print(f"All {len(questions_list)} queries processed in {total_time:.2f}s (mean latency: {avg_latency:.2f} ms/query).")

    # Evaluate metrics
    eval_baseline = evaluate_retrieval_predictions(questions_list, preds_baseline, ks=(5, 10, 20, 40))
    eval_exp_a = evaluate_retrieval_predictions(questions_list, preds_exp_a, ks=(5, 10, 20, 40))
    eval_exp_b = evaluate_retrieval_predictions(questions_list, preds_exp_b, ks=(5, 10, 20, 40))

    # Evaluate 13 failures specifically
    failures_q_map = {str(q["question_id"]): q for q in questions_list if str(q["question_id"]) in FAILURES_13_QIDS}
    failures_res = {}
    for fid in FAILURES_13_QIDS:
        q_row = failures_q_map.get(fid)
        if not q_row:
            continue
        raw_exp = q_row.get("expected_doc_ids")
        if raw_exp is not None:
            try:
                exp_docs = [str(x) for x in raw_exp]
            except Exception:
                exp_docs = []
        else:
            exp_docs = []
        r_base = recall_at_k(preds_baseline.get(fid, []), exp_docs, k=20) or 0.0
        r_exp_a_20 = recall_at_k(preds_exp_a.get(fid, []), exp_docs, k=20) or 0.0
        r_exp_a_40 = recall_at_k(preds_exp_a.get(fid, []), exp_docs, k=40) or 0.0
        r_exp_b_20 = recall_at_k(preds_exp_b.get(fid, []), exp_docs, k=20) or 0.0
        r_exp_b_40 = recall_at_k(preds_exp_b.get(fid, []), exp_docs, k=40) or 0.0
        failures_res[fid] = {
            "question_type": q_row.get("question_type"),
            "num_expected": len(exp_docs),
            "recall_baseline_k20": r_base,
            "recall_exp_a_k20": r_exp_a_20,
            "recall_exp_a_k40": r_exp_a_40,
            "recall_exp_b_k20": r_exp_b_20,
            "recall_exp_b_k40": r_exp_b_40,
        }

    # Summary table stats
    summary = {
        "Baseline (K=20)": {
            "Recall@5": eval_baseline["overall"]["recall_at_k"][5],
            "Recall@10": eval_baseline["overall"]["recall_at_k"][10],
            "Recall@20": eval_baseline["overall"]["recall_at_k"][20],
            "Recall@40": eval_baseline["overall"]["recall_at_k"][40],
            "Hit@10": eval_baseline["overall"]["hit_at_k"][10],
            "MRR@10": eval_baseline["overall"]["mrr_at_k"][10],
            "R@20_basic": eval_baseline["by_category"].get("basic", {}).get("recall_at_k", {}).get(20),
            "R@20_semantic": eval_baseline["by_category"].get("semantic", {}).get("recall_at_k", {}).get(20),
            "R@20_project_related": eval_baseline["by_category"].get("project_related", {}).get("recall_at_k", {}).get(20),
            "R@20_conflicting_info": eval_baseline["by_category"].get("conflicting_info", {}).get("recall_at_k", {}).get(20),
            "R@20_completeness": eval_baseline["by_category"].get("completeness", {}).get("recall_at_k", {}).get(20),
            "mean_13_failures_recall": sum(f["recall_baseline_k20"] for f in failures_res.values()) / len(failures_res),
            "avg_parent_docs_in_candidates": sum(parent_docs_count_baseline) / len(parent_docs_count_baseline),
            "avg_duplicate_chunk_rate": sum(dup_rates_baseline) / len(dup_rates_baseline),
            "avg_retrieval_latency_ms": avg_latency,
        },
        "Experiment A (K=40)": {
            "Recall@5": eval_exp_a["overall"]["recall_at_k"][5],
            "Recall@10": eval_exp_a["overall"]["recall_at_k"][10],
            "Recall@20": eval_exp_a["overall"]["recall_at_k"][20],
            "Recall@40": eval_exp_a["overall"]["recall_at_k"][40],
            "Hit@10": eval_exp_a["overall"]["hit_at_k"][10],
            "MRR@10": eval_exp_a["overall"]["mrr_at_k"][10],
            "R@20_basic": eval_exp_a["by_category"].get("basic", {}).get("recall_at_k", {}).get(20),
            "R@20_semantic": eval_exp_a["by_category"].get("semantic", {}).get("recall_at_k", {}).get(20),
            "R@20_project_related": eval_exp_a["by_category"].get("project_related", {}).get("recall_at_k", {}).get(20),
            "R@20_conflicting_info": eval_exp_a["by_category"].get("conflicting_info", {}).get("recall_at_k", {}).get(20),
            "R@20_completeness": eval_exp_a["by_category"].get("completeness", {}).get("recall_at_k", {}).get(20),
            "mean_13_failures_recall": sum(f["recall_exp_a_k40"] for f in failures_res.values()) / len(failures_res),
            "avg_parent_docs_in_candidates": sum(parent_docs_count_exp_a) / len(parent_docs_count_exp_a),
            "avg_duplicate_chunk_rate": sum(dup_rates_exp_a) / len(dup_rates_exp_a),
            "avg_retrieval_latency_ms": avg_latency,
        },
        "Experiment B (K=40 + Diversity)": {
            "Recall@5": eval_exp_b["overall"]["recall_at_k"][5],
            "Recall@10": eval_exp_b["overall"]["recall_at_k"][10],
            "Recall@20": eval_exp_b["overall"]["recall_at_k"][20],
            "Recall@40": eval_exp_b["overall"]["recall_at_k"][40],
            "Hit@10": eval_exp_b["overall"]["hit_at_k"][10],
            "MRR@10": eval_exp_b["overall"]["mrr_at_k"][10],
            "R@20_basic": eval_exp_b["by_category"].get("basic", {}).get("recall_at_k", {}).get(20),
            "R@20_semantic": eval_exp_b["by_category"].get("semantic", {}).get("recall_at_k", {}).get(20),
            "R@20_project_related": eval_exp_b["by_category"].get("project_related", {}).get("recall_at_k", {}).get(20),
            "R@20_conflicting_info": eval_exp_b["by_category"].get("conflicting_info", {}).get("recall_at_k", {}).get(20),
            "R@20_completeness": eval_exp_b["by_category"].get("completeness", {}).get("recall_at_k", {}).get(20),
            "mean_13_failures_recall": sum(f["recall_exp_b_k40"] for f in failures_res.values()) / len(failures_res),
            "avg_parent_docs_in_candidates": sum(parent_docs_count_exp_b) / len(parent_docs_count_exp_b),
            "avg_duplicate_chunk_rate": sum(dup_rates_exp_b) / len(dup_rates_exp_b),
            "avg_retrieval_latency_ms": avg_latency,
        },
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": len(questions_list),
        "total_questions_with_gold": eval_baseline["overall"]["num_questions_with_gold_docs"],
        "summary_table": summary,
        "eval_baseline": eval_baseline,
        "eval_exp_a": eval_exp_a,
        "eval_exp_b": eval_exp_b,
        "failures_13_comparison": failures_res,
    }

    out_file = BASE_DIR / "evaluation" / "reports" / "retrieval_experiments_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved full report to: {out_file}")

    print("\n" + "=" * 90)
    print("RETRIEVAL EXPERIMENT SUMMARY COMPARISON")
    print("=" * 90)
    header = f"{'Metric':<32} | {'Baseline (K=20)':<18} | {'Exp A (K=40)':<18} | {'Exp B (K=40+Div)':<18}"
    print(header)
    print("-" * len(header))
    metric_map = [
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
        ("13-Failure Mean Recall", "mean_13_failures_recall", "{:.2%}"),
        ("Avg Parent Docs in Pool", "avg_parent_docs_in_candidates", "{:.1f}"),
        ("Duplicate Chunk Rate", "avg_duplicate_chunk_rate", "{:.2%}"),
        ("Retrieval Latency", "avg_retrieval_latency_ms", "{:.1f} ms"),
    ]
    for label, key, fmt in metric_map:
        v_base = summary["Baseline (K=20)"].get(key)
        v_a = summary["Experiment A (K=40)"].get(key)
        v_b = summary["Experiment B (K=40 + Diversity)"].get(key)

        s_base = fmt.format(v_base) if v_base is not None else "N/A"
        s_a = fmt.format(v_a) if v_a is not None else "N/A"
        s_b = fmt.format(v_b) if v_b is not None else "N/A"
        print(f"{label:<32} | {s_base:<18} | {s_a:<18} | {s_b:<18}")
    print("=" * 90)


if __name__ == "__main__":
    run_experiments()
