"""P1.4 & P1.5 Reproducible Indexing and Retrieval Benchmark on corpus_dev_10k.

Indexes all 49,051 chunks from `data/ingested/dev_10k/chunks.parquet` into a persistent
Qdrant collection (`data/qdrant_dev_10k`) with both:
- Dense embeddings (`BAAI/bge-small-en-v1.5`, 384-dim Cosine)
- Sparse BM25 representations (`Qdrant/bm25` with `models.Modifier.IDF`)

Then evaluates `bm25`, `dense`, and `hybrid` (RRF) retrieval on the 193 development
questions (`data/splits/questions_dev.parquet`), saving a complete overall + per-category
comparison report to `evaluation/reports/p1_retrieval_benchmark_dev.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation.metrics import evaluate_retrieval_predictions
from retrieval.indexer import EnterpriseRetriever, RetrievalMode


def run_p1_benchmark(
    chunks_parquet_path: Path,
    dev_questions_parquet_path: Path,
    qdrant_dir: Path,
    index_manifest_path: Path,
    report_output_path: Path,
    rebuild_index: bool = False,
    top_k_docs: int = 20,
) -> Dict[str, Any]:
    chunks_df = pd.read_parquet(chunks_parquet_path)
    questions_df = pd.read_parquet(dev_questions_parquet_path)
    questions_list = questions_df.to_dict(orient="records")

    collection_name = "enterprise_rag_dev_10k"
    retriever = EnterpriseRetriever(
        collection_name=collection_name,
        qdrant_path=qdrant_dir,
    )

    need_index = rebuild_index or (not retriever.client.collection_exists(collection_name))
    if not need_index:
        info = retriever.client.get_collection(collection_name)
        if info.points_count != len(chunks_df):
            need_index = True

    if need_index:
        print(f"Indexing {len(chunks_df)} chunks into {qdrant_dir} ...", flush=True)
        index_stats = retriever.index_chunks_df(
            chunks_df, batch_size=256, reset_collection=True
        )
        index_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        index_manifest_path.write_text(
            json.dumps(index_stats, indent=2) + "\n", encoding="utf-8"
        )
    else:
        info = retriever.client.get_collection(collection_name)
        index_stats = {
            "collection_name": collection_name,
            "qdrant_points_count": info.points_count,
            "chunks_indexed": len(chunks_df),
            "reused_existing_index": True,
        }

    modes: List[RetrievalMode] = ["bm25", "dense", "hybrid"]
    mode_results: Dict[str, Any] = {}

    for mode in modes:
        t0 = time.perf_counter()
        preds_by_qid: Dict[str, List[str]] = {}
        for q in questions_list:
            qid = str(q["question_id"])
            qtext = str(q["question"])
            doc_ids, _ = retriever.retrieve_documents(
                query=qtext,
                mode=mode,
                top_k_docs=top_k_docs,
                chunk_oversample_factor=4,
            )
            preds_by_qid[qid] = doc_ids

        elapsed = time.perf_counter() - t0
        eval_res = evaluate_retrieval_predictions(
            questions=questions_list,
            predictions_by_qid=preds_by_qid,
            ks=(1, 5, 10, 20),
            low_sample_threshold=15,
        )
        mode_results[mode] = {
            "query_time_seconds": round(elapsed, 2),
            "avg_ms_per_query": round((elapsed * 1000.0) / max(1, len(questions_list)), 2),
            "overall": eval_res["overall"],
            "by_category": eval_res["by_category"],
        }

    report = {
        "corpus": "corpus_dev_10k.parquet (strictly unconditioned on held-out test)",
        "question_split": "questions_dev.parquet (n=193, 0 shared gold docs with held-out)",
        "index_stats": index_stats,
        "modes": mode_results,
    }

    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild Qdrant index")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    report_data = run_p1_benchmark(
        chunks_parquet_path=base / "data" / "ingested" / "dev_10k" / "chunks.parquet",
        dev_questions_parquet_path=base / "data" / "splits" / "questions_dev.parquet",
        qdrant_dir=base / "data" / "qdrant_dev_10k",
        index_manifest_path=base / "data" / "ingested" / "dev_10k" / "index_manifest.json",
        report_output_path=base / "evaluation" / "reports" / "p1_retrieval_benchmark_dev.json",
        rebuild_index=args.rebuild,
    )
    for m_name, m_data in report_data["modes"].items():
        ov = m_data["overall"]
        print(
            f"[{m_name.upper():6s}] "
            f"Recall@5={ov['recall_at_k'][5]:.4f} | "
            f"Recall@10={ov['recall_at_k'][10]:.4f} | "
            f"Recall@20={ov['recall_at_k'][20]:.4f} | "
            f"Hit@10={ov['hit_at_k'][10]:.4f} | "
            f"MRR@10={ov['mrr_at_k'][10]:.4f} | "
            f"AvgQueryMs={m_data['avg_ms_per_query']}ms"
        )
