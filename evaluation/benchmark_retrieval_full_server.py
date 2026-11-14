from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


def run() -> dict:
    base = Path(__file__).resolve().parent.parent
    questions = pd.read_parquet(base / "EnterpriseRAG-Bench/data/questions/test.parquet")
    client = QdrantClient(url="http://127.0.0.1:6333", grpc_port=6334, prefer_grpc=True, timeout=120)
    embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    rows = []
    started = time.perf_counter()
    for idx, row in enumerate(questions.to_dict("records"), 1):
        query_vec = next(embedder.query_embed(str(row["question"])))
        hits = client.query_points(
            collection_name="enterprise_rag_full",
            query=models.SparseVector(indices=query_vec.indices.tolist(), values=query_vec.values.tolist()),
            using="bm25", limit=20, with_payload=True,
        ).points
        docs = []
        for hit in hits:
            doc_id = str(hit.payload.get("doc_id"))
            if doc_id not in docs:
                docs.append(doc_id)
        raw_gold = row.get("expected_doc_ids")
        gold = {str(x) for x in ([] if raw_gold is None else list(raw_gold))}
        eligible = bool(gold)
        first_rank = next((i + 1 for i, d in enumerate(docs) if d in gold), None)
        rows.append({"question_id": str(row["question_id"]), "category": str(row.get("question_type", row.get("category", "unknown"))), "gold_count": len(gold), "eligible": eligible, "r1": int(bool(first_rank and first_rank <= 1)), "r5": int(bool(first_rank and first_rank <= 5)), "r10": int(bool(first_rank and first_rank <= 10)), "r20": int(bool(first_rank and first_rank <= 20)), "mrr10": (1.0 / first_rank if first_rank and first_rank <= 10 else 0.0)})
        if idx % 50 == 0:
            print(f"completed {idx}/{len(questions)}", flush=True)
    df = pd.DataFrame(rows)
    eligible_df = df[df["eligible"]]
    summary = {"questions": len(df), "eligible_questions": len(eligible_df), "elapsed_seconds": round(time.perf_counter() - started, 2)}
    for col in ("r1", "r5", "r10", "r20", "mrr10"):
        summary[col] = round(float(eligible_df[col].mean()), 6)
    summary["by_category"] = {str(cat): {"n": int(len(g)), "r5": round(float(g.r5.mean()), 6), "r10": round(float(g.r10.mean()), 6)} for cat, g in df.groupby("category")}
    out = base / "evaluation/reports/full_corpus_bm25_test_retrieval.json"
    out.write_text(json.dumps({"summary": summary, "per_question": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run()
