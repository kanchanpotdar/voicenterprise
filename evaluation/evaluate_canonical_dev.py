"""Compare raw and exact-evidence deduplication on development questions only."""
import json
from pathlib import Path

import pandas as pd
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from retrieval.canonical import evidence_hash
from evaluation.metrics import recall_at_k, hit_at_k, reciprocal_rank_at_k


def run():
    client = QdrantClient(url="http://127.0.0.1:6333", prefer_grpc=True, timeout=120)
    embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    questions = pd.read_parquet("data/splits/questions_dev.parquet")
    records = []
    for row in questions.to_dict("records"):
        vector = next(embedder.query_embed(row["question"]))
        hits = client.query_points("enterprise_rag_full", query=models.SparseVector(
            indices=vector.indices.tolist(), values=vector.values.tolist()), using="bm25",
            limit=160, with_payload=True).points
        rankings = {"raw": [], "canonical": []}
        seen_hashes = set()
        for hit in hits:
            payload = hit.payload
            doc = payload["doc_id"]
            if doc not in rankings["raw"]:
                rankings["raw"].append(doc)
            text = payload.get("chunk_text", "")
            digest = evidence_hash(payload.get("normalized_title", ""), payload.get("section_title", ""), text)
            if text.strip() and digest in seen_hashes:
                continue
            if text.strip():
                seen_hashes.add(digest)
            if doc not in rankings["canonical"]:
                rankings["canonical"].append(doc)
        gold = list(row["expected_doc_ids"])
        record = {"question_id": row["question_id"], "gold_count": len(gold)}
        for mode, docs in rankings.items():
            record[mode] = {"recall@5": recall_at_k(docs, gold, 5),
                            "recall@20": recall_at_k(docs, gold, 20),
                            "hit@5": hit_at_k(docs, gold, 5),
                            "mrr@10": reciprocal_rank_at_k(docs, gold, 10)}
        records.append(record)
    summary = {}
    for mode in ("raw", "canonical"):
        summary[mode] = {}
        for metric in records[0][mode]:
            values = [r[mode][metric] for r in records if r[mode][metric] is not None]
            summary[mode][metric] = sum(values) / len(values) if values else None
    result = {"protocol": "193 development questions; 160 chunks; strict original document IDs; aliases not credited; exact chunk evidence only", "summary": summary, "per_question": records}
    Path("evaluation/reports/canonical_dev_v1.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
