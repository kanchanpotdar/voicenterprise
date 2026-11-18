"""Compare raw and exact-evidence deduplication on development questions only."""
import json
from pathlib import Path

import pandas as pd
from retrieval.indexer import EnterpriseRetriever
from evaluation.metrics import recall_at_k, hit_at_k, reciprocal_rank_at_k


def run():
    retriever = EnterpriseRetriever(collection_name="enterprise_rag_full", qdrant_url="http://127.0.0.1:6333")
    questions = pd.read_parquet("data/splits/questions_dev.parquet")
    records = []
    for row in questions.to_dict("records"):
        rankings, chunks = {}, {}
        for mode in ("raw", "canonical"):
            rankings[mode], chunks[mode] = retriever.retrieve_documents(
                row["question"], mode="bm25", top_k_docs=20,
                chunk_oversample_factor=8, canonical_filter=(mode == "canonical"))
        gold = list(row["expected_doc_ids"])
        record = {"question_id": row["question_id"], "gold_count": len(gold)}
        record["diagnostics"] = {"raw_chunks": len(chunks["raw"]),
            "canonical_chunks": len(chunks["canonical"]),
            "removed_chunks": len(chunks["raw"]) - len(chunks["canonical"]),
            "ranking_changed": rankings["raw"] != rankings["canonical"],
            "alias_groups": sum(len(c.alias_doc_ids) > 1 for c in chunks["canonical"])}
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
    diagnostics = {"removed_chunks": sum(r["diagnostics"]["removed_chunks"] for r in records),
                   "changed_rankings": sum(r["diagnostics"]["ranking_changed"] for r in records),
                   "alias_groups": sum(r["diagnostics"]["alias_groups"] for r in records)}
    result = {"protocol": "production EnterpriseRetriever.retrieve_documents; development split; 160 chunks; original IDs; query-time exact evidence filtering; no persistent payload enrichment", "diagnostics": diagnostics, "summary": summary, "per_question": records}
    Path("evaluation/reports/canonical_dev_v2.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    retriever.client.close()
    print(json.dumps(diagnostics, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
