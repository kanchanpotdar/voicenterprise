"""Fast resumable BM25-only full-corpus indexer for generation evaluation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from retrieval.indexer import deterministic_point_uuid

def run(input_dir: Path, qdrant_dir: Path, batch_size: int = 4096) -> None:
    state_path = input_dir / "bm25_index_manifest.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"completed_batches": {}, "errors": {}}
    client = QdrantClient(path=str(qdrant_dir))
    collection = "enterprise_rag_full"
    embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
    batches = sorted((input_dir / "batches").glob("*_chunks.parquet"))
    if not client.collection_exists(collection):
        client.create_collection(collection_name=collection, sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    for path in batches:
        key = path.stem.split("_")[1]
        if key in state["completed_batches"]:
            continue
        df = pd.read_parquet(path)
        records = df.to_dict("records")
        try:
            for start in range(0, len(records), batch_size):
                rows = records[start:start + batch_size]
                vecs = list(embedder.embed([str(r["contextualized_text"]) for r in rows], batch_size=batch_size))
                points = []
                for row, vec in zip(rows, vecs):
                    points.append(models.PointStruct(id=deterministic_point_uuid(str(row["chunk_id"])), vector={"bm25": models.SparseVector(indices=vec.indices.tolist(), values=vec.values.tolist())}, payload={"chunk_id": str(row["chunk_id"]), "record_id": str(row["record_id"]), "doc_id": str(row["doc_id"]), "chunk_index": int(row["chunk_index"]), "total_chunks_in_doc": int(row["total_chunks_in_doc"]), "source_type": str(row["source_type"]), "normalized_title": str(row["normalized_title"]), "section_title": str(row["section_title"]), "chunk_text": str(row["chunk_text"]), "latest_date": None if pd.isna(row.get("latest_date")) else str(row.get("latest_date")), "date_confidence": str(row.get("date_confidence") or "none"), "slack_channels": [] if row.get("slack_channels") is None else [str(x) for x in row["slack_channels"]], "ticket_ids": [] if row.get("ticket_ids") is None else [str(x) for x in row["ticket_ids"]], "has_duplicate_doc_id_conflict": bool(row.get("has_duplicate_doc_id_conflict", False))}))
                client.upsert(collection_name=collection, points=points)
            state["completed_batches"][key] = {"chunks": len(df)}
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            print(f"indexed BM25 batch {key}: {len(df)} chunks", flush=True)
        except Exception as exc:
            state["errors"][key] = repr(exc)
            state_path.write_text(json.dumps(state, indent=2) + "\n")
            raise
    state["status"] = "complete" if len(state["completed_batches"]) == len(batches) else "partial"
    state_path.write_text(json.dumps(state, indent=2) + "\n")

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,default=Path("data/ingested/full_corpus")); p.add_argument("--qdrant",type=Path,default=Path("data/qdrant_full_bm25")); p.add_argument("--batch-size",type=int,default=4096); a=p.parse_args(); run(a.input,a.qdrant,a.batch_size)
