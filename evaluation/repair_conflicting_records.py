"""Plan or apply a bounded repair of cross-batch conflicting document IDs.

Default is read-only with respect to Qdrant. Original source/batch files remain
available for provenance; repaired chunks are written in a separate directory.
"""
import argparse
import json
import math
from pathlib import Path
import pandas as pd
from ingestion.normalizer import normalize_documents_df
from ingestion.metadata import enrich_dataframe_with_metadata
from ingestion.chunker import chunk_documents_df


def clean_json(value):
    if isinstance(value, dict): return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list): return [clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def run(apply=False):
    root = Path(__file__).resolve().parents[1]
    source = root / "EnterpriseRAG-Bench/data/documents/test.parquet"
    ids = pd.read_parquet(source, columns=["doc_id"])["doc_id"]
    conflicts = ids[ids.duplicated(keep=False)].unique().tolist()
    raw = pd.read_parquet(source, filters=[("doc_id", "in", conflicts)])
    normalized = enrich_dataframe_with_metadata(normalize_documents_df(raw))
    repaired = chunk_documents_df(normalized, max_chunk_chars=2000, overlap_chars=200)
    legacy = []
    for path in sorted((root / "data/ingested/full_corpus/batches").glob("*_chunks.parquet")):
        frame = pd.read_parquet(path, columns=["doc_id", "chunk_id"], filters=[("doc_id", "in", conflicts)])
        legacy.extend(frame.chunk_id.tolist())
    if repaired.chunk_id.duplicated().any():
        raise RuntimeError("Repair generated duplicate chunk IDs")
    out = root / "data/canonical/conflicting_record_repair"
    out.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(out / "chunks.parquet", index=False)
    report = {"conflicting_document_ids": conflicts, "source_rows": len(raw),
              "legacy_chunk_rows": len(legacy), "legacy_unique_chunk_ids": len(set(legacy)),
              "repaired_chunks": len(repaired), "applied": False}
    if apply:
        from qdrant_client import QdrantClient, models
        from fastembed import SparseTextEmbedding
        from retrieval.indexer import deterministic_point_uuid
        from retrieval.canonical import canonical_metadata
        client = QdrantClient(url="http://127.0.0.1:6333", prefer_grpc=True, timeout=300)
        collection = "enterprise_rag_full"
        stale_ids = [deterministic_point_uuid(cid) for cid in set(legacy)]
        before = client.count(collection, exact=True).count
        existing_old = client.retrieve(collection, stale_ids, with_payload=False)
        fresh_ids = [deterministic_point_uuid(cid) for cid in repaired.chunk_id]
        existing_new = client.retrieve(collection, fresh_ids, with_payload=False)
        embedder = SparseTextEmbedding(model_name="Qdrant/bm25")
        for row in repaired.to_dict("records"):
            vector = next(embedder.embed([row["contextualized_text"]]))
            payload = clean_json(json.loads(json.dumps(row, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))))
            payload["canonical_v1"] = canonical_metadata(payload)
            point = models.PointStruct(id=deterministic_point_uuid(row["chunk_id"]),
                vector={"bm25": models.SparseVector(indices=vector.indices.tolist(), values=vector.values.tolist())}, payload=payload)
            client.upsert(collection, [point], wait=True)
        if len(client.retrieve(collection, fresh_ids, with_payload=False)) != len(fresh_ids):
            raise RuntimeError("New points incomplete; legacy points preserved")
        client.delete(collection, models.PointIdsList(points=stale_ids), wait=True)
        after = client.count(collection, exact=True).count
        expected = before + len(fresh_ids) - len(existing_new) - len(existing_old)
        if after != expected:
            raise RuntimeError(f"Point count mismatch: {after} != {expected}")
        report.update(applied=True, points_before=before, points_after=after)
        client.close()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--apply", action="store_true")
    run(p.parse_args().apply)
