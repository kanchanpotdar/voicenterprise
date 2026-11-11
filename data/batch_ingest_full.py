"""Resumable, bounded-memory ingestion of the full EnterpriseRAG corpus.

Each input batch is normalized, metadata-enriched, chunked, and written as
independent Parquet files.  Completed batches are recorded in a manifest so an
interrupted run can resume without repeating completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ingestion.chunker import chunk_documents_df
from ingestion.metadata import enrich_dataframe_with_metadata
from ingestion.normalizer import normalize_documents_df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(input_path: Path, output_dir: Path, batch_rows: int = 5000) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = output_dir / "batches"
    batches_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "ingestion_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "input": str(input_path), "input_sha256": _sha256(input_path),
        "batch_rows": batch_rows, "completed_batches": {}, "errors": {}
    }
    if manifest.get("input_sha256") != _sha256(input_path):
        raise RuntimeError("Input corpus changed since the existing manifest was created")

    parquet = pq.ParquetFile(input_path)
    total_rows = parquet.metadata.num_rows
    manifest["total_rows"] = total_rows
    for batch_no, table in enumerate(parquet.iter_batches(batch_size=batch_rows)):
        key = str(batch_no)
        if key in manifest["completed_batches"]:
            continue
        try:
            raw = table.to_pandas()
            normalized = enrich_dataframe_with_metadata(normalize_documents_df(raw))
            chunks = chunk_documents_df(normalized, max_chunk_chars=2000, overlap_chars=200)
            docs_path = batches_dir / f"batch_{batch_no:05d}_documents.parquet"
            chunks_path = batches_dir / f"batch_{batch_no:05d}_chunks.parquet"
            normalized.to_parquet(docs_path, index=False)
            chunks.to_parquet(chunks_path, index=False)
            manifest["completed_batches"][key] = {
                "rows": len(raw), "documents": len(normalized), "chunks": len(chunks),
                "documents_path": str(docs_path.relative_to(output_dir)),
                "chunks_path": str(chunks_path.relative_to(output_dir)),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"completed batch {batch_no} ({len(raw)} rows, {len(chunks)} chunks)", flush=True)
        except Exception as exc:
            manifest["errors"][key] = repr(exc)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            raise
    manifest["status"] = "complete" if len(manifest["completed_batches"]) == (total_rows + batch_rows - 1) // batch_rows else "partial"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("EnterpriseRAG-Bench/data/documents/test.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/ingested/full_corpus"))
    parser.add_argument("--batch-rows", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, args.batch_rows), indent=2))
