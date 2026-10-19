"""Deterministic P1 ingestion pipeline orchestrator (P1.1 -> P1.2 -> P1.3).

Transforms raw enterprise records from a subset or full Parquet file into:
- `documents_normalized.parquet` (canonical normalized + metadata-enriched documents)
- `chunks.parquet` (structure-aware chunks with parent `doc_id` and section provenance)
- `ingestion_manifest.json` (deterministic counts, hashes, duplicate-ID resolution stats,
  empty-field stats, and metadata extraction coverage)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ingestion.chunker import chunk_documents_df
from ingestion.metadata import enrich_dataframe_with_metadata
from ingestion.normalizer import normalize_documents_df


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_ingestion_pipeline(
    input_parquet_path: Path,
    output_dir: Path,
    max_chunk_chars: int = 2000,
    overlap_chars: int = 200,
) -> Dict[str, Any]:
    """Run full P1.1-P1.3 ingestion deterministically and write Parquet + manifest outputs."""
    raw_df = pd.read_parquet(input_parquet_path)
    norm_df = normalize_documents_df(raw_df)
    enriched_df = enrich_dataframe_with_metadata(norm_df)
    chunks_df = chunk_documents_df(
        enriched_df, max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    docs_out_path = output_dir / "documents_normalized.parquet"
    chunks_out_path = output_dir / "chunks.parquet"

    enriched_df.to_parquet(docs_out_path, index=False)
    chunks_df.to_parquet(chunks_out_path, index=False)

    manifest = {
        "input_file": str(input_parquet_path.name),
        "input_sha256": _file_sha256(input_parquet_path),
        "config": {
            "max_chunk_chars": max_chunk_chars,
            "overlap_chars": overlap_chars,
        },
        "document_counts": {
            "raw_input_rows": len(raw_df),
            "normalized_records": len(enriched_df),
            "unique_parent_doc_ids": int(enriched_df["doc_id"].nunique()),
            "conflicting_duplicate_doc_id_records": int(
                enriched_df["has_duplicate_doc_id_conflict"].sum()
            ),
            "empty_title_synthesized_count": int(enriched_df["title_was_empty"].sum()),
            "empty_content_record_count": int(enriched_df["is_empty_record"].sum()),
        },
        "metadata_extraction_coverage": {
            "docs_with_dates": int(enriched_df["latest_date"].notna().sum()),
            "date_confidence_breakdown": (
                enriched_df["date_confidence"].value_counts().to_dict()
            ),
            "docs_with_slack_channels": int(
                enriched_df["slack_channels"].apply(lambda x: len(x) > 0).sum()
            ),
            "docs_with_ticket_ids": int(
                enriched_df["ticket_ids"].apply(lambda x: len(x) > 0).sum()
            ),
            "docs_with_emails": int(
                enriched_df["email_addresses"].apply(lambda x: len(x) > 0).sum()
            ),
        },
        "chunk_counts": {
            "total_chunks": len(chunks_df),
            "unique_parent_doc_ids_in_chunks": int(chunks_df["doc_id"].nunique()),
            "avg_chunks_per_doc": round(
                len(chunks_df) / max(1, int((~enriched_df["is_empty_record"]).sum())), 2
            ),
            "avg_chunk_chars": round(float(chunks_df["char_length"].mean()), 1),
            "max_chunk_chars_observed": int(chunks_df["char_length"].max()),
            "chunks_by_source_type": (
                chunks_df["source_type"].value_counts().to_dict()
            ),
        },
        "output_files": {
            "documents_normalized.parquet": {
                "size_bytes": docs_out_path.stat().st_size,
                "sha256": _file_sha256(docs_out_path),
            },
            "chunks.parquet": {
                "size_bytes": chunks_out_path.stat().st_size,
                "sha256": _file_sha256(chunks_out_path),
            },
        },
    }

    manifest_path = output_dir / "ingestion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    in_parquet = base / "data" / "subsets" / "corpus_dev_10k.parquet"
    out_dir = base / "data" / "ingested" / "dev_10k"
    summary = run_ingestion_pipeline(in_parquet, out_dir)
    print(json.dumps(summary, indent=2))
