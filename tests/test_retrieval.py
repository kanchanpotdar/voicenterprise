"""Unit and smoke tests for P1.4 Qdrant BM25, Dense, and Hybrid indexing & retrieval."""

import pandas as pd

from ingestion.chunker import chunk_documents_df
from ingestion.metadata import enrich_dataframe_with_metadata
from ingestion.normalizer import normalize_documents_df
from retrieval.indexer import EnterpriseRetriever


def _make_smoke_chunks() -> pd.DataFrame:
    raw_df = pd.DataFrame(
        [
            {
                "doc_id": "dsid_canary",
                "source_type": "confluence",
                "title": "Runbook: perf-canary rollout",
                "content": (
                    "## Telemetry\n"
                    "The new metric added for server-side streaming time limits is "
                    "stream.timebox_finalized with labels for route and model."
                ),
            },
            {
                "doc_id": "dsid_upload",
                "source_type": "github",
                "title": "Multipart upload support on OpenAI-compatible endpoints",
                "content": (
                    "## Configuration\n"
                    "Default per-file upload limit max_file_size is 10 MiB and "
                    "max_total_request_size is 50 MiB."
                ),
            },
            {
                "doc_id": "dsid_kappa",
                "source_type": "linear",
                "title": "Design system elevation tokens",
                "content": (
                    "## Acceptance Criteria\n"
                    "Deliver interactive derivative tokens and elevation.kappa.0-6 "
                    "with explicit shadow parameters for dense DataGrid components."
                ),
            },
        ]
    )
    norm_df = normalize_documents_df(raw_df)
    enriched_df = enrich_dataframe_with_metadata(norm_df)
    return chunk_documents_df(enriched_df, max_chunk_chars=800, overlap_chars=100)


def test_repeatable_indexing_and_three_retrieval_modes():
    chunks_df = _make_smoke_chunks()
    retriever = EnterpriseRetriever(collection_name="smoke_test_col", qdrant_path=None)

    stats1 = retriever.index_chunks_df(chunks_df, reset_collection=True)
    assert stats1["chunks_indexed"] == len(chunks_df)
    assert stats1["qdrant_points_count"] == len(chunks_df)
    assert stats1["unique_parent_docs_indexed"] == 3

    # Re-indexing without resetting must be idempotent (deterministic UUIDv5 point IDs)
    stats2 = retriever.index_chunks_df(chunks_df, reset_collection=False)
    assert stats2["qdrant_points_count"] == len(chunks_df)

    # Exact metric identifier query -> BM25 and Hybrid should rank dsid_canary #1
    for mode in ("bm25", "dense", "hybrid"):
        doc_ids, _ = retriever.retrieve_documents(
            "What is the metric stream.timebox_finalized used for?",
            mode=mode,
            top_k_docs=3,
        )
        assert doc_ids[0] == "dsid_canary", f"Failed top-1 for mode={mode}: {doc_ids}"

    # Verify source_type metadata filter
    filtered_docs, _ = retriever.retrieve_documents(
        "elevation.kappa.0-6 tokens",
        mode="hybrid",
        top_k_docs=3,
        source_types=["linear"],
    )
    assert filtered_docs == ["dsid_kappa"]
