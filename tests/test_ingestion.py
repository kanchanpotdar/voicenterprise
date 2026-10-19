"""Unit tests for P1 ingestion: duplicate resolution, empty fields, metadata, and chunk provenance."""

import pandas as pd
import pytest

from ingestion.chunker import chunk_documents_df
from ingestion.metadata import enrich_dataframe_with_metadata, extract_metadata_for_record
from ingestion.normalizer import normalize_documents_df


def test_duplicate_doc_id_resolution_exact_vs_conflicting():
    raw_df = pd.DataFrame(
        [
            # Exact duplicate of dsid_exact -> should be deduplicated to 1 record
            {
                "doc_id": "dsid_exact",
                "source_type": "slack",
                "title": "deploy thread",
                "content": "Exact same content here.",
            },
            {
                "doc_id": "dsid_exact",
                "source_type": "slack",
                "title": "deploy thread",
                "content": "Exact same content here.",
            },
            # Conflicting versions sharing dsid_conflict -> BOTH must be preserved!
            {
                "doc_id": "dsid_conflict",
                "source_type": "confluence",
                "title": "Incident Review 2026-03-10",
                "content": "Root cause was autoscaler warmup.",
            },
            {
                "doc_id": "dsid_conflict",
                "source_type": "jira",
                "title": "5xx surge ticket",
                "content": "Issue summary: upstream crashloop.",
            },
        ]
    )

    norm_df = normalize_documents_df(raw_df)
    assert len(norm_df) == 3

    exact_rows = norm_df[norm_df["doc_id"] == "dsid_exact"]
    assert len(exact_rows) == 1
    assert exact_rows.iloc[0]["record_id"] == "dsid_exact"
    assert bool(exact_rows.iloc[0]["has_duplicate_doc_id_conflict"]) is False

    conflict_rows = norm_df[norm_df["doc_id"] == "dsid_conflict"].sort_values("version_index")
    assert len(conflict_rows) == 2
    assert list(conflict_rows["record_id"]) == ["dsid_conflict__v1", "dsid_conflict__v2"]
    assert all(conflict_rows["has_duplicate_doc_id_conflict"])
    assert list(conflict_rows["version_count"]) == [2, 2]


def test_empty_title_and_empty_content_handling():
    raw_df = pd.DataFrame(
        [
            {
                "doc_id": "dsid_notitle",
                "source_type": "slack",
                "title": "",
                "content": "Alex T.: quick status check on v2 deprecation plan.",
            },
            {
                "doc_id": "dsid_nocontent",
                "source_type": "slack",
                "title": "general",
                "content": "   ",
            },
        ]
    )

    norm_df = normalize_documents_df(raw_df)
    r_notitle = norm_df[norm_df["doc_id"] == "dsid_notitle"].iloc[0]
    assert r_notitle["raw_title"] == ""
    assert bool(r_notitle["title_was_empty"]) is True
    assert r_notitle["normalized_title"].startswith("[slack] Alex T.:")
    assert bool(r_notitle["is_empty_record"]) is False

    r_nocontent = norm_df[norm_df["doc_id"] == "dsid_nocontent"].iloc[0]
    assert bool(r_nocontent["content_was_empty"]) is True
    assert bool(r_nocontent["is_empty_record"]) is True


def test_metadata_extraction_provenance_and_no_guessing():
    # Record with explicit revision date, channels, tickets, and email
    meta = extract_metadata_for_record(
        record_id="dsid_1",
        doc_id="dsid_1",
        raw_title="Runbook: perf-canary",
        raw_content=(
            "Contact aisha.kapoor@redwood.ai or post in #eng-runtime and #eng-oncall.\n"
            "Related ticket: ENG-4021 and PR #188 (uses SHA-256 and UTF-8).\n"
            "Revision history:\n- 2025-12-18: Initial draft\n- 2026-02-14: Added kill switch"
        ),
    )
    assert meta.latest_date == "2026-02-14"
    assert meta.earliest_date == "2025-12-18"
    assert meta.date_provenance == "header_or_revision"
    assert meta.date_confidence == "high"
    assert meta.slack_channels == ["#eng-oncall", "#eng-runtime"]
    assert meta.ticket_ids == ["ENG-4021", "PR-188"]
    assert "SHA-256" not in meta.ticket_ids
    assert "UTF-8" not in meta.ticket_ids
    assert meta.email_addresses == ["aisha.kapoor@redwood.ai"]

    # Record with no dates -> must NOT guess a date
    meta_none = extract_metadata_for_record(
        record_id="dsid_2",
        doc_id="dsid_2",
        raw_title="Simple note",
        raw_content="No dates or tickets mentioned here.",
    )
    assert meta_none.latest_date is None
    assert meta_none.date_confidence == "none"
    assert meta_none.date_provenance == "none"


def test_structure_aware_chunking_preserves_provenance():
    raw_df = pd.DataFrame(
        [
            {
                "doc_id": "dsid_runbook",
                "source_type": "confluence",
                "title": "Perf Canary Runbook",
                "content": (
                    "## Purpose\n"
                    + ("This is the purpose section. " * 20)
                    + "\n\n## Rollback Procedure\n"
                    + ("Step by step rollback instructions in #eng-releases. " * 35)
                ),
            },
            {
                "doc_id": "dsid_empty",
                "source_type": "slack",
                "title": "general",
                "content": "",
            },
        ]
    )
    norm_df = normalize_documents_df(raw_df)
    enriched_df = enrich_dataframe_with_metadata(norm_df)
    chunks_df = chunk_documents_df(enriched_df, max_chunk_chars=800, overlap_chars=100)

    # Empty record generates 0 chunks; runbook generates multiple chunks with section titles
    assert "dsid_empty" not in set(chunks_df["doc_id"])
    assert set(chunks_df["doc_id"]) == {"dsid_runbook"}
    assert len(chunks_df) >= 2
    assert chunks_df["chunk_id"].is_unique
    assert "Purpose" in set(chunks_df["section_title"])
    assert "Rollback Procedure" in set(chunks_df["section_title"])
    for _, row in chunks_df.iterrows():
        assert row["char_length"] <= 800
        assert "Perf Canary Runbook" in row["contextualized_text"]
