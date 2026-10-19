"""Canonical document normalization and duplicate-ID resolution (P1.1).

Handles findings from P0 dataset audit:
1. Duplicate `doc_id` rows: In EnterpriseRAG-Bench, 4 `doc_id`s appear twice with
   DIFFERENT content and titles (and in 3 cases, different `source_type`s: e.g.,
   Confluence vs. HubSpot, Confluence vs. Google Drive, Confluence vs. Jira).
   - Exact duplicates (same `doc_id` and identical `content_sha256`) are deduplicated.
   - Distinct versions sharing a `doc_id` are BOTH preserved with deterministic
     `record_id` (`{doc_id}__v1`, `{doc_id}__v2`), `version_index`, `version_count > 1`,
     and `has_duplicate_doc_id_conflict = True`, while keeping `doc_id` intact for
     benchmark ground-truth matching.
2. Missing fields:
   - 15 Slack documents have empty `title` (""): raw `title` is preserved in `raw_title`,
     `title_was_empty = True`, and a deterministic fallback `normalized_title` is
     synthesized from the source type and opening line.
   - 1 Slack document (`dsid_33cbedf0709949fd9416c8c864a86cf2`) has empty `content` (""):
     preserved in the normalized table with `content_was_empty = True` and
     `is_empty_record = True` so no record is silently dropped without provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd


@dataclass
class NormalizedDocument:
    """Canonical normalized enterprise document record with provenance."""

    record_id: str
    doc_id: str
    source_type: str
    raw_title: str
    normalized_title: str
    raw_content: str
    content_sha256: str
    char_length: int
    version_index: int
    version_count: int
    has_duplicate_doc_id_conflict: bool
    title_was_empty: bool
    content_was_empty: bool
    is_empty_record: bool
    inclusion_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthesize_fallback_title(source_type: str, content: str) -> str:
    cleaned = " ".join(content.strip().splitlines()[:1]).strip()
    if not cleaned:
        return f"[{source_type}] (untitled)"
    snippet = cleaned[:72].rstrip()
    if len(cleaned) > 72:
        snippet += "..."
    return f"[{source_type}] {snippet}"


def normalize_documents_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw documents DataFrame into the canonical P1 schema."""
    required_cols = {"doc_id", "source_type", "title", "content"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {sorted(missing)}")

    # Prepare intermediate records and compute content hashes
    raw_rows: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            raise ValueError("Encountered record with empty doc_id")
        source_type = str(row.get("source_type") or "unknown").strip().lower()
        raw_title = str(row.get("title") or "")
        raw_content = str(row.get("content") or "")
        inclusion_reason = str(row.get("inclusion_reason") or "full_corpus")
        c_hash = _sha256_text(raw_content)

        raw_rows.append(
            {
                "doc_id": doc_id,
                "source_type": source_type,
                "raw_title": raw_title,
                "raw_content": raw_content,
                "content_sha256": c_hash,
                "inclusion_reason": inclusion_reason,
            }
        )

    # Group by doc_id to resolve exact vs. conflicting duplicate doc_ids deterministically
    by_doc_id: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_rows:
        by_doc_id.setdefault(item["doc_id"], []).append(item)

    normalized_records: List[NormalizedDocument] = []
    for doc_id in sorted(by_doc_id.keys()):
        candidates = by_doc_id[doc_id]
        # Deduplicate exact (source_type, raw_title, content_sha256) matches first
        seen_exact = set()
        unique_versions: List[Dict[str, Any]] = []
        for cand in sorted(
            candidates,
            key=lambda x: (x["source_type"], x["raw_title"], x["content_sha256"]),
        ):
            sig = (cand["source_type"], cand["raw_title"], cand["content_sha256"])
            if sig not in seen_exact:
                seen_exact.add(sig)
                unique_versions.append(cand)

        version_count = len(unique_versions)
        has_conflict = version_count > 1

        for idx, ver in enumerate(unique_versions, start=1):
            record_id = doc_id if version_count == 1 else f"{doc_id}__v{idx}"
            raw_title = ver["raw_title"]
            raw_content = ver["raw_content"]
            title_empty = len(raw_title.strip()) == 0
            content_empty = len(raw_content.strip()) == 0

            normalized_title = (
                _synthesize_fallback_title(ver["source_type"], raw_content)
                if title_empty
                else raw_title.strip()
            )

            normalized_records.append(
                NormalizedDocument(
                    record_id=record_id,
                    doc_id=doc_id,
                    source_type=ver["source_type"],
                    raw_title=raw_title,
                    normalized_title=normalized_title,
                    raw_content=raw_content,
                    content_sha256=ver["content_sha256"],
                    char_length=len(raw_content),
                    version_index=idx,
                    version_count=version_count,
                    has_duplicate_doc_id_conflict=has_conflict,
                    title_was_empty=title_empty,
                    content_was_empty=content_empty,
                    is_empty_record=content_empty,
                    inclusion_reason=ver["inclusion_reason"],
                )
            )

    return pd.DataFrame([r.to_dict() for r in normalized_records])
