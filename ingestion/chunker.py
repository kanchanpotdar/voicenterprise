"""Structure-aware document chunker with section and provenance tracking (P1.3).

Designed specifically for EnterpriseRAG-Bench document structures and `BAAI/bge-small-en-v1.5`
(512-token context window, ~1,800 chars):
- Splits documents first along Markdown section headings (`#`, `##`, `###`) or horizontal
  dividers (`---`), preserving the active `section_title` for every section.
- Packs adjacent small sections up to `max_chunk_chars` (default 1,400 chars, ~350 tokens)
  so short sections don't fragment, while splitting oversized sections along paragraph /
  line boundaries with `overlap_chars` (default 200 chars).
- Preserves both `chunk_text` (exact text from the document) and `contextualized_text`
  (prefixed with `[Source | Title | Section]` for dense & BM25 indexing), along with
  parent `doc_id`, `record_id`, `chunk_index`, and metadata payload fields.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
DIVIDER_RE = re.compile(r"^\s*---+\s*$")


@dataclass
class DocumentChunk:
    """A provenance-preserving chunk of an enterprise document."""

    chunk_id: str
    record_id: str
    doc_id: str
    chunk_index: int
    total_chunks_in_doc: int
    source_type: str
    normalized_title: str
    section_title: str
    chunk_text: str
    contextualized_text: str
    char_length: int
    latest_date: Optional[str]
    date_confidence: str
    slack_channels: List[str]
    ticket_ids: List[str]
    has_duplicate_doc_id_conflict: bool
    inclusion_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_into_sections(content: str) -> List[Tuple[str, str]]:
    """Split markdown/plain text into (section_title, section_body) pairs."""
    lines = content.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    current_title = "Overview"
    current_lines: List[str] = []

    for line in lines:
        if DIVIDER_RE.match(line):
            continue
        m = HEADING_RE.match(line)
        if m:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_title, current_lines))
            current_title = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_title, current_lines))

    return [(title, "\n".join(lns).strip()) for title, lns in sections if "\n".join(lns).strip()]


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Split a text block exceeding max_chars along paragraph/newline boundaries with overlap."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) == 1:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        # If a single paragraph is still larger than max_chars, slice it with character overlap
        if len(para) > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            step = max(1, max_chars - overlap_chars)
            for start in range(0, len(para), step):
                piece = para[start : start + max_chars].strip()
                if piece:
                    chunks.append(piece)
                if start + max_chars >= len(para):
                    break
            continue

        sep_len = 2 if current else 0
        if current_len + sep_len + len(para) <= max_chars:
            current.append(para)
            current_len += sep_len + len(para)
        else:
            chunk_str = "\n\n".join(current)
            chunks.append(chunk_str)
            # Carry over trailing paragraphs up to overlap_chars
            overlap_list: List[str] = []
            overlap_len = 0
            for prev_p in reversed(current):
                if overlap_len + len(prev_p) <= overlap_chars:
                    overlap_list.insert(0, prev_p)
                    overlap_len += len(prev_p) + 2
                else:
                    break
            current = overlap_list + [para]
            current_len = sum(len(x) for x in current) + 2 * max(0, len(current) - 1)

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def chunk_single_document(
    row: Dict[str, Any],
    max_chunk_chars: int = 1400,
    overlap_chars: int = 200,
) -> List[DocumentChunk]:
    """Chunk a single normalized+enriched document row into DocumentChunk objects."""
    if row.get("is_empty_record") or not str(row.get("raw_content") or "").strip():
        return []

    raw_content = str(row["raw_content"])
    sections = _parse_into_sections(raw_content)
    if not sections:
        return []

    # Pack small consecutive sections or split oversized sections
    raw_chunks: List[Tuple[str, str]] = []
    buf_titles: List[str] = []
    buf_texts: List[str] = []
    buf_len = 0

    for sec_title, sec_body in sections:
        if len(sec_body) > max_chunk_chars:
            if buf_texts:
                combined_title = " / ".join(dict.fromkeys(buf_titles))
                raw_chunks.append((combined_title, "\n\n".join(buf_texts)))
                buf_titles, buf_texts, buf_len = [], [], 0
            sub_pieces = _split_long_text(sec_body, max_chunk_chars, overlap_chars)
            for piece in sub_pieces:
                raw_chunks.append((sec_title, piece))
        else:
            sep = 2 if buf_texts else 0
            if buf_len + sep + len(sec_body) <= max_chunk_chars:
                buf_titles.append(sec_title)
                buf_texts.append(sec_body)
                buf_len += sep + len(sec_body)
            else:
                combined_title = " / ".join(dict.fromkeys(buf_titles))
                raw_chunks.append((combined_title, "\n\n".join(buf_texts)))
                buf_titles = [sec_title]
                buf_texts = [sec_body]
                buf_len = len(sec_body)

    if buf_texts:
        combined_title = " / ".join(dict.fromkeys(buf_titles))
        raw_chunks.append((combined_title, "\n\n".join(buf_texts)))

    total_chunks = len(raw_chunks)
    record_id = str(row["record_id"])
    doc_id = str(row["doc_id"])
    source_type = str(row["source_type"])
    norm_title = str(row["normalized_title"])

    out: List[DocumentChunk] = []
    for idx, (sec_title, c_text) in enumerate(raw_chunks):
        chunk_id = f"{record_id}__c{idx:03d}"
        context_header = (
            f"Source: {source_type} | Title: {norm_title} | Section: {sec_title}\n"
        )
        contextualized = f"{context_header}{c_text}"
        out.append(
            DocumentChunk(
                chunk_id=chunk_id,
                record_id=record_id,
                doc_id=doc_id,
                chunk_index=idx,
                total_chunks_in_doc=total_chunks,
                source_type=source_type,
                normalized_title=norm_title,
                section_title=sec_title,
                chunk_text=c_text,
                contextualized_text=contextualized,
                char_length=len(c_text),
                latest_date=row.get("latest_date"),
                date_confidence=str(row.get("date_confidence") or "none"),
                slack_channels=(
                    []
                    if row.get("slack_channels") is None
                    else [str(x) for x in row["slack_channels"]]
                ),
                ticket_ids=(
                    []
                    if row.get("ticket_ids") is None
                    else [str(x) for x in row["ticket_ids"]]
                ),
                has_duplicate_doc_id_conflict=bool(
                    row.get("has_duplicate_doc_id_conflict", False)
                ),
                inclusion_reason=str(row.get("inclusion_reason") or "full_corpus"),
            )
        )
    return out


def chunk_documents_df(
    enriched_df: pd.DataFrame,
    max_chunk_chars: int = 1400,
    overlap_chars: int = 200,
) -> pd.DataFrame:
    """Chunk all documents in an enriched DataFrame into a canonical chunks DataFrame."""
    all_chunks: List[Dict[str, Any]] = []
    for row in enriched_df.to_dict(orient="records"):
        doc_chunks = chunk_single_document(
            row, max_chunk_chars=max_chunk_chars, overlap_chars=overlap_chars
        )
        for ch in doc_chunks:
            all_chunks.append(ch.to_dict())
    return pd.DataFrame(all_chunks)
