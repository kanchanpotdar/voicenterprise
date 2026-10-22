"""Evidence selection, token budgeting, and context assembly for Phase 2 RAG.

Key responsibilities:
- Select and rank evidence chunks within a strict token budget.
- Deduplicate and balance chunks across parent documents (preventing a single
  document from consuming the entire context window).
- Preserve rich document provenance: `doc_id`, `chunk_id`, `source_type`,
  `normalized_title`, `section_title`, and temporal metadata (`latest_date`).
- Support multi-document queries (e.g. completeness, project_related) by
  ensuring diverse distinct parent documents are represented in the budget.
- Never use gold answers or ground truth document labels at inference time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from retrieval.indexer import RetrievedChunk

try:
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-en-v1.5", model_max_length=100000
    )
except Exception:
    _TOKENIZER = None


def estimate_tokens(text: str) -> int:
    """Accurately count tokens using BGE tokenizer or robust whitespace/char fallback."""
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Fallback heuristic: 1 token ~= 4 chars or 0.75 words
    return max(1, len(text) // 4)


@dataclass
class EvidencePassage:
    """A single retrieved chunk packaged as an evidence passage with provenance."""

    passage_index: int
    doc_id: str
    chunk_id: str
    source_type: str
    normalized_title: str
    section_title: str
    chunk_text: str
    score: float
    latest_date: Optional[str] = None
    date_confidence: str = "none"
    token_count: int = 0

    def format_header(self) -> str:
        date_str = f" | Date: {self.latest_date}" if self.latest_date else ""
        section_str = f" | Section: {self.section_title}" if self.section_title else ""
        return (
            f"[Evidence Passage {self.passage_index} | "
            f"Doc ID: {self.doc_id} | "
            f"Chunk ID: {self.chunk_id} | "
            f"Source: {self.source_type} | "
            f"Title: {self.normalized_title}"
            f"{section_str}{date_str}]"
        )

    def to_formatted_passage(self) -> str:
        return f"{self.format_header()}\n{self.chunk_text.strip()}\n"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AssembledContext:
    """The complete assembled context ready for injection into the generator prompt."""

    formatted_context: str
    passages: List[EvidencePassage] = field(default_factory=list)
    total_tokens: int = 0
    unique_doc_ids: List[str] = field(default_factory=list)
    unique_chunk_ids: List[str] = field(default_factory=list)
    doc_id_to_passages: Dict[str, List[int]] = field(default_factory=dict)
    doc_chunk_distribution: Dict[str, int] = field(default_factory=dict)
    max_chunk_concentration: float = 0.0
    distinct_docs_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formatted_context": self.formatted_context,
            "total_tokens": self.total_tokens,
            "num_passages": len(self.passages),
            "unique_doc_ids": self.unique_doc_ids,
            "unique_chunk_ids": self.unique_chunk_ids,
            "passages": [p.to_dict() for p in self.passages],
            "doc_chunk_distribution": self.doc_chunk_distribution,
            "max_chunk_concentration": round(self.max_chunk_concentration, 4),
            "distinct_docs_count": self.distinct_docs_count,
        }


class ContextBuilder:
    """Selects, balances, and formats retrieved chunks into a bounded LLM context."""

    def __init__(
        self,
        max_context_tokens: int = 2500,
        max_chunks_per_doc: int = 3,
        min_distinct_docs: int = 2,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.max_chunks_per_doc = max_chunks_per_doc
        self.min_distinct_docs = min_distinct_docs

    def build_context(
        self,
        retrieved_chunks: Sequence[RetrievedChunk],
        max_tokens_override: Optional[int] = None,
    ) -> AssembledContext:
        """Assemble retrieved chunks into a diverse, bounded evidence context.

        Strategy:
        1. Score-aware multi-document balancing: Group candidates by parent document.
           Ensure high-relevance documents receive their primary sibling chunks,
           preventing low-scoring trailing noise from displacing crucial policy/spec
           sections.
        2. Sibling chunk ordering: Contiguously arrange chunks of each document in
           natural sequential offset order (`chunk_id` ascending).
        3. Strict token budgeting: Format passages and accumulate until the token
           budget is reached. Never truncate mid-passage.
        4. Concentration tracking: Audit chunk distribution across parent documents.
        """
        budget = max_tokens_override or self.max_context_tokens
        if not retrieved_chunks:
            return AssembledContext(formatted_context="", passages=[], total_tokens=0)

        # 1. Group candidate chunks by parent document and determine document best score
        doc_best_score: Dict[str, float] = {}
        doc_candidate_chunks: Dict[str, List[RetrievedChunk]] = {}

        for ch in retrieved_chunks:
            doc_best_score[ch.doc_id] = max(doc_best_score.get(ch.doc_id, float("-inf")), ch.score)
            doc_candidate_chunks.setdefault(ch.doc_id, []).append(ch)

        # Rank parent documents by best chunk score
        ranked_docs = sorted(
            doc_best_score.keys(),
            key=lambda d: doc_best_score[d],
            reverse=True,
        )

        top_score = doc_best_score[ranked_docs[0]] if ranked_docs else 1.0

        # Primary documents: score >= 0.55 * top_score or top min_distinct_docs
        primary_cutoff = 0.55 * top_score
        primary_docs = [
            d for d in ranked_docs
            if (doc_best_score[d] >= primary_cutoff) or (ranked_docs.index(d) < self.min_distinct_docs)
        ]
        secondary_docs = [d for d in ranked_docs if d not in primary_docs]

        # Prioritize candidates:
        # Phase 1: Top chunk from each primary document (guarantees diverse high-relevance evidence)
        # Phase 2: High-scoring sibling chunks from primary documents up to max_chunks_per_doc
        # Phase 3: Candidate chunk from secondary documents
        ordered_candidates: List[RetrievedChunk] = []

        # Determine per-document token allowance if there are many primary documents (e.g. completeness / multi-doc SOPs)
        max_tokens_per_chunk: Optional[int] = None
        if len(primary_docs) >= 5:
            max_tokens_per_chunk = max(300, (budget - 150) // len(primary_docs))

        # Phase 1
        for doc_id in primary_docs:
            if doc_candidate_chunks[doc_id]:
                ordered_candidates.append(doc_candidate_chunks[doc_id][0])

        # Phase 2
        for doc_id in primary_docs:
            for ch in doc_candidate_chunks[doc_id][1: self.max_chunks_per_doc]:
                ordered_candidates.append(ch)

        # Phase 3
        for doc_id in secondary_docs:
            if doc_candidate_chunks[doc_id]:
                ordered_candidates.append(doc_candidate_chunks[doc_id][0])

        # Select passages that fit strictly within token budget
        selected_raw_chunks: List[RetrievedChunk] = []
        current_tokens = 0

        for cand in ordered_candidates:
            cand_chunk_text = cand.chunk_text
            if max_tokens_per_chunk and estimate_tokens(cand_chunk_text) > max_tokens_per_chunk:
                paras = cand_chunk_text.split("\n\n")
                trimmed = ""
                for p in paras:
                    if estimate_tokens(trimmed + "\n\n" + p) <= max_tokens_per_chunk:
                        trimmed += ("\n\n" + p) if trimmed else p
                    else:
                        break
                cand_chunk_text = trimmed or cand_chunk_text[: max_tokens_per_chunk * 4]

            dummy_p = EvidencePassage(
                passage_index=1,
                doc_id=cand.doc_id,
                chunk_id=cand.chunk_id,
                source_type=cand.source_type,
                normalized_title=cand.normalized_title,
                section_title=cand.section_title,
                chunk_text=cand_chunk_text,
                score=cand.score,
                latest_date=cand.latest_date,
                date_confidence=cand.date_confidence,
            )
            cand_tokens = estimate_tokens(dummy_p.to_formatted_passage())
            if current_tokens + cand_tokens > budget:
                if not selected_raw_chunks:
                    selected_raw_chunks.append(
                        RetrievedChunk(
                            chunk_id=cand.chunk_id,
                            record_id=cand.record_id,
                            doc_id=cand.doc_id,
                            score=cand.score,
                            source_type=cand.source_type,
                            normalized_title=cand.normalized_title,
                            section_title=cand.section_title,
                            chunk_text=cand_chunk_text,
                            latest_date=cand.latest_date,
                            date_confidence=cand.date_confidence,
                            slack_channels=cand.slack_channels,
                            ticket_ids=cand.ticket_ids,
                        )
                    )
                    current_tokens += cand_tokens
                break

            selected_raw_chunks.append(
                RetrievedChunk(
                    chunk_id=cand.chunk_id,
                    record_id=cand.record_id,
                    doc_id=cand.doc_id,
                    score=cand.score,
                    source_type=cand.source_type,
                    normalized_title=cand.normalized_title,
                    section_title=cand.section_title,
                    chunk_text=cand_chunk_text,
                    latest_date=cand.latest_date,
                    date_confidence=cand.date_confidence,
                    slack_channels=cand.slack_channels,
                    ticket_ids=cand.ticket_ids,
                )
            )
            current_tokens += cand_tokens

        # Now group the SELECTED passages by parent document (in primary rank order)
        # and sort sibling chunks within each document by chunk_id ascending
        selected_doc_order = list(dict.fromkeys(c.doc_id for c in selected_raw_chunks))
        final_passages: List[EvidencePassage] = []
        accumulated_text = ""
        total_tokens = 0
        passage_idx = 1
        doc_to_indices: Dict[str, List[int]] = {}

        for doc_id in selected_doc_order:
            d_chunks = [c for c in selected_raw_chunks if c.doc_id == doc_id]
            d_chunks.sort(key=lambda c: c.chunk_id)
            for cand in d_chunks:
                p = EvidencePassage(
                    passage_index=passage_idx,
                    doc_id=cand.doc_id,
                    chunk_id=cand.chunk_id,
                    source_type=cand.source_type,
                    normalized_title=cand.normalized_title,
                    section_title=cand.section_title,
                    chunk_text=cand.chunk_text,
                    score=cand.score,
                    latest_date=cand.latest_date,
                    date_confidence=cand.date_confidence,
                )
                formatted = p.to_formatted_passage()
                p.token_count = estimate_tokens(formatted)
                final_passages.append(p)
                accumulated_text += formatted
                total_tokens += p.token_count
                doc_to_indices.setdefault(p.doc_id, []).append(p.passage_index)
                passage_idx += 1

        unique_doc_ids = list(dict.fromkeys(p.doc_id for p in final_passages))
        unique_chunk_ids = [p.chunk_id for p in final_passages]

        # Compute chunk concentration metrics
        doc_dist: Dict[str, int] = {}
        for p in final_passages:
            doc_dist[p.doc_id] = doc_dist.get(p.doc_id, 0) + 1

        total_p = len(final_passages)
        max_conc = (max(doc_dist.values()) / total_p) if total_p > 0 else 0.0

        return AssembledContext(
            formatted_context=accumulated_text.strip(),
            passages=final_passages,
            total_tokens=total_tokens,
            unique_doc_ids=unique_doc_ids,
            unique_chunk_ids=unique_chunk_ids,
            doc_id_to_passages=doc_to_indices,
            doc_chunk_distribution=doc_dist,
            max_chunk_concentration=max_conc,
            distinct_docs_count=len(unique_doc_ids),
        )
