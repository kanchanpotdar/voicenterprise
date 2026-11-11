"""Coverage-Aware Context Construction for Complex Enterprise RAG (Phase 4).

Addresses multi-document and conflicting evidence challenges:
- qst_0436: Multi-document SDK reports across languages
- qst_0437: Postmortem action items and team assignments across incidents
- qst_0447: Multi-phase emergency hotfix SOP across surfaces (Hosted + Dedicated)
- qst_0448: Customer retention exceptions across accounts
- qst_0427: Authoritative specification supersession over obsolete drafts

Strategy:
1. Greedy Evidence Coverage with Novelty Bonus & Redundancy Penalties.
2. Authority & Supersession Precedence: boosts final policies, RFCs, and changelogs.
3. Strict Token Budget Enforcement: fits maximum distinct evidence into context.
4. Detailed Provenance Logging: records selected vs excluded chunks and reasons.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rag.context_builder import AssembledContext, ContextBuilder, EvidencePassage, _TOKENIZER, estimate_tokens
from retrieval.indexer import RetrievedChunk

# Words indicative of explicit supersession or deprecation precedence
AUTHORITY_KEYWORDS = {
    "supersedes", "superseded", "superseding", "governing",
    "deprecation", "obsolete", "deprecated", "replaces", "replaced",
}

# Stopwords for lexical coverage computation
LEXICAL_STOPWORDS = {
    "the", "and", "is", "in", "to", "of", "a", "an", "that", "this", "it",
    "for", "on", "with", "as", "by", "at", "from", "be", "are", "was",
    "were", "or", "which", "will", "would", "must", "should", "can", "could",
    "also", "each", "all", "any", "not", "have", "has", "had", "per", "than",
    "does", "been", "into", "over", "under", "both", "such", "then", "what",
    "how", "when", "where", "who", "why", "which",
}


def _tokenize_words(text: str) -> Set[str]:
    """Extract lowercased alphanumeric words excluding stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", text.lower())
    return {w for w in words if w not in LEXICAL_STOPWORDS}


def _jaccard_set_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


@dataclass
class ChunkSelectionAudit:
    """Detailed trace for why a chunk was selected or excluded."""

    chunk_id: str
    doc_id: str
    score: float
    selected: bool
    reason: str
    token_count: int
    novelty_ratio: float
    redundancy_penalty: float
    has_authority_signal: bool


class CoverageAwareContextBuilder(ContextBuilder):
    """Greedy coverage-aware evidence selection within a strict token budget."""

    def __init__(
        self,
        max_context_tokens: int = 2500,
        max_chunks_per_doc: int = 3,
        min_distinct_docs: int = 2,
        max_primary_docs: int = 7,
        novelty_weight: float = 0.25,
        redundancy_penalty_weight: float = 0.45,
        authority_boost: float = 0.12,
    ) -> None:
        super().__init__(
            max_context_tokens=max_context_tokens,
            max_chunks_per_doc=max_chunks_per_doc,
            min_distinct_docs=min_distinct_docs,
            max_primary_docs=max_primary_docs,
        )
        self.novelty_weight = novelty_weight
        self.redundancy_penalty_weight = redundancy_penalty_weight
        self.authority_boost = authority_boost

    def build_context_with_audit(
        self,
        retrieved_chunks: Sequence[RetrievedChunk],
        query: Optional[str] = None,
        max_tokens_override: Optional[int] = None,
    ) -> Tuple[AssembledContext, List[ChunkSelectionAudit]]:
        """Select chunks by parent-document breadth (Pass 1) + greedy coverage (Pass 2) and return audit log."""
        budget = max_tokens_override or self.max_context_tokens
        if not retrieved_chunks:
            return AssembledContext(formatted_context="", passages=[], total_tokens=0), []

        query_tokens = _tokenize_words(query) if query else set()

        # Step 1: Pre-process candidate chunks and group by parent document
        max_retriever_score = max((ch.score for ch in retrieved_chunks), default=1.0)
        min_retriever_score = min((ch.score for ch in retrieved_chunks), default=0.0)
        score_range = max(0.001, max_retriever_score - min_retriever_score)
        is_rrf_scale = max_retriever_score <= 2.0

        doc_best_score: Dict[str, float] = {}
        doc_lexical_score: Dict[str, float] = {}
        doc_to_candidates: Dict[str, List[Dict[str, Any]]] = {}
        candidate_info: List[Dict[str, Any]] = []

        for ch in retrieved_chunks:
            doc_best_score[ch.doc_id] = max(doc_best_score.get(ch.doc_id, float("-inf")), ch.score)
            if getattr(ch, "lexical_score", None) is not None:
                doc_lexical_score[ch.doc_id] = max(
                    doc_lexical_score.get(ch.doc_id, 0.0), float(ch.lexical_score)
                )
            tokens = _tokenize_words(ch.chunk_text)
            has_auth = any(kw in tokens for kw in AUTHORITY_KEYWORDS)
            norm_score = (ch.score - min_retriever_score) / score_range
            dummy_p = EvidencePassage(
                passage_index=1,
                doc_id=ch.doc_id,
                chunk_id=ch.chunk_id,
                source_type=ch.source_type,
                normalized_title=ch.normalized_title,
                section_title=ch.section_title,
                chunk_text=ch.chunk_text,
                score=ch.score,
                latest_date=ch.latest_date,
                date_confidence=ch.date_confidence,
            )
            est_tokens = estimate_tokens(dummy_p.to_formatted_passage())
            info = {
                "chunk": ch,
                "tokens": tokens,
                "has_authority": has_auth,
                "norm_score": norm_score,
                "est_tokens": est_tokens,
            }
            candidate_info.append(info)
            doc_to_candidates.setdefault(ch.doc_id, []).append(info)

        for d in doc_to_candidates:
            doc_to_candidates[d].sort(key=lambda c: c["chunk"].score, reverse=True)

        ranked_docs = sorted(
            doc_best_score.keys(),
            key=lambda d: doc_best_score[d],
            reverse=True,
        )

        # Determine primary documents (Pass 1 breadth)
        if not is_rrf_scale:
            primary_cutoff = 0.55 * max_retriever_score
            primary_docs = [
                d for i, d in enumerate(ranked_docs)
                if (i < self.min_distinct_docs) or (doc_best_score[d] >= primary_cutoff)
            ][: self.max_primary_docs]
        elif doc_lexical_score:
            primary_docs = [
                d for i, d in enumerate(ranked_docs)
                if (i < self.min_distinct_docs)
                or (doc_best_score[d] >= 0.25 * max_retriever_score)
                or (doc_lexical_score.get(d, 0.0) >= 0.55)
            ][: self.max_primary_docs]
        else:
            top2_focused = (
                len(ranked_docs) >= 2
                and sum(1 for c in retrieved_chunks[:7] if c.doc_id == ranked_docs[0]) >= 3
                and sum(1 for c in retrieved_chunks[:7] if c.doc_id == ranked_docs[1]) >= 3
            )
            if top2_focused:
                primary_docs = ranked_docs[: self.min_distinct_docs]
            else:
                primary_docs = [
                    d for i, d in enumerate(ranked_docs)
                    if (i < self.min_distinct_docs)
                    or (doc_best_score[d] >= 0.22 * max_retriever_score)
                    or (doc_best_score[d] >= 0.14)
                ][: self.max_primary_docs]

        max_tokens_per_chunk: Optional[int] = None
        if len(primary_docs) >= 5:
            cap = 320 if is_rrf_scale else 300
            max_tokens_per_chunk = min(cap, max(280, (budget - 60 * len(primary_docs)) // len(primary_docs)))

        selected_candidates: List[RetrievedChunk] = []
        covered_vocabulary: Set[str] = set()
        doc_counts: Dict[str, int] = {}
        audits: List[ChunkSelectionAudit] = []
        current_tokens = 0
        effective_budget = (budget - 8) if budget >= 400 else budget
        remaining = list(candidate_info)

        def _maybe_trim_chunk(ch: RetrievedChunk, allow_trim: bool) -> Tuple[RetrievedChunk, int]:
            txt = ch.chunk_text
            if allow_trim and max_tokens_per_chunk and estimate_tokens(txt) > max_tokens_per_chunk:
                paras = txt.split("\n\n")
                trimmed = ""
                for p in paras:
                    cand_t = (trimmed + "\n\n" + p) if trimmed else p
                    if estimate_tokens(cand_t) <= max_tokens_per_chunk:
                        trimmed = cand_t
                    else:
                        break
                if trimmed and (not is_rrf_scale or estimate_tokens(trimmed) >= max(160, max_tokens_per_chunk // 2)):
                    txt = trimmed
                elif is_rrf_scale and _TOKENIZER is not None:
                    txt = _TOKENIZER.decode(
                        _TOKENIZER.encode(txt, add_special_tokens=False)[:max_tokens_per_chunk],
                        skip_special_tokens=True,
                    )
                else:
                    txt = trimmed or txt[: max_tokens_per_chunk * 4]
            new_ch = RetrievedChunk(
                chunk_id=ch.chunk_id,
                record_id=ch.record_id,
                doc_id=ch.doc_id,
                score=ch.score,
                source_type=ch.source_type,
                normalized_title=ch.normalized_title,
                section_title=ch.section_title,
                chunk_text=txt,
                latest_date=ch.latest_date,
                date_confidence=ch.date_confidence,
                slack_channels=ch.slack_channels,
                ticket_ids=ch.ticket_ids,
                lexical_score=getattr(ch, "lexical_score", None),
            )
            dp = EvidencePassage(
                passage_index=len(selected_candidates) + 1,
                doc_id=new_ch.doc_id,
                chunk_id=new_ch.chunk_id,
                source_type=new_ch.source_type,
                normalized_title=new_ch.normalized_title,
                section_title=new_ch.section_title,
                chunk_text=new_ch.chunk_text,
                score=new_ch.score,
                latest_date=new_ch.latest_date,
                date_confidence=new_ch.date_confidence,
            )
            return new_ch, estimate_tokens(dp.to_formatted_passage())

        # Pass 1: Ensure top distinct primary parent documents are represented first
        for p_idx, doc_id in enumerate(primary_docs):
            if budget >= 400 and current_tokens >= budget - 70:
                break
            cands = doc_to_candidates.get(doc_id, [])
            if not cands:
                continue
            c = cands[0]
            if c not in remaining:
                continue
            ch, cand_toks = _maybe_trim_chunk(c["chunk"], allow_trim=(not is_rrf_scale or p_idx >= 1))
            if current_tokens + cand_toks > effective_budget:
                if not selected_candidates:
                    max_single_body = max(32, effective_budget - 60)
                    if _TOKENIZER is not None:
                        trimmed_single = _TOKENIZER.decode(
                            _TOKENIZER.encode(ch.chunk_text, add_special_tokens=False)[:max_single_body],
                            skip_special_tokens=True,
                        )
                    else:
                        trimmed_single = ch.chunk_text[: max_single_body * 4]
                    ch = RetrievedChunk(
                        chunk_id=ch.chunk_id,
                        record_id=ch.record_id,
                        doc_id=ch.doc_id,
                        score=ch.score,
                        source_type=ch.source_type,
                        normalized_title=ch.normalized_title,
                        section_title=ch.section_title,
                        chunk_text=trimmed_single,
                        latest_date=ch.latest_date,
                        date_confidence=ch.date_confidence,
                        slack_channels=ch.slack_channels,
                        ticket_ids=ch.ticket_ids,
                        lexical_score=getattr(ch, "lexical_score", None),
                    )
                    dp_single = EvidencePassage(
                        passage_index=1,
                        doc_id=ch.doc_id,
                        chunk_id=ch.chunk_id,
                        source_type=ch.source_type,
                        normalized_title=ch.normalized_title,
                        section_title=ch.section_title,
                        chunk_text=ch.chunk_text,
                        score=ch.score,
                        latest_date=ch.latest_date,
                        date_confidence=ch.date_confidence,
                    )
                    cand_toks = estimate_tokens(dp_single.to_formatted_passage())
                    selected_candidates.append(ch)
                    covered_vocabulary.update(c["tokens"])
                    doc_counts[ch.doc_id] = doc_counts.get(ch.doc_id, 0) + 1
                    current_tokens += cand_toks
                    remaining.remove(c)
                    audits.append(
                        ChunkSelectionAudit(
                            chunk_id=ch.chunk_id,
                            doc_id=ch.doc_id,
                            score=round(ch.score, 4),
                            selected=True,
                            reason="Pass-1 primary document fallback",
                            token_count=cand_toks,
                            novelty_ratio=1.0,
                            redundancy_penalty=0.0,
                            has_authority_signal=c["has_authority"],
                        )
                    )
                continue

            tokens = c["tokens"]
            novel_tokens = tokens - covered_vocabulary
            novelty_ratio = len(novel_tokens) / len(tokens) if tokens else 0.0
            max_jaccard = 0.0
            for sel in selected_candidates:
                jac = _jaccard_set_similarity(tokens, _tokenize_words(sel.chunk_text))
                if jac > max_jaccard:
                    max_jaccard = jac

            selected_candidates.append(ch)
            covered_vocabulary.update(tokens)
            doc_counts[ch.doc_id] = doc_counts.get(ch.doc_id, 0) + 1
            current_tokens += cand_toks
            remaining.remove(c)
            audits.append(
                ChunkSelectionAudit(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    score=round(ch.score, 4),
                    selected=True,
                    reason=f"Pass-1 primary document breadth (rank={p_idx + 1}, novelty={novelty_ratio:.2f})",
                    token_count=cand_toks,
                    novelty_ratio=round(novelty_ratio, 4),
                    redundancy_penalty=round(max_jaccard, 4),
                    has_authority_signal=c["has_authority"],
                )
            )

        # Pass 2: Iterative greedy selection for remaining budget across sibling & secondary chunks
        primary_set = set(primary_docs)
        while remaining:
            if budget >= 400 and current_tokens >= budget - 70:
                break
            best_candidate = None
            best_trimmed_chunk = None
            best_cand_toks = 0
            best_selection_score = float("-inf")
            best_novelty = 0.0
            best_redundancy = 0.0

            for c in remaining:
                ch = c["chunk"]
                tokens = c["tokens"]

                # Constraint 1: max chunks per document
                if doc_counts.get(ch.doc_id, 0) >= self.max_chunks_per_doc:
                    continue

                trimmed_ch, cand_toks = _maybe_trim_chunk(ch, allow_trim=True)

                # Constraint 2: check if formatted passage fits in remaining budget
                if current_tokens + cand_toks > effective_budget:
                    continue

                # Calculate novelty: proportion of tokens not yet covered
                novel_tokens = tokens - covered_vocabulary
                novelty_ratio = len(novel_tokens) / len(tokens) if tokens else 0.0

                # Calculate query alignment bonus
                query_alignment = (
                    len(tokens.intersection(query_tokens)) / len(query_tokens)
                    if query_tokens
                    else 0.0
                )

                # Calculate redundancy penalty: max Jaccard with any already selected chunk
                max_jaccard = 0.0
                for sel in selected_candidates:
                    sel_tokens = _tokenize_words(sel.chunk_text)
                    jac = _jaccard_set_similarity(tokens, sel_tokens)
                    if jac > max_jaccard:
                        max_jaccard = jac

                # Authority signal boost (only for competitive candidates)
                auth_bonus = (
                    self.authority_boost
                    if (c["has_authority"] and c["norm_score"] >= 0.15)
                    else 0.0
                )

                # Primary document sibling bonus so primary docs get their spec/SLA sibling chunks before noise
                primary_sib_bonus = 0.25 if ch.doc_id in primary_set else 0.0

                combined_score = (
                    c["norm_score"]
                    + primary_sib_bonus
                    + (self.novelty_weight * novelty_ratio)
                    + (0.20 * query_alignment)
                    + auth_bonus
                    - (self.redundancy_penalty_weight * max_jaccard)
                )

                if combined_score > best_selection_score:
                    best_selection_score = combined_score
                    best_candidate = c
                    best_trimmed_chunk = trimmed_ch
                    best_cand_toks = cand_toks
                    best_novelty = novelty_ratio
                    best_redundancy = max_jaccard

            if not best_candidate or best_trimmed_chunk is None:
                break

            selected_candidates.append(best_trimmed_chunk)
            covered_vocabulary.update(best_candidate["tokens"])
            doc_counts[best_trimmed_chunk.doc_id] = doc_counts.get(best_trimmed_chunk.doc_id, 0) + 1
            current_tokens += best_cand_toks
            remaining.remove(best_candidate)

            audits.append(
                ChunkSelectionAudit(
                    chunk_id=best_trimmed_chunk.chunk_id,
                    doc_id=best_trimmed_chunk.doc_id,
                    score=round(best_trimmed_chunk.score, 4),
                    selected=True,
                    reason=f"Greedy selection (score={best_selection_score:.3f}, novelty={best_novelty:.2f})",
                    token_count=best_cand_toks,
                    novelty_ratio=round(best_novelty, 4),
                    redundancy_penalty=round(best_redundancy, 4),
                    has_authority_signal=best_candidate["has_authority"],
                )
            )

        # Audit excluded remaining chunks
        for c in remaining:
            ch = c["chunk"]
            reason = (
                "capped_by_doc_limit"
                if doc_counts.get(ch.doc_id, 0) >= self.max_chunks_per_doc
                else "token_budget_exhausted"
            )
            audits.append(
                ChunkSelectionAudit(
                    chunk_id=ch.chunk_id,
                    doc_id=ch.doc_id,
                    score=round(ch.score, 4),
                    selected=False,
                    reason=reason,
                    token_count=c["est_tokens"],
                    novelty_ratio=0.0,
                    redundancy_penalty=0.0,
                    has_authority_signal=c["has_authority"],
                )
            )

        # Step 3: Format selected passages contiguously by document rank order
        # and sort sibling chunks within each document by chunk_id ascending
        while True:
            selected_doc_order = list(dict.fromkeys(c.doc_id for c in selected_candidates))
            passages: List[EvidencePassage] = []
            accumulated_text = ""
            doc_id_to_passages: Dict[str, List[int]] = {}
            unique_doc_ids: List[str] = []
            unique_chunk_ids: List[str] = []
            doc_dist: Dict[str, int] = {}
            p_idx = 1

            for doc_id in selected_doc_order:
                d_chunks = [c for c in selected_candidates if c.doc_id == doc_id]
                d_chunks.sort(key=lambda c: c.chunk_id)
                for ch in d_chunks:
                    p = EvidencePassage(
                        passage_index=p_idx,
                        doc_id=ch.doc_id,
                        chunk_id=ch.chunk_id,
                        source_type=ch.source_type,
                        normalized_title=ch.normalized_title,
                        section_title=ch.section_title,
                        chunk_text=ch.chunk_text,
                        score=ch.score,
                        latest_date=ch.latest_date,
                        date_confidence=ch.date_confidence,
                    )
                    block = p.to_formatted_passage()
                    p.token_count = estimate_tokens(block)
                    passages.append(p)
                    accumulated_text += block

                    if ch.doc_id not in unique_doc_ids:
                        unique_doc_ids.append(ch.doc_id)
                    unique_chunk_ids.append(ch.chunk_id)
                    doc_id_to_passages.setdefault(ch.doc_id, []).append(p_idx)
                    doc_dist[ch.doc_id] = doc_dist.get(ch.doc_id, 0) + 1
                    p_idx += 1

            formatted_context = accumulated_text.strip()
            final_tokens = estimate_tokens(formatted_context) if formatted_context else 0
            if final_tokens <= budget:
                break
            if len(selected_candidates) > 1:
                selected_candidates.pop()
            elif selected_candidates:
                excess = final_tokens - budget + 16
                c0 = selected_candidates[0]
                if _TOKENIZER is not None:
                    toks = _TOKENIZER.encode(c0.chunk_text, add_special_tokens=False)
                    if len(toks) > excess and len(toks) > 32:
                        c0.chunk_text = _TOKENIZER.decode(toks[: max(32, len(toks) - excess)], skip_special_tokens=True)
                        continue
                break
            else:
                break

        total_passages = len(passages)
        max_conc = max((cnt / total_passages for cnt in doc_dist.values()), default=0.0)

        assembled = AssembledContext(
            formatted_context=formatted_context,
            passages=passages,
            total_tokens=final_tokens,
            unique_doc_ids=unique_doc_ids,
            unique_chunk_ids=unique_chunk_ids,
            doc_id_to_passages=doc_id_to_passages,
            doc_chunk_distribution=doc_dist,
            max_chunk_concentration=round(max_conc, 4),
            distinct_docs_count=len(unique_doc_ids),
        )

        return assembled, audits

    def build_context(
        self,
        retrieved_chunks: Sequence[RetrievedChunk],
        max_tokens_override: Optional[int] = None,
        query: Optional[str] = None,
    ) -> AssembledContext:
        """Standard ContextBuilder interface override."""
        assembled, _ = self.build_context_with_audit(
            retrieved_chunks=retrieved_chunks,
            query=query,
            max_tokens_override=max_tokens_override,
        )
        return assembled
