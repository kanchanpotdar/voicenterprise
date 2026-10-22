"""Deterministic citation extraction and validation for Phase 2 RAG.

Key guarantees:
- Extracts explicit document citations (`[dsid_...]`, `[dsid_...__c000]`,
  `[Doc: dsid_...]`, and `[Passage N]`).
- Validates cited documents against the exact evidence supplied to the generator.
- Flags hallucinated citations (citations of documents NOT present in context).
- Computes citation precision and citation coverage.
- Does NOT claim factual entailment just because a document ID is cited;
  reports citation existence and precision separately from answer correctness.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from rag.context_builder import AssembledContext

# Matches standard doc_id format: dsid_ followed by 32 hex chars (supports [dsid_...], 【dsid_...】, etc.)
DOC_ID_RE = re.compile(r"(?:\[|【|\b)(dsid_[a-f0-9]{32})(?:\]|】|\b)", re.IGNORECASE)

# Matches chunk_id format: dsid_...__c000
CHUNK_ID_RE = re.compile(r"(?:\[|【|\b)(dsid_[a-f0-9]{32}__c\d{3})(?:\]|】|\b)", re.IGNORECASE)

# Matches passage index citation format: [Passage 1], [Evidence Passage 2], [Passage: 3], 【Passage 1】
PASSAGE_REF_RE = re.compile(
    r"(?:\[|【)(?:Evidence\s+)?Passage\s*[:#]?\s*(\d{1,3})(?:\]|】)", re.IGNORECASE
)

COMMON_STOPWORDS = {
    "the", "and", "is", "in", "to", "of", "a", "an", "that", "this", "it",
    "for", "on", "with", "as", "by", "at", "from", "be", "are", "was",
    "were", "or", "which", "will", "would", "must", "should", "can", "could",
    "also", "each", "all", "any", "not", "have", "has", "had", "per", "than",
    "must", "does", "been", "into", "over", "under", "both", "such", "then",
}


@dataclass
class SentenceCitation:
    """A sentence with its extracted citations and claim-level support verification."""

    sentence: str
    cited_doc_ids: List[str]
    is_supported_by_context: bool
    is_cited: bool = True
    citation_ids_valid: bool = True
    lexical_overlap_pct: float = 0.0


@dataclass
class CitationValidationResult:
    """Detailed deterministic citation validation report."""

    total_citations_found: int
    valid_doc_ids: List[str] = field(default_factory=list)
    invalid_doc_ids: List[str] = field(default_factory=list)
    cited_chunk_ids: List[str] = field(default_factory=list)
    uncited_supplied_doc_ids: List[str] = field(default_factory=list)
    citation_precision: float = 1.0
    has_hallucinated_citations: bool = False
    sentences: List[SentenceCitation] = field(default_factory=list)
    citation_id_validity: float = 1.0
    supported_sentences_count: int = 0
    unsupported_sentences_count: int = 0
    has_unsupported_claims: bool = False
    is_abstention: bool = False
    total_sentences_count: int = 0
    cited_sentences_count: int = 0
    unsourced_sentences_count: int = 0
    sentence_citation_rate: float = 1.0
    claim_support_rate: float = 1.0
    has_unsourced_claims: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_citations_found": self.total_citations_found,
            "valid_doc_ids": self.valid_doc_ids,
            "invalid_doc_ids": self.invalid_doc_ids,
            "cited_chunk_ids": self.cited_chunk_ids,
            "uncited_supplied_doc_ids": self.uncited_supplied_doc_ids,
            "citation_precision": round(self.citation_precision, 4),
            "citation_id_validity": round(self.citation_id_validity, 4),
            "has_hallucinated_citations": self.has_hallucinated_citations,
            "supported_sentences_count": self.supported_sentences_count,
            "unsupported_sentences_count": self.unsupported_sentences_count,
            "has_unsupported_claims": self.has_unsupported_claims,
            "is_abstention": self.is_abstention,
            "total_sentences_count": self.total_sentences_count,
            "cited_sentences_count": self.cited_sentences_count,
            "unsourced_sentences_count": self.unsourced_sentences_count,
            "sentence_citation_rate": round(self.sentence_citation_rate, 4),
            "claim_support_rate": round(self.claim_support_rate, 4),
            "has_unsourced_claims": self.has_unsourced_claims,
            "num_valid": len(self.valid_doc_ids),
            "num_invalid": len(self.invalid_doc_ids),
        }


LOCAL_ABSTENTION_RE = re.compile(
    r"\b(cannot (?:be )?answer(?:ed)?|not (?:available|found|provided|mentioned|addressed|stated)|insufficient (?:information|evidence|details|context|documentation)|no (?:mention|record|evidence|information|documentation)|does not (?:contain|state|mention|provide|specify))\b",
    re.IGNORECASE,
)


class CitationValidator:
    """Deterministically extracts and validates citations in generated answers."""

    def validate(
        self,
        answer_text: str,
        assembled_context: AssembledContext,
        is_abstention: Optional[bool] = None,
    ) -> CitationValidationResult:
        """Validate all citations in answer_text against assembled_context."""
        supplied_doc_ids = set(assembled_context.unique_doc_ids)
        passage_map: Dict[int, str] = {
            p.passage_index: p.doc_id for p in assembled_context.passages
        }

        # Determine abstention status
        is_abs = (
            is_abstention
            if is_abstention is not None
            else bool(LOCAL_ABSTENTION_RE.search(answer_text))
        )

        # 1. Extract direct doc_ids
        raw_doc_ids = [d.lower() for d in DOC_ID_RE.findall(answer_text)]
        # 2. Extract direct chunk_ids
        raw_chunk_ids = [c.lower() for c in CHUNK_ID_RE.findall(answer_text)]
        # Add parent doc_ids from chunk_ids
        for cid in raw_chunk_ids:
            parent = cid.split("__")[0]
            raw_doc_ids.append(parent)

        # 3. Extract passage references [Passage N] and map to doc_id
        for p_idx_str in PASSAGE_REF_RE.findall(answer_text):
            p_idx = int(p_idx_str)
            if p_idx in passage_map:
                raw_doc_ids.append(passage_map[p_idx])

        # Deduplicate cited doc_ids while preserving order
        unique_cited_docs = list(dict.fromkeys(raw_doc_ids))
        unique_cited_chunks = list(dict.fromkeys(raw_chunk_ids))

        valid_docs: List[str] = []
        invalid_docs: List[str] = []

        for did in unique_cited_docs:
            if did in supplied_doc_ids:
                valid_docs.append(did)
            else:
                invalid_docs.append(did)

        uncited_supplied = [
            did for did in assembled_context.unique_doc_ids if did not in valid_docs
        ]

        total_cited = len(unique_cited_docs)
        if total_cited == 0:
            # If no citations were made:
            # Abstention answers legitimately cite nothing (100% precision)
            # Answers asserting factual claims without citations get 0.0 precision
            precision = 1.0 if is_abs else 0.0
            id_validity = 1.0
        else:
            precision = len(valid_docs) / total_cited
            id_validity = len(valid_docs) / total_cited

        # Pre-build lowercase concatenated text per doc_id for claim support auditing
        doc_texts: Dict[str, str] = {}
        for p in assembled_context.passages:
            doc_texts[p.doc_id] = (doc_texts.get(p.doc_id, "") + " " + p.chunk_text).lower()

        # Inspect sentences
        sentences: List[SentenceCitation] = []
        raw_sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()
        ]
        supported_count = 0
        unsupported_count = 0
        unsourced_count = 0
        cited_count = 0

        for s in raw_sentences:
            s_docs = list(dict.fromkeys([d.lower() for d in DOC_ID_RE.findall(s)]))
            for p_num in PASSAGE_REF_RE.findall(s):
                p_int = int(p_num)
                if p_int in passage_map:
                    s_docs.append(passage_map[p_int])
            s_docs = list(dict.fromkeys(s_docs))

            if s_docs:
                cited_count += 1
                ids_valid = bool(all(d in supplied_doc_ids for d in s_docs))
                # Audit claim-level support against cited documents
                s_words = [
                    w.lower()
                    for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{3,}\b", s)
                    if w.lower() not in COMMON_STOPWORDS and not w.lower().startswith("dsid_")
                ]
                if not s_words:
                    overlap = 1.0
                else:
                    combined_cited = " ".join(doc_texts.get(d, "") for d in s_docs)
                    overlap = sum(1 for w in s_words if w in combined_cited) / len(s_words)

                # Numerical consistency check: if the claim asserts specific quantities, verify they exist in cited text
                s_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", s)
                numbers_in_doc = True
                if s_numbers:
                    combined_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", combined_cited))
                    # If sentence asserts numbers, verify numbers exist in the cited document
                    if not any(num in combined_numbers for num in s_numbers):
                        numbers_in_doc = False

                is_supported = ids_valid and (overlap >= 0.35) and numbers_in_doc
                sentences.append(
                    SentenceCitation(
                        sentence=s,
                        cited_doc_ids=s_docs,
                        is_supported_by_context=is_supported,
                        is_cited=True,
                        citation_ids_valid=ids_valid,
                        lexical_overlap_pct=round(overlap, 4),
                    )
                )
                if is_supported:
                    supported_count += 1
                else:
                    unsupported_count += 1
            else:
                words = [w for w in s.split() if w.isalnum()]
                is_substantive = len(words) > 3 and not is_abs
                if is_substantive:
                    unsourced_count += 1
                sentences.append(
                    SentenceCitation(
                        sentence=s,
                        cited_doc_ids=[],
                        is_supported_by_context=False,
                        is_cited=False,
                        citation_ids_valid=False,
                        lexical_overlap_pct=0.0,
                    )
                )

        total_s = len(raw_sentences)
        s_cite_rate = (cited_count / total_s) if total_s > 0 else (1.0 if is_abs else 0.0)
        c_supp_rate = (supported_count / total_s) if total_s > 0 else (1.0 if is_abs else 0.0)
        has_unsourced = unsourced_count > 0 and not is_abs
        has_unsupported = (unsupported_count > 0 or unsourced_count > 0) and not is_abs

        return CitationValidationResult(
            total_citations_found=total_cited,
            valid_doc_ids=valid_docs,
            invalid_doc_ids=invalid_docs,
            cited_chunk_ids=unique_cited_chunks,
            uncited_supplied_doc_ids=uncited_supplied,
            citation_precision=precision,
            has_hallucinated_citations=len(invalid_docs) > 0,
            sentences=sentences,
            citation_id_validity=id_validity,
            supported_sentences_count=supported_count,
            unsupported_sentences_count=unsupported_count,
            has_unsupported_claims=has_unsupported,
            is_abstention=is_abs,
            total_sentences_count=total_s,
            cited_sentences_count=cited_count,
            unsourced_sentences_count=unsourced_count,
            sentence_citation_rate=s_cite_rate,
            claim_support_rate=c_supp_rate,
            has_unsourced_claims=has_unsourced,
        )
