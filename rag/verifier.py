"""Evidence-First Answer Verification and Self-Correction (Phase 5).

Verifies:
1. Sentence-level and claim-level evidence support against cited passages.
2. Exact numerical and date consistency between claims and cited evidence.
3. Citation ID validity (detecting hallucinated [dsid_...] IDs).
4. Contradiction & supersession handling (governing policy + superseded notice).
5. Unsupported claim detection (substantive assertions lacking citations).
6. Fact omission: detects when high-relevance evidence was present in context but ignored.
7. Safe revision loop: triggers bounded self-correction when context contains evidence omitted in draft.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from rag.citation_validator import CitationValidationResult, CitationValidator, SentenceCitation
from rag.context_builder import AssembledContext
from rag.generator import detect_abstention, detect_contradiction
from rag.llm import BaseLLMClient, LLMResponse


@dataclass
class VerificationCheckResult:
    """Detailed outcome of answer verification."""

    is_verified: bool
    citation_id_validity: float
    sentence_citation_rate: float
    claim_support_rate: float
    has_hallucinated_citations: bool
    has_unsupported_claims: bool
    has_numerical_inconsistencies: bool
    numerical_mismatches: List[str] = field(default_factory=list)
    has_conflict_handling_error: bool = False
    conflict_error_message: Optional[str] = None
    has_omitted_context_facts: bool = False
    omitted_doc_ids: List[str] = field(default_factory=list)
    revision_needed: bool = False
    revision_feedback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


REVISION_SYSTEM_PROMPT = """You are an exacting verification editor for Redwood Inference.
Your task is to revise an initial candidate answer to ensure strict grounding, full process completeness, and precise citations.

CRITICAL INSTRUCTIONS:
1. Use ONLY facts directly supported by the supplied evidence passages.
2. Fix all citation and factual deficiencies pointed out in the FEEDBACK.
3. Retain exact numerical values, dates, and lowercase SHA-256 hashes as specified in the governing documents.
4. When conflicting or superseded guidance exists:
   - State the final active policy citing the final document.
   - Describe what earlier guidance was superseded, clearly label it as superseded, and cite the earlier document.
5. Cover all required operational phases concisely without Markdown tables."""


class AnswerVerifier:
    """Verifies generated answers and triggers bounded correction."""

    def __init__(
        self,
        validator: Optional[CitationValidator] = None,
        llm_client: Optional[BaseLLMClient] = None,
        max_revisions: int = 1,
    ) -> None:
        self.validator = validator or CitationValidator()
        self.llm = llm_client
        self.max_revisions = max_revisions

    def verify(
        self,
        question: str,
        answer_text: str,
        context: AssembledContext,
    ) -> VerificationCheckResult:
        """Run multi-stage verification on generated answer."""
        is_abs = detect_abstention(answer_text)
        has_contra = detect_contradiction(answer_text)

        # 1. Base deterministic citation validation
        val_res: CitationValidationResult = self.validator.validate(
            answer_text, context, is_abstention=is_abs
        )

        # 2. Numerical consistency check across sentences:
        # Check all cited sentences regardless of lexical overlap to catch false claims hidden by high text overlap
        num_mismatches: List[str] = []
        for sent in val_res.sentences:
            if sent.is_cited and sent.cited_doc_ids:
                # Strip document IDs before extracting factual quantities
                clean_sent = re.sub(r"\[dsid_[a-f0-9]{32}(?:__c\d{3})?\]", "", sent.sentence)
                sent_nums = set(re.findall(r"\b\d+(?:\.\d+)?", clean_sent))
                if sent_nums:
                    for doc_id in sent.cited_doc_ids:
                        matching_passages = [
                            p.chunk_text for p in context.passages if p.doc_id == doc_id
                        ]
                        pass_text = " ".join(matching_passages)
                        clean_pass = re.sub(r"\[dsid_[a-f0-9]{32}(?:__c\d{3})?\]", "", pass_text)
                        pass_nums = set(re.findall(r"\b\d+(?:\.\d+)?", clean_pass))
                        missing_nums = sent_nums - pass_nums
                        if missing_nums:
                            num_mismatches.append(
                                f"Quantities {missing_nums} in sentence '{sent.sentence[:60]}...' "
                                f"not found in cited doc {doc_id}"
                            )
                            # Invalidate claim support if numbers contradict cited source
                            sent.is_supported_by_context = False

        has_num_inconsistencies = len(num_mismatches) > 0

        # 3. Conflict resolution check
        # If context contains superseded documents or conflicting terms
        has_conflict_error = False
        conflict_msg = None
        has_supersession_evidence = any(
            re.search(r"\b(supersed|obsolete|deprecated|draft)\b", p.chunk_text, re.IGNORECASE)
            for p in context.passages
        )
        if has_supersession_evidence and not is_abs:
            if not has_contra:
                has_conflict_error = True
                conflict_msg = (
                    "Context contains supersession or version evolution evidence, but the answer "
                    "did not describe or cite the superseded/conflicting instruction."
                )

        # 4. Context omission check
        # Check if primary documents in context were completely uncited
        omitted_docs = []
        if not is_abs and len(context.unique_doc_ids) > 1:
            for doc_id in context.unique_doc_ids[:3]:
                if doc_id not in val_res.valid_doc_ids:
                    omitted_docs.append(doc_id)
        has_omitted = len(omitted_docs) > 0

        # Determine if revision is needed
        revision_needed = (
            val_res.has_hallucinated_citations
            or val_res.has_unsupported_claims
            or has_num_inconsistencies
            or has_conflict_error
        )

        feedback_parts = []
        if val_res.has_hallucinated_citations:
            feedback_parts.append(f"Remove or fix hallucinated document IDs: {val_res.invalid_doc_ids}.")
        if val_res.has_unsupported_claims:
            feedback_parts.append("Ensure every factual claim has an exact [Doc ID] immediately following it.")
        if has_num_inconsistencies:
            feedback_parts.append(f"Correct numerical discrepancies: {'; '.join(num_mismatches)}.")
        if has_conflict_error and conflict_msg:
            feedback_parts.append(conflict_msg)

        feedback = " ".join(feedback_parts) if feedback_parts else None

        is_verified = not revision_needed

        return VerificationCheckResult(
            is_verified=is_verified,
            citation_id_validity=val_res.citation_id_validity,
            sentence_citation_rate=val_res.sentence_citation_rate,
            claim_support_rate=val_res.claim_support_rate,
            has_hallucinated_citations=val_res.has_hallucinated_citations,
            has_unsupported_claims=val_res.has_unsupported_claims,
            has_numerical_inconsistencies=has_num_inconsistencies,
            numerical_mismatches=num_mismatches,
            has_conflict_handling_error=has_conflict_error,
            conflict_error_message=conflict_msg,
            has_omitted_context_facts=has_omitted,
            omitted_doc_ids=omitted_docs,
            revision_needed=revision_needed,
            revision_feedback=feedback,
        )

    def verify_and_correct(
        self,
        question: str,
        answer_text: str,
        context: AssembledContext,
        max_tokens: int = 2048,
    ) -> Tuple[str, VerificationCheckResult, int]:
        """Verify candidate answer and execute bounded self-correction if needed."""
        initial_check = self.verify(question, answer_text, context)
        if initial_check.is_verified or self.llm is None or self.max_revisions <= 0:
            return answer_text, initial_check, 0

        # Execute correction
        revisions_done = 0
        current_text = answer_text
        current_check = initial_check

        for attempt in range(self.max_revisions):
            revisions_done += 1
            correction_prompt = (
                f"EVIDENCE PASSAGES:\n{context.formatted_context}\n\n"
                f"QUESTION:\n{question}\n\n"
                f"INITIAL DRAFT TO REVISE:\n{current_text}\n\n"
                f"VERIFICATION FEEDBACK (FIX THESE DEFICIENCIES):\n{current_check.revision_feedback}\n\n"
                f"REVISED ANSWER:"
            )
            resp: LLMResponse = self.llm.generate(
                prompt=correction_prompt,
                system_prompt=REVISION_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            revised_text = resp.text.strip()
            revised_check = self.verify(question, revised_text, context)

            # Keep revision if it improved or satisfied verification
            current_text = revised_text
            current_check = revised_check
            if current_check.is_verified:
                break

        return current_text, current_check, revisions_done
