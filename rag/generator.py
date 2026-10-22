"""Grounded answer generation with citation enforcement and abstention detection.

Key guarantees:
- Instructs LLMs to synthesize answers strictly from supplied evidence passages.
- Requires explicit `[dsid_...]` citations for all factual claims.
- Enforces safe abstention when information is missing.
- Formulates balanced responses highlighting contradictions when documents conflict.
- Automatically validates citations and flags unsupported statements.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from rag.citation_validator import CitationValidationResult, CitationValidator
from rag.context_builder import AssembledContext
from rag.llm import BaseLLMClient, LLMResponse

GROUNDED_SYSTEM_PROMPT = """You are a precise, evidence-grounded AI knowledge investigator for Redwood Inference.
Your task is to answer the user's question accurately using ONLY the supplied evidence passages.

CRITICAL INSTRUCTIONS:
1. STRICT GROUNDING: State only facts directly supported by the evidence passages. Never extrapolate, speculate, or bring in outside information.
2. CITATION ENFORCEMENT: Every substantive factual claim MUST be followed by the document ID it comes from in brackets, e.g., [dsid_0123456789abcdef0123456789abcdef]. Use the exact Doc ID from the evidence header immediately following each claim. Use standard ASCII brackets [ ] only.
3. MISSING INFORMATION & ABSTENTION: If the supplied evidence does not contain the answer, or only partially addresses the query, you MUST explicitly state that the information is not found or cannot be fully answered from the available documents. Do not attempt to guess.
4. CONFLICT RESOLUTION & SUPERSEDED GUIDANCE: Check all evidence passages for conflicting recommendations, drafts, version evolution, or supersession notices.
   - When one document updates, supersedes, or finalizes another (e.g. final specification vs earlier email proposal/draft), you MUST state the final authoritative requirement citing the final document, AND explicitly describe the earlier guidance that was superseded, clearly label it as superseded, and cite the earlier document. Never omit the superseded version when conflicting guidance exists in the evidence.
   - CITATION TO CLAIM ALIGNMENT: When citing an earlier superseded instruction, proposal, or schedule, you MUST cite the original earlier document where that obsolete draft was proposed. DO NOT cite the newer document or its change log as evidence for the obsolete instruction. The newer document should only be cited for the final, active policy.
   - If documents contradict without clear authority or timestamps, explain the discrepancy, communicate uncertainty, and cite each source.
5. COMPLETE PROCESS COVERAGE & CONCISENESS: When a question asks for a complete end-to-end SOP, checklist, or process across multiple surfaces (e.g. Hosted + Dedicated):
   - Systematically integrate all operational phases from the evidence: Preconditions/Checklist, Approvals, Code PR & Review, Build & Signing, Pre-rollout Validation, Incident Communications, Hosted Rollout & Rollback, Dedicated Rollout & Rollback, Customer Communications, and Post-Change Closeout.
   - Present as concise numbered steps and bullet points. DO NOT output Markdown tables, as wide tables consume formatting tokens and cause output truncation. If any required phase or evidence is missing, state the gap clearly.
6. CONCISE AND FACTUAL: Be concise, clear, and professional. Avoid conversational filler."""

ABSTENTION_PATTERNS = [
    r"\bcannot (?:be )?answer(?:ed)?\b",
    r"\bnot (?:available|found|provided|mentioned|addressed|stated)\b",
    r"\binsufficient (?:information|evidence|details|context|documentation)\b",
    r"\bno (?:mention|record|evidence|information|documentation)\b",
    r"\bdoes not (?:contain|state|mention|provide|specify)\b",
]

CONTRADICTION_PATTERNS = [
    r"\bconflict(?:ing|s|ed)?\b",
    r"\bdisagree(?:s|ment|d)?\b",
    r"\bcontradict(?:s|ory|ion)?\b",
    r"\bdivergen(?:t|ce)\b",
    r"\bwhereas\b",
    r"\bhowever,?\s+(?:an? (?:earlier|updated|other)|in contrast)\b",
    r"\bin contrast\b",
    r"\bon the other hand\b",
    r"\bsupersed(?:ed|es|ing)\b",
    r"\bdeprecat(?:ed|es|ing)\b",
    r"\bpreviously\b",
    r"\bearlier (?:version|draft|standard|implementation|email|policy|recommendation|guidance|proposal)\b",
    r"\bupdated from\b",
    r"\binterim\b",
    r"\bobsolete\b",
    r"\breplaced by\b",
]


@dataclass
class RAGAnswer:
    """The complete structured answer produced by the Phase 2 RAG pipeline."""

    question: str
    answer_text: str
    is_abstention: bool
    has_contradiction: bool
    retrieved_doc_ids: List[str]
    cited_doc_ids: List[str]
    cited_chunk_ids: List[str]
    citation_validation: CitationValidationResult
    context_tokens: int
    context_passages_count: int
    latency_ms: float
    model: str
    cost_estimate_usd: float = 0.0
    retrieval_ms: float = 0.0
    context_builder_ms: float = 0.0
    ttft_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0
    total_e2e_ms: float = 0.0
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    assembled_context: Optional[AssembledContext] = None
    retrieved_chunks: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "question": self.question,
            "answer_text": self.answer_text,
            "is_abstention": self.is_abstention,
            "has_contradiction": self.has_contradiction,
            "retrieved_doc_ids": self.retrieved_doc_ids,
            "cited_doc_ids": self.cited_doc_ids,
            "cited_chunk_ids": self.cited_chunk_ids,
            "citation_validation": self.citation_validation.to_dict(),
            "context_tokens": self.context_tokens,
            "context_passages_count": self.context_passages_count,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "cost_estimate_usd": self.cost_estimate_usd,
            "retrieval_ms": self.retrieval_ms,
            "context_builder_ms": self.context_builder_ms,
            "ttft_ms": self.ttft_ms,
            "generation_ms": self.generation_ms,
            "validation_ms": self.validation_ms,
            "total_e2e_ms": self.total_e2e_ms,
        }
        if self.user_prompt is not None:
            d["user_prompt"] = self.user_prompt
        return d


def detect_abstention(text: str) -> bool:
    """Detect whether the answer expresses abstention or insufficient evidence."""
    lower = text.lower()
    for pat in ABSTENTION_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def detect_contradiction(text: str) -> bool:
    """Detect whether the answer notes conflicting or diverging information."""
    lower = text.lower()
    for pat in CONTRADICTION_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


class AnswerGenerator:
    """Generates grounded answers from assembled context and validates citations."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        validator: Optional[CitationValidator] = None,
        system_prompt: str = GROUNDED_SYSTEM_PROMPT,
    ) -> None:
        self.llm = llm_client
        self.validator = validator or CitationValidator()
        self.system_prompt = system_prompt

    def generate_answer(
        self,
        question: str,
        assembled_context: AssembledContext,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> RAGAnswer:
        """Generate a grounded, cited answer for question given assembled_context."""
        # Edge case: empty context
        if not assembled_context.passages or not assembled_context.formatted_context.strip():
            empty_answer = (
                "I cannot answer this question because no relevant documents were found "
                "in the enterprise knowledge base."
            )
            t_val0 = time.perf_counter()
            val_res = self.validator.validate(empty_answer, assembled_context)
            val_ms = (time.perf_counter() - t_val0) * 1000.0
            return RAGAnswer(
                question=question,
                answer_text=empty_answer,
                is_abstention=True,
                has_contradiction=False,
                retrieved_doc_ids=[],
                cited_doc_ids=[],
                cited_chunk_ids=[],
                citation_validation=val_res,
                context_tokens=0,
                context_passages_count=0,
                latency_ms=0.0,
                model="system_empty_context",
                cost_estimate_usd=0.0,
                retrieval_ms=0.0,
                context_builder_ms=0.0,
                ttft_ms=0.0,
                generation_ms=0.0,
                validation_ms=round(val_ms, 2),
                total_e2e_ms=round(val_ms, 2),
            )

        user_prompt = (
            f"EVIDENCE PASSAGES:\n"
            f"{assembled_context.formatted_context}\n\n"
            f"QUESTION:\n{question}\n\n"
            f"INSTRUCTIONS: Answer the question using ONLY the evidence passages above. Cite supporting documents as [Doc ID].\n"
            f"- If the evidence shows conflicting or superseded guidance, state the final policy citing the final document, and explicitly describe what earlier draft was superseded citing the original earlier document (do not cite the final document for the obsolete instruction).\n"
            f"- When answering SOP or multi-phase process questions, cover all operational phases systematically using concise numbered steps and bullet points. DO NOT use markdown tables.\n\n"
            f"ANSWER:"
        )

        t_gen0 = time.perf_counter()
        llm_resp: LLMResponse = self.llm.generate(
            prompt=user_prompt,
            system_prompt=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            on_token=on_token,
        )
        generation_ms = (time.perf_counter() - t_gen0) * 1000.0

        answer_text = llm_resp.text.strip()
        if llm_resp.finish_reason == "length":
            answer_text += (
                "\n\n[TRUNCATION DETECTED: The generated response reached the maximum output token limit. "
                "Subsequent sections or phases may have been truncated due to length constraints.]"
            )

        is_abs = detect_abstention(answer_text)
        has_contra = detect_contradiction(answer_text)

        t_val0 = time.perf_counter()
        val_result = self.validator.validate(
            answer_text, assembled_context, is_abstention=is_abs
        )
        validation_ms = (time.perf_counter() - t_val0) * 1000.0

        return RAGAnswer(
            question=question,
            answer_text=answer_text,
            is_abstention=is_abs,
            has_contradiction=has_contra,
            retrieved_doc_ids=assembled_context.unique_doc_ids,
            cited_doc_ids=val_result.valid_doc_ids,
            cited_chunk_ids=val_result.cited_chunk_ids,
            citation_validation=val_result,
            context_tokens=assembled_context.total_tokens,
            context_passages_count=len(assembled_context.passages),
            latency_ms=llm_resp.latency_ms,
            model=llm_resp.model,
            cost_estimate_usd=llm_resp.cost_estimate_usd,
            ttft_ms=llm_resp.ttft_ms,
            generation_ms=round(generation_ms, 2),
            validation_ms=round(validation_ms, 2),
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            assembled_context=assembled_context,
        )
