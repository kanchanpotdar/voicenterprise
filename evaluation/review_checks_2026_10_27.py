"""Independent review probes. Offline, synthetic data only; no project artifacts changed.

Run from the repository root:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m evaluation.review_checks_2026_10_27

The output records observed bugs, rather than asserting that current behavior is correct.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
import time

from evaluation.audit_scoring import score_answer
from evaluation.run_controlled_generation_benchmark import generation_cache_key
from rag.citation_validator import CitationValidator
from rag.context_builder import AssembledContext, ContextBuilder, EvidencePassage
import rag.context_builder as context_module
from rag.verifier import AnswerVerifier
from retrieval.indexer import RetrievedChunk
from retrieval.query_expansion import MultiQueryRetriever, QueryExpansionResult
from voice.voice_pipeline import VoiceRAGPipeline


def main() -> None:
    results = {}
    params = dict(model="synthetic", system_prompt="system", user_prompt="prompt", temperature=0.0)
    results["cache_key_ignores_max_tokens"] = (
        generation_cache_key(**params, max_tokens=32) == generation_cache_key(**params, max_tokens=4096)
    )
    score = score_answer(
        "synthetic", "basic",
        "The production access expires after 14 business days.\n[TRUNCATION DETECTED: output cut off]",
        ["The production access expires after 14 business days."],
    )
    results["truncated_answer"] = {"truncated": score.truncated, "marked_correct": score.answer_correct_strict}
    score = score_answer(
        "synthetic", "info_not_found",
        "The undisclosed secret code is 123456. The owner is not specified.", [],
    )
    results["fabrication_plus_unrelated_caveat_marked_correct"] = score.answer_correct_strict

    doc_id = "dsid_" + "a" * 32
    passage = EvidencePassage(
        1, doc_id, doc_id + "__c000", "confluence", "Policy", "Limits",
        "The policy allows 10 requests and 20 users.", 1.0,
    )
    context = AssembledContext(passage.to_formatted_passage(), [passage], 100, unique_doc_ids=[doc_id])
    validator = CitationValidator()
    validation = validator.validate(f"The policy allows 10 requests and 999 users [{doc_id}].", context)
    results["wrong_number_marked_supported"] = validation.supported_sentences_count == 1
    validation = validator.validate(f"The policy allows 10 requests [{doc_id}__c000].", context)
    results["chunk_citation"] = {
        "valid_doc_ids": validation.valid_doc_ids, "cited_sentence_count": validation.cited_sentences_count,
    }
    validation = validator.validate(f"The policy allows 10 requests [{doc_id}__v1__c000].", context)
    results["versioned_chunk_citation_count"] = validation.total_citations_found
    results["empty_answer_marked_verified"] = AnswerVerifier().verify("How many users?", "", context).is_verified

    chunk = RetrievedChunk(
        doc_id + "__c000", doc_id, doc_id, 10.0, "confluence", "Header " * 80,
        "Section", "body " * 100, None, "none", [], [],
    )
    for tokenizer_mode in ("cached_tokenizer", "fallback"):
        original_tokenizer = context_module._TOKENIZER
        try:
            if tokenizer_mode == "fallback":
                context_module._TOKENIZER = None
                context_module.estimate_tokens.cache_clear()
            assembled = ContextBuilder(max_context_tokens=50).build_context([chunk])
            results[f"budget_{tokenizer_mode}"] = {"budget": 50, "actual": assembled.total_tokens}
        finally:
            context_module._TOKENIZER = original_tokenizer
            context_module.estimate_tokens.cache_clear()

    def make_chunk(name, score):
        return RetrievedChunk(
            name + "__c000", name, name, score, "confluence", "Title", "Section",
            "The documented answer is available here.", None, "none", [], [],
        )

    class Expander:
        def expand(self, *args, **kwargs):
            return QueryExpansionResult(
                original_query="original", mode="alternative_queries", expanded_queries=["expanded"],
            )

    class Retriever:
        def retrieve_documents(self, query, **kwargs):
            if query == "original":
                return ["A", "B"], [make_chunk("A", 10), make_chunk("B", 9)]
            return ["B", "C"], [make_chunk("B", 900), make_chunk("C", 1000)]

    documents, chunks, *_ = MultiQueryRetriever(Retriever(), Expander()).retrieve("original")
    results["multiquery_rank_lost"] = {
        "fused_order": documents, "context_order": ContextBuilder().build_context(chunks).unique_doc_ids,
    }

    class Pipeline:
        def query(self, **kwargs):
            kwargs["on_token"]("The first sentence has arrived. Next sentence ")
            time.sleep(0.12)
            return SimpleNamespace(
                retrieval_ms=0, context_builder_ms=0, ttft_ms=0, generation_ms=120,
                answer_text="The first sentence has arrived. Next sentence completed.",
            )

    class STT:
        def transcribe_audio_file(self, *args):
            return "query", 0

    class TTS:
        def synthesize_to_file(self, *args):
            return "synthetic-no-file-written", 1

    response = VoiceRAGPipeline(Pipeline(), STT(), TTS()).process_voice_query("synthetic-no-file")
    results["voice_timing"] = {
        "reported_ttfa_ms": round(response.metrics.total_ttfa_ms, 2),
        "actual_pipeline_ms": round(response.metrics.total_e2e_ms, 2),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
