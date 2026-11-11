"""Controlled Answer-Generation Benchmark: Baseline BM25 (A) vs Corroborated Hybrid + Updated ContextBuilder (E).

Protocol & Safeguards:
1. Uses DEVELOPMENT split only (`data/splits/questions_dev.parquet`). Never touches held-out test data.
2. Evaluates the deduplicated union of:
   - The deterministic 40-question stratified development sample (`get_stratified_sample(df_all, 40)`,
     4 questions per category across all 10 categories, including `info_not_found`).
   - The 8 diagnostic questions (`qst_0427`, `qst_0214`, `qst_0203`, `qst_0282`, `qst_0436`,
     `qst_0437`, `qst_0447`, `qst_0448`).
   - Overlap (`qst_0436`) is deduplicated so each question is evaluated and counted once (`n = 47` unique).
3. Compares two controlled configurations with identical generator model, prompts, temperature (0.0),
   max_tokens (2048), context budget (2500), and scoring procedure:
   - `A_Baseline_BM25_k20`: BM25 (`top_k_docs=20`, `top_k_chunks=80`) + standard `ContextBuilder` (`budget=2500`).
   - `E_CorroboratedHybrid_k40_UpdatedCB`: Corroborated Hybrid RRF (`fuse_bm25_and_dense_chunks`,
     `top_k_docs=40`, `top_k_chunks=160`) + updated `ContextBuilder` (`budget=2500`).
4. Explicitly configures the generator model (no silent model switching).
5. Caches validated LLM responses in `evaluation/reports/cache/controlled_generation_cache.json`
   keyed by `(model, system_prompt_sha256, user_prompt_sha256, temperature, max_tokens)`.
   Records model, prompt hash, cache status, input/output tokens, latency, estimated cost, and API errors.
6. Separates retrieval/prompt-evidence metrics from heuristic answer-quality and citation-grounding metrics.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from evaluation.audit_scoring import (
    AuditScore,
    abstains,
    evidence_fact_coverage,
    is_empty_or_truncated,
    score_answer,
)
from evaluation.eval_rag import DeterministicJudge, split_facts_and_constraints
from evaluation.evaluation_harness import load_development_benchmark
from evaluation.inspect_p1_checks import get_stratified_sample
from evaluation.metrics import hit_at_k, overall_answer_score, recall_at_k
from rag.citation_validator import CitationValidator
from rag.context_builder import AssembledContext, ContextBuilder, estimate_tokens
from rag.generator import GROUNDED_SYSTEM_PROMPT, AnswerGenerator, detect_abstention, detect_contradiction
from rag.llm import LLMResponse, OpenAICompatibleLLM, PRICING_PER_1K
from retrieval.indexer import RetrievedChunk, fuse_bm25_and_dense_chunks
from retrieval.query_expansion import ExpansionCache, QueryExpander

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "evaluation" / "reports"
CACHE_DIR = REPORTS_DIR / "cache"
RETRIEVAL_CACHE_PATH = CACHE_DIR / "offline_retrieval_chunks.json.gz"
GEN_CACHE_PATH = CACHE_DIR / "controlled_generation_cache.json"
EXP_SMOKE_CACHE_PATH = CACHE_DIR / "query_expansion_smoke_cache.json"
SMOKE_REPORT_PATH = REPORTS_DIR / "controlled_generation_smoke_report.json"
FULL_REPORT_PATH = REPORTS_DIR / "controlled_generation_benchmark_47q.json"

DIAGNOSTIC_8_QIDS = [
    "qst_0427",
    "qst_0214",
    "qst_0203",
    "qst_0282",
    "qst_0436",
    "qst_0437",
    "qst_0447",
    "qst_0448",
]

SMOKE_QIDS = [
    "qst_0427",  # Conflicting info (ClearEdge PO fingerprint SHA-256 vs unhashed Gmail draft)
    "qst_0447",  # Multi-doc completeness (6 gold docs, 10-phase emergency hotfix SOP)
    "qst_0481",  # Unanswerable / info_not_found (appropriate abstention check)
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generation_cache_key(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int = 4096,
) -> str:
    payload = json.dumps(
        {
            "v": 1,
            "model": model,
            "sys_sha": _sha256(system_prompt),
            "usr_sha": _sha256(user_prompt),
            "temperature": float(temperature),
        },
        sort_keys=True,
    )
    return _sha256(payload)


class GenerationResponseCache:
    """Persistent atomic JSON cache of validated LLM generation responses."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
            except Exception:
                self._data = {}

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        entry = self._data.get(key)
        if not isinstance(entry, dict):
            self.misses += 1
            return None
        text = str(entry.get("text", "")).strip()
        flags = is_empty_or_truncated(text)
        if flags["empty"] or flags["truncated"] or entry.get("finish_reason") == "length":
            self.misses += 1
            return None
        self.hits += 1
        return entry

    def put(self, key: str, resp: LLMResponse, user_prompt_sha: str, sys_prompt_sha: str) -> bool:
        flags = is_empty_or_truncated(resp.text)
        if flags["empty"] or flags["truncated"] or resp.finish_reason == "length":
            return False
        self._data[key] = {
            "v": 1,
            "text": resp.text,
            "model": resp.model,
            "prompt_tokens": int(resp.prompt_tokens),
            "completion_tokens": int(resp.completion_tokens),
            "latency_ms": float(resp.latency_ms),
            "ttft_ms": float(resp.ttft_ms),
            "finish_reason": str(resp.finish_reason),
            "cost_estimate_usd": float(resp.cost_estimate_usd),
            "user_prompt_sha": user_prompt_sha,
            "sys_prompt_sha": sys_prompt_sha,
            "cached_at_epoch": round(time.time(), 2),
        }
        self.writes += 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        return True


def build_deduplicated_question_subset(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the deduplicated union of the 40-question stratified sample and 8 diagnostic questions."""
    strat_40 = get_stratified_sample(df_all, num_samples=40)
    strat_40_qids = list(strat_40["question_id"].astype(str))
    diag_8_qids = list(DIAGNOSTIC_8_QIDS)

    overlap_qids = [q for q in diag_8_qids if q in set(strat_40_qids)]
    combined_qids = list(dict.fromkeys(strat_40_qids + diag_8_qids))

    df_by_qid = df_all.set_index("question_id")
    subset_df = df_by_qid.loc[combined_qids].reset_index()

    metadata = {
        "stratified_40_count": len(strat_40_qids),
        "diagnostic_8_count": len(diag_8_qids),
        "overlap_count": len(overlap_qids),
        "overlap_question_ids": overlap_qids,
        "total_deduplicated_questions": len(combined_qids),
        "stratified_40_question_ids": strat_40_qids,
        "diagnostic_8_question_ids": diag_8_qids,
        "deduplicated_question_ids": combined_qids,
        "category_counts_in_deduplicated_47": subset_df["question_type"].value_counts().to_dict(),
    }
    return subset_df, metadata


def get_retrieval_stream_from_cache(
    qid: str,
    config_name: str,
    retrieval_cache: Dict[str, Any],
) -> Tuple[List[str], List[RetrievedChunk], int]:
    """Return `(pool_doc_ids, chunks, top_k_docs)` for Configuration A or Configuration E."""
    entry = retrieval_cache[qid]
    if config_name == "A_Baseline_BM25_k20":
        top_k = 20
        doc_ids = list(entry["bm25_k20"]["doc_ids"])[:top_k]
        chunks = [RetrievedChunk(**c) for c in entry["bm25_k20"]["chunks"]]
        return doc_ids, chunks, top_k
    elif config_name == "E_CorroboratedHybrid_k40_UpdatedCB":
        top_k = 40
        bm25_chunks = [RetrievedChunk(**c) for c in entry["bm25_k40"]["chunks"]]
        dense_chunks = [RetrievedChunk(**c) for c in entry["dense_k40"]["chunks"]]
        fused_chunks = fuse_bm25_and_dense_chunks(
            bm25_chunks=bm25_chunks,
            dense_chunks=dense_chunks,
            top_k_chunks=160,
        )
        seen_docs: List[str] = []
        for ch in fused_chunks:
            if ch.doc_id not in seen_docs:
                seen_docs.append(ch.doc_id)
                if len(seen_docs) >= top_k:
                    break
        return seen_docs, fused_chunks, top_k
    else:
        raise ValueError(f"Unsupported configuration: {config_name}")


def classify_failure_category(
    question_type: str,
    gold_docs: List[str],
    pool_doc_ids: List[str],
    prompt_doc_ids: List[str],
    audit_score: AuditScore,
    has_hallucinated_citations: bool,
) -> str:
    """Classify the primary failure category when `answer_correct_strict` is False."""
    if audit_score.answer_correct_strict:
        return "none"
    if audit_score.empty or audit_score.truncated:
        return "generation_empty_or_truncated"
    if question_type == "info_not_found":
        return "failed_to_abstain_on_unanswerable"
    if audit_score.abstains:
        gold_set = set(gold_docs)
        if gold_set and not (gold_set & set(pool_doc_ids)):
            return "retrieval_miss"
        if gold_set and not (gold_set & set(prompt_doc_ids)):
            return "context_packing_loss"
        return "false_abstention_on_answerable"
    if audit_score.suspected_constraint_violations:
        return "contradiction_or_prohibited_violation"
    gold_set = set(gold_docs)
    if gold_set and not (gold_set <= set(pool_doc_ids)):
        return "retrieval_miss"
    if gold_set and not (gold_set <= set(prompt_doc_ids)):
        return "context_packing_loss"
    if has_hallucinated_citations:
        return "generator_hallucinated_citation"
    return "missing_fact_synthesis"


def verify_query_expansion_reporting(llm_client: OpenAICompatibleLLM) -> Dict[str, Any]:
    """Run a bounded 2-query check verifying QueryExpander status, cache hits, and fallback reporting."""
    if EXP_SMOKE_CACHE_PATH.exists():
        EXP_SMOKE_CACHE_PATH.unlink()
    exp_cache = ExpansionCache(path=EXP_SMOKE_CACHE_PATH)
    expander = QueryExpander(
        llm_client=llm_client,
        enabled=True,
        default_mode="terminology",
        max_expansions=3,
        cache=exp_cache,
        strict=False,
    )
    q_test = "Why would a customer's session stay locked to a faulty node?"
    # 1. Live call or cache miss -> write
    r1 = expander.expand(q_test, mode="terminology")
    # 2. Second identical call -> must be a cache_hit without a second LLM call
    r2 = expander.expand(q_test, mode="terminology")
    # 3. Disabled call -> explicit 'disabled' status
    r3 = expander.expand(q_test, mode="none")

    return {
        "model": llm_client.model,
        "first_call_status": r1.status,
        "first_call_cache_status": r1.cache_status,
        "first_call_failed": r1.failed,
        "first_call_fell_back": r1.fell_back_to_original,
        "first_call_expanded_queries": r1.expanded_queries,
        "second_call_status": r2.status,
        "second_call_cache_status": r2.cache_status,
        "disabled_call_status": r3.status,
        "expander_stats": expander.stats_report(),
        "cache_stats": {
            "entries": len(exp_cache),
            "hits": exp_cache.hits,
            "misses": exp_cache.misses,
            "writes": exp_cache.writes,
            "rejected": exp_cache.rejected,
        },
    }


def evaluate_single_question_config(
    row: Any,
    config_name: str,
    retrieval_cache: Dict[str, Any],
    context_builder: ContextBuilder,
    llm_client: OpenAICompatibleLLM,
    gen_cache: GenerationResponseCache,
    validator: CitationValidator,
    det_judge: DeterministicJudge,
    in_stratified_40: bool,
    in_diagnostic_8: bool,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    cache_only: bool = False,
) -> Dict[str, Any]:
    qid = str(row.question_id)
    qtype = str(row.question_type)
    question = str(row.question)
    gold_answer = str(getattr(row, "gold_answer", "") or "")
    raw_gold = getattr(row, "expected_doc_ids", None)
    gold_docs = list(dict.fromkeys(str(x) for x in raw_gold)) if raw_gold is not None and len(raw_gold) > 0 else []
    raw_facts = getattr(row, "answer_facts", None)
    facts = [str(x) for x in raw_facts] if raw_facts is not None and len(raw_facts) > 0 else []
    positive_facts, prohibited_constraints = split_facts_and_constraints(facts)

    # 1. Retrieval & Context Assembly (deterministic from cached local Qdrant streams)
    t_ctx0 = time.perf_counter()
    pool_doc_ids, retrieved_chunks, top_k_docs = get_retrieval_stream_from_cache(
        qid=qid,
        config_name=config_name,
        retrieval_cache=retrieval_cache,
    )
    assembled: AssembledContext = context_builder.build_context(retrieved_chunks=retrieved_chunks)
    context_ms = (time.perf_counter() - t_ctx0) * 1000.0

    # Retrieval & Prompt Evidence Metrics (independent of generation)
    gold_set = set(gold_docs)
    in_pool = gold_set & set(pool_doc_ids)
    in_prompt = gold_set & set(assembled.unique_doc_ids)
    pool_recall = (len(in_pool) / len(gold_set)) if gold_set else None
    pool_hit = (1.0 if in_pool else 0.0) if gold_set else None
    prompt_recall = (len(in_prompt) / len(gold_set)) if gold_set else None
    prompt_complete = (len(in_prompt) == len(gold_set)) if gold_set else None
    lost_in_packing = sorted(in_pool - in_prompt) if gold_set else []
    ev_fact_cov = evidence_fact_coverage(facts, assembled.formatted_context)
    ev_fact_cov_val = round(ev_fact_cov, 4) if (ev_fact_cov == ev_fact_cov) else None
    actual_fmt_tokens = estimate_tokens(assembled.formatted_context) if assembled.formatted_context else 0

    # 2. Construct exact prompt identical to AnswerGenerator.generate_answer
    user_prompt = (
        f"EVIDENCE PASSAGES:\n"
        f"{assembled.formatted_context}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"INSTRUCTIONS: Answer the question using ONLY the evidence passages above. Cite supporting documents as [Doc ID].\n"
        f"- If the evidence shows conflicting or superseded guidance, state the final policy citing the final document, and explicitly describe what earlier draft was superseded citing the original earlier document (do not cite the final document for the obsolete instruction).\n"
        f"- When answering SOP or multi-phase process questions, cover all operational phases systematically using concise numbered steps and bullet points. DO NOT use markdown tables.\n\n"
        f"ANSWER:"
    )
    sys_prompt = GROUNDED_SYSTEM_PROMPT
    usr_sha = _sha256(user_prompt)
    sys_sha = _sha256(sys_prompt)
    c_key = generation_cache_key(
        model=llm_client.model,
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    cached_entry = gen_cache.get(c_key)
    api_error: Optional[str] = None
    if cached_entry is not None:
        cache_status = "hit"
        raw_text = str(cached_entry["text"])
        resp_model = str(cached_entry["model"])
        prompt_tokens = int(cached_entry["prompt_tokens"])
        completion_tokens = int(cached_entry["completion_tokens"])
        gen_latency_ms = float(cached_entry["latency_ms"])
        finish_reason = str(cached_entry.get("finish_reason", "stop"))
        cost_usd = float(cached_entry.get("cost_estimate_usd", 0.0))
    elif cache_only:
        cache_status = "skipped_quota_or_cache_only"
        api_error = "TPD quota reached for openai/gpt-oss-120b (200,000 TPD limit); skipped live API call."
        raw_text = ""
        resp_model = llm_client.model
        prompt_tokens = 0
        completion_tokens = 0
        gen_latency_ms = 0.0
        finish_reason = "skipped_quota"
        cost_usd = 0.0
    else:
        cache_status = "miss"
        try:
            llm_resp = llm_client.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            raw_text = llm_resp.text.strip()
            resp_model = llm_resp.model
            prompt_tokens = llm_resp.prompt_tokens
            completion_tokens = llm_resp.completion_tokens
            gen_latency_ms = llm_resp.latency_ms
            finish_reason = llm_resp.finish_reason
            cost_usd = llm_resp.cost_estimate_usd
            stored = gen_cache.put(c_key, llm_resp, user_prompt_sha=usr_sha, sys_prompt_sha=sys_sha)
            cache_status = "miss_write" if stored else "miss_not_stored"
        except Exception as e:
            api_error = str(e)[:400]
            raw_text = ""
            resp_model = llm_client.model
            prompt_tokens = 0
            completion_tokens = 0
            gen_latency_ms = 0.0
            finish_reason = "error"
            cost_usd = 0.0

    answer_text = raw_text
    if finish_reason == "length":
        answer_text += (
            "\n\n[TRUNCATION DETECTED: The generated response reached the maximum output token limit. "
            "Subsequent sections or phases may have been truncated due to length constraints.]"
        )

    # 3. Validate citations & score answer independently of retrieval
    has_contra = detect_contradiction(answer_text)
    audit_sc: AuditScore = score_answer(
        question_id=qid,
        question_type=qtype,
        answer_text=answer_text,
        answer_facts=facts,
        assembled_context=assembled,
        validator=validator,
    )
    is_abs = audit_sc.abstains
    val_res = validator.validate(answer_text, assembled, is_abstention=is_abs)

    judge_correct, judge_comp_pct, judge_expl = det_judge.evaluate(
        question=question,
        question_type=qtype,
        generated_answer=answer_text,
        gold_answer=gold_answer,
        answer_facts=facts,
        is_abstention=is_abs,
        has_contradiction=has_contra,
    )

    strict_comp = 0.0 if (audit_sc.fact_completeness_pct != audit_sc.fact_completeness_pct) else float(audit_sc.fact_completeness_pct)
    strict_overall = overall_answer_score(audit_sc.answer_correct_strict, strict_comp)
    judge_overall = overall_answer_score(judge_correct, judge_comp_pct)

    failure_category = (
        "api_error"
        if api_error
        else classify_failure_category(
            question_type=qtype,
            gold_docs=gold_docs,
            pool_doc_ids=pool_doc_ids,
            prompt_doc_ids=assembled.unique_doc_ids,
            audit_score=audit_sc,
            has_hallucinated_citations=val_res.has_hallucinated_citations,
        )
    )

    return {
        "question_id": qid,
        "question_type": qtype,
        "question": question,
        "in_stratified_40": in_stratified_40,
        "in_diagnostic_8": in_diagnostic_8,
        "config_name": config_name,
        "retrieval_and_context": {
            "top_k_docs": top_k_docs,
            "gold_docs_count": len(gold_set),
            "gold_docs": gold_docs,
            "pool_gold_count": len(in_pool),
            "prompt_gold_count": len(in_prompt),
            "pool_recall_at_k": round(pool_recall, 4) if pool_recall is not None else None,
            "pool_hit_at_k": round(pool_hit, 4) if pool_hit is not None else None,
            "prompt_gold_recall": round(prompt_recall, 4) if prompt_recall is not None else None,
            "prompt_complete_gold": prompt_complete,
            "pool_gold_lost_in_packing": lost_in_packing,
            "evidence_fact_coverage": ev_fact_cov_val,
            "distinct_docs_in_prompt": assembled.distinct_docs_count,
            "passages_in_prompt": len(assembled.passages),
            "prompt_context_tokens": actual_fmt_tokens,
            "prompt_doc_ids": assembled.unique_doc_ids,
            "context_assembly_ms": round(context_ms, 2),
        },
        "generation_and_usage": {
            "model": resp_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt_sha256": sys_sha[:16],
            "user_prompt_sha256": usr_sha[:16],
            "cache_key": c_key[:16],
            "cache_status": cache_status,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "finish_reason": finish_reason,
            "latency_ms": round(gen_latency_ms, 2),
            "cost_estimate_usd": round(cost_usd, 6),
            "api_error": api_error,
        },
        "answer_quality_heuristic": {
            "scoring_label": "automated_lexical_anchor_heuristic_not_human_verified",
            "answer_correct_strict": audit_sc.answer_correct_strict,
            "fact_completeness_pct": round(strict_comp, 2),
            "overall_score_strict": round(strict_overall, 2),
            "judge_answer_correct": judge_correct,
            "judge_completeness_pct": round(judge_comp_pct, 2),
            "overall_score_judge": round(judge_overall, 2),
            "judge_explanation": judge_expl,
            "positive_facts_total": len(positive_facts),
            "positive_facts_covered": len(audit_sc.covered_facts),
            "missing_facts": audit_sc.missing_facts,
            "prohibited_constraints_total": len(prohibited_constraints),
            "suspected_constraint_violations": audit_sc.suspected_constraint_violations,
            "has_prohibited_violation": len(audit_sc.suspected_constraint_violations) > 0,
            "empty_response": audit_sc.empty,
            "truncated_response": audit_sc.truncated,
            "abstains": audit_sc.abstains,
            "has_contradiction_cue": has_contra,
            "failure_category": failure_category,
        },
        "citation_and_grounding_heuristic": {
            "total_citations_found": val_res.total_citations_found,
            "cited_doc_ids": val_res.valid_doc_ids,
            "hallucinated_doc_ids": val_res.invalid_doc_ids,
            "has_hallucinated_citations": val_res.has_hallucinated_citations,
            "citation_precision": round(val_res.citation_precision, 4),
            "citation_id_validity": round(val_res.citation_id_validity, 4),
            "sentence_citation_rate": round(val_res.sentence_citation_rate, 4),
            "claim_support_rate_all_sentences": round(val_res.claim_support_rate, 4),
            "unsupported_substantive_sentence_rate": (
                round(audit_sc.unsupported_sentence_rate, 4)
                if audit_sc.unsupported_sentence_rate is not None
                else None
            ),
            "num_substantive_sentences": audit_sc.num_substantive_sentences,
            "has_unsourced_claims": val_res.has_unsourced_claims,
        },
        "answer_text": answer_text,
    }


def aggregate_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute separate retrieval/context, answer-quality, citation, and API-usage aggregates."""
    n = len(records)
    if n == 0:
        return {}

    with_gold = [r for r in records if r["retrieval_and_context"]["pool_recall_at_k"] is not None]
    multi_gold = [r for r in with_gold if r["retrieval_and_context"]["gold_docs_count"] > 1]
    ev_vals = [
        r["retrieval_and_context"]["evidence_fact_coverage"]
        for r in records
        if r["retrieval_and_context"]["evidence_fact_coverage"] is not None
    ]
    answerable = [r for r in records if r["question_type"] != "info_not_found"]
    unanswerable = [r for r in records if r["question_type"] == "info_not_found"]
    unsup_vals = [
        r["citation_and_grounding_heuristic"]["unsupported_substantive_sentence_rate"]
        for r in records
        if r["citation_and_grounding_heuristic"]["unsupported_substantive_sentence_rate"] is not None
    ]

    failure_counts: Dict[str, int] = {}
    for r in records:
        fc = r["answer_quality_heuristic"]["failure_category"]
        failure_counts[fc] = failure_counts.get(fc, 0) + 1

    cache_counts: Dict[str, int] = {}
    for r in records:
        cs = r["generation_and_usage"]["cache_status"]
        cache_counts[cs] = cache_counts.get(cs, 0) + 1

    return {
        "num_questions": n,
        "num_answerable": len(answerable),
        "num_answerable_with_gold_docs": len(with_gold),
        "num_multi_doc_with_gold": len(multi_gold),
        "num_unanswerable_info_not_found": len(unanswerable),
        "retrieval_and_prompt_coverage": {
            "pool_recall_at_k_pct": (
                round(100.0 * sum(r["retrieval_and_context"]["pool_recall_at_k"] for r in with_gold) / len(with_gold), 2)
                if with_gold
                else None
            ),
            "pool_hit_at_k_pct": (
                round(100.0 * sum(r["retrieval_and_context"]["pool_hit_at_k"] for r in with_gold) / len(with_gold), 2)
                if with_gold
                else None
            ),
            "prompt_gold_recall_pct": (
                round(100.0 * sum(r["retrieval_and_context"]["prompt_gold_recall"] for r in with_gold) / len(with_gold), 2)
                if with_gold
                else None
            ),
            "prompt_complete_gold_pct": (
                round(100.0 * sum(1 for r in with_gold if r["retrieval_and_context"]["prompt_complete_gold"]) / len(with_gold), 2)
                if with_gold
                else None
            ),
            "multi_doc_prompt_gold_recall_pct": (
                round(100.0 * sum(r["retrieval_and_context"]["prompt_gold_recall"] for r in multi_gold) / len(multi_gold), 2)
                if multi_gold
                else None
            ),
            "multi_doc_prompt_complete_gold_pct": (
                round(100.0 * sum(1 for r in multi_gold if r["retrieval_and_context"]["prompt_complete_gold"]) / len(multi_gold), 2)
                if multi_gold
                else None
            ),
            "questions_losing_pool_gold_in_packing": sum(
                1 for r in with_gold if len(r["retrieval_and_context"]["pool_gold_lost_in_packing"]) > 0
            ),
            "mean_evidence_fact_coverage_pct": (
                round(100.0 * sum(ev_vals) / len(ev_vals), 2) if ev_vals else None
            ),
            "mean_distinct_docs_in_prompt": round(
                sum(r["retrieval_and_context"]["distinct_docs_in_prompt"] for r in records) / n, 2
            ),
            "mean_prompt_context_tokens": round(
                sum(r["retrieval_and_context"]["prompt_context_tokens"] for r in records) / n, 1
            ),
        },
        "answer_quality_heuristic": {
            "note": "Automated lexical and numeric/identifier anchor heuristic scores (not human-verified semantic entailment).",
            "answer_correct_strict_pct": round(
                100.0 * sum(1 for r in records if r["answer_quality_heuristic"]["answer_correct_strict"]) / n, 2
            ),
            "answerable_correct_strict_pct": (
                round(
                    100.0 * sum(1 for r in answerable if r["answer_quality_heuristic"]["answer_correct_strict"]) / len(answerable),
                    2,
                )
                if answerable
                else None
            ),
            "fact_completeness_pct": round(
                sum(r["answer_quality_heuristic"]["fact_completeness_pct"] for r in records) / n, 2
            ),
            "answerable_fact_completeness_pct": (
                round(
                    sum(r["answer_quality_heuristic"]["fact_completeness_pct"] for r in answerable) / len(answerable),
                    2,
                )
                if answerable
                else None
            ),
            "overall_score_strict": round(
                sum(r["answer_quality_heuristic"]["overall_score_strict"] for r in records) / n, 2
            ),
            "judge_answer_correct_pct": round(
                100.0 * sum(1 for r in records if r["answer_quality_heuristic"]["judge_answer_correct"]) / n, 2
            ),
            "judge_completeness_pct": round(
                sum(r["answer_quality_heuristic"]["judge_completeness_pct"] for r in records) / n, 2
            ),
            "overall_score_judge": round(
                sum(r["answer_quality_heuristic"]["overall_score_judge"] for r in records) / n, 2
            ),
            "prohibited_violation_count": sum(
                1 for r in records if r["answer_quality_heuristic"]["has_prohibited_violation"]
            ),
            "prohibited_violation_rate_pct": round(
                100.0 * sum(1 for r in records if r["answer_quality_heuristic"]["has_prohibited_violation"]) / n, 2
            ),
            "empty_response_count": sum(1 for r in records if r["answer_quality_heuristic"]["empty_response"]),
            "truncated_response_count": sum(1 for r in records if r["answer_quality_heuristic"]["truncated_response"]),
            "appropriate_abstention_count_on_info_not_found": sum(
                1 for r in unanswerable if r["answer_quality_heuristic"]["abstains"]
            ),
            "appropriate_abstention_rate_on_info_not_found_pct": (
                round(100.0 * sum(1 for r in unanswerable if r["answer_quality_heuristic"]["abstains"]) / len(unanswerable), 2)
                if unanswerable
                else None
            ),
            "false_abstention_count_on_answerable": sum(
                1 for r in answerable if r["answer_quality_heuristic"]["abstains"]
            ),
            "failure_category_counts": failure_counts,
        },
        "citation_and_grounding_heuristic": {
            "citation_precision_pct": round(
                100.0 * sum(r["citation_and_grounding_heuristic"]["citation_precision"] for r in records) / n, 2
            ),
            "citation_id_validity_pct": round(
                100.0 * sum(r["citation_and_grounding_heuristic"]["citation_id_validity"] for r in records) / n, 2
            ),
            "hallucinated_citation_question_count": sum(
                1 for r in records if r["citation_and_grounding_heuristic"]["has_hallucinated_citations"]
            ),
            "sentence_citation_rate_pct": round(
                100.0 * sum(r["citation_and_grounding_heuristic"]["sentence_citation_rate"] for r in records) / n, 2
            ),
            "claim_support_rate_all_sentences_pct": round(
                100.0 * sum(r["citation_and_grounding_heuristic"]["claim_support_rate_all_sentences"] for r in records) / n, 2
            ),
            "unsupported_substantive_claim_rate_pct": (
                round(100.0 * sum(unsup_vals) / len(unsup_vals), 2) if unsup_vals else 0.0
            ),
            "unsourced_claim_question_count": sum(
                1 for r in records if r["citation_and_grounding_heuristic"]["has_unsourced_claims"]
            ),
        },
        "api_usage_and_caching": {
            "cache_status_counts": cache_counts,
            "api_error_count": sum(1 for r in records if r["generation_and_usage"]["api_error"] is not None),
            "total_prompt_tokens": sum(r["generation_and_usage"]["prompt_tokens"] for r in records),
            "total_completion_tokens": sum(r["generation_and_usage"]["completion_tokens"] for r in records),
            "total_tokens": sum(r["generation_and_usage"]["total_tokens"] for r in records),
            "total_estimated_cost_usd": round(
                sum(r["generation_and_usage"]["cost_estimate_usd"] for r in records), 6
            ),
            "mean_generation_latency_ms": round(
                sum(r["generation_and_usage"]["latency_ms"] for r in records) / n, 1
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled answer-generation benchmark (A vs E).")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Explicit generator model identifier (no silent fallback).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the 3-question smoke test + query expansion status verification first.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max completion tokens per answer.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Score all cached generation responses and all 47 retrieval/context records without making new LLM API calls.",
    )
    args = parser.parse_args()

    t_start = time.time()
    with gzip.open(RETRIEVAL_CACHE_PATH, "rt", encoding="utf-8") as f:
        retrieval_cache = json.load(f)

    df_all, _, _ = load_development_benchmark()
    subset_df, subset_meta = build_deduplicated_question_subset(df_all)

    strat_40_set = set(subset_meta["stratified_40_question_ids"])
    diag_8_set = set(subset_meta["diagnostic_8_question_ids"])

    if args.smoke:
        eval_df = df_all[df_all["question_id"].isin(SMOKE_QIDS)].copy()
        eval_df["sort_order"] = eval_df["question_id"].map(lambda x: SMOKE_QIDS.index(x))
        eval_df = eval_df.sort_values("sort_order").reset_index(drop=True)
    else:
        eval_df = subset_df

    # Explicit LLM client — no silent model switching
    llm_client = OpenAICompatibleLLM(model=args.model, timeout_seconds=90.0)
    if llm_client.model != args.model:
        raise RuntimeError(f"Silent model switch detected: requested {args.model}, got {llm_client.model}")

    print(f"Explicit Generator Model : {llm_client.model} (base_url={llm_client.base_url})")
    print(f"Evaluation Mode          : {'SMOKE TEST (3 questions)' if args.smoke else 'FULL CONTROLLED BENCHMARK (47 deduplicated questions)'}")

    exp_verification = None
    if args.smoke:
        print("\n--- Verifying QueryExpander Status, Caching & Fallback Reporting ---")
        exp_verification = verify_query_expansion_reporting(llm_client)
        print(json.dumps(exp_verification, indent=2))

    context_builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3, min_distinct_docs=2, max_primary_docs=7)
    gen_cache = GenerationResponseCache(GEN_CACHE_PATH)
    validator = CitationValidator()
    det_judge = DeterministicJudge()

    configs = [
        "A_Baseline_BM25_k20",
        "E_CorroboratedHybrid_k40_UpdatedCB",
    ]

    results_by_config: Dict[str, List[Dict[str, Any]]] = {c: [] for c in configs}
    consecutive_bad = 0
    quota_halted = bool(args.cache_only)
    quota_halt_reason: Optional[str] = (
        "Run in --cache-only mode after Groq TPD quota limit (200,000 TPD on openai/gpt-oss-120b) was reached."
        if args.cache_only
        else None
    )

    for idx, row in enumerate(eval_df.itertuples(), start=1):
        qid = str(row.question_id)
        in_s40 = qid in strat_40_set
        in_d8 = qid in diag_8_set
        for cfg in configs:
            rec = evaluate_single_question_config(
                row=row,
                config_name=cfg,
                retrieval_cache=retrieval_cache,
                context_builder=context_builder,
                llm_client=llm_client,
                gen_cache=gen_cache,
                validator=validator,
                det_judge=det_judge,
                in_stratified_40=in_s40,
                in_diagnostic_8=in_d8,
                temperature=0.0,
                max_tokens=args.max_tokens,
                cache_only=quota_halted,
            )
            results_by_config[cfg].append(rec)

            aq = rec["answer_quality_heuristic"]
            gu = rec["generation_and_usage"]
            rc = rec["retrieval_and_context"]
            print(
                f"[{idx:02d}/{len(eval_df):02d}] {cfg[:18]:18s} | {qid} ({row.question_type[:12]:12s}) | "
                f"cache={gu['cache_status'][:12]:12s} | pr_rec={str(rc['prompt_gold_recall']):6s} | "
                f"strict={str(aq['answer_correct_strict']):5s} | comp={aq['fact_completeness_pct']:6.1f}% | "
                f"judge={str(aq['judge_answer_correct']):5s} | tok={gu['prompt_tokens']}+{gu['completion_tokens']} | "
                f"fail={aq['failure_category']}",
                flush=True,
            )

            if not quota_halted:
                if gu["api_error"] or aq["empty_response"] or aq["truncated_response"]:
                    consecutive_bad += 1
                    if consecutive_bad >= 2:
                        quota_halted = True
                        quota_halt_reason = (
                            f"Stopped live API calls after {consecutive_bad} consecutive errors/truncations "
                            f"at {qid} ({cfg}): {gu['api_error']}"
                        )
                        print(f"\n[SAFEGUARD TRIGGERED] {quota_halt_reason}", flush=True)
                else:
                    consecutive_bad = 0

    # Determine paired completed questions (questions where both A and E have valid generated answers)
    by_qid_a = {r["question_id"]: r for r in results_by_config["A_Baseline_BM25_k20"]}
    by_qid_e = {r["question_id"]: r for r in results_by_config["E_CorroboratedHybrid_k40_UpdatedCB"]}
    paired_completed_qids = [
        qid
        for qid in by_qid_a
        if by_qid_a[qid]["generation_and_usage"]["api_error"] is None
        and by_qid_e[qid]["generation_and_usage"]["api_error"] is None
    ]
    paired_set = set(paired_completed_qids)

    summaries: Dict[str, Any] = {}
    for cfg in configs:
        recs = results_by_config[cfg]
        paired_recs = [r for r in recs if r["question_id"] in paired_set]
        by_cat_paired: Dict[str, Any] = {}
        for cat in sorted(set(r["question_type"] for r in paired_recs)):
            by_cat_paired[cat] = aggregate_records([r for r in paired_recs if r["question_type"] == cat])

        summaries[cfg] = {
            "retrieval_all_47_deduplicated": aggregate_records(recs)["retrieval_and_prompt_coverage"],
            "retrieval_stratified_40": aggregate_records([r for r in recs if r["in_stratified_40"]])["retrieval_and_prompt_coverage"],
            "retrieval_diagnostic_8": aggregate_records([r for r in recs if r["in_diagnostic_8"]])["retrieval_and_prompt_coverage"],
            "paired_completed_generation": aggregate_records(paired_recs),
            "paired_stratified_40_completed": aggregate_records([r for r in paired_recs if r["in_stratified_40"]]),
            "paired_diagnostic_completed": aggregate_records([r for r in paired_recs if r["in_diagnostic_8"]]),
            "paired_by_category": by_cat_paired,
        }

    # Head-to-head question-level win/loss/tie analysis (E vs A) on paired completed questions
    comparisons: List[Dict[str, Any]] = []
    e_wins_strict = 0
    a_wins_strict = 0
    e_wins_judge = 0
    a_wins_judge = 0
    e_wins_comp = 0
    a_wins_comp = 0
    identical_prompts_all47 = 0
    identical_prompts_paired = 0

    for qid in by_qid_a:
        ra = by_qid_a[qid]
        re_ = by_qid_e[qid]
        same_prompt = (
            ra["generation_and_usage"]["user_prompt_sha256"]
            == re_["generation_and_usage"]["user_prompt_sha256"]
        )
        if same_prompt:
            identical_prompts_all47 += 1

        is_paired = qid in paired_set
        sa = ra["answer_quality_heuristic"]["answer_correct_strict"] if is_paired else None
        se = re_["answer_quality_heuristic"]["answer_correct_strict"] if is_paired else None
        ja = ra["answer_quality_heuristic"]["judge_answer_correct"] if is_paired else None
        je = re_["answer_quality_heuristic"]["judge_answer_correct"] if is_paired else None
        ca = ra["answer_quality_heuristic"]["fact_completeness_pct"] if is_paired else None
        ce = re_["answer_quality_heuristic"]["fact_completeness_pct"] if is_paired else None

        if is_paired:
            if same_prompt:
                identical_prompts_paired += 1
            if se and not sa:
                e_wins_strict += 1
            elif sa and not se:
                a_wins_strict += 1

            if je and not ja:
                e_wins_judge += 1
            elif ja and not je:
                a_wins_judge += 1

            if ce is not None and ca is not None:
                if ce > ca + 1e-6:
                    e_wins_comp += 1
                elif ca > ce + 1e-6:
                    a_wins_comp += 1

        comparisons.append(
            {
                "question_id": qid,
                "question_type": ra["question_type"],
                "in_stratified_40": ra["in_stratified_40"],
                "in_diagnostic_8": ra["in_diagnostic_8"],
                "paired_generation_completed": is_paired,
                "same_assembled_prompt": same_prompt,
                "A_prompt_gold_recall": ra["retrieval_and_context"]["prompt_gold_recall"],
                "E_prompt_gold_recall": re_["retrieval_and_context"]["prompt_gold_recall"],
                "A_evidence_fact_coverage": ra["retrieval_and_context"]["evidence_fact_coverage"],
                "E_evidence_fact_coverage": re_["retrieval_and_context"]["evidence_fact_coverage"],
                "A_strict_correct": sa,
                "E_strict_correct": se,
                "A_fact_completeness_pct": ca,
                "E_fact_completeness_pct": ce,
                "delta_fact_completeness_pct": round(ce - ca, 2) if (ce is not None and ca is not None) else None,
                "A_judge_correct": ja,
                "E_judge_correct": je,
                "A_failure_category": ra["answer_quality_heuristic"]["failure_category"] if is_paired else "skipped_quota",
                "E_failure_category": re_["answer_quality_heuristic"]["failure_category"] if is_paired else "skipped_quota",
            }
        )

    out_path = SMOKE_REPORT_PATH if args.smoke else FULL_REPORT_PATH
    report = {
        "benchmark": "VoiceEnterprise Controlled Generation Benchmark (A vs E)",
        "mode": "smoke" if args.smoke else "full_47q",
        "status": "COMPLETED" if len(paired_completed_qids) == len(eval_df) else "PARTIAL_STOPPED_ON_TPD_QUOTA",
        "quota_halt_reason": quota_halt_reason,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(time.time() - t_start, 2),
        "generator_model": llm_client.model,
        "generation_parameters": {
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "max_context_tokens": 2500,
            "max_chunks_per_doc": 3,
        },
        "question_set_metadata": subset_meta,
        "query_expansion_verification": exp_verification,
        "head_to_head_summary": {
            "total_questions_in_subset": len(comparisons),
            "paired_generation_completed_count": len(paired_completed_qids),
            "paired_completed_question_ids": paired_completed_qids,
            "identical_assembled_prompt_count_all47": identical_prompts_all47,
            "identical_assembled_prompt_count_paired": identical_prompts_paired,
            "differing_assembled_prompt_count_paired": len(paired_completed_qids) - identical_prompts_paired,
            "E_wins_strict_correctness": e_wins_strict,
            "A_wins_strict_correctness": a_wins_strict,
            "E_wins_judge_correctness": e_wins_judge,
            "A_wins_judge_correctness": a_wins_judge,
            "E_wins_fact_completeness": e_wins_comp,
            "A_wins_fact_completeness": a_wins_comp,
        },
        "summaries_by_config": summaries,
        "per_question_comparison": comparisons,
        "results_by_config": results_by_config,
        "generation_cache_stats": {
            "cache_path": str(GEN_CACHE_PATH),
            "total_cached_entries": len(gen_cache._data),
            "session_hits": gen_cache.hits,
            "session_misses": gen_cache.misses,
            "session_writes": gen_cache.writes,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved controlled generation report to: {out_path}")


if __name__ == "__main__":
    main()
