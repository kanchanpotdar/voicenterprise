"""Phase 6: End-to-End Evaluation Across Configurations E0 to E5.

Configurations evaluated:
- E0 (Baseline): BM25 (K=20) + standard ContextBuilder + Generator (No expansion, No verifier)
- E1 (Best Retrieval): Hybrid (K=40 + Div) + standard ContextBuilder + Generator
- E2 (Retrieval + Expansion): Hybrid (K=40) + Query Expansion (C1) + standard ContextBuilder + Generator
- E3 (Decomposition): Hybrid (K=40) + Sub-question Decomposition (D1) + standard ContextBuilder + Generator
- E4 (Coverage Context): Hybrid (K=40) + Query Expansion + CoverageAwareContextBuilder + Generator
- E5 (Full System + Verifier): CoverageAwareContextBuilder + AnswerVerifier (grounding verification, conflict resolution, numerical verification, bounded self-correction)

Measures:
- Answer Correctness (%)
- Answer Completeness (%)
- Overall Benchmark Score (correctness * completeness)
- Valid Citation Rate (%)
- Claim Support Rate (%)
- Error Attribution Breakdown:
  * Retrieval Miss
  * Context Truncation
  * Generator Hallucination
  * Contradiction Selection
  * Missing Numerical Fact
- Special validation on qst_0427 (ClearEdge PO fingerprint)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from evaluation.eval_rag import split_facts_and_constraints
from evaluation.evaluation_harness import (
    DIAGNOSTIC_QIDS,
    load_development_benchmark,
)
from evaluation.metrics import overall_answer_score
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.coverage_context_builder import CoverageAwareContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import OpenAICompatibleLLM
from rag.verifier import AnswerVerifier
from retrieval.indexer import EnterpriseRetriever
from retrieval.query_expansion import MultiQueryRetriever, QueryExpander

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_answer_facts(
    answer_text: str,
    raw_facts: List[str],
    question_type: str = "basic",
) -> Tuple[bool, float, List[str], List[str]]:
    """Score answer correctness and completeness against positive facts and negative constraints.

    Delegates to `evaluation.audit_scoring.score_answer` so that:
    - Empty or truncated outputs score (False, 0.0).
    - Abstentions on answerable questions score (False, 0.0).
    - Required facts require numeric/identifier anchors and >=0.50-0.60 stem recall
      (never single-token overlap).
    - Prohibited constraints check per-sentence with negation/supersession cues.
    """
    from evaluation.audit_scoring import score_answer

    res = score_answer(
        question_id="eval",
        question_type=question_type,
        answer_text=answer_text,
        answer_facts=raw_facts,
    )
    comp = 0.0 if (res.fact_completeness_pct != res.fact_completeness_pct) else float(res.fact_completeness_pct)
    return (
        res.answer_correct_strict,
        comp,
        res.missing_facts,
        res.suspected_constraint_violations,
    )


def run_phase6_evaluation():
    print("=" * 80)
    print("PHASE 6: END-TO-END BENCHMARK EVALUATION (CONFIGURATIONS E0 - E5)")
    print("=" * 80)

    df_all, df_group_a, df_group_b = load_development_benchmark()

    # Evaluation Cohort: Group B (14 Diagnostic Questions)
    df_eval = df_group_b.copy()
    print(f"Evaluated cohort: {len(df_eval)} diagnostic benchmark questions (Group B).\n")

    # Initialize components
    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )
    llm = OpenAICompatibleLLM(model="openai/gpt-oss-20b", timeout_seconds=45.0)
    print(f"Primary generator model: {llm.model}", flush=True)

    generator = AnswerGenerator(llm_client=llm)
    expander = QueryExpander(llm_client=llm)
    mq_retriever = MultiQueryRetriever(retriever=retriever, expander=expander)
    base_builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3)
    coverage_builder = CoverageAwareContextBuilder(
        max_context_tokens=2500,
        max_chunks_per_doc=3,
        novelty_weight=0.40,
        redundancy_penalty_weight=0.50,
        authority_boost=0.30,
    )
    validator = CitationValidator()
    verifier = AnswerVerifier(llm_client=llm)

    configs = [
        ("E0_Baseline", "bm25", 20, False, "none", base_builder, False),
        ("E1_BestRetrieval", "hybrid", 40, True, "none", base_builder, False),
        ("E2_RetrievalExpansion", "hybrid", 40, True, "alternative_queries", base_builder, False),
        ("E3_Decomposition", "hybrid", 40, True, "decomposition", base_builder, False),
        ("E4_CoverageContext", "hybrid", 40, True, "alternative_queries", coverage_builder, False),
        ("E5_FullSystemVerifier", "hybrid", 40, True, "alternative_queries", coverage_builder, True),
    ]

    results_by_config: Dict[str, Dict[str, Any]] = {c[0]: {} for c in configs}
    checkpoint_path = REPORTS_DIR / "phase6_checkpoint.json"
    if checkpoint_path.exists():
        try:
            saved_cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            for k in results_by_config:
                if k in saved_cp and isinstance(saved_cp[k], dict):
                    results_by_config[k] = saved_cp[k]
            print(f"Loaded existing checkpoint: { {k: len(v) for k, v in results_by_config.items()} }", flush=True)
        except Exception as e:
            print(f"Notice: Failed to load checkpoint: {e}", flush=True)

    for cfg_name, r_mode, top_k, div_filt, exp_mode, c_builder, use_verifier in configs:
        print(f"\n================ Running Configuration: {cfg_name} ================", flush=True)
        t_cfg0 = time.perf_counter()

        for idx, row in df_eval.iterrows():
            qid = str(row["question_id"])
            if qid in results_by_config[cfg_name] and "overall_score" in results_by_config[cfg_name][qid]:
                print(f"  [{cfg_name}] {qid:<10} | (loaded from checkpoint)", flush=True)
                continue
            query = str(row["question"])
            raw_g = row.get("expected_doc_ids")
            gold_docs = [str(x) for x in raw_g] if raw_g is not None and len(raw_g) > 0 else []
            raw_f = row.get("answer_facts")
            raw_facts = [str(x) for x in raw_f] if raw_f is not None and len(raw_f) > 0 else []

            # 1. Retrieval
            t0 = time.perf_counter()
            if exp_mode != "none":
                doc_ids, chunks, _, _ = mq_retriever.retrieve(
                    query=query,
                    mode=r_mode,
                    expansion_mode=exp_mode,
                    top_k_docs=top_k,
                    diversity_filter=div_filt,
                )
            else:
                doc_ids, chunks = retriever.retrieve_documents(
                    query=query,
                    mode=r_mode,
                    top_k_docs=top_k,
                    diversity_filter=div_filt,
                )
            retrieval_ms = (time.perf_counter() - t0) * 1000

            # 2. Context Construction
            t_c0 = time.perf_counter()
            if isinstance(c_builder, CoverageAwareContextBuilder):
                assembled = c_builder.build_context(retrieved_chunks=chunks, query=query)
            else:
                assembled = c_builder.build_context(retrieved_chunks=chunks)
            context_ms = (time.perf_counter() - t_c0) * 1000

            # 3. Generation (no silent model fallback: failures are raised explicitly)
            t_g0 = time.perf_counter()
            raw_answer = generator.generate_answer(
                question=query,
                assembled_context=assembled,
                temperature=0.0,
            )
            gen_ms = (time.perf_counter() - t_g0) * 1000

            # 4. Verification & Self-Correction (E5 only)
            ver_ms = 0.0
            ver_report = None
            final_answer = raw_answer
            if use_verifier:
                t_v0 = time.perf_counter()
                corr_ans_text, rep, rev_cnt = verifier.verify_and_correct(
                    question=query,
                    answer_text=raw_answer.answer_text,
                    context=assembled,
                    max_tokens=512,
                )
                ver_ms = (time.perf_counter() - t_v0) * 1000
                raw_answer.answer_text = corr_ans_text
                final_answer = raw_answer
                ver_report = rep.to_dict()

            # 5. Citation validation
            val_result = validator.validate(final_answer.answer_text, assembled)

            # 6. Fact & Constraint Scoring
            q_type = str(row.get("question_type") or "basic")
            is_correct, completeness_pct, missing_facts, violated_constraints = evaluate_answer_facts(
                answer_text=final_answer.answer_text,
                raw_facts=raw_facts,
                question_type=q_type,
            )
            score = overall_answer_score(is_correct, completeness_pct)

            # 7. Error Attribution
            # Check what caused failure if score < 100
            from evaluation.audit_scoring import is_empty_or_truncated

            flags = is_empty_or_truncated(final_answer.answer_text)
            error_attribution = "none"
            gold_in_retrieval = all(g in doc_ids for g in gold_docs) if gold_docs else True
            context_docs = {p.doc_id for p in assembled.passages}
            gold_in_context = all(g in context_docs for g in gold_docs) if gold_docs else True

            if not is_correct or completeness_pct < 80.0:
                if flags["empty"] or flags["truncated"]:
                    error_attribution = "generation_empty_or_truncated"
                elif not gold_in_retrieval:
                    error_attribution = "retrieval_miss"
                elif not gold_in_context:
                    error_attribution = "context_truncation"
                elif violated_constraints:
                    error_attribution = "contradiction_selection"
                elif val_result.invalid_doc_ids:
                    error_attribution = "generator_hallucination"
                elif missing_facts:
                    error_attribution = "missing_fact_synthesis"

            results_by_config[cfg_name][qid] = {
                "question": query,
                "question_type": row.get("question_type"),
                "answer_text": final_answer.answer_text,
                "gold_docs": gold_docs,
                "retrieved_docs": doc_ids[:10],
                "cited_passages": val_result.valid_doc_ids,
                "hallucinated_citations": val_result.invalid_doc_ids,
                "citation_validity_rate": val_result.citation_id_validity,
                "claim_support_rate": val_result.claim_support_rate,
                "has_unsourced_claims": val_result.has_unsourced_claims,
                "is_correct": is_correct,
                "completeness_pct": completeness_pct,
                "overall_score": score,
                "missing_facts": missing_facts,
                "violated_constraints": violated_constraints,
                "error_attribution": error_attribution,
                "tokens_used": assembled.total_tokens,
                "latencies_ms": {
                    "retrieval": round(retrieval_ms, 1),
                    "context": round(context_ms, 1),
                    "generation": round(gen_ms, 1),
                    "verifier": round(ver_ms, 1),
                    "total": round(retrieval_ms + context_ms + gen_ms + ver_ms, 1),
                },
                "verifier_report": ver_report,
            }

            print(f"  [{cfg_name}] {qid:<10} | Correct: {str(is_correct):<5} | Comp: {completeness_pct:>5.1f}% | Score: {score:>5.1f} | Error: {error_attribution}", flush=True)

            # Checkpoint results to disk after every question
            checkpoint_path = REPORTS_DIR / "phase6_checkpoint.json"
            checkpoint_path.write_text(json.dumps(results_by_config, indent=2), encoding="utf-8")

        t_cfg_elapsed = time.perf_counter() - t_cfg0
        print(f"Finished {cfg_name} in {t_cfg_elapsed:.1f}s", flush=True)

    # Aggregate summaries
    summary_by_config = {}
    for cfg_name, _ in [c[:2] for c in configs]:
        recs = list(results_by_config[cfg_name].values())
        mean_correctness = sum(1 for r in recs if r["is_correct"]) / len(recs)
        mean_completeness = sum(r["completeness_pct"] for r in recs) / len(recs)
        mean_overall = sum(r["overall_score"] for r in recs) / len(recs)
        mean_valid_cit = sum(r["citation_validity_rate"] for r in recs) / len(recs)
        mean_claim_sup = sum(r["claim_support_rate"] for r in recs) / len(recs)

        # Attribution counts
        attr_counts = {}
        for r in recs:
            ea = r["error_attribution"]
            attr_counts[ea] = attr_counts.get(ea, 0) + 1

        # Diagnostic 14 subset
        diag_recs = [results_by_config[cfg_name][qid] for qid in DIAGNOSTIC_QIDS if qid in results_by_config[cfg_name]]
        diag_correctness = sum(1 for r in diag_recs if r["is_correct"]) / len(diag_recs) if diag_recs else 0.0
        diag_score = sum(r["overall_score"] for r in diag_recs) / len(diag_recs) if diag_recs else 0.0

        summary_by_config[cfg_name] = {
            "overall_score": mean_overall,
            "correctness_rate": mean_correctness,
            "completeness_pct": mean_completeness,
            "citation_validity_rate": mean_valid_cit,
            "claim_support_rate": mean_claim_sup,
            "diagnostic_14_score": diag_score,
            "diagnostic_14_correctness": diag_correctness,
            "error_attribution_counts": attr_counts,
        }

    # Print summary table
    print("\n" + "=" * 105)
    print(f"{'Configuration':<24} | {'Overall Score':<14} | {'Correctness':<12} | {'Completeness':<12} | {'Claim Support':<14} | {'Diag Score':<12}")
    print("-" * 105)
    for cfg_name, sm in summary_by_config.items():
        print(
            f"{cfg_name:<24} | "
            f"{sm['overall_score']:>12.2f} | "
            f"{sm['correctness_rate']*100:>10.1f}% | "
            f"{sm['completeness_pct']:>10.1f}% | "
            f"{sm['claim_support_rate']*100:>12.1f}% | "
            f"{sm['diagnostic_14_score']:>10.2f}"
        )
    print("=" * 105)

    # ClearEdge qst_0427 verification check
    print("\n--- ClearEdge qst_0427 Verification Check ---")
    for cfg_name, _ in [c[:2] for c in configs]:
        ans_427 = results_by_config[cfg_name].get("qst_0427", {}).get("answer_text", "")
        has_hash_format = "sha-256" in ans_427.lower() or "pofingerprint" in ans_427.lower()
        has_superseded = "supersede" in ans_427.lower() or "june" in ans_427.lower()
        print(f"  {cfg_name:<24}: Mention SHA-256/poFingerprint: {has_hash_format} | Acknowledged Superseded: {has_superseded}")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary_by_config,
        "results_by_config": results_by_config,
    }

    report_path = REPORTS_DIR / "phase6_end_to_end_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nPhase 6 Report saved to {report_path}")


if __name__ == "__main__":
    run_phase6_evaluation()
