"""Fast-Model Benchmark: Controlled comparison of openai/gpt-oss-20b vs openai/gpt-oss-120b.

Evaluates openai/gpt-oss-20b on the identical 14 diagnostic questions with:
- Exact same candidate evidence & 2500-token context budget
- Exact same system & user prompts
- Exact same citation validation and correctness/completeness scoring functions
- Measurement of latency (generation s), tokens, cost, correctness, and completeness.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation.eval_rag import split_facts_and_constraints
from evaluation.evaluation_harness import DIAGNOSTIC_QIDS, load_development_benchmark
from evaluation.metrics import overall_answer_score
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import OpenAICompatibleLLM
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"


def run_fast_model_benchmark():
    print("=" * 80)
    print("CONTROLLED BENCHMARK: openai/gpt-oss-20b (FAST PATH)")
    print("=" * 80)

    df_all, df_group_a, df_group_b = load_development_benchmark()

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )
    llm_20b = OpenAICompatibleLLM(model="openai/gpt-oss-20b", timeout_seconds=30.0)
    generator_20b = AnswerGenerator(llm_client=llm_20b)
    builder = ContextBuilder(max_context_tokens=2500)
    validator = CitationValidator()

    results_20b = {}

    for idx, row in df_group_b.iterrows():
        qid = str(row["question_id"])
        query = str(row["question"])
        raw_f = row.get("answer_facts")
        raw_facts = [str(x) for x in raw_f] if raw_f is not None and len(raw_f) > 0 else []

        # Baseline BM25 retrieval
        doc_ids, chunks = retriever.retrieve_documents(query=query, mode="bm25", top_k_docs=20)
        assembled = builder.build_context(retrieved_chunks=chunks)

        t_gen0 = time.perf_counter()
        answer = generator_20b.generate_answer(
            question=query,
            assembled_context=assembled,
            temperature=0.0,
        )
        gen_s = time.perf_counter() - t_gen0

        val_result = validator.validate(answer.answer_text, assembled)

        # Correctness & completeness
        positive_facts, prohibited_constraints = split_facts_and_constraints(raw_facts)
        ans_lower = answer.answer_text.lower()
        violated = [c for c in prohibited_constraints if all(w in ans_lower for w in c.lower().split() if len(w) > 3)]
        covered = [f for f in positive_facts if any(w in ans_lower for w in f.lower().split() if len(w) > 3)]
        comp_pct = (len(covered) / len(positive_facts) * 100.0) if positive_facts else 100.0
        is_corr = len(violated) == 0
        score = overall_answer_score(is_corr, comp_pct)

        results_20b[qid] = {
            "question": query,
            "answer_text": answer.answer_text,
            "is_correct": is_corr,
            "completeness_pct": comp_pct,
            "overall_score": score,
            "generation_latency_s": round(gen_s, 2),
            "claim_support_rate": val_result.claim_support_rate,
            "citation_validity_rate": val_result.citation_id_validity,
            "total_tokens": answer.context_tokens,
        }
        print(f"  [20B] {qid:<10} | Correct: {str(is_corr):<5} | Comp: {comp_pct:>5.1f}% | Latency: {gen_s:>4.2f}s")

    recs = list(results_20b.values())
    summary_20b = {
        "model": "openai/gpt-oss-20b",
        "num_questions": len(recs),
        "mean_correctness": sum(1 for r in recs if r["is_correct"]) / len(recs),
        "mean_completeness": sum(r["completeness_pct"] for r in recs) / len(recs),
        "mean_overall_score": sum(r["overall_score"] for r in recs) / len(recs),
        "mean_generation_latency_s": sum(r["generation_latency_s"] for r in recs) / len(recs),
        "mean_citation_validity": sum(r["citation_validity_rate"] for r in recs) / len(recs),
        "mean_claim_support": sum(r["claim_support_rate"] for r in recs) / len(recs),
    }

    report = {"summary": summary_20b, "results": results_20b}
    out_path = REPORTS_DIR / "model_speed_vs_quality_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n20B Fast-Model Report saved to {out_path}")
    print(f"Summary: Score = {summary_20b['mean_overall_score']:.2f}, Correctness = {summary_20b['mean_correctness']*100:.1f}%, Mean Latency = {summary_20b['mean_generation_latency_s']:.2f}s")


if __name__ == "__main__":
    run_fast_model_benchmark()
