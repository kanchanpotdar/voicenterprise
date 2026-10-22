"""Benchmark the fast model (openai/gpt-oss-20b) vs flagship model (openai/gpt-oss-120b).

Tracks latency stage by stage:
- retrieval_ms
- context_builder_ms
- ttft_ms (time to first token via SSE streaming)
- generation_ms
- validation_ms
- total_e2e_ms

Evaluates accuracy, completeness, citation precision, and domain assertions across
the 10 representative queries (conflict resolution, long SOP, candidate cutoff rescues,
dense-dependent, standard single/multi-doc, and abstentions).
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evaluation.eval_rag import DeterministicJudge
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import OpenAICompatibleLLM
from rag.pipeline import RAGPipeline
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent

REPRESENTATIVE_QIDS = [
    "qst_0427",  # ClearEdge conflict resolution (SHA-256 vs unhashed)
    "qst_0447",  # Long SOP synthesis (10 phases across 6 gold docs)
    "qst_0436",  # Cutoff failure rescued at K=28
    "qst_0437",  # Cutoff failure rescued at K=21
    "qst_0448",  # Cutoff failure rescued at K=46
    "qst_0001",  # Standard single-doc query
    "qst_0016",  # Standard multi-doc query
    "qst_0214",  # Dense/hybrid-dependent query
    "qst_0499",  # Abstention / info_not_found
    "qst_0481",  # Abstention / info_not_found
]


def run_fast_model_benchmark():
    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    df_dev = pd.read_parquet(dev_path)

    # Filter to representative queries in order
    df_rep = df_dev[df_dev["question_id"].isin(REPRESENTATIVE_QIDS)].copy()
    # Sort according to REPRESENTATIVE_QIDS order
    df_rep["sort_order"] = df_rep["question_id"].map(lambda x: REPRESENTATIVE_QIDS.index(x))
    df_rep = df_rep.sort_values("sort_order").reset_index(drop=True)

    print(f"Loaded {len(df_rep)} representative questions for fast-model comparison.")

    # Shared retriever with diversity filter and top_k_docs=40
    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )
    context_builder = ContextBuilder(max_context_tokens=2500)
    judge = DeterministicJudge()

    models_to_test = [
        ("openai/gpt-oss-20b", "GPT-OSS 20B (Fast)"),
        ("openai/gpt-oss-120b", "GPT-OSS 120B (Flagship)"),
    ]

    results_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for model_name, model_label in models_to_test:
        print(f"\n==========================================", flush=True)
        print(f"Evaluating {model_label} ({model_name})...", flush=True)
        print(f"==========================================", flush=True)

        llm_client = OpenAICompatibleLLM(model=model_name, timeout_seconds=90.0)
        generator = AnswerGenerator(llm_client=llm_client)
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=generator,
            context_builder=context_builder,
            default_mode="bm25",
        )

        model_results = []

        for idx, row in df_rep.iterrows():
            qid = row["question_id"]
            question = row["question"]
            gold_answer = str(row.get("gold_answer") or "")
            category = row["question_type"]

            raw_facts = row.get("answer_facts")
            if raw_facts is not None and hasattr(raw_facts, "tolist"):
                raw_facts = raw_facts.tolist()
            answer_facts = list(raw_facts) if raw_facts is not None else []

            raw_expected = row.get("expected_doc_ids")
            expected_doc_ids = (
                [str(x) for x in raw_expected]
                if raw_expected is not None and len(raw_expected) > 0
                else []
            )

            print(f"[{model_name}] ({idx+1}/{len(df_rep)}) {qid} [{category}]: {question[:60]}...", flush=True)

            try:
                # Run pipeline with streaming=True to measure exact TTFT
                ans = pipeline.query(
                    question=question,
                    mode="bm25",
                    top_k_docs=40,
                    diversity_filter=True,
                    stream=True,
                )

                # Evaluate with DeterministicJudge
                correct, comp, expl = judge.evaluate(
                    question=question,
                    question_type=category,
                    generated_answer=ans.answer_text,
                    gold_answer=gold_answer,
                    answer_facts=answer_facts,
                    is_abstention=ans.is_abstention,
                    has_contradiction=ans.has_contradiction,
                )
                overall_score = round(comp, 2) if correct else 0.0

                # Specific domain checks
                special_checks = {}
                if qid == "qst_0427":
                    # ClearEdge check
                    lower = ans.answer_text.lower()
                    has_sha256 = "sha-256" in lower or "sha256" in lower or "hexadecimal" in lower
                    has_po_fingerprint = "pofingerprint" in lower or "fingerprint" in lower
                    mentions_unhashed_superseded = "po#|lineuuid" in lower or "un-hashed" in lower or "superseded" in lower
                    special_checks["qst_0427_sha256_correct"] = has_sha256 and has_po_fingerprint
                    special_checks["qst_0427_superseded_documented"] = mentions_unhashed_superseded
                    # Check citation precision on final vs draft
                    final_doc = "dsid_d93a61f5c6a1b853874c728e124803d1"
                    gmail_doc = "dsid_86f0c3d9a105c317be9ef282542a1b9c"
                    special_checks["cites_final_confluence"] = final_doc in ans.cited_doc_ids
                    special_checks["cites_gmail_draft"] = gmail_doc in ans.cited_doc_ids

                elif qid == "qst_0447":
                    # SOP coverage check: check how many key operational phases are present
                    sop_phases = [
                        "preconditions", "pr", "signing", "validation", "hosted", "dedicated", "customer", "closeout"
                    ]
                    phases_found = sum(1 for p in sop_phases if p in ans.answer_text.lower())
                    special_checks["qst_0447_sop_phases_found"] = phases_found
                    special_checks["qst_0447_no_truncation"] = "[TRUNCATION DETECTED" not in ans.answer_text

                record = {
                    "question_id": qid,
                    "category": category,
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": ans.answer_text,
                    "answer_correct": correct,
                    "completeness_pct": comp,
                    "overall_score": overall_score,
                    "judge_explanation": expl,
                    "citation_validity_rate": ans.citation_validation.citation_id_validity,
                    "has_hallucinated_citations": ans.citation_validation.has_hallucinated_citations,
                    "valid_doc_ids": ans.citation_validation.valid_doc_ids,
                    "invalid_doc_ids": ans.citation_validation.invalid_doc_ids,
                    "is_abstention": ans.is_abstention,
                    "has_contradiction": ans.has_contradiction,
                    "retrieval_ms": ans.retrieval_ms,
                    "context_builder_ms": ans.context_builder_ms,
                    "ttft_ms": ans.ttft_ms,
                    "generation_ms": ans.generation_ms,
                    "validation_ms": ans.validation_ms,
                    "total_e2e_ms": ans.total_e2e_ms,
                    "context_tokens": ans.context_tokens,
                    "special_checks": special_checks,
                }
                model_results.append(record)

                print(
                    f"    Score: {overall_score:.1f}% | Correct: {correct} | "
                    f"TTFT: {ans.ttft_ms:.0f}ms | Gen: {ans.generation_ms:.0f}ms | E2E: {ans.total_e2e_ms:.0f}ms",
                    flush=True,
                )
            except Exception as e:
                print(f"    ERROR on {qid}: {e}", flush=True)
                continue

        results_by_model[model_name] = model_results

    # Compute aggregates
    summary = {}
    for model_name, records in results_by_model.items():
        n = len(records)
        correct_n = sum(1 for r in records if r["answer_correct"])
        mean_score = sum(r["overall_score"] for r in records) / n
        mean_completeness = sum(r["completeness_pct"] for r in records) / n
        mean_citation_validity = sum(r["citation_validity_rate"] for r in records) / n
        hallucination_count = sum(1 for r in records if r["has_hallucinated_citations"])
        mean_retrieval_ms = sum(r["retrieval_ms"] for r in records) / n
        mean_context_ms = sum(r["context_builder_ms"] for r in records) / n
        mean_ttft_ms = sum(r["ttft_ms"] for r in records) / n
        mean_generation_ms = sum(r["generation_ms"] for r in records) / n
        mean_validation_ms = sum(r["validation_ms"] for r in records) / n
        mean_e2e_ms = sum(r["total_e2e_ms"] for r in records) / n

        summary[model_name] = {
            "num_questions": n,
            "correct_count": correct_n,
            "accuracy_pct": round((correct_n / n) * 100.0, 2),
            "mean_completeness_pct": round(mean_completeness, 2),
            "mean_overall_score": round(mean_score, 2),
            "mean_citation_validity_pct": round(mean_citation_validity * 100.0, 2),
            "hallucinated_citation_questions": hallucination_count,
            "mean_retrieval_ms": round(mean_retrieval_ms, 1),
            "mean_context_ms": round(mean_context_ms, 1),
            "mean_ttft_ms": round(mean_ttft_ms, 1),
            "mean_generation_ms": round(mean_generation_ms, 1),
            "mean_validation_ms": round(mean_validation_ms, 1),
            "mean_total_e2e_ms": round(mean_e2e_ms, 1),
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "representative_questions": REPRESENTATIVE_QIDS,
        "results_by_model": results_by_model,
    }

    out_path = BASE_DIR / "evaluation" / "reports" / "fast_model_benchmark_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport successfully saved to {out_path}")


if __name__ == "__main__":
    run_fast_model_benchmark()
