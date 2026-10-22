"""Milestone 1.2: Controlled Verification and Downstream Evaluation.

Compares Milestone 1.1 Baseline vs. Milestone 1.2 Updated Implementation.
Evaluates on fixed development set queries spanning:
- Conflict resolution: qst_0427, qst_0413
- Long SOP & Completeness: qst_0447, qst_0436, qst_0437, qst_0448
- Basic & Constrained: qst_0001, qst_0016, qst_0043, qst_0139
- Unanswerable (info_not_found): qst_0499, qst_0488
- Semantic: qst_0214

Tracks:
- Answer Correctness (% binary)
- Completeness (% fact coverage)
- Overall Score (Correctness * Completeness)
- Citation ID Validity (%)
- Sentence Citation Rate (%)
- Lexical Claim Support Heuristic (%)
- Abstention Accuracy on info_not_found
- Contradiction / Supersession Accuracy on conflicting_info
- Truncation / Gap Reporting
- Latency (ms)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation.eval_rag import DeterministicJudge
from evaluation.metrics import overall_answer_score
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import OpenAICompatibleLLM, load_env_file
from rag.pipeline import RAGPipeline
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent

# Fixed stratified evaluation set
EVAL_QIDS = [
    # Critical targets from tasks
    "qst_0427",  # ClearEdge conflict & supersession
    "qst_0447",  # Hotfix SOP (6 gold docs, 22 facts)
    "qst_0436",  # SDK auth bug reports (10 gold docs)
    "qst_0437",  # Incident action items (9 gold docs)
    "qst_0448",  # Retention exceptions (7 gold docs)
    # Unanswerable
    "qst_0499",  # Kafka topic info_not_found
    "qst_0488",  # Unanswerable billing info_not_found
    # Rank cutoff & dense rescues
    "qst_0043",  # p99 latency jump
    "qst_0139",  # PR review FP threshold
    "qst_0214",  # Runtime rollback build
    # Basic benchmark anchors
    "qst_0001",  # File upload limits
    "qst_0016",  # Tenant rate limit
]


def run_milestone_1_2_benchmark():
    load_env_file()
    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    dev_df = pd.read_parquet(dev_path)
    q_map = {str(r["question_id"]): r for r in dev_df.to_dict(orient="records")}

    llm = OpenAICompatibleLLM(model="openai/gpt-oss-120b")
    pipeline = RAGPipeline.from_defaults(llm_client=llm)
    judge = DeterministicJudge()

    results: List[Dict[str, Any]] = []

    print(f"Starting Milestone 1.2 Evaluation across {len(EVAL_QIDS)} benchmark questions...")
    print("Model: openai/gpt-oss-120b | Retrieval: BM25 (top-20 docs) | Context: 2500 tokens\n")

    t0_all = time.perf_counter()

    for idx, qid in enumerate(EVAL_QIDS, start=1):
        q_row = q_map.get(qid)
        if not q_row:
            continue
        qtext = str(q_row["question"])
        qtype = str(q_row["question_type"])
        raw_exp = q_row.get("expected_doc_ids")
        exp_docs = [str(x) for x in raw_exp] if raw_exp is not None and len(raw_exp) > 0 else []
        raw_facts = q_row.get("answer_facts")
        facts = [str(x) for x in raw_facts] if raw_facts is not None and len(raw_facts) > 0 else []

        print("=" * 80)
        print(f"[{idx}/{len(EVAL_QIDS)}] Question ID: {qid} ({qtype})")
        print(f"Question: {qtext}")
        print(f"Expected Gold Docs ({len(exp_docs)}): {exp_docs}")

        t_q0 = time.perf_counter()
        ans = pipeline.query(question=qtext, mode="bm25", top_k_docs=20)
        t_q_elapsed = time.perf_counter() - t_q0

        # Judge
        correct, completeness, expl = judge.evaluate(
            question=qtext,
            question_type=qtype,
            generated_answer=ans.answer_text,
            gold_answer=str(q_row.get("expected_answer") or ""),
            answer_facts=facts,
            is_abstention=ans.is_abstention,
            has_contradiction=ans.has_contradiction,
        )
        ov_score = overall_answer_score(correct, completeness)

        # Check ClearEdge specific alignment for qst_0427
        extra_checks = {}
        if qid == "qst_0427":
            confluence_id = "dsid_d93aed75845e4a0b8126a80d8a7a5a41"
            gmail_id = "dsid_86f033ed92b84f3a8b381d09b0b95357"
            confluence_cited_for_monday = False
            gmail_cited_for_superseded = False
            for s in ans.citation_validation.sentences:
                if "monday" in s.sentence.lower():
                    if confluence_id in s.cited_doc_ids:
                        confluence_cited_for_monday = True
                    if gmail_id in s.cited_doc_ids:
                        gmail_cited_for_superseded = True
            extra_checks["confluence_falsely_cited_for_monday"] = confluence_cited_for_monday
            extra_checks["gmail_correctly_cited_for_superseded"] = gmail_cited_for_superseded

        # Check truncation for qst_0447
        if qid == "qst_0447":
            extra_checks["truncation_detected"] = "[TRUNCATION DETECTED" in ans.answer_text
            extra_checks["markdown_table_used"] = "|---" in ans.answer_text
            extra_checks["gold_docs_assembled"] = [d for d in exp_docs if d in ans.retrieved_doc_ids]

        res_item = {
            "question_id": qid,
            "question_type": qtype,
            "question": qtext,
            "num_expected_docs": len(exp_docs),
            "expected_doc_ids": exp_docs,
            "retrieved_doc_ids": ans.retrieved_doc_ids[:8],
            "cited_doc_ids": ans.cited_doc_ids,
            "answer_correct": correct,
            "completeness_pct": completeness,
            "overall_score": ov_score,
            "citation_id_validity": ans.citation_validation.citation_id_validity,
            "sentence_citation_rate": ans.citation_validation.sentence_citation_rate,
            "lexical_claim_support_heuristic": ans.citation_validation.claim_support_rate,
            "has_unsourced_claims": ans.citation_validation.has_unsourced_claims,
            "is_abstention": ans.is_abstention,
            "has_contradiction": ans.has_contradiction,
            "context_tokens": ans.context_tokens,
            "passages_count": ans.context_passages_count,
            "latency_ms": round(t_q_elapsed * 1000.0, 1),
            "explanation": expl,
            "extra_checks": extra_checks,
            "answer_text": ans.answer_text,
        }
        results.append(res_item)

        print(f"Result: Correct={correct} | Completeness={completeness:.1f}% | Score={ov_score:.1f} | Citations={ans.cited_doc_ids}")
        print(f"Explanation: {expl}")
        print(f"Latency: {res_item['latency_ms']} ms")
        time.sleep(0.5)

    total_time = time.perf_counter() - t0_all

    # Aggregate metrics
    answerable = [r for r in results if r["question_type"] != "info_not_found"]
    unanswerable = [r for r in results if r["question_type"] == "info_not_found"]
    conflicts = [r for r in results if r["question_type"] == "conflicting_info"]
    completeness_q = [r for r in results if r["question_type"] == "completeness"]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "num_evaluated": len(results),
        "total_time_seconds": round(total_time, 2),
        "overall": {
            "overall_score": round(_mean([r["overall_score"] for r in results]), 2),
            "answer_correctness_pct": round(_mean([100.0 if r["answer_correct"] else 0.0 for r in results]), 2),
            "answer_completeness_pct": round(_mean([r["completeness_pct"] for r in results]), 2),
            "citation_id_validity_pct": round(_mean([r["citation_id_validity"] * 100.0 for r in results]), 2),
            "sentence_citation_rate_pct": round(_mean([r["sentence_citation_rate"] * 100.0 for r in results]), 2),
            "claim_support_heuristic_pct": round(_mean([r["lexical_claim_support_heuristic"] * 100.0 for r in results]), 2),
            "avg_latency_ms": round(_mean([r["latency_ms"] for r in results]), 1),
        },
        "subsets": {
            "answerable": {
                "count": len(answerable),
                "overall_score": round(_mean([r["overall_score"] for r in answerable]), 2),
                "correctness": round(_mean([100.0 if r["answer_correct"] else 0.0 for r in answerable]), 2),
                "completeness": round(_mean([r["completeness_pct"] for r in answerable]), 2),
            },
            "unanswerable": {
                "count": len(unanswerable),
                "abstention_accuracy": round(_mean([100.0 if r["is_abstention"] else 0.0 for r in unanswerable]), 2),
                "correctness": round(_mean([100.0 if r["answer_correct"] else 0.0 for r in unanswerable]), 2),
            },
            "conflict_resolution": {
                "count": len(conflicts),
                "contradiction_detection_rate": round(_mean([100.0 if r["has_contradiction"] else 0.0 for r in conflicts]), 2),
                "overall_score": round(_mean([r["overall_score"] for r in conflicts]), 2),
            },
            "multi_doc_completeness": {
                "count": len(completeness_q),
                "overall_score": round(_mean([r["overall_score"] for r in completeness_q]), 2),
                "completeness": round(_mean([r["completeness_pct"] for r in completeness_q]), 2),
            },
        },
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "per_question": results,
    }

    out_file = BASE_DIR / "evaluation" / "reports" / "milestone_1_2_evaluation.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved Milestone 1.2 evaluation report to: {out_file}")

    print("\n" + "=" * 80)
    print("MILESTONE 1.2 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Overall Score             : {summary['overall']['overall_score']}")
    print(f"Answer Correctness        : {summary['overall']['answer_correctness_pct']}%")
    print(f"Answer Completeness       : {summary['overall']['answer_completeness_pct']}%")
    print(f"Citation ID Validity      : {summary['overall']['citation_id_validity_pct']}%")
    print(f"Sentence Citation Coverage: {summary['overall']['sentence_citation_rate_pct']}%")
    print(f"Claim Support Proxy       : {summary['overall']['claim_support_heuristic_pct']}%")
    print(f"Abstention Accuracy       : {summary['subsets']['unanswerable']['abstention_accuracy']}%")
    print(f"Multi-Doc Completeness Avg: {summary['subsets']['multi_doc_completeness']['completeness']}%")
    print(f"Avg Latency               : {summary['overall']['avg_latency_ms']} ms")
    print("=" * 80)


if __name__ == "__main__":
    run_milestone_1_2_benchmark()
