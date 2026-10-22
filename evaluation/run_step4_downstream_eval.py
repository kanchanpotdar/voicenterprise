"""VoiceEnterprise - Downstream Answer Quality Evaluation (Step 4).

Evaluates representative queries spanning the key failure modes and behaviors:
1. Rank-cutoff failure: qst_0043 (recovered at rank 39)
2. Rank-cutoff failure: qst_0139 (recovered at rank 36)
3. Dense-only rescue: qst_0214 (runtime rollback build version; BM25 >50, Dense rank 7)
4. Terminology gap: qst_0178 (low bit math / FP8 pass rate; BM25 >50, requires query expansion)
5. Multi-doc completeness: qst_0436 (SDK auth bug reports across 10 docs)
6. Multi-doc completeness: qst_0437 (postmortem action items across 9 docs)
7. Multi-doc completeness: qst_0448 (retention policy exceptions across 7 docs)
8. Conflicting document versions: qst_0427 (SHA-256 hash, Tuesday 07:00 PT, superseded Gmail note)
9. Unanswerable: qst_0499 (profiling.signals Kafka topic; info_not_found clean abstention)

Evaluates answers across:
- Baseline (BM25, K=20)
- Experiment A (BM25, K=40)
- Experiment B (BM25, K=40 + Diversity Filter)
- Plus Hybrid (K=40) for dense-rescue and terminology gap queries.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from evaluation.eval_rag import DeterministicJudge
from evaluation.metrics import overall_answer_score
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator, RAGAnswer
from rag.llm import OpenAICompatibleLLM, load_env_file
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent

TARGET_QIDS = [
    ("qst_0043", "rank_cutoff"),
    ("qst_0139", "rank_cutoff"),
    ("qst_0214", "dense_rescue"),
    ("qst_0178", "lexical_gap"),
    ("qst_0436", "completeness"),
    ("qst_0437", "completeness"),
    ("qst_0448", "completeness"),
    ("qst_0427", "conflicting_info"),
    ("qst_0499", "unanswerable"),
]


def run_step4_evaluation():
    load_env_file()
    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    dev_df = pd.read_parquet(dev_path)
    q_map = {str(r["question_id"]): r for r in dev_df.to_dict(orient="records")}

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )
    context_builder = ContextBuilder(max_context_tokens=2500)

    # Use openai/gpt-oss-120b on Groq
    llm = OpenAICompatibleLLM(model="openai/gpt-oss-120b")
    generator = AnswerGenerator(llm_client=llm)
    judge = DeterministicJudge()

    results: List[Dict[str, Any]] = []

    print("Starting Step 4 Downstream Answer Quality Evaluation...")
    print(f"Model: openai/gpt-oss-120b | Context Budget: 2500 tokens\n")

    for qid, category_note in TARGET_QIDS:
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
        print(f"[{qid}] ({category_note} | {qtype})")
        print(f"Question: {qtext}")
        print(f"Expected Docs ({len(exp_docs)}): {exp_docs}")

        # Configs to run: Baseline, Exp A, Exp B (and Hybrid for rescue/gap)
        configs = [
            ("Baseline (K=20)", "bm25", 20, False),
            ("Exp A (K=40)", "bm25", 40, False),
            ("Exp B (K=40+Div)", "bm25", 40, True),
        ]
        if category_note in {"dense_rescue", "lexical_gap"}:
            configs.append(("Hybrid (K=40)", "hybrid", 40, True))

        for cfg_name, mode, top_k, div_filt in configs:
            t0 = time.perf_counter()

            # 1. Retrieve
            doc_ids, retrieved_chunks = retriever.retrieve_documents(
                query=qtext,
                mode=mode,
                top_k_docs=top_k,
                diversity_filter=div_filt,
            )

            # Gold recall
            exp_set = set(exp_docs)
            hits = len(exp_set.intersection(doc_ids))
            rec = hits / len(exp_set) if exp_set else (1.0 if not doc_ids else 0.0)

            # 2. Context assembly
            assembled = context_builder.build_context(retrieved_chunks=retrieved_chunks)

            # 3. Generation
            ans = generator.generate_answer(
                question=qtext,
                assembled_context=assembled,
                temperature=0.0,
                max_tokens=1024,
            )
            elapsed = time.perf_counter() - t0

            # 4. Scoring
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

            res_entry = {
                "question_id": qid,
                "category_note": category_note,
                "config": cfg_name,
                "mode": mode,
                "top_k_docs": top_k,
                "diversity_filter": div_filt,
                "recall": round(rec, 4),
                "hits": hits,
                "num_expected": len(exp_docs),
                "retrieved_doc_ids": doc_ids[:10],
                "cited_doc_ids": ans.cited_doc_ids,
                "citation_precision": ans.citation_validation.citation_precision,
                "is_abstention": ans.is_abstention,
                "has_contradiction": ans.has_contradiction,
                "answer_correct": correct,
                "completeness_pct": completeness,
                "overall_score": ov_score,
                "latency_ms": round(elapsed * 1000.0, 1),
                "explanation": expl,
                "generated_answer_snippet": ans.answer_text[:280].replace("\n", " "),
                "full_answer": ans.answer_text,
            }
            results.append(res_entry)

            print(f"  [{cfg_name:<16}] Recall: {hits}/{len(exp_docs)} ({rec:.1%}) | Correct: {correct} | Completeness: {completeness:.1f}% | Score: {ov_score:.1f} | Citations: {ans.cited_doc_ids}")
            print(f"    Snippet: {ans.answer_text[:160]}...")
            time.sleep(0.5)  # slight pause for Groq rate limits

    out_path = BASE_DIR / "evaluation" / "reports" / "step4_downstream_eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nStep 4 evaluation completed! Saved results to {out_path}")


if __name__ == "__main__":
    run_step4_evaluation()
