"""Head-to-head comparative evaluation of Groq models on EnterpriseRAG-Bench.

Models evaluated:
1. openai/gpt-oss-120b
2. qwen/qwen3.8-27b
3. openai/gpt-oss-20b

Design:
- Uses 30 stratified dev questions across all 10 question categories.
- Ensures identical evidence retrieval (BM25) and identical prompts for all 3 models.
- Compares:
  * Answer Correctness (% binary)
  * Answer Completeness (% fact coverage)
  * Overall Score (Correctness x Completeness)
  * Citation Precision (% of cited doc IDs validly referencing retrieved evidence)
  * Abstention on unanswerable questions (info_not_found)
  * Conflict/Contradiction detection on conflicting_info questions
  * Generation Latency (ms)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from evaluation.eval_rag import DeterministicJudge
from evaluation.metrics import overall_answer_score
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator, RAGAnswer
from rag.llm import OpenAICompatibleLLM, load_env_file
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

CATEGORY_QUOTAS = [
    ("basic", 5),
    ("semantic", 5),
    ("constrained", 3),
    ("intra_document_reasoning", 3),
    ("project_related", 3),
    ("conflicting_info", 3),
    ("completeness", 3),
    ("info_not_found", 3),
    ("miscellaneous", 1),
    ("high_level", 1),
]


def load_stratified_questions(sample_target: int = 30) -> pd.DataFrame:
    """Load stratified sample of development questions."""
    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    if not dev_path.exists():
        raise FileNotFoundError(f"Questions parquet not found at {dev_path}")

    df = pd.read_parquet(dev_path)
    selected_indices: List[int] = []

    for cat, count in CATEGORY_QUOTAS:
        sub = df[df["question_type"] == cat]
        if not sub.empty:
            take = min(count, len(sub))
            sampled = sub.sample(n=take, random_state=42)
            selected_indices.extend(sampled.index.tolist())

    # Fill remainder if needed
    if len(selected_indices) < sample_target:
        remaining = [i for i in df.index if i not in selected_indices]
        needed = sample_target - len(selected_indices)
        selected_indices.extend(remaining[:needed])

    return df.loc[selected_indices[:sample_target]].copy().reset_index(drop=True)


def run_comparison(sample_size: int = 30) -> Dict[str, Any]:
    load_env_file()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        sys.exit("ERROR: GROQ_API_KEY not set in environment or .env file.")

    qdrant_dir = BASE_DIR / "data" / "qdrant_dev_10k"
    if not qdrant_dir.exists():
        sys.exit(f"ERROR: Qdrant collection directory not found at {qdrant_dir}")

    df = load_stratified_questions(sample_size)
    print(f"Loaded {len(df)} stratified questions across {df['question_type'].nunique()} categories.")

    # Initialize shared retriever and context builder
    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=qdrant_dir,
    )
    context_builder = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=2)
    validator = CitationValidator()
    judge = DeterministicJudge()

    # Initialize generators for each model
    generators: Dict[str, AnswerGenerator] = {}
    for m in MODELS:
        llm = OpenAICompatibleLLM(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
            model=m,
            timeout_seconds=45.0,
        )
        generators[m] = AnswerGenerator(llm_client=llm, validator=validator)

    results_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in MODELS}

    print("\nStarting comparative evaluation on identical evidence context...")
    print("=" * 88)

    for q_idx, row in df.iterrows():
        qid = row["question_id"]
        qtype = row["question_type"]
        question = row["question"]
        gold_ans = row["gold_answer"]
        facts = list(row["answer_facts"]) if row["answer_facts"] is not None else []
        expected_docs = list(row["expected_doc_ids"]) if row["expected_doc_ids"] is not None else []

        print(f"\n[{q_idx+1}/{len(df)}] QID: {qid} ({qtype})")
        print(f"  Q: {question[:80]}...")

        # 1. Retrieve candidates ONCE for this question
        _, retrieved_chunks = retriever.retrieve_documents(
            query=question,
            mode="bm25",
            top_k_docs=20,
        )

        # 2. Build context ONCE for this question
        assembled_context = context_builder.build_context(retrieved_chunks)

        # 3. Query all models with the identical assembled context
        for m in MODELS:
            gen = generators[m]
            t0 = time.perf_counter()
            try:
                ans: RAGAnswer = gen.generate_answer(
                    question=question,
                    assembled_context=assembled_context,
                    temperature=0.0,
                    max_tokens=1024,
                )
                latency = ans.latency_ms
                err = None
            except Exception as ex:
                err = str(ex)
                latency = (time.perf_counter() - t0) * 1000.0
                ans = RAGAnswer(
                    question=question,
                    answer_text=f"ERROR: {err}",
                    is_abstention=False,
                    has_contradiction=False,
                    retrieved_doc_ids=assembled_context.unique_doc_ids,
                    cited_doc_ids=[],
                    cited_chunk_ids=[],
                    citation_validation=validator.validate("", assembled_context),
                    context_tokens=assembled_context.total_tokens,
                    context_passages_count=len(assembled_context.passages),
                    latency_ms=latency,
                    model=m,
                )

            # Evaluate with deterministic judge
            is_correct, completeness, explanation = judge.evaluate(
                question=question,
                question_type=qtype,
                generated_answer=ans.answer_text,
                gold_answer=gold_ans,
                answer_facts=facts,
                is_abstention=ans.is_abstention,
                has_contradiction=ans.has_contradiction,
            )
            score = overall_answer_score(is_correct, completeness)

            entry = {
                "question_id": qid,
                "question_type": qtype,
                "model": m,
                "answer_text": ans.answer_text,
                "answer_correct": is_correct,
                "completeness_pct": round(completeness, 2),
                "overall_score": round(score, 4),
                "cited_doc_ids": ans.cited_doc_ids,
                "citation_precision": round(ans.citation_validation.citation_precision, 4),
                "has_hallucinated_citations": ans.citation_validation.has_hallucinated_citations,
                "is_abstention": ans.is_abstention,
                "has_contradiction": ans.has_contradiction,
                "latency_ms": round(latency, 1),
                "judge_explanation": explanation,
            }
            results_by_model[m].append(entry)

            short_m = m.split("/")[-1]
            print(f"    [{short_m}] Score: {score:.2f} | Latency: {latency:.0f}ms | Correct: {is_correct} | Cited: {len(ans.cited_doc_ids)}")

            # Gentle pause between requests to prevent rate limit spikes
            time.sleep(0.5)

    if hasattr(retriever, "client") and retriever.client is not None:
        try:
            retriever.client.close()
        except Exception:
            pass

    # Aggregate metrics
    summary: Dict[str, Any] = {}
    for m in MODELS:
        items = results_by_model[m]
        n = len(items)
        if n == 0:
            continue
        avg_score = sum(i["overall_score"] for i in items) / n
        correct_pct = sum(1 for i in items if i["answer_correct"]) / n * 100.0
        avg_comp = sum(i["completeness_pct"] for i in items) / n
        avg_prec = sum(i["citation_precision"] for i in items) / n * 100.0
        avg_latency = sum(i["latency_ms"] for i in items) / n
        halluc_cites = sum(1 for i in items if i["has_hallucinated_citations"])

        # Abstention metrics on info_not_found
        inf_items = [i for i in items if i["question_type"] == "info_not_found"]
        inf_abstain_acc = (
            sum(1 for i in inf_items if i["is_abstention"]) / len(inf_items) * 100.0
            if inf_items
            else 0.0
        )

        # Contradiction metrics on conflicting_info
        conf_items = [i for i in items if i["question_type"] == "conflicting_info"]
        conf_det_acc = (
            sum(1 for i in conf_items if i["has_contradiction"]) / len(conf_items) * 100.0
            if conf_items
            else 0.0
        )

        summary[m] = {
            "num_questions": n,
            "overall_score": round(avg_score, 4),
            "answer_correct_pct": round(correct_pct, 2),
            "completeness_pct": round(avg_comp, 2),
            "citation_precision_pct": round(avg_prec, 2),
            "hallucinated_citation_count": halluc_cites,
            "abstention_accuracy_pct": round(inf_abstain_acc, 2),
            "contradiction_accuracy_pct": round(conf_det_acc, 2),
            "mean_latency_ms": round(avg_latency, 1),
        }

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": sample_size,
        "summary": summary,
        "results": results_by_model,
    }

    out_path = BASE_DIR / "evaluation" / "reports" / "groq_model_comparison_30q.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved comparison report to: {out_path}")

    # Print summary table
    print("\n" + "=" * 96)
    print("GROQ MODEL COMPARISON SUMMARY (30 Stratified Questions)")
    print("=" * 96)
    header = f"{'Model':<22} | {'Overall':<7} | {'Correct%':<8} | {'Compl%':<7} | {'CitePrec%':<9} | {'Abstain%':<8} | {'Contra%':<7} | {'Latency':<8}"
    print(header)
    print("-" * 96)
    for m in MODELS:
        s = summary[m]
        print(
            f"{m:<22} | {s['overall_score']:<7.4f} | {s['answer_correct_pct']:<8.1f} | "
            f"{s['completeness_pct']:<7.1f} | {s['citation_precision_pct']:<9.1f} | "
            f"{s['abstention_accuracy_pct']:<8.1f} | {s['contradiction_accuracy_pct']:<7.1f} | "
            f"{s['mean_latency_ms']:<6.0f}ms"
        )
    print("=" * 96)

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare Groq models on EnterpriseRAG-Bench")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of questions (default: 30)")
    args = parser.parse_args()

    run_comparison(sample_size=args.sample_size)
