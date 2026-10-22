"""Command-Line Interface for VoiceEnterprise Phase 2 RAG.

Usage Examples:
  # 1. Ask a single question using BM25 (default)
  .venv/bin/python3 -m rag.cli --query "What are the default size limits for file uploads?"

  # 2. Ask using Hybrid retrieval with debug mode
  .venv/bin/python3 -m rag.cli --query "What is the rollback procedure for Serving Runtime?" --mode hybrid --debug

  # 3. Test without API keys using deterministic Mock LLM
  .venv/bin/python3 -m rag.cli --query "What are the size limits?" --provider mock --debug

  # 4. Run reproducible evaluation on the development set (zero cost with deterministic judge)
  .venv/bin/python3 -m rag.cli --eval --eval-sample-size 20 --mode hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress local SQLite warnings for clean CLI experience
warnings.filterwarnings(
    "ignore",
    message="Local mode is not recommended for collections with more than 20,000 points.*",
    category=UserWarning,
)

import pandas as pd

from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator, RAGAnswer
from rag.llm import BaseLLMClient, MockLLM, OpenAICompatibleLLM
from rag.pipeline import RAGPipeline
from retrieval.indexer import EnterpriseRetriever, RetrievalMode

BASE_DIR = Path(__file__).resolve().parent.parent


def build_pipeline_from_args(args: argparse.Namespace) -> RAGPipeline:
    """Build and validate RAG pipeline from CLI arguments."""
    qdrant_dir = BASE_DIR / "data" / "qdrant_dev_10k"
    if not qdrant_dir.exists():
        sys.exit(
            f"ERROR: Qdrant collection directory not found at {qdrant_dir}.\n"
            f"Run Phase 1 indexing first: python -m retrieval.benchmark_p1"
        )

    # Resolve LLM provider
    provider = args.provider.lower()
    llm_client: BaseLLMClient

    if provider == "mock":
        llm_client = MockLLM()
    elif provider in ("openai", "groq", "auto"):
        from rag.llm import is_valid_key

        openai_key = os.getenv("OPENAI_API_KEY") if is_valid_key(os.getenv("OPENAI_API_KEY")) else None
        groq_key = os.getenv("GROQ_API_KEY") if is_valid_key(os.getenv("GROQ_API_KEY")) else None
        api_key = openai_key or groq_key

        if not api_key:
            if provider == "auto":
                print(
                    "Note: No valid OPENAI_API_KEY or GROQ_API_KEY found in environment. "
                    "Defaulting to deterministic MockLLM for local testing. "
                    "Set OPENAI_API_KEY or GROQ_API_KEY in .env to enforce live LLM.",
                    file=sys.stderr,
                )
                llm_client = MockLLM()
            else:
                sys.exit(
                    f"ERROR: Provider '{provider}' requested, but neither OPENAI_API_KEY nor "
                    f"GROQ_API_KEY is set in environment or .env file."
                )
        else:
            model = args.model
            if groq_key and not openai_key and model == "gpt-4o-mini":
                model = "openai/gpt-oss-120b"
            llm_client = OpenAICompatibleLLM(
                api_key=api_key,
                model=model,
            )
    else:
        sys.exit(f"ERROR: Unknown provider '{provider}'. Choose 'openai', 'groq', 'mock', or 'auto'.")

    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=qdrant_dir,
    )

    context_builder = ContextBuilder(
        max_context_tokens=args.token_budget,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )

    generator = AnswerGenerator(llm_client=llm_client)

    return RAGPipeline(
        retriever=retriever,
        generator=generator,
        context_builder=context_builder,
        default_mode=args.mode,
    )


def handle_single_query(pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    """Execute and display a single question through the RAG pipeline."""
    question = args.query.strip()
    effective_model = getattr(pipeline.generator.llm, "model", args.model)
    print("=" * 88)
    print(f"QUERY: {question}")
    print(f"MODE : {args.mode.upper()} | MODEL: {effective_model} | BUDGET: {args.token_budget} tokens")
    print("=" * 88)

    t0 = time.perf_counter()
    answer: RAGAnswer = pipeline.query(
        question=question,
        mode=args.mode,
        top_k_docs=args.top_k,
    )
    total_elapsed = (time.perf_counter() - t0) * 1000.0

    print("\nFINAL ANSWER:")
    print("-" * 88)
    print(answer.answer_text)
    print("-" * 88)

    print("\nCITATIONS & GROUNDING:")
    val = answer.citation_validation
    print(f"  Valid Cited Docs     : {val.valid_doc_ids or '(none)'}")
    if val.invalid_doc_ids:
        print(f"  WARNING: Invalid/Hallucinated Doc Citations: {val.invalid_doc_ids}")
    print(f"  Citation Precision   : {val.citation_precision*100:.1f}%")
    print(f"  Abstention Detected  : {answer.is_abstention}")
    print(f"  Contradiction Noted  : {answer.has_contradiction}")

    if args.debug or args.show_chunks:
        print("\n" + "=" * 88)
        print("RETRIEVED CANDIDATE CHUNKS (TOP-K)")
        print("=" * 88)
        if answer.retrieved_chunks:
            for idx, ch in enumerate(answer.retrieved_chunks[: args.top_k_display], start=1):
                preview = ch.chunk_text.replace("\n", " ").strip()[:180]
                sec = f" | Section: {ch.section_title}" if ch.section_title else ""
                dt = f" | Date: {ch.latest_date}" if ch.latest_date else ""
                print(f"[{idx}] Score: {ch.score:.4f} | Source: {ch.source_type} | Doc: {ch.doc_id} | Chunk: {ch.chunk_id}")
                print(f"    Title: {ch.normalized_title}{sec}{dt}")
                print(f"    Snippet: \"{preview}...\"\n")
        else:
            print(f"  Candidate Docs: {answer.retrieved_doc_ids}")

    if args.debug or args.show_prompt:
        print("\n" + "=" * 88)
        print("FULL PROMPT TRANSMITTED TO LLM")
        print("=" * 88)
        if answer.system_prompt:
            print("--- [SYSTEM PROMPT] ---")
            print(answer.system_prompt)
            print("-----------------------")
        if answer.user_prompt:
            print("\n--- [USER PROMPT & EVIDENCE CONTEXT] ---")
            print(answer.user_prompt)
            print("----------------------------------------")

    if args.debug:
        print("\n" + "=" * 88)
        print("DEBUG METRICS")
        print("=" * 88)
        print(f"Total Pipeline Latency : {total_elapsed:.1f} ms (LLM: {answer.latency_ms:.1f} ms)")
        print(f"Context Tokens Used    : {answer.context_tokens} tokens across {answer.context_passages_count} passages")
        print(f"Candidate Docs Retrieved: {len(answer.retrieved_doc_ids)} docs: {answer.retrieved_doc_ids[:6]}...")
        if answer.cost_estimate_usd > 0:
            print(f"Estimated Cost         : ${answer.cost_estimate_usd:.6f}")


def handle_evaluation(pipeline: RAGPipeline, args: argparse.Namespace) -> None:
    """Execute evaluation over development questions."""
    from evaluation.eval_rag import run_rag_evaluation

    dev_path = BASE_DIR / "data" / "splits" / "questions_dev.parquet"
    if not dev_path.exists():
        sys.exit(f"ERROR: Dev questions parquet not found at {dev_path}")

    dev_df = pd.read_parquet(dev_path)
    sample_size = args.eval_sample_size or len(dev_df)

    print("=" * 88)
    print(f"PHASE 2 RAG EVALUATION BENCHMARK")
    print(f"Dataset      : {dev_path.name} (n={sample_size} of {len(dev_df)} questions)")
    print(f"Retrieval    : {args.mode.upper()}")
    print(f"Judge Type   : {args.judge.upper()}")
    print("=" * 88)

    output_path = args.output_report
    if not output_path:
        output_path = (
            BASE_DIR
            / "evaluation"
            / "reports"
            / f"p2_rag_eval_{args.mode}_{sample_size}q.json"
        )

    report = run_rag_evaluation(
        pipeline=pipeline,
        questions_df=dev_df,
        mode=args.mode,
        judge_type=args.judge,
        max_questions=args.eval_sample_size,
        output_report_path=output_path,
    )

    ov = report["overall"]
    print("\nAGGREGATE BENCHMARK RESULTS:")
    print("-" * 60)
    print(f"  Overall Answer Score (Score = Correct * Comp): {ov['overall_score']:.2f}")
    print(f"  Answer Correctness                          : {ov['answer_correctness_pct']:.2f}%")
    print(f"  Answer Completeness                         : {ov['answer_completeness_pct']:.2f}%")
    print(f"  Gold Doc Recall@5                           : {ov['recall_at_5']:.2f}%")
    print(f"  Gold Doc Recall@20                          : {ov['recall_at_20']:.2f}%")
    print(f"  Citation Validity Rate                      : {ov['citation_validity_pct']:.2f}%")
    print(f"  Average Query Latency                       : {ov['avg_latency_ms']:.1f} ms")
    print(f"  Total Estimated Generation Cost             : ${ov['total_estimated_cost_usd']:.5f}")
    print(f"  Execution Elapsed Time                      : {report['execution_time_seconds']:.2f} s")
    print("-" * 60)
    print(f"Saved complete reproducible report to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceEnterprise Phase 2 Grounded Text RAG System")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Single question to query the RAG system.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="bm25",
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval strategy: 'bm25' (default baseline), 'dense', or 'hybrid'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "groq", "mock"],
        help="LLM provider: 'auto' (detects API keys), 'openai', 'groq', or 'mock'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of parent documents to retrieve (default: 20).",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=2500,
        help="Maximum prompt token budget for evidence context (default: 2500).",
    )
    parser.add_argument(
        "--max-chunks-per-doc",
        type=int,
        default=2,
        help="Maximum chunks to include per document to ensure diversity (default: 2).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable full verbose inspection: shows retrieval scores, chunks, full prompt, and validation.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Display the exact system prompt and user prompt (including assembled context) sent to the LLM.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Display the top-k retrieved chunks with scores, sources, and text previews.",
    )
    parser.add_argument(
        "--top-k-display",
        type=int,
        default=5,
        help="Number of retrieved chunks to display when inspecting (default: 5).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation benchmark over the development set.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=None,
        help="Number of development questions to evaluate (default: all 193).",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="deterministic",
        choices=["deterministic", "llm"],
        help="Judge type for evaluation: 'deterministic' (free, zero API cost) or 'llm'.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Output path for the evaluation report JSON.",
    )

    args = parser.parse_args()

    if not args.query and not args.eval:
        parser.print_help()
        print("\nError: Please provide either --query \"your question\" or --eval.", file=sys.stderr)
        sys.exit(1)

    pipeline = build_pipeline_from_args(args)
    try:
        if args.eval:
            handle_evaluation(pipeline, args)
        elif args.query:
            handle_single_query(pipeline, args)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
