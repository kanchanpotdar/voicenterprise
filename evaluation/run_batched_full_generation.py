"""Checkpointed generation over the complete 500-question benchmark.

This runner assumes a separately built ``enterprise_rag_full`` collection. It
writes one JSON object per question immediately, allowing safe resume after
network failures or provider rate limits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from rag.llm import OpenAICompatibleLLM, load_env_file
from rag.pipeline import RAGPipeline


def run(questions_path: Path, output_path: Path, qdrant_path: Path, model: str, qdrant_url: str | None = None,
        batch_size: int = 10, max_tokens: int = 1024) -> None:
    load_env_file()
    llm = OpenAICompatibleLLM(
        model=model, base_url="https://api.groq.com/openai/v1", timeout_seconds=90.0
    )
    pipeline = RAGPipeline.from_defaults(
        qdrant_path=qdrant_path, qdrant_url=qdrant_url, collection_name="enterprise_rag_full",
        default_mode="bm25", llm_client=llm, model_name=model,
    )
    df = pd.read_parquet(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["question_id"])
    pending = [r for r in df.to_dict(orient="records") if r["question_id"] not in done]
    print(f"total={len(df)} completed={len(done)} pending={len(pending)}")
    with output_path.open("a", encoding="utf-8") as out:
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            for row in batch:
                qid = str(row["question_id"])
                started = time.perf_counter()
                try:
                    answer = pipeline.query(str(row["question"]), top_k_docs=40,
                                            max_tokens=max_tokens, temperature=0.0)
                    record = {"question_id": qid, "question": row["question"],
                              "expected_doc_ids": row.get("expected_doc_ids", []),
                              "answer": answer.to_dict(), "error": None}
                except Exception as exc:
                    record = {"question_id": qid, "question": row["question"],
                              "expected_doc_ids": row.get("expected_doc_ids", []),
                              "answer": None, "error": repr(exc)}
                record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                out.write(json.dumps(record, default=str) + "\n")
                out.flush()
                print(f"completed {qid}", flush=True)
            print(f"batch {offset // batch_size + 1} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("EnterpriseRAG-Bench/data/questions/test.parquet"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/runs/full_generation.jsonl"))
    parser.add_argument("--qdrant", type=Path, default=Path("data/qdrant_full_bm25"))
    parser.add_argument("--qdrant-url", default=None)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    run(args.questions, args.output, args.qdrant, args.model, args.qdrant_url, args.batch_size, args.max_tokens)
