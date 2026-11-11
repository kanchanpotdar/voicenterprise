"""Build an OFFLINE retrieval cache for the 178 answerable dev questions.

Local only: Qdrant (local path) + local embedding model. No LLM / API calls. HF hub access is disabled.
Output: evaluation/reports/cache/offline_retrieval_chunks.json.gz  (new file; nothing is overwritten)

Modes cached (no diversity filter, no query expansion):
    bm25_k20, bm25_k40, hybrid_k40
Resumable; each record stores the chunks exactly as returned by `retrieve_documents`.
"""
from __future__ import annotations

import gzip
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from evaluation.evaluation_harness import load_development_benchmark  # noqa: E402
from retrieval.indexer import EnterpriseRetriever  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE / "evaluation" / "reports" / "cache"
CACHE_PATH = CACHE_DIR / "offline_retrieval_chunks.json.gz"
SETUPS = {"bm25_k20": ("bm25", 20), "bm25_k40": ("bm25", 40), "hybrid_k40": ("hybrid", 40)}


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with gzip.open(CACHE_PATH, "rt", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp.replace(CACHE_PATH)


def main() -> None:
    _, df_a, _ = load_development_benchmark()
    retriever = EnterpriseRetriever(collection_name="enterprise_rag_dev_10k", qdrant_path=BASE / "data" / "qdrant_dev_10k")
    cache = load_cache()
    t0 = time.time()
    for i, row in enumerate(df_a.itertuples()):
        if row.question_id in cache and all(k in cache[row.question_id] for k in SETUPS):
            continue
        rec = {}
        for name, (mode, k) in SETUPS.items():
            ids, chunks = retriever.retrieve_documents(query=row.question, mode=mode, top_k_docs=k)
            rec[name] = {"doc_ids": ids, "chunks": [asdict(c) for c in chunks]}
        cache[row.question_id] = rec
        if (i + 1) % 10 == 0:
            save_cache(cache)
            print(f"[{i+1}/{len(df_a)}] {time.time()-t0:.0f}s", flush=True)
    save_cache(cache)
    print("saved", CACHE_PATH, len(cache), "questions")


if __name__ == "__main__":
    main()
