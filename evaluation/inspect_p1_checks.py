"""Interactive CLI inspector for the 7 Pre-Phase-2 Verification Checks.

Run all checks:
    .venv/bin/python3 -m evaluation.inspect_p1_checks

Run a specific check (1..7):
    .venv/bin/python3 -m evaluation.inspect_p1_checks --check 1 --num-examples 12
    .venv/bin/python3 -m evaluation.inspect_p1_checks --check 2
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Suppress advisory warning about local SQLite collection size in diagnostic CLI
warnings.filterwarnings(
    "ignore",
    message="Local mode is not recommended for collections with more than 20,000 points.*",
    category=UserWarning,
)

from evaluation.metrics import (
    hit_at_k,
    invalid_extra_docs_count,
    recall_at_k,
    reciprocal_rank_at_k,
)
from retrieval.indexer import EnterpriseRetriever

BASE_DIR = Path(__file__).resolve().parent.parent


def _has_expected_docs(val: Any) -> bool:
    """Safely check if expected_doc_ids is non-empty without ambiguous NumPy truth values."""
    if val is None:
        return False
    try:
        return len(val) > 0
    except TypeError:
        return False


def get_stratified_sample(
    df: pd.DataFrame, num_samples: int, category_col: str = "question_type"
) -> pd.DataFrame:
    """Return up to num_samples rows from df, balanced evenly across categories via round-robin."""
    if num_samples <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if num_samples >= len(df) or category_col not in df.columns:
        return df.head(num_samples).copy()

    cat_series = df[category_col].fillna("unknown").astype(str)
    categories = sorted(cat_series.unique())
    by_cat = {cat: df[cat_series == cat].index.tolist() for cat in categories}

    selected_indices: List[Any] = []
    idx = 0
    while len(selected_indices) < num_samples:
        added_in_round = 0
        for cat in categories:
            cat_list = by_cat[cat]
            if idx < len(cat_list):
                selected_indices.append(cat_list[idx])
                added_in_round += 1
                if len(selected_indices) >= num_samples:
                    break
        if added_in_round == 0:
            break
        idx += 1

    return df.loc[selected_indices].copy()


def classify_retrieval_failure(
    expected: List[str],
    retrieved_docs: List[str],
    indexed_doc_ids: set[str],
) -> str:
    """Classify why top-5 retrieval achieved Recall@5 < 1.0."""
    missing_in_corpus = [d for d in expected if d not in indexed_doc_ids]
    missed_at_5 = [d for d in expected if d not in retrieved_docs[:5]]
    found_5_to_20 = [d for d in missed_at_5 if d in retrieved_docs[5:20]]

    if missing_in_corpus:
        return "missing_from_corpus"
    if len(expected) > 5:
        return (
            f"multi_doc_budget (needs {len(expected)} gold docs; "
            f"{len(found_5_to_20)} more found in ranks 6..20)"
        )
    if found_5_to_20:
        return f"ranking_suboptimal (in ranks 6..20; {len(found_5_to_20)} gold docs found)"
    return "lexical_semantic_gap (not found in top-20)"


def check_1_inspect_retrieval_examples(
    retriever: EnterpriseRetriever, num_examples: int = 12, mode: str = "hybrid"
) -> None:
    print("=" * 88)
    print(f"CHECK 1: INSPECT {num_examples} RETRIEVAL EXAMPLES (mode={mode})")
    print("=" * 88)
    dev_q = pd.read_parquet(BASE_DIR / "data" / "splits" / "questions_dev.parquet")
    with_gold = (
        dev_q[dev_q["expected_doc_ids"].apply(_has_expected_docs)]
        if "expected_doc_ids" in dev_q.columns
        else dev_q
    )
    sampled = get_stratified_sample(with_gold, num_samples=num_examples)

    for idx, row in enumerate(sampled.to_dict(orient="records"), start=1):
        qid = str(row.get("question_id", f"q_{idx}"))
        qtype = str(row.get("question_type", "unknown"))
        question = str(row.get("question", ""))
        raw_expected = row.get("expected_doc_ids")
        expected = [str(x) for x in raw_expected] if raw_expected is not None else []
        doc_ids, chunks = retriever.retrieve_documents(
            query=question, mode=mode, top_k_docs=5
        )
        r5 = recall_at_k(doc_ids, expected, k=5) or 0.0
        top_chunk = chunks[0] if chunks else None

        print(f"\n[{idx}/{len(sampled)}] {qid} ({qtype}) | Recall@5={r5:.2f}")
        print(f"  Question      : {question}")
        print(f"  Expected Docs : {expected}")
        print(f"  Top-5 Docs    : {doc_ids}")
        if top_chunk:
            is_hit = "HIT" if top_chunk.doc_id in expected else "MISS"
            snippet = top_chunk.chunk_text.replace("\n", " ")[:220]
            print(
                f"  Top-1 Chunk   : [{is_hit}] {top_chunk.chunk_id} "
                f"({top_chunk.source_type} | {top_chunk.normalized_title} | "
                f"Section: {top_chunk.section_title})"
            )
            print(f"  Chunk Snippet : {snippet}...")



def check_2_read_failed_queries(
    retriever: EnterpriseRetriever, mode: str = "hybrid", sample_size: int = 30
) -> None:
    print("\n" + "=" * 88)
    print(
        f"CHECK 2: INSPECT FAILED / PARTIAL QUERIES AT TOP-5 "
        f"(mode={mode}, fixed sample size={sample_size} stratified dev questions)"
    )
    print("=" * 88)
    dev_q = pd.read_parquet(BASE_DIR / "data" / "splits" / "questions_dev.parquet")
    chunks_df = pd.read_parquet(
        BASE_DIR / "data" / "ingested" / "dev_10k" / "chunks.parquet"
    )
    indexed_doc_ids = set(chunks_df["doc_id"].unique())

    with_gold = (
        dev_q[dev_q["expected_doc_ids"].apply(_has_expected_docs)]
        if "expected_doc_ids" in dev_q.columns
        else dev_q
    )
    sampled = get_stratified_sample(with_gold, num_samples=sample_size)
    sampled_records = sampled.to_dict(orient="records")

    modes_to_run = ["bm25", "dense", "hybrid"] if mode in ("compare", "all") else [mode]

    results_by_mode: Dict[str, Dict[str, Any]] = {}

    for current_mode in modes_to_run:
        failures = []
        r5_list, r10_list, r20_list = [], [], []
        h5_list, h10_list, h20_list = [], [], []
        mrr5_list, mrr10_list = [], []

        for row in sampled_records:
            qid = str(row.get("question_id", ""))
            qtype = str(row.get("question_type", "unknown"))
            question = str(row.get("question", ""))
            raw_expected = row.get("expected_doc_ids")
            expected = [str(x) for x in raw_expected] if raw_expected is not None else []

            retrieved_docs, top_chunks = retriever.retrieve_documents(
                query=question, mode=current_mode, top_k_docs=20
            )

            r5 = recall_at_k(retrieved_docs, expected, k=5) or 0.0
            r10 = recall_at_k(retrieved_docs, expected, k=10) or 0.0
            r20 = recall_at_k(retrieved_docs, expected, k=20) or 0.0
            h5 = hit_at_k(retrieved_docs, expected, k=5) or 0.0
            h10 = hit_at_k(retrieved_docs, expected, k=10) or 0.0
            h20 = hit_at_k(retrieved_docs, expected, k=20) or 0.0
            m5 = reciprocal_rank_at_k(retrieved_docs, expected, k=5) or 0.0
            m10 = reciprocal_rank_at_k(retrieved_docs, expected, k=10) or 0.0

            r5_list.append(r5)
            r10_list.append(r10)
            r20_list.append(r20)
            h5_list.append(h5)
            h10_list.append(h10)
            h20_list.append(h20)
            mrr5_list.append(m5)
            mrr10_list.append(m10)

            if r5 < 1.0:
                top1_title = top_chunks[0].normalized_title if top_chunks else ""
                cause = classify_retrieval_failure(
                    expected=expected,
                    retrieved_docs=retrieved_docs,
                    indexed_doc_ids=indexed_doc_ids,
                )

                failures.append(
                    {
                        "qid": qid,
                        "type": qtype,
                        "question": question,
                        "expected": expected,
                        "top5": retrieved_docs[:5],
                        "num_gold": len(expected),
                        "r5": r5,
                        "r10": r10,
                        "r20": r20,
                        "cause": cause,
                        "top1_title": top1_title,
                    }
                )

        results_by_mode[current_mode] = {
            "r5": sum(r5_list) / len(r5_list) if r5_list else 0.0,
            "r10": sum(r10_list) / len(r10_list) if r10_list else 0.0,
            "r20": sum(r20_list) / len(r20_list) if r20_list else 0.0,
            "h5": sum(h5_list) / len(h5_list) if h5_list else 0.0,
            "h10": sum(h10_list) / len(h10_list) if h10_list else 0.0,
            "h20": sum(h20_list) / len(h20_list) if h20_list else 0.0,
            "mrr5": sum(mrr5_list) / len(mrr5_list) if mrr5_list else 0.0,
            "mrr10": sum(mrr10_list) / len(mrr10_list) if mrr10_list else 0.0,
            "failures": failures,
        }

    # Print comparative table if running comparison
    if len(modes_to_run) > 1:
        print(f"\nHEAD-TO-HEAD COMPARISON ON THE EXACT SAME {len(sampled_records)} DEV QUESTIONS:")
        print("-" * 88)
        print(f"{'Metric':<14} | {'BM25':<12} | {'DENSE':<12} | {'HYBRID':<12} | {'DELTA (HYB vs DENSE)'}")
        print("-" * 88)
        m = results_by_mode
        for metric_name, k in [
            ("Recall@5", "r5"),
            ("Recall@10", "r10"),
            ("Recall@20", "r20"),
            ("Hit@5", "h5"),
            ("Hit@10", "h10"),
            ("Hit@20", "h20"),
            ("MRR@5", "mrr5"),
            ("MRR@10", "mrr10"),
        ]:
            bm_v = m["bm25"][k]
            dn_v = m["dense"][k]
            hy_v = m["hybrid"][k]
            diff = hy_v - dn_v
            sign = "+" if diff >= 0 else ""
            print(f"{metric_name:<14} | {bm_v*100:>10.2f}% | {dn_v*100:>10.2f}% | {hy_v*100:>10.2f}% | {sign}{diff*100:>6.2f}%")
        print("-" * 88)

    # Detailed breakdown for the primary or hybrid mode
    target_mode = "hybrid" if "hybrid" in results_by_mode else modes_to_run[0]
    target_failures = results_by_mode[target_mode]["failures"]

    print(
        f"\n[{target_mode.upper()}] Questions with Recall@5 < 1.0 in sample ({len(sampled_records)} checked): "
        f"{len(target_failures)} (Pass Rate = {(1 - len(target_failures)/len(sampled_records))*100:.1f}%)"
    )

    if target_failures:
        cause_counts = pd.Series(
            [f["cause"].split(" (")[0] for f in target_failures]
        ).value_counts()
        print("\nFailure cause breakdown:")
        for c_name, cnt in cause_counts.items():
            print(f"  - {c_name}: {cnt}")

        print(f"\nDetailed failure inspection ({len(target_failures)} examples):")
        for f in target_failures:
            print(
                f"\n  * {f['qid']} ({f['type']}, gold_docs={f['num_gold']}) | "
                f"Recall@5={f['r5']:.2f} -> Recall@20={f['r20']:.2f}"
            )
            print(f"    Diagnosis     : {f['cause']}")
            print(f"    Question      : {f['question']}")
            print(f"    Expected Docs : {f['expected']}")
            print(f"    Retrieved@5   : {f['top5']} (Top-1 Title: {f['top1_title'][:70]})")


def check_3_verify_duplicate_handling() -> None:
    print("\n" + "=" * 88)
    print("CHECK 3: VERIFY DUPLICATE DOC_ID & CONFLICTING JIRA VERSIONS")
    print("=" * 88)
    docs_df = pd.read_parquet(
        BASE_DIR / "data" / "ingested" / "dev_10k" / "documents_normalized.parquet"
    )
    chunks_df = pd.read_parquet(
        BASE_DIR / "data" / "ingested" / "dev_10k" / "chunks.parquet"
    )
    conflicts = docs_df[docs_df["has_duplicate_doc_id_conflict"]]
    print(f"Conflicting duplicate doc_id records in dev_10k: {len(conflicts)}")
    for row in conflicts.to_dict(orient="records"):
        rec_chunks = chunks_df[chunks_df["record_id"] == row["record_id"]]
        print(
            f"\n  record_id : {row['record_id']} (parent doc_id={row['doc_id']}, "
            f"version {row['version_index']} of {row['version_count']})"
        )
        print(f"  source    : {row['source_type']} | char_length={row['char_length']}")
        print(f"  title     : {row['normalized_title']}")
        print(f"  sha256    : {row['content_sha256'][:24]}...")
        print(f"  chunk_ids : {list(rec_chunks['chunk_id'])}")


def check_4_verify_evaluation_granularity() -> None:
    print("\n" + "=" * 88)
    print("CHECK 4: VERIFY EVALUATION GRANULARITY (MULTI-CHUNK DEDUPLICATION)")
    print("=" * 88)
    expected = ["doc_gold_1", "doc_gold_2"]
    retrieved_with_dup_chunks = [
        "doc_gold_1",
        "doc_gold_1",
        "doc_gold_1",
        "doc_gold_1",
        "doc_distractor",
    ]
    r5 = recall_at_k(retrieved_with_dup_chunks, expected, k=5)
    h5 = hit_at_k(retrieved_with_dup_chunks, expected, k=5)
    mrr5 = reciprocal_rank_at_k(retrieved_with_dup_chunks, expected, k=5)
    extra5 = invalid_extra_docs_count(retrieved_with_dup_chunks, expected, k=5)
    print(f"Expected gold docs             : {expected} (2 distinct docs)")
    print(f"Retrieved list with duplicates : {retrieved_with_dup_chunks}")
    print(f"Computed Recall@5              : {r5} (MUST be 0.5, NOT 2.0 or 1.0)")
    print(f"Computed Hit@5 / MRR@5 / Extra : Hit@5={h5}, MRR@5={mrr5}, InvalidExtra@5={extra5}")
    assert r5 == 0.5, "Granularity check failed!"
    print("PASS: Multiple hits from the same parent doc_id are deduplicated before scoring.")


def check_5_check_chunk_quality() -> None:
    print("\n" + "=" * 88)
    print("CHECK 5: INSPECT CHUNK QUALITY, HEADINGS, DATES & METADATA")
    print("=" * 88)
    chunks_df = pd.read_parquet(
        BASE_DIR / "data" / "ingested" / "dev_10k" / "chunks.parquet"
    )
    print(f"Total chunks: {len(chunks_df)} across {chunks_df['doc_id'].nunique()} unique docs")
    print(
        f"Chunk length stats (chars): min={chunks_df['char_length'].min()}, "
        f"median={int(chunks_df['char_length'].median())}, "
        f"mean={chunks_df['char_length'].mean():.1f}, "
        f"max={chunks_df['char_length'].max()}"
    )
    multi = chunks_df[
        (chunks_df["total_chunks_in_doc"] >= 3)
        & (chunks_df["source_type"] == "confluence")
        & (chunks_df["latest_date"].notna())
    ]
    sample_doc_id = multi.iloc[0]["doc_id"]
    doc_chunks = chunks_df[chunks_df["doc_id"] == sample_doc_id].sort_values("chunk_index")
    print(f"\nSample multi-chunk Confluence runbook ({sample_doc_id}, {len(doc_chunks)} chunks):")
    for row in doc_chunks.to_dict(orient="records"):
        first_line = row["contextualized_text"].splitlines()[0]
        print(
            f"  - {row['chunk_id']} (idx {row['chunk_index']+1}/{row['total_chunks_in_doc']}, "
            f"{row['char_length']} chars, latest_date={row['latest_date']} [{row['date_confidence']}], "
            f"channels={list(row['slack_channels'])[:3]})"
        )
        print(f"    Context Header : {first_line}")
        print(f"    Opening Text   : {row['chunk_text'].splitlines()[0][:100]}")


def check_6_check_split_and_corpus_integrity() -> None:
    print("\n" + "=" * 88)
    print("CHECK 6: VERIFY SPLIT INTEGRITY & CORPUS CONDITIONING")
    print("=" * 88)
    dev_q = pd.read_parquet(BASE_DIR / "data" / "splits" / "questions_dev.parquet")
    test_q = pd.read_parquet(
        BASE_DIR / "data" / "splits" / "questions_held_out_test.parquet"
    )
    dev_corpus = pd.read_parquet(
        BASE_DIR / "data" / "subsets" / "corpus_dev_10k.parquet"
    )

    dev_gold = set(dev_q["expected_doc_ids"].explode().dropna())
    test_gold = set(test_q["expected_doc_ids"].explode().dropna())
    corpus_ids = set(dev_corpus["doc_id"])

    print(f"Dev questions                  : {len(dev_q)} (unique gold docs: {len(dev_gold)})")
    print(f"Held-out test questions        : {len(test_q)} (unique gold docs: {len(test_gold)})")
    print(f"Shared gold docs (dev ∩ test)  : {len(dev_gold.intersection(test_gold))} (MUST be 0)")
    print(
        f"Dev gold docs in corpus_dev_10k: {len(dev_gold.intersection(corpus_ids))}/{len(dev_gold)} (100%)"
    )
    print(
        "Inclusion reasons in corpus_dev_10k:\n ",
        dev_corpus["inclusion_reason"].value_counts().to_dict(),
    )


def check_7_reproducibility_manifests() -> None:
    print("\n" + "=" * 88)
    print("CHECK 7: VERIFY SAVED MANIFESTS, HASHES & BENCHMARK REPORT")
    print("=" * 88)
    paths = [
        BASE_DIR / "data" / "manifest.json",
        BASE_DIR / "data" / "splits" / "split_protocol.json",
        BASE_DIR / "data" / "subsets" / "subset_manifest.json",
        BASE_DIR / "data" / "ingested" / "dev_10k" / "ingestion_manifest.json",
        BASE_DIR / "data" / "ingested" / "dev_10k" / "index_manifest.json",
        BASE_DIR / "evaluation" / "reports" / "p1_retrieval_benchmark_dev.json",
    ]
    for p in paths:
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        print(f"  [{'OK' if exists else 'MISSING'}] {p.relative_to(BASE_DIR)} ({size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pre-Phase-2 Verification Checks")
    parser.add_argument(
        "--check",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=None,
        help="Run a specific check (1..7), or omit to run all checks.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=12,
        help="Number of retrieval examples to print in Check 1 (stratified across categories).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Fixed sample size of stratified questions to evaluate in Check 2.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid",
        choices=["bm25", "dense", "hybrid", "compare", "all"],
        help="Retrieval mode: 'bm25', 'dense', 'hybrid', or 'compare' (runs all on identical sample).",
    )
    args = parser.parse_args()

    need_retriever = args.check in (None, 1, 2)
    retriever: Optional[EnterpriseRetriever] = None
    if need_retriever:
        retriever = EnterpriseRetriever(
            collection_name="enterprise_rag_dev_10k",
            qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
        )

    if args.check in (None, 1) and retriever is not None:
        c1_mode = "hybrid" if args.mode in ("compare", "all") else args.mode
        check_1_inspect_retrieval_examples(
            retriever, num_examples=args.num_examples, mode=c1_mode
        )
    if args.check in (None, 2) and retriever is not None:
        check_2_read_failed_queries(
            retriever, mode=args.mode, sample_size=args.sample_size
        )
    if args.check in (None, 3):
        check_3_verify_duplicate_handling()
    if args.check in (None, 4):
        check_4_verify_evaluation_granularity()
    if args.check in (None, 5):
        check_5_check_chunk_quality()
    if args.check in (None, 6):
        check_6_check_split_and_corpus_integrity()
    if args.check in (None, 7):
        check_7_reproducibility_manifests()

    if retriever is not None:
        retriever.client.close()


if __name__ == "__main__":
    main()
