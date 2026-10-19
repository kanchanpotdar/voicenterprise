"""Deterministic corpus subset builder (P0.3).

Addresses Correction #1 (preventing silent held-out leakage into subset construction):
1. Builds `data/subsets/corpus_dev_10k.parquet` (10,000 documents) conditioned ONLY on
   the development question split (`questions_dev.parquet`):
   - Layer 1 (`dev_gold`): All expected_doc_ids from dev questions (~280 docs).
   - Layer 2 (`dev_hard_negative`): Documents in the corpus that share exact normalized
     title prefixes, ticket/PR identifiers, or distinctive title tokens with `dev_gold`
     documents (~2,200 docs) to preserve near-duplicates and conflicting records.
   - Layer 3 (`stratified_background`): Proportional stratified sample across all 9
     `source_type`s to reach exactly 10,000 documents.
   Zero information from `questions_held_out_test.parquet` is used in `corpus_dev_10k.parquet`.

2. Builds `data/subsets/corpus_benchmark_conditioned_12k.parquet` (~12,000 documents),
   which adds `held_out_gold` and `held_out_hard_negative` documents on top of
   `corpus_dev_10k.parquet` with an explicit `inclusion_reason` provenance column and
   records exact file sizes and limitations in `data/subsets/subset_manifest.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import duckdb
import pandas as pd

STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under",
    "prod", "production", "staging", "runbook", "update", "weekly", "notes", "sync",
    "meeting", "thread", "summary", "review", "issue", "ticket", "request", "pull",
    "redwood", "inference", "team", "channel", "general", "random", "internal",
    "status", "plan", "draft", "doc", "document", "guide", "playbook", "checklist",
}


def _extract_distinctive_terms(titles: List[str], max_terms: int = 80) -> List[str]:
    """Extract distinctive alphanumeric/hyphenated tokens from gold document titles."""
    counts: Dict[str, int] = {}
    for title in titles:
        for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]{3,}", title.lower()):
            if tok in STOPWORDS or tok.isdigit():
                continue
            counts[tok] = counts.get(tok, 0) + 1
    # Prefer terms that appear in 1-5 gold titles (specific project/service names)
    ranked = sorted(
        counts.items(),
        key=lambda kv: (1 <= kv[1] <= 5, len(kv[0]), kv[0]),
        reverse=True,
    )
    return [tok for tok, _ in ranked[:max_terms]]


def build_corpus_subsets(
    documents_parquet_path: Path,
    splits_dir: Path,
    output_dir: Path,
    target_dev_subset_size: int = 10000,
    target_dev_hard_negatives: int = 2200,
    target_test_hard_negatives: int = 1500,
) -> Dict[str, Any]:
    dev_q_df = pd.read_parquet(splits_dir / "questions_dev.parquet")
    test_q_df = pd.read_parquet(splits_dir / "questions_held_out_test.parquet")

    dev_gold_ids: Set[str] = set(dev_q_df["expected_doc_ids"].explode().dropna())
    test_gold_ids: Set[str] = set(test_q_df["expected_doc_ids"].explode().dropna())
    all_gold_ids: Set[str] = dev_gold_ids.union(test_gold_ids)

    con = duckdb.connect(":memory:")
    con.execute("PRAGMA threads=4;")

    # Load full corpus metadata into a DuckDB view
    con.execute(
        f"CREATE VIEW docs AS SELECT * FROM read_parquet('{documents_parquet_path.as_posix()}')"
    )

    # Register dev_gold and test_gold tables
    con.register("dev_gold_df", pd.DataFrame({"doc_id": sorted(dev_gold_ids)}))
    con.register("test_gold_df", pd.DataFrame({"doc_id": sorted(test_gold_ids)}))

    # Fetch dev gold docs and their titles to mine hard negatives deterministically
    dev_gold_records = con.execute(
        """
        SELECT d.doc_id, d.source_type, d.title, d.content, 'dev_gold' AS inclusion_reason
        FROM docs d
        INNER JOIN dev_gold_df g ON d.doc_id = g.doc_id
        """
    ).fetchdf()

    test_gold_records = con.execute(
        """
        SELECT d.doc_id, d.source_type, d.title, d.content, 'held_out_gold' AS inclusion_reason
        FROM docs d
        INNER JOIN test_gold_df g ON d.doc_id = g.doc_id
        """
    ).fetchdf()

    dev_terms = _extract_distinctive_terms(dev_gold_records["title"].tolist(), max_terms=60)
    test_terms = _extract_distinctive_terms(test_gold_records["title"].tolist(), max_terms=60)

    def _query_hard_negatives(
        terms: List[str],
        exclude_ids: Set[str],
        limit_n: int,
        reason_label: str,
    ) -> pd.DataFrame:
        if not terms:
            return pd.DataFrame(columns=["doc_id", "source_type", "title", "content", "inclusion_reason"])
        con.register("exclude_df", pd.DataFrame({"doc_id": sorted(exclude_ids)}))
        regex_pattern = "|".join(re.escape(t) for t in terms)
        query = f"""
        SELECT
            d.doc_id,
            d.source_type,
            d.title,
            d.content,
            '{reason_label}' AS inclusion_reason
        FROM docs d
        LEFT JOIN exclude_df ex ON d.doc_id = ex.doc_id
        WHERE ex.doc_id IS NULL
          AND regexp_matches(lower(d.title), '{regex_pattern}')
        ORDER BY md5(d.doc_id)
        LIMIT {int(limit_n)}
        """
        return con.execute(query).fetchdf()

    # Select dev hard negatives WITHOUT excluding test_gold_ids (to remain 100% blind to test_gold_ids!)
    dev_hard_neg_df = _query_hard_negatives(
        dev_terms,
        exclude_ids=dev_gold_ids,
        limit_n=target_dev_hard_negatives,
        reason_label="dev_hard_negative",
    )

    # Compute remaining slots for stratified background noise to reach target_dev_subset_size
    already_in_dev = set(dev_gold_records["doc_id"]).union(set(dev_hard_neg_df["doc_id"]))
    remaining_for_dev = max(0, target_dev_subset_size - len(already_in_dev))

    # Get source distribution of full corpus
    source_counts = con.execute(
        "SELECT source_type, COUNT(*) AS cnt FROM docs GROUP BY source_type ORDER BY cnt DESC"
    ).fetchall()
    total_corpus_docs = sum(r[1] for r in source_counts)

    # Allocate stratified background counts per source_type proportionally
    bg_targets: Dict[str, int] = {}
    allocated = 0
    for idx, (stype, cnt) in enumerate(source_counts):
        if idx == len(source_counts) - 1:
            bg_targets[stype] = remaining_for_dev - allocated
        else:
            share = int(round(remaining_for_dev * (cnt / total_corpus_docs)))
            bg_targets[stype] = share
            allocated += share

    con.register("already_dev_df", pd.DataFrame({"doc_id": sorted(already_in_dev)}))
    bg_dfs: List[pd.DataFrame] = []
    for stype, quota in bg_targets.items():
        if quota <= 0:
            continue
        sub_df = con.execute(
            f"""
            SELECT
                d.doc_id,
                d.source_type,
                d.title,
                d.content,
                'stratified_background' AS inclusion_reason
            FROM docs d
            LEFT JOIN already_dev_df ex ON d.doc_id = ex.doc_id
            WHERE ex.doc_id IS NULL
              AND d.source_type = '{stype}'
            ORDER BY md5('seed42_' || d.doc_id)
            LIMIT {int(quota)}
            """
        ).fetchdf()
        bg_dfs.append(sub_df)

    bg_all_df = pd.concat(bg_dfs, ignore_index=True)

    # Assemble strict dev-conditioned subset (zero knowledge of held-out test questions)
    # Note: deduplicate on (doc_id, source_type, title, content) so distinct versions
    # sharing a doc_id (e.g. conflicting_info records) are BOTH preserved.
    corpus_dev_df = pd.concat(
        [dev_gold_records, dev_hard_neg_df, bg_all_df], ignore_index=True
    )
    corpus_dev_df = (
        corpus_dev_df.drop_duplicates(
            subset=["doc_id", "source_type", "title", "content"], keep="first"
        )
        .sort_values(["doc_id", "source_type", "title"])
        .reset_index(drop=True)
    )

    # Now build the benchmark-conditioned evaluation subset by adding held_out_gold
    # and held_out_hard_negative documents not already in corpus_dev_df
    in_dev_subset_ids = set(corpus_dev_df["doc_id"])
    missing_test_gold_df = test_gold_records[
        ~test_gold_records["doc_id"].isin(in_dev_subset_ids)
    ].copy()

    exclude_for_test_hn = in_dev_subset_ids.union(test_gold_ids)
    test_hard_neg_df = _query_hard_negatives(
        test_terms,
        exclude_ids=exclude_for_test_hn,
        limit_n=target_test_hard_negatives,
        reason_label="held_out_hard_negative",
    )

    corpus_benchmark_df = pd.concat(
        [corpus_dev_df, missing_test_gold_df, test_hard_neg_df], ignore_index=True
    )
    corpus_benchmark_df = (
        corpus_benchmark_df.drop_duplicates(
            subset=["doc_id", "source_type", "title", "content"], keep="first"
        )
        .sort_values(["doc_id", "source_type", "title"])
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    dev_subset_path = output_dir / "corpus_dev_10k.parquet"
    bench_subset_path = output_dir / "corpus_benchmark_conditioned_12k.parquet"

    corpus_dev_df.to_parquet(dev_subset_path, index=False)
    corpus_benchmark_df.to_parquet(bench_subset_path, index=False)

    dev_gold_coverage_in_dev = len(dev_gold_ids.intersection(set(corpus_dev_df["doc_id"])))
    test_gold_coverage_in_dev = len(test_gold_ids.intersection(set(corpus_dev_df["doc_id"])))
    all_gold_coverage_in_bench = len(
        all_gold_ids.intersection(set(corpus_benchmark_df["doc_id"]))
    )

    subset_manifest = {
        "subsets": {
            "corpus_dev_10k.parquet": {
                "path": str(dev_subset_path.relative_to(output_dir.parent.parent)),
                "num_documents": len(corpus_dev_df),
                "size_bytes": dev_subset_path.stat().st_size,
                "size_mb": round(dev_subset_path.stat().st_size / (1024 * 1024), 2),
                "conditioned_on_held_out_questions": False,
                "dev_gold_docs_present": dev_gold_coverage_in_dev,
                "dev_gold_docs_total": len(dev_gold_ids),
                "incidental_held_out_gold_docs_present": test_gold_coverage_in_dev,
                "by_inclusion_reason": (
                    corpus_dev_df["inclusion_reason"].value_counts().to_dict()
                ),
                "by_source_type": (
                    corpus_dev_df["source_type"].value_counts().to_dict()
                ),
            },
            "corpus_benchmark_conditioned_12k.parquet": {
                "path": str(bench_subset_path.relative_to(output_dir.parent.parent)),
                "num_documents": len(corpus_benchmark_df),
                "size_bytes": bench_subset_path.stat().st_size,
                "size_mb": round(bench_subset_path.stat().st_size / (1024 * 1024), 2),
                "conditioned_on_held_out_questions": True,
                "all_gold_docs_present": all_gold_coverage_in_bench,
                "all_gold_docs_total": len(all_gold_ids),
                "methodological_note": (
                    "This subset includes expected_doc_ids and title-matched hard negatives "
                    "for both dev and held-out test questions so held-out questions are "
                    "answerable without indexing all 511,962 documents. For strictly "
                    "unconditioned final evaluation, index the full documents/test.parquet."
                ),
                "by_inclusion_reason": (
                    corpus_benchmark_df["inclusion_reason"].value_counts().to_dict()
                ),
                "by_source_type": (
                    corpus_benchmark_df["source_type"].value_counts().to_dict()
                ),
            },
        }
    }

    manifest_path = output_dir / "subset_manifest.json"
    manifest_path.write_text(json.dumps(subset_manifest, indent=2) + "\n", encoding="utf-8")
    return subset_manifest


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    docs_path = base / "EnterpriseRAG-Bench" / "data" / "documents" / "test.parquet"
    splits_path = base / "data" / "splits"
    out_path = base / "data" / "subsets"
    res = build_corpus_subsets(docs_path, splits_path, out_path)
    print(json.dumps(res, indent=2))
