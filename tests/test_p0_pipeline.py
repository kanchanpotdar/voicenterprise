"""Integration tests for Phase 0 deliverables (P0.1, P0.2, P0.3)."""

import json
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def test_p0_1_manifest_integrity():
    manifest_path = BASE_DIR / "data" / "manifest.json"
    assert manifest_path.exists(), "data/manifest.json should exist"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["files"]["questions/test.parquet"]["sha256_verified"] is True
    assert manifest["files"]["documents/test.parquet"]["sha256_verified"] is True
    assert manifest["questions_audit"]["total_rows"] == 500
    assert manifest["documents_audit"]["total_rows"] == 511962
    assert manifest["referential_integrity"]["all_expected_docs_present"] is True
    assert manifest["referential_integrity"]["total_unique_expected_docs"] == 722


def test_p0_2_zero_leakage_between_dev_and_held_out_test():
    splits_dir = BASE_DIR / "data" / "splits"
    dev_df = pd.read_parquet(splits_dir / "questions_dev.parquet")
    test_df = pd.read_parquet(splits_dir / "questions_held_out_test.parquet")

    assert len(dev_df) + len(test_df) == 500
    assert set(dev_df["question_id"]).isdisjoint(set(test_df["question_id"]))

    dev_docs = set(dev_df["expected_doc_ids"].explode().dropna())
    test_docs = set(test_df["expected_doc_ids"].explode().dropna())
    assert dev_docs.isdisjoint(test_docs), "Dev and held-out test must share 0 expected_doc_ids"

    # Verify all 10 categories are represented in both splits
    assert dev_df["question_type"].nunique() == 10
    assert test_df["question_type"].nunique() == 10

    # Verify 5-fold CV indices on dev
    assert set(dev_df["dev_fold"].unique()) == {0, 1, 2, 3, 4}


def test_p0_3_subsets_and_provenance():
    subsets_dir = BASE_DIR / "data" / "subsets"
    sub_manifest_path = subsets_dir / "subset_manifest.json"
    assert sub_manifest_path.exists()
    sub_manifest = json.loads(sub_manifest_path.read_text(encoding="utf-8"))

    dev_sub_info = sub_manifest["subsets"]["corpus_dev_10k.parquet"]
    assert dev_sub_info["conditioned_on_held_out_questions"] is False
    assert dev_sub_info["dev_gold_docs_present"] == dev_sub_info["dev_gold_docs_total"]

    bench_sub_info = sub_manifest["subsets"]["corpus_benchmark_conditioned_12k.parquet"]
    assert bench_sub_info["conditioned_on_held_out_questions"] is True
    assert bench_sub_info["all_gold_docs_present"] == 722
