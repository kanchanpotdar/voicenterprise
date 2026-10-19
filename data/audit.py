"""Dataset audit and reproducible manifest generator (P0.1).

Verifies:
- File sizes and SHA-256 hashes against Hugging Face Git LFS pointers
- Parquet schemas, row counts, unique IDs, and null/empty string counts
- Source-type distributions across all 511,962 documents
- Question-category distributions across all 500 questions
- Referential integrity: confirms how many expected_doc_ids exist in documents/test.parquet
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import duckdb
import pyarrow.parquet as pq

EXPECTED_LFS_SHA256 = {
    "questions/test.parquet": "e25066f4eff3843dd0f3df0d1348113471e072e75007ffe390a0aa83f2a80af2",
    "documents/test.parquet": "6b0747bf160af9427b12101537d53056ac592ada9831c1a98ae01fa50a8d2a9f",
}


def compute_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def run_dataset_audit(hf_repo_dir: Path, output_manifest_path: Path) -> Dict[str, Any]:
    q_path = hf_repo_dir / "data" / "questions" / "test.parquet"
    d_path = hf_repo_dir / "data" / "documents" / "test.parquet"

    q_sha256 = compute_sha256(q_path)
    d_sha256 = compute_sha256(d_path)

    q_meta = pq.read_metadata(q_path)
    d_meta = pq.read_metadata(d_path)

    con = duckdb.connect(":memory:")

    # 1. Audit questions
    q_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT question_id) AS unique_question_ids,
            SUM(CASE WHEN question_id IS NULL OR TRIM(question_id) = '' THEN 1 ELSE 0 END) AS empty_question_id,
            SUM(CASE WHEN question IS NULL OR TRIM(question) = '' THEN 1 ELSE 0 END) AS empty_question,
            SUM(CASE WHEN gold_answer IS NULL OR TRIM(gold_answer) = '' THEN 1 ELSE 0 END) AS empty_gold_answer
        FROM read_parquet('{q_path.as_posix()}')
        """
    ).fetchone()

    q_categories = con.execute(
        f"""
        SELECT
            question_type,
            COUNT(*) AS question_count,
            MIN( array_length(expected_doc_ids) ) AS min_expected_docs,
            MEDIAN( array_length(expected_doc_ids) ) AS median_expected_docs,
            MAX( array_length(expected_doc_ids) ) AS max_expected_docs,
            SUM( array_length(expected_doc_ids) ) AS total_expected_doc_refs,
            AVG( array_length(answer_facts) ) AS avg_answer_facts
        FROM read_parquet('{q_path.as_posix()}')
        GROUP BY question_type
        ORDER BY question_count DESC, question_type ASC
        """
    ).fetchall()

    # 2. Audit documents
    d_stats = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT doc_id) AS unique_doc_ids,
            SUM(CASE WHEN doc_id IS NULL OR TRIM(doc_id) = '' THEN 1 ELSE 0 END) AS empty_doc_id,
            SUM(CASE WHEN source_type IS NULL OR TRIM(source_type) = '' THEN 1 ELSE 0 END) AS empty_source_type,
            SUM(CASE WHEN title IS NULL OR TRIM(title) = '' THEN 1 ELSE 0 END) AS empty_title,
            SUM(CASE WHEN content IS NULL OR TRIM(content) = '' THEN 1 ELSE 0 END) AS empty_content,
            AVG(LENGTH(content)) AS avg_content_chars,
            MEDIAN(LENGTH(content)) AS median_content_chars,
            MAX(LENGTH(content)) AS max_content_chars
        FROM read_parquet('{d_path.as_posix()}')
        """
    ).fetchone()

    d_sources = con.execute(
        f"""
        SELECT
            source_type,
            COUNT(*) AS doc_count,
            ROUND(AVG(LENGTH(content)), 1) AS avg_chars,
            CAST(MEDIAN(LENGTH(content)) AS INTEGER) AS median_chars
        FROM read_parquet('{d_path.as_posix()}')
        GROUP BY source_type
        ORDER BY doc_count DESC
        """
    ).fetchall()

    # 3. Referential integrity between questions.expected_doc_ids and documents.doc_id
    ref_integrity = con.execute(
        f"""
        WITH expected AS (
            SELECT DISTINCT UNNEST(expected_doc_ids) AS expected_doc_id
            FROM read_parquet('{q_path.as_posix()}')
        )
        SELECT
            COUNT(*) AS total_unique_expected_docs,
            SUM(CASE WHEN d.doc_id IS NOT NULL THEN 1 ELSE 0 END) AS found_in_corpus,
            SUM(CASE WHEN d.doc_id IS NULL THEN 1 ELSE 0 END) AS missing_from_corpus
        FROM expected e
        LEFT JOIN (
            SELECT DISTINCT doc_id FROM read_parquet('{d_path.as_posix()}')
        ) d ON e.expected_doc_id = d.doc_id
        """
    ).fetchone()

    manifest: Dict[str, Any] = {
        "dataset_repo": "onyx-dot-app/EnterpriseRAG-Bench",
        "git_commit": get_git_commit(hf_repo_dir),
        "license": "MIT",
        "files": {
            "questions/test.parquet": {
                "size_bytes": q_path.stat().st_size,
                "sha256": q_sha256,
                "expected_lfs_sha256": EXPECTED_LFS_SHA256["questions/test.parquet"],
                "sha256_verified": q_sha256 == EXPECTED_LFS_SHA256["questions/test.parquet"],
                "num_rows": q_meta.num_rows,
                "num_row_groups": q_meta.num_row_groups,
                "columns": [q_meta.schema.column(i).name for i in range(q_meta.num_columns)],
            },
            "documents/test.parquet": {
                "size_bytes": d_path.stat().st_size,
                "sha256": d_sha256,
                "expected_lfs_sha256": EXPECTED_LFS_SHA256["documents/test.parquet"],
                "sha256_verified": d_sha256 == EXPECTED_LFS_SHA256["documents/test.parquet"],
                "num_rows": d_meta.num_rows,
                "num_row_groups": d_meta.num_row_groups,
                "columns": [d_meta.schema.column(i).name for i in range(d_meta.num_columns)],
            },
        },
        "questions_audit": {
            "total_rows": int(q_stats[0]),
            "unique_question_ids": int(q_stats[1]),
            "empty_question_id": int(q_stats[2]),
            "empty_question": int(q_stats[3]),
            "empty_gold_answer": int(q_stats[4]),
            "categories": [
                {
                    "question_type": row[0],
                    "count": int(row[1]),
                    "min_expected_docs": int(row[2] or 0),
                    "median_expected_docs": float(row[3] or 0.0),
                    "max_expected_docs": int(row[4] or 0),
                    "total_expected_doc_refs": int(row[5] or 0),
                    "avg_answer_facts": round(float(row[6] or 0.0), 2),
                }
                for row in q_categories
            ],
        },
        "documents_audit": {
            "total_rows": int(d_stats[0]),
            "unique_doc_ids": int(d_stats[1]),
            "empty_doc_id": int(d_stats[2]),
            "empty_source_type": int(d_stats[3]),
            "empty_title": int(d_stats[4]),
            "empty_content": int(d_stats[5]),
            "avg_content_chars": round(float(d_stats[6]), 1),
            "median_content_chars": int(d_stats[7]),
            "max_content_chars": int(d_stats[8]),
            "sources": [
                {
                    "source_type": row[0],
                    "doc_count": int(row[1]),
                    "share_pct": round(100.0 * int(row[1]) / int(d_stats[0]), 2),
                    "avg_chars": float(row[2]),
                    "median_chars": int(row[3]),
                }
                for row in d_sources
            ],
        },
        "referential_integrity": {
            "total_unique_expected_docs": int(ref_integrity[0]),
            "found_in_corpus": int(ref_integrity[1]),
            "missing_from_corpus": int(ref_integrity[2]),
            "all_expected_docs_present": int(ref_integrity[2]) == 0,
        },
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    hf_dir = base / "EnterpriseRAG-Bench"
    out_manifest = base / "data" / "manifest.json"
    manifest_data = run_dataset_audit(hf_dir, out_manifest)
    print(json.dumps(manifest_data, indent=2))
