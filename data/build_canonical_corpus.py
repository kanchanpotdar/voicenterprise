"""Build a provenance-preserving canonical retrieval corpus.

The raw corpus is never modified. Exact duplicates are collapsed; conflicting
versions remain linked by record_group_id. Near-duplicate candidates are
identified with deterministic token-shingle signatures for later review.
"""
from __future__ import annotations

import argparse, hashlib, json, re
from pathlib import Path
import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9_:/.-]+")

def norm(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))

def signature(text: str, width: int = 5) -> str:
    toks = norm(text).split()
    shingles = {" ".join(toks[i:i + width]) for i in range(max(0, len(toks) - width + 1))}
    return hashlib.sha256("\n".join(sorted(shingles)).encode()).hexdigest()

def build(input_path: Path, output_path: Path) -> dict:
    df = pd.read_parquet(input_path)
    required = {"doc_id", "source_type", "title", "content"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    work = df.copy()
    work["title"] = work["title"].fillna("").astype(str)
    work["content"] = work["content"].fillna("").astype(str)
    work["normalized_title"] = work["title"].map(norm)
    work["normalized_content"] = work["content"].map(norm)
    work["content_sha256"] = work["normalized_content"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    work["record_group_id"] = work["doc_id"].map(lambda x: f"record_group:{x}")
    work["exact_duplicate_key"] = work.apply(lambda r: f"{r.doc_id}|{r.content_sha256}|{r.normalized_title}", axis=1)
    before = len(work)
    work = work.drop_duplicates("exact_duplicate_key", keep="first").reset_index(drop=True)
    work["version_index"] = work.groupby("record_group_id").cumcount() + 1
    work["version_count"] = work.groupby("record_group_id")["record_group_id"].transform("size")
    work["has_conflicting_versions"] = work["version_count"] > 1
    work["near_duplicate_signature"] = work["normalized_content"].map(signature)
    work["canonical_priority"] = work["version_index"].map(lambda x: 1.0 / max(1, int(x)))
    work["retrieval_status"] = "canonical_candidate"
    work.loc[work["has_conflicting_versions"], "retrieval_status"] = "versioned_conflict"
    work = work.drop(columns=["normalized_content", "exact_duplicate_key"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work.to_parquet(output_path, index=False)
    summary = {"input_rows": before, "output_rows": len(work), "exact_duplicates_removed": before - len(work), "record_groups": int(work.record_group_id.nunique()), "conflicting_groups": int(work.loc[work.has_conflicting_versions, "record_group_id"].nunique()), "near_duplicate_signatures": int(work.near_duplicate_signature.nunique())}
    output_path.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args(); build(a.input, a.output)
