from __future__ import annotations
import argparse, hashlib, json, unicodedata
from pathlib import Path
import pandas as pd

def h(x: str) -> str:
    return hashlib.sha256(str(x).encode()).hexdigest()

def normalize(text: str) -> str:
    """Preserve case, punctuation and identifiers; normalize canonical Unicode/spacing."""
    return " ".join(unicodedata.normalize("NFC", text).split())

def run(path: Path, out: Path) -> None:
    df = pd.read_parquet(path, columns=["doc_id", "source_type", "title", "content"])
    for c in ("doc_id", "source_type", "title", "content"):
        df[c] = df[c].fillna("").astype(str)
    df["title_norm"] = df["title"].map(normalize)
    df["content_norm"] = df["content"].map(normalize)
    df["content_hash"] = df["content_norm"].map(h)
    df["exact_key"] = df["title_norm"].map(h) + "|" + df["content_hash"]
    exact_dupe_rows = int(df.duplicated("exact_key").sum())
    groups = df.groupby("doc_id", sort=False)
    conflict_groups = groups["exact_key"].nunique()
    conflicts = conflict_groups[conflict_groups > 1]
    sample_key = (df["title_norm"] + "|" + df["content_norm"].str.slice(0, 256)).map(h)
    bucket_sizes = sample_key.value_counts()
    entity_patterns = {"ticket_ids": r"\b[A-Z]{2,8}-\d{2,6}\b", "emails": r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "iso_dates": r"\b202[0-9]-\d{2}-\d{2}\b", "channels": r"#[a-z][a-z0-9_-]{2,45}\b"}
    entity_coverage = {name: int(df["title"].str.contains(pat, regex=True, na=False).add(df["content"].str.contains(pat, regex=True, na=False)).gt(0).sum()) for name, pat in entity_patterns.items()}
    report = {"input_file": str(path), "raw_rows": int(len(df)), "unique_doc_ids": int(df.doc_id.nunique()), "exact_duplicate_rows": exact_dupe_rows, "exact_unique_records": int(df.exact_key.nunique()), "conflicting_doc_id_groups": int(len(conflicts)), "conflicting_rows": int(df[df.doc_id.isin(conflicts.index)].shape[0]), "conflict_group_sizes": {str(k): int(v) for k, v in conflicts.items()}, "source_type_counts": {str(k): int(v) for k, v in df.source_type.value_counts().items()}, "near_duplicate_candidate_buckets": int((bucket_sizes > 1).sum()), "rows_in_near_duplicate_candidate_buckets": int(bucket_sizes[bucket_sizes > 1].sum()), "entity_coverage_rows": entity_coverage, "canonical_output": {"exact_dedup_rows": int(df.exact_key.nunique()), "rows_requiring_version_linking": int(df[df.doc_id.isin(conflicts.index)].shape[0]), "raw_corpus_preserved": True}}
    report["protocol"] = {"version": 2, "exact_key": "NFC and whitespace normalized title plus content; case preserved; excludes doc_id", "candidate_method": "equal title and first 256 normalized content characters; unverified candidates only", "canonical_corpus_written": False}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report, indent=2))

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); run(a.input,a.output)
