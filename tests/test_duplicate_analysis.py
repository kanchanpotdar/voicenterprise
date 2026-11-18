"""Regression coverage for conservative corpus duplicate classification."""
import json
import pandas as pd
from data.analyze_corpus_duplicates import run, normalize


def test_cross_id_duplicates_and_conflicts(tmp_path):
    source = tmp_path / "input.parquet"
    output = tmp_path / "report.json"
    pd.DataFrame([
        {"doc_id": "a", "title": "Limits", "content": "Max  10 MB", "source_type": "jira"},
        {"doc_id": "b", "title": "Limits", "content": "Max 10 MB", "source_type": "jira"},
        {"doc_id": "a", "title": "Limits", "content": "Max 20 MB", "source_type": "jira"},
        {"doc_id": "c", "title": "Limits", "content": "max 10 MB", "source_type": "jira"},
    ]).to_parquet(source)
    run(source, output)
    result = json.loads(output.read_text())
    assert result["exact_duplicate_rows"] == 1
    assert result["conflicting_doc_id_groups"] == 1
    assert result["conflicting_rows"] == 2
    assert result["exact_unique_records"] == 3


def test_normalization_preserves_identifiers():
    assert normalize(" A\n B ") == "A B"
    assert normalize("e\u0301") == normalize("é")
    assert normalize("API_Key=10") != normalize("api_key=10")
    assert normalize("10.0") != normalize("100")
