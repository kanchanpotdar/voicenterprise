# Audit assessment and repair record

Priority is based on data loss and invalid evaluation before ranking, latency,
and maintenance. Historical reports are preserved; fixes do not retroactively
validate their numbers.

| Priority | Finding | Assessment and repair |
| --- | --- | --- |
| P0 | Cross-batch record collisions | Confirmed: 27 source chunks shared 16 IDs. Global conflict normalization now precedes batch ingestion. Legacy resumes fail explicitly. The eight affected source rows were re-chunked with versioned IDs and repaired in Qdrant, verifying fresh points before removing stale IDs. Exact collection count is now 1,622,256. |
| P0 | Chunk overlap exceeds body limit | Confirmed; overlap paragraphs are discarded until the next paragraph fits. Invalid chunk/overlap settings fail explicitly. Historical Parquet chunks are not silently regenerated. |
| P1 | Truncated answers marked correct | Confirmed; strict correctness rejects truncation. |
| P1 | Fabrication accepted with unrelated caveat | Confirmed; no-context info-not-found correctness now requires every sentence to be an abstention. Mixed factual/caveated answers cannot establish correctness without grounding. This remains a lexical heuristic, not semantic proof. |
| P1 | Any matching number passes citation support | Confirmed; every extracted quantity must occur in cited evidence. |
| P1 | Versioned/chunk citations lost | Confirmed; regex accepts versioned chunk IDs and sentence-level parsing resolves their parent IDs. Citation syntax is removed before number extraction. |
| P1 | Empty verification and regressing revisions | Confirmed; empty answers require revision, empty/worse revisions are rejected, supersession inflections are recognized. |
| P1 | Legacy fast scorers | `run_fast_model_benchmark.py` used the old heuristic and now uses audit scoring. `benchmark_fast_model.py` already used the corrected deterministic judge; the audit overgeneralized. |
| P1 | Live/cached hybrid candidate mismatch | Confirmed; live hybrid uses max(requested chunks,160) per branch, and cache generation includes dense_k40. Old retrieval caches are invalidated by a protocol marker. |
| P1 | Multi-query order lost in context | Confirmed; returned chunks carry fused document scores. Context sorting preserves fused ranking. |
| P1 | max_tokens absent from generation key | Confirmed; token budget now contributes to the key, invalidating old keys. |
| P1 | Prompt coverage mislabeled retrieval recall | Confirmed; pipeline records the actual retrieved document ranking separately from context assembly. |
| P1 | Voice timing, double playback, fake files | Confirmed; one playback per requested response; elapsed completion/launch timing includes generation; no fabricated WAV fallback. This prototype still does not measure acoustic onset or overlap generation and TTS. |
| P2 | Dev folds split connected components | Confirmed; fold assignment now allocates whole components. Existing frozen split artifacts are preserved; regenerate folds in a new output directory before CV. |
| P2 | Tiny context budgets overflow on headers | Confirmed; builders return empty evidence if even one formatted passage cannot fit after trimming. |
| P2 | Invalid dates / omitted 2030 | Confirmed; ISO dates are calendar-validated, and matching includes 2030. |
| P2 | Silent provider model substitution | Confirmed; implicit Groq gpt-4o-mini replacement now raises a configuration error. Base URL alone no longer counts as a credential. |
| P2 | Partial operational artifacts | Confirmed. Old embedded BM25 checkpoint is not the Docker checkpoint. Generation remains 73 successful / 427 quota failures. Payload enrichment remains paused/partial; these are not completed quality evaluations. |
| P2 | Large Qdrant blobs in Git history | Not reproduced by local tracked-path/history checks on available refs. No history rewrite performed. New ignore rules prevent accidental store/run/recovery-bundle additions. |
| P2 | Stale README | Corrected test badge, implementation scope, deprecated quality/TTFA claims; added this assessment. |
| P3 | 128 static findings | Ruff E9,F,B,PLE fixes: unused names/imports, duplicate set entries, explicit strict zip. Clean on ingestion/retrieval/rag/voice/evaluation/data/tests. |

## Validation

- Full suite: 98 passed.
- Compileall: passed for application/source packages.
- Ruff: `ruff check ingestion retrieval rag voice evaluation data tests --select E9,F,B,PLE` passed.
- Independent offline probes are now regression assertions in `tests/test_audit_repairs.py`.
- Real collision repair reports are under `data/canonical/conflicting_record_repair/`.
- No new external LLM generation, full-corpus reingestion, or frozen-split overwrite.

## Operational follow-up

The existing batch files and old reports remain historical evidence. Use protocol-2
ingestion in a new output directory for a full rebuild; do not resume an old
batch-local-ID manifest. After modifying collection membership, the old enrichment
cursor is not sufficient to certify coverage: perform a new full coverage pass
before claiming all points are enriched. Canonical metadata is optional for search.
New retrieval and generation baselines must be measured against the repaired
collection before making quality comparisons. The four conflicting IDs alone do
not establish that one version supersedes another.
