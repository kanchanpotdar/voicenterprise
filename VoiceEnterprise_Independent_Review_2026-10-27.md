# VoiceEnterprise: independent history, experiment, and code review

Reviewed 27 October 2026. All commit times below are IST (UTC+05:30).

## Assessment

The project has progressed from a dataset/retrieval foundation into a substantial RAG experiment framework. Its strongest work is the explicit dataset protocol, preservation of duplicate document versions, measured sparse/dense retrieval, and growing separation of retrieval, prompt coverage, answer completeness, and citation quality.

The next milestone should be **reliable, auditable answers**, before additional voice or agent architecture. There is evidence that the new corroborated hybrid preserves more relevant documents in prompts across the 178-question development set. There is not yet reliable evidence that it produces better answers overall. Evaluation bugs and scoring drift currently make that conclusion unsafe.

This review inspected every commit reachable from main, the changes to tracked files, new local Python modules/tests, saved experiment reports, and selected development source documents. GitHub `refs/heads/main` was checked with `git ls-remote`: it matches local HEAD, `cf6f70818220a2ed42cdd52b26b2619eb57fea4f`. There are exactly four main commits, with no merges. Main has not advanced beyond the local base.

No application code was changed, no commits or pushes were made, no existing reports/caches were overwritten, and no external LLM calls were made. This review adds this document, a synthetic offline probe script, and a separate evidence JSON. Tests generated ordinary temporary/cache files. The existing Phase 0 integrity test loads the held-out Parquet to check ID/category/document separation; I did not inspect held-out questions/answers, tune against them, or run held-out answer evaluation. Historical unit-test execution cannot be independently reconstructed without old run logs or CI; test counts below establish what the snapshots contain.

## 1. Complete main-branch history

| Commit and time | What actually entered the repository | Tests/evidence at that snapshot |
|---|---|---|
| `a31b7ea`, 19 Oct 18:13:46 | Initial 24-file snapshot: dataset hash/schema/reference audit; connected-component dev/held-out split; dev-conditioned corpus and hard negatives; document normalization/version preservation; date/entity extraction; structure-aware chunking; local Qdrant sparse BM25 and dense BGE index; retrieval metrics and benchmark entry point. This root commit already contains Phases 0 and 1, so earlier implementation steps are not individually recoverable from Git. | Four test files, 14 test functions. Manifest and split/subset protocols are tracked. The initial `pyproject.toml` has pytest configuration only, without declared package dependencies. |
| `6eb3937`, 19 Oct 18:15:26 | README formalization and academic presentation; no runtime code changes. | Same four files / 14 test functions. No evidence of a new experiment introduced by this commit. |
| `6c2c455`, 21 Oct 19:10:29 | Phase 1 benchmark/diagnostic artifacts, ingestion/index manifests, stratified diagnostic tooling, dependency declarations, README expansion. Fixed NumPy-array handling in `evaluate_retrieval_predictions` by removing ambiguous `raw_exp or []` truth testing. | Five files / 19 test functions. Saved full dev retrieval report and 30-question diagnostic sample. |
| `cf6f708`, 22 Oct 23:13:27 | Text RAG orchestration, context assembly, grounded generation, provider client, citation validation, CLI; candidate-depth/diversity experiments and failure audits; model comparison and milestone runners; streaming LLM support; Groq Whisper STT and macOS TTS prototype; numerous reports. | Seven files / 47 test functions. Retrieval depth, RAG smoke/milestone, model-comparison, and voice reports described below. Commit title says “verified baselines,” but report sample sizes and measurement limitations still matter. |

The much larger Phase 3–6 research/audit work is **local and uncommitted**, not additional Git history. Filesystem timestamps and report timestamps can suggest order; they cannot provide commit-level provenance for these changes.

## 2. Experiments recovered from code and artifacts

### Dataset and ingestion foundation

- Source manifest pins dataset revision `69916e31c68aa5963c00248fd7f0bc12d04fd235`: 511,962 rows, 511,958 unique document IDs, 500 questions, nine source types, 722 distinct gold document IDs.
- Four duplicate IDs have divergent records. Normalization preserves conflicting versions, rather than silently dropping one. The development corpus preserves both rows for one such ID.
- Main development split: 193 questions, including 178 with gold document IDs; 307 held-out. Main split groups questions sharing gold documents into connected components, preventing shared gold IDs across dev/held-out.
- Development corpus: 10,001 rows / 10,000 IDs, including dev gold, 2,200 hard-negative rows, and source-stratified background. This is a **development-conditioned 10k corpus**, not a demonstrated 512k-document deployment.
- Separate approximately 12k corpus deliberately includes held-out gold documents. It is not an unbiased test corpus for generalization claims.
- Ingestion manifest uses 2,000-character chunks / 200-character overlap and records 34,354 chunks, with maximum observed body size 2,187 characters. The low-level chunker defaults to 1,400 characters, while the orchestration explicitly passes 2,000; preserve the actual configuration when comparing runs.

### Retrieval baselines

`p1_retrieval_benchmark_dev.json` evaluates 193 questions, with recall averaged over the 178 gold-labelled questions:

| Retriever | Recall@5 | Recall@10 | Recall@20 | MRR@10 |
|---|---:|---:|---:|---:|
| BM25 | 82.64% | 91.77% | 94.03% | 0.8353 |
| Dense BGE-small | 70.19% | 75.30% | 78.56% | 0.6751 |
| Original hybrid RRF | 85.09% | 89.04% | 92.98% | 0.8134 |

Dense retrieval complements lexical discovery but is weaker alone here. Original hybrid improves @5 and loses to BM25 @10/@20. A later full-dev report has slightly different shallow hybrid results (85.18% @5 / MRR 0.8194); these are separate recorded runs, not identical baseline numbers.

### Candidate depth, diversity, and hard cases

`retrieval_experiments_report.json`, `full_dev_retrieval_comparison.json`, and `step1_failure_audit_13q.json` examine K=20 versus K=40 and per-document chunk caps/Jaccard deduplication:

- BM25 recall rises from 94.03% @20 to 95.28% @40.
- Original hybrid reaches about 95.30% @40.
- BM25's 13-hard-case mean recall rises from 18.30% to 35.31%; hybrid @40 is 35.63%.
- Diversity reduces redundant chunks but does not improve document recall in the saved BM25 comparison. Do not claim an independent accuracy gain from this ablation.
- `run_step4_downstream_eval.py` and `benchmark_fast_model.py` contain further diagnostic experiment entry points. A runnable script alone does not establish that a corresponding full experiment completed; their expected final result artifacts are not all present.

### Text RAG smoke tests and early model comparisons

- `p2_rag_eval_bm25_10q.json`, `p2_rag_eval_bm25_20q.json`, and `p2_rag_eval_hybrid_20q.json`: 10/20/20-question smoke artifacts with zero answer scores, zero reported cost, and constant 12 ms latency. These are not credible live-model quality baselines; they are consistent with deterministic/mock-path checks.
- `5_questions_inspection.json`: five actual answers/prompts/citation inspections.
- `groq_model_comparison_30q.json`: **sample_size is 5**, with five answers per model, not 30. Models recorded are GPT-OSS-120B, `qwen/qwen3.8-27b`, and GPT-OSS-20B. Saved overall scores are 40 for each; mean latencies are approximately 2.83 s, 7.99 s, and 1.30 s. This tiny, legacy-scored run does not establish a model ranking, and does not substantiate the different 30-question figures quoted in older prose.
- `milestone_1_1_evaluation.json`: five questions; legacy overall score 80, completeness 85.45%, claim-support heuristic 34%.
- `milestone_1_2_evaluation.json`: **11 questions**, legacy score 58.68, correctness 63.64%, completeness 76.86%, claim support 37.22%. An older audit calls 58.68 a full-benchmark result; the JSON does not support that description.

### Query rewriting and decomposition

`phase3_query_expansion_report.json` compares original query, alternative rewrites, terminology expansion, and decomposition. The actual report contains **17 unique target questions**, despite script comments saying 21. Alternative rewrites contain expansions for 13 questions; terminology and decomposition contain none.

The report claims diagnostic Recall@40 rises from 40.33% to 49.85% under alternatives. However, the later full-dev audit records **146/178 fallbacks** in alternative expansion. These results require explicit success/fallback separation before supporting a general expansion recommendation. Current code has improved status tracking, strict mode, successful-response caching, and unit tests; those are engineering improvements, not proof that historical expansion succeeded.

The latest controlled A/E comparison leaves expansion disabled. It should remain a separately measured candidate until validated.

### Context selection, verifier, and invalid Phase 6 conclusions

- `phase4_context_coverage_report.json`: first coverage-aware packer actually **regressed** mean gold coverage from 21.07% to 17.94% on the diagnostic cohort; complete coverage falls from 16.67% to 8.33%. Its runner requests **40 chunks**, not 40 parent documents. This distinction matters when comparing it with later K=40 document runs.
- `phase6_end_to_end_report.json`: 14 diagnostic questions × six configurations E0–E5, comparing deeper hybrid, rewriting, decomposition, coverage packing, and bounded verifier correction. The original scorecard's 100% claims are invalid.
- `audit_phase6_rescored.json`: stored-answer audit reduces answerable strict correctness to 16.67% for E0 and 8.33% for E4/E5; E4 has six answerable abstentions, E5 seven. This is evidence against the old “100% breakthrough” narrative. The rescoring is itself lexical, not human-certified truth.
- `run_fast_model_benchmark.py` still contains the discredited any-one-word coverage logic and marks correctness from absence of prohibited violations. Do not use its output to choose a fast model. Its claimed final `model_speed_vs_quality_report.json` is absent.
- `audit_full_dev_protocol_report.json` and `audit_full_dev_summary.json`: 178Q retrieval/prompt audit; partial generation with quota/truncation problems. “COMPLETE” describes the retrieval protocol, not a full 178-question answer-generation benchmark.

### Current 178-question offline ablation: independently rerun

I reran all six configurations from the existing cached streams, with no API calls and a separate output path. It completed in about 109 seconds. Retrieval, prompt-document coverage, and budget counts match the saved report. Lexical evidence-fact scores drift, as explained below.

| Configuration | Pool gold recall | Prompt gold recall | Complete prompt coverage | Complete multi-doc coverage | Questions losing pool gold during packing |
|---|---:|---:|---:|---:|---:|
| A: BM25@20 + legacy packer | 94.03% | 85.87% | 82.02% | 66.67% | 22 |
| B: BM25@40 + legacy packer | 95.28% | 85.87% | 82.02% | 66.67% | 25 |
| C: original hybrid@40 + legacy packer | 95.30% | 81.76% | 76.40% | 41.03% | 36 |
| D: original hybrid@40 + updated packer | 95.30% | 85.62% | 80.34% | 53.85% | 29 |
| E: corroborated hybrid@40 + updated packer | 95.22% | 89.59% | 85.39% | 66.67% | 19 |
| F: corroborated hybrid@40 + coverage packer | 95.22% | 89.59% | 85.39% | 66.67% | 19 |

There are 39 multi-document questions. E recovers substantial prompt coverage relative to original hybrid, and improves overall prompt recall over A by 3.72 percentage points. Complete multi-document coverage remains tied with BM25. F has no document-coverage advantage over E. A/B each have one over-budget context; D/E/F have none on these inputs.

This verifies **cache-to-context computation**. It does not prove that the cached streams match the current live index/retriever. No index rebuild or live embedding retrieval was performed: the offline BM25 integration test cannot load its model cache.

### Current controlled generation: 47 planned, 30 paired completed

The local JSON is newer than the handoff's 28-pair account: **30 completed pairs**, 17 pairs outstanding. It contains 60 answer records for completed pairs and 59 distinct response-cache entries because A/E share one identical prompt. All cached models are recorded as GPT-OSS-120B.

The 47-question set combines a stratified 40-question development sample with eight diagnostics, deduplicating one overlap. It is not a random representative test set, and quota-limited completed pairs do not cover the full planned sample.

| Measurement | Saved A | Saved E | Current offline rescore A | Current offline rescore E |
|---|---:|---:|---:|---:|
| Strict heuristic correctness, completed 30 pairs | 10/30 | 10/30 | 10/30 | 11/30 |
| Mean fact completeness, all completed questions | 59.01% | 58.32% | 56.61% | 59.26% |
| Claim-support heuristic, saved completed answers | 44.20% | 36.14% | — | — |
| Full 47Q candidate-pool recall, gold-eligible questions | 83.02% | 88.44% | 83.02% | 88.44% |
| Full 47Q prompt gold recall | 78.28% | 77.85% | 78.28% | 77.85% |
| Full 47Q complete prompt coverage | 74.36% | 71.79% | 74.36% | 71.79% |

The apparent extra E win is **qst_0485**, using the same cached answer: the current scorer accepts its missing-detail caveat. A/E answer text did not improve. `qst_0433`, `qst_0312`, and `qst_0447` also change score under the current scorer, with identical stored answers. Lexical evidence coverage in the 178Q report changes too: for E, 75.12% saved versus 74.89% recomputed.

Do not present any of these scores as semantic accuracy or a proven E win. Freeze and calibrate the scorer before spending quota completing the remaining pairs. The current generation runner uses the **updated ContextBuilder for both A and E**, whereas the offline A row explicitly uses a legacy packer; these are different baselines.

### Voice prototype

`voice_ttfa_report.json` records one question with reported TTFA 5.065 seconds and end-to-end time 5.236 seconds. That is not evidence of sub-1.5-second voice readiness. More seriously, TTS runs after full RAG generation while TTFA uses an earlier first-sentence timestamp, so the metric omits real waiting time. This is a file/STT/LLM/macOS-TTS prototype, without demonstrated full-duplex interruption handling or LiveKit integration.

## 3. What is currently uncommitted

At the start of review: nine modified tracked files and 33 untracked file entries. Nothing was staged.

| Area | Local additions/changes | Assessment |
|---|---|---|
| Retrieval | Custom `fuse_bm25_and_dense_chunks`, lexical corroboration metadata, local-first model loading, replacement of native hybrid path; new query expansion/decomposition/cache module | Promising prompt preservation; hybrid behavior changed globally and needs an explicit version/configuration identity. |
| Context | Updated standard packer; new coverage-aware packer, trimming/breadth/authority heuristics, formatted token accounting | Better measured coverage; still heuristic and budget edge cases remain. |
| Generation/provider | 4096-token generator default; usage/finish-reason tracking; placeholder-key checks; long Retry-After rejection | Useful reliability work; pipeline still passes its own 2048-token default and model switching remains in provider/CLI. |
| Evaluation | Complete-gold coverage metrics, stricter judge, audit scorer/harness, historical rescoring/full-dev protocol, offline/context and controlled-generation runners | Necessary progress, but inconsistent scorer definitions and missing run fingerprints remain major blockers. |
| Grounding | Citation bug fix; new bounded answer verifier | Does not establish claim entailment or safe correction. |
| Tests | Four new suites: scorer, query expansion, coverage context, verifier | Good coverage growth, but important counterexamples below are missing. |
| Artifacts/hygiene | Local reports, audit/handoff Markdown, ignored cache/checkpoint patterns | Preserve historical evidence; reconcile or clearly retract conflicting prose before sharing results. |

## 4. Errors and concrete improvements

The accompanying `evaluation/review_checks_2026_10_27.py` reproduces the synthetic failures below without external calls. Findings in inherited code are distinguished from new modules by their paths and Git history; not every problem was introduced by the uncommitted work.

### High priority: evaluation and answer trust

1. **Truncated answers can still score correct.** `evaluation/audit_scoring.py:238–245` computes `flags['truncated']` but never requires it to be false. A synthetic answer containing a truncation marker and a matching fact receives `answer_correct_strict=True`. The controlled runner also considers a pair completed solely from absence of `api_error`, so nonempty/truncated invalid outputs can enter the paired cohort. Reject invalid generations before fact scoring and require successful validated completion for paired membership.

2. **An unrelated caveat can excuse a fabricated answer.** The `info_not_found` branch in `evaluation/audit_scoring.py:239` accepts any missing-detail caveat without checking whether the requested information was fabricated. “The undisclosed secret code is 123456. The owner is not specified.” passes. Match the caveat to the requested unavailable field, check prohibited/fabricated assertions, and calibrate against manually reviewed mixed-answer examples. This does not imply that all partial answers should be rejected.

3. **Numeric citation checks accept a wrong number if one number matches.** `rag/citation_validator.py:217` uses `any`, not a requirement that all claimed numeric values be supported. Source: “10 requests and 20 users”; answer: “10 requests and 999 users” is marked supported. Check all numeric claims after excluding citation/list metadata, and align values with units/entities. Even an all-numbers-present test is only a guard, not entailment.

4. **An empty answer can be marked verified, and a bad revision can replace a good draft.** `rag/verifier.py:141–161` never rejects empty output. `verify_and_correct`, lines 212–217, always keeps the returned revision, regardless of finish reason or whether it improves the original. A synthetic empty answer verifies successfully. Validate content/finish reason first, preserve the last acceptable draft, and only accept a revision under an explicit acceptance rule. The supersession regex at line 121 matches `supersed` as a whole word, missing “supersedes” and “superseded”; detecting the word “draft” also does not prove a relevant policy conflict.

5. **Generation cache keys ignore output limits.** `evaluation/run_controlled_generation_benchmark.py:84–101` accepts `max_tokens` but omits it from the key; limits 32 and 4096 produce the same key. Cached responses do not record that limit either. Include all generation-affecting settings and provider identity in a versioned key. Preserve the old cache; migrate or explicitly classify old entries rather than invalidating them silently.

6. **Scores are mutable without scorer provenance.** Identical cached answers produce changed correctness/completeness. Saved reports omit scorer/source fingerprints. Version the scorer and question/fact snapshot; store code/diff hashes, dependency/model/index versions, prompt/tokenizer identity, configuration, and corpus/cache hashes. Historical recalculations should become new reports, not overwrite old conclusions. The new evidence JSON records hashes for this review.

### High priority: experiment equivalence and retrieval flow

7. **Live and cached “hybrid” are not the same controlled configuration.** Current `retrieval/indexer.py:440` fetches 320 BM25 and 320 dense chunks when asking for 160 fused chunks; the controlled benchmark fuses cached streams of 160 each. `offline_retrieval_cache.py` does not generate `dense_k40` at all, though the current controlled runner requires it; its `hybrid` stream now also means the new corroborated fusion, while old C/D reports use original Qdrant RRF. Give original/corroborated algorithms separate names, freeze depths, add manifest validation, and make the cache producer reproduce every required stream. Verify live/cached equivalence before treating the benchmark as production-equivalent.

8. **Multiquery RRF ranking is discarded by the context builder.** `retrieval/query_expansion.py:531–557` computes fused document ranks but returns chunks with branch-native scores. `rag/context_builder.py:149–172` re-sorts by those scores, which are not comparable between queries. A synthetic fused order B,A,C becomes prompt order C,A,B. Carry fused document and chunk ranking explicitly through packing; retain native scores separately for diagnostics. Test the combined retriever-to-prompt path, not just the returned document IDs.

9. **Old experiment entry points still mix historical and new behavior.** For example, Phase 2 loads old hybrid predictions for some configurations and executes today's changed `hybrid` for another. Phase 6 can resume a checkpoint keyed only by config/question, and its current generator is 20B despite prose describing 120B runs. The older fast-model runner still has invalid scoring. Archive or clearly mark these runners, require immutable run manifests, and reject incompatible checkpoints.

### Medium priority: context, citations, and data protocol

10. **Strict context budgets fail for long headers or tiny budgets.** `rag/context_builder.py:377–385` gives up if trimming the only body cannot fit the header. A 50-token budget with a long title returns 179 BGE tokens; fallback counting returns 215. Equivalent final-loop logic exists in the coverage builder. Return an empty context or explicitly reject an impossible budget, with a test for header-only overflow and no-tokenizer behavior. BGE token counts are not the target generator's actual prompt token counts; call this a BGE-budget metric or use the generator tokenizer.

11. **Citation parsing loses chunk/version provenance.** Unversioned chunk citations are recognized globally but not by the sentence path, which only reads document IDs and passage references. Versioned IDs such as `dsid_…__v1__c000` are not recognized by the chunk regex at all. Parent-ID aggregation also merges distinct record versions for support checking. Parse citations once into document/record/chunk/passage references, validate exact supplied passages, and preserve version/source roles. I independently checked the actual ClearEdge development documents: final Confluence is `dsid_d93aed75845e4a0b8126a80d8a7a5a41`; earlier Gmail is `dsid_86f033ed92b84f3a8b381d09b0b95357`. The saved current A/E answer assigns those roles correctly; older prose reverses them.

12. **Dev cross-validation does not preserve the main split's component isolation.** `data/split_questions.py:130–136` assigns folds to individual rows. In the actual dev data, **13 gold document IDs span multiple dev folds**. This does not invalidate the main dev/held-out split. It does invalidate claims that those five dev folds are also free of shared-gold leakage. Assign connected components to folds and report group-level isolation.

13. **The main evaluation path confuses retrieved pool and selected prompt IDs.** `AnswerGenerator` populates `RAGAnswer.retrieved_doc_ids` with assembled context IDs. `RAGPipeline.query` has actual retrieved IDs but does not assign them, and `evaluation/eval_rag.py:347` computes Recall@K from that field. Historical recall reported by that path is therefore prompt recall, with a shorter/reordered list, rather than candidate retrieval recall. Store `retrieved_doc_ids` and `prompt_doc_ids` separately and score each at its own stage.

14. **The current suite is not fully offline-reproducible.** The full command produced **79 passes, one failure, in 6.21 seconds**. Failure: `tests/test_retrieval.py::test_repeatable_indexing_and_three_retrieval_modes`, because FastEmbed cannot find `Qdrant/bm25` under its temporary cache directory in offline mode. This is an environment/dependency-cache failure, not evidence that retrieval ranking is wrong. Configure a persistent prepared model cache and separate hermetic unit tests from model-backed integration tests. The manifests test reads recorded hash-verification booleans, not freshly recomputed corpus hashes; distinguish a manifest test from current binary-integrity verification. No CI workflow or dependency lockfile is tracked.

### Historical voice claim and smaller reliability issues

15. **TTFA is optimistic and TTS can silently become fake audio.** `voice/voice_pipeline.py:310–350` waits for full generation before synthesizing, but uses the earlier first-sentence timestamp to report TTFA. A synthetic 120 ms generation tail reports about 1 ms TTFA even though TTS cannot have started before that tail ends. `StreamingTTS` catches synthesis failures and writes mock RIFF bytes, so a test checking nonempty output can pass without valid audio. When voice work resumes, emit actual first audio during generation, time its delivery, report fallback status, decode-check produced audio, and add barge-in/cancellation tests.

16. **Provider/default settings remain inconsistent.** `rag/llm.py:123–126` silently maps `gpt-4o-mini` to GPT-OSS-120B under Groq; the CLI also switches it. The pipeline treats a base URL as evidence of credentials even though the client requires a key. Unify provider/model validation, expose the effective config, and explicitly gate mock mode. Current ordinary pipeline defaults remain BM25@20 / 2048 output tokens, not the controlled E benchmark's hybrid@40 / 4096. This can be intentional, but must be visible.

17. **Temporal metadata should not imply authority.** Date regexes accept invalid calendar dates and omit 2030 despite documentation saying the supported range includes it. `latest_date` may mean the selected title/header date or a body mention, not publication/effective date. Validate actual dates and preserve publication, revision, effective, referenced-event dates separately where supported. Authority keywords and newer timestamps are weak signals; use explicit project/version/supersession evidence for policy precedence.

## 5. Verification performed in this review

| Check | Result and limitation |
|---|---|
| Remote main vs local HEAD | Exact match at `cf6f708`; read-only remote-ref query |
| Main history | All four commits inspected, including file/test evolution |
| Current complete pytest suite | 79 passed / 1 failed; missing offline BM25 model cache |
| 178Q offline A–F ablation | Reproduced retrieval/document coverage and budget counts; lexical evidence scoring drift identified |
| Controlled generation cache-only replay | No API calls; all 30 completed pairs reconstructed; score changes traced to identical answers |
| Synthetic failure probes | Cache collision, truncated-answer acceptance, fabrication+caveat acceptance, wrong numeric support, chunk/version citation gaps, empty verification, header budget overflow, multiquery rank loss, and TTFA timing error reproduced |
| Dev fold isolation | 13 gold document IDs span dev folds |
| ClearEdge provenance | Read the two actual dev-corpus sources; final/draft IDs and current answer roles checked |
| Historical full live experiments | Not rerun; quoted as stored evidence with limitations |

Reproduce the failure probes:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m evaluation.review_checks_2026_10_27
```

Reproduce the existing suite:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m pytest tests -q
```

The evidence file `evaluation/reports/independent_review_2026-10-27.json` contains recomputed offline aggregates, saved/recomputed generation comparisons, synthetic probe outputs, and relevant source/cache hashes. Existing benchmark reports were preserved. The existing ablation/controlled runners default to overwriting their historical report paths; this review redirected outputs to new temporary paths before summarizing them into the new evidence file.

## 6. Recommended objective and execution order

**Near-term objective:** Given an enterprise question, identify the required facts, retrieve and preserve the supporting passages, resolve source/version conflicts, answer each supported part with passage-level citations, and explicitly identify requested details that remain unknown. Demonstrate that behavior with calibrated development evaluation and a reproducible baseline.

1. **Repair the measurement layer first.** Fix strict-scoring validity gates, field-specific abstention, numeric support, cache identity, paired-completion membership, and verifier acceptance. Add targeted regression tests from the reproduced failures. Preserve historical scores under their original scorer identity.
2. **Freeze a reproducible A/E protocol.** Explicitly define whether A is legacy or updated packing; pin query/fact snapshot, retrieval algorithm/depth, prompt, tokenizer, model, output limit, cache/index identity, and scorer. Validate the live/cache equivalence and persistent model-backed smoke test. Do not silently regenerate a legacy hybrid cache with the new algorithm.
3. **Calibrate actual answer and citation quality.** Manually label a small development set spanning numerical swaps, negation, partial missing details, conflicts, and bullet-list citations. Measure atomic facts, wrong claims, source-role correctness, supported citations, and calibrated abstention separately. A lexical score cannot certify correctness.
4. **Then complete the remaining 17 matched pairs.** First determine whether existing cached responses have sufficient configuration provenance to continue the same experiment. If not, explicitly version a new comparison rather than mixing protocols. Keep valid old responses; do not rerun paid generations merely to rescore. Future live API usage can be separately authorized; none was needed for this review.
5. **Optimize the demonstrated bottleneck: evidence surviving packing.** Run one-variable comparisons of relevance-aware sentence/passage selection, sibling evidence retrieval, and larger/adaptive context budgets for 6–10-source questions. Evaluate coverage, real completeness, unsupported claims, tokens, and cost together. Corroborated hybrid E remains a candidate, with BM25 retained as the control.
6. **Only after that, test expansion/reranking/iterative investigation independently.** Expansion must show actual successful branch calls and preserved identifiers. Reranking should use a frozen pool; it cannot recover missing candidates. Decomposition should maintain an explicit fact/evidence checklist and pass fused ranks into packing. No graph database or multi-agent rewrite is justified by current measurements alone.
7. **Validate generalization before claiming enterprise readiness.** After selecting a stable development configuration, use a larger corpus with realistic distractors and then a frozen held-out protocol. Real enterprise ingestion will also need document permissions, tenant isolation, freshness/deletion propagation, and source-instruction isolation; the current synthetic benchmark does not demonstrate those capabilities.
8. **Return to voice after answer quality is stable.** Make speech a delivery layer over the same traceable answer/evidence system, with actual streaming-audio measurements, cancellation, and readable source transcripts. The present 20B/120B speed claims and optimistic TTFA are insufficient to choose a production voice architecture.

The core engineering opportunity is clear: retrieval recall is already strong on the dev-conditioned subset. The largest remaining gains are likely in selecting the right evidence within the prompt, synthesizing all required facts, preserving source/version attribution, and measuring these behaviors reliably.
