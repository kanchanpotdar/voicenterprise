# VoiceEnterprise — Codex Engineering Handoff

**Prepared:** 2026-10-27  
**Repository:** <https://github.com/kanchanpotdar/voicenterprise>  
**Project:** VoiceEnterprise / Voice Rag — voice-first enterprise knowledge investigator  
**Immediate scope:** RAG ingestion, retrieval, context construction, generation, grounding, and evaluation **only**. **Do not begin voice features yet.**  
**Basis:** prior engineering conversations, repository descriptions, Antigravity execution reports, benchmark/audit reports, and the latest controlled-generation report. **This handoff is not a fresh inspection of the current local Git working tree.**

> **Critical reading rule:** "Reported implemented" means the previous coding agent said it changed the code and ran the specified checks. It is **not** evidence that the modification is committed, pushed, present in the remote repository, or independently validated here. "Measured" means a named evaluation reported a numeric result under its stated protocol; model-judged or lexical answer scores are **not human-certified correctness**. "Proposed" means not yet demonstrated. Where reports conflict, prefer the **later integrity audit**, then the **latest reproducible controlled runs**, not earlier impressive-looking Phase 6 tables.

---

## 1. Executive state — read first

VoiceEnterprise already has a developed RAG research/engineering foundation:

- **Dataset:** `onyx-dot-app/EnterpriseRAG-Bench`, a synthetic corpus about *Redwood Inference*: **511,962 source document rows**, **500 benchmark questions**, **9 enterprise source types**.
- **Evaluation partition:** **193 development questions** and **307 held-out questions**, assigned by shared-gold-document connected components. The 193 dev questions comprise **178 with annotated gold documents** plus **10 `info_not_found` and 5 `high_level` questions without gold IDs**. Held-out questions were reported untouched. Do not inspect, tune on, or evaluate the held-out set during development.
- **Working corpus:** development-conditioned `corpus_dev_10k.parquet`, **10,001 rows / 10,000 unique document IDs**, approximately **34,354 chunks** in the initial index. Results are about this **subset**, not the full 512k-document corpus.
- **Retrieval stack:** persistent local **Qdrant**; `Qdrant/bm25` sparse vectors with IDF; `BAAI/bge-small-en-v1.5` dense vectors (384-dimensional cosine); parent-document aggregation; hybrid **RRF** and a later lexical-corroborated hybrid variant.
- **RAG stack:** context packers, Groq-compatible LLM abstraction, answer generator, citation validator, pipeline, query expansion/decomposition components, heuristic evaluators, and a response-cached controlled benchmark runner.
- **Latest benchmarked candidate:** **`E_CorroboratedHybrid_k40_UpdatedCB`** (BM25+dense hybrid, corroborated RRF, top 40 candidate documents, updated 2,500-token ContextBuilder, Groq `openai/gpt-oss-120b`, temperature 0, max output tokens 4096, **query expansion disabled**).
- **Latest test status reported:** **77 passing tests**. Latest retrieval/context run covers all **178 answerable dev questions**; latest end-to-end run evaluates **28 of 47 paired questions** (56 answers), stopped by a Groq tokens-per-day quota.
- **Most important result:** E substantially improves retrieval-to-prompt gold-document preservation versus the original vanilla hybrid, but the completed **28-question answer-generation comparison is a tie on overall strict correctness (9/28 each)**. E is *promising*, **not proven superior end to end**.
- **Git status:** latest agent report described changes as **uncommitted/unpushed**. This is a historical status, **not** a live verification. Do not commit or push without a new explicit instruction from the user.

**Priorities:** factual correctness, full multi-document evidence coverage, contradiction/supersession handling, accurate citations, calibrated abstention, reproducibility, and leakage prevention **ahead of latency**. No voice streaming/STT/TTS/LiveKit work yet.

---

## 2. Product intent and scope boundaries

Long-term product goal: ask natural-language or spoken questions across Slack, Gmail, Linear, Jira, Confluence, Google Drive, GitHub, HubSpot, and Fireflies; investigate documents across systems; reconcile conflicting/newer/older information; cite precisely; abstain if the requested fact is absent; eventually return a concise spoken response.

**Currently in scope:**
1. Deterministic corpus prep and searchable indexes.
2. Comparative lexical / semantic / hybrid retrieval and, when separately justified, reformulation or reranking.
3. Multi-document evidence assembly inside a limited context window.
4. Source-grounded answers and accurate citation-to-claim attribution.
5. Evaluation at **separate** retrieval, prompt, answer, and citation levels.

**Currently out of scope:** voice agents, microphone UI, STT/TTS, LiveKit, a graph database, speculative architectural rewrites, production latency optimization, final held-out scoring, and automatically publishing code.

---

## 3. Data, integrity protocol, and known limitations

### 3.1 Source dataset and anomalies

- Dataset: `onyx-dot-app/EnterpriseRAG-Bench`; an early repository description pinned source snapshot at `69916e31c68aa5963c00248fd7f0bc12d04fd235` (**verify pin locally**).
- Nine systems: `slack`, `gmail`, `linear`, `google_drive`, `hubspot`, `fireflies`, `github`, `jira`, `confluence`.
- Source data reported **511,962 rows, 511,958 unique doc IDs**. Four duplicated IDs refer to distinct records/versions, including a gold-associated Jira conflict; do not silently deduplicate distinct versions into one content record.
- Other reported anomalies: 15 Slack rows with empty titles, one row with empty content. Normalization retains original fields and synthesizes safe fallback titles as needed.
- All **722 distinct gold document IDs** across the 500 questions were reported to exist in the source corpus.

### 3.2 Leakage-aware split

`data/split_questions.py` reportedly builds a question–gold-document graph, groups connected questions sharing gold documents, and assigns whole components using seed **42**. This gives 193 dev / 307 held-out, **0 gold-doc overlap** across split partitions. The dev split has **277 unique gold document IDs**, though the dev subset retains **278 dev-gold rows** because one ID has two content versions.

**Rule:** Never inspect `data/splits/questions_test.parquet` / similarly named held-out question file (reports use both `questions_test.parquet` and `questions_held_out_test.parquet`) to optimize prompts, retrieval, parameters, heuristics, or models. First locate the canonical actual filename; **do not touch held-out contents**. Maintain question-component integrity in cross-validation and model selection.

### 3.3 Development subsets

| File | Reported composition | Correct use |
|---|---|---|
| `data/subsets/corpus_dev_10k.parquet` | **10,001 records / 10,000 unique IDs**; 278 dev-gold rows, 2,200 dev hard-negative rows, 7,523 background rows | Main development retrieval benchmark; sampled/conditioned on **dev**, not an unbiased full-corpus test |
| `data/subsets/corpus_benchmark_conditioned_12k.parquet` | **11,937 rows**; expands with held-out gold docs and hard negatives | **Do not use** for normal model selection or claim unbiased generalization; label explicitly as gold-conditioned if ever studied under an approved protocol |

The dev 10k background can incidentally include some held-out gold IDs (a prior report mentioned **9**). Its *selection procedure* was reported not to use held-out question labels; nonetheless, treat data leakage carefully and never evaluate on held-out questions until final lock.

---

## 4. Architecture and repository map

### 4.1 Data flow

```text
EnterpriseRAG-Bench raw Parquet
  -> data/audit.py (hash/schema/referential-integrity manifest)
  -> data/split_questions.py (connected-component dev vs. held-out split)
  -> data/build_subset.py (dev subset, hard negatives, background)
  -> ingestion/normalizer.py (canonical schema, duplicate-version preservation)
  -> ingestion/metadata.py (deterministic dates, entities, source metadata)
  -> ingestion/chunker.py (section-aware 2,000-char chunks / 200-char overlap)
  -> ingestion/pipeline.py (34,354 indexed chunks initially)
  -> retrieval/indexer.py (Qdrant sparse BM25 + dense BGE vectors)
  -> parent-document aggregation / retrieval ranking (BM25, dense, RRF hybrid)
  -> optional QueryExpander / decomposition (off for latest A-vs-E benchmark)
  -> rag/context_builder.py or rag/coverage_context_builder.py
  -> rag/generator.py via rag/llm.py (Groq LLM)
  -> rag/citation_validator.py, optional verifier
  -> rag/pipeline.py / CLI
  -> evaluation/* (retrieval, prompt, generation, grounding, error categories)
```

### 4.2 File map (reported paths; verify before editing)

**Dataset/ingestion**
- `data/audit.py`, `data/manifest.json`: source hashes/schema and reference checks.
- `data/split_questions.py`, `data/splits/*`: leakage-aware split artifacts.
- `data/build_subset.py`, `data/subsets/*`: development corpus selection.
- `ingestion/normalizer.py`: canonical documents, versioned duplicate IDs.
- `ingestion/metadata.py`: extraction with confidence tiers.
- `ingestion/chunker.py`, `ingestion/pipeline.py`: structure-aware chunking pipeline.

**Retrieval/RAG**
- `retrieval/indexer.py`: `EnterpriseRetriever`, dense/sparse vector index, base Qdrant RRF, later custom corroborated fusion.
- `retrieval/query_expansion.py`: expansions, status, cache, failure handling. Older reports also mention question decomposition; search actual implementation.
- `retrieval/benchmark_p1.py`: baseline retrieval benchmark.
- `rag/context_builder.py`: standard and updated context packing.
- `rag/coverage_context_builder.py`: alternative coverage-aware packing.
- `rag/llm.py`: API client, explicit model selection, token usage and price metadata.
- `rag/generator.py`: system prompt, generation, output/citation metadata.
- `rag/citation_validator.py`: citation-ID and lexical-support checks.
- `rag/pipeline.py`: orchestrates query -> documents -> context -> answer.
- `rag/verifier.py` (verify actual path): self-correction / verification component from prior Phase 6 experiments; **benefit not established**.

**Evaluation**
- `evaluation/metrics.py`: Recall@K, Hit@K, MRR@K, extra docs, fact metrics (watch legacy implementation).
- `evaluation/evaluation_harness.py`, `evaluation/eval_rag.py`: scoring and runner functions.
- `evaluation/audit_scoring.py`: stricter deterministic answer/integrity heuristics.
- `evaluation/inspect_p1_checks.py`: diagnostics, failure inspection.
- `evaluation/run_offline_context_ablation.py`: reproducible 178Q retrieval/context variants.
- `evaluation/run_controlled_generation_benchmark.py`: paired A/E live generation with response caching, resumability, and quota safeguards.
- `evaluation/run_phase6_end_to_end_suite.py`, `evaluation/audit_rescore_phase6.py`, `evaluation/audit_full_dev_protocol.py`, `evaluation/audit_aggregate.py`: **historical experiments/audit**, not the current canonical generation conclusion.
- `evaluation/reports/`: historical JSON reports and `cache/` directories; do not overwrite baseline artifacts or publish response caches by default.
- `tests/`: Phase 0/1 retrieval tests plus RAG, expansion, context packing, verifier, and scoring regressions.

### 4.3 Embeddings, indexing, chunking, ranking

- **Chunker:** Markdown headings/dividers/transcript sections, target max **2,000 characters** with **200-character overlap**, source/title/section headers; observed maximum **2,187** chars including formatting. Initial dev corpus produced **34,354 chunks** (approximately 3.44 per unique doc).
- **Index:** local Qdrant persistence (older path `data/qdrant_dev_10k`, collection `enterprise_rag_dev_10k`); deterministic UUIDv5 chunk IDs; named vectors `dense` and `bm25`.
- **Sparse:** FastEmbed `Qdrant/bm25` with Qdrant `Modifier.IDF`.
- **Dense:** `BAAI/bge-small-en-v1.5`, **384-dim** cosine; previous Mac indexing used MPS FP16 optimizations. Query instructions reportedly use standard BGE retrieval prefix.
- **Original hybrid:** Qdrant prefetch + reciprocal rank fusion of dense/BM25 chunks. Parent-document aggregation/deduplication prevents several chunks from masquerading as independent documents in recall metrics.
- **New hybrid:** custom corroborated RRF, retaining dense discoveries but demoting dense-only chunks without top-15 lexical corroboration. Latest A-vs-E benchmark describes **lexical discount 0.45** on uncorroborated dense-only chunks, with `top_k_docs=40`. Verify actual formula/rank positions from code; never equate it with an independent cross-encoder reranker.
- **Context packing:** 2,500-token budget, format-aware token accounting, chunk trimming including single-chunk overflow, selective document breadth and relevance anchoring. Reported E breadth pass uses up to **8 distinct parent docs**. This is a selection policy, not an additional retrieval model.

---

## 5. Benchmark protocols and metric semantics

Always specify **which set** (`178 answerable`, `193 total`, `47 stratified+diagnostics`, or `28 completed pairs`), **which corpus**, **candidate cutoff**, **prompt budget**, **same/different model**, and **actual generated answer count**.

### Retrieval stage (document IDs)
- **Recall@K:** fraction of expected **distinct parent document IDs** retrieved among top K; per-question score aggregated across gold-eligible questions.
- **Hit@K:** fraction of eligible questions with at least one gold doc in top K.
- **MRR@K:** reciprocal rank to first relevant parent doc, aggregated.
- **Complete gold coverage:** all annotated gold doc IDs present in the retrieved pool or final prompt.
- **Multi-doc coverage:** compute separately on questions requiring >=2 source documents.

### Context selection (prompt, not pool)
- **Prompt gold-doc recall** / **complete prompt coverage**: which retrieved gold parents survived text packing.
- **Gold docs lost in packing:** pool hit(s) dropped or not represented in final context.
- **Evidence fact coverage:** lexical evidence-anchor coverage inside prompt; **proxy, not a guarantee of semantic support**.
- **Distinct parent docs**, context tokens, over-budget count: diagnose crowding and truncation.

### Generated-answer stage
- **Strict correctness:** deterministic check of required facts and prohibited constraints (with known lexical false positives/negatives), abstention rules; not true exact human correctness.
- **Fact completeness:** anchor/stem overlap against ground-truth fact statements; some paraphrases and concise yet valid answers receive false negatives.
- **Citation-ID validity:** cited `[dsid_...]` exists in provided context; **not** proof that citation backs the right claim.
- **Sentence citation rate / lexical claim support:** measure citation attachment and lexical overlap, can mis-handle uncited sub-bullets under cited headings, reversed source roles, and paraphrases.
- **Unsupported substantive sentence rate:** lexical proxy; needs stronger claim-level entailment / manual calibration.
- **Abstention correctness:** distinguish genuinely unavailable requested attributes from general background context.
- **Generation reliability:** API errors, explicit model used, `finish_reason`, empty/truncated outputs, token usage, latency, cache status.

**Never collapse retrieval metrics, completeness, citation validity, and actual correctness into one unqualified "accuracy" number.**

---

## 6. Verified/reproducible baseline retrieval experiments (Phase 1)

Source report: `evaluation/reports/p1_retrieval_benchmark_dev.json`; benchmark covers **193 dev queries**, **178 with gold docs**; the following rows are *document-level* metrics, not full answer quality.

| Metric | BM25 | Dense BGE-small | Vanilla hybrid RRF |
|---|---:|---:|---:|
| Recall@1 | **65.92%** | 50.35% | 62.10% |
| Recall@5 | 82.64% | 70.19% | **85.09%** |
| Recall@10 | **91.77%** | 75.30% | 89.04% |
| Recall@20 | **94.03%** | 78.56% | 92.98% |
| Hit@5 | 86.52% | 74.72% | **89.33%** |
| Hit@10 | **93.82%** | 79.21% | 91.57% |
| MRR@10 | **0.8353** | 0.6751 | 0.8134 |

**Interpretation:** Dense-only underperformed BM25 on this dev subset. Vanilla RRF improved shallow top-5 discovery but was **worse at @10/@20**, and later proved vulnerable to candidate-to-prompt loss. Deeper candidate pools helped: in an intermediate retrieval-only study, vanilla hybrid Recall@40 was reported around **95.30%** vs Recall@20 **92.98%**; BM25 Recall@40 was around **95.28%**. **Do not conflate pool recall with evidence surviving the prompt.**

Historical Phase 2 results also identified 13 BM25 Recall@20 misses involving rank cutoffs, vocabulary mismatch, and multi-document breadth. The **later integrity audit explicitly retracted an earlier detailed rank table** for these cases; use current generated diagnostics, not that table.

---

## 7. Experiment status: what has and has not been demonstrated

| Technique | Implementation/test history | Evidence status | Practical conclusion |
|---|---|---|---|
| Sparse BM25 | Indexed and benchmarked against gold | **Measured** | Strong baseline; retain as control |
| Dense BGE | Indexed and benchmarked | **Measured** | Weak standalone here; still retrieves complementary semantic candidates |
| Vanilla hybrid RRF | Qdrant native RRF, K=20/40 trials | **Measured** | Pool retrieval may improve, but context diversity/coverage can deteriorate |
| Corroborated RRF / custom fusion | Changed `retrieval/indexer.py`; 178Q offline comparison, 47Q retrieval subset | **Measured retrieval/prompt** | E is the latest promising retrieval variant; not equivalent to a trained reranker |
| Standard vs coverage-aware context builders | Earlier coverage-aware variant failed important gold-coverage checks; updated builders later compared | **Measured, protocol-dependent** | Latest E/F tied on key gold-document metrics; no proof CoverageAware is better for answers |
| Candidate depth 20 vs 40 | Multiple comparisons | **Measured** | Higher pool recall does not automatically raise prompt gold coverage |
| LLM query expansion / alternative reformulation | Implemented; old C1 gain claims later discredited due to silent fallback; status/cache tests later passed | **Feature functional; retrieval benefit unproven in latest A/E test** | Need explicit controlled expanded-vs-original A/B with measured success rate |
| Question decomposition | Historical experiments existed | **No trustworthy standalone improvement** | Old Phase 6 scores invalid; rerun only after independent ablation design |
| Query rewriting | Related to alternative-query reformulation, but not separately isolated from multiquery expansion | **Not separately proven** | Define distinct transformation and evaluation |
| Metadata extraction | Implemented during ingestion | **Measured as extracted counts**, not as retrieval filtering | No documented metadata-filter vs no-filter quality comparison |
| Metadata filtering | No confirmed controlled benchmark | **Not established** | Test verified explicit constraints carefully, guarding against false-negative exclusion |
| Cross-encoder / LLM reranking | No reliable implementation+benchmark shown | **Not benchmarked** | Candidate experiment, not part of E |
| Answer verifier / correction | Earlier E5 experiment; audited as unreliable/confounded | **Not established** | Do not assume it helps correctness |
| Graph retrieval, entity graph, recursive agents | Aspirational | **Not established** | Defer until simpler failures are addressed |
| Voice STT/TTS/LiveKit | Long-term roadmap | **Not implemented/validated in this handoff** | Out of scope now |

**Important provenance note:** A pre-audit report claimed C1 query reformulation improved 14-question diagnostic Recall@40 from ~40.33% to ~49.85%, and an older E4 reached "100% correctness/completeness". **Do not cite these as verified improvements.** Subsequent integrity work showed expansion was frequently failing silently, Phase 6 mixed models/checkpoints, and legacy answer-scoring rules were unsound.

---

## 8. Major integrity audit: why some old results must be rejected

The file `VoiceEnterprise_Benchmark_Integrity_Audit.md` (previously shared in conversation as `Pasted markdown (2)(20261023-160722).md`) explicitly retracted or downgraded major old Phase 5–6 findings.

**Audited defects:**
1. **Legacy fact scorer** counted almost any overlapping word as fact coverage. It let abstentions or irrelevant answers score 100%. A branch returned "correct / 100%" when no raw facts existed. Negative constraints were conflated with affirmative factual requirements.
2. **Silent model substitution and mixed checkpoints:** Phase 6 configured one model but switched to Qwen on failures, reused partial checkpoints across runs, and did not retain a trustworthy per-answer model identity. Therefore E0–E5 were **not controlled fixed-model comparisons**.
3. **Generation truncation misclassified as retrieval failure:** `min(max_tokens, 800)` silently capped output including reasoning tokens. Empty/truncated answers were sometimes mislabeled as retrieval misses/context truncation. This cap was subsequently reported removed, with `finish_reason` retained.
4. **Query expansion fallback:** Original expander returned only the user query after failures, without marking them. **146/178** dev cases showed `expansion_fallback_to_original_only`; earlier Phase 3 improvement claims lacked reliable evidence that expansion had actually run.
5. **Citation role inversion:** earlier `qst_0427` answers gave correct factual contents but attached citations to the *wrong* source roles (final vs superseded document), which an ID-validity-only check could not catch.
6. **Old 13-case rank table, verbatim `qst_0427` quote, "expansion crowding" proof, and 20B-vs-120B speed/quality figures were retracted.** Do not use them for decisions.
7. **Old benchmark `groq_model_comparison_30q.json` claimed 30 questions but stored only 5**, under legacy scoring; the model leaderboard from that artifact is invalid.

The integrity audit reported **70 passing tests** (superseded by **77** later), full-dev retrieval/context comparisons, and only **9 valid / 40 planned** live generations in one previous run because **31 attempts failed with Groq 429**. This incomplete, confounded earlier run does **not** constitute a model or configuration ranking.

The audit's earlier 178Q context snapshot showed vanilla hybrid disproportionately packing 1–2 parent docs and collapsing multi-doc complete prompt coverage to ~41%; this drove the later retrieval+packing rewrite.

**Additional unresolved inconsistency to verify in underlying documents:** Early and late textual reports assign opposite `dsid_...` strings to final Confluence versus superseded Gmail in `qst_0427`. Do **not** trust either prose label blindly; check raw provenance, dates, actual cited IDs and claims using **development** data only. The latest benchmark report says its generated A/E answers had correctly aligned roles; independent validation of the underlying IDs is still needed.

---

## 9. Fixes reported implemented after the audit

These changes were described as implemented and tested, but **must be confirmed in the actual working tree by Codex**.

### 9.1 Evaluation integrity

- `evaluation/audit_scoring.py`: `is_empty_or_truncated`, abstention checks, stricter `score_fact_in_text` using numeric/code anchors and lexical stems, per-sentence prohibited/superseded checks, zero strict correctness for empty/truncated and inappropriate abstaining answers.
- `evaluation/eval_rag.py`, `evaluation/run_phase6_end_to_end_suite.py`: consume integrity status and failure taxonomy (`generation_empty_or_truncated`), avoid silently mutating model selection on API failures.
- Later scoring edge-case normalization for Unicode narrow no-break spaces and spaces before `%`, clauses such as "when ... not available" that are not refusals, tail notes in SOP answers, and positive clarifications following "is not for ...".
- Test files such as `tests/test_audit_scoring.py`; latest offline suite **77 passing**.
- **Still a limitation:** lexical matching incorrectly penalizes valid paraphrases and omission of unimportant policy code labels; citation support isn't semantic entailment.

### 9.2 Query expansion reliability (NOT a proven retrieval-quality win)

- `retrieval/query_expansion.py`: telemetry statuses `success`, `cache_hit`, `disabled`, `empty_expansion`, `invalid_output`, `api_error`; explicit `cache_status` and failure/fallback flags; warning logs; optional `strict=True` raising `QueryExpansionError`.
- Stable persistent cache SHA-256 key reportedly includes normalized query, expansion mode, model identity, system-prompt hash, and maximum expansions; writes atomically; rejects failed/empty/truncated responses; handles corrupt cache.
- Unit tests for cached repeats, invalidation, errors, truncation, and warning statuses.
- Latest smoke test reported live expansion status `success` -> cached repeat `cache_hit` -> disabled `disabled`, but **latest A/E generation benchmark set `use_query_expansion=False`**.

### 9.3 Hybrid candidate ranking and context assembly

- `retrieval/indexer.py`: custom RRF with strong dense contribution only with top lexical corroboration; dense-only ranks discounted rather than discarded.
- `rag/context_builder.py`: scale-aware RRF thresholding; increased primary-document breadth; rank/lexical selection; preserve top chunk unless trimming is necessary; headers counted in token budget.
- `rag/coverage_context_builder.py`: parent diversity selection, capped/trimmed chunks, removed generic authority cue boosting (`policy`, `final`, etc.) that elevated irrelevant passages.
- Fix for one **single giant chunk** (`qst_0229`, 2,604 tokens) exceeding 2,500-token cap, even when only one chunk was selected.
- Retrieval cache now reportedly uses exact Qdrant `dense_k40` response streams instead of reconstructed approximations. Recomputed vanilla-hybrid metrics differ from earlier before/after runs; compare values **within** the same experimental report, not across incompatible snapshots.

### 9.4 Generation and tracking

- Removed hidden 800-token cap, raised default generation `max_tokens` to **4096** after 2048 truncated a multi-phase SOP reasoning output; explicitly record `prompt_tokens`, `completion_tokens`, `finish_reason` in `RAGAnswer`.
- `rag/pipeline.py`: explicit model configuration and improved API-key placeholder validation; removed conflicting default `llama-3.1-8b-instant` override.
- `rag/llm.py`: reported price metadata for GPT-OSS 120B/20B; prices are **historical values**, confirm with provider before financial reporting.
- Created response-cached, quota-aware `evaluation/run_controlled_generation_benchmark.py` with paired configurations, a smoke mode, report saving, and a cache-only rescore.

---

## 10. Latest **178-question** reproducible retrieval/context results

**Primary contemporary source:** latest reported reproduction in `evaluation/reports/offline_retrieval_context_comparison_178q.json`, run with `evaluation/run_offline_context_ablation.py` on the **178 answerable** development questions and a **2,500-token** prompt budget. These are **no-model-call** retrieval and prompt metrics.

| Metric | A: BM25@20 + legacy builder | C: vanilla hybrid@40 + legacy builder | E: corroborated hybrid@40 + updated builder | F: same hybrid + updated coverage builder |
|---|---:|---:|---:|---:|
| Pool gold-doc recall | 94.03% | 95.22% | **95.22%** | **95.22%** |
| Prompt gold-doc recall | 85.87% | 81.76% | **89.59%** | **89.59%** |
| Complete gold-document coverage in prompt | 82.02% | 76.40% | **85.39%** | **85.39%** |
| Multi-doc prompt gold recall | 84.22% | 65.48% | **85.80%** | **85.80%** |
| Multi-doc complete prompt coverage | 66.67% | 41.03% | **66.67%** | **66.67%** |
| Gold documents lost during packing | 22 | 34 | **19** | **19** |
| Evidence-fact coverage (lexical proxy) | 70.65% | 66.85% | **75.12%** | **75.12%** |
| Mean distinct docs per prompt | 5.24 | 4.89 | **5.36** | **5.36** |
| Prompts exceeding 2,500 tokens | 0 | 0 | **0** | **0** |

**Interpretation:** relative to C, E improves prompt gold recall by **7.83 percentage points**, complete prompt gold coverage by **8.99 points**, and multi-doc complete prompt coverage by **25.64 points**; **15 fewer** packing losses in this *latest exact-cache reproduction* (34 -> 19). E also outperforms BM25 A on average gold-prompt recall by **3.72 points**, although multi-doc complete coverage **ties** BM25, so important unsolved broad questions remain.

**Caution on cross-run numbers:** An immediately preceding offline report, before exact Qdrant stream reconciliation, had C evidence coverage **80.32%** and only **2.78** distinct docs, with **36** losses; E evidence coverage **74.66%**, 5.35 docs, 19 losses. The last run reported C **66.85%**, **4.89** distinct docs, **34** losses and E **75.12%**, **5.36** docs, **19** losses. These differ because reconstruction/caching and accounting were corrected. Preserve historical snapshots, but adopt **latest internally consistent run** for comparisons and check the actual JSON to resolve discrepancies.

**Important:** E and F have the same key gold-document metrics here, so the extra complexity of F is **not justified as the default** by this test alone.

---

## 11. Latest **47-question retrieval / 28-pair answer** controlled benchmark

**Report:** `evaluation/reports/controlled_generation_benchmark_47q.json`  
**Runner:** `evaluation/run_controlled_generation_benchmark.py`  
**Cache:** `evaluation/reports/cache/controlled_generation_cache.json`

### 11.1 Protocol

- Target questions: **40** seed-42 stratified dev questions (4 from each of 10 categories) + **8** fixed diagnostics (`qst_0427`, `qst_0214`, `qst_0203`, `qst_0282`, `qst_0436`, `qst_0437`, `qst_0447`, `qst_0448`). `qst_0436` overlapped, so **47 unique**: **39 gold-annotated, 4 `high_level`, 4 `info_not_found`**.
- Compare **A** `mode="bm25", top_k_docs=20, legacy ContextBuilder` versus **E** `mode="hybrid", top_k_docs=40, updated ContextBuilder(corroborated_coverage=True)`.
- **Same generator for both:** Groq `openai/gpt-oss-120b`, `temperature=0`, `max_tokens=4096`, **`use_query_expansion=False`**. Context max 2,500 tokens.
- Smoke checks: two paired questions (`qst_0427`, `qst_0447`); successful with no reported empty/truncated answers in the final smoke pass.
- Full retrieval/prompt comparison done for **47**; *live generation completed for only **28 matched pairs** (56 cached answers)*. Remaining **19 pairs / 38 model calls** were not completed in the provided report after Groq 429 tokens-per-day limit (reported on-demand limit 200,000 TPD). **Do not represent the generation benchmark as 47/47 complete.**

### 11.2 Candidate retrieval and prompt coverage — **all 47** questions

| Metric | A | E | E − A |
|---|---:|---:|---:|
| Pool recall@K on gold-eligible | 83.02% | **88.44%** | +5.42 pp |
| Pool hit@K | 87.18% | **94.87%** | +7.69 pp |
| Prompt gold-doc recall | **78.28%** | 77.85% | −0.43 pp |
| Complete prompt gold coverage | **69.23%** | **69.23%** | tie |
| Multi-doc prompt gold recall | **81.47%** | 79.31% | −2.16 pp |
| Gold docs lost in packing | **4** | 6 | 2 more lost in E |
| Prompt evidence-fact coverage (proxy) | 63.15% | **66.34%** | +3.19 pp |
| Mean distinct docs per prompt | 5.26 | **5.55** | +0.29 |

**Important:** E has better **candidate pool** recall on these 47 questions but not better **prompt gold-document** recall. This is a real qualification to the stronger 178Q average, and the remaining 19 generation cases include major semantic rescue diagnostics. Preserve both findings.

### 11.3 Answer quality — **28 completed matched pairs** only

Of these 28, **25 answerable** and **3 `info_not_found`** questions were reported.

| Deterministic/lexical measure | A: BM25 | E: corroborated hybrid | Interpretation |
|---|---:|---:|---|
| Strict correctness, all 28 | **9/28 = 32.14%** | **9/28 = 32.14%** | **Tie overall** |
| Strict correctness, 25 answerable | 6/25 = 24.00% | **7/25 = 28.00%** | E +1 question, not decisive |
| Mean fact completeness, answerable | 50.81% | **53.99%** | E +3.18 pp, heuristic |
| `DeterministicJudge` correctness | **23/28 = 82.14%** | 22/28 = 78.57% | Different permissive scoring definition; don't conflate with strict correctness |
| Valid cited document IDs | 100% | 100% | Neither hallucinated doc IDs in this subset; not entailment |
| Sentence citation rate | **51.43%** | 44.48% | E worse |
| All-sentence lexical claim-support proxy | **45.18%** | 36.54% | E worse by 8.64 pp |
| Claim support on cited sentences only | **78.90%** | 75.56% | E slightly worse |
| Unsupported substantive sentence rate, proxy | **58.88%** | 66.78% | E worse; uncited structured sub-bullets may be scoring artifact |
| Correct abstentions on 3 `info_not_found` | **3/3** | 2/3 | E regresses on `qst_0485` |
| Empty/truncated outputs | 0 / 0 | 0 / 0 | 4096 output token limit worked on completed pairs |
| Suspected prohibited/outdated violations | 0 | 0 | Only according to heuristic; not human assurance |

**Takeaway:** More candidate coverage and slightly higher factual completeness do **not** yet establish more accurate, safer, or better-cited answers. Strict overall accuracy is tied; citation-support and abstention trade off in E. These are small-sample, heuristic results. Do not claim a statistically significant E win.

### 11.4 Token use and cost from the run

- A: 79,580 prompt + 22,921 completion = **102,501 tokens**.
- E: 82,420 prompt + 22,094 completion = **104,514 tokens**.
- Total cached validated answers: **206,855 tokens**; estimated **$0.0579** at the pricing assumptions used by the report. This is not a bank/payment statement, and the daily-rate cap differs from aggregated reported usage because of timing/other calls.

### 11.5 Diagnostic wins, failures, known hard cases

| Question(s) | Reported observation | What to do |
|---|---|---|
| `qst_0447` | E and A include all 6 gold docs in latest generation run; E fact completeness **18/22 (81.82%)** vs A **12/22 (54.55%)**; E cites 6/6 vs A 3/6 | Maintain multi-phase SOP completeness and correctly attributed citations |
| `qst_0411` | Conflict-resolution fact completeness E **100%** vs A **50%**, citing 22% superseded memo and 30% final allocation | Preserve precedence and source roles |
| `qst_0413` | E 50% vs A 25% completeness in report | Check project/date resolution and missing facts |
| `qst_0179` | Semantic retrieval rescue raised fact completeness 0% -> 28.57%, still far from complete | More than recall is needed; improve synthesis |
| `qst_0485` | E retrieved a near-match defining `delayed_with_handover` token concept but **not exact wire layout**, and partially answered instead of unambiguously abstaining | Test field-specific answerability and required-detail abstention; avoid blanket refusal when a subset *is* answerable |
| `qst_0433` | E dropped one gold document in the completed 28 pairs after packing | Inspect ranking/packing trace; ensure budget-constrained diversity |
| `qst_0343` | E was more concise and omitted required schema field identifiers; lexical completeness fell 62.50% -> 18.75% | Separate actual omissions from scoring false negatives; require atomic-field enumeration when asked |
| `qst_0001` | Both answers gave "14 business days" but strict scorer failed because prose lacked policy-code anchors that were only in cited document | Calibrate evaluator against manually reviewed gold facts; avoid irrelevant code requirements |
| `qst_0178` | Both failed to retrieve "low bit math" gold doc and abstained | Terminology gap, expansion/reranking candidate |
| `qst_0282` | "stay locked" wording misses gold doc under latest earlier offline retrieval; later 47Q candidate may recover some | Test alternate query rewrites with exact rank traces |
| `qst_0203`, `qst_0214` | Earlier evidence in candidate pool but at ranks 13–19 / outside final 2.5k prompt; latest 47Q hybrid pool gains on these | Optimize evidence selection; check whether final prompt contains **all** facts |
| `qst_0436`, `qst_0437` | 9–10 gold documents can exceed practical candidate or context capacity | Test structured per-entity coverage, adaptive budget, decomposition with evidence bookkeeping |
| `qst_0448` | Hybrid recovered more relevant candidates than BM25, but not all seven gold sources | Candidate recall + constrained packing; remaining live generation not done |
| `qst_0427` | Temporal/citation-role conflict case. Latest report *claims* correct roles; earlier audit found inverted citations, and source doc IDs are inconsistent across prose reports | Independently inspect actual source IDs and sentence-to-citation mapping |

---

## 12. Latest benchmark configuration — do not confuse with older "E4"

```yaml
name: E_CorroboratedHybrid_k40_UpdatedCB
status: last_benchmarked_candidate_not_proven_production_default
corpus: corpus_dev_10k.parquet
retrieval:
  mode: hybrid
  sparse: Qdrant/bm25 with IDF
  dense: BAAI/bge-small-en-v1.5
  fusion: corroborated_RRF
  top_k_docs: 40
  uncorroborated_dense_only_discount_reported: 0.45
context:
  builder: updated_ContextBuilder
  max_context_tokens: 2500
  corroborated_coverage: true
  diversity_breadth_docs_reported: 8
generation:
  provider: Groq
  model: openai/gpt-oss-120b
  temperature: 0.0
  max_tokens: 4096
  use_query_expansion: false
evaluation:
  offline_dev_gold_questions: 178
  controlled_total_development_questions: 47
  completed_live_generation_pairs: 28
  held_out_used: false
```

**Baseline A:** `mode=bm25`, `top_k_docs=20`, legacy `ContextBuilder`, same 2,500-token prompt and the same GPT-OSS-120B generation settings.

**Naming hazard:** Old `E0`–`E5` Phase 6 configuration labels and new `A`–`F` offline ablation labels **are not the same experiments**. Earlier "E4 best" conclusion was audited as invalid. The *latest* E is **corroborated hybrid + updated standard builder**, with **query expansion off**.

Also do not assume the CLI/default application pipeline uses E until the actual `rag/pipeline.py` default arguments are inspected. The **benchmark invocation** is verified by report; production configuration is not.

---

## 13. Open problems (confirmed or credible from reports)

### P0 — correctness/integrity blockers

1. **Answer quality unresolved:** 19/47 matched generation pairs unrun; available pairs show no overall strict correctness gain for E; citation-support and abstention regressions persist.
2. **Citation-to-claim source-role verification:** quoted/well-formed `[dsid_...]` can point to a source with the wrong authority, date, or assertion. `qst_0427` deserves manual gold provenance verification.
3. **Evaluation validity:** lexical anchor fact scoring false-negatives correct paraphrases and false-positives overlapping wrong answers. Separate *document citation validity*, *claim-to-passage support*, and *actual answer truth*. Need a human-labelled calibration set.
4. **Context selection failures:** retrieved gold docs can be dropped in a 2,500-token context, especially multi-source questions. E shows a modest regression in 47Q prompt gold recall despite improved pool recall.
5. **Insufficient-evidence abstention:** hybrid semantic near-misses can encourage ungrounded partial answers (`qst_0485`).
6. **Reproducibility and configuration provenance:** ensure no model fallback, silent query-expansion failure, mutable cache keys, hidden truncation, test contamination, or report overwrite.

### P1 — retrieval/coverage opportunities, **unproven**

7. Metadata filtering based on reliable explicit project/source/date constraints; need hard-negative recall safeguards.
8. Successful and separately evaluated alternative query rewriting/expansion versus no expansion, with status and cache correctness.
9. Cross-encoder reranking after BM25/hybrid top-40 candidate generation; this cannot restore a doc absent from the pool.
10. Query decomposition/iterative retrieval with per-subquestion evidence coverage, if single-query packer fails 9–10-document questions.
11. Adaptive/evidence-aware context allocation vs fixed 2,500-token cap, measured against token cost and completeness.
12. Better provenance-aware conflict resolution including supersession edges and document-version identity (without prematurely adding a graph DB).

### P2 — later only

13. Larger corpus evaluations after development protocol is locked; never claim 10k-corpus performance equals full 512k-corpus performance.
14. Voice interface and latency-specific dual-model routing; previous 20B/120B comparative figures were discredited. Re-test only when accuracy and citations are stable.
15. Final one-time held-out evaluation after configuration and evaluation protocol are frozen.

---

## 14. Prioritized Codex work plan (proposed; **do not treat as completed**)

### Phase 0: Audit working tree and freeze evidence

- `git status --short`, `git diff --stat`, `git diff`, `git log -5 --oneline`.
- Identify code files vs untracked reports/caches; verify whether reported changes are actually present.
- Confirm no secrets, API credentials, source-only model caches, customer/private files, generated large Qdrant data, or response caches would be inadvertently published.
- Reconcile **last reported base commit** (`cf6f708` in an older audit) with the public repo history; **do not assume it is still HEAD**.
- Record hashes/config of dataset, retrieval cache, model settings, evaluation question lists, and scoring version. **Preserve all historical result files.**

### Phase 1: Offline regression and retrieval reproducibility

- Rerun all tests. Expected historical state: **77 passing**, not a guarantee for current checkout.
- Reproduce the 178Q A/C/E/F retrieval-prompt table with no Groq calls, validate the exact Qdrant dense+BM25 source cache against live local index, and compare per-question differences, not only means.
- Assert formatted context <=2,500 tokens, no single-chunk overflow, no silent dropped expected metadata.
- Check 47Q gold-doc regressions (`qst_0433`, `qst_0214`, `qst_0203`), no held-out access.

### Phase 2: Complete paired generation without uncontrolled changes

- If API quota is available **and user authorizes API usage**, resume cached 47Q A-vs-E test using same `openai/gpt-oss-120b`, temp 0, max_tokens 4096, fixed questions, identical scoring, and expansion off. Historical command:

  ```bash
  .venv/bin/python -m evaluation.run_controlled_generation_benchmark --model openai/gpt-oss-120b
  ```

- Reuse 56 existing cached responses (validate cache key includes config/context/model/prompt/scoring version); run only **remaining 19 pairs** if present. Preserve report lineage, no overwrites. Quota may differ now: inspect errors/limits rather than assume 200k remains available.
- Do not modify generator prompts or retrieval tuning mid-comparison; otherwise treat as a **new benchmark version** and rerun both arms.

### Phase 3: Correctness/grounding calibration

- Create a small (e.g., 30–50 answers) **manually labelled development-only** evaluation set spanning facts, abstentions, numerical/code claims, document authority, citation-to-claim support, and structured bullets. Ensure gold citations are verified at doc and passage level.
- Diagnose `qst_0485` and add **targeted** field-specific abstention guard; avoid requiring boilerplate refusal for an answer that properly distinguishes known and unknown elements, unless benchmark rubric explicitly requires strict abstention.
- Decide whether `qst_0001`, `qst_0343`, etc. are real content misses or lexical-scoring artifacts.
- Check `qst_0427` source IDs and temporal precedence by reading dev source documents; implement/regression-test source-role checking.

### Phase 4: One-variable-at-a-time retrieval ablations

After stable baseline, test each on the **same frozen dev questions** and track: pool gold recall, *prompt* gold recall, complete multi-doc coverage, evidence-fact coverage, real answer factual completeness/correctness, unsupported claims, and abstention.

1. **Query expansion/rewrite:** with vs. without; only score cases where expansion status = success/cache-hit; report fallback rate. Compare C1 alternative rewrites separately from terminology expansions and decomposition.
2. **Reranker:** deterministic candidate set (BM25 and/or E@40) -> cross-encoder top-20/40 ordering -> identical packer, no generation prompt drift.
3. **Metadata filtering:** only well-attested fields (source, exact project, date) and a safe fallback when parsing uncertain; report false exclusions and recovered precision.
4. **Context builder/adaptive budget:** account for all headers and preserve multiple gold parents; avoid "gold-aware" selection at inference (gold IDs may be used **only for offline scoring**, not retrieval decisions).
5. Combine techniques only after measuring independent improvements; don't declare a new default on a single anecdotal win.

### Phase 5: Release hygiene — **requires explicit user permission to publish**

- Document exact architecture, evaluation protocols, limitations, and truthfully measured scores in README.
- Stage **reviewed code/tests/lightweight reproducible manifests** only; `.gitignore` `.env`, keys, large corpora, Qdrant indices, logs, user-private paths, and large/generated caches.
- Verify tests and sample reruns, ask user before `git commit`/`git push`. Publication of current local changes **has not been authorized by this handoff**.

---

## 15. Commands and environment (historical; inspect repo for differences)

Project originally used Python **3.11+** and `pyproject.toml` with editable developer extras; early Phase 0/1 needed no API key or Docker.

```bash
# From the checked-out repository root
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Safe initial inspection
git status --short
git diff --stat
git log -5 --oneline

# Offline tests (reported 77 passed)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/pytest tests -q

# Earlier Phase 0/1 reproduction (these regenerate artifacts; inspect configs first)
.venv/bin/python -m data.audit
.venv/bin/python -m data.split_questions
.venv/bin/python -m data.build_subset
.venv/bin/python -m ingestion.pipeline
.venv/bin/python -m retrieval.benchmark_p1

# Offline latest retrieval/context comparison; inspect CLI args first
.venv/bin/python -m evaluation.run_offline_context_ablation

# Quota-consuming only after user approval; response cache should resume
.venv/bin/python -m evaluation.run_controlled_generation_benchmark --model openai/gpt-oss-120b
```

**Secrets:** `.env` should contain provider credentials locally. Never display, commit, send, or log API keys. The historic code reportedly recognized `"your-openai-api-key"` as placeholder, not a valid secret.

**Repository paths may differ** if Codex runs on a new machine, a new branch, or the latest remote only. Historical macOS work was described under `.../Downloads/voice ai /`; **do not hard-code this into scripts**. Reports may be local/untracked and unavailable after a fresh Git clone.

---

## 16. Artifact inventory and trust ranking

Search for these files in the local project or prior attachments; their existence in the remote repo **has not been confirmed**.

**Tier A — current most useful reports**

| Path | Meaning / caution |
|---|---|
| `evaluation/reports/p1_retrieval_benchmark_dev.json` | Original 193Q dev BM25/dense/RRF baseline |
| `evaluation/reports/offline_retrieval_context_comparison_178q.json` | Most recent 178Q context/candidate comparison; distinguish exact-cache revision from earlier snapshot |
| `evaluation/reports/controlled_generation_benchmark_47q.json` | 47 retrieval evaluations, only 28 complete live generation pairs in supplied report |
| `evaluation/reports/controlled_generation_smoke_report.json` | Smoke checks and expansion status/cache telemetry |
| `evaluation/reports/cache/controlled_generation_cache.json` | 56 cached validated A/E responses; likely large and should remain local |
| `evaluation/reports/cache/offline_retrieval_chunks.json.gz` | Cached exact Qdrant retrieval streams, including `dense_k40` |
| `evaluation/run_controlled_generation_benchmark.py` | Current paired benchmark runner |
| `evaluation/run_offline_context_ablation.py` | Offline rerunnable retrieval/context ablation |

**Tier B — historical / diagnostic only**

| Path/name | Meaning / caution |
|---|---|
| `VoiceEnterprise_Benchmark_Integrity_Audit.md` | Canonical **retraction and failure analysis** for older E0–E5 scorecard |
| `evaluation/reports/audit_phase6_rescored.json` | Demonstrates old metric failures; mixed-model historical outputs, not reliable model ranking |
| `evaluation/reports/audit_full_dev_summary.json`, `audit_full_dev_protocol_report.json` | Old audit full-dev retrieval data and tiny partial generation run |
| `evaluation/reports/phase2_retrieval_experiments_report.json` | Retrieval depth and early experiment data |
| `evaluation/reports/phase3_query_expansion_report.json` | Claims **not validated after silent fallback audit** |
| `evaluation/reports/phase4_context_coverage_report.json` | Early builder tradeoffs; not latest builder implementation |
| `evaluation/reports/phase6_end_to_end_report.json` | **Invalid E0–E5 "100%" scoring**; retained for audit only |
| `evaluation/reports/groq_model_comparison_30q.json` | "30Q" title but contained only 5 evaluated questions, legacy metric |

Do not use an old agent's Markdown prose as proof that the actual generated answer says a particular quote. Check stored answers and underlying gold docs.

---

## 17. Non-negotiable operational instructions for Codex

1. **First inspect, then edit.** Never assume report filenames, committed state, model defaults, or hidden flags match current checkout.
2. **No held-out access or tuning.** Use only dev question IDs/labels for development and regression tests. Never put gold IDs into live retrieval logic.
3. **No silent model switching, query expansion fallback, scoring default-to-success, or empty-output masquerade.** Every error category and model identity must be explicit in results.
4. **Do not overwrite baselines or prior generated responses.** Version reports/config hashes; retain paired question IDs, tokens, prompts, model, cached/live flags, and scoring-version identity.
5. **Compare at four independent levels:** retrieved pool, prompt evidence, generated facts, citation/abstention integrity. A higher Recall@40 is not a validated answer improvement.
6. **Never cite historical Phase 6 E4 100% scores as valid.** They were retracted.
7. **Do not introduce voice, rerankers, metadata filtering, decomposition, or architecture expansions without scoped experiment rationale and the user's direction.** These are candidates, not requirements implemented by default.
8. **Respect quota and cost.** Offline work unless live calls are explicitly in scope; stop safely on rate limits, reuse valid caches, and preserve partial results.
9. **No commits, pushes, deployments, or API-key exposure without explicit user authorization.**
10. **Accuracy, completeness and evidence grounding first; latency later.**

---

## 18. The first Codex request to issue after opening this handoff

> Read `VoiceEnterprise_Codex_Handoff_2026-10-27.md` and inspect the current repository and its uncommitted changes. Do not commit, push, access held-out questions, or call external LLM APIs. Verify the latest reported E configuration, 77-test suite, exact Qdrant offline retrieval cache, and the 178Q A/C/E/F metrics. Identify discrepancies between this handoff and the actual working tree; report which changes exist, which artifacts are missing, which conclusions are reproducible, and the smallest next step to complete the 47Q A-vs-E benchmark without invalidating its controls. Do not implement unrelated changes.

---

## 19. Bottom line

VoiceEnterprise is past a toy RAG proof-of-concept: it has a corpus integrity protocol, provenance-aware ingestion, a dual-vector retrieval index, measured BM25/dense/hybrid comparisons, custom evidence packing, generation and citation plumbing, plus progressively stronger evaluation safeguards. Its largest engineering lesson is that **retrieval recall, prompt evidence preservation, answer fact completeness, and citation correctness are different bottlenecks**.

**Keep E as the latest benchmarked experimental candidate, and A as the trusted BM25 control.** Do **not** claim E is proven best end to end, do **not** resurrect retracted E4 scores, and do **not** publish the working tree blindly. Confirm exact code and reports, finish the 19 remaining paired generations using the frozen protocol, calibrate answer/citation metrics, then assess reranking, metadata filtering, and query reformulation as **separate, controlled proposals**.
