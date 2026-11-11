"""Deterministic offline retrieval & context-selection evaluation on all 178 answerable dev questions.

Zero LLM/API calls. Uses cached Qdrant retrieval streams (`evaluation/reports/cache/offline_retrieval_chunks.json.gz`)
to compare:
- Legacy ContextBuilder & Legacy CoverageAwareContextBuilder (before fixes)
- Updated ContextBuilder & Updated CoverageAwareContextBuilder on:
  1. BM25 (top_k=20)
  2. BM25 (top_k=40)
  3. Unweighted Qdrant Hybrid RRF (top_k=40)
  4. Corroborated Hybrid RRF (`fuse_bm25_and_dense_chunks`, top_k=40)

Saves full aggregate, per-category, and 8-diagnostic-question breakdown to
`evaluation/reports/offline_retrieval_context_comparison_178q.json`.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from evaluation.audit_scoring import evidence_fact_coverage
from evaluation.evaluation_harness import load_development_benchmark
from rag.context_builder import AssembledContext, ContextBuilder, EvidencePassage, estimate_tokens
from rag.coverage_context_builder import CoverageAwareContextBuilder
from retrieval.indexer import RetrievedChunk, fuse_bm25_and_dense_chunks

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "evaluation" / "reports" / "cache" / "offline_retrieval_chunks.json.gz"
OUT_PATH = ROOT / "evaluation" / "reports" / "offline_retrieval_context_comparison_178q.json"

DIAGNOSTIC_QIDS = [
    "qst_0427",
    "qst_0214",
    "qst_0203",
    "qst_0282",
    "qst_0436",
    "qst_0437",
    "qst_0447",
    "qst_0448",
]


def _legacy_context_builder(
    retrieved_chunks: Sequence[RetrievedChunk],
    budget: int = 2500,
    max_chunks_per_doc: int = 3,
    min_distinct_docs: int = 2,
) -> AssembledContext:
    """Exact reproduction of pre-fix ContextBuilder.build_context (`0.55 * top_score` cutoff)."""
    if not retrieved_chunks:
        return AssembledContext(formatted_context="", passages=[], total_tokens=0)

    doc_best_score: Dict[str, float] = {}
    doc_candidate_chunks: Dict[str, List[RetrievedChunk]] = {}
    for ch in retrieved_chunks:
        doc_best_score[ch.doc_id] = max(doc_best_score.get(ch.doc_id, float("-inf")), ch.score)
        doc_candidate_chunks.setdefault(ch.doc_id, []).append(ch)

    ranked_docs = sorted(doc_best_score.keys(), key=lambda d: doc_best_score[d], reverse=True)
    top_score = doc_best_score[ranked_docs[0]] if ranked_docs else 1.0
    primary_cutoff = 0.55 * top_score
    primary_docs = [
        d for d in ranked_docs
        if (doc_best_score[d] >= primary_cutoff) or (ranked_docs.index(d) < min_distinct_docs)
    ]
    secondary_docs = [d for d in ranked_docs if d not in primary_docs]

    ordered_candidates: List[RetrievedChunk] = []
    max_tokens_per_chunk = max(300, (budget - 150) // len(primary_docs)) if len(primary_docs) >= 5 else None

    for doc_id in primary_docs:
        if doc_candidate_chunks[doc_id]:
            ordered_candidates.append(doc_candidate_chunks[doc_id][0])
    for doc_id in primary_docs:
        for ch in doc_candidate_chunks[doc_id][1:max_chunks_per_doc]:
            ordered_candidates.append(ch)
    for doc_id in secondary_docs:
        if doc_candidate_chunks[doc_id]:
            ordered_candidates.append(doc_candidate_chunks[doc_id][0])

    selected_raw_chunks: List[RetrievedChunk] = []
    current_tokens = 0
    for cand in ordered_candidates:
        cand_chunk_text = cand.chunk_text
        if max_tokens_per_chunk and estimate_tokens(cand_chunk_text) > max_tokens_per_chunk:
            paras = cand_chunk_text.split("\n\n")
            trimmed = ""
            for p in paras:
                if estimate_tokens(trimmed + "\n\n" + p) <= max_tokens_per_chunk:
                    trimmed += ("\n\n" + p) if trimmed else p
                else:
                    break
            cand_chunk_text = trimmed or cand_chunk_text[: max_tokens_per_chunk * 4]

        dummy_p = EvidencePassage(
            passage_index=1,
            doc_id=cand.doc_id,
            chunk_id=cand.chunk_id,
            source_type=cand.source_type,
            normalized_title=cand.normalized_title,
            section_title=cand.section_title,
            chunk_text=cand_chunk_text,
            score=cand.score,
            latest_date=cand.latest_date,
            date_confidence=cand.date_confidence,
        )
        cand_tokens = estimate_tokens(dummy_p.to_formatted_passage())
        if current_tokens + cand_tokens > budget:
            if not selected_raw_chunks:
                selected_raw_chunks.append(cand)
                current_tokens += cand_tokens
            break
        selected_raw_chunks.append(
            RetrievedChunk(
                chunk_id=cand.chunk_id,
                record_id=cand.record_id,
                doc_id=cand.doc_id,
                score=cand.score,
                source_type=cand.source_type,
                normalized_title=cand.normalized_title,
                section_title=cand.section_title,
                chunk_text=cand_chunk_text,
                latest_date=cand.latest_date,
                date_confidence=cand.date_confidence,
                slack_channels=cand.slack_channels,
                ticket_ids=cand.ticket_ids,
            )
        )
        current_tokens += cand_tokens

    selected_doc_order = list(dict.fromkeys(c.doc_id for c in selected_raw_chunks))
    final_passages: List[EvidencePassage] = []
    accumulated_text = ""
    total_tokens = 0
    passage_idx = 1
    doc_to_indices: Dict[str, List[int]] = {}
    for doc_id in selected_doc_order:
        d_chunks = [c for c in selected_raw_chunks if c.doc_id == doc_id]
        d_chunks.sort(key=lambda c: c.chunk_id)
        for cand in d_chunks:
            p = EvidencePassage(
                passage_index=passage_idx,
                doc_id=cand.doc_id,
                chunk_id=cand.chunk_id,
                source_type=cand.source_type,
                normalized_title=cand.normalized_title,
                section_title=cand.section_title,
                chunk_text=cand.chunk_text,
                score=cand.score,
                latest_date=cand.latest_date,
                date_confidence=cand.date_confidence,
            )
            formatted = p.to_formatted_passage()
            p.token_count = estimate_tokens(formatted)
            final_passages.append(p)
            accumulated_text += formatted
            total_tokens += p.token_count
            doc_to_indices.setdefault(p.doc_id, []).append(p.passage_index)
            passage_idx += 1

    unique_doc_ids = list(dict.fromkeys(p.doc_id for p in final_passages))
    doc_dist: Dict[str, int] = {}
    for p in final_passages:
        doc_dist[p.doc_id] = doc_dist.get(p.doc_id, 0) + 1
    total_p = len(final_passages)
    max_conc = (max(doc_dist.values()) / total_p) if total_p > 0 else 0.0
    return AssembledContext(
        formatted_context=accumulated_text.strip(),
        passages=final_passages,
        total_tokens=total_tokens,
        unique_doc_ids=unique_doc_ids,
        unique_chunk_ids=[p.chunk_id for p in final_passages],
        doc_id_to_passages=doc_to_indices,
        doc_chunk_distribution=doc_dist,
        max_chunk_concentration=max_conc,
        distinct_docs_count=len(unique_doc_ids),
    )


def _reconstruct_corroborated_hybrid_chunks(
    bm25_raw: List[Dict[str, Any]],
    hybrid_raw: List[Dict[str, Any]],
    dense_raw: List[Dict[str, Any]] | None = None,
) -> Tuple[List[str], List[RetrievedChunk]]:
    """Fuse BM25 and Dense chunks via fuse_bm25_and_dense_chunks (using exact dense_k40 when cached)."""
    bm25_chunks = [RetrievedChunk(**c) for c in bm25_raw]
    if dense_raw is not None:
        dense_chunks = [RetrievedChunk(**c) for c in dense_raw]
    else:
        b_rank_map = {c.chunk_id: i + 1 for i, c in enumerate(bm25_chunks)}
        dense_with_rank: List[Tuple[float, RetrievedChunk]] = []
        for c_dict in hybrid_raw:
            cid = c_dict["chunk_id"]
            br = b_rank_map.get(cid)
            b_rrf = (1.0 / (1.0 + br)) if br is not None else 0.0
            d_rrf = max(0.0, float(c_dict["score"]) - b_rrf)
            if d_rrf > 1e-6:
                d_rank = (1.0 / d_rrf) - 1.0
                ch = RetrievedChunk(**c_dict)
                ch.score = d_rrf
                dense_with_rank.append((d_rank, ch))

        dense_with_rank.sort(key=lambda x: x[0])
        dense_chunks = [ch for _, ch in dense_with_rank]

    fused_chunks = fuse_bm25_and_dense_chunks(
        bm25_chunks=bm25_chunks,
        dense_chunks=dense_chunks,
        top_k_chunks=160,
    )
    seen_docs: List[str] = []
    for ch in fused_chunks:
        if ch.doc_id not in seen_docs:
            seen_docs.append(ch.doc_id)
            if len(seen_docs) >= 40:
                break
    return seen_docs, fused_chunks


def evaluate_configuration(
    name: str,
    df_answerable: Any,
    cache: Dict[str, Any],
    stream_mode: str,
    top_k_docs: int,
    builder_fn: Callable[[Sequence[RetrievedChunk], str], AssembledContext],
) -> Dict[str, Any]:
    pool_recalls: List[float] = []
    pool_hits: List[float] = []
    prompt_recalls: List[float] = []
    prompt_completes: List[float] = []
    multi_pool_recalls: List[float] = []
    multi_prompt_recalls: List[float] = []
    multi_prompt_completes: List[float] = []
    ev_covs: List[float] = []
    actual_tokens_list: List[int] = []
    reported_tokens_list: List[int] = []
    distinct_docs_list: List[int] = []
    lost_in_packing_qids: List[str] = []
    over_budget_qids: List[str] = []
    by_category: Dict[str, Dict[str, List[float]]] = {}
    diagnostics: Dict[str, Any] = {}

    for row in df_answerable.itertuples():
        qid = row.question_id
        qtype = row.question_type
        qtext = row.question
        gold = list(dict.fromkeys(row.expected_doc_ids))
        gold_set = set(gold)
        facts = list(row.answer_facts) if row.answer_facts is not None else []

        if stream_mode == "corroborated_hybrid_k40":
            pool_doc_ids, chunks = _reconstruct_corroborated_hybrid_chunks(
                cache[qid]["bm25_k40"]["chunks"],
                cache[qid]["hybrid_k40"]["chunks"],
                cache[qid].get("dense_k40", {}).get("chunks"),
            )
            pool_doc_ids = pool_doc_ids[:top_k_docs]
        else:
            pool_doc_ids = cache[qid][stream_mode]["doc_ids"][:top_k_docs]
            chunks = [RetrievedChunk(**c) for c in cache[qid][stream_mode]["chunks"]]

        ctx = builder_fn(chunks, qtext)
        actual_fmt_tokens = estimate_tokens(ctx.formatted_context) if ctx.formatted_context else 0

        in_pool = gold_set & set(pool_doc_ids)
        in_ctx = gold_set & set(ctx.unique_doc_ids)
        p_rec = len(in_pool) / len(gold_set)
        pr_rec = len(in_ctx) / len(gold_set)
        pr_comp = 1.0 if len(in_ctx) == len(gold_set) else 0.0
        ev = evidence_fact_coverage(facts, ctx.formatted_context)

        pool_recalls.append(p_rec)
        pool_hits.append(1.0 if in_pool else 0.0)
        prompt_recalls.append(pr_rec)
        prompt_completes.append(pr_comp)
        if len(gold_set) > 1:
            multi_pool_recalls.append(p_rec)
            multi_prompt_recalls.append(pr_rec)
            multi_prompt_completes.append(pr_comp)
        if in_pool - in_ctx:
            lost_in_packing_qids.append(qid)
        if ev == ev:
            ev_covs.append(ev)
        actual_tokens_list.append(actual_fmt_tokens)
        reported_tokens_list.append(ctx.total_tokens)
        if actual_fmt_tokens > 2500:
            over_budget_qids.append(qid)
        distinct_docs_list.append(ctx.distinct_docs_count)

        cat = by_category.setdefault(qtype, {"pool_rec": [], "prompt_rec": [], "prompt_comp": [], "ev_cov": []})
        cat["pool_rec"].append(p_rec)
        cat["prompt_rec"].append(pr_rec)
        cat["prompt_comp"].append(pr_comp)
        if ev == ev:
            cat["ev_cov"].append(ev)

        if qid in DIAGNOSTIC_QIDS:
            diagnostics[qid] = {
                "question_type": qtype,
                "gold_docs_total": len(gold_set),
                "gold_docs_in_pool": len(in_pool),
                "gold_docs_in_prompt": len(in_ctx),
                "pool_recall": round(p_rec, 4),
                "prompt_gold_recall": round(pr_rec, 4),
                "prompt_complete_gold": bool(len(in_ctx) == len(gold_set)),
                "evidence_fact_coverage": round(ev, 4) if ev == ev else None,
                "distinct_docs_in_prompt": ctx.distinct_docs_count,
                "passages_in_prompt": len(ctx.passages),
                "actual_formatted_tokens": actual_fmt_tokens,
            }

    cat_summary = {
        k: {
            "count": len(v["pool_rec"]),
            "pool_recall": round(sum(v["pool_rec"]) / len(v["pool_rec"]), 4),
            "prompt_gold_recall": round(sum(v["prompt_rec"]) / len(v["prompt_rec"]), 4),
            "prompt_complete_gold_pct": round(100.0 * sum(v["prompt_comp"]) / len(v["prompt_comp"]), 2),
            "evidence_fact_coverage": round(sum(v["ev_cov"]) / len(v["ev_cov"]), 4) if v["ev_cov"] else 0.0,
        }
        for k, v in by_category.items()
    }

    return {
        "config_name": name,
        "stream_mode": stream_mode,
        "top_k_docs": top_k_docs,
        "num_questions": len(pool_recalls),
        "num_multi_doc_questions": len(multi_prompt_recalls),
        "pool_recall_at_k": round(sum(pool_recalls) / len(pool_recalls), 4),
        "pool_hit_at_k": round(sum(pool_hits) / len(pool_hits), 4),
        "prompt_gold_recall": round(sum(prompt_recalls) / len(prompt_recalls), 4),
        "prompt_complete_gold_pct": round(100.0 * sum(prompt_completes) / len(prompt_completes), 2),
        "multi_doc_pool_recall": round(sum(multi_pool_recalls) / len(multi_pool_recalls), 4),
        "multi_doc_prompt_gold_recall": round(sum(multi_prompt_recalls) / len(multi_prompt_recalls), 4),
        "multi_doc_prompt_complete_gold_pct": round(
            100.0 * sum(multi_prompt_completes) / len(multi_prompt_completes), 2
        ),
        "questions_losing_pool_gold_in_packing": len(lost_in_packing_qids),
        "mean_evidence_fact_coverage": round(sum(ev_covs) / len(ev_covs), 4),
        "mean_distinct_docs_in_prompt": round(sum(distinct_docs_list) / len(distinct_docs_list), 2),
        "mean_reported_tokens": round(sum(reported_tokens_list) / len(reported_tokens_list), 1),
        "mean_actual_formatted_tokens": round(sum(actual_tokens_list) / len(actual_tokens_list), 1),
        "over_budget_2500_count": len(over_budget_qids),
        "per_category": cat_summary,
        "diagnostic_questions": diagnostics,
    }


def main() -> None:
    t0 = time.time()
    with gzip.open(CACHE_PATH, "rt", encoding="utf-8") as f:
        cache = json.load(f)

    _, df_answerable, _ = load_development_benchmark()

    new_cb = ContextBuilder(max_context_tokens=2500)
    new_cov_cb = CoverageAwareContextBuilder(max_context_tokens=2500)

    configs = [
        (
            "A_Baseline_BM25_k20_LegacyCB",
            "bm25_k20",
            20,
            lambda chs, q: _legacy_context_builder(chs, budget=2500),
        ),
        (
            "B_BM25_k40_LegacyCB",
            "bm25_k40",
            40,
            lambda chs, q: _legacy_context_builder(chs, budget=2500),
        ),
        (
            "C_UnweightedHybrid_k40_LegacyCB",
            "hybrid_k40",
            40,
            lambda chs, q: _legacy_context_builder(chs, budget=2500),
        ),
        (
            "D_UnweightedHybrid_k40_UpdatedCB",
            "hybrid_k40",
            40,
            lambda chs, q: new_cb.build_context(chs),
        ),
        (
            "E_CorroboratedHybrid_k40_UpdatedCB",
            "corroborated_hybrid_k40",
            40,
            lambda chs, q: new_cb.build_context(chs),
        ),
        (
            "F_CorroboratedHybrid_k40_UpdatedCoverageCB",
            "corroborated_hybrid_k40",
            40,
            lambda chs, q: new_cov_cb.build_context(chs, query=q),
        ),
    ]

    results = {}
    for name, mode, k, fn in configs:
        res = evaluate_configuration(name, df_answerable, cache, mode, k, fn)
        results[name] = res
        print(
            f"{name:42s} | pool={res['pool_recall_at_k']:.4f} | "
            f"pr_rec={res['prompt_gold_recall']:.4f} | "
            f"pr_comp={res['prompt_complete_gold_pct']:5.2f}% | "
            f"m_rec={res['multi_doc_prompt_gold_recall']:.4f} | "
            f"m_comp={res['multi_doc_prompt_complete_gold_pct']:5.2f}% | "
            f"lost={res['questions_losing_pool_gold_in_packing']:2d} | "
            f"ev_cov={res['mean_evidence_fact_coverage']:.4f} | "
            f"ddocs={res['mean_distinct_docs_in_prompt']:.2f} | "
            f"over2500={res['over_budget_2500_count']}"
        )

    payload = {
        "generated_at_epoch": round(time.time(), 2),
        "elapsed_sec": round(time.time() - t0, 2),
        "split": "dev_answerable_178",
        "max_context_tokens": 2500,
        "external_api_calls": 0,
        "configurations": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved full comparison report to {OUT_PATH} in {payload['elapsed_sec']:.2f}s")


if __name__ == "__main__":
    main()
