"""Audit protocol: controlled comparison A-E on the 178 answerable DEV questions.

Stage 1 (all 178, no answer generation): retrieval pool + final-prompt coverage.
    A  BM25   K=20          + standard ContextBuilder
    B  BM25   K=40          + standard ContextBuilder
    C  Hybrid K=40          + standard ContextBuilder
    D  Hybrid K=40 + alternative-query reformulation + standard ContextBuilder
    E  (same retrieved chunks as D) + CoverageAwareContextBuilder
  No diversity filter anywhere (so B/C isolate depth / retriever); token budget 2500; max 3 chunks/doc.
  D and E share one retrieval call so the context builder is the only difference.
Stage 2 (--generate): fixed model, NO fallback, on the 8 required diagnostic questions only (quota-bound).
  F (verifier) is deliberately not run: the verifier's decisions rely on the same lexical validator
  whose reliability is unestablished (see audit report).

Never touches the held-out split. Resumable: records are keyed by question id and config; a record is
only reused if it carries the same `protocol_version`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evaluation.audit_scoring import evidence_fact_coverage, score_answer
from evaluation.evaluation_harness import load_development_benchmark
from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.coverage_context_builder import CoverageAwareContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import OpenAICompatibleLLM
from retrieval.indexer import EnterpriseRetriever
from retrieval.query_expansion import MultiQueryRetriever, QueryExpander

BASE = Path(__file__).resolve().parent.parent
REP = BASE / "evaluation" / "reports"
PROTOCOL_VERSION = "audit_v1"
CKPT = REP / "audit_full_dev_checkpoint.json"
FINAL = REP / "audit_full_dev_protocol_report.json"
GEN_MODEL = "openai/gpt-oss-120b"
GEN_SUBSET = ["qst_0427", "qst_0214", "qst_0203", "qst_0282", "qst_0436", "qst_0437", "qst_0447", "qst_0448"]
CONFIGS = ["A_bm25_k20", "B_bm25_k40", "C_hybrid_k40", "D_hybrid_k40_altq", "E_D_plus_coverage_ctx"]


def ctx_metrics(assembled, gold, facts, doc_ids, k):
    gold_set = set(gold)
    in_pool = gold_set & set(doc_ids[:k])
    ctx_docs = set(p.doc_id for p in assembled.passages)
    in_ctx = gold_set & ctx_docs
    return {
        "gold_total": len(gold_set),
        "gold_in_pool": len(in_pool),
        "pool_recall": len(in_pool) / len(gold_set),
        "pool_complete": len(in_pool) == len(gold_set),
        "gold_in_prompt": len(in_ctx),
        "prompt_recall": len(in_ctx) / len(gold_set),
        "prompt_complete": len(in_ctx) == len(gold_set),
        "pool_gold_lost_in_context": sorted(in_pool - in_ctx),
        "prompt_tokens": assembled.total_tokens,
        "prompt_distinct_docs": len(ctx_docs),
        "evidence_fact_coverage": evidence_fact_coverage(facts, assembled.formatted_context),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    _, df_a, _ = load_development_benchmark()
    if args.limit:
        df_a = df_a.head(args.limit)
    retriever = EnterpriseRetriever(collection_name="enterprise_rag_dev_10k", qdrant_path=BASE / "data" / "qdrant_dev_10k")
    exp_llm = OpenAICompatibleLLM(model="openai/gpt-oss-20b", timeout_seconds=45.0)
    mq = MultiQueryRetriever(retriever=retriever, expander=QueryExpander(llm_client=exp_llm))
    std = ContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3)
    cov = CoverageAwareContextBuilder(max_context_tokens=2500, max_chunks_per_doc=3, novelty_weight=0.40, redundancy_penalty_weight=0.50, authority_boost=0.30)
    gen = AnswerGenerator(llm_client=OpenAICompatibleLLM(model=GEN_MODEL, timeout_seconds=90.0)) if args.generate else None
    validator = CitationValidator()

    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}
    ck = {q: r for q, r in ck.items() if r.get("protocol_version") == PROTOCOL_VERSION}
    t0 = time.time()
    for i, row in enumerate(df_a.itertuples()):
        qid, query = row.question_id, row.question
        gold = list(dict.fromkeys(row.expected_doc_ids))
        facts = list(row.answer_facts) if row.answer_facts is not None else []
        rec = ck.get(qid, {"protocol_version": PROTOCOL_VERSION, "question_type": row.question_type, "configs": {}})
        need_gen = args.generate and qid in GEN_SUBSET and not all("generation" in rec["configs"].get(c, {}) for c in CONFIGS)
        if len(rec["configs"]) == len(CONFIGS) and not need_gen:
            continue
        chunks_by = {}
        t = time.perf_counter()
        ids_a, ch_a = retriever.retrieve_documents(query=query, mode="bm25", top_k_docs=20)
        ids_b, ch_b = retriever.retrieve_documents(query=query, mode="bm25", top_k_docs=40)
        ids_c, ch_c = retriever.retrieve_documents(query=query, mode="hybrid", top_k_docs=40)
        ids_d, ch_d, exp, _ = mq.retrieve(query=query, mode="hybrid", expansion_mode="alternative_queries", top_k_docs=40, diversity_filter=False)
        ret_s = time.perf_counter() - t
        setups = {
            "A_bm25_k20": (ids_a, ch_a, 20, std), "B_bm25_k40": (ids_b, ch_b, 40, std),
            "C_hybrid_k40": (ids_c, ch_c, 40, std), "D_hybrid_k40_altq": (ids_d, ch_d, 40, std),
            "E_D_plus_coverage_ctx": (ids_d, ch_d, 40, cov),
        }
        for cname, (ids, chs, k, builder) in setups.items():
            asm = builder.build_context(retrieved_chunks=chs, query=query) if builder is cov else builder.build_context(retrieved_chunks=chs)
            m = ctx_metrics(asm, gold, facts, ids, k)
            if cname.startswith(("D_", "E_")):
                m["expansion_fallback_to_original_only"] = len(exp.expanded_queries) == 0
                m["expanded_queries"] = exp.expanded_queries
            if need_gen:
                try:
                    ans = gen.generate_answer(question=query, assembled_context=asm, temperature=0.0)
                    sc = score_answer(qid, row.question_type, ans.answer_text, facts, asm, validator)
                    m["generation"] = {"model": GEN_MODEL, "answer_text": ans.answer_text, "score": sc.to_dict()}
                except Exception as e:  # recorded, never silently replaced by another model
                    m["generation"] = {"model": GEN_MODEL, "error": str(e)[:300]}
            rec["configs"][cname] = m
        rec["retrieval_seconds"] = round(ret_s, 1)
        ck[qid] = rec
        CKPT.write_text(json.dumps(ck, default=str))
        print(f"[{i+1}/{len(df_a)}] {qid} done in {time.time()-t0:.0f}s", flush=True)

    done = [q for q in df_a.question_id if q in ck and len(ck[q]["configs"]) == len(CONFIGS)]
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "questions_expected": len(df_a), "questions_completed": len(done),
        "status": "COMPLETE" if len(done) == len(df_a) else "INCOMPLETE",
        "generation_stage": {"model": GEN_MODEL, "questions": GEN_SUBSET, "run": bool(args.generate)},
        "records": {q: ck[q] for q in done},
    }
    FINAL.write_text(json.dumps(report, default=str))
    print("saved", FINAL, report["status"], len(done))


if __name__ == "__main__":
    main()
