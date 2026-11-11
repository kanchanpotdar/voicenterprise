"""Aggregate audit_full_dev_protocol_report.json into audit_full_dev_summary.json (no model calls)."""
import json, math
from pathlib import Path
REP = Path(__file__).resolve().parent.parent / "evaluation" / "reports"
r = json.loads((REP / "audit_full_dev_protocol_report.json").read_text())
recs = r["records"]; cfgs = ["A_bm25_k20","B_bm25_k40","C_hybrid_k40","D_hybrid_k40_altq","E_D_plus_coverage_ctx"]
def mean(xs): xs=[x for x in xs if x==x]; return sum(xs)/len(xs) if xs else float("nan")
out = {"status": r["status"], "n": len(recs), "configs": {}, "generation_subset": {}}
for c in cfgs:
    m = [v["configs"][c] for v in recs.values()]
    multi = [x for x in m if x["gold_total"] > 1]
    out["configs"][c] = {
      "pool_recall": mean(x["pool_recall"] for x in m), "pool_complete_gold": mean(x["pool_complete"] for x in m),
      "prompt_gold_recall": mean(x["prompt_recall"] for x in m), "prompt_complete_gold": mean(x["prompt_complete"] for x in m),
      "multi_doc_n": len(multi), "multi_doc_prompt_recall": mean(x["prompt_recall"] for x in multi),
      "multi_doc_prompt_complete": mean(x["prompt_complete"] for x in multi),
      "n_questions_gold_lost_pool_to_prompt": sum(1 for x in m if x["pool_gold_lost_in_context"]),
      "evidence_fact_coverage": mean(x["evidence_fact_coverage"] for x in m),
      "mean_prompt_tokens": mean(x["prompt_tokens"] for x in m), "mean_distinct_docs": mean(x["prompt_distinct_docs"] for x in m),
      "expansion_fallback_count": sum(1 for x in m if x.get("expansion_fallback_to_original_only")),
    }
for q, v in recs.items():
    for c in cfgs:
        g = v["configs"][c].get("generation")
        if g: out["generation_subset"].setdefault(q, {})[c] = ({"error": g["error"]} if "error" in g else {k: g["score"][k] for k in ("answer_correct_strict","fact_completeness_pct","abstains","empty","truncated","unsupported_sentence_rate","claim_support_rate_all_sentences","citation_id_validity","suspected_constraint_violations")} | {"gold_in_prompt": f'{v["configs"][c]["gold_in_prompt"]}/{v["configs"][c]["gold_total"]}', "evidence_fact_coverage": v["configs"][c]["evidence_fact_coverage"], "answer_chars": len(g["answer_text"])})
(REP / "audit_full_dev_summary.json").write_text(json.dumps(out, indent=2, default=str))
for c, s in out["configs"].items(): print(c, {k: (round(x,3) if isinstance(x,float) else x) for k,x in s.items()})
print()
for q, d in out["generation_subset"].items():
    print(q)
    for c, s in d.items(): print("  ", c[:2], s)
