"""Re-score the stored Phase 6 answers with the audit scorer (no new model calls).

This is a RE-SCORING of historical outputs, not a new experiment.
"""
import json
import time
from pathlib import Path

import pandas as pd

from evaluation.audit_scoring import score_answer

BASE = Path(__file__).resolve().parent.parent
REP = BASE / "evaluation" / "reports"


def main():
    src = json.loads((REP / "phase6_end_to_end_report.json").read_text())
    res = src["results_by_config"]
    df = pd.read_parquet(BASE / "data" / "splits" / "questions_dev.parquet")
    meta = {r.question_id: (r.question_type, list(r.answer_facts)) for r in df.itertuples()}

    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": "phase6_end_to_end_report.json (stored answers, re-scored)", "per_question": {}, "summary": {}}
    for cfg, qs in res.items():
        rows = []
        for qid, r in qs.items():
            qt, facts = meta[qid]
            s = score_answer(qid, qt, r["answer_text"], facts)
            d = s.to_dict()
            d.update(
                legacy_correct=r["is_correct"], legacy_completeness=r["completeness_pct"],
                legacy_attribution=r["error_attribution"], legacy_claim_support=r["claim_support_rate"],
                gold_docs=len(r["gold_docs"]), answer_chars=len(r["answer_text"]),
                gen_ms=r["latencies_ms"]["generation"],
            )
            rows.append(d)
            out["per_question"].setdefault(qid, {})[cfg] = d
        n = len(rows)
        ans = [x for x in rows if x["question_type"] != "info_not_found"]
        inf = [x for x in rows if x["question_type"] == "info_not_found"]
        out["summary"][cfg] = {
            "n_questions": n,
            "n_answerable": len(ans),
            "n_info_not_found": len(inf),
            "legacy_correctness_pct": 100 * sum(x["legacy_correct"] for x in rows) / n,
            "legacy_completeness_pct": sum(x["legacy_completeness"] for x in rows) / n,
            "strict_correct_answerable_pct": 100 * sum(x["answer_correct_strict"] for x in ans) / len(ans),
            "strict_fact_completeness_answerable_pct": sum(x["fact_completeness_pct"] for x in ans) / len(ans),
            "abstention_correct_info_not_found": f"{sum(x['answer_correct_strict'] for x in inf)}/{len(inf)}",
            "answerable_abstained": sum(x["abstains"] for x in ans),
            "empty_or_truncated_outputs": sum(x["empty"] for x in rows),
            "legacy_attribution_counts_total": {k: sum(1 for x in rows if x["legacy_attribution"] == k) for k in {x["legacy_attribution"] for x in rows}},
            "suspected_constraint_violations": sum(len(x["suspected_constraint_violations"]) for x in rows),
            "mean_claim_support_all_sentences_legacy": sum(x["legacy_claim_support"] for x in rows) / n,
        }
    p = REP / "audit_phase6_rescored.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["summary"], indent=1))
    print("\nper-question strict correct (A=abstain, E=empty):")
    for qid, cfgs in out["per_question"].items():
        line = []
        for c, x in cfgs.items():
            tag = "E" if x["empty"] else ("A" if x["abstains"] else ("Y" if x["answer_correct_strict"] else "n"))
            line.append(f"{c[:2]}:{tag}/{x['fact_completeness_pct']:.0f}")
        print(f"{qid} gold={cfgs['E0_Baseline']['gold_docs']:>2} " + " ".join(line))
    print("saved", p)


if __name__ == "__main__":
    main()
