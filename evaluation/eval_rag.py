"""Reproducible Phase 2 RAG evaluation suite on the 193-question dev set.

Evaluates and compares:
1. BM25 + Generation
2. Dense + Generation
3. Hybrid (RRF) + Generation

Metrics reported per question and in aggregate:
- Official EnterpriseRAG-Bench Overall Score: mean(answer_correct * completeness_pct)
- Answer Correctness (% binary)
- Answer Completeness (% 0-100 based on answer_facts coverage)
- Gold Document Recall@5 and Recall@20
- Citation Validity (% of citations referencing supplied evidence)
- Abstention Accuracy on `info_not_found` questions
- Contradiction Detection on `conflicting_info` questions
- Latency and generation cost tracking
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from evaluation.metrics import overall_answer_score, recall_at_k
from rag.citation_validator import CitationValidator
from rag.generator import RAGAnswer
from rag.llm import BaseLLMClient, MockLLM, OpenAICompatibleLLM
from rag.pipeline import RAGPipeline
from retrieval.indexer import EnterpriseRetriever, RetrievalMode

BASE_DIR = Path(__file__).resolve().parent.parent

JUDGE_SYSTEM_PROMPT = """You are an objective evaluation judge for enterprise question answering.
Evaluate the candidate answer against the gold reference answer and required answer facts.

Output your evaluation strictly in the following JSON format:
{
  "answer_correct": true or false,
  "completeness_pct": 0.0 to 100.0,
  "explanation": "Brief explanation of errors or missing facts"
}

Scoring criteria:
1. `answer_correct`: Set to true ONLY if the candidate answer does not contain false, contradictory, or fabricated claims relative to the gold answer. For unanswerable queries ("info_not_found"), correct is true if the answer states the information is missing.
2. `completeness_pct`: The percentage (0-100) of required answer facts covered by the candidate answer. If correct is false, completeness will still be recorded, but overall score is 0.
"""


@dataclass
class QuestionEvaluationResult:
    """Evaluation output for a single question."""

    question_id: str
    question_type: str
    question: str
    gold_answer: str
    answer_facts: List[str]
    expected_doc_ids: List[str]
    generated_answer: str
    retrieved_doc_ids: List[str]
    cited_doc_ids: List[str]
    recall_at_5: Optional[float]
    recall_at_20: Optional[float]
    answer_correct: bool
    completeness_pct: float
    overall_score: float
    citation_validity_rate: float
    has_hallucinated_citations: bool
    is_abstention: bool
    has_contradiction: bool
    latency_ms: float
    cost_usd: float
    judge_explanation: str
    sentence_citation_rate: float = 1.0
    claim_support_rate: float = 1.0
    has_unsourced_claims: bool = False
    max_chunk_concentration: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


NEGATIVE_CONSTRAINT_RE = re.compile(
    r"^(the answer must not|the answer should not|do not|must not|should not|cannot)\b",
    re.IGNORECASE,
)


def split_facts_and_constraints(facts: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Separate positive answer facts from negative prohibited constraints."""
    positive_facts: List[str] = []
    prohibited_constraints: List[str] = []
    for f in facts:
        f_strip = f.strip()
        if NEGATIVE_CONSTRAINT_RE.search(f_strip):
            prohibited_constraints.append(f_strip)
        else:
            positive_facts.append(f_strip)
    return positive_facts, prohibited_constraints


class DeterministicJudge:
    """Heuristic rule-based judge for testing and offline evaluation without paid API calls."""

    def evaluate(
        self,
        question: str,
        question_type: str,
        generated_answer: str,
        gold_answer: str,
        answer_facts: List[str],
        is_abstention: bool,
        has_contradiction: bool,
    ) -> Tuple[bool, float, str]:
        """Evaluate correctness and completeness deterministically."""
        # Special case 1: info_not_found
        if question_type == "info_not_found":
            if is_abstention:
                return (
                    True,
                    100.0,
                    "Correctly abstained on query where information was not available.",
                )
            else:
                return (
                    False,
                    0.0,
                    "Failed to abstain on an unanswerable query (hallucination risk).",
                )

        # Special case 2: conflicting_info
        if question_type == "conflicting_info":
            if has_contradiction:
                return (
                    True,
                    100.0,
                    "Correctly noted contradiction between divergent sources.",
                )

        # Separate positive facts from negative constraints
        raw_facts = answer_facts if answer_facts else [gold_answer]
        positive_facts, prohibited_constraints = split_facts_and_constraints(raw_facts)
        if not positive_facts and gold_answer:
            positive_facts = [gold_answer]

        covered = 0
        gen_lower = generated_answer.lower()

        # 1. Evaluate positive fact coverage
        for fact in positive_facts:
            words = [
                w
                for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{4,}\b", fact.lower())
                if w not in {"that", "this", "from", "with", "must", "should", "answer"}
            ]
            if not words:
                covered += 1
                continue
            matched = sum(1 for w in words if w in gen_lower)
            if matched / len(words) >= 0.40:
                covered += 1

        completeness = (covered / len(positive_facts)) * 100.0 if positive_facts else 100.0

        # 2. Check for prohibited distractor claims
        violated_constraints: List[str] = []
        for constraint in prohibited_constraints:
            body = NEGATIVE_CONSTRAINT_RE.sub("", constraint).strip()
            c_words = [
                w
                for w in re.findall(r"\b[a-zA-Z0-9_\-\.]{4,}\b", body.lower())
                if w not in {
                    "that", "this", "from", "with", "must", "should", "answer",
                    "claim", "state", "mention", "discuss", "attribute", "explain"
                }
            ]
            if not c_words:
                continue
            matched_c = sum(1 for w in c_words if w in gen_lower)
            # High lexical containment indicates model asserted the prohibited claim
            if len(c_words) >= 3 and (matched_c / len(c_words) >= 0.60):
                violated_constraints.append(constraint[:60] + "...")

        has_violations = len(violated_constraints) > 0
        correct = (completeness >= 30.0) and (not has_violations) and not (is_abstention and positive_facts)

        if has_violations:
            expl = f"Failed negative constraint: asserted prohibited claim ({violated_constraints[0]})"
        else:
            expl = f"Deterministic evaluation: {covered}/{len(positive_facts)} positive facts addressed."

        return (
            correct,
            round(completeness, 2),
            expl,
        )


class LLMJudge:
    """LLM-as-a-judge using OpenAI/Groq or configured provider."""

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm = llm_client

    def evaluate(
        self,
        question: str,
        question_type: str,
        generated_answer: str,
        gold_answer: str,
        answer_facts: List[str],
        is_abstention: bool,
        has_contradiction: bool,
    ) -> Tuple[bool, float, str]:
        facts_text = "\n".join(f"- {f}" for f in answer_facts)
        prompt = (
            f"QUESTION: {question}\n\n"
            f"QUESTION TYPE: {question_type}\n\n"
            f"GOLD REFERENCE ANSWER:\n{gold_answer}\n\n"
            f"REQUIRED ANSWER FACTS:\n{facts_text}\n\n"
            f"CANDIDATE ANSWER TO EVALUATE:\n{generated_answer}\n\n"
            f"OUTPUT (JSON only):"
        )
        try:
            resp = self.llm.generate(
                prompt=prompt,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=256,
            )
            # Parse JSON from response
            m = re.search(r"\{.*\}", resp.text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                correct = bool(data.get("answer_correct", False))
                comp = float(data.get("completeness_pct", 0.0))
                comp = max(0.0, min(100.0, comp))
                expl = str(data.get("explanation", ""))
                return correct, comp, expl
        except Exception as e:
            # Fallback to deterministic judge if judge API fails
            pass

        return DeterministicJudge().evaluate(
            question=question,
            question_type=question_type,
            generated_answer=generated_answer,
            gold_answer=gold_answer,
            answer_facts=answer_facts,
            is_abstention=is_abstention,
            has_contradiction=has_contradiction,
        )


def run_rag_evaluation(
    pipeline: RAGPipeline,
    questions_df: pd.DataFrame,
    mode: RetrievalMode = "bm25",
    judge_type: str = "deterministic",
    judge_llm: Optional[BaseLLMClient] = None,
    max_questions: Optional[int] = None,
    output_report_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run evaluation over a questions DataFrame and compute official benchmark metrics."""
    if max_questions:
        from rag.context_builder import ContextBuilder

        # Take stratified sample if max_questions specified
        from evaluation.inspect_p1_checks import get_stratified_sample

        eval_df = get_stratified_sample(questions_df, num_samples=max_questions)
    else:
        eval_df = questions_df

    judge = (
        LLMJudge(judge_llm)
        if judge_type == "llm" and judge_llm is not None
        else DeterministicJudge()
    )

    records = eval_df.to_dict(orient="records")
    eval_results: List[QuestionEvaluationResult] = []

    t_start = time.perf_counter()

    for idx, row in enumerate(records, start=1):
        qid = str(row["question_id"])
        qtype = str(row["question_type"])
        question = str(row["question"])
        gold_ans = str(row.get("gold_answer") or "")
        raw_facts = row.get("answer_facts")
        facts = [str(f) for f in raw_facts] if raw_facts is not None else []
        raw_exp = row.get("expected_doc_ids")
        expected_docs = [str(d) for d in raw_exp] if raw_exp is not None else []

        rag_ans: RAGAnswer = pipeline.query(question=question, mode=mode)

        r5 = recall_at_k(rag_ans.retrieved_doc_ids, expected_docs, k=5)
        r20 = recall_at_k(rag_ans.retrieved_doc_ids, expected_docs, k=20)

        correct, completeness, expl = judge.evaluate(
            question=question,
            question_type=qtype,
            generated_answer=rag_ans.answer_text,
            gold_answer=gold_ans,
            answer_facts=facts,
            is_abstention=rag_ans.is_abstention,
            has_contradiction=rag_ans.has_contradiction,
        )

        ov_score = overall_answer_score(correct, completeness)

        eval_results.append(
            QuestionEvaluationResult(
                question_id=qid,
                question_type=qtype,
                question=question,
                gold_answer=gold_ans,
                answer_facts=facts,
                expected_doc_ids=expected_docs,
                generated_answer=rag_ans.answer_text,
                retrieved_doc_ids=rag_ans.retrieved_doc_ids,
                cited_doc_ids=rag_ans.cited_doc_ids,
                recall_at_5=r5,
                recall_at_20=r20,
                answer_correct=correct,
                completeness_pct=completeness,
                overall_score=ov_score,
                citation_validity_rate=rag_ans.citation_validation.citation_precision,
                has_hallucinated_citations=rag_ans.citation_validation.has_hallucinated_citations,
                is_abstention=rag_ans.is_abstention,
                has_contradiction=rag_ans.has_contradiction,
                latency_ms=rag_ans.latency_ms,
                cost_usd=rag_ans.cost_estimate_usd,
                judge_explanation=expl,
                sentence_citation_rate=rag_ans.citation_validation.sentence_citation_rate,
                claim_support_rate=rag_ans.citation_validation.claim_support_rate,
                has_unsourced_claims=rag_ans.citation_validation.has_unsourced_claims,
                max_chunk_concentration=rag_ans.assembled_context.max_chunk_concentration if rag_ans.assembled_context else 0.0,
            )
        )

    total_time = time.perf_counter() - t_start

    # Aggregate metrics
    n = len(eval_results)
    avg_ov_score = sum(r.overall_score for r in eval_results) / n if n else 0.0
    avg_correct = (sum(1 for r in eval_results if r.answer_correct) / n * 100.0) if n else 0.0
    avg_completeness = sum(r.completeness_pct for r in eval_results) / n if n else 0.0

    r5_vals = [r.recall_at_5 for r in eval_results if r.recall_at_5 is not None]
    avg_r5 = (sum(r5_vals) / len(r5_vals) * 100.0) if r5_vals else 0.0

    r20_vals = [r.recall_at_20 for r in eval_results if r.recall_at_20 is not None]
    avg_r20 = (sum(r20_vals) / len(r20_vals) * 100.0) if r20_vals else 0.0

    avg_citation_validity = (
        sum(r.citation_validity_rate for r in eval_results) / n * 100.0 if n else 0.0
    )
    avg_sentence_citation_rate = (
        sum(r.sentence_citation_rate for r in eval_results) / n * 100.0 if n else 0.0
    )
    avg_claim_support_rate = (
        sum(r.claim_support_rate for r in eval_results) / n * 100.0 if n else 0.0
    )
    total_cost = sum(r.cost_usd for r in eval_results)
    avg_latency = sum(r.latency_ms for r in eval_results) / n if n else 0.0

    # Per-category aggregation
    by_category: Dict[str, Any] = {}
    cats = sorted(set(r.question_type for r in eval_results))
    for c in cats:
        c_recs = [r for r in eval_results if r.question_type == c]
        cn = len(c_recs)
        c_r5 = [r.recall_at_5 for r in c_recs if r.recall_at_5 is not None]
        by_category[c] = {
            "num_questions": cn,
            "overall_score": sum(r.overall_score for r in c_recs) / cn if cn else 0.0,
            "answer_correct_pct": (
                sum(1 for r in c_recs if r.answer_correct) / cn * 100.0 if cn else 0.0
            ),
            "completeness_pct": (
                sum(r.completeness_pct for r in c_recs) / cn if cn else 0.0
            ),
            "recall_at_5": (sum(c_r5) / len(c_r5) * 100.0) if c_r5 else None,
            "abstention_count": sum(1 for r in c_recs if r.is_abstention),
            "contradiction_count": sum(1 for r in c_recs if r.has_contradiction),
        }

    report = {
        "benchmark": "EnterpriseRAG-Bench",
        "mode": mode,
        "judge_type": judge_type,
        "num_evaluated": n,
        "execution_time_seconds": round(total_time, 2),
        "overall": {
            "overall_score": round(avg_ov_score, 2),
            "answer_correctness_pct": round(avg_correct, 2),
            "answer_completeness_pct": round(avg_completeness, 2),
            "recall_at_5": round(avg_r5, 2),
            "recall_at_20": round(avg_r20, 2),
            "citation_validity_pct": round(avg_citation_validity, 2),
            "sentence_citation_rate_pct": round(avg_sentence_citation_rate, 2),
            "claim_support_rate_pct": round(avg_claim_support_rate, 2),
            "total_estimated_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
        },
        "by_category": by_category,
        "per_question": [r.to_dict() for r in eval_results],
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        output_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report
