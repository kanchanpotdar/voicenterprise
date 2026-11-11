"""Audit scoring (benchmark-integrity audit, post Phase 6).

This module does NOT replace historical scoring; it provides a stricter, separately
reported scorer used only by the audit scripts. Every metric here is a deterministic
lexical heuristic and is documented with its failure modes. Nothing here establishes
semantic entailment.

Why it exists
-------------
The Phase 6 scorer (`evaluation/run_phase6_end_to_end_suite.py::evaluate_answer_facts`)
counted a required fact as "covered" if ANY single token longer than 3 characters from the
fact appeared in the answer, and set `is_correct = no constraint violated` (always True for
questions without negative constraints). Abstaining answers therefore scored 100%.

Metrics (all per answer; aggregate by simple mean over questions)
-----------------------------------------------------------------
- empty / truncated: generation produced no usable text, or hit the token limit.
- abstains: answer states the information is unavailable (phrase match in first 400 chars).
- fact_covered(fact): ALL anchor tokens of the fact (tokens containing a digit, e.g.
  `0.92`, `2025-03-24`, `runtime-1.21.4`, `SUP-4821`, `07:00`; plus snake_case identifiers)
  must appear in the normalised answer AND the stem-level recall of the fact's remaining
  content words must be >= 0.6 (0.5 when anchors are present).
- fact_completeness = covered positive facts / positive facts.
- constraint_violation_suspected: all anchors/keywords of a prohibited claim appear in a
  sentence of the answer that has no negation / supersession cue.
- answer_correct_strict (answerable): not empty, not abstaining, completeness == 1.0, no
  suspected violations.  (info_not_found): answer abstains or caveats.
- unsupported_sentence_rate (lexical proxy): over substantive, non-abstention sentences,
  the fraction that are uncited, cite an ID absent from the prompt, or have <35% lexical
  overlap / unmatched numbers against the cited passages.  A sentence can pass this check
  and still be false, and a correct sentence can fail it.  It measures citation
  *attachment*, not truth.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from evaluation.eval_rag import split_facts_and_constraints
from rag.citation_validator import LOCAL_ABSTENTION_RE, CitationValidator

_STOP = {
    "the", "and", "that", "this", "with", "from", "must", "should", "state", "answer",
    "indicate", "specify", "also", "have", "been", "into", "which", "their", "there",
    "than", "then", "will", "would", "each", "more", "most", "such", "only", "when",
    "after", "before", "while", "were", "does", "mention", "mentions", "describe",
    "describes", "includes", "include", "required", "using", "used", "uses", "for",
}

_ABSTAIN_EXTRA = re.compile(
    r"(not possible to determine|cannot be determined|can(?:no|')t be determined|"
    r"is \*{0,2}not found\*{0,2}|not found in the (?:available )?documents|"
    r"does not contain|do not contain|no (?:document|passage|evidence) (?:provides|states|mentions))",
    re.IGNORECASE,
)
_NEG_CUE = re.compile(
    r"\b(not|no|never|don't|do not|must not|cannot|can't|superseded|supersedes|earlier|previous|"
    r"previously|older|prior|draft|instead of|rather than|replaced|deprecated)\b",
    re.IGNORECASE,
)
_ANSWER_PREFIX = re.compile(
    r"^(the answer )?(must|should)( also)?( (state|indicate|specify|mention|note|describe))?( that)?\s*",
    re.IGNORECASE,
)
_ANCHOR_RE = re.compile(r"[A-Za-z0-9_.:/%\-]*\d[A-Za-z0-9_.:/%\-]*|[a-z]+_[a-z0-9_]+|[a-z0-9_]+(?:/[a-z0-9_]+){2,}")


def normalise(text: str) -> str:
    t = text.lower()
    for ch in ("\u2011", "\u2010", "\u2013", "\u2014"):
        t = t.replace(ch, "-")
    for sp in ("\u202f", "\u00a0", "\u2009", "\u2007"):
        t = t.replace(sp, " ")
    t = t.replace("\u2019", "'").replace("`", "").replace("*", "")
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    t = re.sub(r"(\d)\s+%", r"\1%", t)
    return t


def _stem(w: str) -> str:
    return w[:5]


def _content_stems(text: str) -> List[str]:
    words = re.findall(r"[a-z][a-z\-]{3,}", normalise(text))
    return [_stem(w) for w in words if w not in _STOP]


def _anchors(fact: str) -> List[str]:
    out: List[str] = []
    for m in _ANCHOR_RE.findall(normalise(fact)):
        a = m.strip(".:,-/")
        if a and a not in out:
            out.append(a)
    return out


def is_empty_or_truncated(answer: str) -> Dict[str, bool]:
    marker = "[TRUNCATION DETECTED"
    body = answer.split(marker)[0].strip() if marker in answer else answer.strip()
    return {"empty": len(body) < 20, "truncated": marker in answer}


_DOC_REFUSAL_RE = re.compile(
    r"(cannot (?:be )?answer(?:ed)?|can(?:no|')t be determined|not possible to determine|"
    r"not (?:found|available|provided|mentioned|stated) in the (?:available |supplied |provided )?(?:documents|evidence|passages|context)|"
    r"(?:supplied|provided|available) (?:evidence|documents|passages|context) (?:does|do) not (?:contain|provide|state|mention|specify)|"
    r"no (?:document|passage|evidence) (?:among|in|provides|states|mentions))",
    re.IGNORECASE,
)
_DETAIL_CAVEAT_RE = re.compile(
    r"(?:not (?:spelled out|specified|provided|found|mentioned|stated|documented|detailed|defined|available)|"
    r"does not (?:specify|contain|state|mention|provide|detail)|"
    r"cannot be (?:determined|found|answered)|"
    r"omitted from the (?:available |supplied )?(?:documents|evidence|passages))",
    re.IGNORECASE,
)


def caveats_missing_detail(answer: str) -> bool:
    """Detect if an answer explicitly caveats that a specific requested detail is omitted or not specified."""
    return bool(_DETAIL_CAVEAT_RE.search(answer))


def abstains(answer: str, head_chars: int = 400) -> bool:
    """Detect if the answer as a whole is a full refusal/abstention.

    Does NOT flag answers that begin with substantive, cited factual assertions
    before expressing a secondary caveat or noting an omitted detail.
    """
    lines = [p.strip() for p in answer.split("\n\n") if p.strip()]
    if not lines:
        return False
    first = lines[0]
    m_first = _DOC_REFUSAL_RE.search(first) or (
        len(answer.strip()) < 600 and (_ABSTAIN_EXTRA.search(first) or LOCAL_ABSTENTION_RE.search(first))
    )
    if m_first:
        text_before = first[: m_first.start()]
        if "dsid_" in text_before and len(text_before.split()) > 6:
            return False
        return True

    # If first paragraph has substantive cited content, it is not a full refusal
    if "dsid_" in first and len(first.split()) > 10:
        return False

    head = answer[:head_chars]
    m_head = _DOC_REFUSAL_RE.search(head)
    if m_head:
        text_before = head[: m_head.start()]
        if "dsid_" in text_before and len(text_before.split()) > 6:
            return False
        return True
    if len(answer.strip()) < 600 and (LOCAL_ABSTENTION_RE.search(head) or _ABSTAIN_EXTRA.search(head)):
        m = LOCAL_ABSTENTION_RE.search(head) or _ABSTAIN_EXTRA.search(head)
        text_before = head[: m.start()]
        if "dsid_" in text_before and len(text_before.split()) > 6:
            return False
        return True
    return False



def fact_covered(fact: str, answer: str) -> bool:
    fact_clean = _ANSWER_PREFIX.sub("", fact.strip())
    ans = normalise(answer)
    anchors = _anchors(fact_clean)
    for a in anchors:
        if a not in ans:
            return False
    stems = _content_stems(re.sub(_ANCHOR_RE, " ", normalise(fact_clean)))
    if not stems:
        return bool(anchors)
    ans_stems = set(_content_stems(answer))
    recall = sum(1 for s in stems if s in ans_stems) / len(stems)
    return recall >= (0.5 if anchors else 0.6)


def constraint_violation_suspected(constraint: str, answer: str) -> bool:
    body = re.sub(r"^.*?\b(must not|should not|do not|not)\b\s*(claim|state|say|assert|attribute|introduce)?( that)?", "", constraint, flags=re.IGNORECASE)
    body = re.split(r";\s*it is\b|,\s*not to\b", body, flags=re.IGNORECASE)[0]
    anchors = _anchors(body)
    stems = _content_stems(re.sub(_ANCHOR_RE, " ", normalise(body)))
    for sent in re.split(r"(?<=[.!?])\s+|\n+", answer):
        s = normalise(sent)
        if anchors and not all(a in s for a in anchors):
            continue
        if not anchors and stems:
            have = set(_content_stems(sent))
            if sum(1 for x in stems if x in have) / len(stems) < 0.8:
                continue
        if not anchors and not stems:
            continue
        if not _NEG_CUE.search(sent):
            return True
    return False


@dataclass
class AuditScore:
    question_id: str
    question_type: str
    empty: bool
    truncated: bool
    abstains: bool
    fact_completeness_pct: float
    covered_facts: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    suspected_constraint_violations: List[str] = field(default_factory=list)
    answer_correct_strict: bool = False
    citation_id_validity: Optional[float] = None
    claim_support_rate_all_sentences: Optional[float] = None
    unsupported_sentence_rate: Optional[float] = None
    num_substantive_sentences: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_answer(
    question_id: str,
    question_type: str,
    answer_text: str,
    answer_facts: Sequence[str],
    assembled_context: Any = None,
    validator: Optional[CitationValidator] = None,
) -> AuditScore:
    flags = is_empty_or_truncated(answer_text)
    abst = abstains(answer_text)
    positives, constraints = split_facts_and_constraints(list(answer_facts))
    covered = [f for f in positives if fact_covered(f, answer_text)]
    missing = [f for f in positives if f not in covered]
    violated = [c for c in constraints if constraint_violation_suspected(c, answer_text)]

    if question_type == "info_not_found":
        is_handled = (abst or caveats_missing_detail(answer_text)) and not flags["empty"]
        completeness = 100.0 if is_handled else 0.0
        correct = is_handled
        covered, missing = ([], []) if correct else ([], list(positives))
    else:
        completeness = 100.0 * len(covered) / len(positives) if positives else float("nan")
        correct = (not flags["empty"]) and (not abst) and bool(positives) and not missing and not violated

    cit_valid = claim_all = unsup = None
    n_sub = 0
    if assembled_context is not None:
        validator = validator or CitationValidator()
        res = validator.validate(answer_text, assembled_context)
        cit_valid = res.citation_id_validity if res.total_citations_found else None
        claim_all = res.claim_support_rate
        subs = []
        for s in res.sentences:
            words = [w for w in s.sentence.split() if w.strip("*#-:`").isalnum()]
            if len(words) <= 3 or LOCAL_ABSTENTION_RE.search(s.sentence) or _ABSTAIN_EXTRA.search(s.sentence):
                continue
            subs.append(s)
        n_sub = len(subs)
        if subs and not flags["empty"]:
            bad = sum(1 for s in subs if not (s.is_cited and s.is_supported_by_context))
            unsup = bad / len(subs)
    return AuditScore(
        question_id=question_id,
        question_type=question_type,
        empty=flags["empty"],
        truncated=flags["truncated"],
        abstains=abst,
        fact_completeness_pct=completeness,
        covered_facts=covered,
        missing_facts=missing,
        suspected_constraint_violations=violated,
        answer_correct_strict=bool(correct),
        citation_id_validity=cit_valid,
        claim_support_rate_all_sentences=claim_all,
        unsupported_sentence_rate=unsup,
        num_substantive_sentences=n_sub,
    )


def evidence_fact_coverage(facts: Sequence[str], context_text: str) -> float:
    """Fraction of positive facts whose anchors + content words appear in the PROMPT context.

    Upper bound on what a perfect reader could answer from the supplied evidence (lexical).
    """
    positives, _ = split_facts_and_constraints(list(facts))
    if not positives:
        return float("nan")
    return sum(1 for f in positives if fact_covered(f, context_text)) / len(positives)
