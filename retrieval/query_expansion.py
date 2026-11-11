"""Configurable LLM Query Expansion and Question Decomposition for Enterprise RAG.

Key guarantees:
- Optional and configurable: can be disabled with a single flag (enabled=False).
- Preserves the original user query as the primary search branch.
- Structured JSON output with strict validation:
    * original_query
    * expanded_queries (2-4 bounded alternatives)
    * key_entities
    * exact_identifiers
    * constraints & negations
    * uncertainties
- Modes supported:
    * C0: None (original query only)
    * C1: Alternative query reformulation
    * C2: Enterprise terminology-oriented expansion (e.g. colloquial -> architectural terms)
    * D1: Multi-part question decomposition
- Safe fallback: If the LLM fails, times out, or returns invalid JSON, falls back to the original query.
- Reciprocal Rank Fusion (RRF) across query branches with source-query provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rag.llm import BaseLLMClient, MockLLM, OpenAICompatibleLLM
from retrieval.indexer import EnterpriseRetriever, RetrievedChunk


@dataclass
class SubQuestion:
    """A decomposed subquestion target."""

    subquestion: str
    requested_information: str
    entities_identifiers: List[str] = field(default_factory=list)
    expected_evidence_type: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryExpansionResult:
    """Structured result of query expansion or question decomposition."""

    original_query: str
    mode: str
    expanded_queries: List[str] = field(default_factory=list)
    key_entities: List[str] = field(default_factory=list)
    exact_identifiers: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    negations: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    subquestions: List[SubQuestion] = field(default_factory=list)
    provenance_log: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["subquestions"] = [sq.to_dict() for sq in self.subquestions]
        return d

    @property
    def all_search_queries(self) -> List[str]:
        """Return the deduplicated list of all queries to search, always starting with original."""
        queries = [self.original_query]
        for q in self.expanded_queries:
            q_clean = q.strip()
            if q_clean and q_clean not in queries:
                queries.append(q_clean)
        for sq in self.subquestions:
            sq_clean = sq.subquestion.strip()
            if sq_clean and sq_clean not in queries:
                queries.append(sq_clean)
        return queries

    @property
    def status(self) -> str:
        """Explicit expansion status ('success', 'cache_hit', 'disabled', 'success_empty', or 'fallback_*')."""
        return str(self.provenance_log.get("status", "unknown"))

    @property
    def failed(self) -> bool:
        """True if query expansion encountered an LLM/parse/truncation error."""
        return bool(self.provenance_log.get("failed", False)) or (self.status in FAILED_STATUSES)

    @property
    def fell_back_to_original(self) -> bool:
        """True if retrieval fell back to the original query only despite expansion being requested."""
        return bool(self.provenance_log.get("fallback_to_original", False)) or (
            self.mode != "none" and len(self.all_search_queries) == 1
        )

    @property
    def cache_status(self) -> str:
        """Cache status ('hit', 'write', 'miss', 'not_stored', or 'disabled')."""
        return str(self.provenance_log.get("cache", "disabled"))


ALTERNATIVE_QUERIES_SYSTEM_PROMPT = """You are an expert enterprise search query analyst for Redwood Inference.
Your task is to analyze the user's question and generate 2 to 3 high-precision alternative search formulations.

CRITICAL RULES:
1. ALWAYS preserve exact identifiers, ticket numbers, error codes, and technical flags (e.g., dsid_..., PR#, INT8, FP8, vLLM).
2. NEVER invent imaginary incident IDs, customer names, or root causes.
3. Formulate complementary queries targeting likely enterprise documentation phrasing (e.g., Confluence wiki, postmortems, Slack logs, PR descriptions).
4. Identify any negative constraints or phrases (e.g., "must not", "excluding", "without").
5. Output STRICT JSON only with the following schema:
{
  "expanded_queries": ["reformulation 1", "reformulation 2"],
  "key_entities": ["entity1", "entity2"],
  "exact_identifiers": ["ident1"],
  "constraints": ["constraint1"],
  "negations": ["negation1"],
  "uncertainties": ["uncertainty1"]
}"""

TERMINOLOGY_EXPANSION_SYSTEM_PROMPT = """You are an enterprise terminology and architecture specialist for Redwood Inference.
Your task is to bridge vocabulary gaps between colloquial user questions and formal engineering documentation.

CRITICAL RULES:
1. Identify colloquial, informal, or imprecise phrasing (e.g. "low bit math", "cloud catalog", "stay locked", "speed up", "crash").
2. Map to corresponding enterprise and infrastructure terminology:
   - "low bit math" -> "FP8 INT8 quantization safe mode inference precision"
   - "cloud catalog" -> "marketplace listing partner integration SaaS catalog"
   - "stay locked" -> "session pinning sticky sessions affinity routing"
3. Preserve all exact identifiers, dates, versions, and constraints.
4. Output STRICT JSON only with the following schema:
{
  "expanded_queries": ["terminology query 1", "terminology query 2"],
  "key_entities": ["entity1"],
  "exact_identifiers": ["ident1"],
  "constraints": ["constraint1"],
  "negations": ["negation1"],
  "uncertainties": ["uncertainty1"]
}"""

DECOMPOSITION_SYSTEM_PROMPT = """You are a technical research decomposition specialist for complex enterprise queries.
Your task is to break down multi-part or multi-document questions into 2 to 4 independently searchable subquestions.

CRITICAL RULES:
1. Decompose ONLY if the question asks for multiple distinct pieces of evidence across different dimensions (e.g. root cause + mitigation + assigned team; or SDK feature comparison across languages; or multi-phase SOP).
2. If the question is simple or single-fact, return an empty subquestions list.
3. Each subquestion must be self-contained and independently searchable in a document database.
4. Preserve exact constraints and identifiers across subquestions.
5. Output STRICT JSON only with the following schema:
{
  "expanded_queries": [],
  "key_entities": ["entity1"],
  "exact_identifiers": ["ident1"],
  "constraints": ["constraint1"],
  "negations": ["negation1"],
  "uncertainties": [],
  "subquestions": [
    {
      "subquestion": "Subquestion text targeting specific evidence",
      "requested_information": "What specific fact is needed",
      "entities_identifiers": ["ident1"],
      "expected_evidence_type": "postmortem | confluence | code | ticket",
      "dependencies": [],
      "constraints": []
    }
  ]
}"""


class QueryExpansionError(RuntimeError):
    """Raised by QueryExpander(strict=True) when expansion fails instead of falling back."""


# Provenance `status` values (legacy names kept for backward compatibility):
#   "success"            LLM call parsed and validated, >=1 usable alternative query / sub-question
#   "success_empty"      LLM call valid but produced nothing new (NOT a failure, but NOT an expansion)
#   "cache_hit"          validated result served from ExpansionCache (no LLM call)
#   "disabled"           expansion switched off / mode "none"
#   "fallback_no_json"   LLM replied without a JSON object        -> FAILED
#   "fallback_truncated" LLM output hit the token limit            -> FAILED
#   "fallback_invalid"   JSON present but failed schema validation -> FAILED
#   "fallback_error"     exception (HTTP 429, timeout, bad JSON…)  -> FAILED
FAILED_STATUSES = frozenset({"fallback_no_json", "fallback_truncated", "fallback_invalid", "fallback_error"})
_CACHE_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)


def normalize_query_for_key(query: str) -> str:
    """Whitespace-normalised query (case preserved: identifiers are case-sensitive)."""
    return " ".join(query.split())


def expansion_cache_key(query: str, mode: str, model: str, system_prompt: str, limit: int) -> str:
    """Stable key: depends on normalised query, mode, model id, prompt text and expansion limit."""
    payload = json.dumps(
        {
            "v": _CACHE_SCHEMA_VERSION,
            "q": normalize_query_for_key(query),
            "mode": mode,
            "model": model,
            "prompt_sha": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "limit": int(limit),
        },
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ExpansionCache:
    """Persistent JSON cache of SUCCESSFUL, validated expansions only.

    - Failures, fallbacks and empty expansions are never stored.
    - Entries failing schema validation on load are ignored (and logged), never returned.
    - A corrupt cache file is left untouched on disk and treated as empty (logged).
    - Writes are atomic (temp file + replace).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path else None
        self._data: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.rejected = 0
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {k: v for k, v in raw.items() if isinstance(k, str)}
                else:
                    logger.warning("Expansion cache %s is not a JSON object; ignoring it.", self.path)
            except Exception as e:  # corrupt file
                logger.warning("Expansion cache %s unreadable (%s); starting empty (file left untouched).", self.path, e)

    @staticmethod
    def _valid_entry(e: Any) -> bool:
        if not isinstance(e, dict) or e.get("v") != _CACHE_SCHEMA_VERSION:
            return False
        eq = e.get("expanded_queries")
        sq = e.get("subquestions", [])
        if not isinstance(eq, list) or not all(isinstance(x, str) and x.strip() for x in eq):
            return False
        if not isinstance(sq, list) or not all(isinstance(x, dict) and str(x.get("subquestion", "")).strip() for x in sq):
            return False
        return bool(eq or sq)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        e = self._data.get(key)
        if e is None:
            self.misses += 1
            return None
        if not self._valid_entry(e):
            self.rejected += 1
            self.misses += 1
            logger.warning("Ignoring invalid expansion cache entry %s…", key[:12])
            return None
        self.hits += 1
        return e

    def put(self, key: str, result: "QueryExpansionResult") -> bool:
        entry = {
            "v": _CACHE_SCHEMA_VERSION,
            "expanded_queries": list(result.expanded_queries),
            "key_entities": list(result.key_entities),
            "exact_identifiers": list(result.exact_identifiers),
            "constraints": list(result.constraints),
            "negations": list(result.negations),
            "uncertainties": list(result.uncertainties),
            "subquestions": [sq.to_dict() for sq in result.subquestions],
        }
        if not self._valid_entry(entry):
            return False  # never cache empty / invalid results
        self._data[key] = entry
        self.writes += 1
        self._flush()
        return True

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)


class QueryExpander:
    """Configurable query expander with explicit status reporting.

    A failed expansion still returns the original query (so retrieval keeps working) but is
    NEVER silent: it is logged at WARNING, flagged `provenance_log["failed"] = True`, and counted
    in `self.stats`. With `strict=True` a failure raises `QueryExpansionError` instead.
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        enabled: bool = True,
        default_mode: str = "alternative_queries",
        max_expansions: int = 3,
        cache: Optional[ExpansionCache] = None,
        strict: bool = False,
    ) -> None:
        self.llm = llm_client or OpenAICompatibleLLM(model="openai/gpt-oss-20b", timeout_seconds=30.0)
        self.enabled = enabled
        self.default_mode = default_mode
        self.max_expansions = max_expansions
        self.cache = cache
        self.strict = strict
        self.stats: Dict[str, int] = {
            "requests": 0, "success": 0, "success_empty": 0, "cache_hit": 0, "disabled": 0, "failed": 0,
        }

    def stats_report(self) -> Dict[str, Any]:
        s = dict(self.stats)
        s["failure_rate"] = (s["failed"] / s["requests"]) if s["requests"] else 0.0
        return s

    def _fail(self, query: str, mode: str, status: str, **extra: Any) -> QueryExpansionResult:
        self.stats["failed"] += 1
        msg = f"Query expansion FAILED (status={status}, mode={mode}): {extra.get('error') or extra.get('raw', '')[:120]}"
        if self.strict:
            raise QueryExpansionError(msg)
        logger.warning("%s — falling back to the original query only.", msg)
        return QueryExpansionResult(
            original_query=query,
            mode=mode,
            provenance_log={"status": status, "failed": True, "fallback_to_original": True, "cache": "miss", **extra},
        )

    def expand(
        self,
        query: str,
        mode: Optional[str] = None,
        max_expansions: Optional[int] = None,
    ) -> QueryExpansionResult:
        """Expand a query into structured alternatives."""
        selected_mode = mode or self.default_mode
        limit = max_expansions or self.max_expansions
        self.stats["requests"] += 1

        if not self.enabled or selected_mode == "none":
            self.stats["disabled"] += 1
            return QueryExpansionResult(
                original_query=query,
                mode="none",
                expanded_queries=[],
                provenance_log={"status": "disabled", "failed": False, "fallback_to_original": False, "cache": "disabled"},
            )

        if selected_mode == "alternative_queries":
            sys_prompt = ALTERNATIVE_QUERIES_SYSTEM_PROMPT
        elif selected_mode in ("terminology", "terminology_expansion"):
            sys_prompt = TERMINOLOGY_EXPANSION_SYSTEM_PROMPT
        elif selected_mode == "decomposition":
            sys_prompt = DECOMPOSITION_SYSTEM_PROMPT
        else:
            raise ValueError(f"Unknown query expansion mode: {selected_mode}")

        key = None
        if self.cache is not None:
            key = expansion_cache_key(query, selected_mode, str(getattr(self.llm, "model", "unknown")), sys_prompt, limit)
            hit = self.cache.get(key)
            if hit is not None:
                self.stats["cache_hit"] += 1
                return QueryExpansionResult(
                    original_query=query,
                    mode=selected_mode,
                    expanded_queries=[q for q in hit["expanded_queries"] if q.strip() != query][:limit],
                    key_entities=list(hit.get("key_entities") or []),
                    exact_identifiers=list(hit.get("exact_identifiers") or []),
                    constraints=list(hit.get("constraints") or []),
                    negations=list(hit.get("negations") or []),
                    uncertainties=list(hit.get("uncertainties") or []),
                    subquestions=[
                        SubQuestion(
                            subquestion=str(sq["subquestion"]).strip(),
                            requested_information=str(sq.get("requested_information", "")),
                            entities_identifiers=list(sq.get("entities_identifiers") or []),
                            expected_evidence_type=sq.get("expected_evidence_type"),
                            dependencies=list(sq.get("dependencies") or []),
                            constraints=list(sq.get("constraints") or []),
                        )
                        for sq in hit.get("subquestions", [])
                    ],
                    provenance_log={"status": "cache_hit", "failed": False, "fallback_to_original": False, "cache": "hit"},
                )

        user_prompt = f"QUESTION TO ANALYZE:\n{query}\n\nSTRICT JSON OUTPUT:"

        try:
            resp = self.llm.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception as e:
            return self._fail(query, selected_mode, "fallback_error", error=str(e)[:300])

        if getattr(resp, "finish_reason", "stop") == "length":
            return self._fail(query, selected_mode, "fallback_truncated", raw=(resp.text or "")[:200])

        clean_text = (resp.text or "").strip()
        m = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not m:
            return self._fail(query, selected_mode, "fallback_no_json", raw=clean_text[:200])

        try:
            data = json.loads(m.group(0))
            if not isinstance(data, dict):
                raise ValueError("JSON root is not an object")
            raw_exp = data.get("expanded_queries", [])
            raw_sq = data.get("subquestions", [])
            if not isinstance(raw_exp, list) or not isinstance(raw_sq, list):
                raise ValueError("expanded_queries / subquestions must be lists")
        except Exception as e:
            return self._fail(query, selected_mode, "fallback_error", error=f"invalid JSON: {e}"[:300])

        valid_exp = [str(q).strip() for q in raw_exp if q and str(q).strip() and str(q).strip() != query][:limit]

        subquestions: List[SubQuestion] = []
        for sq in raw_sq[:limit]:
            if isinstance(sq, dict) and sq.get("subquestion"):
                subquestions.append(
                    SubQuestion(
                        subquestion=str(sq["subquestion"]).strip(),
                        requested_information=str(sq.get("requested_information", "")),
                        entities_identifiers=list(sq.get("entities_identifiers") or []),
                        expected_evidence_type=sq.get("expected_evidence_type"),
                        dependencies=list(sq.get("dependencies") or []),
                        constraints=list(sq.get("constraints") or []),
                    )
                )

        result = QueryExpansionResult(
            original_query=query,
            mode=selected_mode,
            expanded_queries=valid_exp,
            key_entities=list(data.get("key_entities") or []),
            exact_identifiers=list(data.get("exact_identifiers") or []),
            constraints=list(data.get("constraints") or []),
            negations=list(data.get("negations") or []),
            uncertainties=list(data.get("uncertainties") or []),
            subquestions=subquestions,
            provenance_log={"latency_ms": getattr(resp, "latency_ms", 0.0), "failed": False, "fallback_to_original": False},
        )
        if not valid_exp and not subquestions:
            self.stats["success_empty"] += 1
            logger.warning("Query expansion returned no usable new queries (mode=%s); retrieval uses the original query only.", selected_mode)
            result.provenance_log.update(status="success_empty", fallback_to_original=True, cache="not_stored")
            return result

        self.stats["success"] += 1
        stored = self.cache.put(key, result) if self.cache is not None and key else False
        result.provenance_log.update(status="success", cache="write" if stored else ("disabled" if self.cache is None else "not_stored"))
        return result


class MultiQueryRetriever:
    """Executes multi-query retrieval with provenance tracking and Reciprocal Rank Fusion."""

    def __init__(
        self,
        retriever: EnterpriseRetriever,
        expander: Optional[QueryExpander] = None,
    ) -> None:
        self.retriever = retriever
        self.expander = expander or QueryExpander()

    def retrieve(
        self,
        query: str,
        mode: str = "bm25",
        expansion_mode: str = "none",
        top_k_docs: int = 40,
        diversity_filter: bool = True,
        max_expansions: int = 3,
    ) -> Tuple[List[str], List[RetrievedChunk], QueryExpansionResult, Dict[str, Any]]:
        """Retrieve documents using query expansion or decomposition with RRF fusion."""
        # 1. Expand query
        exp_res = self.expander.expand(query, mode=expansion_mode, max_expansions=max_expansions)
        all_queries = exp_res.all_search_queries

        # If only original query (C0), run standard retrieval
        if len(all_queries) == 1:
            doc_ids, chunks = self.retriever.retrieve_documents(
                query=query,
                mode=mode,
                top_k_docs=top_k_docs,
                diversity_filter=diversity_filter,
            )
            provenance = {query: {"retrieved_docs": doc_ids[:10], "branch_weight": 1.0}}
            return doc_ids, chunks, exp_res, provenance

        # Multi-query execution
        branch_results: List[Tuple[str, float, List[str], List[RetrievedChunk]]] = []
        provenance = {}

        for idx, q_branch in enumerate(all_queries):
            # Original query has weight 1.0, expansions have weight 0.8
            weight = 1.0 if idx == 0 else 0.8
            docs, chunks = self.retriever.retrieve_documents(
                query=q_branch,
                mode=mode,
                top_k_docs=top_k_docs,
                diversity_filter=diversity_filter,
            )
            branch_results.append((q_branch, weight, docs, chunks))
            provenance[q_branch] = {
                "branch_weight": weight,
                "retrieved_docs": docs[:10],
                "num_chunks": len(chunks),
            }

        # 2. Reciprocal Rank Fusion across all query branches
        # score(d) = sum( weight / (60 + rank) )
        doc_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}
        doc_best_chunk: Dict[str, RetrievedChunk] = {}

        for q_branch, weight, docs, chunks in branch_results:
            for rank, doc_id in enumerate(docs, start=1):
                rrf_val = weight / (60.0 + rank)
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + rrf_val

            for ch in chunks:
                if ch.chunk_id not in chunk_map:
                    chunk_map[ch.chunk_id] = ch
                if ch.doc_id not in doc_best_chunk or ch.score > doc_best_chunk[ch.doc_id].score:
                    doc_best_chunk[ch.doc_id] = ch

        # Sort documents by fused RRF score
        sorted_doc_ids = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)[:top_k_docs]

        # Order chunks corresponding to fused ranking
        ordered_chunks: List[RetrievedChunk] = []
        seen_chunks = set()
        for doc_id in sorted_doc_ids:
            # Gather chunks belonging to this document from all branches
            doc_chunks = [ch for ch in chunk_map.values() if ch.doc_id == doc_id]
            doc_chunks.sort(key=lambda c: c.score, reverse=True)
            for ch in doc_chunks:
                if ch.chunk_id not in seen_chunks:
                    seen_chunks.add(ch.chunk_id)
                    ordered_chunks.append(ch)

        return sorted_doc_ids, ordered_chunks, exp_res, provenance
