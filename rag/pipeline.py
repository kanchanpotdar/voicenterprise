"""Modular end-to-end RAG pipeline orchestrator for Phase 2.

Pipeline Flow:
User Question
  -> 1. Retrieval (BM25 default, Dense, or Hybrid)
  -> 2. Evidence Selection & Context Assembly (balanced across parent docs)
  -> 3. Grounded LLM Generation (with citation enforcement)
  -> 4. Deterministic Citation Validation
  -> 5. Structured Final Answer
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator, RAGAnswer
from rag.llm import BaseLLMClient, MockLLM, OpenAICompatibleLLM
from retrieval.indexer import EnterpriseRetriever, RetrievalMode

BASE_DIR = Path(__file__).resolve().parent.parent


class RAGPipeline:
    """End-to-end, modular Phase 2 RAG system."""

    def __init__(
        self,
        retriever: EnterpriseRetriever,
        generator: AnswerGenerator,
        context_builder: Optional[ContextBuilder] = None,
        default_mode: RetrievalMode = "bm25",
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.context_builder = context_builder or ContextBuilder()
        self.default_mode = default_mode

    @classmethod
    def from_defaults(
        cls,
        qdrant_path: Optional[Path] = None,
        collection_name: str = "enterprise_rag_dev_10k",
        default_mode: RetrievalMode = "bm25",
        llm_client: Optional[BaseLLMClient] = None,
        model_name: str = "gpt-4o-mini",
        max_context_tokens: int = 2500,
    ) -> RAGPipeline:
        """Create a pipeline using standard project paths and environment configuration."""
        q_path = qdrant_path or (BASE_DIR / "data" / "qdrant_dev_10k")

        # Initialize retriever
        retriever = EnterpriseRetriever(
            collection_name=collection_name,
            qdrant_path=q_path,
        )

        # Initialize LLM client: use provided, or OpenAI/Groq if API key exists, otherwise MockLLM
        if llm_client is None:
            has_api_key = bool(
                os.getenv("OPENAI_API_KEY")
                or os.getenv("GROQ_API_KEY")
                or os.getenv("OPENAI_BASE_URL")
            )
            if has_api_key:
                effective_model = model_name
                if (
                    os.getenv("GROQ_API_KEY")
                    and not os.getenv("OPENAI_API_KEY")
                    and model_name == "gpt-4o-mini"
                ):
                    effective_model = "llama-3.1-8b-instant"
                llm_client = OpenAICompatibleLLM(model=effective_model)
            else:
                llm_client = MockLLM()

        context_builder = ContextBuilder(max_context_tokens=max_context_tokens)
        generator = AnswerGenerator(llm_client=llm_client)

        return cls(
            retriever=retriever,
            generator=generator,
            context_builder=context_builder,
            default_mode=default_mode,
        )

    def query(
        self,
        question: str,
        mode: Optional[RetrievalMode] = None,
        top_k_docs: int = 20,
        max_context_tokens: Optional[int] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        diversity_filter: bool = False,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> RAGAnswer:
        """Run the end-to-end RAG pipeline for a question with stage latency tracking."""
        t_pipeline_start = time.perf_counter()
        retrieval_mode = mode or self.default_mode

        # 1. Retrieve candidates
        t_ret0 = time.perf_counter()
        ordered_doc_ids, retrieved_chunks = self.retriever.retrieve_documents(
            query=question,
            mode=retrieval_mode,
            top_k_docs=top_k_docs,
            diversity_filter=diversity_filter,
        )
        retrieval_ms = (time.perf_counter() - t_ret0) * 1000.0

        # 2. Select evidence and assemble context within token budget
        t_ctx0 = time.perf_counter()
        assembled_context = self.context_builder.build_context(
            retrieved_chunks=retrieved_chunks,
            max_tokens_override=max_context_tokens,
        )
        context_builder_ms = (time.perf_counter() - t_ctx0) * 1000.0

        # 3. Generate grounded answer and validate citations
        answer = self.generator.generate_answer(
            question=question,
            assembled_context=assembled_context,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            on_token=on_token,
        )
        answer.retrieved_chunks = list(retrieved_chunks)
        answer.retrieval_ms = round(retrieval_ms, 2)
        answer.context_builder_ms = round(context_builder_ms, 2)
        answer.total_e2e_ms = round((time.perf_counter() - t_pipeline_start) * 1000.0, 2)

        return answer

    def close(self) -> None:
        """Clean up Qdrant client connection."""
        if hasattr(self.retriever, "client") and self.retriever.client is not None:
            try:
                self.retriever.client.close()
            except Exception:
                pass
