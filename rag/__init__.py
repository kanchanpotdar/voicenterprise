"""Phase 2: Baseline Text RAG System.

Modules:
- `rag.context_builder`: Token-budgeted evidence selection and multi-doc balancing.
- `rag.citation_validator`: Deterministic citation extraction and provenance validation.
- `rag.llm`: Replaceable LLM interface with OpenAI-compatible and mock implementations.
- `rag.generator`: Grounded generation, citation enforcement, and abstention handling.
- `rag.pipeline`: End-to-end RAG orchestrator.
"""

from rag.citation_validator import CitationValidationResult, CitationValidator
from rag.context_builder import AssembledContext, ContextBuilder, EvidencePassage
from rag.generator import AnswerGenerator, RAGAnswer
from rag.llm import BaseLLMClient, LLMResponse, MockLLM, OpenAICompatibleLLM
from rag.pipeline import RAGPipeline

__all__ = [
    "AssembledContext",
    "ContextBuilder",
    "EvidencePassage",
    "CitationValidationResult",
    "CitationValidator",
    "BaseLLMClient",
    "LLMResponse",
    "OpenAICompatibleLLM",
    "MockLLM",
    "AnswerGenerator",
    "RAGAnswer",
    "RAGPipeline",
]
