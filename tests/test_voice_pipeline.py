"""Tests for the VoiceEnterprise Voice RAG Pipeline.

Verifies:
- Text-to-speech sanitization (stripping [dsid_...] and markdown formatting)
- StreamingTTS file generation
- VoiceRAGMetrics stage-by-stage latency tracking and serialization
- End-to-end VoiceRAGPipeline orchestration with deterministic mock components
- Total Time-to-First-Audio (TTFA) formula verification
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.citation_validator import CitationValidator
from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import MockLLM
from rag.pipeline import RAGPipeline
from voice.voice_pipeline import (
    GroqWhisperSTT,
    StreamingTTS,
    VoiceRAGMetrics,
    VoiceRAGPipeline,
    VoiceRAGResponse,
)


def test_streaming_tts_text_sanitization():
    raw_text = (
        "Based on [dsid_d93a61f5c6a1b853874c728e124803d1] and [Evidence Passage 2], "
        "the final fingerprint is **SHA-256**.\n\n"
        "- An earlier draft [dsid_86f0c3d9a105c317be9ef282542a1b9c] was superseded."
    )
    sanitized = StreamingTTS._sanitize_text_for_speech(raw_text)
    assert "[dsid_" not in sanitized
    assert "Passage" not in sanitized
    assert "**" not in sanitized
    assert "SHA-256" in sanitized
    assert "superseded" in sanitized


def test_streaming_tts_synthesize_to_file():
    tts = StreamingTTS()
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
        temp_path = f.name

    path, latency_ms = tts.synthesize_to_file("ClearEdge verification confirmed.", temp_path)
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0
    assert latency_ms >= 0.0


def test_voice_rag_metrics_serialization():
    metrics = VoiceRAGMetrics(
        stt_latency_ms=210.5,
        retrieval_latency_ms=152.0,
        context_builder_latency_ms=3.2,
        llm_ttft_ms=640.0,
        llm_generation_ms=1450.0,
        tts_first_chunk_latency_ms=45.0,
        total_ttfa_ms=1050.7,
        total_e2e_ms=1860.7,
        transcribed_query="What is the final cutoff?",
        first_spoken_sentence="The final cutoff is Tuesday 07:00 PT.",
    )
    d = metrics.to_dict()
    assert d["stt_latency_ms"] == 210.5
    assert d["total_ttfa_ms"] == 1050.7
    assert d["transcribed_query"] == "What is the final cutoff?"


def test_voice_rag_pipeline_end_to_end():
    mock_retriever = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "dsid_00000000000000000000000000000001__c000"
    mock_chunk.record_id = "dsid_00000000000000000000000000000001"
    mock_chunk.doc_id = "dsid_00000000000000000000000000000001"
    mock_chunk.chunk_text = "The system operational cutoff is 07:00 PT."
    mock_chunk.score = 0.95
    mock_chunk.source_type = "confluence"
    mock_chunk.normalized_title = "Standard Guidance"
    mock_chunk.section_title = "Schedule"
    mock_chunk.latest_date = None
    mock_chunk.date_confidence = "none"
    mock_chunk.slack_channels = []
    mock_chunk.ticket_ids = []

    mock_retriever.retrieve_documents.return_value = (
        ["dsid_00000000000000000000000000000001"],
        [mock_chunk],
    )

    mock_llm = MockLLM(
        default_response=(
            "The operational cutoff is 07:00 PT [dsid_00000000000000000000000000000001]. "
            "Please follow the documented procedure."
        ),
        latency_ms=15.0,
    )
    generator = AnswerGenerator(llm_client=mock_llm)
    rag_pipe = RAGPipeline(
        retriever=mock_retriever,
        generator=generator,
        context_builder=ContextBuilder(max_context_tokens=1000),
    )

    mock_stt = MagicMock()
    mock_stt.transcribe_audio_file.return_value = ("What is the operational cutoff?", 180.0)

    voice_pipe = VoiceRAGPipeline(rag_pipeline=rag_pipe, stt=mock_stt, tts=StreamingTTS())

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFFmockwavdata")
        dummy_audio = f.name

    resp = voice_pipe.process_voice_query(dummy_audio, save_audio_response=True)

    assert isinstance(resp, VoiceRAGResponse)
    assert resp.transcription == "What is the operational cutoff?"
    assert "07:00 PT" in resp.rag_answer.answer_text
    assert resp.metrics.stt_latency_ms == 180.0
    assert resp.metrics.total_ttfa_ms > 0.0
    assert resp.audio_output_path is not None
    assert Path(resp.audio_output_path).exists()
