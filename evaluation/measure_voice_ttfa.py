"""Measure real end-to-end Voice RAG latency and Time-to-First-Audio (TTFA).

Flow:
1. Synthesize audio speech input query (.aiff / .wav)
2. Transcribe via Groq Whisper API (whisper-large-v3-turbo)
3. Retrieve evidence (K=40 + parent diversity)
4. Build context
5. Stream generation via openai/gpt-oss-20b
6. Synthesize first spoken phrase to audio
7. Output complete stage-by-stage latency and TTFA breakdown
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from rag.context_builder import ContextBuilder
from rag.generator import AnswerGenerator
from rag.llm import OpenAICompatibleLLM, load_env_file
from rag.pipeline import RAGPipeline
from retrieval.indexer import EnterpriseRetriever
from voice.voice_pipeline import GroqWhisperSTT, StreamingTTS, VoiceRAGPipeline

load_env_file()
BASE_DIR = Path(__file__).resolve().parent.parent


def measure_voice_ttfa():
    audio_dir = BASE_DIR / "evaluation" / "audio_tests"
    audio_dir.mkdir(parents=True, exist_ok=True)

    input_audio_path = audio_dir / "query_clearedge.aiff"
    query_text = "What is the final PO fingerprint format for the ClearEdge 2027 pilot?"

    print(f"Generating spoken audio query: '{query_text}'...", flush=True)
    subprocess.run(
        ["/usr/bin/say", "-v", "Samantha", "-o", str(input_audio_path), query_text],
        check=True,
    )
    print(f"Created audio input: {input_audio_path} ({input_audio_path.stat().st_size} bytes)", flush=True)

    # Initialize components
    retriever = EnterpriseRetriever(
        collection_name="enterprise_rag_dev_10k",
        qdrant_path=BASE_DIR / "data" / "qdrant_dev_10k",
    )
    context_builder = ContextBuilder(max_context_tokens=2500)
    llm = OpenAICompatibleLLM(model="openai/gpt-oss-20b", timeout_seconds=60.0)
    generator = AnswerGenerator(llm_client=llm)
    rag_pipe = RAGPipeline(
        retriever=retriever,
        generator=generator,
        context_builder=context_builder,
        default_mode="bm25",
    )

    stt = GroqWhisperSTT(model="whisper-large-v3-turbo")
    tts = StreamingTTS(voice_name="Samantha")
    voice_pipe = VoiceRAGPipeline(rag_pipeline=rag_pipe, stt=stt, tts=tts)

    print("\nRunning End-to-End Voice RAG Pipeline...", flush=True)
    voice_resp = voice_pipe.process_voice_query(
        audio_input_path=input_audio_path,
        save_audio_response=True,
    )

    m = voice_resp.metrics
    print("\n=======================================================", flush=True)
    print("VOICE RAG STAGE-BY-STAGE LATENCY & TTFA BREAKDOWN", flush=True)
    print("=======================================================", flush=True)
    print(f"Transcribed Input Query:     {voice_resp.transcription}", flush=True)
    print(f"First Spoken Audio Phrase:   {m.first_spoken_sentence}", flush=True)
    print(f"Audio Response File:         {voice_resp.audio_output_path}", flush=True)
    print("-------------------------------------------------------", flush=True)
    print(f"1. Speech-to-Text (STT):     {m.stt_latency_ms:.1f} ms", flush=True)
    print(f"2. Retrieval (BM25 K=40):    {m.retrieval_latency_ms:.1f} ms", flush=True)
    print(f"3. Context Builder:          {m.context_builder_latency_ms:.1f} ms", flush=True)
    print(f"4. LLM Time-to-First-Token:  {m.llm_ttft_ms:.1f} ms", flush=True)
    print(f"5. First Phrase TTS Synth:   {m.tts_first_chunk_latency_ms:.1f} ms", flush=True)
    print("-------------------------------------------------------", flush=True)
    print(f">> TIME TO FIRST AUDIO (TTFA): {m.total_ttfa_ms:.1f} ms  ({m.total_ttfa_ms/1000.0:.2f} s)", flush=True)
    print(f">> TOTAL END-TO-END LATENCY:   {m.total_e2e_ms:.1f} ms  ({m.total_e2e_ms/1000.0:.2f} s)", flush=True)
    print("=======================================================", flush=True)

    # Save metrics report
    report_path = BASE_DIR / "evaluation" / "reports" / "voice_ttfa_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metrics": m.to_dict(),
                "answer_text": voice_resp.rag_answer.answer_text,
                "audio_output_path": voice_resp.audio_output_path,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved Voice TTFA Report to: {report_path}", flush=True)
    rag_pipe.close()


if __name__ == "__main__":
    measure_voice_ttfa()
