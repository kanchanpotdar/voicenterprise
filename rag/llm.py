"""Replaceable LLM client interface and implementations for Phase 2 RAG.

Key guarantees:
- Never hardcodes API keys; reads from environment variables.
- Lightweight: uses `httpx` for standard OpenAI-compatible endpoints (OpenAI,
  Groq, Ollama, vLLM, DeepSeek).
- Includes a fully deterministic `MockLLM` for testing and offline evaluation
  without paid API calls or network dependencies.
- Measures latency, token consumption, and estimated generation costs.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent


def load_env_file(env_path: Optional[Path] = None) -> None:
    """Lightweight pure-Python .env file loader."""
    path = env_path or (BASE_DIR / ".env")
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


# Automatically load .env on module import
load_env_file()

# Cost estimates per 1k tokens (approximate reference)
PRICING_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.0100},
    "llama-3.1-8b-instant": {"prompt": 0.00005, "completion": 0.00008},
    "mock": {"prompt": 0.0, "completion": 0.0},
}


def is_valid_key(key: Optional[str]) -> bool:
    """Return True if key is non-empty and not a placeholder string."""
    if not key:
        return False
    k = key.strip().strip("'\"")
    return bool(k and not k.startswith("your-"))


@dataclass
class LLMResponse:
    """Standardized response from an LLM provider."""

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cost_estimate_usd: float = 0.0
    ttft_ms: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        """Generate a completion for the given prompt."""
        pass


class OpenAICompatibleLLM(BaseLLMClient):
    """Client for OpenAI and any OpenAI-compatible API (Groq, Ollama, vLLM, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 45.0,
    ) -> None:
        openai_key = os.getenv("OPENAI_API_KEY") if is_valid_key(os.getenv("OPENAI_API_KEY")) else None
        groq_key = os.getenv("GROQ_API_KEY") if is_valid_key(os.getenv("GROQ_API_KEY")) else None

        self.api_key = api_key or openai_key or groq_key
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY or GROQ_API_KEY in environment or .env"
            )

        # Default base URL based on key or environment
        if base_url:
            self.base_url = base_url.rstrip("/")
        elif os.getenv("OPENAI_BASE_URL"):
            self.base_url = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
        elif groq_key and not openai_key:
            self.base_url = "https://api.groq.com/openai/v1"
            if model == "gpt-4o-mini":
                model = "openai/gpt-oss-120b"
        else:
            self.base_url = "https://api.openai.com/v1"

        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        t0 = time.perf_counter()
        url = f"{self.base_url}/chat/completions"
        max_retries = 5
        backoff = 2.0

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    if not stream:
                        resp = client.post(url, headers=headers, json=payload)
                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        if resp.status_code == 429 and attempt < max_retries - 1:
                            retry_after = resp.headers.get("retry-after")
                            try:
                                sleep_time = float(retry_after) if retry_after else (backoff * (attempt + 1))
                            except (ValueError, TypeError):
                                sleep_time = backoff * (attempt + 1)
                            time.sleep(max(1.0, sleep_time))
                            continue

                        if resp.status_code != 200:
                            raise RuntimeError(
                                f"LLM API request failed ({resp.status_code}): {resp.text}"
                            )

                        data = resp.json()
                        choice = data["choices"][0]
                        text = choice["message"]["content"] or ""
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)

                        # Estimate cost
                        rates = PRICING_PER_1K.get(self.model, {"prompt": 0.001, "completion": 0.002})
                        cost = (prompt_tokens / 1000.0) * rates["prompt"] + (
                            completion_tokens / 1000.0
                        ) * rates["completion"]

                        return LLMResponse(
                            text=text.strip(),
                            model=self.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            latency_ms=round(latency_ms, 2),
                            finish_reason=choice.get("finish_reason", "stop"),
                            cost_estimate_usd=round(cost, 6),
                            raw=data,
                            ttft_ms=round(latency_ms, 2),
                        )
                    else:
                        # Streaming mode: captures TTFT directly
                        ttft_ms = 0.0
                        collected_chunks: List[str] = []
                        finish_reason = "stop"
                        with client.stream("POST", url, headers=headers, json=payload) as resp:
                            if resp.status_code == 429 and attempt < max_retries - 1:
                                retry_after = resp.headers.get("retry-after")
                                try:
                                    sleep_time = float(retry_after) if retry_after else (backoff * (attempt + 1))
                                except (ValueError, TypeError):
                                    sleep_time = backoff * (attempt + 1)
                                time.sleep(max(1.0, sleep_time))
                                continue

                            if resp.status_code != 200:
                                resp.read()
                                raise RuntimeError(
                                    f"LLM API streaming request failed ({resp.status_code}): {resp.text}"
                                )

                            import json
                            for line in resp.iter_lines():
                                line = line.strip()
                                if not line or not line.startswith("data:"):
                                    continue
                                data_str = line[5:].strip()
                                if data_str == "[DONE]":
                                    break
                                try:
                                    chunk = json.loads(data_str)
                                    choices = chunk.get("choices", [])
                                    if not choices:
                                        continue
                                    delta = choices[0].get("delta", {})
                                    content_piece = delta.get("content", "")
                                    if content_piece:
                                        if ttft_ms == 0.0:
                                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                                        collected_chunks.append(content_piece)
                                        if on_token:
                                            on_token(content_piece)
                                    if choices[0].get("finish_reason"):
                                        finish_reason = choices[0]["finish_reason"]
                                except Exception:
                                    continue

                        total_latency_ms = (time.perf_counter() - t0) * 1000.0
                        full_text = "".join(collected_chunks).strip()
                        prompt_tokens = len(prompt) // 4
                        completion_tokens = len(full_text) // 4
                        rates = PRICING_PER_1K.get(self.model, {"prompt": 0.001, "completion": 0.002})
                        cost = (prompt_tokens / 1000.0) * rates["prompt"] + (
                            completion_tokens / 1000.0
                        ) * rates["completion"]

                        return LLMResponse(
                            text=full_text,
                            model=self.model,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            latency_ms=round(total_latency_ms, 2),
                            finish_reason=finish_reason,
                            cost_estimate_usd=round(cost, 6),
                            raw={"streaming": True},
                            ttft_ms=round(ttft_ms if ttft_ms > 0 else total_latency_ms, 2),
                        )
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(backoff * (attempt + 1))


class MockLLM(BaseLLMClient):
    """Deterministic mock LLM for testing and offline evaluation without API costs."""

    def __init__(
        self,
        default_response: Optional[str] = None,
        response_fn: Optional[Callable[[str, Optional[str]], str]] = None,
        canned_responses: Optional[Dict[str, str]] = None,
        latency_ms: float = 12.0,
    ) -> None:
        self.default_response = default_response
        self.response_fn = response_fn
        self.canned_responses = canned_responses or {}
        self.latency_ms = latency_ms

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        # Check canned responses by prompt substring
        text = None
        for key, resp in self.canned_responses.items():
            if key in prompt:
                text = resp
                break

        if text is None and self.response_fn is not None:
            text = self.response_fn(prompt, system_prompt)

        if text is None:
            if self.default_response is not None:
                text = self.default_response
            else:
                # Default mock behavior: extract and echo the first document citation from the prompt
                import re

                doc_ids = re.findall(r"\b(dsid_[a-f0-9]{32})\b", prompt)
                if doc_ids:
                    text = (
                        f"Based on the provided documentation [{doc_ids[0]}], "
                        f"the verified operational procedure is documented in the source record."
                    )
                else:
                    text = "I cannot answer this query based on the provided evidence."

        if stream and on_token:
            for word in text.split(" "):
                on_token(word + " ")

        return LLMResponse(
            text=text,
            model="mock-llm-v1",
            prompt_tokens=len(prompt) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=self.latency_ms,
            finish_reason="stop",
            cost_estimate_usd=0.0,
            ttft_ms=min(self.latency_ms, 2.0),
        )
