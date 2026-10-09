"""Minimal OpenAI-compatible client (chat + embeddings) using only the stdlib.

Targets a local llama.cpp server (`llama-server`), but works with any
OpenAI-compatible endpoint (LM Studio, Ollama's /v1, vLLM, ...).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str = "local", api_key: str | None = None, timeout: float = 120.0,
                 retries: int = 4, backoff: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries  # for rate limits (429), 5xx and timeouts, which hosted endpoints do return
        self.backoff = backoff
        # Never route loopback traffic through a system/corporate proxy (a common Windows trap).
        host = urlparse(self.base_url).hostname or ""
        handlers = [urllib.request.ProxyHandler({})] if host in _LOOPBACK else []
        self._opener = urllib.request.build_opener(*handlers)

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}{path}"
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with self._opener.open(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")[:500]
                if e.code in _RETRY_STATUS and attempt < self.retries:
                    time.sleep(_retry_after(e) or self.backoff * 2**attempt)
                    continue
                hint = {401: " (missing or invalid API key: set OLLAMA_API_KEY or pass --api-key)",
                        404: " (unknown model or endpoint: check --llm-model / --embed-url)"}.get(e.code, "")
                raise LLMError(f"HTTP {e.code} from {url}{hint}: {body}") from e
            except TimeoutError as e:
                if attempt < self.retries:
                    time.sleep(self.backoff * 2**attempt)
                    continue
                raise LLMError(f"Timed out after {self.timeout}s: {url}") from e
            except (urllib.error.URLError, ConnectionError) as e:
                raise LLMError(f"Cannot reach {url}: {e}") from e
        raise AssertionError("unreachable")

    def chat(self, messages: list[dict], temperature: float = 0.9, max_tokens: int = 80) -> str:
        out = self._post(
            "/chat/completions",
            {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
        )
        try:
            content = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected chat response shape: {str(out)[:300]}") from e
        # Reasoning models (Qwen3, DeepSeek-R1 distills) may leak their scratchpad.
        return _THINK_RE.sub("", content).strip()

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = self._post("/embeddings", {"model": self.model, "input": texts})
        try:
            rows = sorted(out["data"], key=lambda r: r["index"])
            vecs = [r["embedding"] for r in rows]
        except (KeyError, TypeError) as e:
            raise LLMError(f"Unexpected embeddings response shape: {str(out)[:300]}") from e
        if len(vecs) != len(texts):
            raise LLMError(f"Asked for {len(texts)} embeddings, got {len(vecs)}")
        if vecs and vecs[0] and isinstance(vecs[0][0], list):
            raise LLMError(
                "Server returned per-token embeddings. Start llama-server with "
                "--embeddings --pooling mean (or cls/last, per the model card)."
            )
        return vecs


def _retry_after(e: urllib.error.HTTPError) -> float | None:
    try:
        return min(float(e.headers.get("Retry-After", "")), 60.0)
    except (TypeError, ValueError):
        return None
