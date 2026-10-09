"""Minimal OpenAI-compatible client (chat + embeddings) using only the stdlib.

Targets a local llama.cpp server (`llama-server`), but works with any
OpenAI-compatible endpoint (LM Studio, Ollama's /v1, vLLM, ...).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from urllib.parse import urlparse

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class LLMError(RuntimeError):
    pass


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str = "local", api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        # Never route loopback traffic through a system/corporate proxy (a common Windows trap).
        host = urlparse(self.base_url).hostname or ""
        handlers = [urllib.request.ProxyHandler({})] if host in _LOOPBACK else []
        self._opener = urllib.request.build_opener(*handlers)

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers, method="POST")
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"HTTP {e.code} from {self.base_url}{path}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise LLMError(f"Cannot reach {self.base_url}{path}: {e}") from e

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
