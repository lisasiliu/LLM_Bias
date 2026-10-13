"""
llm_client.py -- Lightweight OpenAI-compatible HTTP client for lurkingwolf.

Provides:
    client.chat.completions.create(model=..., messages=..., ...)

Works with any OpenAI-compatible endpoint: vLLM, Ollama, LiteLLM,
TGI, OpenRouter, or the OpenAI API itself.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import requests

# One pooled session per process: a replay issues hundreds of calls, and a new TCP
# connection for each of them overflows the server's accept queue once dozens of
# replays share one endpoint, which the client sees as a closed connection.
_SESSION = requests.Session()
_SESSION.mount("http://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16))
_SESSION.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16))
_CONNECTION_ATTEMPTS = int(os.getenv("LLM_CONNECT_ATTEMPTS", "4"))


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _CompletionResponse:
    choices: list


class _Completions:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    def create(self, *, model: str, messages: list,
               temperature: float = 0.7, max_tokens: int = 300,
               seed: Optional[int] = None, top_p: Optional[float] = None,
               extra_body: Optional[dict] = None,
               **kwargs) -> _CompletionResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key and self._api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        if extra_body:
            payload.update(extra_body)

        timeout = int(os.getenv("LLM_TIMEOUT", "300"))   # seconds; raise it when many replays share one server
        url = f"{self._base_url}/v1/chat/completions"
        for attempt in range(1, _CONNECTION_ATTEMPTS + 1):
            try:
                resp = _SESSION.post(url, headers=headers, json=payload, timeout=timeout)
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
                # The connection was closed before the server answered, so this request was
                # never served: asking again returns a sample for the same prompt.
                if attempt == _CONNECTION_ATTEMPTS:
                    raise
                time.sleep(min(30.0, 2.0 ** attempt) * (0.5 + random.random()))
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _CompletionResponse(
            choices=[_Choice(message=_Message(content=text))]
        )


class _Chat:
    def __init__(self, base_url: str, api_key: str):
        self.completions = _Completions(base_url, api_key)


class LLMClient:
    """OpenAI-compatible HTTP client for any /v1/chat/completions endpoint."""

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._base_url = base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000")
        self._api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.chat = _Chat(self._base_url, self._api_key)
