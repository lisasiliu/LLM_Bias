"""
vllm_client.py -- Lightweight vLLM OpenAI-compatible client for lurkingwolf.

Provides:
    client.chat.completions.create(model=..., messages=..., ...)

Uses direct HTTP requests to the vLLM /v1/chat/completions endpoint.
"""

import os
from dataclasses import dataclass
from typing import Optional

import requests


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

        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _CompletionResponse(
            choices=[_Choice(message=_Message(content=text))]
        )


class _Chat:
    def __init__(self, base_url: str, api_key: str):
        self.completions = _Completions(base_url, api_key)


class VLLMClient:
    """OpenAI-compatible client targeting a vLLM server."""

    def __init__(self, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self._base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000")
        self._api_key = api_key or os.getenv("VLLM_API_KEY", "EMPTY")
        self.chat = _Chat(self._base_url, self._api_key)
