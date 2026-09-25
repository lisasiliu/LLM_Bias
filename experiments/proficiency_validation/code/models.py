"""Model registry for English validation, including independently hosted vLLM models.

PROFICIENCY_MODEL_CONFIG points to a JSON mapping of named models. Endpoint
locations may change between HPC jobs; model identities and inference settings
must match the frozen configuration. Local servers use endpoint-specific
authentication settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from components.judges.judge_model import (
    JudgeModel, ModelSpec, Completion, JudgeModelError, JudgeTransportError,
    THINK_OPEN, resolve_spec,
)
from lurkingwolf.utils.speech_nl import strip_reasoning


def registry():
    path = os.getenv("PROFICIENCY_MODEL_CONFIG")
    return json.loads(Path(path).read_text(encoding="utf-8")).get("models", {}) if path else {}


def model_identity(handle):
    entry = registry().get(handle)
    if entry is None:
        spec = resolve_spec(handle)
        return {"handle": handle, "model": spec.model, "family": spec.family,
                "revision": None, "tokenizer_revision": None, "serving_version": None,
                "precision": None, "chat_template": None, "extra_body": {}}
    return {"handle": handle, **{key: entry.get(key) for key in (
        "model", "family", "revision", "tokenizer_revision", "serving_version",
        "precision", "chat_template")}, "extra_body": entry.get("extra_body", {})}


def completion_url(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1") or base.endswith("/v1beta/openai"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class ValidationModel(JudgeModel):
    def __init__(self, spec, *, identity=None, extra_body=None, ledger=None, purpose="other", **kwargs):
        super().__init__(spec, **kwargs)
        self.identity_record = identity or model_identity(spec.handle)
        self.extra_body = dict(extra_body or {})
        forbidden = {"model", "messages", "temperature", "max_tokens", "stream"} & self.extra_body.keys()
        if forbidden:
            raise ValueError(f"extra_body must not override {sorted(forbidden)}")
        self.ledger, self.purpose = ledger, purpose

    @classmethod
    def resolve(cls, handle, **kwargs):
        entry = registry().get(handle)
        if entry is None:
            spec = resolve_spec(handle)
        else:
            base = os.getenv(entry.get("base_url_env", "")) or entry.get("base_url")
            if not base or not entry.get("model") or not entry.get("family"):
                raise ValueError(f"{handle}: model, family and a configured server address are required")
            spec = ModelSpec(handle=handle, provider="local", model=entry["model"],
                             base_url=base, family=entry["family"],
                             api_key_env=tuple([entry["api_key_env"]]) if entry.get("api_key_env") else ())
        return cls(spec, identity=model_identity(handle),
                   extra_body=(entry or {}).get("extra_body"), **kwargs)

    def complete(self, *, system, user, max_tokens=None, temperature=None):
        payload = {**self.extra_body, "model": self.spec.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "temperature": self.temperature if temperature is None else temperature,
                   "max_tokens": self.max_tokens if max_tokens is None else max_tokens}
        headers = self._headers()
        for attempt in range(self.retries + 1):
            try:
                response = requests.post(completion_url(self.spec.base_url), headers=headers,
                                         json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.retries:
                    raise JudgeTransportError(str(exc)) from exc
                self._sleep(self._backoff * 2 ** attempt)
                continue
            break
        self.calls += 1
        self.transport_retries += attempt
        self.model_served = str(data.get("model") or "")
        if self.model_served != self.spec.model:
            raise JudgeModelError(f"requested {self.spec.model!r}, server returned {self.model_served!r}")
        usage = dict(data.get("usage") or {})
        if self.ledger is not None:
            self.ledger.record(self.spec.provider, self.model_served, self.purpose, usage)
        try:
            choice = data["choices"][0]
            raw = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeModelError("malformed completion") from exc
        if choice.get("finish_reason") == "length":
            raise JudgeModelError("completion exceeded the frozen token budget")
        text = strip_reasoning(raw).strip()
        if not text or THINK_OPEN.search(text):
            raise JudgeModelError("empty or incomplete completion")
        return Completion(text, self.spec.model, self.model_served, usage, attempt)

    def preflight(self):
        # Check the served model without requiring a reasoning model to answer in eight tokens.
        url = completion_url(self.spec.base_url).removesuffix("chat/completions") + "models"
        try:
            response = requests.get(url, headers=self._headers(), timeout=min(self.timeout, 30))
            response.raise_for_status()
            ids = {m["id"] for m in response.json()["data"]}
            if self.spec.model not in ids:
                return False, f"server does not expose {self.spec.model!r}"
            return True, "model is served; revision metadata is operator-declared"
        except (requests.RequestException, ValueError, KeyError, JudgeModelError) as exc:
            return False, str(exc)

    def provenance(self):
        return {**super().provenance(), "identity": self.identity_record,
                "revision_verification": "operator-declared; archive the vLLM launch command and environment"}
