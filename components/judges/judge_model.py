"""One model behind a provider-independent call, so judges can swap families.

    model = JudgeModel.resolve("deepseek-chat")   # or "openai:gpt-5-mini"

Any OpenAI-compatible /v1/chat/completions endpoint. Records which model
actually answered, since a provider may serve a different one than requested.

Fails closed: transport errors raise and are retried; an empty reply or one cut
off inside an unterminated <think> block raises and is never retried, because
re-asking until the model answers is resampling until success.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

# A pooled session: the gate issues several judge calls per rewrite, and one new TCP
# connection per call overflows the accept queue of a locally served model.
_SESSION = requests.Session()
for _scheme in ("http://", "https://"):
    _SESSION.mount(_scheme, requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16))

from lurkingwolf.utils.speech_nl import strip_reasoning

THINK_OPEN = re.compile(r"</?think\b", re.IGNORECASE)


class JudgeTransportError(RuntimeError):
    """The endpoint could not be reached or returned an infrastructure error."""


class JudgeModelError(RuntimeError):
    """The endpoint answered, but the answer is unusable (empty / truncated)."""


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to call one model, and nothing else."""

    handle: str                 # short name used on the command line
    provider: str               # "deepseek" | "openai" | "local" | ...
    model: str                  # wire model id sent in the request body
    base_url: str
    api_key_env: tuple = ()     # env / .env names to try, in order
    family: str = ""            # for the same-family-judging caveat (PART 5)
    notes: str = ""

    @property
    def identity(self) -> str:
        return f"{self.provider}:{self.model}"


REGISTRY: Dict[str, ModelSpec] = {
    "deepseek-chat": ModelSpec(
        handle="deepseek-chat", provider="deepseek", model="deepseek-chat",
        base_url="https://api.deepseek.com",
        api_key_env=("DEEPSEEK_API_KEY", "DEEPSEEK"),
        family="deepseek",
        notes="pilot default; this account serves it as deepseek-flash",
    ),
    "gpt-5.6-luna": ModelSpec(
        handle="gpt-5.6-luna", provider="openai", model="gpt-5.6-luna",
        base_url="https://api.openai.com",
        api_key_env=("OPENAI_API_KEY", "OPENAI"),
        family="openai",
        notes="the LLM politeness judge. No dated version is published; accepts "
              "only the default temperature, so one pass is stochastic and the "
              "judge uses a 3-pass majority",
    ),
    "deepseek-reasoner": ModelSpec(
        handle="deepseek-reasoner", provider="deepseek", model="deepseek-reasoner",
        base_url="https://api.deepseek.com",
        api_key_env=("DEEPSEEK_API_KEY", "DEEPSEEK"),
        family="deepseek",
        notes="emits reasoning; needs a larger max_tokens",
    ),
}

PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "deepseek": {"base_url": "https://api.deepseek.com",
                 "api_key_env": ("DEEPSEEK_API_KEY", "DEEPSEEK"), "family": "deepseek"},
    "openai": {"base_url": "https://api.openai.com",
               "api_key_env": ("OPENAI_API_KEY", "OPENAI"), "family": "openai"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "api_key_env": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "family": "gemini"},
    "local": {"base_url": "", "api_key_env": ("LLM_API_KEY",), "family": "local"},
    # OpenRouter serves the OpenAI models under "openai/<name>"; same request style as the OpenAI provider.
    "openrouter": {"base_url": "https://openrouter.ai/api",
                   "api_key_env": ("OPENROUTER_API_KEY", "OPENROUTER"), "family": "openai"},
}

# The judge model used by the in-game gate and the blind strategy judge. The same gpt-5.6-luna, reached through OpenRouter
# while the OpenAI account has no credit; set it back to "gpt-5.6-luna" to use the OpenAI account.
LUNA = "openrouter:openai/gpt-5.6-luna"


def _dotenv_values() -> Dict[str, str]:
    """`.env` contents, if python-dotenv and a `.env` are both available."""
    try:
        from dotenv import dotenv_values, find_dotenv
    except ImportError:                                    # pragma: no cover
        return {}
    path = find_dotenv(usecwd=True)
    return {k: v for k, v in dotenv_values(path).items() if v} if path else {}


def read_key(names) -> str:
    """First non-empty value among ``names``, from the environment then `.env`."""
    env_file = None
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
        if env_file is None:
            env_file = _dotenv_values()
        value = (env_file.get(name) or "").strip()
        if value:
            return value
    return ""


def resolve_spec(handle: str) -> ModelSpec:
    """A registry handle, or ``"<provider>:<model>"`` for anything else."""
    handle = (handle or "").strip()
    if handle in REGISTRY:
        return REGISTRY[handle]
    if ":" in handle:
        provider, model = handle.split(":", 1)
        defaults = PROVIDER_DEFAULTS.get(provider)
        if defaults:
            base_url = defaults["base_url"] or os.getenv("LLM_BASE_URL", "")
            if not base_url:
                raise ValueError(
                    f"Provider {provider!r} has no base URL; set LLM_BASE_URL.")
            return ModelSpec(handle=handle, provider=provider, model=model,
                             base_url=base_url, api_key_env=defaults["api_key_env"],
                             family=defaults["family"])
    raise ValueError(
        f"Unknown model {handle!r}. Known handles: {sorted(REGISTRY)}; "
        f"or use '<provider>:<model>' with provider in {sorted(PROVIDER_DEFAULTS)}."
    )


@dataclass
class Completion:
    """One reply, plus the provenance a report has to be able to print."""

    text: str
    model_requested: str
    model_served: str
    usage: Dict[str, Any] = field(default_factory=dict)
    transport_retries: int = 0

    def to_dict(self) -> dict:
        return {"model_requested": self.model_requested,
                "model_served": self.model_served,
                "usage": self.usage,
                "transport_retries": self.transport_retries}


class JudgeModel:
    """One model, callable by any judge. Provider-independent by construction."""

    def __init__(self, spec: ModelSpec, *, temperature: float = 0.0,
                 max_tokens: int = 512, retries: int = 2, timeout: Optional[int] = None,
                 backoff: float = 0.5, sleep=time.sleep):
        self.spec = spec
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.retries = int(retries)
        # Seconds per request; LLM_TIMEOUT raises it when many replays share one server.
        self.timeout = int(timeout if timeout is not None else os.getenv("LLM_TIMEOUT", "180"))
        self._backoff = backoff
        self._sleep = sleep
        self.transport_retries = 0
        self.calls = 0
        self.model_served = ""
        # Optional callback ``(model_requested, response_json)`` after each call, so a
        # caller can keep its own usage/cost ledger.
        self.on_usage = None

    @classmethod
    def resolve(cls, handle: str, **kwargs) -> "JudgeModel":
        return cls(resolve_spec(handle), **kwargs)

    @property
    def identity(self) -> str:
        """What a report should print: requested id, plus who actually answered."""
        if self.model_served and self.model_served != self.spec.model:
            return f"{self.spec.identity} (served: {self.model_served})"
        return self.spec.identity

    @property
    def family(self) -> str:
        return self.spec.family or self.spec.provider

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = read_key(self.spec.api_key_env)
        if key and key != "EMPTY":
            headers["Authorization"] = f"Bearer {key}"
        elif self.spec.provider not in ("local",):
            raise JudgeModelError(
                f"No API key for {self.spec.identity}: set one of "
                f"{list(self.spec.api_key_env)} in the environment or `.env`.")
        return headers

    def complete(self, *, system: str, user: str,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None) -> Completion:
        """One chat completion. Raises rather than returning unusable text."""
        budget = self.max_tokens if max_tokens is None else int(max_tokens)
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        if self.spec.provider in ("openai", "openrouter"):
            # GPT-5-family models reject temperature=0 and the `max_tokens`
            # name, and reasoning tokens count against the completion budget.
            payload["max_completion_tokens"] = max(budget, 2000)
        else:
            payload["temperature"] = (self.temperature if temperature is None
                                      else float(temperature))
            payload["max_tokens"] = budget
        headers = self._headers()
        url = f"{self.spec.base_url.rstrip('/')}/v1/chat/completions"

        retried = 0
        for attempt in range(self.retries + 1):
            try:
                response = _SESSION.post(url, headers=headers, json=payload,
                                         timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except Exception as exc:
                if attempt == self.retries:
                    raise JudgeTransportError(f"{type(exc).__name__}: {exc}") from exc
                retried += 1
                self._sleep(self._backoff * (2 ** attempt))
                continue
            break

        self.calls += 1
        self.transport_retries += retried
        try:
            raw = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeModelError(f"malformed completion body: {exc}") from exc
        self.model_served = str(data.get("model") or self.spec.model)
        if self.on_usage is not None:
            self.on_usage(self.spec.model, data)

        text = strip_reasoning(raw).strip()
        if THINK_OPEN.search(text):
            raise JudgeModelError(
                f"response was cut off inside a reasoning block "
                f"(max_tokens={budget}); raise the budget")
        if not text:
            raise JudgeModelError("model returned an empty response")
        return Completion(text=text, model_requested=self.spec.model,
                          model_served=self.model_served,
                          usage=dict(data.get("usage") or {}),
                          transport_retries=retried)

    def complete_json(self, *, system: str, user: str,
                      max_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> tuple:
        """``(parsed_or_None, completion)``. ``None`` means unusable output."""
        completion = self.complete(system=system, user=user,
                                   max_tokens=max_tokens, temperature=temperature)
        return extract_json(completion.text), completion

    def preflight(self) -> tuple:
        """One cheap call, so a dead endpoint fails before a sample is paid for."""
        try:
            completion = self.complete(system="Reply with one word.",
                                       user="Reply with one word: ready",
                                       max_tokens=8)
        except (JudgeTransportError, JudgeModelError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, completion.text[:80]

    def provenance(self) -> dict:
        return {
            "handle": self.spec.handle,
            "provider": self.spec.provider,
            "family": self.family,
            "model_requested": self.spec.model,
            "model_served": self.model_served,
            "base_url": self.spec.base_url,
            "temperature": (None if self.spec.provider == "openai"
                            else self.temperature),
            "max_tokens": self.max_tokens,
            "calls": self.calls,
            "transport_retries": self.transport_retries,
        }


def extract_json(raw: str) -> Optional[dict]:
    """Pull the first JSON object out of a model response. ``None`` if unusable."""
    if not raw:
        return None
    text = strip_reasoning(raw)
    text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start:idx + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def known_handles() -> List[str]:
    return sorted(REGISTRY)


# ── three-pass majority vote over a stochastic classifier ──
# GPT-5-family models cannot run at temperature 0, so the same question is asked several times and the label
# that a strict majority of the passes agree on is kept.

NO_MAJORITY = "no_majority"
FAILED = "failed"


def majority(labels: Sequence[Optional[str]], *, passes: int = 3) -> Tuple[str, int]:
    """``(label, votes)`` for the label with a strict majority of ``passes``.

    Passes that errored (``None``) count as no vote. Returns ``FAILED`` when no
    pass produced a label and ``NO_MAJORITY`` when none has more than half of
    the passes asked for.
    """
    votes = Counter(l for l in labels if l)
    if not votes:
        return FAILED, 0
    label, n = votes.most_common(1)[0]
    if n * 2 > passes:
        return label, n
    return NO_MAJORITY, n
