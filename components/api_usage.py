"""Lightweight API usage accounting.

One record per (experiment, run, provider, model, purpose), appended to a JSONL
ledger. Deliberately not a framework: a dict of counters, an append, and a
reader.

    usage = UsageLedger("natural_politeness", run_id="pilot_01")
    usage.record("deepseek", "deepseek-chat", "game_agent", completion.usage)
    ...
    usage.write()          # -> <experiment>/data/api_usage.jsonl

Token counts come from the provider's own `usage` block where the endpoint
returns one, which is what `JudgeModel` already receives. Where a caller can
only supply locally counted tokens, it says so via ``counted="local"`` and the
record carries that, because a tiktoken estimate and a provider count are not
the same number and should never be silently added together.

Cost is only ever computed when a price is known for that model. Otherwise
``estimated_cost_usd`` is null rather than zero -- an unknown price is not a
free call. Every estimate records the price used and when that price was read,
so a later change in provider pricing cannot silently rewrite history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# USD per 1M tokens. Add a model here to start costing it; leave it out and the
# ledger records tokens with a null cost.
PRICING: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {"input": 0.28, "output": 0.42, "cached_input": 0.028,
                      "as_of": "2026-09-22", "source": "api-docs.deepseek.com/quick_start/pricing"},
}

PURPOSES = ("game_agent", "surface_generation", "rewrite",
            "politeness_classifier", "semantic_validator", "inverse_parser",
            "listener_probe", "other")


@dataclass
class UsageRecord:
    experiment: str
    run_id: str
    provider: str
    model: str
    purpose: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    counted: str = "provider"          # "provider" | "local"
    provider_reported_cost_usd: Optional[float] = None

    def add(self, usage: Optional[Dict[str, Any]] = None, *, calls: int = 1) -> None:
        usage = usage or {}
        self.calls += calls
        self.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_tokens += int(details.get("cached_tokens")
                                  or usage.get("prompt_cache_hit_tokens") or 0)
        if usage.get("cost") is not None:
            self.provider_reported_cost_usd = (
                (self.provider_reported_cost_usd or 0.0) + float(usage["cost"]))

    def to_dict(self) -> dict:
        price = PRICING.get(self.model)
        estimated = None
        if price:
            billed_input = max(self.input_tokens - self.cached_tokens, 0)
            estimated = round(
                billed_input / 1e6 * price["input"]
                + self.cached_tokens / 1e6 * price.get("cached_input", price["input"])
                + self.output_tokens / 1e6 * price["output"], 6)
        return {
            "experiment": self.experiment, "run_id": self.run_id,
            "provider": self.provider, "model": self.model, "purpose": self.purpose,
            "calls": self.calls, "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens, "cached_tokens": self.cached_tokens,
            "token_counts_from": self.counted,
            "provider_reported_cost_usd": self.provider_reported_cost_usd,
            "estimated_cost_usd": estimated,
            "pricing_usd_per_1m": price and {k: v for k, v in price.items()
                                             if k in ("input", "output", "cached_input")},
            "pricing_as_of": price and price["as_of"],
            "pricing_source": price and price["source"],
            "recorded_on": date.today().isoformat(),
        }


class UsageLedger:
    """Accumulates usage for one run and appends it to the experiment's ledger."""

    def __init__(self, experiment: str, run_id: str, path: Optional[Path] = None):
        self.experiment, self.run_id = experiment, run_id
        self.path = Path(path) if path else Path(
            "experiments") / experiment / "data" / "api_usage.jsonl"
        self._records: Dict[tuple, UsageRecord] = {}

    def record(self, provider: str, model: str, purpose: str,
               usage: Optional[Dict[str, Any]] = None, *, calls: int = 1,
               counted: str = "provider") -> None:
        key = (provider, model, purpose)
        rec = self._records.get(key)
        if rec is None:
            rec = self._records[key] = UsageRecord(
                self.experiment, self.run_id, provider, model, purpose, counted=counted)
        rec.add(usage, calls=calls)

    def totals(self) -> dict:
        rows = [r.to_dict() for r in self._records.values()]
        known = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
        return {
            "calls": sum(r["calls"] for r in rows),
            "input_tokens": sum(r["input_tokens"] for r in rows),
            "output_tokens": sum(r["output_tokens"] for r in rows),
            "cached_tokens": sum(r["cached_tokens"] for r in rows),
            "estimated_cost_usd": round(sum(known), 4) if known else None,
            "models_without_pricing": sorted({r["model"] for r in rows
                                              if r["estimated_cost_usd"] is None}),
        }

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            for rec in self._records.values():
                handle.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return self.path


def read_ledger(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: Iterable[dict]) -> Dict[str, Any]:
    """Totals overall and by purpose, keeping unpriced models visible."""
    rows = list(records)
    by_purpose: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        acc = by_purpose.setdefault(r["purpose"], {
            "calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
            "priced": True})
        acc["calls"] += r["calls"]
        acc["input_tokens"] += r["input_tokens"]
        acc["output_tokens"] += r["output_tokens"]
        if r["estimated_cost_usd"] is None:
            acc["priced"] = False
        else:
            acc["estimated_cost_usd"] += r["estimated_cost_usd"]
    priced = [r["estimated_cost_usd"] for r in rows if r["estimated_cost_usd"] is not None]
    return {
        "runs": sorted({r["run_id"] for r in rows}),
        "calls": sum(r["calls"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "estimated_cost_usd": round(sum(priced), 4) if priced else None,
        "unpriced_models": sorted({r["model"] for r in rows
                                   if r["estimated_cost_usd"] is None}),
        "by_purpose": by_purpose,
    }
