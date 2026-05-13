"""
token_tracker.py -- Token counting and cost estimation for LLM game runs.

Uses tiktoken for tokenization. Falls back to cl100k_base for models
without a native tiktoken encoding (Qwen, etc.).
"""

import tiktoken

# ── Pricing table (USD per 1M tokens) ──────────────────────────
# Self-hosted models have zero API cost. Add entries here if you
# use a pay-per-token API.

PRICING: dict[str, dict[str, float]] = {
    "qwen3.5-9b": {"input": 0.0, "output": 0.0},
    "Qwen/Qwen3.5-9B": {"input": 0.0, "output": 0.0},
    "qwen3.5-27b": {"input": 0.0, "output": 0.0},
    "Qwen/Qwen3.5-27B": {"input": 0.0, "output": 0.0},
}

# ── Tiktoken encoding selection ────────────────────────────────
# Map model prefixes to tiktoken encoding names.
_ENCODING_MAP: dict[str, str] = {}

_DEFAULT_ENCODING = "cl100k_base"

_encoding_cache: dict[str, tiktoken.Encoding] = {}


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for the given model string."""
    # Try tiktoken's built-in model lookup first
    if model in _encoding_cache:
        return _encoding_cache[model]

    # Match by prefix
    enc_name = _DEFAULT_ENCODING
    for prefix, name in _ENCODING_MAP.items():
        if prefix in model.lower():
            enc_name = name
            break

    if enc_name not in _encoding_cache:
        _encoding_cache[enc_name] = tiktoken.get_encoding(enc_name)
    _encoding_cache[model] = _encoding_cache[enc_name]
    return _encoding_cache[model]


def count_tokens(text: str, model: str) -> int:
    """Count the number of tokens in *text* for the given model."""
    enc = _get_encoding(model)
    return len(enc.encode(text))


# ── Per-game tracker ───────────────────────────────────────────

class TokenBudgetExceeded(Exception):
    """Raised when a game exceeds its configured token budget."""
    pass


class TokenTracker:
    """Accumulates token counts and computes estimated costs for a game."""

    def __init__(self, model: str, max_tokens: int | None = None):
        self.model = model
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        self.max_tokens = max_tokens  # None = no limit

    def record(self, input_text: str, output_text: str) -> dict:
        """Record one LLM call. Returns a dict with token counts for this call.

        Raises TokenBudgetExceeded if the cumulative total exceeds max_tokens.
        """
        input_tokens = count_tokens(input_text, self.model)
        output_tokens = count_tokens(output_text, self.model)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1
        if self.max_tokens is not None and self.total_tokens > self.max_tokens:
            raise TokenBudgetExceeded(
                f"Token budget exceeded: {self.total_tokens:,} > {self.max_tokens:,} "
                f"(after {self.call_count} calls)"
            )
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def estimated_cost(self) -> float:
        """Return estimated cost in USD based on the pricing table."""
        prices = PRICING.get(self.model)
        if prices is None:
            return 0.0
        input_cost = (self.total_input_tokens / 1_000_000) * prices["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * prices["output"]
        return input_cost + output_cost

    def summary(self) -> dict:
        """Return a summary dict suitable for JSON serialisation."""
        cost = self.estimated_cost()
        prices = PRICING.get(self.model)
        return {
            "model": self.model,
            "calls": self.call_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(cost, 6),
            "pricing_per_1m": prices if prices else "unknown",
        }

    def print_summary(self) -> None:
        """Print a human-readable cost summary to stdout."""
        s = self.summary()
        print(f"\n{'-'*50}")
        print(f"  Token Usage & Cost Estimate")
        print(f"{'-'*50}")
        print(f"  Model:          {s['model']}")
        print(f"  API calls:      {s['calls']}")
        print(f"  Input tokens:   {s['input_tokens']:,}")
        print(f"  Output tokens:  {s['output_tokens']:,}")
        print(f"  Total tokens:   {s['total_tokens']:,}")
        if s["pricing_per_1m"] == "unknown":
            print(f"  Estimated cost: (pricing not available for this model)")
        elif s["estimated_cost_usd"] == 0.0 and isinstance(s["pricing_per_1m"], dict) \
                and s["pricing_per_1m"]["input"] == 0.0:
            print(f"  Estimated cost: $0.00 (self-hosted / no API cost)")
        else:
            print(f"  Estimated cost: ${s['estimated_cost_usd']:.4f}")
        print(f"{'-'*50}")
