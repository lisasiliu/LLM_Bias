"""
gen_config.py -- Shared generation settings for NL output.

Used by both pred_to_nl.py (standalone test) and agent.py (in-game NL generation)
to ensure identical behavior. Thinking mode is OFF by default — temp=0 Qwen3.5
tends to loop on short NL prompts, blowing the context window with garbage.
Set ENABLE_THINKING=1 to opt in when you actually want traces.
"""

import os

NL_SEED = 101
TEMPERATURE = 0
TOP_P = 1
MAX_TOKENS_NL = 10000           # with thinking enabled (reason + answer)
MAX_TOKENS_NL_NOTHINK = 1000    # without thinking (just the answer)
MAX_TOKENS_STAGE1 = 10000       # predicate generation with thinking
MAX_TOKENS_STAGE1_NOTHINK = 1000 # predicate generation without thinking


def thinking_disabled() -> bool:
    """Return True unless ENABLE_THINKING env var is set. Off by default."""
    return os.environ.get("ENABLE_THINKING", "").strip().lower() not in ("1", "true", "yes")


def is_thinking_model(model: str) -> bool:
    """Return True for models that ship with a thinking/reasoning mode."""
    name = model.lower()
    # Qwen3/3.5 and DeepSeek-R1 have enable_thinking; Qwen2.5 does not
    return "qwen3" in name or "deepseek-r1" in name


def thinking_extra_body(model: str) -> dict:
    """Return extra_body kwargs to enable/disable thinking for thinking models.

    Returns a dict with 'extra_body' key if applicable, empty dict otherwise.
    Can be unpacked into any chat.completions.create() call.
    """
    if is_thinking_model(model):
        enable = not thinking_disabled()
        return {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable},
            }
        }
    return {}


def nl_max_tokens(model: str) -> int:
    """Max tokens for Stage 2 NL generation."""
    if thinking_disabled() or not is_thinking_model(model):
        return MAX_TOKENS_NL_NOTHINK
    return MAX_TOKENS_NL


def stage1_max_tokens(model: str, default: int) -> int:
    """Max tokens for Stage 1 predicate generation.

    Thinking on  -> bump to MAX_TOKENS_STAGE1 (10000).
    Thinking off -> bump to MAX_TOKENS_STAGE1_NOTHINK (1000) if agent default is lower.
    """
    if thinking_disabled() or not is_thinking_model(model):
        return max(default, MAX_TOKENS_STAGE1_NOTHINK)
    return max(default, MAX_TOKENS_STAGE1)


def deterministic_kwargs(model: str) -> dict:
    """Build generation kwargs for deterministic NL output.

    With thinking enabled: high max_tokens so model can reason + answer.
    With thinking disabled: low max_tokens for efficiency.
    """
    kw = dict(
        model=model,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        seed=NL_SEED,
        max_tokens=nl_max_tokens(model),
    )
    kw.update(thinking_extra_body(model))
    return kw
