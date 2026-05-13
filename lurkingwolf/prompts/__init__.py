"""Prompt loading and templates for LurkingWolf agents."""

import json
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


def load_prompts() -> dict:
    with open(_PROMPT_DIR / "prompt.json", encoding="utf-8") as f:
        return json.load(f)
