"""Shared parsing and cleanup helpers for game speech."""

import re
from typing import Dict


def parse_predicate(action: str) -> Dict[str, object]:
    action = (action or "").strip()
    match = re.match(r"(\w+)\((.*)\)$", action)
    if not match:
        return {"name": action, "args": [], "raw": action}
    args = []
    depth = 0
    current = []
    for ch in match.group(2):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return {"name": match.group(1), "args": args, "raw": action}


def strip_think_tags(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def clean_natural_text(text: str) -> str:
    cleaned = strip_think_tags(text)
    cleaned = re.sub(r"^\s*\[.*?\]\s*(?:->)?\s*", "", cleaned).strip()
    cleaned = re.sub(r"^\s*\w+\([^)]*\)\s*[:\-]?\s*", "", cleaned).strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
        inner = cleaned[1:-1].strip()
        if inner:
            cleaned = inner
    return cleaned


def strip_reasoning(text: str) -> str:
    """Return only what a reasoning model emitted AFTER its reasoning.

    ``strip_think_tags`` removes a CLOSED ``<think>...</think>`` pair, which
    assumes the opening tag is present in the response.  Many chat templates
    pre-open the tag for the model, so the reply starts mid-reasoning and the
    ONLY tag that ever appears is the closing one.  The reasoning then survives
    every cleanup and is treated as the answer.

    Rule: a closed reasoning block ends at its LAST ``</think>`` and everything
    before that is discarded reasoning -- including drafts the model went on to
    revise.  If an UNTERMINATED ``<think>`` remains afterwards the response was
    cut off mid-reasoning; the tag is left in place so the caller can recognise
    the truncation rather than silently receiving a fragment.

    Additive helper: nothing in the game engine calls it, so game behaviour is
    unchanged.
    """
    if not isinstance(text, str):
        return ""
    closes = list(re.finditer(r"</think\s*>", text, flags=re.IGNORECASE))
    if closes:
        text = text[closes[-1].end():]
    return text.strip()
