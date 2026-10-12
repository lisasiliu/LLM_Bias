"""Blind pairwise proficiency probe: Family C listener-side perception (PLAN 9).

A listener model reads two matched passages, never told which condition either
came from, and says which speaker's English shows the more limited range.
Each pair is run in both presentation orders by the caller; the probe itself
is stateless.  Listener models are perception INSTRUMENTS, kept separate from
the rewrite generator and the semantic judges.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .judge_model import JudgeModel, JudgeModelError, JudgeTransportError

PROMPT_VERSION = "range_pairwise_v2"

_SYSTEM = """\
You are a blind rater in a language study. You will read two short passages, A and B,
each spoken by a different player during a Werewolf (Mafia) game.

Judge LANGUAGE only, not content. Ignore who is accused, whether the reasoning is
good, whether the claims are true, and how polite or forceful the speaker is.

Question: which passage shows a MORE LIMITED range of vocabulary and sentence
structure (more basic and repetitive
word choice; simpler, more uniform sentence patterns)?

Assess vocabulary and sentence range; grammatical correctness is assessed separately.
Do not infer the person's overall English ability from this task. Do not prefer a passage
because it is shorter or longer. If they seem equal, still choose the one that is
even slightly more limited.

Answer with JSON only, exactly:
{"more_limited": "A" | "B",
 "confidence": "low" | "medium" | "high",
 "reason": "<=20 words about the language, not the content"}
"""


def system_prompt() -> str:
    return _SYSTEM


def probe_prompt_sha256() -> str:
    return hashlib.sha256(_SYSTEM.encode("utf-8")).hexdigest()


def build_user_prompt(text_a: str, text_b: str) -> str:
    return f"Passage A:\n{text_a.strip()}\n\nPassage B:\n{text_b.strip()}\n\nJSON:"


CONFIDENCE_SCORE = {"low": 1, "medium": 2, "high": 3}


def parse_choice(value) -> Optional[str]:
    """``"A"`` / ``"B"`` from ``"A"``, ``"passage B"``, ``"passage_a"``, ``"(B)"``; else ``None``."""
    text = str(value or "").strip().upper()
    text = text.replace("PASSAGE", "").replace("SPEAKER", "")
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) == 1 and letters[0] in ("A", "B"):
        return letters[0]
    return None


@dataclass
class ProbeResult:
    ok: bool
    choice: Optional[str] = None          # "A" | "B"
    confidence: str = ""
    reason: str = ""
    error: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "choice": self.choice, "confidence": self.confidence,
                "confidence_score": CONFIDENCE_SCORE.get(self.confidence),
                "reason": self.reason, "error": self.error,
                "provenance": dict(self.provenance)}


class BlindProficiencyProbe:
    """Ask one listener model which of two passages is more limited."""

    name = "blind_proficiency_probe"

    # 400 leaves room for a reasoning model's hidden trace; the visible answer is tiny.
    def __init__(self, model: JudgeModel, max_tokens: int = 400):
        self.model = model
        self.max_tokens = int(max_tokens)

    def compare(self, text_a: str, text_b: str) -> ProbeResult:
        if not (text_a or "").strip() or not (text_b or "").strip():
            return ProbeResult(ok=False, error="empty_passage")
        try:
            parsed, completion = self.model.complete_json(
                system=_SYSTEM, user=build_user_prompt(text_a, text_b),
                max_tokens=self.max_tokens)
        except (JudgeTransportError, JudgeModelError) as exc:
            return ProbeResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        provenance = completion.to_dict()
        provenance["prompt_version"] = PROMPT_VERSION
        if parsed is None:
            return ProbeResult(ok=False, error="unparseable_json", provenance=provenance)
        choice = parse_choice(parsed.get("more_limited"))
        if choice is None:
            return ProbeResult(ok=False, error=f"unknown_choice:{str(parsed.get('more_limited'))[:20]}",
                               provenance=provenance)
        return ProbeResult(ok=True, choice=choice,
                           confidence=str(parsed.get("confidence", "")).strip().lower()[:10],
                           reason=str(parsed.get("reason", ""))[:200], provenance=provenance)
