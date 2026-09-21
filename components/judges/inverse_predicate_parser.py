"""Recovers the structured predicate from an utterance, never having seen it.

Shown only the public utterance, the prior public record and the allowed
grammar, the parser reconstructs the predicate from scratch. Preservation then
becomes a comparison of two structured objects rather than a model's opinion,
and the judge itself can be scored against ground truth.

Exact recovery is strong evidence when it succeeds; a legitimate indirect
realization can make it impossible, so a mismatch is a discrepancy signal
rather than a rejection. `compare` scores a result against a true predicate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .judge_model import JudgeModel, JudgeModelError, JudgeTransportError
from components.rewriters.politeness_rewriter import predicate_entry
from lurkingwolf.utils.speech_nl import parse_predicate

PARSEABLE_PREDICATES: tuple = ("estimate", "question", "agree", "disagree",
                               "recommend_action")

GIVEN_SLOTS: frozenset = frozenset({"Phase", "Player", "Speaker"})

DIAGNOSTIC_SLOTS: frozenset = frozenset({"RefID"})


def signature_slots(name: str) -> List[str]:
    """Slot NAMES for a predicate, read from prompt.json's signature."""
    sig = predicate_entry(name).get("sig", "")
    return [s.strip() for s in sig.strip("()").split(",") if s.strip()]


def _enum_from_desc(name: str, label: str) -> List[str]:
    """Pull ``Label: a|b|c`` out of a predicate description in prompt.json."""
    desc = predicate_entry(name).get("desc", "")
    match = re.search(rf"{label}:\s*([a-z_|]+)", desc)
    return match.group(1).split("|") if match else []


def grammar_block() -> str:
    """The allowed predicate grammar, rendered for the prompt."""
    lines = []
    for name in PARSEABLE_PREDICATES:
        entry = predicate_entry(name)
        lines.append(f"- {name}{entry.get('sig', '')}: {entry.get('desc', '')}")
    return "\n".join(lines)


_PARSER_SYSTEM = """\
You are a semantic parser for a Werewolf (Mafia) game. You convert ONE public
utterance into the single structured predicate that it realizes.

ALLOWED GRAMMAR — the utterance realizes exactly one of these five predicates:
%(grammar)s

Slot rules:
- Phase and the speaker slot (Player / Speaker) are given to you in the input.
  Copy them exactly.
- Player ids are written P followed by a number, e.g. P7.
- RefID is the id of the ONE prior statement the utterance responds to. It must
  be copied from the listed prior statements (e.g. s11). Choose by what the
  utterance actually responds to, not merely by topic overlap.
- Role must be one of: %(roles)s
- Action must be one of: %(actions)s

How to choose the predicate:
- estimate: the speaker asserts what role some target player has.
- question: the speaker challenges a target about a specific earlier statement.
- agree: the speaker publicly supports a specific earlier statement.
- disagree: the speaker publicly opposes a specific earlier statement.
- recommend_action: the speaker recommends that some actor take a game action
  toward some target.

Read the utterance on its own terms. Recover what the speaker is DOING, not what
would be reasonable for them to do. If the utterance is indirect, hedged, polite
or merely hinted, still recover the predicate it is driving at, and say so with
"recoverable": true. Use "recoverable": false only when the underlying predicate
genuinely cannot be determined, and list the slots you could not fill in
"unresolved".

Answer with JSON only, no prose, exactly:
{"predicate_name": "<one of the five, or null>",
 "arguments": {"<SlotName>": "<value>", ...},
 "recoverable": true|false,
 "unresolved": ["<SlotName>", ...],
 "notes": "<=20 words"}

"arguments" must use exactly the slot names from the signature of the predicate
you chose, and must contain every slot in that signature.
"""


def build_system_prompt() -> str:
    roles = "|".join(_enum_from_desc("estimate", "Role"))
    actions = "|".join(_enum_from_desc("recommend_action", "Action"))
    return _PARSER_SYSTEM % {"grammar": grammar_block(), "roles": roles,
                             "actions": actions}


_PHASE_TOKEN = re.compile(r"^[DN]\d+$", re.IGNORECASE)


def context_phase(source) -> str:
    """The game phase token (``D1``, ``N2``) that the Phase slot is written in."""
    first = (getattr(source, "predicate_args", None) or [""])[0].strip()
    return first if _PHASE_TOKEN.match(first) else str(getattr(source, "phase", "") or "")


def build_user_prompt(utterance: str, *, phase: str, speaker_id: str,
                      public_history: Optional[List[str]] = None) -> str:
    """The utterance plus ONLY the context needed to resolve its arguments."""
    history = [h for h in (public_history or []) if str(h).strip()]
    block = "\n".join(f"  {h}" for h in history) or "  (none)"
    return (
        f"Phase: {phase}\n"
        f"Speaker: {speaker_id}\n"
        f"Prior public statements (use these to resolve RefID):\n{block}\n\n"
        f"Utterance: {utterance}\n\n"
        "JSON:"
    )


_PLAYER_RE = re.compile(r"^\s*(?:player\s*#?\s*|p\s*|#\s*)?(\d+)\s*$", re.IGNORECASE)
_SPEECH_RE = re.compile(r"^\s*(?:s|speech\s*|statement\s*)?(\d+)\s*$", re.IGNORECASE)


def normalize_slot(slot: str, value: Any) -> str:
    """Canonical form for one slot value, so formatting is not scored as error."""
    text = ("" if value is None else str(value)).strip()
    if not text:
        return ""
    if slot in ("Player", "Speaker", "Target", "Actor", "T1", "T2"):
        match = _PLAYER_RE.match(text)
        return f"P{int(match.group(1))}" if match else text.upper()
    if slot == "RefID":
        match = _SPEECH_RE.match(text)
        return f"s{int(match.group(1))}" if match else text.lower()
    if slot == "Phase":
        return text.upper().replace(" ", "")
    return text.strip().lower().replace(" ", "_").replace("-", "_")


def truth_slots(predicate_text: str) -> Dict[str, str]:
    """``{slot: normalized value}`` for a ground-truth predicate string."""
    parsed = parse_predicate(predicate_text)
    name = str(parsed.get("name", ""))
    args = list(parsed.get("args", []))
    slots = signature_slots(name)
    return {slot: normalize_slot(slot, arg) for slot, arg in zip(slots, args)}


@dataclass
class ParseResult:
    """One parser attempt. ``ok=False`` means no usable structured answer."""

    ok: bool
    predicate_name: Optional[str] = None
    arguments: Dict[str, str] = field(default_factory=dict)
    recoverable: Optional[bool] = None
    unresolved: List[str] = field(default_factory=list)
    notes: str = ""
    error: str = ""
    raw: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def predicate_text(self) -> str:
        """The recovered predicate rendered back into LURK's surface syntax."""
        if not self.predicate_name:
            return ""
        slots = signature_slots(self.predicate_name)
        args = [self.arguments.get(s, "?") for s in slots]
        return f"{self.predicate_name}({', '.join(args)})"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "predicate_name": self.predicate_name,
            "predicate_text": self.predicate_text,
            "arguments": dict(self.arguments),
            "recoverable": self.recoverable,
            "unresolved": list(self.unresolved),
            "notes": self.notes,
            "error": self.error,
            "provenance": dict(self.provenance),
        }


def compare(result: ParseResult, truth_predicate_text: str) -> dict:
    """Structured comparison of a recovered predicate against ground truth."""
    truth = truth_slots(truth_predicate_text)
    truth_name = str(parse_predicate(truth_predicate_text).get("name", ""))
    slots = signature_slots(truth_name)

    name_match = bool(result.ok and result.predicate_name == truth_name)
    per_slot: Dict[str, dict] = {}
    if name_match:
        for slot in slots:
            got = normalize_slot(slot, result.arguments.get(slot))
            per_slot[slot] = {"expected": truth.get(slot, ""), "got": got,
                              "match": got == truth.get(slot, ""),
                              "given": slot in GIVEN_SLOTS}
    else:
        for slot in slots:
            per_slot[slot] = {"expected": truth.get(slot, ""), "got": "",
                              "match": False, "given": slot in GIVEN_SLOTS}

    content = [s for s in slots if s not in GIVEN_SLOTS]
    gated = [s for s in content if s not in DIAGNOSTIC_SLOTS]
    return {
        "truth_predicate": truth_predicate_text,
        "truth_name": truth_name,
        "recovered_predicate": result.predicate_text,
        "recovered_name": result.predicate_name,
        "parsed_ok": result.ok,
        "recoverable": result.recoverable,
        "name_match": name_match,
        "structural_match": name_match and all(per_slot[s]["match"] for s in gated),
        "gated_slots": gated,
        "content_match": name_match and all(per_slot[s]["match"] for s in content),
        "exact_match": name_match and all(v["match"] for v in per_slot.values()),
        "refid_match": (per_slot["RefID"]["match"] if "RefID" in per_slot else None),
        "slots": per_slot,
        "mismatched_slots": sorted(s for s, v in per_slot.items() if not v["match"]),
        "error": result.error,
    }


class InversePredicateParser:
    """Recover a structured predicate from a public utterance."""

    name = "inverse_predicate_parser"

    def __init__(self, model: JudgeModel, max_tokens: int = 400):
        self.model = model
        self.max_tokens = int(max_tokens)

    def parse(self, utterance: str, *, phase: str, speaker_id: str,
              public_history: Optional[List[str]] = None) -> ParseResult:
        if not (utterance or "").strip():
            return ParseResult(ok=False, error="empty_utterance")

        system = build_system_prompt()
        user = build_user_prompt(utterance, phase=phase, speaker_id=speaker_id,
                                 public_history=public_history)
        try:
            parsed, completion = self.model.complete_json(
                system=system, user=user, max_tokens=self.max_tokens)
        except (JudgeTransportError, JudgeModelError) as exc:
            return ParseResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        provenance = completion.to_dict()
        if parsed is None:
            return ParseResult(ok=False, error="unparseable_json",
                               raw=completion.text[:400], provenance=provenance)

        name = parsed.get("predicate_name")
        name = str(name).strip() if name is not None else None
        if name in ("", "null", "None"):
            name = None
        if name is not None and name not in PARSEABLE_PREDICATES:
            return ParseResult(ok=False, error=f"predicate_outside_grammar:{name}",
                               raw=completion.text[:400], provenance=provenance)

        raw_args = parsed.get("arguments")
        arguments: Dict[str, str] = {}
        if isinstance(raw_args, dict) and name:
            for slot in signature_slots(name):
                if slot in raw_args:
                    arguments[slot] = normalize_slot(slot, raw_args[slot])

        unresolved = parsed.get("unresolved")
        unresolved = [str(u) for u in unresolved] if isinstance(unresolved, list) else []
        recoverable = parsed.get("recoverable")
        recoverable = recoverable if isinstance(recoverable, bool) else None

        return ParseResult(
            ok=True,
            predicate_name=name,
            arguments=arguments,
            recoverable=recoverable,
            unresolved=unresolved,
            notes=str(parsed.get("notes", ""))[:200],
            raw=completion.text[:400],
            provenance=provenance,
        )
