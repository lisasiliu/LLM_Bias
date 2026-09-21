"""Deterministic preservation checks driven by the structured predicate.

A rewrite may not LOSE a slot the original utterance actually realized on the
surface. Slots the original never spelled out (a bare speech id, usually) cannot
be checked here and are left to the semantic judge, so this never invents a
requirement the untreated text did not itself satisfy.

``commitment_may_be_indirect`` relaxes the added-hedge and slot rules for a
strategy where indirectness is the treatment (politeness ``off_record``).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from components.helpers import CheckResult, SourceRecord


_PLAYER_RE = re.compile(r"^P\d+$", re.IGNORECASE)
_SPEECH_RE = re.compile(r"^s\d+$", re.IGNORECASE)
_PHASE_RE = re.compile(r"^[DN]\d+$", re.IGNORECASE)

ROLE_SURFACE: Dict[str, List[str]] = {
    "werewolf": ["werewolf", "werewolves", "wolf", "wolves"],
    "white_wolf_king": ["white wolf king", "white-wolf king", "wolf king", "wwk"],
    "villager": ["villager", "villagers", "village", "good", "townsfolk"],
    "seer": ["seer", "prophet"],
    "guard": ["guard", "bodyguard", "protector"],
    "witch": ["witch"],
    "hunter": ["hunter"],
    "wolf_camp": ["wolf camp", "wolf side", "wolf team", "werewolf", "werewolves", "wolf", "wolves"],
    "villager_camp": ["villager camp", "village side", "village team", "villager", "villagers", "good"],
}

ACTION_SURFACE: Dict[str, List[str]] = {
    "vote_sheriff": ["vote for sheriff", "sheriff vote", "vote", "badge", "sheriff"],
    "vote_execute": ["vote out", "vote to execute", "vote for", "execute", "exile", "vote", "lynch"],
    "divine": ["divine", "check", "inspect", "verify", "look at"],
    "protect": ["protect", "guard", "shield", "save"],
    "use_antidote": ["antidote", "save", "heal", "revive"],
    "use_poison": ["poison"],
    "shoot": ["shoot", "shot", "gun", "take down"],
    "pass_badge": ["pass the badge", "pass badge", "hand the badge", "give the badge", "badge"],
    "tear_badge": ["tear the badge", "tear badge", "destroy the badge", "rip the badge", "badge"],
    "question": ["question", "ask", "press", "challenge"],
    "pressure": ["pressure", "press", "push"],
    "run_sheriff": ["run for sheriff", "run", "campaign", "sheriff"],
    "withdraw_sheriff": ["withdraw", "drop out", "step down", "pull out"],
    "hide": ["hide", "stay hidden", "keep quiet", "stay quiet"],
    "reveal": ["reveal", "come out", "claim", "show yourself", "speak up"],
    "hold": ["hold", "wait", "hold off", "stand by"],
    "act_now": ["act now", "act", "move now", "do it now"],
}

HEDGE_PATTERNS: List[str] = [
    r"\bmaybe\b", r"\bmight\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b",
    r"\bpotentially\b", r"\bseems?\b", r"\bseemed\b", r"\bappears?\b",
    r"\bi think\b", r"\bi guess\b", r"\bi feel\b", r"\bi suspect\b",
    r"\bi'?m not sure\b", r"\bnot certain\b", r"\bunsure\b", r"\bcould be\b",
    r"\bmay be\b", r"\bkind of\b", r"\bsort of\b", r"\bpretty sure\b",
    r"\bif i'?m right\b", r"\blikely\b",
]

_AGREEMENT_FLIP = [
    r"\bi disagree\b", r"\bi don'?t agree\b", r"\bi do not agree\b",
    r"\bthat'?s wrong\b", r"\bi oppose\b", r"\bi can'?t back\b",
]
_DISAGREEMENT_FLIP = [
    r"\bi agree\b", r"\bi'?m with (you|p\d+)\b", r"\byou'?re right\b",
    r"\bexactly right\b", r"\bi support\b", r"\bi back\b",
]
_CONTRAST = [r"\bbut\b", r"\bhowever\b", r"\bthough\b", r"\byet\b", r"\bstill\b", r"\bexcept\b"]


def classify_arg(arg: str) -> str:
    arg = (arg or "").strip()
    if _PLAYER_RE.match(arg):
        return "player"
    if _SPEECH_RE.match(arg):
        return "speech_id"
    if _PHASE_RE.match(arg):
        return "phase"
    if arg.lower() in ROLE_SURFACE:
        return "role"
    if arg.lower() in ACTION_SURFACE:
        return "action"
    return "literal"


def surface_forms(arg: str, kind: str) -> List[str]:
    """Regex alternatives that count as realizing ``arg`` on the surface."""
    value = (arg or "").strip()
    low = value.lower()
    if kind == "player":
        num = re.sub(r"\D", "", value)
        return [rf"\bp\s*{num}\b", rf"\bplayer\s*#?\s*{num}\b", rf"#\s*{num}\b"]
    if kind == "speech_id":
        num = re.sub(r"\D", "", value)
        return [rf"\bs\s*{num}\b", rf"\bstatement\s*{num}\b"]
    if kind == "role":
        return [rf"\b{re.escape(form)}\b" for form in ROLE_SURFACE.get(low, [low])]
    if kind == "action":
        return [rf"\b{re.escape(form)}\b" for form in ACTION_SURFACE.get(low, [low])]
    return [rf"\b{re.escape(low)}\b"]


_CONTRACTIONS: List[tuple[str, str]] = [
    (r"\bi am\b", "i'm"), (r"\bwe are\b", "we're"), (r"\byou are\b", "you're"),
    (r"\bthey are\b", "they're"), (r"\bthat is\b", "that's"), (r"\bit is\b", "it's"),
    (r"\bthere is\b", "there's"), (r"\blet us\b", "let's"),
    (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"), (r"\bwas not\b", "wasn't"),
    (r"\bcan not\b", "can't"), (r"\bcannot\b", "can't"), (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"), (r"\bcould not\b", "couldn't"),
    (r"\bhave not\b", "haven't"), (r"\bhas not\b", "hasn't"),
    (r"\bi would\b", "i'd"), (r"\bi will\b", "i'll"), (r"\bi have\b", "i've"),
]


def text_variants(text: str) -> List[str]:
    """The lowered text plus its contracted spelling (deduplicated)."""
    lowered = (text or "").lower()
    contracted = lowered
    for pattern, replacement in _CONTRACTIONS:
        contracted = re.sub(pattern, replacement, contracted)
    return [lowered] if contracted == lowered else [lowered, contracted]


def _matches(text: str, patterns: List[str]) -> List[str]:
    variants = text_variants(text)
    return [p for p in patterns if any(re.search(p, v) for v in variants)]


def _realized(text: str, patterns: List[str]) -> bool:
    return bool(_matches(text, patterns))


def structural_check(
    source: SourceRecord,
    rewritten_nl: str,
    *,
    commitment_may_be_indirect: bool = False,
) -> CheckResult:
    """Deterministic preservation checks driven by the structured predicate."""
    reasons: List[str] = []
    details: Dict[str, Any] = {
        "slots": [],
        "unverifiable_slots": [],
        "added_hedges": [],
        "commitment_may_be_indirect": commitment_may_be_indirect,
        "identical_to_original": False,
    }

    text = (rewritten_nl or "").strip()
    if not text:
        return CheckResult(ok=False, reasons=["empty_rewrite"], details=details)

    original = source.original_nl or ""
    details["identical_to_original"] = _tokens(text) == _tokens(original)

    for index, arg in enumerate(source.predicate_args):
        kind = classify_arg(arg)
        if kind == "phase":
            continue
        patterns = surface_forms(arg, kind)
        in_original = _realized(original, patterns)
        in_rewrite = _realized(text, patterns)
        slot = {
            "index": index,
            "arg": arg,
            "kind": kind,
            "in_original": in_original,
            "in_rewrite": in_rewrite,
        }
        details["slots"].append(slot)
        if not in_original:
            details["unverifiable_slots"].append(arg)
            continue
        if in_rewrite:
            continue
        if commitment_may_be_indirect and kind in ("role", "action", "literal"):
            details["unverifiable_slots"].append(arg)
            continue
        reasons.append(f"lost_{kind}_slot:{arg}")

    added = [h for h in _matches(text, HEDGE_PATTERNS) if not _matches(original, [h])]
    details["added_hedges"] = added
    if added and not commitment_may_be_indirect:
        reasons.append("added_epistemic_hedge:" + ",".join(h.strip("\\b") for h in added))

    if source.predicate_name == "agree" and _matches(text, _AGREEMENT_FLIP):
        reasons.append("polarity_flip_suspected:agreement_became_disagreement")
    if source.predicate_name == "disagree":
        if _matches(text, _DISAGREEMENT_FLIP) and not _matches(text, _CONTRAST):
            reasons.append("polarity_flip_suspected:disagreement_became_agreement")

    if commitment_may_be_indirect and _is_hedged_copy(original, text):
        reasons.append("off_record_is_only_hedged_original")

    return CheckResult(ok=not reasons, reasons=reasons, details=details)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_hedged_copy(original: str, rewrite: str) -> bool:
    """True when the rewrite is the original plus only hedge/filler words."""
    hedge_words = {
        "maybe", "might", "perhaps", "possibly", "probably", "potentially",
        "seems", "seem", "appears", "think", "guess", "feel", "suspect",
        "could", "may", "likely", "i'm", "im", "not", "sure", "just", "kind",
        "sort", "of", "a", "bit",
    }
    original_tokens = set(_tokens(original))
    rewrite_tokens = _tokens(rewrite)
    if not rewrite_tokens or not original_tokens:
        return False
    novel = [t for t in rewrite_tokens if t not in original_tokens and t not in hedge_words]
    kept = sum(1 for t in original_tokens if t in set(rewrite_tokens))
    keeps_most = kept >= 0.7 * len(original_tokens)
    return keeps_most and len(novel) <= 2


