"""Tokenization and the player-id normalization used by the grammar instruments.

Player ids (``P3``) and statement ids (``s12``) are game notation, not English.
Lexical measures drop them; the grammar/acceptability instruments see them
replaced by a fixed, common, gender-neutral first name (PLAN 8.1) so that a
tool does not count ``P3`` as a spelling error or an odd noun.  The same id
always maps to the same name across Original / Neutral / Low / High, and the
mapping is used ONLY for those instruments -- semantic checks see the raw ids.
"""

from __future__ import annotations

import re
from typing import Iterable, List

# Game ids (``P7``, ``s12``) are single tokens; a possessive stays attached so
# that ``P7's`` does not leave a stray ``s`` behind.
WORD_RE = re.compile(
    r"\b[Pp]\d{1,2}(?:['’]s)?\b|\bs\d{1,3}\b|[A-Za-z][A-Za-z'’\-]*[A-Za-z]|[A-Za-z]|\d+")
PLAYER_ID_RE = re.compile(r"\bP(\d{1,2})\b")
SPEECH_ID_RE = re.compile(r"\bs(\d{1,3})\b")
_PLAYER_TOKEN_RE = re.compile(r"^P\d{1,2}(?:['’]s)?$", re.IGNORECASE)
_SPEECH_TOKEN_RE = re.compile(r"^s\d{1,3}$")

# Fixed, common, gender-neutral English first names, one per seat.
PLAYER_NAMES: dict = {
    1: "Alex", 2: "Sam", 3: "Jordan", 4: "Taylor", 5: "Casey", 6: "Riley",
    7: "Morgan", 8: "Jamie", 9: "Avery", 10: "Quinn", 11: "Rowan", 12: "Skyler",
}

# Werewolf domain vocabulary (PLAN 7.1).  Lexical-frequency measures report
# these separately: their frequency in SUBTLEX reflects film subtitles, not the
# game, and every condition is required to keep them.
DOMAIN_TERMS: frozenset = frozenset({
    "werewolf", "werewolves", "wolf", "wolves", "villager", "villagers", "village",
    "seer", "witch", "guard", "hunter", "sheriff", "badge", "divine", "divined",
    "divination", "divinations", "antidote", "poison", "poisoned", "execute",
    "executed", "execution", "vote", "votes", "voted", "voting", "night", "day",
    "claim", "claimed", "claimant", "claimants", "camp", "wolfish", "lynch",
    "townsfolk", "alignment", "role", "roles", "mafia", "read", "reads",
    "seat", "seats", "flip", "flipped", "clear", "cleared", "check", "checked",
})


def tokenize(text: str) -> List[str]:
    """Word tokens (letters, apostrophes, hyphens) and bare numbers; no punctuation."""
    return WORD_RE.findall(text or "")


def is_player_id(token: str) -> bool:
    return bool(_PLAYER_TOKEN_RE.match(token or ""))


def is_speech_id(token: str) -> bool:
    return bool(_SPEECH_TOKEN_RE.match(token or ""))


def is_game_id(token: str) -> bool:
    return is_player_id(token) or is_speech_id(token)


def word_count(text: str) -> int:
    return len(tokenize(text))


def same_wording(a: str, b: str) -> bool:
    """Ignore case/spacing, but retain punctuation that changes sentence structure."""
    return surface_key(a) == surface_key(b)


def surface_key(text: str) -> tuple:
    return tuple(re.findall(r"[\w]+(?:['’][\w]+)*|[^\w\s]", (text or "").lower()))


def content_word_count(text: str) -> int:
    """Words excluding player and statement ids."""
    return sum(1 for t in tokenize(text) if not is_game_id(t))


def player_name(seat: int) -> str:
    if seat in PLAYER_NAMES:
        return PLAYER_NAMES[seat]
    # Seats beyond 12 never occur in LURK (7-12 players); keep it deterministic anyway.
    return f"Player{seat}"


def normalize_player_ids(text: str) -> str:
    """``P3`` -> ``Jordan``; ``s12`` -> ``statement 12``.  Grammar instruments only."""
    text = PLAYER_ID_RE.sub(lambda m: player_name(int(m.group(1))), text or "")
    text = SPEECH_ID_RE.sub(lambda m: f"statement {int(m.group(1))}", text)
    # Possessive forms survive: "P3's" -> "Jordan's".
    return text


def strip_game_ids(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if not is_game_id(t)]


def split_sentences(text: str) -> List[str]:
    """Regex sentence split for when spaCy is unavailable. Conservative."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(P])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]
