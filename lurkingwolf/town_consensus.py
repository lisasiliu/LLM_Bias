"""Town consensus: how the town, publicly, has treated each player so far, from a wolf's point of view.

Engine code, no model calls. The score is cumulative from game start and is recomputed at each rating
round (a snapshot). It is never shown to any agent. Only events by players outside the wolf's team (the
wolf and its teammates) count, so the wolf side's own talk never moves it.

    estimate(S, T, werewolf)            T -1       estimate(S, T, <town role>)   T +1
    claim_divine(S, T, wolf_camp)       T -1       claim_divine(S, T, villager_camp)   T +1
    recommend_action(S, ., vote_execute, T)   T -1
    daytime vote S -> T                 T -1
    disagree / agree with a statement authored by T     T -1 / +1
    sheriff_set(T)                      T +1
"""

from __future__ import annotations

import re
from typing import Dict

from .engine import GameState, WOLF_ROLES
from .utils.speech_nl import parse_predicate

_SPEECH = re.compile(r"^\[(s\d+)\|(P\d+)\]\s*(.*)$", re.S)
_PLAYER = re.compile(r"P\d+$")


def _bare(pred_text: str) -> str:
    """The predicate itself, without the ': natural language' tail."""
    m = re.match(r"(\w+\([^)]*\))", pred_text.strip())
    return m.group(1) if m else pred_text.strip()


def town_consensus(state: GameState) -> Dict[str, int]:
    """player -> score. One table serves every wolf; teammates are scored like anyone else."""
    team = {p for p, r in state.roles.items() if r in WOLF_ROLES}
    score = {p: 0 for p in state.players}

    def add(target, delta):
        if target in score:
            score[target] += delta

    for entry in state.public_history:
        if not isinstance(entry, str):
            continue
        m = _SPEECH.match(entry)
        if not m:                                           # system results: only sheriff_set counts
            p = parse_predicate(_bare(entry))
            if p["name"] == "sheriff_set" and len(p["args"]) >= 2:
                add(p["args"][1], +1)
            continue
        sid, speaker, rest = m.groups()
        if speaker in team:
            continue
        p = parse_predicate(_bare(rest))
        name, a = p["name"], p["args"]
        if name == "estimate" and len(a) >= 4:
            add(a[2], -1 if a[3] in WOLF_ROLES else +1)
        elif name == "claim_divine" and len(a) >= 4:
            add(a[2], -1 if a[3] == "wolf_camp" else +1 if a[3] == "villager_camp" else 0)
        elif name == "recommend_action" and len(a) >= 5 and a[3] == "vote_execute":
            add(a[4], -1)
        elif name in ("agree", "disagree") and len(a) >= 3:
            author = (state.speech_registry.get(a[2]) or {}).get("speaker")
            if author:
                add(author, +1 if name == "agree" else -1)
    for day in state.vote_log:
        for voter, target in day.votes.items():
            if voter not in team and target and _PLAYER.match(str(target)):
                add(target, -1)
    return score
