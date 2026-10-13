"""What the other players say about a treated speaker, and trust read as a rank.

The day's vote is one decision per replay. The discussion that precedes it is tens of
targeted predicates per replay, each naming who says what about whom, so it carries far
more of the reaction to a rewritten speaker than the vote does. This module reads those
predicates out of the recorded speeches and turns them into outcomes paired the same way
the vote outcomes are.

It also reads the reliability matrix as a rank inside each assessor's own row. Assessors
differ in how generously they score, and that spread is noise for a contrast between two
replays of the same checkpoint; a rank removes it.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from components.stats import mean
from lurkingwolf.agent import parse_action

#: ``[s13|P1] estimate(D1, P1, P2, werewolf): free text``
SPEECH_LINE = re.compile(r"^\[(?P<id>[^|\]]*)\|(?P<speaker>P\d+)\]\s*(?P<predicate>[a-z_]+\([^)]*\))")

#: predicate -> the argument position that names the player it is about
TARGET_ARG = {"estimate": 2, "question": 2, "answer": 2, "allied": 2, "unallied": 2,
              "dissociate": 2, "claim_divine": 2, "claim_guard": 2, "claim_antidote": 2,
              "claim_poison": 2, "recommend_action": 4}
#: predicate -> the argument position that names the utterance it answers, whose speaker is the target
SPEECH_ARG = {"agree": 2, "disagree": 2}
WOLF_WORDS = {"werewolf", "white_wolf_king", "wolf"}


def speech_acts(result: dict) -> List[dict]:
    """One row per recorded predicate: who said it, about whom, and what kind it is.

    `agree` and `disagree` name the utterance they answer rather than its speaker, so the
    speech ids are resolved first; they are the most direct reaction a player can make and
    are a fifth of all predicates.
    """
    speeches = (result or {}).get("speeches", [])
    speaker_of: Dict[str, str] = {}
    for speech in speeches:
        for line in speech.get("predicates") or []:
            match = SPEECH_LINE.match(line.strip())
            if match:
                speaker_of[match.group("id")] = match.group("speaker")
    acts: List[dict] = []
    for order, speech in enumerate(speeches):
        for line in speech.get("predicates") or []:
            match = SPEECH_LINE.match(line.strip())
            if not match:
                continue
            parsed = parse_action(match.group("predicate"))
            name, args = parsed["name"], [str(a).strip() for a in parsed["args"]]
            index = TARGET_ARG.get(name)
            target = args[index] if index is not None and len(args) > index else None
            speech_index = SPEECH_ARG.get(name)
            if speech_index is not None and len(args) > speech_index:
                target = speaker_of.get(args[speech_index])
            if target is not None and not re.fullmatch(r"P\d+", target):
                target = None
            role_claim = args[3].lower() if name == "estimate" and len(args) > 3 else ""
            acts.append({"order": order, "speaker": match.group("speaker"), "predicate": name,
                         "target": target, "role_guess": role_claim,
                         "disagrees": name == "disagree",
                         "accuses": name == "estimate" and role_claim in WOLF_WORDS,
                         "defends": name == "estimate" and bool(role_claim) and role_claim not in WOLF_WORDS,
                         "urges_execution": name == "recommend_action" and "vote_execute" in args})
    return acts


def speaker_position(result: dict, speaker: str) -> Optional[int]:
    for order, speech in enumerate((result or {}).get("speeches", [])):
        if speech.get("speaker") == speaker:
            return order
    return None


#: outcomes this module adds, all about the treated speaker
RESPONSE_OUTCOMES = ("accused_by_others", "defended_by_others", "execution_urged",
                     "disagreed_with", "mentions_after_speaking")
#: trust read as a rank; it measures the same matrix as `trust_in_speaker`, so it is reported
#: beside it rather than inside the discussion family.
EXTRA_OUTCOMES = ("trust_rank", "share_accused_after_speaking")


def response_outcomes(result: dict, speaker: str) -> dict:
    """How the others talk about ``speaker`` in this replay.

    The counts restricted to "after speaking" use only the players who spoke after the
    treated speaker, so they are a reaction to the rewritten speech rather than the day's
    opening positions.
    """
    acts = [a for a in speech_acts(result) if a["speaker"] != speaker]
    about = [a for a in acts if a["target"] == speaker]
    position = speaker_position(result, speaker)
    after = [a for a in about if position is not None and a["order"] > position]
    return {"accused_by_others": sum(a["accuses"] for a in about),
            "defended_by_others": sum(a["defends"] for a in about),
            "execution_urged": sum(a["urges_execution"] for a in about),
            "disagreed_with": sum(a["disagrees"] for a in about),
            "mentions_after_speaking": len(after),
            "share_accused_after_speaking": (sum(a["accuses"] for a in after) / len(after)
                                             if after else None),
            "trust_rank": trust_rank(result, speaker)}


def trust_rank(result: dict, speaker: str) -> Optional[float]:
    """The speaker's average rank in the assessors' own orderings, 0 = least trusted.

    Each village assessor alive after the night ranks the players it scored; the rank is
    rescaled to 0..1 so assessors who rate few players count the same as those who rate many.
    Ties share the midpoint of the positions they occupy.
    """
    from experiments.language_flip.code.analyze import assessors
    scores = result.get("reliability") or {}
    ranks = []
    for assessor in assessors(result, speaker):
        row = {t: v for t, v in (scores.get(assessor) or {}).items() if v is not None}
        if speaker not in row or len(row) < 2:
            continue
        values = sorted(row.values())
        below = sum(1 for v in values if v < row[speaker])
        equal = sum(1 for v in values if v == row[speaker])
        ranks.append((below + (equal - 1) / 2) / (len(values) - 1))
    return mean(ranks)
