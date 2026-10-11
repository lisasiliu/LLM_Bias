"""Trust-based politeness policy for village-side speakers.

For each utterance a village-side agent addresses to one player, the policy ranks
the speaker's current 1-10 reliability ratings of the living players, calls the
top half "trusted", the bottom half "distrusted" and (with an odd number) the middle
one "middle", and rewrites the utterance into the politeness strategy the chosen
arm assigns to that group. Every rewrite must pass the admission gate; otherwise
the bald rewrite is used, and if that fails too the original text is kept. Nothing
is dropped silently: every touched utterance is logged.
"""

from __future__ import annotations

import json
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.helpers import SourceRecord, parse_history_entry
from components.rewriters.politeness_rewriter import (
    PolitenessRewriter, predicate_definition, predicate_entry)
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import parse_predicate

BALD, POSITIVE, NEGATIVE = "bald_on_record", "positive_politeness", "negative_politeness"
TOUCHED_ACTS = ("question", "disagree", "recommend_action")
# D009 primary and secondary tiers: every speech act the flip experiment may rewrite
FLIP_ACTS = ("estimate", "question", "agree", "disagree", "recommend_action",
             "address_role", "allied", "unallied", "dissociate")
WOLF_ROLES = {"werewolf", "white_wolf_king"}

# arm -> strategy for each rank group (an unrated addressee is treated as "middle")
ARMS: Dict[str, Dict[str, str]] = {
    "all_bald": {"trusted": BALD, "distrusted": BALD, "middle": BALD},
    "trust_positive": {"trusted": POSITIVE, "distrusted": BALD, "middle": BALD},
    "trust_bald": {"trusted": BALD, "distrusted": POSITIVE, "middle": BALD},
    "all_positive": {"trusted": POSITIVE, "distrusted": POSITIVE, "middle": POSITIVE,
                    "unrated": POSITIVE, "unaddressed": POSITIVE},
    "bl_written": {"trusted": POSITIVE, "distrusted": NEGATIVE, "middle": BALD},
}


def addressee_of(name: str, args: List[str], speaker: str,
                 ref_speaker: Optional[str]) -> Optional[str]:
    """The single player an utterance is addressed to, or None.

    question, estimate, dissociate -> the player named; disagree/agree -> author of
    the referenced statement; recommend_action -> the actor, if not the speaker.
    address_role, allied and unallied name a group or two players: no single one.
    """
    if name in ("question", "recommend_action", "estimate", "dissociate") and len(args) >= 3:
        target = args[2].strip()
    elif name in ("disagree", "agree"):
        target = ref_speaker
    else:
        return None
    return target if target and re.fullmatch(r"P\d+", target) and target != speaker else None


def referenced_statement(history: List[str], ref_id: str) -> Optional[Dict[str, str]]:
    for entry in reversed(history):
        item = parse_history_entry(entry) if isinstance(entry, str) else {}
        if item and item.get("sid") == ref_id:
            return {"speech_id": item["sid"], "speaker": item["speaker"],
                    "predicate": item["pred"], "text": item.get("text", "")}
    return None


def rank_groups(rel: Dict[str, int], alive_others: List[str], seed_key: str) -> Dict[str, str]:
    """trusted / distrusted / middle for the rated living players. Ties are broken by
    a deterministic shuffle seeded by ``seed_key``."""
    rated = [p for p in alive_others if p in rel]
    order = list(rated)
    random.Random(seed_key).shuffle(order)
    order.sort(key=lambda p: -rel[p])          # stable: the shuffle decides ties
    half = len(order) // 2
    groups = {p: "middle" for p in order}
    for p in order[:half]:
        groups[p] = "trusted"
    for p in order[len(order) - half:] if half else []:
        groups[p] = "distrusted"
    return groups


@register
class TrustPolitenessPolicy(Rewriter):
    name = "trust_politeness_policy"

    def __init__(self, llm_client=None, arm: str = "trust_positive", seed: str = "",
                 model: str = "deepseek-chat", gate=None, log_path=None,
                 agents: Optional[dict] = None, retries: int = 1,
                 dry_run: bool = False, acts=TOUCHED_ACTS, include_wolves: bool = False,
                 treat_unaddressed: bool = False, **kwargs):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
        self.arm, self.seed, self.model = arm, str(seed), model
        self.gate, self.agents, self.retries = gate, agents or {}, int(retries)
        self.dry_run = dry_run
        self.acts, self.include_wolves = tuple(acts), bool(include_wolves)
        self.treat_unaddressed = bool(treat_unaddressed)
        self.log_path = Path(log_path) if log_path else None
        self._writers = {s: PolitenessRewriter(llm_client=llm_client, strategy=s, model=model)
                         for s in {BALD, POSITIVE, NEGATIVE}}
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = {}

    @property
    def applies_to_predicates(self):
        return set(self.acts)

    def bind(self, agents: dict) -> None:
        self.agents = agents

    def to_config(self) -> dict:
        return {"class": "TrustPolitenessPolicy",
                "params": {"arm": self.arm, "seed": self.seed, "model": self.model,
                           "acts": list(self.acts), "include_wolves": self.include_wolves,
                           "treat_unaddressed": self.treat_unaddressed}}

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    # ── rewrite ──

    def rewrite(self, ctx: RewriteContext) -> str:
        parsed = parse_predicate(ctx.predicate)
        name, args = str(parsed.get("name", "")), list(parsed.get("args", []))
        role = ctx.speaker_role_claimed or ""
        if (role in WOLF_ROLES and not self.include_wolves) or role == "" or name not in self.acts:
            self._count("untouched:wolf_or_other_act")
            return ctx.original_nl
        ref = None
        if name in ("disagree", "question", "agree") and args:
            ref = referenced_statement(ctx.public_history, args[-1].strip())
        addressee = addressee_of(name, args, ctx.speaker_id, (ref or {}).get("speaker"))
        if not addressee and not self.treat_unaddressed:
            self._count("untouched:no_addressee")
            return ctx.original_nl

        # village speakers rank by their own ratings; wolves by the engine's town-consensus score
        wolf_speaker = role in WOLF_ROLES
        rel = (ctx.town_consensus if wolf_speaker else ctx.speaker_reliability) or {}
        if wolf_speaker and not ctx.town_consensus:
            self._count("wolf_score_missing")
        alive_others = [p for p in (ctx.alive_players or []) if p != ctx.speaker_id]
        groups = rank_groups(rel, alive_others,
                             f"{self.seed}|{ctx.day}|{ctx.phase}|{ctx.speaker_id}")
        group = groups.get(addressee, "unrated") if addressee else "unaddressed"
        assigned = ARMS[self.arm].get(group, BALD)

        source = SourceRecord(
            source_id=f"{self.seed}:{ctx.speaker_id}:{len(ctx.public_history)}",
            predicate_name=name, predicate_text=ctx.predicate, predicate_args=args,
            predicate_definition=predicate_definition(name),
            nl_hint=predicate_entry(name).get("nl_hint", ""),
            original_nl=ctx.original_nl, phase=ctx.phase, speaker_id=ctx.speaker_id,
            speaker_role_claimed=role, game_id=self.seed,
            public_history=list(ctx.public_history), referenced_speech=ref)

        attempts: List[Dict[str, Any]] = []
        final, delivered = ctx.original_nl, "original_fallback"
        order = [assigned] if assigned == BALD else [assigned, BALD]
        for k, strategy in enumerate(order):
            attempt = self._attempt(ctx, source, strategy)
            attempts.append(attempt)
            if attempt["admitted"]:
                final = attempt["text"]
                delivered = "rewrite" if k == 0 else "bald_fallback"
                break
        self._count(f"touched:{self.arm}:{delivered}")
        self._log(ctx, name, addressee, group, rel.get(addressee), assigned, delivered,
                  final, attempts)
        return final

    def _attempt(self, ctx: RewriteContext, source: SourceRecord, strategy: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"strategy": strategy, "text": "", "admitted": False,
                               "error": "", "checks": None}
        text = ""
        if self.dry_run:                                   # no model calls in a dry run
            out.update(text=ctx.original_nl, admitted=True)
            return out
        for _ in range(self.retries + 1):
            try:
                text = self._writers[strategy].rewrite(ctx)
                out["error"] = ""
                break
            except Exception as exc:                      # a failed generation is a failure
                out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        if not text:
            return out
        out["text"] = text
        if self.gate is None:                              # dry runs only
            out["admitted"] = True
            return out
        try:
            out["checks"] = self.gate.check(source, text)
            out["admitted"] = bool(out["checks"]["admitted"])
        except Exception as exc:
            out["error"] = f"gate {type(exc).__name__}: {exc}"[:300]
        return out

    def _log(self, ctx, name, addressee, group, score, assigned, delivered, final, attempts):
        speaker_agent = self.agents.get(ctx.speaker_id)
        addr_agent = self.agents.get(addressee)
        record = {
            "seed": self.seed, "arm": self.arm, "day": ctx.day, "phase": ctx.phase,
            "speaker": ctx.speaker_id, "speaker_role": ctx.speaker_role_claimed,
            "addressee": addressee,
            "addressee_role": getattr(addr_agent, "role", None),
            "speech_act": name, "predicate": ctx.predicate,
            "speaker_score_for_addressee": score, "rank_group": group,
            "score_kind": "town_consensus" if ctx.speaker_role_claimed in WOLF_ROLES else "reliability",
            "assigned_strategy": assigned, "delivered_source": delivered,
            "original_text": ctx.original_nl, "final_text": final,
            "addressee_rating_of_speaker_before":
                (getattr(addr_agent, "reliability", {}) or {}).get(ctx.speaker_id),
            "addressee_ratings_so_far": len(getattr(addr_agent, "reasoning_history", []) or []),
            "attempts": attempts,
        }
        if self.log_path:
            with self._lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
