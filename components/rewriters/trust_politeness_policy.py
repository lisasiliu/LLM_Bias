"""Trust-based politeness policy.

For each eligible utterance (any of the nine D009 speech acts, any speaker, wolves included) the policy
ranks the speaker's current 1-10 reliability ratings of the living players. The top half (the middle player
of an odd count included) is "trusted", the bottom half "distrusted". The chosen setting assigns a politeness
strategy to each group. The rewrite must pass the gate; otherwise the original text is kept. Every
touched utterance is logged.
"""

from __future__ import annotations

import json
import re
import random
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.helpers import SourceRecord, parse_history_entry
from components.rewriters.politeness_rewriter import (
    ELIGIBLE_PREDICATES, PolitenessRewriter, predicate_definition, predicate_entry)
from lurkingwolf.engine import WOLF_ROLES
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import parse_predicate

BALD, POSITIVE = "bald_on_record", "positive_politeness"

# setting -> strategy for each rank group. A listener who is unrated or absent gets bald in the trust settings.
ARMS: Dict[str, Dict[str, str]] = {
    "all_bald": {"trusted": BALD, "distrusted": BALD},
    "all_positive": {"trusted": POSITIVE, "distrusted": POSITIVE, "unrated": POSITIVE, "unaddressed": POSITIVE},
    "trust_positive": {"trusted": POSITIVE, "distrusted": BALD},
    "trust_bald": {"trusted": BALD, "distrusted": POSITIVE},
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
    """trusted / distrusted for the rated living players: the top half by score is trusted, the rest
    distrusted, so the middle player of an odd count is trusted. Ties are broken by a shuffle seeded
    with ``seed_key``."""
    order = [p for p in alive_others if p in rel]
    random.Random(seed_key).shuffle(order)
    order.sort(key=lambda p: -rel[p])
    half = (len(order) + 1) // 2
    return {p: "trusted" if i < half else "distrusted" for i, p in enumerate(order)}


@register
class TrustPolitenessPolicy(Rewriter):
    name = "trust_politeness_policy"

    def __init__(self, llm_client=None, arm: str = "trust_positive", seed: str = "",
                 model: str = "deepseek-chat", gate=None, log_path=None,
                 agents: Optional[dict] = None, retries: int = 1,
                 dry_run: bool = False, **kwargs):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
        self.arm, self.seed, self.model = arm, str(seed), model
        self.gate, self.agents, self.retries = gate, agents or {}, int(retries)
        self.dry_run = dry_run
        self.log_path = Path(log_path) if log_path else None
        self._writers = {s: PolitenessRewriter(llm_client=llm_client, strategy=s, model=model)
                         for s in (BALD, POSITIVE)}
        self._lock = threading.Lock()
        self.counters: Dict[str, int] = {}

    @property
    def applies_to_predicates(self):
        return set(ELIGIBLE_PREDICATES)

    def bind(self, agents: dict) -> None:
        self.agents = agents

    def to_config(self) -> dict:
        return {"class": "TrustPolitenessPolicy",
                "params": {"arm": self.arm, "seed": self.seed, "model": self.model}}

    def _count(self, key: str) -> None:
        self.counters[key] = self.counters.get(key, 0) + 1

    # ── rewrite ──

    def rewrite(self, ctx: RewriteContext) -> str:
        parsed = parse_predicate(ctx.predicate)
        name, args = str(parsed.get("name", "")), list(parsed.get("args", []))
        role = ctx.speaker_role_claimed or ""
        if role == "" or name not in ELIGIBLE_PREDICATES:
            self._count("untouched:not_eligible")
            return ctx.original_nl
        ref = None
        if name in ("disagree", "question", "agree") and args:
            ref = referenced_statement(ctx.public_history, args[-1].strip())
        addressee = addressee_of(name, args, ctx.speaker_id, (ref or {}).get("speaker"))

        # A village speaker ranks listeners by their own reliability scores. A wolf's reliability scores are
        # its role guesses, not trust, so a wolf ranks by its own guess at the town consensus.
        alive_others = [p for p in (ctx.alive_players or []) if p != ctx.speaker_id]
        tie_key = f"{self.seed}|{ctx.speaker_id}|{len(ctx.public_history)}"
        wolf_speaker = role in WOLF_ROLES

        def group_of(scores):
            if not scores:
                return None
            return rank_groups(scores, alive_others, tie_key).get(addressee, "unrated") if addressee else "unaddressed"

        by_reliability, by_consensus = group_of(ctx.speaker_reliability), group_of(ctx.town_consensus)
        rel = (ctx.town_consensus if wolf_speaker else ctx.speaker_reliability) or {}
        if wolf_speaker and not ctx.town_consensus:
            self._count("wolf_consensus_missing")
        group = (by_consensus if wolf_speaker else by_reliability) or ("unaddressed" if not addressee else "unrated")
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
        for k in range(2):        # one regeneration with the same frozen prompt; then the original text
            attempt = self._attempt(ctx, source, assigned, addressee)
            attempts.append(attempt)
            if attempt["admitted"]:
                final, delivered = attempt["text"], "rewrite"
                break
        self._count(f"touched:{self.arm}:{delivered}")
        self._log(ctx, name, addressee, group, rel.get(addressee), assigned, delivered,
                  final, attempts, by_reliability, by_consensus)
        return final

    def _attempt(self, ctx: RewriteContext, source: SourceRecord, strategy: str,
                 addressee: Optional[str] = None) -> Dict[str, Any]:
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
            out["checks"] = self.gate.check(source, text, addressee=addressee)
            out["admitted"] = bool(out["checks"]["admitted"])
        except Exception as exc:
            out["error"] = f"gate {type(exc).__name__}: {exc}"[:300]
        return out

    def _log(self, ctx, name, addressee, group, score, assigned, delivered, final, attempts, by_reliability=None, by_consensus=None):
        speaker_agent = self.agents.get(ctx.speaker_id)
        addr_agent = self.agents.get(addressee)
        record = {
            "seed": self.seed, "arm": self.arm, "day": ctx.day, "phase": ctx.phase,
            "speaker": ctx.speaker_id, "speaker_role": ctx.speaker_role_claimed,
            "addressee": addressee,
            "addressee_role": getattr(addr_agent, "role", None),
            "speech_act": name, "predicate": ctx.predicate,
            "speaker_score_for_addressee": score, "rank_group": group,
            "reliability_for_addressee": (ctx.speaker_reliability or {}).get(addressee),
            "town_consensus_for_addressee": (ctx.town_consensus or {}).get(addressee),
            "rank_group_by_reliability": by_reliability,
            "rank_group_by_consensus": by_consensus,
            "assigned_strategy": assigned, "delivered_source": delivered,
            "delivered": "rewrite" if delivered == "rewrite" else "original",
            "delivered_attempt": next((i + 1 for i, a in enumerate(attempts) if a["admitted"]), None),
            "gate_failure_reasons": [(a.get("checks") or {}).get("gate_failure_reasons") or ([a["error"]] if a.get("error") else [])
                                     for a in attempts],
            "original_text": ctx.original_nl, "final_text": final,
            "addressee_rating_of_speaker_before":
                (getattr(addr_agent, "reliability", {}) or {}).get(ctx.speaker_id),
            "addressee_ratings_so_far": len(getattr(addr_agent, "reasoning_history", []) or []),
            "attempts": attempts,
        }
        if self.log_path:
            with self._lock, self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
