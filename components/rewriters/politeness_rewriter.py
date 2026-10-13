"""Politeness rewriting for LURK Werewolf.

`PolitenessRewriter` rewrites one utterance into one of four strategies while holding the underlying predicate/game content fixed.
`TrustPolitenessPolicy` (below) decides which strategy each utterance gets: it ranks the listener among the speaker's trusted and
distrusted players and applies one of four settings (all_bald, all_positive, trust_positive, trust_bald), with the rewrite checked by
the gate and the original text delivered when it fails.

The rewriter is driven by the REAL ``lurkingwolf.rewriter.RewriteContext``, which
carries the full predicate string but no predicate definition.  The predicate
name is therefore parsed from ``ctx.predicate`` and its definition/``nl_hint``
is read from the single source of truth, ``lurkingwolf/prompts/prompt.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import random
import threading
from collections.abc import Mapping
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Set

from components.helpers import SourceRecord, parse_history_entry, predicate_definition, predicate_entry
from components.judges.gate import GateUnavailable
from lurkingwolf.engine import WOLF_ROLES
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import (
    clean_natural_text,
    parse_predicate,
    strip_reasoning,
)

THINK_OPEN = re.compile(r"</?think\b", re.IGNORECASE)

# Stands in for the predicate definition when hashing the prompt TEMPLATE, so one
# hash per strategy covers every predicate. Same literal as the proficiency rewriter's.
DEFINITION_PLACEHOLDER = "{predicate_definition}"


POLITENESS_STRATEGIES: dict[str, str] = {
    "bald_on_record": (
        "Realize the speech act directly and without politeness mitigation. "
        "Use a plain, concise, unambiguous formulation. Do not add hedges, "
        "apologies, deference, rapport-building, insults, or extra force. "
        "Directness must not strengthen the proposition, certainty, or directive "
        "beyond the original."
    ),
    "positive_politeness": (
        "Frame the same speech act to support solidarity, common ground, or social "
        "connection with the listener. Appropriate mechanisms can include inclusive "
        "language, in-group framing, or brief rapport-oriented wording, but do not "
        "invent praise, honesty, competence, motives, shared beliefs, shared goals, "
        "or any other factual or epistemic claim that is absent from the original."
    ),
    "negative_politeness": (
        "Frame the same speech act to respect the listener's autonomy and reduce "
        "social imposition. Appropriate mechanisms can include deference, apology, "
        "softening of the request/act, or conventional indirectness. Soften the "
        "FACE-THREATENING ACT, not the truth conditions: do not add maybe, might, "
        "probably, seems, or other uncertainty unless that uncertainty already "
        "exists in the original."
    ),
    "off_record": (
        "Realize the same communicative intent indirectly through a hint, implication, "
        "or observation that allows plausible deniability. The same intended target, "
        "underlying game proposition, polarity, and relevant evidence/reason relation "
        "must remain recoverable. Explicit assertoric commitment may become indirect "
        "because plausible deniability is part of this treatment. Do not reduce the "
        "condition to ordinary uncertainty hedging such as merely adding maybe or "
        "might, and do not introduce new evidence."
    ),
}

# D009 — the speech acts the politeness treatment applies to (primary and secondary tiers).
ELIGIBLE_PREDICATES: tuple[str, ...] = (
    "estimate",
    "question",
    "agree",
    "disagree",
    "recommend_action",
    "address_role",
    "allied",
    "unallied",
    "dissociate",
)


def normalize_strategy(strategy: str) -> str:
    normalized = strategy.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bald": "bald_on_record",
        "positive": "positive_politeness",
        "negative": "negative_politeness",
        "offrecord": "off_record",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POLITENESS_STRATEGIES:
        raise ValueError(
            f"Unsupported politeness strategy {strategy!r}; expected one of "
            f"{sorted(POLITENESS_STRATEGIES)}"
        )
    return normalized


# Backwards-compatible private alias (was the only name in the published file).
_normalize_strategy = normalize_strategy


def _ctx_mapping(ctx: RewriteContext) -> Mapping:
    for attr in ("metadata", "meta", "extra"):
        value = getattr(ctx, attr, None)
        if isinstance(value, Mapping):
            return value
    return {}


def _ctx_value(ctx: RewriteContext, *names: str) -> str:
    """Read a field from RewriteContext directly or from a metadata mapping."""
    for name in names:
        value = getattr(ctx, name, None)
        if value not in (None, ""):
            if isinstance(value, Mapping) and "name" in value:
                return str(value["name"])
            return str(value)

    metadata = _ctx_mapping(ctx)
    for name in names:
        value = metadata.get(name)
        if value not in (None, ""):
            if isinstance(value, Mapping) and "name" in value:
                return str(value["name"])
            return str(value)
    return ""


def context_predicate_name(ctx: RewriteContext) -> str:
    """Predicate NAME for a context whose ``predicate`` is a full predicate string."""
    explicit = _ctx_value(ctx, "predicate_name", "speech_act")
    if explicit:
        return explicit.strip()
    raw = _ctx_value(ctx, "predicate")
    return str(parse_predicate(raw).get("name", "")).strip()


def _thinking_kwargs(model: str) -> dict:
    """The engine's thinking switch for the served model (off during games)."""
    try:
        from lurkingwolf.utils.gen_config import thinking_extra_body
    except ImportError:
        return {}
    return thinking_extra_body(model)


def _engine_log_path() -> Optional[Path]:
    """``rewrites.jsonl`` next to the engine subprocess's ``--output`` (replay) target."""
    argv = sys.argv
    if "--output" in argv:
        target = Path(argv[argv.index("--output") + 1])
        return (target.parent if target.suffix == ".json" else target) / "rewrites.jsonl"
    return None


@register
class PolitenessRewriter(Rewriter):
    """Rewrite one self utterance using one politeness strategy."""

    name = "politeness"

    def __init__(
        self,
        llm_client=None,
        strategy: str = "bald_on_record",
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 300,
        applies_to: Optional[list] = None,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.strategy = normalize_strategy(strategy)
        # Inside a LURK game or replay subprocess the engine's model name is the served model.
        self.model = model if model != "default" else os.getenv("LLM_MODEL", model)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._applies_to = set(applies_to) if applies_to else None
        self._template_hash = self.template_sha256()
        self.log_path = _engine_log_path()
        if self.log_path and self.log_path.exists():
            # The engine restarts an interrupted replay from its checkpoint: one log per attempt.
            self.log_path.unlink()

    @property
    def applies_to_predicates(self) -> Optional[Set[str]]:
        return set(self._applies_to) if self._applies_to else None

    # ── Prompting ──

    def _system_prompt(self, definition: str) -> str:
        """The system prompt for this strategy, with ``definition`` in the predicate slot."""
        if self.strategy == "off_record":
            certainty_rule = (
                "For off-record only, preserve the underlying intended proposition, "
                "target, polarity, and evidence/reason relation, but explicit assertoric "
                "commitment may become indirect because plausible deniability is part "
                "of this treatment. The proposition must remain recoverable. Do not "
                "merely add ordinary uncertainty words such as maybe or might."
            )
        else:
            certainty_rule = (
                "Preserve epistemic certainty. Do not make a claim more or less certain "
                "merely to sound polite, direct, or indirect."
            )

        system_prompt = dedent(
            f"""\
            You are a pragmatic-style controller for a Werewolf game research experiment.

            TASK:
            Rewrite exactly one self utterance so it uses the requested politeness strategy while preserving the same game content and CEFR/register level.

            PREDICATE SCOPE:
            {definition}
            Only express content licensed by this predicate and the original utterance.

            SEMANTIC CONTRACT - MANDATORY:
            1. Preserve speaker, addressee, target, role/alignment claims, actions, vote targets, night actions, referenced statements, evidence/reasons, polarity, and directive force.
            2. {certainty_rule}
            3. Preserve the original speech act. Agreement must remain agreement; disagreement must remain disagreement; a recommendation must remain a recommendation; a question must remain a question.
            4. Do not add or remove game facts, motives, evidence, social judgments, or claims about another player's honesty, competence, intentions, or values.
            5. Do not intentionally simplify or complexify vocabulary or grammar. CEFR/proficiency is a separate experimental treatment.
            6. Keep approximately the same informational content and amount of reasoning. Interpersonal framing may change only as required by the target strategy.
            7. Keep every player number from the original utterance (P3 stays P3).
            8. Write one short, natural spoken utterance for the same live-game context.

            POLITENESS STRATEGY - {self.strategy}:
            {POLITENESS_STRATEGIES[self.strategy]}

            OUTPUT:
            Return the rewritten utterance only. No JSON, labels, commentary, or explanation.
            """
        ).strip()

        return system_prompt

    def build_prompts(self, ctx: RewriteContext) -> tuple[str, str]:
        """Return (system_prompt, user_prompt). Raises if the context is unusable."""
        predicate_name = context_predicate_name(ctx)
        if not predicate_name:
            raise ValueError(
                "RewriteContext must expose the predicate (directly or in metadata)"
            )

        definition = _ctx_value(ctx, "predicate_definition", "predicate_description")
        if not definition:
            definition = predicate_definition(predicate_name)
        if not definition:
            raise ValueError(
                f"No predicate definition available for {predicate_name!r}; the "
                "politeness rewriter needs it to preserve the predicate's semantic scope"
            )

        if not (ctx.original_nl or "").strip():
            raise ValueError("RewriteContext.original_nl is empty; nothing to rewrite")

        system_prompt = self._system_prompt(definition)

        user_prompt = dedent(
            f"""\
            Predicate: {ctx.predicate}
            Speaker: {ctx.speaker_id}
            Phase: {ctx.phase}
            Original: {ctx.original_nl}

            Rewrite using the {self.strategy} strategy:
            """
        ).strip()

        return system_prompt, user_prompt

    def prompt_sha256(self, ctx: RewriteContext) -> str:
        """Hash of the system prompt as sent for THIS context (definition included)."""
        system, _ = self.build_prompts(ctx)
        return hashlib.sha256(system.encode("utf-8")).hexdigest()

    def template_sha256(self) -> str:
        """Hash of the system prompt with the predicate-definition slot left as a
        placeholder: one value per strategy, whatever the predicate."""
        return hashlib.sha256(self._system_prompt(DEFINITION_PLACEHOLDER).encode("utf-8")).hexdigest()

    # ── Rewriter interface ──

    def rewrite(self, ctx: RewriteContext) -> str:
        # Fail closed (D007): never substitute untreated text under a treatment label.
        if self.llm_client is None:
            raise RuntimeError("PolitenessRewriter requires an llm_client")

        system_prompt, user_prompt = self.build_prompts(ctx)

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **_thinking_kwargs(self.model),
        )
        raw = response.choices[0].message.content or ""
        # `clean_natural_text` only removes a CLOSED <think>...</think> PAIR, so it
        # misses the common case where the chat template pre-opened the tag and the
        # reply starts mid-reasoning, ending `...reasoning</think>\n<utterance>`.
        # `strip_reasoning` cuts at the last closing tag, so the utterance is what
        # survives and the later cleanups apply to it rather than to the trace.
        text = clean_natural_text(strip_reasoning(raw))
        # An UNTERMINATED <think> survives that cut: the model was still reasoning
        # when max_tokens ended the response, so the whole trace would otherwise be
        # returned as the public utterance.  That is a failed generation (D007).
        if THINK_OPEN.search(text):
            raise RuntimeError(
                "Politeness rewriter got an unterminated reasoning block -- the "
                f"response was cut off at max_tokens={self.max_tokens}. Raise "
                "max_tokens, or use a model that does not emit <think>."
            )
        if not text:
            raise RuntimeError("Politeness rewriter returned an empty response")
        self._record(ctx, text)
        return text

    def _record(self, ctx: RewriteContext, text: str) -> None:
        """One row per rewritten utterance in the replay directory's rewrites.jsonl."""
        if not self.log_path:
            return
        row = {"speaker": getattr(ctx, "speaker_id", None), "phase": getattr(ctx, "phase", None),
               "predicate": getattr(ctx, "predicate", None), "strategy": self.strategy,
               "condition": self.strategy, "model": self.model, "temperature": self.temperature,
               "prompt_sha256": self._template_hash,
               "status": "ok", "original_text": ctx.original_nl, "rewritten_text": text}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def to_config(self) -> dict:
        params = {
            "strategy": self.strategy,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self._applies_to:
            params["applies_to"] = sorted(self._applies_to)
        return {"class": self.__class__.__name__, "params": params}


# ======================================================================
# Trust-based setting policy
# ======================================================================

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
        # The second attempt uses the same frozen prompt and only samples at temperature 0.7.
        self._retry_writers = {s: PolitenessRewriter(llm_client=llm_client, strategy=s, model=model, temperature=0.7)
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
            attempt = self._attempt(ctx, source, assigned, addressee, retry=bool(attempts))
            attempts.append(attempt)
            if attempt["admitted"]:
                final, delivered = attempt["text"], "rewrite"
                break
        self._count(f"touched:{self.arm}:{delivered}")
        self._log(ctx, name, addressee, group, rel.get(addressee), assigned, delivered,
                  final, attempts, by_reliability, by_consensus)
        return final

    def _attempt(self, ctx: RewriteContext, source: SourceRecord, strategy: str,
                 addressee: Optional[str] = None, retry: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {"strategy": strategy, "text": "", "admitted": False,
                               "error": "", "checks": None}
        text = ""
        if self.dry_run:                                   # no model calls in a dry run
            out.update(text=ctx.original_nl, admitted=True)
            return out
        for _ in range(self.retries + 1):
            try:
                text = (self._retry_writers if retry else self._writers)[strategy].rewrite(ctx)
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
        except GateUnavailable as exc:                     # the replay must fail and be rerun; see flip_replay
            self._count("gate_unavailable")
            out["error"] = f"gate_unavailable: {exc}"[:300]
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
