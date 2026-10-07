"""Politeness Stage-3 rewriter for LURK Werewolf.

Politeness is a local, per-utterance pragmatic treatment.  The caller chooses
one of four strategies for the focal utterance while holding the underlying
predicate/game content fixed.

The rewriter is driven by the REAL ``lurkingwolf.rewriter.RewriteContext``, which
carries the full predicate string but no predicate definition.  The predicate
name is therefore parsed from ``ctx.predicate`` and its definition/``nl_hint``
is read from the single source of truth, ``lurkingwolf/prompts/prompt.json``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from functools import lru_cache
from textwrap import dedent
from typing import Optional, Set

from lurkingwolf.prompts import load_prompts
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import (
    clean_natural_text,
    parse_predicate,
    strip_reasoning,
)

THINK_OPEN = re.compile(r"</?think\b", re.IGNORECASE)


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

# D009 — predicates eligible for the symmetric four-way primary validation.
PRIMARY_PREDICATES: tuple[str, ...] = (
    "estimate",
    "question",
    "agree",
    "disagree",
    "recommend_action",
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


@lru_cache(maxsize=1)
def _predicate_registry() -> dict[str, dict]:
    """Flatten prompt.json's predicate groups into {name: entry}.

    Placeholders such as ``{valid_roles}`` are filled from the engine so the
    definition handed to the rewriter reads the same way Stage 1/2 sees it.
    """
    from lurkingwolf.engine import VALID_ROLES

    try:  # ADDRESS_ROLE_TARGETS is only used by address_role
        from lurkingwolf.agent import ADDRESS_ROLE_TARGETS
        target_roles = "|".join(sorted(ADDRESS_ROLE_TARGETS))
    except Exception:  # pragma: no cover - defensive, agent imports are heavy
        target_roles = ""

    roles = "|".join(sorted(VALID_ROLES))
    registry: dict[str, dict] = {}
    for group, entries in load_prompts().get("predicates", {}).items():
        if not isinstance(entries, Mapping):
            continue
        for name, entry in entries.items():
            filled = dict(entry)
            for key in ("desc", "nl_hint"):
                if isinstance(filled.get(key), str):
                    filled[key] = (
                        filled[key]
                        .replace("{valid_roles}", roles)
                        .replace("{valid_target_roles}", target_roles)
                    )
            filled["group"] = group
            registry[name] = filled
    return registry


def predicate_entry(name: str) -> dict:
    """Return the prompt.json entry for a predicate name ({} if unknown)."""
    return dict(_predicate_registry().get(name, {}))


def predicate_definition(name: str) -> str:
    """Signature + description + nl_hint for a predicate name."""
    entry = _predicate_registry().get(name)
    if not entry:
        return ""
    parts = [f"{name}{entry.get('sig', '')}: {entry.get('desc', '')}".strip()]
    if entry.get("nl_hint"):
        parts.append(entry["nl_hint"])
    return " ".join(p for p in parts if p).strip()


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


@register
class PolitenessRewriter(Rewriter):
    """Rewrite one focal utterance using one politeness strategy."""

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
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._applies_to = set(applies_to) if applies_to else None

    @property
    def applies_to_predicates(self) -> Optional[Set[str]]:
        return set(self._applies_to) if self._applies_to else None

    # ── Prompting ──

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
            Rewrite exactly one focal utterance so it uses the requested politeness strategy while preserving the same game content and CEFR/register level.

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
        return text

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
