"""Politeness Stage-3 rewriter for LURK Werewolf.

Politeness is a local, per-utterance pragmatic treatment.  The caller chooses
one of four strategies for the focal utterance while holding the underlying
predicate/game content fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
from textwrap import dedent

from lurkingwolf.rewriter import Rewriter, RewriteContext, register


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


def _normalize_strategy(strategy: str) -> str:
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
        **kwargs,
    ):
        self.llm_client = llm_client
        self.strategy = _normalize_strategy(strategy)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def rewrite(self, ctx: RewriteContext) -> str:
        # Fail closed: never substitute untreated text under a treatment label.
        if self.llm_client is None:
            raise RuntimeError("PolitenessRewriter requires an llm_client")

        predicate = _ctx_value(ctx, "predicate_name", "predicate", "speech_act")
        predicate_definition = _ctx_value(
            ctx,
            "predicate_definition",
            "predicate_description",
            "nl_hint",
        )

        if not predicate:
            raise ValueError(
                "RewriteContext must expose the predicate name (directly or in metadata)"
            )
        if not predicate_definition:
            raise ValueError(
                "RewriteContext must expose predicate_definition/nl_hint so the "
                "politeness rewriter can preserve the predicate's semantic scope"
            )

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
            {predicate_definition}
            Only express content licensed by this predicate and the original utterance.

            SEMANTIC CONTRACT - MANDATORY:
            1. Preserve speaker, addressee, target, role/alignment claims, actions, vote targets, night actions, referenced statements, evidence/reasons, polarity, and directive force.
            2. {certainty_rule}
            3. Preserve the original speech act. Agreement must remain agreement; disagreement must remain disagreement; a recommendation must remain a recommendation; a question must remain a question.
            4. Do not add or remove game facts, motives, evidence, social judgments, or claims about another player's honesty, competence, intentions, or values.
            5. Do not intentionally simplify or complexify vocabulary or grammar. CEFR/proficiency is a separate experimental treatment.
            6. Keep approximately the same informational content and amount of reasoning. Interpersonal framing may change only as required by the target strategy.
            7. Write one short, natural spoken utterance for the same live-game context.

            POLITENESS STRATEGY - {self.strategy}:
            {POLITENESS_STRATEGIES[self.strategy]}

            OUTPUT:
            Return the rewritten utterance only. No JSON, labels, commentary, or explanation.
            """
        ).strip()

        user_prompt = dedent(
            f"""\
            Predicate: {predicate}
            Original: {ctx.original_nl}

            Rewrite using the {self.strategy} strategy:
            """
        ).strip()

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = response.choices[0].message.content or ""
        text = text.strip()
        if not text:
            raise RuntimeError("Politeness rewriter returned an empty response")
        return text

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "params": {
                "strategy": self.strategy,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }
