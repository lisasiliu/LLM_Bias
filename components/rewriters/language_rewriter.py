"""CEFR Stage-3 rewriter for LURK Werewolf.

The treatment is a persistent speaker-level proficiency cue.  The caller is
responsible for assigning one CEFR band (A/B/C) to the self speaker and
reusing that same rewriter configuration for that speaker's later utterances.
"""

from __future__ import annotations

from collections.abc import Mapping
from textwrap import dedent

from lurkingwolf.rewriter import Rewriter, RewriteContext, register


CEFR_LEVEL_SPECS: dict[str, dict[str, str]] = {
    "A": {
        "General linguistic range": (
            "Ranges from a very basic repertoire of simple, concrete expressions "
            "and one-clause structures, sometimes with reduced elements, to a basic "
            "repertoire that can handle predictable familiar situations with brief "
            "expressions and basic sentence patterns. Some searching, repetition, "
            "or simplification of formulation is acceptable, but the game meaning "
            "must stay intact."
        ),
        "Vocabulary range": (
            "Ranges from basic words and short phrases for concrete situations to "
            "enough vocabulary for familiar, routine topics and basic communicative "
            "needs. Prefer common words outside protected Werewolf terminology; "
            "required game terms, player identifiers, and role names may remain "
            "specialized."
        ),
        "Grammatical accuracy": (
            "Ranges from limited control of a few simple learned structures and "
            "sentence patterns to using some simple structures correctly while "
            "still allowing basic mistakes. Meaning must remain clear. Do not "
            "manufacture errors and do not enforce an error quota."
        ),
        "Vocabulary control": (
            "Use a narrow, controlled repertoire for concrete needs. Prefer "
            "straightforward lexical choices except where protected game terms are "
            "required for semantic fidelity."
        ),
        "Coherence and cohesion": (
            "Ranges from linking words or short groups with very basic linear "
            "connectors to linking simple sentences with frequent connectors such "
            "as and, but, because, or then."
        ),
        "Thematic development": (
            "When the source contains more than one point, present the existing "
            "points as a simple sequence or list. Do not invent examples, reasons, "
            "or supporting details."
        ),
    },
    "B": {
        "General linguistic range": (
            "Ranges from enough language to handle familiar and some unpredictable "
            "situations, explain main points with reasonable precision, and express "
            "thoughts despite occasional hesitation or repetition, to a sufficiently "
            "broad range to describe, argue, and express viewpoints clearly using "
            "some complex sentence forms without conspicuous searching for words."
        ),
        "Vocabulary range": (
            "Ranges from a good range of vocabulary for familiar topics and everyday "
            "situations, with circumlocution when needed, to a broad range for general "
            "topics that allows varied formulation and reduced repetition. Required "
            "Werewolf terminology should remain exact."
        ),
        "Grammatical accuracy": (
            "Ranges from reasonable accuracy in familiar contexts, where errors may "
            "occur but the intended meaning stays clear, to good control of simple "
            "language and some complex forms, with only occasional slips or "
            "non-systematic errors that do not cause misunderstanding."
        ),
        "Vocabulary control": (
            "Ranges from good control of elementary vocabulary, with larger errors "
            "mainly on more complex or unfamiliar ideas, to generally high lexical "
            "accuracy with occasional incorrect word choice that does not hinder "
            "communication."
        ),
        "Coherence and cohesion": (
            "Ranges from linking shorter elements into a connected linear sequence "
            "with a limited set of cohesive devices to using a wider variety of "
            "linking expressions efficiently so relationships between existing ideas "
            "are clear."
        ),
        "Thematic development": (
            "Ranges from presenting a straightforward description or argument as a "
            "clear sequence of points to developing the source's argument "
            "systematically with appropriate emphasis on the points and support that "
            "are already present. Never add new support or evidence."
        ),
    },
    "C": {
        "General linguistic range": (
            "Ranges from a broad range of complex grammatical structures and "
            "formulations used appropriately and flexibly, with little need to "
            "restrict what is said, to comprehensive mastery of a very wide range of "
            "language for precise emphasis, differentiation, and disambiguation."
        ),
        "Vocabulary range": (
            "Ranges from a broad lexical repertoire with ready circumlocution, "
            "multiple formulation options, and appropriate use of less common or "
            "idiomatic vocabulary to a very broad repertoire that also handles "
            "colloquial and connotative meaning precisely. Do not replace protected "
            "game terms with stylistic synonyms."
        ),
        "Grammatical accuracy": (
            "Ranges from consistently high grammatical accuracy with errors that are "
            "rare and hard to notice to consistent control of complex language even "
            "when the content is demanding."
        ),
        "Vocabulary control": (
            "Ranges from appropriate idiomatic use of less common vocabulary with "
            "only minor slips to consistently correct and appropriate lexical choice."
        ),
        "Coherence and cohesion": (
            "Ranges from clear, smoothly flowing, well-structured language with "
            "controlled use of connectors and cohesive devices to flexible use of a "
            "wide range of organizational patterns and cohesive devices."
        ),
        "Thematic development": (
            "Organize the source's existing ideas clearly and, when the source itself "
            "contains several points, integrate them smoothly and emphasize their "
            "relationships. Do not introduce new subpoints, reasons, or examples."
        ),
    },
}


def _normalize_level(level: str) -> str:
    level = level.strip().upper()
    if level not in CEFR_LEVEL_SPECS:
        raise ValueError(f"Unsupported CEFR band {level!r}; expected A, B, or C")
    return level


def _render_spec(level: str) -> str:
    return "\n".join(
        f"- {dimension}: {description}"
        for dimension, description in CEFR_LEVEL_SPECS[level].items()
    )


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
class CEFRRewriter(Rewriter):
    """Rewrite one Stage-2 utterance into a requested broad CEFR band."""

    name = "cefr"

    def __init__(
        self,
        llm_client=None,
        level: str = "B",
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 300,
        **kwargs,
    ):
        self.llm_client = llm_client
        self.level = _normalize_level(level)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def rewrite(self, ctx: RewriteContext) -> str:
        # Fail closed: returning ctx.original_nl here would silently create an
        # untreated stimulus carrying a CEFR treatment label.
        if self.llm_client is None:
            raise RuntimeError("CEFRRewriter requires an llm_client")

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
                "RewriteContext must expose predicate_definition/nl_hint so the CEFR "
                "rewriter can preserve the predicate's semantic scope"
            )

        system_prompt = dedent(
            f"""\
            You are a CEFR language-level controller for a Werewolf game research experiment.

            TASK:
            Rewrite the utterance so its LANGUAGE matches CEFR band {self.level} while preserving the same game content, pragmatic strategy, and epistemic stance.

            BAND INTERPRETATION:
            A, B, and C are broad CEFR bands, not single point targets. Any natural realization anywhere inside the requested band is acceptable. Do not force the output toward either endpoint of the band and do not try to display every listed property at once. If the original already naturally fits the requested band, a minimal rewrite or no surface change is acceptable.

            PREDICATE SCOPE:
            {predicate_definition}
            Only express content that fits this predicate's scope. Do not add content from other predicate types.

            SEMANTIC CONTRACT - MANDATORY:
            1. Preserve every game-relevant proposition in the original: player identities, addressee, target, role/alignment claims, actions, vote targets, night actions, referenced statements, evidence/reasons, polarity, and directive force.
            2. Preserve epistemic certainty. Do not turn a definite claim into a possibility, or a possibility into a definite claim, unless the original already does so.
            3. Preserve required Werewolf terminology even if the word itself is more advanced than the target band. Simplify surrounding language instead.
            4. Do not add facts, strategy, motives, praise, blame, social judgments, or evidence.
            5. Do not intentionally change politeness, warmth, deference, directness, formality, or interpersonal stance. Politeness is a separate experimental treatment.
            6. Do not simulate an accent, nationality, or first-language stereotype. Do not deliberately insert grammatical errors merely to signal low proficiency.
            7. Keep the utterance natural spoken dialogue for the same player in the same live game. Do not turn it into an essay or explanation.

            CEFR BAND {self.level} SPECIFICATION:
            {_render_spec(self.level)}

            OUTPUT:
            Return the rewritten utterance only. No JSON, labels, commentary, or explanation.
            """
        ).strip()

        user_prompt = dedent(
            f"""\
            Predicate: {predicate}
            Original: {ctx.original_nl}

            Rewrite at CEFR band {self.level}:
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
            raise RuntimeError("CEFR rewriter returned an empty response")
        return text

    def to_config(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "params": {
                "level": self.level,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }
