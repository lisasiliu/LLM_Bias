"""Linguistic-proficiency-cue Stage-3 rewriter for LURK Werewolf.

Conditions (experiments/proficiency_validation/PLAN.md, section 1):

    low      the treatment: a RESTRICTED linguistic range -- basic, repetitive
             vocabulary and narrow, simple syntax -- that stays grammatical,
             natural and fully understandable
    neutral  the paraphrase control: NEW surface wording at the SAME range,
             register, politeness and certainty as the original
    high     the optional extension: a WIDE, precise range

``original`` is not a rewriter condition: the untreated Stage-2 text is used
directly.  The cue is a SPEAKER-LEVEL characteristic, so the caller assigns one
condition to the focal speaker and reuses the same configuration for every
later utterance by that speaker in the branch.

The rewriter shares the politeness rewriter's context parser and prompt.json
predicate-definition lookup, so both treatments run under one predicate
contract.  It knows nothing about the semantic gate: gating, corrective retries
and the ``treatment_unavailable`` policy live in
``experiments.proficiency_validation.code.semantic_gate.GatedRewriter``.

The neutral condition generates several candidate paraphrases in ONE call (so
temperature 0 still yields distinct candidates) and picks the one whose
normalized token edit distance from the original is closest to a target,
normally the distance of the Low rewrite of the same utterance (PLAN 3.3).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from textwrap import dedent
from typing import List, Optional, Set

from components.judges.judge_model import THINK_OPEN
from components.metrics.edit_distance import normalized_token_edit_distance
from components.metrics.text_normalization import same_wording, surface_key, tokenize
from components.rewriters.politeness_rewriter import (
    _ctx_value,
    context_predicate_name,
    predicate_definition,
)
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import clean_natural_text, strip_reasoning

CONDITIONS: tuple = ("low", "neutral", "high")

PROMPT_VERSION = "range_v2"

# What each condition asks for.  These are the construct definitions; the paper
# quotes them.  Nothing here mentions errors, accents or learners: the cue is a
# restricted RANGE, and grammaticality is a guardrail, not a manipulation.
CONDITION_SPECS: dict = {
    "low": dedent("""\
        Represent a speaker with a RESTRICTED linguistic range who is nevertheless clear and correct.
        - Vocabulary: use basic, high-frequency, concrete words. Where the original uses a less common, abstract, figurative or idiomatic word or phrase, replace it with a common plain word or phrase that carries the same meaning. Reuse the same simple words rather than varying them with synonyms.
        - Syntax: use short sentences, mostly one clause each, in plain subject-verb-object order with simple tense forms. Link ideas with common connectors (and, but, so, because, then). Avoid relative clauses, nested subordination, participial phrases, inversion, cleft constructions and long noun phrases.
        - Cohesion: present the original's points as a simple sequence. Repeat the name or the noun instead of using elaborate reference.
        - Grammar: fully correct and natural. Do not insert errors. A restricted range is not broken English.
        - Content: every game-relevant point of the original must still be expressed. If a complex sentence has to be split into several simple ones, the text may become longer; that is acceptable. Do not drop a point to keep it short."""),
    "neutral": dedent("""\
        Produce a paraphrase control: NEW surface wording at the SAME linguistic range as the original.
        - Change the wording: use different words, phrasings and sentence constructions from the original wherever the meaning allows. The paraphrase must not be a near-copy of the original.
        - Do not change the level: keep vocabulary sophistication, sentence length, number of sentences, syntactic complexity, cohesion, idiomaticity and register approximately equal to the original. Do not simplify and do not elaborate.
        - Keep politeness, directness, certainty and interpersonal stance exactly as in the original."""),
    "high": dedent("""\
        Represent a speaker with a WIDE and PRECISE linguistic range.
        - Vocabulary: use precise, varied and, where natural, less common or idiomatic vocabulary; avoid repeating a word when a precise alternative exists. Do not replace protected game terms with stylistic synonyms.
        - Syntax: use flexible, varied sentence structure with controlled subordination, relative clauses and varied cohesive devices, where they express the same reasoning.
        - Cohesion: integrate the original's points smoothly and mark their logical relations explicitly, without adding new points.
        - Grammar: consistently accurate.
        - Content: do not inflate length, and do not add nuance, distinctions, motives or evidence in order to sound advanced. Fluent spoken dialogue, not academic prose."""),
}

PROTECTED_TERMS_NOTE = (
    "werewolf, wolf, villager, village, seer, witch, guard, hunter, sheriff, badge, "
    "divine/divination, antidote, poison, execute, vote"
)


def normalize_condition(condition: str) -> str:
    normalized = (condition or "").strip().lower()
    aliases = {"l": "low", "n": "neutral", "h": "high", "paraphrase": "neutral",
               "neutral_paraphrase": "neutral"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in CONDITIONS:
        raise ValueError(
            f"Unsupported proficiency condition {condition!r}; expected one of {CONDITIONS}")
    return normalized


# Dedented BEFORE substitution: the multi-line specification would otherwise
# defeat ``dedent`` and leave every line of the prompt indented.
_SYSTEM_TEMPLATE = dedent("""\
    You are a controlled language-proficiency rewriter for a Werewolf game research experiment.

    The proficiency cue is a SPEAKER-LEVEL CHARACTERISTIC: in this game the same speaker keeps the same cue across all of their messages. You are rewriting one of those messages.

    TASK:
    Rewrite the utterance so that its LINGUISTIC RANGE matches the %(COND)s specification below, while preserving the same game content, speech act, epistemic stance and interpersonal stance.

    PREDICATE SCOPE:
    %(definition)s
    Only express content licensed by this predicate and the original utterance.

    SEMANTIC CONTRACT - MANDATORY:
    1. Preserve speaker, addressee, target, role/alignment claims, actions, vote targets, night actions, referenced statements, evidence/reasons, polarity and directive force.
    2. Preserve epistemic certainty exactly. Do not turn a definite claim into a possibility or a possibility into a definite claim. Do not add or remove certainty words such as maybe, might, probably, seems, I think, clearly, definitely.
    3. Preserve the speech act. An estimate stays an estimate; a question stays a question; a disagreement stays a disagreement; a recommendation stays a recommendation with the same actor, action and target.
    4. Do not add facts, evidence, motives, praise, blame, social judgments, or claims about any player's honesty, competence or intentions. Do not delete any game-relevant proposition.
    5. Keep every player id exactly as written (P3 stays P3) and every statement id (s12 stays s12). Keep required Werewolf terms (%(protected)s) even where they are advanced words; change the language AROUND them instead.
    6. Do not intentionally change politeness, warmth, deference, directness, aggression or formality. Politeness is a separate experimental treatment; keep the interpersonal stance of the original.
    7. Do not simulate an accent, nationality or first-language stereotype. Do not insert grammatical errors, misspellings or broken English. Every condition must be natural, grammatical English.
    8. Write natural spoken dialogue for the same player in the same live game. No essay framing, greetings, labels or explanations. Stop when the source content is fully expressed; do not pad.

    %(COND)s SPECIFICATION:
    %(spec)s

    OUTPUT:
    %(output)s
    """)


DEFINITION_PLACEHOLDER = "{predicate_definition}"
SINGLE_OUTPUT_BLOCK = "Return the rewritten utterance only. No JSON, labels, commentary or explanation."


def _candidate_output_block(n: int) -> str:
    return (f"Return a JSON array of exactly {n} strings and nothing else. Each string is one "
            "complete paraphrase of the whole utterance. The paraphrases must differ from "
            "each other and from the original in wording. No commentary.")


def _shared_system_prompt(condition: str, definition: str, output_block: str) -> str:
    return (_SYSTEM_TEMPLATE % {
        "COND": condition.upper(), "definition": definition.strip(),
        "protected": PROTECTED_TERMS_NOTE, "spec": CONDITION_SPECS[condition].strip(),
        "output": output_block.strip()}).strip()


def _user_header(ctx: RewriteContext) -> str:
    """Joined line by line: ``dedent`` after interpolating a multi-line original
    would leave the whole prompt indented."""
    return "\n".join([f"Predicate: {ctx.predicate}", f"Speaker: {ctx.speaker_id}",
                      f"Phase: {ctx.phase}", f"Original: {ctx.original_nl}"])


def _feedback_block(feedback: Optional[dict], condition: str) -> str:
    """Corrective-retry text (PLAN 4.3): the gate's failure reasons, verbatim."""
    if not feedback:
        return ""
    reasons = feedback.get("reasons") or []
    previous = (feedback.get("previous") or "").strip()
    lines = ["", "A previous rewrite failed these semantic checks: "
             + ("; ".join(str(r) for r in reasons) or "unspecified") + "."]
    if previous:
        lines.append(f"Previous rewrite: {previous}")
    lines.append(f"Produce a corrected rewrite that fixes exactly these problems while "
                 f"still meeting the {condition.upper()} specification.")
    return "\n".join(lines)


@dataclass
class NeutralSelection:
    """Everything the neutral matching step decided, for the record."""

    chosen: str
    candidates: List[str] = field(default_factory=list)
    distances: List[float] = field(default_factory=list)
    target_distance: Optional[float] = None
    rule: str = "closest_to_target"           # or "median_distance"
    dropped: List[dict] = field(default_factory=list)   # {text, reason}

    @property
    def chosen_distance(self) -> Optional[float]:
        if self.chosen in self.candidates and len(self.distances) == len(self.candidates):
            return self.distances[self.candidates.index(self.chosen)]
        return None

    def to_dict(self) -> dict:
        return {"chosen": self.chosen, "chosen_distance": self.chosen_distance,
                "candidates": list(self.candidates), "distances": list(self.distances),
                "target_distance": self.target_distance, "rule": self.rule,
                "dropped": list(self.dropped)}


def parse_candidates(raw: str) -> List[str]:
    """A JSON array of strings, or a numbered / bulleted list as a fallback.

    ``clean_natural_text`` is applied per candidate, never to the whole reply:
    it strips a leading ``[...]`` block, which would eat the JSON array itself.
    """
    text = strip_reasoning(raw or "")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    items: List[str] = []
    # The first complete JSON array OF STRINGS, wherever it starts; trailing
    # commentary (even with brackets in it) and any nested numeric array in a
    # preamble are ignored rather than breaking the parse.
    decoder = json.JSONDecoder()
    arrays: List[list] = []
    start = text.find("[")
    while start != -1:
        try:
            parsed, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
            continue
        if isinstance(parsed, list):
            arrays.append(parsed)
        start = text.find("[", start + 1)
    string_arrays = [a for a in arrays if a and all(isinstance(x, str) for x in a)]
    if string_arrays:
        items = list(string_arrays[0])
    elif arrays:
        items = [str(x) for x in arrays[0]]
    if not items:
        for line in text.splitlines():
            match = re.match(r"^\s*(?:\d+[.)]|[-*])\s*(.+?)\s*$", line)
            if match:
                items.append(match.group(1).strip().strip('"'))
    cleaned = [clean_natural_text(item) for item in items]
    return [c for c in cleaned if c]


def select_candidate(candidates: List[str], original: str,
                     target_distance: Optional[float]) -> NeutralSelection:
    """Drop unusable candidates, then match on normalized token edit distance."""
    kept: List[str] = []
    dropped: List[dict] = []
    seen: Set[tuple] = set()
    for cand in candidates:
        toks = surface_key(cand)
        if not toks:
            dropped.append({"text": cand, "reason": "empty"})
        elif same_wording(cand, original):
            dropped.append({"text": cand, "reason": "identical_to_original"})
        elif toks in seen:
            dropped.append({"text": cand, "reason": "duplicate"})
        elif THINK_OPEN.search(cand):
            dropped.append({"text": cand, "reason": "reasoning_tag"})
        else:
            seen.add(toks)
            kept.append(cand)
    if not kept:
        raise RuntimeError("No usable neutral paraphrase candidate "
                           f"({len(dropped)} dropped: "
                           f"{sorted({d['reason'] for d in dropped})})")
    distances = [normalized_token_edit_distance(c, original) for c in kept]
    if target_distance is None:
        order = sorted(range(len(kept)), key=lambda i: distances[i])
        chosen = kept[order[len(order) // 2]]
        rule = "median_distance"
    else:
        chosen = min(kept, key=lambda c: abs(distances[kept.index(c)] - target_distance))
        rule = "closest_to_target"
    return NeutralSelection(chosen=chosen, candidates=kept, distances=distances,
                            target_distance=target_distance, rule=rule, dropped=dropped)


@register
class LanguageProficiencyRewriter(Rewriter):
    """Rewrite one utterance into the ``low``, ``neutral`` or ``high`` range."""

    name = "proficiency"

    def __init__(
        self,
        llm_client=None,
        condition: str = "low",
        model: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 400,
        n_neutral_candidates: int = 5,
        neutral_match_to: Optional[str] = "low",
        target_edit_distance: Optional[float] = None,
        applies_to: Optional[list] = None,
        prompt_version: Optional[str] = None,
        neutral_anchor_conditions: Optional[list] = None,
    ):
        # No **kwargs: an unknown parameter (e.g. a stale CEFR ``level="B"``) must
        # raise instead of silently running every utterance as ``low``.
        # ``prompt_version`` is what ``to_config`` records; a checkpoint written
        # under another prompt version must not resume as if nothing changed.
        if prompt_version is not None and prompt_version != PROMPT_VERSION:
            raise ValueError(f"config was written with prompt version {prompt_version!r}; "
                             f"this code has {PROMPT_VERSION!r}")
        self.llm_client = llm_client
        self.condition = normalize_condition(condition)
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.n_neutral_candidates = int(n_neutral_candidates)
        if neutral_match_to not in (None, "low", "high"):
            raise ValueError("neutral_match_to must be None, 'low' or 'high'")
        self.neutral_match_to = neutral_match_to
        self.neutral_anchor_conditions = (list(neutral_anchor_conditions) if neutral_anchor_conditions is not None
                                          else ([neutral_match_to] if neutral_match_to else []))
        if any(c not in ("low", "high") for c in self.neutral_anchor_conditions):
            raise ValueError("neutral anchors must be low and/or high")
        self.target_edit_distance = (None if target_edit_distance is None
                                     else float(target_edit_distance))
        self._applies_to = set(applies_to) if applies_to else None
        self.last_selection: Optional[NeutralSelection] = None
        self.last_raw: str = ""

    @property
    def applies_to_predicates(self) -> Optional[Set[str]]:
        return set(self._applies_to) if self._applies_to else None

    # ── Prompting ──

    def _definition(self, ctx: RewriteContext) -> str:
        predicate_name = context_predicate_name(ctx)
        if not predicate_name:
            raise ValueError("RewriteContext must expose the predicate (directly or in metadata)")
        definition = _ctx_value(ctx, "predicate_definition", "predicate_description")
        if not definition:
            definition = predicate_definition(predicate_name)
        if not definition:
            raise ValueError(
                f"No predicate definition available for {predicate_name!r}; the "
                "proficiency rewriter needs it to preserve the predicate's semantic scope")
        if not (ctx.original_nl or "").strip():
            raise ValueError("RewriteContext.original_nl is empty; nothing to rewrite")
        return definition

    def build_prompts(self, ctx: RewriteContext, feedback: Optional[dict] = None,
                      condition: Optional[str] = None) -> tuple:
        """``(system, user)`` for a single rewrite. Raises if the context is unusable."""
        condition = normalize_condition(condition or self.condition)
        definition = self._definition(ctx)
        system = _shared_system_prompt(condition, definition, SINGLE_OUTPUT_BLOCK)
        user = _user_header(ctx) + _feedback_block(feedback, condition) + \
            f"\n\nRewrite at the {condition.upper()} range:"
        return system, user

    def build_candidate_prompts(self, ctx: RewriteContext, n: Optional[int] = None,
                                feedback: Optional[dict] = None) -> tuple:
        """``(system, user)`` asking for ``n`` distinct neutral paraphrases at once."""
        n = int(n or self.n_neutral_candidates)
        definition = self._definition(ctx)
        system = _shared_system_prompt("neutral", definition, _candidate_output_block(n))
        user = _user_header(ctx) + _feedback_block(feedback, "neutral") + \
            f"\n\nWrite {n} different NEUTRAL paraphrases as a JSON array:"
        return system, user

    def prompt_sha256(self, ctx: RewriteContext, condition: Optional[str] = None, *,
                      candidates: bool = False) -> str:
        """Hash of the system prompt as sent for THIS context (definition included).
        ``candidates=True`` hashes the neutral multi-candidate prompt, the one a
        neutral rewrite is actually generated from."""
        if candidates:
            system, _ = self.build_candidate_prompts(ctx)
        else:
            system, _ = self.build_prompts(ctx, condition=condition)
        return hashlib.sha256(system.encode("utf-8")).hexdigest()

    def template_sha256(self, condition: Optional[str] = None, *, candidates: bool = False,
                        n: Optional[int] = None) -> str:
        """Hash of the system prompt with the predicate-definition slot left as a
        placeholder: the same for every predicate, so it is the artifact a frozen
        config pins and every rewrite row records. ``candidates=True`` hashes the
        neutral multi-candidate prompt for ``n`` candidates instead."""
        condition = normalize_condition(condition or self.condition)
        if candidates:
            n = int(n or self.n_neutral_candidates)
            system = _shared_system_prompt("neutral", DEFINITION_PLACEHOLDER, _candidate_output_block(n))
        else:
            system = _shared_system_prompt(condition, DEFINITION_PLACEHOLDER, SINGLE_OUTPUT_BLOCK)
        return hashlib.sha256(system.encode("utf-8")).hexdigest()

    # ── Transport ──

    def _complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        # Fail closed: returning ctx.original_nl here would silently create an
        # untreated stimulus carrying a treatment label.
        if self.llm_client is None:
            raise RuntimeError("LanguageProficiencyRewriter requires an llm_client")
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature,
            max_tokens=int(max_tokens or self.max_tokens),
        )
        raw = response.choices[0].message.content or ""
        self.last_raw = raw
        text = strip_reasoning(raw)
        if THINK_OPEN.search(text):
            raise RuntimeError(
                "Proficiency rewriter got an unterminated reasoning block -- the response "
                f"was cut off at max_tokens={max_tokens or self.max_tokens}. Raise "
                "max_tokens, or use a model that does not emit <think>.")
        return text

    # ── Rewriter interface ──

    def rewrite(self, ctx: RewriteContext, feedback: Optional[dict] = None) -> str:
        if self.condition == "neutral":
            return self.rewrite_neutral(ctx, feedback=feedback).chosen
        text = clean_natural_text(self._complete(*self.build_prompts(ctx, feedback)))
        if not text:
            raise RuntimeError("Proficiency rewriter returned an empty response")
        return text

    def rewrite_single(self, ctx: RewriteContext, condition: str,
                       feedback: Optional[dict] = None) -> str:
        """One rewrite in an explicit condition (used to target neutral matching)."""
        text = clean_natural_text(
            self._complete(*self.build_prompts(ctx, feedback, condition=condition)))
        if not text:
            raise RuntimeError("Proficiency rewriter returned an empty response")
        return text

    def generate_candidates(self, ctx: RewriteContext, n: Optional[int] = None,
                            feedback: Optional[dict] = None) -> List[str]:
        n = int(n or self.n_neutral_candidates)
        system, user = self.build_candidate_prompts(ctx, n, feedback)
        # Several paraphrases need room: scale the budget with the request.
        raw = self._complete(system, user, max_tokens=self.max_tokens * max(2, n))
        candidates = parse_candidates(raw)
        if not candidates:
            raise RuntimeError("Proficiency rewriter returned no parseable neutral candidates")
        return candidates

    def neutral_target_distance(self, ctx: RewriteContext) -> Optional[float]:
        """The distance the neutral paraphrase should match (PLAN 3.3)."""
        if self.target_edit_distance is not None:
            return self.target_edit_distance
        if self.neutral_match_to is None:
            return None
        anchor = self.rewrite_single(ctx, self.neutral_match_to)
        return normalized_token_edit_distance(anchor, ctx.original_nl)

    def rewrite_neutral(self, ctx: RewriteContext, feedback: Optional[dict] = None,
                        target_distance: Optional[float] = None,
                        candidates: Optional[List[str]] = None) -> NeutralSelection:
        target = (target_distance if target_distance is not None
                  else self.neutral_target_distance(ctx))
        if candidates is None:
            candidates = self.generate_candidates(ctx, feedback=feedback)
        selection = select_candidate(candidates, ctx.original_nl, target)
        self.last_selection = selection
        return selection

    def to_config(self) -> dict:
        params = {
            "condition": self.condition,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_version": PROMPT_VERSION,
        }
        if self.condition == "neutral":
            params.update({"n_neutral_candidates": self.n_neutral_candidates,
                           "neutral_match_to": self.neutral_match_to,
                           "target_edit_distance": self.target_edit_distance})
            params["neutral_anchor_conditions"] = self.neutral_anchor_conditions
        if self._applies_to:
            params["applies_to"] = sorted(self._applies_to)
        return {"class": self.__class__.__name__, "params": params}
