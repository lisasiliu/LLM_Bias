"""Per-dimension semantic preservation: the half a parser cannot see.

A model judges one boolean per dimension -- commitment, act force, evidence
relation, referenced statement, invented content, deleted content, speaker and
addressee -- and is asked only the dimensions that apply to the predicate at
hand. It is never asked for an overall "is this equivalent" verdict, which is
what previously let dropped directives through.

Fails closed: an unreachable model, unparseable output or a missing applicable
dimension yields ``ok=None``, which can never produce a validated record.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from components.judges.judge_model import JudgeModel, JudgeModelError, JudgeTransportError
from components.helpers import CheckResult, SourceRecord

SEMANTIC_RUBRIC_VERSION = "v2_speaker_perspective"


SEMANTIC_DIMENSIONS: Dict[str, str] = {
    "commitment_preserved": (
        "Epistemic commitment. Does the speaker still put the claim forward as their "
        "own position, as strongly as in the original?\n"
        "  PRESERVED when the claim is still asserted, even if a conventional stance "
        "marker softens how it lands on the hearer (\"I think\", \"I believe\", "
        "\"I'd say\", \"it seems to me\"), or a solidarity tag is appended to a "
        "position the speaker has already stated. Softening the FACE THREAT is what "
        "the treatment is for; it does not touch the truth claim.\n"
        "  NOT PRESERVED when the claim itself is downgraded: reduced to mere "
        "possibility (\"might be\", \"could be\", \"maybe\") where the original "
        "asserted; made conditional or hypothetical; attributed to someone else; or "
        "replaced by a request for the hearer to supply the judgement, so that the "
        "speaker never actually states it."
    ),
    "act_force_preserved": (
        "Force of the act. Does the rewrite still PERFORM the act the original "
        "performed, rather than merely gesturing at its subject matter? A "
        "recommendation that becomes an observation, a challenge that becomes a "
        "comment, or a disagreement that becomes a neutral remark about the same "
        "topic has lost its force. The act may be performed indirectly; what "
        "matters is that it is still being performed."
    ),
    "evidence_relation_preserved": (
        "Evidence/reason relation. If the original supports its point with a "
        "specific reason, is it the SAME reason, standing in the same relation to "
        "the same point? A different reason, or a reason attached to a different "
        "claim, is not preserved. Answer \"n/a\" if the original gives no reason."
    ),
    "same_referenced_statement": (
        "Referenced statement. The original responds to one specific earlier "
        "statement, whose text you are given. Does the rewrite still respond to "
        "THAT statement's content? Judge by content, not by any label or id: the "
        "id is never spoken aloud. Responding to a different earlier point is not "
        "preserved; neither is wording so unfocused that a listener could not tell "
        "which earlier statement is being responded to at all. Recoverability by a "
        "listener is the standard, not explicitness."
    ),
    "speaker_perspective_preserved": (
        "Speaker and addressee. The utterance is spoken BY one named player, who "
        "is given to you, and may be addressed TO another. Does the rewrite keep "
        "the same person speaking and the same person addressed?\n"
        "  NOT PRESERVED when the speaker ends up referring to themself in the "
        "second or third person; when something the original speaker announced "
        "they would do themself becomes a request for someone else to do it (or "
        "the reverse); when the speaker appears to be addressing themself; or "
        "when the rewrite is directed at a different player than the original.\n"
        "  Changing only how politely the addressee is approached is fine -- that "
        "is the treatment. What must not move is WHO is speaking and WHO is "
        "being spoken to."
    ),
    "no_invented_content": (
        "No invented GAME-RELEVANT content. The test is whether a listener's beliefs "
        "about the game could change because of something the rewrite added.\n"
        "  ALLOWED, and not a failure, is non-propositional rapport or common-ground "
        "framing, which the politeness treatment is licensed to introduce: "
        "\"let's work this out\", \"we're all trying to figure this out\", "
        "\"we can sort this out\", \"help me out here\". These assert nothing about "
        "the game and carry no information a listener could act on.\n"
        "  NOT ALLOWED is newly asserted information: game facts, evidence, or "
        "motives the original did not contain, and social or epistemic claims "
        "presented as true -- \"I trust you\", \"you've been honest\", "
        "\"we both know X\", \"we're on the same team\", invented praise or claims "
        "about another player's reliability, or an invented shared observation or "
        "shared belief. Answer false only for these."
    ),
    "no_deleted_content": (
        "No deleted content. Does the rewrite keep the game-relevant content of "
        "the original? Answer false if something a listener needs in order to act "
        "on the utterance has been dropped."
    ),
}

DIMENSIONS_BY_PREDICATE: Dict[str, Tuple[str, ...]] = {
    "estimate": (
        "commitment_preserved", "act_force_preserved",
        "evidence_relation_preserved", "no_invented_content", "no_deleted_content", "speaker_perspective_preserved",
    ),
    "agree": (
        "same_referenced_statement", "commitment_preserved", "act_force_preserved",
        "evidence_relation_preserved", "no_invented_content", "no_deleted_content", "speaker_perspective_preserved",
    ),
    "disagree": (
        "same_referenced_statement", "commitment_preserved", "act_force_preserved",
        "evidence_relation_preserved", "no_invented_content", "no_deleted_content", "speaker_perspective_preserved",
    ),
    "question": (
        "same_referenced_statement", "act_force_preserved",
        "no_invented_content", "no_deleted_content", "speaker_perspective_preserved",
    ),
    "recommend_action": (
        "act_force_preserved", "evidence_relation_preserved",
        "no_invented_content", "no_deleted_content", "speaker_perspective_preserved",
    ),
}

DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "commitment_preserved", "act_force_preserved",
    "evidence_relation_preserved", "no_invented_content", "no_deleted_content",
    "speaker_perspective_preserved",
)


def dimensions_for(predicate_name: str) -> Tuple[str, ...]:
    return DIMENSIONS_BY_PREDICATE.get(predicate_name, DEFAULT_DIMENSIONS)


def semantic_prompt_sha() -> str:
    """Content hash of the rubric: dimension texts, per-predicate sets, template."""
    payload = json.dumps({
        "version": SEMANTIC_RUBRIC_VERSION,
        "dimensions": SEMANTIC_DIMENSIONS,
        "by_predicate": {k: list(v) for k, v in DIMENSIONS_BY_PREDICATE.items()},
        "default": list(DEFAULT_DIMENSIONS),
        "system": _SYSTEM,
        "direct_rule": _DIRECT_COMMITMENT_RULE,
        "indirect_rule": _INDIRECT_COMMITMENT_RULE,
    }, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SYSTEM = """\
You check whether a rewritten Werewolf (Mafia) game utterance preserved specific
aspects of the original's meaning.

The rewrite was produced by a pragmatic/style treatment, so it is SUPPOSED to
sound different. Do not judge style, politeness, fluency, register, word choice,
or whether the claim is true in the game. Another check already verifies the
utterance's structured content (who it is about, what role or action is named).

Judge ONLY the dimensions listed below, each on its own. Do not give an overall
"is this equivalent" opinion.

DIMENSIONS TO JUDGE:
%(dimensions)s

For each, answer true (preserved), false (broken by the rewrite), or the string
"n/a" if the dimension genuinely does not apply to this utterance. Use "n/a"
sparingly and only where the question has no content — not as a way to avoid
deciding.

%(commitment_rule)s

Answer with JSON only, no prose, exactly:
{%(schema)s,
 "notes": "<=25 words, name the dimension you were least sure about"}
"""

_DIRECT_COMMITMENT_RULE = (
    "This rewrite uses a DIRECT treatment. Commitment must be equivalent: added "
    "uncertainty where the original asserted plainly is a failure."
)

_INDIRECT_COMMITMENT_RULE = (
    "This rewrite uses an INDIRECT (off-record) treatment, where leaving the point "
    "to be inferred IS the treatment. Mark commitment_preserved true whenever a "
    "listener could still recover what the speaker is committing to. Hypothetical "
    "framing, a pointed observation, or an invitation to draw a conclusion are "
    "licensed off-record devices, not weakened assertions. Judge recoverability, "
    "not explicitness. The one thing that is NOT recoverable indirectness is simply "
    "appending \"maybe\" or \"might\" to the original sentence, which downgrades "
    "the claim without implying anything."
)


def build_prompts(source: SourceRecord, rewritten_nl: str, *,
                  commitment_may_be_indirect: bool = False,
                  dimensions: Optional[Tuple[str, ...]] = None) -> Tuple[str, str]:
    """``(system, user)``. Only the applicable dimensions are asked about."""
    names = dimensions if dimensions is not None else dimensions_for(source.predicate_name)
    listed = "\n".join(f"- {name}: {SEMANTIC_DIMENSIONS[name]}" for name in names)
    schema = ",\n ".join(f'"{name}": true|false|"n/a"' for name in names)
    rule = (_INDIRECT_COMMITMENT_RULE if commitment_may_be_indirect
            else _DIRECT_COMMITMENT_RULE)
    system = _SYSTEM % {"dimensions": listed, "commitment_rule": rule, "schema": schema}

    reference = ""
    if "same_referenced_statement" in names:
        referenced = source.referenced_speech or {}
        text = (referenced.get("text") or "").strip()
        if text:
            reference = (f"The earlier statement being responded to, by "
                         f"{referenced.get('speaker', 'another player')}:\n"
                         f"  \"{text}\"\n")
        else:
            reference = ("The earlier statement being responded to is not "
                         "available as text; judge from the utterances alone.\n")

    speaker = (source.speaker_id or "").strip()
    spoken_by = f"Spoken by: {speaker}\n" if speaker else ""
    user = (
        f"Speech act: {source.predicate_name}\n"
        f"Definition: {source.predicate_definition}\n"
        f"{spoken_by}"
        f"{reference}"
        f"ORIGINAL (untreated): {source.original_nl}\n"
        f"REWRITTEN: {rewritten_nl}\n\n"
        "JSON verdict:"
    )
    return system, user


def _as_tristate(value: Any) -> Optional[str]:
    """``"true"`` / ``"false"`` / ``"na"``, or ``None`` when unreadable."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes"):
            return "true"
        if low in ("false", "no"):
            return "false"
        if low in ("n/a", "na", "not applicable", "none", "null"):
            return "na"
    return None


class DimensionalSemanticValidator:
    """Per-dimension semantic preservation check."""

    name = "dimensional_semantic_validator"

    def __init__(self, model: Optional[JudgeModel] = None, max_tokens: int = 400):
        self.model = model
        self.max_tokens = int(max_tokens)

    def build_prompts(self, source: SourceRecord, rewritten_nl: str,
                      commitment_may_be_indirect: bool = False) -> Tuple[str, str]:
        return build_prompts(source, rewritten_nl,
                             commitment_may_be_indirect=commitment_may_be_indirect)

    def validate(self, source: SourceRecord, rewritten_nl: str, *,
                 commitment_may_be_indirect: bool = False) -> CheckResult:
        names = dimensions_for(source.predicate_name)
        details: Dict[str, Any] = {"validator": self.name, "dimensions_asked": list(names)}

        if self.model is None:
            return CheckResult(ok=None, reasons=["semantic_validator_unavailable"],
                               details=details)
        if not (rewritten_nl or "").strip():
            return CheckResult(ok=False, reasons=["empty_rewrite"], details=details)

        system, user = self.build_prompts(source, rewritten_nl, commitment_may_be_indirect)
        try:
            verdict, _completion = self.model.complete_json(
                system=system, user=user, max_tokens=self.max_tokens)
        except (JudgeTransportError, JudgeModelError) as exc:
            return CheckResult(ok=None, reasons=[f"semantic_validator_error:{exc}"],
                               details=details)
        if verdict is None:
            return CheckResult(ok=None, reasons=["semantic_validator_unparseable_output"],
                               details=details)

        details["notes"] = str(verdict.get("notes", ""))[:200]
        reasons: List[str] = []
        missing: List[str] = []
        not_applicable: List[str] = []
        for name in names:
            state = _as_tristate(verdict.get(name))
            details[name] = state
            if state is None:
                missing.append(name)
            elif state == "na":
                not_applicable.append(name)
            elif state == "false":
                reasons.append(f"not_preserved:{name}")
        details["not_applicable"] = not_applicable

        if missing:
            return CheckResult(
                ok=None,
                reasons=[f"semantic_validator_missing_dimensions:{','.join(missing)}"],
                details=details)
        if len(not_applicable) == len(names):
            return CheckResult(ok=None,
                               reasons=["semantic_validator_answered_na_for_everything"],
                               details=details)
        return CheckResult(ok=not reasons, reasons=reasons, details=details)
