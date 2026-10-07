"""Labels the politeness strategy of one utterance, blind to what was requested.

Never told the requested condition, so the condition can be compared against the
label instead of acting as the answer key. Two labels exist beyond the four
strategies: `ambiguous_or_mixed` when none dominates, and `failed_manipulation`
when the text is not a usable in-game utterance.

DIAGNOSTIC ONLY: recorded for every rewrite, never used to accept or reject one.
Manual annotation is the manipulation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .judge_model import JudgeModel, JudgeModelError, JudgeTransportError

LABELS: tuple = (
    "bald_on_record",
    "positive_politeness",
    "negative_politeness",
    "off_record",
    "ambiguous_or_mixed",
    "failed_manipulation",
)

STRATEGY_LABELS: tuple = LABELS[:4]

PROMPT_VERSION = "act_then_strategy"

_CLASSIFIER_SYSTEM = """\
You label how a speech act is REALIZED in ONE utterance from a Werewolf (Mafia)
game, using Brown & Levinson's strategies. The strategies are kinds, not degrees.

Work in two steps, in this order.

STEP 1 — identify the speech act.
What is the speaker DOING? For example: asserting what role a player has;
challenging someone about an earlier statement; supporting or opposing an earlier
statement; recommending an action. Name it in at most 12 words.

STEP 2 — decide how THAT act is realized.
- bald_on_record: the target act is performed directly, without redress.
- positive_politeness: the target act is performed through affiliation,
  agreement, solidarity, or common ground.
- negative_politeness: the target act is CLEAR, but softened through deference,
  apology, or minimizing the imposition.
- off_record: the target act ITSELF is substantially left for the listener to
  infer.
- ambiguous_or_mixed: no one category dominates.
- failed_manipulation: not a usable in-game utterance at all — meta-commentary
  about the task, a refusal, an echoed instruction, or incoherent text.

DECISION RULE — do NOT label from politeness markers alone.
Ask whether the act you identified in STEP 1 is STATED or must be INFERRED.
- If the act is stated outright, it is NOT off_record, however the utterance
  opens. "Interesting that ...", "I noticed ...", "It's worth remembering ...",
  a tag question ("..., right?"), or a hedge ("I think") do NOT make a stated
  act off-record. Those are redress on a clear act, or no redress at all.
- Reserve off_record for utterances where the listener must do real inferential
  work to recover what is being claimed or asked for.
- Act clear, redress builds solidarity or common ground -> positive_politeness.
- Act clear, redress defers or reduces imposition -> negative_politeness.
- Act clear, no redress -> bald_on_record. Bluntness, aggression and rudeness
  are still bald_on_record, not failures.
- Markers of another category may co-occur naturally; that alone does not make
  the utterance mixed.

You are NOT told what the utterance was supposed to be. Do not guess at an
intended category; label what is actually there.

Answer with JSON only, no prose, exactly:
{"speech_act": "<=12 words naming what the speaker is doing>",
 "act_left_to_inference": true|false,
 "label": "<one of the six labels>",
 "confidence": "low|medium|high",
 "evidence": "<=15 words quoted from the utterance>",
 "notes": "<=20 words"}

"act_left_to_inference" records STEP 2's test: true only when the act itself must
be inferred. Setting it false and then labelling off_record is a contradiction.
"""


_MARKER_V1_SYSTEM = """\
You classify the politeness strategy of ONE Werewolf game utterance, using Brown &
Levinson's categories. The four categories are kinds, not degrees.

- bald_on_record: the act is performed directly, with no mitigation and no indirectness.
- positive_politeness: the act is framed to build solidarity, common ground, or in-group
  connection with the hearer.
- negative_politeness: the act is framed to reduce imposition and respect the hearer's
  autonomy (deference, apology, conventional indirectness, softened request).
- off_record: the act is performed indirectly, as a hint or implication, leaving the
  speaker plausible deniability, while the intended point stays recoverable.
- ambiguous_or_mixed: two or more strategies are equally present, or none dominates.
- failed_manipulation: not a usable in-game utterance at all.

Rules:
- Judge the utterance as a whole. Markers from another category may co-occur
  naturally; that alone is not a failure.
- Decide which single category the utterance MAINLY realizes.
- For off_record, merely adding "maybe" or "might" to a direct statement is NOT
  off-record. Genuine off-record leaves the point implied rather than stated.
- Aggression, bluntness or rudeness is still bald_on_record, not a failure.

Answer with JSON only, no prose, exactly:
{"label": "<one of the six labels>",
 "confidence": "low|medium|high",
 "evidence": "<=15 words quoted from the utterance>",
 "notes": "<=20 words"}
"""

PROMPTS: dict = {
    # Step 1 names the speech act, step 2 decides how that act is realized.
    "act_then_strategy": _CLASSIFIER_SYSTEM,
    # Baseline that labels from politeness markers.
    "marker_based": _MARKER_V1_SYSTEM,
}


def system_prompt(version: str = PROMPT_VERSION) -> str:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version {version!r}; have {sorted(PROMPTS)}")
    return PROMPTS[version]


def build_user_prompt(utterance: str, *, phase: str = "", speaker_id: str = "") -> str:
    """The utterance and nothing that could reveal the requested condition."""
    header = ""
    if speaker_id:
        header += f"Speaker: {speaker_id}\n"
    if phase:
        header += f"Phase: {phase}\n"
    return f"{header}Utterance: {utterance}\n\nJSON:"


@dataclass
class ClassificationResult:
    ok: bool
    label: Optional[str] = None
    speech_act: str = ""
    act_left_to_inference: Optional[bool] = None
    confidence: str = ""
    evidence: str = ""
    notes: str = ""
    error: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def inconsistent(self) -> bool:
        """The model called the act explicit and then labelled it off-record."""
        if self.label is None or self.act_left_to_inference is None:
            return False
        return (self.label == "off_record") != bool(self.act_left_to_inference)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "label": self.label, "speech_act": self.speech_act,
                "act_left_to_inference": self.act_left_to_inference,
                "inconsistent": self.inconsistent, "confidence": self.confidence,
                "evidence": self.evidence, "notes": self.notes, "error": self.error,
                "provenance": dict(self.provenance)}


class BlindPolitenessClassifier:
    """Label one utterance, never told what it was asked to be."""

    name = "blind_politeness_classifier"

    def __init__(self, model: JudgeModel, max_tokens: int = 250,
                 prompt_version: str = PROMPT_VERSION):
        self.model = model
        self.max_tokens = int(max_tokens)
        self.prompt_version = prompt_version
        self.system = system_prompt(prompt_version)

    def classify(self, utterance: str, *, phase: str = "",
                 speaker_id: str = "") -> ClassificationResult:
        if not (utterance or "").strip():
            return ClassificationResult(ok=False, error="empty_utterance")
        try:
            parsed, completion = self.model.complete_json(
                system=self.system,
                user=build_user_prompt(utterance, phase=phase, speaker_id=speaker_id),
                max_tokens=self.max_tokens)
        except (JudgeTransportError, JudgeModelError) as exc:
            return ClassificationResult(ok=False, error=f"{type(exc).__name__}: {exc}")

        provenance = completion.to_dict()
        if parsed is None:
            return ClassificationResult(ok=False, error="unparseable_json",
                                        provenance=provenance)
        label = str(parsed.get("label", "")).strip().lower()
        if label not in LABELS:
            return ClassificationResult(ok=False, error=f"unknown_label:{label}",
                                        provenance=provenance)
        inference = parsed.get("act_left_to_inference")
        return ClassificationResult(
            ok=True, label=label,
            speech_act=str(parsed.get("speech_act", ""))[:120],
            act_left_to_inference=inference if isinstance(inference, bool) else None,
            confidence=str(parsed.get("confidence", ""))[:10],
            evidence=str(parsed.get("evidence", ""))[:200],
            notes=str(parsed.get("notes", ""))[:200],
            provenance=provenance,
        )


