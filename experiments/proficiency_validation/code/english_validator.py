"""English-only semantic and pragmatic fidelity checks.

The politeness experiment has a different intervention contract. Its validator
is deliberately not modified or used as the English fidelity rubric.
"""

from __future__ import annotations

import hashlib
import json

from components.helpers import CheckResult
from components.judges.judge_model import JudgeModelError, JudgeTransportError

SEMANTIC_RUBRIC_VERSION = "english_fidelity_v1"
DIMENSIONS = {
    "content_preserved": "All original propositions, targets, roles, actions, reasons and logical relations are preserved. Nothing is added or deleted, including interpersonal framing.",
    "commitment_preserved": "Certainty and evidential strength are unchanged in BOTH directions. Adding or removing 'I think', 'I believe', 'maybe', 'clearly' or similar stance markers is a change, even if the main claim remains recoverable.",
    "act_force_preserved": "The same speech act is performed with the same directive force and directness. Recoverable indirectness alone is insufficient.",
    "speaker_perspective_preserved": "The same speaker speaks to the same addressee about the same targets. First, second and third person reference remains correct.",
    "reference_preserved": "Responses address the same earlier statement and preserve how it supports or contradicts the claim. Use n/a only if the original has no reference to an earlier statement.",
    "politeness_preserved": "Politeness, warmth, deference, solidarity and aggression stay the same. Added rapport, requests for cooperation, praise, insults or courtesy formulas are changes even without new game facts.",
    "register_preserved": "Formality and interpersonal register stay the same. Simpler or more varied vocabulary and syntax are allowed; making the speaker more formal, deferential, academic or patronizing is not.",
}
_SYSTEM = """You assess fidelity in a controlled English language-range experiment.
Only vocabulary and syntactic range may change. Politeness is NOT the treatment.
Do not infer preservation merely because both passages express the same predicate.
Judge every dimension below against the original, including the strength and
interpersonal presentation of the claim. Do not judge whether game claims are true.
Return JSON with one true/false value per dimension, plus a short 'notes' string.
Only reference_preserved may be 'n/a', and only when no earlier statement is referenced.
Missing, unclear or inapplicable answers for other dimensions cannot pass.
"""


def semantic_prompt_sha():
    return hashlib.sha256(json.dumps([SEMANTIC_RUBRIC_VERSION, _SYSTEM, DIMENSIONS],
                                     sort_keys=True).encode()).hexdigest()


class EnglishFidelityValidator:
    name = "english_fidelity_validator"

    def __init__(self, model=None, max_tokens=800):
        self.model = model
        self.max_tokens = max_tokens

    def build_prompts(self, source, rewritten_nl, commitment_may_be_indirect=False):
        if commitment_may_be_indirect:
            raise ValueError("English range manipulation never licenses indirectness")
        system = _SYSTEM + "\n" + "\n".join(f"{k}: {v}" for k, v in DIMENSIONS.items())
        user = json.dumps({"speaker": source.speaker_id, "speech_act": source.predicate_name,
                           "definition": source.predicate_definition,
                           "referenced_statement": source.referenced_speech,
                           "original": source.original_nl, "rewrite": rewritten_nl}, ensure_ascii=False)
        return system, user

    def validate(self, source, rewritten_nl, *, commitment_may_be_indirect=False):
        if self.model is None:
            return CheckResult(ok=None, reasons=["english_validator_unavailable"])
        system, user = self.build_prompts(source, rewritten_nl, commitment_may_be_indirect)
        try:
            result, completion = self.model.complete_json(system=system, user=user,
                                                          max_tokens=self.max_tokens)
        except (JudgeModelError, JudgeTransportError) as exc:
            return CheckResult(ok=None, reasons=[f"english_validator_error:{exc}"])
        if not isinstance(result, dict):
            return CheckResult(ok=None, reasons=["english_validator_unparseable_output"])
        details = {"rubric_version": SEMANTIC_RUBRIC_VERSION, "notes": result.get("notes", ""),
                   "provenance": completion.to_dict()}
        missing, broken = [], []
        for name in DIMENSIONS:
            value = result.get(name)
            if isinstance(value, str) and value.lower() in ("true", "false"):
                value = value.lower() == "true"
            if name == "reference_preserved" and value == "n/a" and not source.referenced_speech:
                details[name] = "n/a"
            elif isinstance(value, bool):
                details[name] = value
                if not value:
                    broken.append(f"not_preserved:{name}")
            else:
                missing.append(name)
        if missing:
            return CheckResult(ok=None, reasons=["english_validator_missing:" + ",".join(missing)], details=details)
        return CheckResult(ok=not broken, reasons=broken, details=details)
