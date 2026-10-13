"""Model access, the in-game gate, and the judges that take a model.

Swap the model in one place; the judges are unchanged:

    model = JudgeModel.resolve("deepseek-chat")
    InversePredicateParser(model)
    BlindPolitenessClassifier(model)
    AdmissionGate(model)
"""

from .judge_model import (
    Completion, JudgeModel, JudgeModelError, JudgeTransportError, ModelSpec,
    extract_json, known_handles, resolve_spec,
)
from .blind_politeness_classifier import BlindPolitenessClassifier
from .inverse_predicate_parser import InversePredicateParser
from .gate import AdmissionGate, GateUnavailable, structural_check

__all__ = [
    "AdmissionGate", "BlindPolitenessClassifier", "Completion", "GateUnavailable", "InversePredicateParser", "JudgeModel",
    "JudgeModelError", "JudgeTransportError", "ModelSpec", "extract_json", "known_handles", "resolve_spec", "structural_check",
]
