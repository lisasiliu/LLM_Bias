"""Model access, and the judges that take one.

Swap the model in one place; the judges are unchanged:

    model = JudgeModel.resolve("deepseek-chat")
    InversePredicateParser(model)
    DimensionalSemanticValidator(model)
    BlindPolitenessClassifier(model)

`structural_check` is the deterministic counterpart and needs no model.
"""

from .judge_model import (
    Completion, JudgeModel, JudgeModelError, JudgeTransportError, ModelSpec,
    extract_json, known_handles, resolve_spec,
)
from .blind_politeness_classifier import BlindPolitenessClassifier
from .dimensional_semantic_validator import DimensionalSemanticValidator
from .inverse_predicate_parser import InversePredicateParser
from .structural_check import structural_check

__all__ = [
    "BlindPolitenessClassifier", "Completion", "DimensionalSemanticValidator",
    "InversePredicateParser", "JudgeModel", "JudgeModelError",
    "JudgeTransportError", "ModelSpec", "extract_json", "known_handles",
    "resolve_spec", "structural_check",
]
