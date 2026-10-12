"""The in-game gate: a rewrite is delivered only if a model that did not write it reads it back correctly.

The inverse predicate parser sees only the rewrite and the public record and must recover the original
predicate; the dimensional semantic validator checks commitment, act force, evidence, referenced statement,
speaker perspective and invented or deleted content. Both must agree. A rewrite that fails is not delivered.
"""

from __future__ import annotations

from typing import Any, Dict

from components.helpers import SourceRecord, decide


class AdmissionGate:
    def __init__(self, model):
        from components.judges.dimensional_semantic_validator import DimensionalSemanticValidator
        from components.judges.inverse_predicate_parser import InversePredicateParser
        self.parser = InversePredicateParser(model)
        self.validator = DimensionalSemanticValidator(model)

    def check(self, source: SourceRecord, rewrite: str, *,
              commitment_may_be_indirect: bool = False) -> Dict[str, Any]:
        from components.judges.inverse_predicate_parser import compare, context_phase
        text = (rewrite or "").strip()
        if not text:
            return {"admitted": False, "parser_status": "not_run", "semantic_status": "not_run",
                    "adjudication_reasons": ["empty_rewrite"]}
        parsed = self.parser.parse(
            text, phase=context_phase(source), speaker_id=source.speaker_id,
            public_history=source.public_history)
        comparison = compare(parsed, source.predicate_text) if parsed.ok else None
        semantic = self.validator.validate(
            source, text, commitment_may_be_indirect=commitment_may_be_indirect)
        verdict = decide(
            parser_structural_match=bool(comparison and comparison["structural_match"]),
            semantic_ok=semantic.ok)
        return {
            "admitted": verdict.auto_accepted,
            "identical_to_original": text == (source.original_nl or "").strip(),
            "parser_status": verdict.parser_status,
            "parser_error": parsed.error,
            "parser_mismatched_slots": (comparison or {}).get("mismatched_slots"),
            "semantic_status": verdict.semantic_status,
            "semantic_reasons": list(semantic.reasons),
            "adjudication_reasons": list(verdict.reasons),
        }
