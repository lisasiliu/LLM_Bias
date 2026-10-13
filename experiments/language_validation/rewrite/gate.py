"""Semantic gate (structural, treatment, parser and fidelity checks with corrective retries), GatedRewriter and generation QC."""
from __future__ import annotations
import hashlib
import os
from collections import Counter
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from components.helpers import CheckResult, SourceRecord, parse_history_entry
from experiments.language_validation.rewrite.fidelity import SEMANTIC_RUBRIC_VERSION, EnglishFidelityValidator, semantic_prompt_sha
from experiments.language_validation.rewrite.runtime import ValidationModel
from components.judges.inverse_predicate_parser import PARSEABLE_PREDICATES, InversePredicateParser, build_system_prompt as parser_system_prompt, compare, context_phase
from components.api_usage import UsageLedger
from components.judges.judge_model import Completion, JudgeModel
from experiments.language_validation.rewrite.fidelity import NEUTRAL_DIMENSIONS, proficiency_check
from components.rewriters.politeness_rewriter import addressee_of
from components.metrics.edit_distance import normalized_token_edit_distance
from components.metrics.text_normalization import same_wording, surface_key
from components.rewriters.language_rewriter import LanguageProficiencyRewriter, NeutralSelection, select_candidate
from components.rewriters.politeness_rewriter import predicate_definition, predicate_entry
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import parse_predicate
from experiments.language_validation.rewrite.runtime import history_before, sha256_text


META_PATTERNS = [
    r"^\s*(rewrite|rewritten|paraphrase|original|output|answer)\s*[:\-]",
    r"\bas an ai\b", r"\bhere is\b.*\brewrite\b", r"\bCEFR\b", r"\bproficiency\b",
    r"\bspecification\b", r"\bjson\b", r"</?think\b",
]
_META_RE = re.compile("|".join(META_PATTERNS), re.IGNORECASE)


class TreatmentUnavailable(RuntimeError):
    """All attempts failed the gate and the caller asked to fail closed."""


def source_from_context(ctx: RewriteContext) -> SourceRecord:
    """Rebuild the ``SourceRecord`` the judges need from a live ``RewriteContext``."""
    parsed = parse_predicate(ctx.predicate)
    name = str(parsed.get("name", ""))
    args = [str(a) for a in parsed.get("args", [])]
    # The engine's public history also carries vote tallies as dicts; the corpus loader keeps the
    # string entries only, and the judges read the same form here.
    history = [entry for entry in (ctx.public_history or []) if isinstance(entry, str)]
    ref = None
    ref_arg = args[-1].strip() if args else ""
    if re.fullmatch(r"s\d+", ref_arg):
        for entry in history:
            item = parse_history_entry(entry)
            if item.get("sid") == ref_arg:
                ref = {"speech_id": item["sid"], "speaker": item["speaker"],
                       "predicate": item["pred"], "text": item.get("text", "")}
                break
    return SourceRecord(
        addressee=addressee_of(name, args, ctx.speaker_id, (ref or {}).get("speaker")),
        source_id=f"ctx:{ctx.speaker_id}:{ctx.predicate}",
        predicate_name=name, predicate_text=ctx.predicate, predicate_args=args,
        predicate_definition=predicate_definition(name),
        nl_hint=predicate_entry(name).get("nl_hint", ""),
        original_nl=ctx.original_nl, phase=ctx.phase, speaker_id=ctx.speaker_id,
        speaker_role_claimed=ctx.speaker_role_claimed,
        public_history=history, referenced_speech=ref,
        provenance="runtime_context")


def treatment_checks(source: SourceRecord, rewritten: str, condition: str, *,
                     identical: Optional[bool] = None) -> CheckResult:
    """Cheap, deterministic checks specific to the proficiency treatments.

    ``identical`` is the structural check's own ``identical_to_original`` flag;
    the gate passes it so there is ONE definition of "a copy of the original".
    """
    reasons: List[str] = []
    text = (rewritten or "").strip()
    if not text:
        return CheckResult(ok=False, reasons=["empty_rewrite"])
    if (same_wording(text, source.original_nl) if identical is None else identical):
        reasons.append("identical_to_original")
    if _META_RE.search(text):
        reasons.append("meta_text_or_leaked_instruction")
    if text.count("\n") >= 3:
        reasons.append("multi_paragraph_output")
    return CheckResult(ok=not reasons, reasons=reasons, details={"condition": condition})


PARSER_ABSTENTION_POLICIES = ("defer", "no_verdict")


@dataclass
class GateResult:
    ok: Optional[bool]                 # True pass, False fail, None no verdict
    reasons: List[str] = field(default_factory=list)
    structural: Dict[str, Any] = field(default_factory=dict)
    treatment: Dict[str, Any] = field(default_factory=dict)
    parser: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)
    second_opinion: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.ok is True

    @property
    def parser_status(self) -> Optional[str]:
        """recovered_match | recovered_mismatch | abstained | no_verdict | outside_grammar | not_run."""
        if "parser_predicate_outside_grammar" in self.skipped:
            return "outside_grammar"
        if not self.parser:
            return "not_run"
        if self.parser.get("abstained"):
            return "abstained"
        if not self.parser.get("ok"):
            return "no_verdict"
        return "recovered_match" if self.parser.get("structural_match") else "recovered_mismatch"

    @property
    def permanent_transport_failure(self) -> bool:
        errors = [self.parser.get("transport"), self.semantic.get("dimensions", {}).get("transport")]
        return any(isinstance(e, dict) and e.get("retryable") is False for e in errors)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "structural": self.structural,
                "treatment": self.treatment, "parser": self.parser, "parser_status": self.parser_status,
                "semantic": self.semantic, "second_opinion": self.second_opinion, "skipped": list(self.skipped)}


class LedgerJudgeModel(JudgeModel):
    """A ``JudgeModel`` that books every completion's usage under one purpose."""

    def __init__(self, spec, *, ledger: Optional[UsageLedger] = None, purpose: str = "other", **kwargs):
        super().__init__(spec, **kwargs)
        self.ledger = ledger
        self.purpose = purpose

    def complete(self, **kwargs) -> Completion:
        completion = super().complete(**kwargs)
        if self.ledger is not None:
            self.ledger.record(self.spec.provider, completion.model_served or self.spec.model,
                               self.purpose, completion.usage)
        return completion


class SemanticGate:
    """Run the four checks and combine them into one explicit verdict."""

    def __init__(self, parser: Optional[InversePredicateParser] = None,
                 validator: Optional[EnglishFidelityValidator] = None, *,
                 require_parser: bool = True, require_validator: bool = True,
                 judge_handle: Optional[str] = None, parser_abstention: str = "defer",
                 second_opinion: Optional[EnglishFidelityValidator] = None):
        if parser_abstention not in PARSER_ABSTENTION_POLICIES:
            raise ValueError(f"parser_abstention must be one of {PARSER_ABSTENTION_POLICIES}")
        self.parser = parser
        self.validator = validator
        # An independent-family fidelity judge consulted only when the parser abstained and the
        # primary judge passed: agreement passes, disagreement or no answer leaves no verdict.
        self.second_opinion = second_opinion
        # Active-paraphrase floor for Neutral: a gate-passing candidate that changed fewer than this
        # share of the tokens is a near-copy and is not selectable.
        self.neutral_exposure_floor = 0.05
        # Word-frequency scorer (text -> mean SUBTLEX Lg10CD) that ranks the gate-passing Neutral candidates.
        self.neutral_frequency = None
        self.subtlex_path = None
        self.neutral_spacy_model = "en_core_web_sm"
        # Prior-history budget for the parser and the act inventory (words). Referenced speech first,
        # then the current phase, then the most recent remaining entries; speech order is restored.
        self.parser_context_words = 6000
        self.require_parser = require_parser
        self.require_validator = require_validator
        # "defer": when the parser runs but recovers no predicate from the candidate, the English
        # fidelity judge decides alone (as it already does for predicates outside the parser
        # grammar) and the row records the abstention. "no_verdict": such a candidate gets no
        # verdict (the stricter pre-October policy). A recovered predicate that mismatches is a
        # rejection under both policies.
        self.parser_abstention = parser_abstention
        self.judge_handle = judge_handle
        self._original_parses = {}
        self._original_inventories = {}
        # When set (an AdmissionGate), the parser and fidelity judge are replaced by the shared in-game gate:
        # the action appeal and the reasons check. The English fidelity judge is then asked only for
        # Neutral's three range labels (the paraphrase control must stay at the original's range).
        self.admission = None

    def set_neutral_frequency(self, subtlex_path, spacy_model: str = "en_core_web_sm") -> None:
        """Load SUBTLEX and the tokenizer once for ranking Neutral candidates."""
        if not subtlex_path:
            raise ValueError("Neutral selection needs a SUBTLEX file")
        from experiments.language_validation.rewrite.__main__ import word_frequency_scorer
        self.subtlex_path = str(subtlex_path)
        self.neutral_frequency = word_frequency_scorer(subtlex_path, spacy_model)
        self.neutral_spacy_model = spacy_model

    @classmethod
    def from_handle(cls, handle: str = "deepseek-chat", *, ledger: Optional[UsageLedger] = None,
                    max_tokens: int = 3072, **kwargs) -> "SemanticGate":
        """Parser and validator on the same model, each booking its own usage purpose."""
        parser_model = ValidationModel.resolve(handle, ledger=ledger, purpose="inverse_parser")
        validator_model = ValidationModel.resolve(handle, ledger=ledger, purpose="semantic_validator")
        return cls(InversePredicateParser(parser_model, max_tokens=max_tokens),
                   EnglishFidelityValidator(validator_model, max_tokens=max_tokens),
                   judge_handle=handle, **kwargs)

    def to_config(self) -> dict:
        """Record gate policy, enabled checks and judge budgets."""
        secondary = None
        if self.second_opinion is not None:
            model = self.second_opinion.model
            handle = getattr(getattr(model, "spec", None), "handle", None)
            if not handle:
                raise ValueError("second-opinion model needs a registered handle for serialization")
            secondary = {"handle": handle, "identity": getattr(model, "identity_record", None),
                         "max_tokens": self.second_opinion.max_tokens}
        return {"judge_handle": self.judge_handle, "semantic_rubric_version": SEMANTIC_RUBRIC_VERSION,
                "parser_abstention": self.parser_abstention, "neutral_exposure_floor": self.neutral_exposure_floor,
                "subtlex_path": self.subtlex_path,
                "neutral_spacy_model": self.neutral_spacy_model,
                "parser_context_words": self.parser_context_words, "require_parser": self.require_parser,
                "require_validator": self.require_validator, "parser_enabled": self.parser is not None,
                "validator_enabled": self.validator is not None,
                "parser_max_tokens": getattr(self.parser, "max_tokens", None),
                "validator_max_tokens": getattr(self.validator, "max_tokens", None), "second_opinion": secondary}

    @classmethod
    def from_config(cls, config: dict) -> "SemanticGate":
        """Restore a serialized gate and check the recorded rubric and secondary model."""
        handle = config.get("judge_handle")
        if not handle:
            raise ValueError("gate config has no judge_handle")
        recorded = config.get("semantic_rubric_version")
        if recorded and recorded != SEMANTIC_RUBRIC_VERSION:
            raise ValueError(f"gate config was frozen with semantic rubric {recorded!r}; "
                             f"this code has {SEMANTIC_RUBRIC_VERSION!r}")
        gate = cls.from_handle(handle, parser_abstention=config.get("parser_abstention", "defer"),
                               require_parser=config.get("require_parser", True),
                               require_validator=config.get("require_validator", True))
        for name in ("parser", "validator"):
            if not config.get(f"{name}_enabled", True):
                setattr(gate, name, None)
            elif config.get(f"{name}_max_tokens") is not None:
                getattr(gate, name).max_tokens = int(config[f"{name}_max_tokens"])
        gate.neutral_exposure_floor = float(config.get("neutral_exposure_floor", .05))
        if config.get("subtlex_path"):
            gate.set_neutral_frequency(config["subtlex_path"], config.get("neutral_spacy_model", "en_core_web_sm"))
        gate.parser_context_words = int(config.get("parser_context_words", 6000))
        secondary = config.get("second_opinion")
        if secondary:
            model = ValidationModel.resolve(secondary["handle"], purpose="semantic_second_opinion")
            if secondary.get("identity") is not None and secondary["identity"] != model.identity_record:
                raise ValueError("second-opinion model identity changed")
            gate.second_opinion = EnglishFidelityValidator(model, max_tokens=int(secondary["max_tokens"]))
        return gate

    # ── individual judges ──

    def _run_parser(self, source: SourceRecord, rewritten: str) -> Optional[Dict[str, Any]]:
        """Parser verdict, or ``None`` when the predicate is outside its grammar.

        The parser is shown ONLY the history before this utterance: the full
        game history a source carries includes the entry that realizes the
        utterance itself, which would hand the parser the answer.
        """
        if self.parser is None or source.predicate_name not in PARSEABLE_PREDICATES:
            return None
        history, context_record = parser_context(source, budget_words=self.parser_context_words)
        key = (source.source_id, source.predicate_text, source.original_nl, tuple(history))
        baseline = self._original_parses.get(key)
        if baseline is None:
            baseline = self.parser.parse(source.original_nl, phase=context_phase(source),
                                         speaker_id=source.speaker_id, public_history=history)
            # Transport failures are retriable and must not become a cached baseline.
            if baseline.ok:
                self._original_parses[key] = baseline
        parsed = (baseline if rewritten == source.original_nl else self.parser.parse(
            rewritten, phase=context_phase(source), speaker_id=source.speaker_id, public_history=history))
        comparison = compare(parsed, source.predicate_text)
        original_comparison = compare(baseline, source.predicate_text)
        baseline_usable = bool(baseline.ok and baseline.predicate_name and baseline.recoverable is True
                               and not baseline.unresolved and "?" not in baseline.predicate_text)
        candidate_usable = bool(parsed.ok and parsed.predicate_name and parsed.recoverable is True
                                and not parsed.unresolved and "?" not in parsed.predicate_text)
        original_match = (compare(parsed, baseline.predicate_text)["content_match"]
                          if baseline_usable and candidate_usable else False)
        # A single-label parser can select a secondary act in both texts. The full
        # English fidelity judgment remains mandatory for this comparison basis.
        same_original_act = bool(not original_comparison["structural_match"] and original_match)
        structural_match = bool(candidate_usable and (comparison["structural_match"] or same_original_act))
        inventory = None
        basis = "original_parse" if same_original_act else "source_predicate"
        model = getattr(self.parser, "model", None)
        if not structural_match and model is not None:
            from experiments.language_validation.rewrite.fidelity import extract_inventory
            if key not in self._original_inventories:
                original_inventory = extract_inventory(model, source.original_nl, phase=context_phase(source),
                    speaker=source.speaker_id, history=history, max_tokens=self.parser.max_tokens)
                if original_inventory["ok"]:
                    self._original_inventories[key] = original_inventory
            else:
                original_inventory = self._original_inventories[key]
            candidate_inventory = extract_inventory(model, rewritten, phase=context_phase(source),
                speaker=source.speaker_id, history=history, max_tokens=self.parser.max_tokens)
            inventory = {"original": original_inventory, "rewrite": candidate_inventory}
            if original_inventory["ok"] and candidate_inventory["ok"]:
                structural_match = original_inventory["signatures"] == candidate_inventory["signatures"]
                candidate_usable = True
                basis = "act_inventory"
        return {"ok": candidate_usable, "structural_match": structural_match,
                "exact_match": comparison["exact_match"],
                "recovered_predicate": comparison["recovered_predicate"],
                "mismatched_slots": (["predicate_name"] if not comparison["name_match"] else
                                     [s for s in comparison["mismatched_slots"] if s in comparison["gated_slots"]]),
                "comparison_basis": basis, "act_inventory": inventory,
                "source_predicate_match": original_comparison["structural_match"],
                "original_parse": baseline.to_dict(), "candidate_parse": parsed.to_dict(),
                "original_comparison": original_comparison,
                "candidate_comparison": comparison,
                "error": parsed.error or ("predicate_not_recoverable" if not candidate_usable else ""),
                "provenance": parsed.provenance,
                "transport": getattr(getattr(self.parser, "model", None), "last_error", None),
                "history_entries": len(history), "context": context_record, "source_id": source.source_id}

    def _run_validator(self, source: SourceRecord, rewritten: str, condition=None) -> Optional[Dict[str, Any]]:
        if self.validator is None:
            return None
        kwargs = {"condition": condition} if getattr(self.validator, "supports_condition", False) else {}
        verdict = self.validator.validate(source, rewritten, commitment_may_be_indirect=False, **kwargs)
        return {"ok": verdict.ok, "reasons": verdict.reasons,
                "dimensions": {k: v for k, v in verdict.details.items()
                               if k not in ("validator", "dimensions_asked")},
                "rubric_version": SEMANTIC_RUBRIC_VERSION}

    def _evaluate_admission(self, source: SourceRecord, rewritten: str, condition: str, structural,
                            result: "GateResult") -> "GateResult":
        reasons = result.reasons
        info: Dict[str, Any] = {"action_appeals": [], "evidence": None, "neutral_range": None}
        for signal in structural.details.get("semantic_signals", []):
            if signal.startswith("lost_action_slot:"):
                appeal = self.admission.appeal(source, rewritten, signal.split(":", 1)[1])
                info["action_appeals"].append(appeal)
                if appeal["answer"] is not True:
                    reasons.append(signal)
        info["evidence"] = self.admission.reasons_check(source, rewritten)
        if info["evidence"]["kept"] is False:
            reasons.append("evidence_dropped")
        no_verdict = False
        if condition == "neutral" and not reasons and self.validator is not None:
            verdict = self.validator.validate(source, rewritten, commitment_may_be_indirect=False, condition=condition)
            labels = {name: (verdict.details or {}).get(name) for name in NEUTRAL_DIMENSIONS}
            info["neutral_range"] = labels
            if any(v not in ("simpler", "same", "more_advanced") for v in labels.values()):
                no_verdict = True
                reasons.append("neutral_range_no_verdict")
            else:
                reasons += [f"{name}:{label}" for name, label in labels.items() if label != "same"]
        result.semantic = {"ok": None if no_verdict else (not reasons), "reasons": list(reasons),
                           "dimensions": {}, "admission": info}
        result.skipped += ["parser", "fidelity_judge"]
        result.ok = False if [r for r in reasons if r != "neutral_range_no_verdict"] else (None if no_verdict else True)
        result.reasons = reasons
        return result

    def judge_all(self, source: SourceRecord, rewritten: str) -> Dict[str, Any]:
        """Every judge on the same text, independently and without short-circuit.

        For calibration (the challenge set): the structural check, the parser
        and the validator each give their own verdict on every item, and the
        treatment checks are reported but not applied -- an original-vs-original
        positive control is a legitimate benchmark item, not a failed rewrite.
        """
        structural = proficiency_check(source, rewritten)
        treatment = treatment_checks(source, rewritten, "benchmark",
                                     identical=same_wording(rewritten, source.original_nl))
        return {"structural": {"ok": structural.ok, "reasons": structural.reasons,
                               "details": structural.details},
                "treatment": {"ok": treatment.ok, "reasons": treatment.reasons},
                "parser": self._run_parser(source, rewritten),
                "semantic": self._run_validator(source, rewritten)}

    @property
    def judge_model(self) -> Optional[JudgeModel]:
        if self.parser is not None:
            return self.parser.model
        if self.validator is not None:
            return self.validator.model
        return None

    def evaluate(self, source: SourceRecord, rewritten: str, condition: str, *,
                 cheap_only: bool = False) -> GateResult:
        reasons: List[str] = []
        skipped: List[str] = []

        structural = proficiency_check(source, rewritten)
        reasons += structural.reasons
        treatment = treatment_checks(source, rewritten, condition,
                                     identical=same_wording(rewritten, source.original_nl))
        reasons += treatment.reasons
        result = GateResult(ok=None, reasons=reasons,
                            structural={"ok": structural.ok, "reasons": structural.reasons,
                                        "details": structural.details},
                            treatment={"ok": treatment.ok, "reasons": treatment.reasons},
                            skipped=skipped)
        if cheap_only or reasons:
            # Model judges are not spent on a candidate the cheap checks reject.
            result.ok = False if reasons else None
            if cheap_only:
                skipped += ["parser", "semantic"]
            return result

        if self.admission is not None:
            return self._evaluate_admission(source, rewritten, condition, structural, result)

        # The verdict is derived from each judge's own ok/None signal, never from
        # the wording of its reason strings.
        no_verdict = False
        hard_fail = False
        # The parser and the fidelity judge are independent model calls: run them side by side.
        if self.parser is not None and self.validator is not None:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                parser_future = pool.submit(self._run_parser, source, rewritten)
                validator_future = pool.submit(self._run_validator, source, rewritten, condition)
                parser_result, semantic_result = parser_future.result(), validator_future.result()
        else:
            parser_result = self._run_parser(source, rewritten)
            semantic_result = self._run_validator(source, rewritten, condition)
        if parser_result is not None:
            result.parser = parser_result
            if not parser_result["ok"] and (self.parser_abstention == "defer"
                                            and parser_result["error"] == "predicate_not_recoverable"):
                parser_result["abstained"] = True
                skipped.append("parser_abstained")
            elif not parser_result["ok"]:
                no_verdict = True
                reasons.append(f"parser_no_verdict:{parser_result['error']}")
            elif not parser_result["structural_match"]:
                hard_fail = True
                reasons.append("parser_mismatch:" + ",".join(parser_result["mismatched_slots"]))
        elif self.parser is not None:
            skipped.append("parser_predicate_outside_grammar")
        else:
            skipped.append("parser")

        if semantic_result is not None:
            result.semantic = semantic_result
            if semantic_result["ok"] is None:
                no_verdict = True
                reasons += semantic_result["reasons"]
            elif semantic_result["ok"] is False:
                hard_fail = True
                reasons += semantic_result["reasons"]
        else:
            skipped.append("semantic")

        needs_semantic = (structural.details.get("requires_semantic_adjudication")
                          or "parser_abstained" in skipped
                          or (parser_result or {}).get("comparison_basis") in ("original_parse", "act_inventory"))
        if needs_semantic and (semantic_result or {}).get("ok") is None:
            no_verdict = True
            reasons.append("semantic_adjudication_required")
        if ("parser_abstained" in skipped and (semantic_result or {}).get("ok") is True
                and self.second_opinion is not None and not hard_fail):
            verdict = self.second_opinion.validate(source, rewritten, condition=condition)
            spec = getattr(getattr(self.second_opinion, "model", None), "spec", None)
            result.second_opinion = {"ok": verdict.ok, "reasons": list(verdict.reasons),
                                     "rubric_version": (verdict.details or {}).get("rubric_version"),
                                     "handle": getattr(spec, "handle", None), "family": getattr(spec, "family", None)}
            if verdict.ok is not True:
                no_verdict = True
                reasons.append("second_opinion_conflict" if verdict.ok is False else "second_opinion_unavailable")

        if hard_fail:
            result.ok = False
        elif no_verdict:
            result.ok = None
        elif (self.require_parser and "parser" in skipped) or (self.require_validator and "semantic" in skipped):
            result.ok = None
            reasons.append("required_judge_unavailable")
        else:
            result.ok = True
        result.reasons = reasons
        return result

    def provenance(self) -> dict:
        model = self.judge_model
        return {
            "judge_handle": self.judge_handle,
            "judge_model": model.provenance() if model else None,
            "semantic_rubric_version": SEMANTIC_RUBRIC_VERSION,
            "semantic_rubric_sha256": semantic_prompt_sha(),
            "parser_prompt_sha256": sha256_text(parser_system_prompt()),
            "commitment_may_be_indirect": False,
            "validator_max_tokens": getattr(self.validator, "max_tokens", None),
            "parser_max_tokens": getattr(self.parser, "max_tokens", None),
            "parser_abstention": self.parser_abstention,
            "neutral_exposure_floor": self.neutral_exposure_floor,
            "parser_context_words": self.parser_context_words,
            "second_opinion": (getattr(self.second_opinion.model, "identity_record", None)
                               if self.second_opinion is not None else None),
        }


@dataclass
class Attempt:
    index: int
    text: Optional[str]
    gate: Optional[GateResult]
    error: str = ""
    feedback_reasons: List[str] = field(default_factory=list)
    neutral_selection: Optional[dict] = None

    def to_dict(self) -> dict:
        return {"attempt": self.index, "text": self.text,
                "gate": self.gate.to_dict() if self.gate else None, "error": self.error,
                "feedback_reasons": list(self.feedback_reasons),
                "neutral_selection": self.neutral_selection}


STATUS_OK = "ok"
STATUS_TREATMENT_UNAVAILABLE = "treatment_unavailable"           # the rewrites failed the gate
# Neither a pass nor any real failing verdict: generator or judge infrastructure
# (transport errors, unparseable judge output) prevented every attempt.
STATUS_INFRASTRUCTURE_UNAVAILABLE = "infrastructure_unavailable"
STATUS_JUDGE_UNAVAILABLE = STATUS_INFRASTRUCTURE_UNAVAILABLE      # backwards-compatible alias


@dataclass
class GenerationOutcome:
    condition: str
    status: str                        # STATUS_OK | STATUS_TREATMENT_UNAVAILABLE | STATUS_INFRASTRUCTURE_UNAVAILABLE
    text: Optional[str]
    attempts: List[Attempt]
    fallback_to_original: bool = False

    @property
    def first_pass_ok(self) -> bool:
        return bool(self.attempts) and self.attempts[0].gate is not None and self.attempts[0].gate.passed

    @property
    def n_retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> dict:
        return {"condition": self.condition, "status": self.status, "text": self.text,
                "failure_kind": (None if self.status == STATUS_OK else
                    "fidelity_failure" if self.status == STATUS_TREATMENT_UNAVAILABLE else "infrastructure_failure"),
                "first_pass_ok": self.first_pass_ok, "n_attempts": len(self.attempts),
                "n_retries": self.n_retries, "fallback_to_original": self.fallback_to_original,
                "attempts": [a.to_dict() for a in self.attempts]}


def generate_with_gate(rewriter: LanguageProficiencyRewriter, gate: SemanticGate,
                       ctx: RewriteContext, source: SourceRecord, *,
                       max_corrective_retries: int = 2,
                       retry_temperature: Optional[float] = None) -> GenerationOutcome:
    """Gated generation; the rewriter's own temperature is restored whatever the outcome."""
    first = rewriter.temperature
    try:
        return _generate_with_gate(rewriter, gate, ctx, source, max_corrective_retries=max_corrective_retries,
                                   retry_temperature=retry_temperature)
    finally:
        rewriter.temperature = first


def _generate_with_gate(rewriter: LanguageProficiencyRewriter, gate: SemanticGate,
                        ctx: RewriteContext, source: SourceRecord, *,
                        max_corrective_retries: int = 2,
                        retry_temperature: Optional[float] = None) -> GenerationOutcome:
    """Gated generation for one utterance in one condition.

    Low/High: rewrite -> gate -> corrective retry.  Neutral: generate all
    candidates -> cheap-check each -> model-judge the survivors -> choose among
    the passers by the neutrality rule; retry with the failure reasons when
    nothing passes. Every condition is generated independently of the others.
    """
    attempts: List[Attempt] = []
    feedback: Optional[dict] = None
    # Scope verdict reuse to this source and generation call. Exact text keys
    # retain the punctuation and casing seen by the judges.
    neutral_verdicts: Dict[str, GateResult] = {}
    neutral_history: List[str] = []
    condition = rewriter.condition
    if condition == "neutral" and not callable(getattr(gate, "neutral_frequency", None)):
        raise ValueError("Neutral requires a SUBTLEX word-frequency scorer before generation")
    saw_real_failure = False
    first_temperature = rewriter.temperature
    for index in range(1 + max_corrective_retries):
        prior_reasons = list((feedback or {}).get("reasons", []))
        # With retry_temperature, a retry is the same prompt sampled hotter, with no corrective feedback.
        rewriter.temperature = retry_temperature if (retry_temperature is not None and index > 0) else first_temperature
        try:
            if condition == "neutral":
                text, selection, gate_result = _neutral_attempt(
                    rewriter, gate, ctx, source, feedback,
                    verdict_cache=neutral_verdicts, history=neutral_history)
                attempt = Attempt(index, text, gate_result, feedback_reasons=prior_reasons,
                                  neutral_selection=selection)
            else:
                text = rewriter.rewrite(ctx, feedback=feedback)
                gate_result = _evaluate_with_rejudge(gate, source, text, condition)
                attempt = Attempt(index, text, gate_result, feedback_reasons=prior_reasons)
        except Exception as exc:
            # A generation/transport failure is an attempt that produced nothing.
            # It is NOT semantic feedback: the next prompt keeps the previous
            # feedback rather than quoting an exception message as a check.
            attempts.append(Attempt(index, None, None, error=f"{type(exc).__name__}: {exc}",
                                    feedback_reasons=prior_reasons))
            details = getattr(exc, "details", None)
            if isinstance(details, dict) and details.get("retryable") is False:
                break
            continue
        attempts.append(attempt)
        if attempt.gate.passed:
            return GenerationOutcome(condition, STATUS_OK, attempt.text, attempts)
        if attempt.gate.ok is None:
            # No verdict even after re-judging: the judges, not the rewrite, are
            # at fault. Do not feed infrastructure errors back as failure reasons.
            if (attempt.gate.permanent_transport_failure or
                    attempt.gate.semantic.get("dimensions", {}).get("failure_category") == "model_output"):
                break
            continue
        saw_real_failure = True
        if attempt.gate.permanent_transport_failure:
            break
        if retry_temperature is None:
            feedback = {"reasons": list(attempt.gate.reasons), "previous": attempt.text or ""}
    rewriter.temperature = first_temperature
    status = STATUS_TREATMENT_UNAVAILABLE if saw_real_failure else STATUS_INFRASTRUCTURE_UNAVAILABLE
    return GenerationOutcome(condition, status, None, attempts)


def parser_context(source: SourceRecord, *, budget_words: Optional[int] = 6000):
    """Prior public history under a word budget. Priority: the speech this utterance answers,
    then entries of the current phase, then the most recent remaining entries. The kept
    entries are returned in speech order with a record of what was included."""
    history = history_before(source)
    words = [len((entry or "").split()) for entry in history]
    total = sum(words)
    if budget_words is None or total <= budget_words:
        return history, {"budget_words": budget_words, "words": total, "n_included": len(history), "n_dropped": 0,
                         "included_speech_ids": [parse_history_entry(e).get("sid") for e in history]}
    referenced = {(source.referenced_speech or {}).get("speech_id")} - {None}
    phase = context_phase(source)
    def priority(index):
        entry = history[index]
        sid = parse_history_entry(entry).get("sid")
        if sid in referenced:
            return 0
        if phase and f"({phase}," in entry:
            return 1
        return 2
    order = sorted(range(len(history)), key=lambda i: (priority(i), -i))
    kept, used = set(), 0
    for index in order:
        if used + words[index] <= budget_words:
            kept.add(index)
            used += words[index]
    selected = [history[i] for i in range(len(history)) if i in kept]
    return selected, {"budget_words": budget_words, "words": used, "n_included": len(selected),
                      "n_dropped": len(history) - len(selected),
                      "included_speech_ids": [parse_history_entry(e).get("sid") for e in selected]}


def _evaluate_with_rejudge(gate: SemanticGate, source: SourceRecord, text: str,
                           condition: str, *, rejudge: int = 1) -> GateResult:
    """Evaluate; on a no-verdict (judge error) re-judge the SAME text up to ``rejudge`` times."""
    result = gate.evaluate(source, text, condition)
    for _ in range(rejudge):
        if (result.ok is not None or result.permanent_transport_failure or
                result.semantic.get("dimensions", {}).get("failure_category") == "model_output"):
            break
        result = gate.evaluate(source, text, condition)
    return result


def _neutral_attempt(rewriter, gate, ctx, source, feedback, *, verdict_cache=None, history=None):
    """One Neutral round: generate the candidates, cheap-check all of them, model-judge the
    survivors, and choose among the passers by the neutrality rule. Nothing about Low, its
    text or its edit distance enters this step; edit distances are recorded as diagnostics."""
    # The within-round key ignores case and spacing but retains punctuation.
    # Cross-round reuse requires an exact text match to the judged candidate.
    verdict_cache = {} if verdict_cache is None else verdict_cache
    history = [] if history is None else history
    generation_feedback = {**(feedback or {}), "previous_candidates": list(history)}
    unique: Dict[tuple, str] = {}
    for cand in rewriter.generate_candidates(ctx, feedback=generation_feedback):
        unique.setdefault(surface_key(cand), cand)
    candidates = list(unique.values())
    history.extend(c for c in candidates if c not in history)
    reused = {c for c in candidates if c in verdict_cache}
    # Cheap checks on every candidate; model judges only on survivors.
    verdicts: Dict[str, GateResult] = {}
    for cand in candidates:
        verdicts[cand] = verdict_cache.get(cand) or gate.evaluate(source, cand, "neutral", cheap_only=True)
    floor = getattr(gate, "neutral_exposure_floor", 0.0) or 0.0
    distance_of = {c: normalized_token_edit_distance(c, ctx.original_nl) for c in candidates}
    # Near-copies (below the active-paraphrase floor) can never be selected, so they are not
    # sent to the model judges; they are recorded with the floor as their reason.
    for cand in candidates:
        if not verdicts[cand].reasons and distance_of[cand] < floor - 1e-12:
            verdicts[cand] = GateResult(ok=False, reasons=[f"below_exposure_floor:only {100 * distance_of[cand]:.0f}% of the tokens "
                                                           f"changed (floor {100 * floor:.0f}%); reword more of the utterance"],
                                        structural=verdicts[cand].structural, treatment={"ok": False, "reasons": ["below_exposure_floor"]},
                                        skipped=["parser", "semantic"])
    survivors = [c for c in candidates if not verdicts[c].reasons]
    to_judge = [c for c in survivors if c not in reused]
    if to_judge:
        # Every surviving candidate is judged at the same time; each judgment is itself independent.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(to_judge)) as pool:
            futures = {c: pool.submit(_evaluate_with_rejudge, gate, source, c, "neutral") for c in to_judge}
            for c, future in futures.items():
                verdicts[c] = future.result()
    for cand in survivors:
        if verdicts[cand].permanent_transport_failure:
            record = {"candidate_gates": [{"text": c, **verdicts[c].to_dict()} for c in candidates],
                      "n_candidates": len(candidates), "stopped_on_transport_error": True}
            return cand, record, verdicts[cand]
    # A missing verdict remains retryable. A definite verdict is never sampled
    # repeatedly until the judge happens to accept an unchanged candidate.
    verdict_cache.update({c: v for c, v in verdicts.items() if v.ok is not None})
    passers = [c for c in survivors if verdicts[c].passed]
    seed = int(hashlib.sha256(f"{source.source_id}".encode()).hexdigest()[:8], 16)
    frequency = getattr(gate, "neutral_frequency", None)
    if frequency is None:
        raise ValueError("the gate needs a word-frequency scorer for Neutral selection (set_neutral_frequency)")
    if passers:
        selection = select_candidate(passers, ctx.original_nl, seed=seed, frequency=frequency)
        chosen = selection.chosen
        verdict = verdicts[chosen]
    else:
        # Nothing passed. Prefer a candidate with a REAL failure verdict as the
        # thing to report and feed back; only when every judged candidate got no
        # verdict at all is the attempt itself a no-verdict.
        failed = [c for c in candidates if verdicts[c].ok is False]
        pool = failed or candidates
        try:
            selection = select_candidate(pool, ctx.original_nl, seed=seed, frequency=frequency)
            chosen = selection.chosen
        except RuntimeError:
            # Every candidate was dropped (e.g. all copies of the original): keep a
            # complete record anyway, distances included, so to_dict() cannot fail.
            selection = NeutralSelection(
                chosen=pool[0] if pool else "", candidates=list(pool),
                distances=[normalized_token_edit_distance(c, ctx.original_nl) for c in pool],
                rule="fallback_all_dropped", seed=seed)
            chosen = selection.chosen
        verdict = verdicts.get(chosen, GateResult(ok=False, reasons=["no_candidate"]))
    record = selection.to_dict()
    record["candidate_gates"] = [{"text": c, **verdicts[c].to_dict(),
                                  "edit_distance": normalized_token_edit_distance(c, ctx.original_nl),
                                  "reused_verdict": c in reused} for c in candidates]
    record["n_reused_verdicts"] = len(reused)
    record["n_candidates"] = len(candidates)
    record["n_passed_gate"] = len(passers)
    record["exposure_floor"] = floor
    record["n_below_floor"] = sum(1 for c in candidates if any(r.startswith("below_exposure_floor") for r in verdicts[c].reasons))
    record["n_no_verdict"] = sum(1 for c in survivors if verdicts[c].ok is None)
    rewriter.last_selection = selection
    return chosen, record, verdict


def _engine_defaults() -> dict:
    """Log path and identifiers for a rewriter built by ``lurkingwolf.eval.replay_cli``
    or ``lurkingwolf.run_game`` (``--output`` and ``--config-name`` are theirs)."""
    import sys
    argv = sys.argv
    out = {}
    try:
        if "--output" in argv:
            target = Path(argv[argv.index("--output") + 1])
            run_dir = target.parent if target.suffix == ".json" else target
            out["log_path"] = str(run_dir / "rewrites.jsonl")
        if "--config-name" in argv:
            out["branch_id"] = argv[argv.index("--config-name") + 1]
        if "--checkpoint" in argv:
            # .../<experiment root>/<comparison>/baseline/checkpoints/day_N_start.pkl
            ckpt = Path(argv[argv.index("--checkpoint") + 1])
            out["game_id"] = ckpt.resolve().parents[3].name if len(ckpt.resolve().parents) > 3 else None
        elif "--seed" in argv:
            out["game_id"] = argv[argv.index("--seed") + 1]
    except (IndexError, ValueError):
        return {}
    return {k: v for k, v in out.items() if v}


@register
class GatedRewriter(Rewriter):
    """Generate, validate and record each rewritten utterance during a game.

    ``base`` accepts a rewriter, condition or configuration. ``gate`` accepts a
    SemanticGate, judge handle or configuration. ``on_unavailable`` selects an
    exception or a logged return of the original utterance after failed attempts.
    Neutral loads its word-frequency scorer from ``subtlex_path`` or ``SUBTLEX``.
    """

    name = "gated_proficiency"

    def __init__(self, base=None, gate="deepseek-chat", *, llm_client=None,
                 max_corrective_retries: int = 2, on_unavailable: str = "raise",
                 retry_temperature: Optional[float] = None, admission=None,
                 log_path: Optional[str] = None,
                 subtlex_path: Optional[str] = None, spacy_model: str = "en_core_web_sm",
                 frozen_path: Optional[str] = None, generator_handle: Optional[str] = None,
                 game_id: Optional[str] = None, branch_id: Optional[str] = None,
                 source_builder: Callable[[RewriteContext], SourceRecord] = source_from_context,
                 log: Optional[List[dict]] = None, **base_params):
        # Inside a LURK replay or game subprocess the engine never passes these, so take
        # them from that process's own command line: the log lands next to its outputs.
        inferred = _engine_defaults()
        engine_log = not log_path and inferred.get("log_path")
        log_path = log_path or inferred.get("log_path")
        game_id = game_id or inferred.get("game_id")
        branch_id = branch_id or inferred.get("branch_id")
        self.frozen_path, self.generator_handle = frozen_path, generator_handle
        self.game_id, self.branch_id = game_id, branch_id
        frozen = None
        if frozen_path:
            from experiments.language_validation.validation.protocol import enforce_frozen
            frozen = enforce_frozen(Path(frozen_path), check_models=True)
            if not game_id or not branch_id or not log_path:
                raise ValueError("frozen replay requires game_id, branch_id and log_path")
            generator_handle = frozen["models"]["generator"]
            self.generator_handle = generator_handle
            gate = SemanticGate.from_handle(frozen["models"]["judge"], max_tokens=frozen["judge_max_tokens"],
                                            parser_abstention=frozen.get("parser_abstention_policy", "no_verdict"))
            gate.neutral_exposure_floor = frozen["neutral_exposure_floor"]
            gate.set_neutral_frequency(frozen["instruments"]["subtlex_path"], frozen["instruments"]["spacy_model"])
            gate.parser_context_words = frozen["parser_context_words"]
            max_corrective_retries = frozen["max_corrective_retries"]
            params = dict(base.to_config()["params"] if isinstance(base, LanguageProficiencyRewriter)
                          else base.get("params", base) if isinstance(base, dict) else {"condition": base or "low"})
            if params.get("condition", "low") not in frozen["conditions"]:
                raise ValueError("replay condition is outside the frozen design")
            if params.get("applies_to") or params.get("applies_to_predicates"):
                raise ValueError("frozen persistent treatment must cover every public speech predicate")
            params.update(temperature=frozen["generation_temperature"], max_tokens=frozen["generation_max_tokens"],
                          n_neutral_candidates=frozen["n_neutral_candidates"])
            base = params
        if generator_handle:
            from experiments.language_validation.rewrite.__main__ import JudgeModelClient
            generator_model = ValidationModel.resolve(generator_handle)
            llm_client = JudgeModelClient(generator_model)
            base_params["model"] = generator_model.spec.model
        if on_unavailable not in ("original", "raise"):
            raise ValueError("on_unavailable must be 'original' or 'raise'")
        if on_unavailable == "original" and not log_path:
            raise ValueError("on_unavailable='original' requires log_path: a fallback that is not "
                             "written to disk is an untreated stimulus under a treatment label")
        if isinstance(base, LanguageProficiencyRewriter):
            if llm_client is not None and (base.llm_client is None or generator_handle):
                base.llm_client = llm_client
            if generator_handle:
                base.model = generator_model.spec.model
        else:
            params = dict(base_params)
            if isinstance(base, dict):
                params.update(base.get("params", base))
            elif base is not None:
                params["condition"] = str(base)          # ``base="low"`` shorthand
            if generator_handle:
                params["model"] = generator_model.spec.model
            base = LanguageProficiencyRewriter(llm_client=llm_client, **params)
        if isinstance(gate, SemanticGate):
            pass
        elif isinstance(gate, dict):
            gate = SemanticGate.from_config(gate)
        else:
            gate = SemanticGate.from_handle(str(gate))
        if base.condition == "neutral" and not callable(gate.neutral_frequency):
            from components.judges.judge_model import _dotenv_values
            path = subtlex_path or os.getenv("SUBTLEX") or _dotenv_values().get("SUBTLEX")
            if not path:
                raise ValueError("Neutral requires subtlex_path, or SUBTLEX in the environment or .env")
            gate.set_neutral_frequency(path, spacy_model)
            if not callable(gate.neutral_frequency):
                raise ValueError("Neutral requires a SUBTLEX word-frequency scorer")
        self.base = base
        self.gate = gate
        if admission is not None:
            gate.admission = admission
        self.retry_temperature = retry_temperature
        self.log_path = log_path
        self.max_corrective_retries = int(max_corrective_retries)
        self.on_unavailable = on_unavailable
        self.source_builder = source_builder
        self.log: List[dict] = log if log is not None else []
        if engine_log and Path(log_path).exists():
            # The engine restarts an interrupted replay from its checkpoint, so the log of
            # this subprocess holds this attempt only.
            Path(log_path).unlink()
        self.sequence = sum(1 for _ in Path(log_path).open()) if log_path and Path(log_path).exists() else 0
        self.frozen = frozen

    @property
    def condition(self) -> str:
        return self.base.condition

    @property
    def applies_to_predicates(self):
        return self.base.applies_to_predicates

    def rewrite(self, ctx: RewriteContext) -> str:
        source = self.source_builder(ctx)
        self.sequence += 1
        source.game_id = self.game_id
        source.source_id = f"{self.game_id}:{self.branch_id}:{self.sequence}"
        outcome = generate_with_gate(self.base, self.gate, ctx, source,
                                     max_corrective_retries=self.max_corrective_retries,
                                     retry_temperature=self.retry_temperature)
        entry = {"game_id": self.game_id, "branch_id": self.branch_id, "source_id": source.source_id,
                 "sequence": self.sequence, "speaker": ctx.speaker_id, "phase": ctx.phase,
                 "predicate": ctx.predicate, "original_nl": ctx.original_nl, "original_text": ctx.original_nl,
                 "rewritten_text": outcome.text, "public_history": history_before(source),
                 "gate_provenance": self.gate.provenance(),
                 "generator_handle": self.generator_handle, "generator_temperature": self.base.temperature,
                 "prompt_template_sha256": self.base.template_sha256(candidates=self.condition == "neutral"),
                 **outcome.to_dict()}
        if self.frozen:
            from experiments.language_validation.rewrite.runtime import digest
            entry["frozen_sha256"] = digest(self.frozen)
        if outcome.status == STATUS_OK:
            self._record(entry)
            return outcome.text or ""
        if self.on_unavailable == "raise":
            entry["fallback_to_original"] = False
            self._record(entry)
            last = outcome.attempts[-1] if outcome.attempts else None
            detail = (last.gate.reasons if last and last.gate else (last.error if last else "no attempts"))
            raise TreatmentUnavailable(
                f"{self.condition} rewrite {outcome.status} for {ctx.speaker_id} {ctx.predicate} after "
                f"{len(outcome.attempts)} attempts: {detail}")
        entry["fallback_to_original"] = True
        self._record(entry)
        return ctx.original_nl

    def _record(self, entry: dict) -> None:
        """Keep the entry in memory and, when a path is set, append it to disk NOW."""
        self.log.append(entry)
        if self.log_path:
            import json
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def to_config(self) -> dict:
        """Serialize the rewriter, gate policy and run settings."""
        return {"class": self.__class__.__name__,
                "params": {"base": self.base.to_config(),
                           "gate": self.gate.to_config(),
                           "max_corrective_retries": self.max_corrective_retries,
                           "on_unavailable": self.on_unavailable,
                           "log_path": self.log_path, "frozen_path": self.frozen_path,
                           "generator_handle": self.generator_handle,
                           "game_id": self.game_id, "branch_id": self.branch_id}}


def failure_cause(row):
    """Describe the recorded failures."""
    if row.get("status") == STATUS_OK:
        return "accepted"
    attempts = row.get("attempts") or []
    gate = (attempts[-1].get("gate") or {}) if attempts else {}
    reasons = gate.get("reasons", [])
    if reasons and set(reasons) == {"identical_to_original"}:
        return "unchanged_candidate"
    if reasons and all(r.startswith("below_exposure_floor") for r in reasons):
        return "near_copy_only"
    if any(r.startswith(("english_validator_missing:", "parser_no_verdict:predicate_not_recoverable")) for r in reasons):
        return "evaluation_output_failure"
    if row.get("status") == STATUS_TREATMENT_UNAVAILABLE:
        return "recorded_gate_rejection"
    return "generation_or_transport_failure"


def generation_qc(outcomes: List[dict]) -> Dict[str, dict]:
    """First-pass, retry, unavailable and fallback rates per condition."""
    out: Dict[str, dict] = {}
    for condition in sorted({o["condition"] for o in outcomes}):
        rows = [o for o in outcomes if o["condition"] == condition]
        n = len(rows)
        out[condition] = {
            "n": n,
            "final_success": sum(o.get("status") == STATUS_OK for o in rows),
            "final_success_rate": sum(o.get("status") == STATUS_OK for o in rows) / n if n else None,
            "own_failed_attempts": sum(o.get("status") != STATUS_OK and bool(o.get("n_attempts")) for o in rows),
            "first_pass_success": sum(1 for o in rows if o.get("first_pass_ok")),
            "first_pass_success_rate": sum(1 for o in rows if o.get("first_pass_ok")) / n if n else None,
            "needed_retry": sum(1 for o in rows if o.get("n_retries", 0) > 0),
            "retry_rate": sum(1 for o in rows if o.get("n_retries", 0) > 0) / n if n else None,
            "treatment_unavailable": sum(1 for o in rows if o.get("status") == STATUS_TREATMENT_UNAVAILABLE),
            "treatment_unavailable_rate": (sum(1 for o in rows if o.get("status") == STATUS_TREATMENT_UNAVAILABLE) / n
                                           if n else None),
            "infrastructure_unavailable": sum(1 for o in rows if o.get("status") == STATUS_INFRASTRUCTURE_UNAVAILABLE),
            "accepted_with_parser_abstention": sum(
                1 for o in rows if o.get("status") == STATUS_OK and "parser_abstained" in
                (((o.get("attempts") or [{}])[-1].get("gate") or {}).get("skipped") or [])),
            "accepted_by_parser_status": dict(Counter(
                ((o.get("attempts") or [{}])[-1].get("gate") or {}).get("parser_status") or "unknown"
                for o in rows if o.get("status") == STATUS_OK)),
            "second_opinion": dict(Counter(
                {True: "agree", False: "conflict", None: "unavailable"}[((a.get("gate") or {}).get("second_opinion") or {}).get("ok")]
                for o in rows for a in (o.get("attempts") or []) if (a.get("gate") or {}).get("second_opinion"))),
            "fallback_to_original": sum(1 for o in rows if o.get("fallback_to_original")),
            "mean_attempts": (sum(o.get("n_attempts", 0) for o in rows) / n) if n else None,
        }
    return out
