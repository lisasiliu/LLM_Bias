"""Semantic gate (structural, treatment, parser and fidelity checks with corrective retries), GatedRewriter and generation QC."""
from __future__ import annotations
import re
from math import isclose
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from components.helpers import CheckResult, SourceRecord, parse_history_entry
from experiments.proficiency_validation.rewrite.fidelity import SEMANTIC_RUBRIC_VERSION, EnglishFidelityValidator, semantic_prompt_sha
from experiments.proficiency_validation.rewrite.runtime import ValidationModel
from components.judges.inverse_predicate_parser import PARSEABLE_PREDICATES, InversePredicateParser, build_system_prompt as parser_system_prompt, compare, context_phase
from components.api_usage import UsageLedger
from components.judges.judge_model import Completion, JudgeModel
from experiments.proficiency_validation.rewrite.fidelity import proficiency_check
from components.metrics.edit_distance import normalized_token_edit_distance
from components.metrics.text_normalization import same_wording, surface_key
from components.rewriters.language_rewriter import LanguageProficiencyRewriter, NeutralSelection, select_candidate
from components.rewriters.politeness_rewriter import predicate_definition, predicate_entry
from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.speech_nl import parse_predicate
from experiments.proficiency_validation.rewrite.runtime import history_before, sha256_text


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
    ref = None
    ref_arg = args[-1].strip() if args else ""
    if re.fullmatch(r"s\d+", ref_arg):
        for entry in ctx.public_history or []:
            item = parse_history_entry(entry)
            if item.get("sid") == ref_arg:
                ref = {"speech_id": item["sid"], "speaker": item["speaker"],
                       "predicate": item["pred"], "text": item.get("text", "")}
                break
    return SourceRecord(
        source_id=f"ctx:{ctx.speaker_id}:{ctx.predicate}",
        predicate_name=name, predicate_text=ctx.predicate, predicate_args=args,
        predicate_definition=predicate_definition(name),
        nl_hint=predicate_entry(name).get("nl_hint", ""),
        original_nl=ctx.original_nl, phase=ctx.phase, speaker_id=ctx.speaker_id,
        speaker_role_claimed=ctx.speaker_role_claimed,
        public_history=list(ctx.public_history or []), referenced_speech=ref,
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


@dataclass
class GateResult:
    ok: Optional[bool]                 # True pass, False fail, None no verdict
    reasons: List[str] = field(default_factory=list)
    structural: Dict[str, Any] = field(default_factory=dict)
    treatment: Dict[str, Any] = field(default_factory=dict)
    parser: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.ok is True

    @property
    def permanent_transport_failure(self) -> bool:
        errors = [self.parser.get("transport"), self.semantic.get("dimensions", {}).get("transport")]
        return any(isinstance(e, dict) and e.get("retryable") is False for e in errors)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reasons": list(self.reasons), "structural": self.structural,
                "treatment": self.treatment, "parser": self.parser, "semantic": self.semantic,
                "skipped": list(self.skipped)}


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
                 judge_handle: Optional[str] = None):
        self.parser = parser
        self.validator = validator
        self.require_parser = require_parser
        self.require_validator = require_validator
        self.judge_handle = judge_handle
        self._original_parses = {}
        self._original_inventories = {}
        self.neutral_edit_tolerance = None

    @classmethod
    def from_handle(cls, handle: str = "deepseek-chat", *, ledger: Optional[UsageLedger] = None,
                    max_tokens: int = 3072, **kwargs) -> "SemanticGate":
        """Parser and validator on the same model, each booking its own usage purpose."""
        parser_model = ValidationModel.resolve(handle, ledger=ledger, purpose="inverse_parser")
        validator_model = ValidationModel.resolve(handle, ledger=ledger, purpose="semantic_validator")
        return cls(InversePredicateParser(parser_model, max_tokens=max_tokens),
                   EnglishFidelityValidator(validator_model, max_tokens=max_tokens),
                   judge_handle=handle, **kwargs)

    # ── individual judges ──

    def _run_parser(self, source: SourceRecord, rewritten: str) -> Optional[Dict[str, Any]]:
        """Parser verdict, or ``None`` when the predicate is outside its grammar.

        The parser is shown ONLY the history before this utterance: the full
        game history a source carries includes the entry that realizes the
        utterance itself, which would hand the parser the answer.
        """
        if self.parser is None or source.predicate_name not in PARSEABLE_PREDICATES:
            return None
        history = history_before(source)
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
            from experiments.proficiency_validation.rewrite.fidelity import extract_inventory
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
                "original_parse": baseline.to_dict(), "original_comparison": original_comparison,
                "candidate_comparison": comparison,
                "error": parsed.error or ("predicate_not_recoverable" if not candidate_usable else ""),
                "provenance": parsed.provenance,
                "transport": getattr(getattr(self.parser, "model", None), "last_error", None),
                "history_entries": len(history_before(source)), "source_id": source.source_id}

    def _run_validator(self, source: SourceRecord, rewritten: str, condition=None) -> Optional[Dict[str, Any]]:
        if self.validator is None:
            return None
        kwargs = {"condition": condition} if getattr(self.validator, "supports_condition", False) else {}
        verdict = self.validator.validate(source, rewritten, commitment_may_be_indirect=False, **kwargs)
        return {"ok": verdict.ok, "reasons": verdict.reasons,
                "dimensions": {k: v for k, v in verdict.details.items()
                               if k not in ("validator", "dimensions_asked")},
                "rubric_version": SEMANTIC_RUBRIC_VERSION}

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

        # The verdict is derived from each judge's own ok/None signal, never from
        # the wording of its reason strings.
        no_verdict = False
        hard_fail = False
        parser_result = self._run_parser(source, rewritten)
        if parser_result is not None:
            result.parser = parser_result
            if not parser_result["ok"]:
                no_verdict = True
                reasons.append(f"parser_no_verdict:{parser_result['error']}")
            elif not parser_result["structural_match"]:
                hard_fail = True
                reasons.append("parser_mismatch:" + ",".join(parser_result["mismatched_slots"]))
        elif self.parser is not None:
            skipped.append("parser_predicate_outside_grammar")
        else:
            skipped.append("parser")

        semantic_result = self._run_validator(source, rewritten, condition)
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
                          or (parser_result or {}).get("comparison_basis") in ("original_parse", "act_inventory"))
        if needs_semantic and (semantic_result or {}).get("ok") is None:
            no_verdict = True
            reasons.append("semantic_adjudication_required")

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
            "neutral_edit_tolerance": self.neutral_edit_tolerance,
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
    target_edit_distance: Optional[float] = None
    anchor_outcomes: Dict[str, dict] = field(default_factory=dict)

    @property
    def first_pass_ok(self) -> bool:
        return bool(self.attempts) and self.attempts[0].gate is not None and self.attempts[0].gate.passed

    @property
    def n_retries(self) -> int:
        return max(0, len(self.attempts) - 1)

    def to_dict(self) -> dict:
        last_gate = self.attempts[-1].gate if self.attempts else None
        matching_failure = bool(last_gate and any(
            r.startswith("neutral_edit_distance_mismatch") for r in last_gate.reasons))
        return {"condition": self.condition, "status": self.status, "text": self.text,
                "failure_kind": (None if self.status == STATUS_OK else
                    "anchor_unavailable" if self.anchor_outcomes and not self.attempts else
                    "control_matching_failure" if matching_failure else
                    "fidelity_failure" if self.status == STATUS_TREATMENT_UNAVAILABLE else "infrastructure_failure"),
                "first_pass_ok": self.first_pass_ok, "n_attempts": len(self.attempts),
                "n_retries": self.n_retries, "fallback_to_original": self.fallback_to_original,
                "target_edit_distance": self.target_edit_distance,
                "anchor_outcomes": self.anchor_outcomes,
                "attempts": [a.to_dict() for a in self.attempts]}


def generate_with_gate(rewriter: LanguageProficiencyRewriter, gate: SemanticGate,
                       ctx: RewriteContext, source: SourceRecord, *,
                       max_corrective_retries: int = 2,
                       target_edit_distance: Optional[float] = None,
                       anchor_outcomes: Optional[Dict[str, dict]] = None) -> GenerationOutcome:
    """Gated generation for one utterance in one condition.

    Low/High: rewrite -> gate -> corrective retry.  Neutral: generate all
    candidates -> cheap-check each -> model-judge the survivors -> choose, among
    the passers, the one closest to the target edit distance; retry
    with the closest failure's reasons when nothing passes.
    """
    attempts: List[Attempt] = []
    feedback: Optional[dict] = None
    # Scope verdict reuse to this source and generation call. Exact text keys
    # retain the punctuation and casing seen by the judges.
    neutral_verdicts: Dict[str, GateResult] = {}
    neutral_history: List[str] = []
    condition = rewriter.condition
    saw_real_failure = False
    need_anchor = condition == "neutral" and target_edit_distance is None
    anchors = dict(anchor_outcomes or {})
    if need_anchor:
        if len(rewriter.neutral_anchor_conditions) != 1:
            raise ValueError("Neutral requires one treatment anchor; run Low and High with separate controls")
        anchors = {c: anchors[c] for c in rewriter.neutral_anchor_conditions if c in anchors}
        for anchor_condition in rewriter.neutral_anchor_conditions:
            if anchor_condition not in anchors:
                params = rewriter.to_config()["params"]
                params["condition"] = anchor_condition
                anchor_rw = LanguageProficiencyRewriter(llm_client=rewriter.llm_client, **params)
                anchor = generate_with_gate(anchor_rw, gate, ctx, source,
                                            max_corrective_retries=max_corrective_retries)
                anchors[anchor_condition] = anchor.to_dict()
            if anchors[anchor_condition]["status"] != STATUS_OK:
                return GenerationOutcome(condition, anchors[anchor_condition]["status"], None, [],
                                         anchor_outcomes=anchors)
        anchor = anchors[rewriter.neutral_anchor_conditions[0]]
        target_edit_distance = normalized_token_edit_distance(anchor["text"], ctx.original_nl)
        need_anchor = False
    for index in range(1 + max_corrective_retries):
        prior_reasons = list((feedback or {}).get("reasons", []))
        try:
            if condition == "neutral":
                text, selection, gate_result = _neutral_attempt(
                    rewriter, gate, ctx, source, feedback, target_edit_distance,
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
            return GenerationOutcome(condition, STATUS_OK, attempt.text, attempts,
                                     target_edit_distance=target_edit_distance, anchor_outcomes=anchors)
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
        feedback = {"reasons": list(attempt.gate.reasons), "previous": attempt.text or ""}
    status = STATUS_TREATMENT_UNAVAILABLE if saw_real_failure else STATUS_INFRASTRUCTURE_UNAVAILABLE
    return GenerationOutcome(condition, status, None, attempts,
                             target_edit_distance=target_edit_distance, anchor_outcomes=anchors)


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


def _neutral_attempt(rewriter, gate, ctx, source, feedback, target, *,
                     verdict_cache=None, history=None):
    # The within-round key ignores case and spacing but retains punctuation.
    # Cross-round reuse requires an exact text match to the judged candidate.
    verdict_cache = {} if verdict_cache is None else verdict_cache
    history = [] if history is None else history
    previous_passers = [c for c, v in verdict_cache.items() if v.passed]
    best = select_candidate(previous_passers, ctx.original_nl, target).chosen if previous_passers else None
    unique: Dict[tuple, str] = {}
    generation_feedback = {**(feedback or {}), "target_edit_distance": target,
                           "edit_tolerance": getattr(gate, "neutral_edit_tolerance", None),
                           "best_candidate": best, "previous_candidates": list(history)}
    for cand in rewriter.generate_candidates(ctx, feedback=generation_feedback):
        unique.setdefault(surface_key(cand), cand)
    candidates = list(unique.values())
    history.extend(c for c in candidates if c not in history)
    reused = {c for c in candidates if c in verdict_cache}
    # Cheap checks on every candidate; model judges only on survivors.
    verdicts: Dict[str, GateResult] = {}
    for cand in candidates:
        verdicts[cand] = verdict_cache.get(cand) or gate.evaluate(source, cand, "neutral", cheap_only=True)
    survivors = [c for c in candidates if not verdicts[c].reasons]
    for cand in survivors:
        if cand not in reused:
            verdicts[cand] = _evaluate_with_rejudge(gate, source, cand, "neutral")
        if verdicts[cand].permanent_transport_failure:
            record = {"candidate_gates": [{"text": c, **verdicts[c].to_dict()} for c in candidates],
                      "n_candidates": len(candidates), "stopped_on_transport_error": True}
            return cand, record, verdicts[cand]
    # A missing verdict remains retryable. A definite verdict is never sampled
    # repeatedly until the judge happens to accept an unchanged candidate.
    verdict_cache.update({c: v for c, v in verdicts.items() if v.ok is not None})
    passers = [c for c in survivors if verdicts[c].passed]
    pool = passers + ([best] if best is not None and best not in passers else [])
    if pool:
        selection = select_candidate(pool, ctx.original_nl, target)
        chosen = selection.chosen
        verdict = verdict_cache[chosen]
        tolerance = getattr(gate, "neutral_edit_tolerance", None)
        gap = abs(normalized_token_edit_distance(chosen, ctx.original_nl) - target) if target is not None else None
        if (tolerance is not None and gap is not None and gap > tolerance
                and not isclose(gap, tolerance, rel_tol=0., abs_tol=1e-12)):
            verdict = GateResult(ok=False,
                reasons=[f"neutral_edit_distance_mismatch:gap={gap:.6f},tolerance={tolerance}"],
                structural=verdict.structural, parser=verdict.parser, semantic=verdict.semantic,
                treatment={"ok": False, "reasons": ["neutral_edit_distance_mismatch"]})
    else:
        # Nothing passed. Prefer a candidate with a REAL failure verdict as the
        # thing to report and feed back; only when every judged candidate got no
        # verdict at all is the attempt itself a no-verdict.
        failed = [c for c in candidates if verdicts[c].ok is False]
        pool = failed or candidates
        try:
            selection = select_candidate(pool, ctx.original_nl, target)
            chosen = selection.chosen
        except RuntimeError:
            # Every candidate was dropped (e.g. all copies of the original): keep a
            # complete record anyway, distances included, so to_dict() cannot fail.
            selection = NeutralSelection(
                chosen=pool[0] if pool else "", candidates=list(pool),
                distances=[normalized_token_edit_distance(c, ctx.original_nl) for c in pool],
                target_distance=target, rule="fallback_all_dropped")
            chosen = selection.chosen
        verdict = verdicts.get(chosen, GateResult(ok=False, reasons=["no_candidate"]))
    record = selection.to_dict()
    record["candidate_gates"] = [{"text": c, **verdicts[c].to_dict(),
                                  "edit_distance": normalized_token_edit_distance(c, ctx.original_nl),
                                  "reused_verdict": c in reused} for c in candidates]
    record["carried_candidate"] = ({"text": best, **verdict_cache[best].to_dict()}
                                    if best is not None and best not in candidates else None)
    record["edit_tolerance"] = getattr(gate, "neutral_edit_tolerance", None)
    record["n_reused_verdicts"] = len(reused)
    record["n_candidates"] = len(candidates)
    record["n_passed_gate"] = len(passers)
    record["n_no_verdict"] = sum(1 for c in survivors if verdicts[c].ok is None)
    rewriter.last_selection = selection
    return chosen, record, verdict


@register
class GatedRewriter(Rewriter):
    """Runtime wrapper: base rewriter + gate + retry/fallback policy + log.

    Registered under ``gated_proficiency`` so the engine's configuration paths
    (``--rewriter gated_proficiency --rewriter-params '{...}'``, YAML
    ``rewriter: GatedRewriter``, ``Rewriter.from_config``) build the GATED
    rewriter, never the bare one by accident. ``base`` may be a rewriter or its
    config/params dict; ``gate`` may be a ``SemanticGate``, a judge handle or a
    dict with ``judge_handle``. ``llm_client`` is forwarded to the base rewriter.

    The default policy FAILS CLOSED (``on_unavailable="raise"``): untreated text
    is never returned under a treatment label unless the operator explicitly
    asks for ``"original"``, and then only with a ``log_path`` so that every
    fallback is written to disk as it happens -- the engine itself records only
    the returned string.
    """

    name = "gated_proficiency"

    def __init__(self, base=None, gate="deepseek-chat", *, llm_client=None,
                 max_corrective_retries: int = 2, on_unavailable: str = "raise",
                 log_path: Optional[str] = None,
                 frozen_path: Optional[str] = None, generator_handle: Optional[str] = None,
                 game_id: Optional[str] = None, branch_id: Optional[str] = None,
                 source_builder: Callable[[RewriteContext], SourceRecord] = source_from_context,
                 log: Optional[List[dict]] = None, **base_params):
        self.frozen_path, self.generator_handle = frozen_path, generator_handle
        self.game_id, self.branch_id = game_id, branch_id
        frozen = None
        if frozen_path:
            from experiments.proficiency_validation.validation.protocol import enforce_frozen
            frozen = enforce_frozen(Path(frozen_path), check_models=True)
            if not game_id or not branch_id or not log_path:
                raise ValueError("frozen replay requires game_id, branch_id and log_path")
            generator_handle = frozen["models"]["generator"]
            self.generator_handle = generator_handle
            gate = SemanticGate.from_handle(frozen["models"]["judge"], max_tokens=frozen["judge_max_tokens"])
            gate.neutral_edit_tolerance = frozen["neutral_edit_distance_tolerance"]
            max_corrective_retries = frozen["max_corrective_retries"]
            params = dict(base.to_config()["params"] if isinstance(base, LanguageProficiencyRewriter)
                          else base.get("params", base) if isinstance(base, dict) else {"condition": base or "low"})
            if params.get("condition", "low") not in frozen["conditions"]:
                raise ValueError("replay condition is outside the frozen design")
            if params.get("applies_to") or params.get("applies_to_predicates"):
                raise ValueError("frozen persistent treatment must cover every public speech predicate")
            params.update(temperature=frozen["generation_temperature"], max_tokens=frozen["generation_max_tokens"],
                          n_neutral_candidates=frozen["n_neutral_candidates"],
                          neutral_anchor_conditions=[frozen["neutral_anchor_condition"]],
                          target_edit_distance=None)
            base = params
        if generator_handle:
            from experiments.proficiency_validation.rewrite.__main__ import JudgeModelClient
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
            handle = gate.get("judge_handle")
            if not handle:
                raise ValueError("gate config has no judge_handle; refusing to substitute a default judge")
            recorded = gate.get("semantic_rubric_version")
            if recorded and recorded != SEMANTIC_RUBRIC_VERSION:
                raise ValueError(f"gate config was frozen with semantic rubric {recorded!r}; the code "
                                 f"now has {SEMANTIC_RUBRIC_VERSION!r}")
            gate = SemanticGate.from_handle(handle)
        else:
            gate = SemanticGate.from_handle(str(gate))
        self.base = base
        self.gate = gate
        self.log_path = log_path
        self.max_corrective_retries = int(max_corrective_retries)
        self.on_unavailable = on_unavailable
        self.source_builder = source_builder
        self.log: List[dict] = log if log is not None else []
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
                                     max_corrective_retries=self.max_corrective_retries)
        entry = {"game_id": self.game_id, "branch_id": self.branch_id, "source_id": source.source_id,
                 "sequence": self.sequence, "speaker": ctx.speaker_id, "phase": ctx.phase,
                 "predicate": ctx.predicate, "original_nl": ctx.original_nl, "original_text": ctx.original_nl,
                 "rewritten_text": outcome.text, "public_history": history_before(source),
                 "gate_provenance": self.gate.provenance(),
                 "generator_handle": self.generator_handle, "generator_temperature": self.base.temperature,
                 "prompt_template_sha256": self.base.template_sha256(candidates=self.condition == "neutral"),
                 **outcome.to_dict()}
        if self.frozen:
            from experiments.proficiency_validation.rewrite.runtime import digest
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
        """Reconstructible by ``Rewriter.from_config``: ``gate`` is the judge handle."""
        return {"class": self.__class__.__name__,
                "params": {"base": self.base.to_config(),
                           "gate": {"judge_handle": self.gate.judge_handle,
                                    "semantic_rubric_version": SEMANTIC_RUBRIC_VERSION},
                           "max_corrective_retries": self.max_corrective_retries,
                           "on_unavailable": self.on_unavailable,
                           "log_path": self.log_path, "frozen_path": self.frozen_path,
                           "generator_handle": self.generator_handle,
                           "game_id": self.game_id, "branch_id": self.branch_id}}


def failure_cause(row):
    """Describe recorded failures without changing archived acceptance decisions."""
    if row.get("status") == STATUS_OK:
        return "accepted"
    attempts = row.get("attempts") or []
    if row.get("anchor_outcomes") and not attempts:
        return "anchor_unavailable"
    gate = (attempts[-1].get("gate") or {}) if attempts else {}
    reasons = gate.get("reasons", [])
    if any(r.startswith("neutral_edit_distance_mismatch") for r in reasons):
        return "control_matching_failure"
    if reasons and set(reasons) == {"identical_to_original"}:
        return "unchanged_candidate"
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
            "anchor_blocked": sum(o.get("status") != STATUS_OK and bool(o.get("anchor_outcomes"))
                                  and not o.get("n_attempts") for o in rows),
            "own_failed_attempts": sum(o.get("status") != STATUS_OK and bool(o.get("n_attempts")) for o in rows),
            "first_pass_success": sum(1 for o in rows if o.get("first_pass_ok")),
            "first_pass_success_rate": sum(1 for o in rows if o.get("first_pass_ok")) / n if n else None,
            "needed_retry": sum(1 for o in rows if o.get("n_retries", 0) > 0),
            "retry_rate": sum(1 for o in rows if o.get("n_retries", 0) > 0) / n if n else None,
            "treatment_unavailable": sum(1 for o in rows if o.get("status") == STATUS_TREATMENT_UNAVAILABLE),
            "treatment_unavailable_rate": (sum(1 for o in rows if o.get("status") == STATUS_TREATMENT_UNAVAILABLE) / n
                                           if n else None),
            "infrastructure_unavailable": sum(1 for o in rows if o.get("status") == STATUS_INFRASTRUCTURE_UNAVAILABLE),
            "fallback_to_original": sum(1 for o in rows if o.get("fallback_to_original")),
            "mean_attempts": (sum(o.get("n_attempts", 0) for o in rows) / n) if n else None,
        }
    return out
