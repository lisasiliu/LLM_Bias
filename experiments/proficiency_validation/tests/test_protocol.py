import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from components.helpers import write_jsonl
from components.judges.judge_model import Completion, JudgeModelError
from components.stats import cluster_bootstrap, tost_equivalence, proportion_bootstrap
from components.metrics.grammar import ColaScorer
from experiments.proficiency_validation.validation.analysis import listener_analysis, frozen_criteria, verdict, integrity_checks, paired_clusters
from experiments.proficiency_validation.rewrite.runtime import PRIMARY_RANGE_METRICS, history_before
from experiments.proficiency_validation.rewrite.fidelity import EnglishFidelityValidator, DIMENSIONS
from experiments.proficiency_validation.validation.protocol import frozen_hashes
from experiments.proficiency_validation.rewrite.runtime import ValidationModel, completion_url
from experiments.proficiency_validation.validation.protocol import validate_design
from experiments.proficiency_validation.rewrite.runtime import record_artifact, artifact_provenance
from experiments.proficiency_validation.rewrite.gate import SemanticGate, generate_with_gate
from components.rewriters.language_rewriter import LanguageProficiencyRewriter
from experiments.proficiency_validation.tests.test_offline import source, FakeClient
from components.helpers import build_context


def _constant_frequency(text):
    return 0.0

def design(treatment="low"):
    """Synthetic decisions for tests only, not proposed study thresholds."""
    metrics = list(PRIMARY_RANGE_METRICS)
    handles = ["generator", "judge", "listener1", "listener2", "listener3"]
    identities = {h: {"handle": h, "model": h, "family": h, "revision": "a" * 40,
        "tokenizer_revision": "b" * 40, "serving_version": "test-version", "precision": "float32",
        "chat_template": "tokenizer-default", "extra_body": {}} for h in handles}
    splits = {stage: [{"bundle_id": f"{stage}_g{i}:P1", "game_id": f"{stage}_g{i}", "speaker": "P1",
                       "source_ids": [f"{stage}_s{i}"], "window": [150, None], "selection": "whole_game"} for i in range(6 if stage == "pilot" else 4)]
              for stage in ("pilot", "set_a", "set_b")}
    return {"schema_version": 7, "cefr": {"enabled": False}, "validation_scope": "automated", "conditions": ["original", "neutral", treatment],
        "perception_constructs": ["range"],
        "estimand": "equal_game_weight", "treatment_condition": treatment, "neutral_generation": "independent",
        "parser_abstention_policy": "no_verdict", "neutral_exposure_floor": .05, "parser_context_words": 6000,
        "primary_range_metrics": metrics, "range_effect_thresholds": {c: {m: .1 for m in metrics}
            for c in ("low_vs_neutral", "high_vs_neutral")},
        "equivalence_bounds": {m: .1 for m in metrics + ["grammar_errors_per_100_words", "cola_mean"]},
        "noninferiority_margins": {"grammar_errors_per_100_words": .5, "cola_mean": .05},
        "noninferiority_rationale": "synthetic test margins, not a study decision",
        "bound_rationale": {m: "synthetic test rationale, not a study decision" for m in metrics + ["grammar_errors_per_100_words", "cola_mean"]},
        "min_games": 4, "min_bundles": 4, "min_valid_fraction": 1., "min_subtlex_coverage": .9,
        "listener_equivalence_margin": .1, "listener_min_effect": .1, "position_bias_margin": .2,
        "min_treatment_availability": .9, "max_differential_attrition": .05,
        "listener_effect_rationale": "synthetic meaningful separation, not an empirical choice",
        "availability_rationale": "synthetic availability limits, not an empirical choice",
        "mattr_window": 50, "max_bundles_per_game": 1, "n_neutral_candidates": 2, "max_corrective_retries": 2,
        "generation_temperature": 0., "generation_max_tokens": 400, "judge_max_tokens": 800, "probe_max_tokens": 400,
        "bundle_window_words": [150, None], "bundle_selection": "whole_game",
        "power_rationale": "test data are not an empirical power calculation",
        "listener_margin_rationale": "test margin, not a recommended empirical choice",
        "absolute_quality_rationale": "test limits, not recommended empirical thresholds",
        "absolute_quality": {"grammar_errors_max": .5, "cola_mean_min": .8},
        "statistics": {"n_boot": 1000, "n_perm": 1000, "seed": 101},
        "instruments": {"spacy_model": "en_core_web_sm", "spacy_model_version": "test", "spacy_version": "test",
            "languagetool_version": "LanguageTool test", "languagetool_language": "en-US", "cola_model": "test-cola",
            "cola_commit_hash": "c"*40, "cola_acceptable_index": 1, "cola_max_length": 512,
            "cola_label_rationale": "test metadata with an explicitly verified label", "subtlex_sha256": "d"*64, "subtlex_path": "subtlex.txt",
            "subtlex_measure": "Lg10CD", "include_domain_terms": False},
        "models": {"generator": "generator", "judge": "judge", "listeners": handles[2:],
                   "identities": identities, "behavioral_families": ["behavior"]}, "splits": splits,
        "sources_sha256": {s: "e"*64 for s in splits}}


class StatisticalBoundaryTests(unittest.TestCase):
    def test_single_game_cannot_confirm_any_interval(self):
        self.assertIsNone(cluster_bootstrap({"g": [1, 1]}, n_boot=30)["ci_low"])
        self.assertIsNone(tost_equivalence({"g": [0]}, .1, n_boot=30)["equivalent"])
        rows = [{"bundle_id": "g:P1", "game_id": "g", "pair": "low:neutral", "treatment": "low",
                 "comparator": "neutral", "listener": "listener1", "order": o, "ok": True,
                 "chosen_condition": "low", "chose_treatment": True} for o in ("ab", "ba")]
        report = {"listener": listener_analysis(rows, n_boot=30, seed=1)}
        criteria = frozen_criteria(report, design())
        self.assertFalse(verdict(criteria)["confirmed"])
        criterion = next(c for c in criteria if c["criterion"] == "Listener listener1: A vs B, both orders")
        self.assertIsNone(criterion["passed"])

    def test_unanimous_two_games_does_not_imply_certainty(self):
        ci = proportion_bootstrap({"a": [1.], "b": [1.]}, n_boot=30)
        self.assertLess(ci["ci_low"], .5)
        self.assertEqual(ci["ci_high"], 1.)

    def test_wide_control_ci_is_not_equivalence(self):
        f = design()
        row = {"n_games": 75, "n_bundles": 200, "signed_ci90": [-.5, .6],
               "by_order": {o: {"n_games": 75, "n_bundles": 200, "ci90": [-.5, .6]} for o in ("ab", "ba")}}
        report = {"listener": {"neutral:original": {"listeners": {"listener1": row}}}}
        c = next(c for c in frozen_criteria(report, f) if c["criterion"].startswith("Listener listener1: B vs Original"))
        self.assertIs(c["passed"], False)

    def test_high_is_required_and_perception_direction_reversed(self):
        f = design("high")
        c = frozen_criteria({}, f)
        self.assertTrue(any(x["criterion"].startswith("C vs B") and x["passed"] is None for x in c))
        row = {"n_games": 75, "n_bundles": 200, "signed_preference": -.8, "signed_ci95": [-.9, -.6],
               "by_order": {o: {"n_games": 75, "n_bundles": 200, "signed": -.8, "ci95": [-.9, -.6]} for o in ("ab", "ba")}}
        report = {"listener": {"high:neutral": {"listeners": {"listener1": row}}}}
        c = next(c for c in frozen_criteria(report, f) if c["criterion"] == "Listener listener1: C vs B, both orders")
        self.assertIs(c["passed"], True)

    def test_duplicate_bundle_is_not_silently_overwritten(self):
        row = {"bundle_id": "g:p", "game_id": "g", "condition": "low", "mattr": .5}
        with self.assertRaises(ValueError):
            paired_clusters([row, row], "mattr", "low", "neutral")


class EnglishContractTests(unittest.TestCase):
    def test_required_pragmatics_and_stance_dimensions(self):
        class Model:
            def complete_json(self, **kwargs):
                result = {k: True for k in DIMENSIONS}
                result["politeness_preserved"] = False
                result["alignments"] = [{"original": source().original_nl, "rewrite": "candidate",
                                         "equivalent": True, "reason": "fixture"}]
                return result, Completion("", "fake", "fake")
        check = EnglishFidelityValidator(Model()).validate(source(), "candidate")
        self.assertIs(check.ok, False)
        self.assertIn("not_preserved:politeness_preserved", check.reasons)
        class NA:
            def complete_json(self, **kwargs):
                return {k: "n/a" for k in DIMENSIONS}, Completion("", "fake", "fake")
        self.assertIsNone(EnglishFidelityValidator(NA()).validate(source(), "candidate").ok)

    def test_missing_history_location_fails_closed(self):
        src = source()
        src.speech_id = None
        src.public_history = ["[s20|P7] estimate(D3, P7, P3, wolf): A future statement."]
        with self.assertRaises(ValueError):
            history_before(src)

    def test_sentence_boundary_change_is_not_copy(self):
        src = source(text="P7 is a werewolf; P2 is a villager.")
        gate = SemanticGate(require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        result = gate.evaluate(src, "P7 is a werewolf. P2 is a villager.", "low")
        self.assertTrue(result.passed)

    def test_neutral_is_generated_without_any_treatment_anchor(self):
        src = source()
        neutral = "P7 is a werewolf since he pushed the vote onto P2 without giving any read of his own."
        client = FakeClient([json.dumps([neutral])])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="neutral")
        gate = SemanticGate(require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        result = generate_with_gate(rw, gate, build_context(src), src)
        self.assertEqual((result.status, result.text), ("ok", neutral))
        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("target_edit_distance", result.to_dict())
        self.assertNotIn("anchor_outcomes", result.to_dict())
        prompt = client.calls[0]["messages"][1]["content"]
        for word in ("edit distance", "Low", "anchor"):
            self.assertNotIn(word, prompt)


class FrozenProtocolTests(unittest.TestCase):
    def test_confirmation_requires_all_automated_criteria(self):
        f = design()
        self.assertEqual(validate_design(f), [])
        sample = {"n_games": 4, "n_bundles": 4}
        metric = {**sample, "reject_fdr_05": True, "direction_confirmed": True,
                  "cohens_dz": .5, "equivalence": {"equivalent": True}, "noninferiority": {"noninferior": True}}
        r = {"expected_bundles": 4, "range": {}, "guardrails": {}, "listener": {},
             "generation_qc": {c: {"final_success_rate": 1.} for c in ("low", "neutral")},
             "absolute_quality": {},
             "integrity": [{"check": "synthetic fixture only", "passed": True}]}
        for treatment in ("low",):
            contrast = f"{treatment}_vs_neutral"
            r["range"][contrast] = {"metrics": {m: dict(metric) for m in f["primary_range_metrics"]}}
            r["guardrails"][contrast] = {"metrics": {m: dict(metric) for m in ("grammar_errors_per_100_words", "cola_mean")}}
            interval = [.6, .9] if treatment == "low" else [-.9, -.6]
            bias = {"position_bias": {"score": 0., "ci90": [-.05, .05]}}
            row = {**sample, **bias, "signed_preference": .8 if treatment == "low" else -.8, "signed_ci95": interval,
                   "by_order": {o: {**sample, "signed": .8 if treatment == "low" else -.8, "ci95": interval} for o in ("ab", "ba")}}
            r["listener"][f"{treatment}:neutral"] = {"listeners": {h: row for h in f["models"]["listeners"] + ["all_models"]}}
        r["range"]["neutral_vs_original"] = {"metrics": {m: dict(metric) for m in f["primary_range_metrics"]}}
        r["guardrails"]["neutral_vs_original"] = {"metrics": {m: dict(metric) for m in ("grammar_errors_per_100_words", "cola_mean")}}
        control = {**sample, **bias, "signed_ci90": [-.05, .05],
                   "by_order": {o: {**sample, "ci90": [-.05, .05]} for o in ("ab", "ba")}}
        r["listener"]["neutral:original"] = {"listeners": {h: control for h in f["models"]["listeners"] + ["all_models"]}}
        r["absolute_quality"] = {c: {**sample, "grammar_mean": .1, "cola_mean": .9} for c in f["conditions"]}
        criteria = frozen_criteria(r, f)
        self.assertTrue(verdict(criteria)["confirmed"])
        r["listener"]["low:neutral"]["listeners"]["unregistered"] = row
        self.assertFalse(verdict(frozen_criteria(r, f))["confirmed"])
        del r["listener"]["low:neutral"]["listeners"]["unregistered"]
        control_metrics = r["guardrails"]["neutral_vs_original"]["metrics"]
        for name in list(control_metrics):
            saved = control_metrics[name]
            for value in (False, None):
                control_metrics[name] = {**saved, "equivalence": {"equivalent": value}}
                self.assertFalse(verdict(frozen_criteria(r, f))["confirmed"])
            del control_metrics[name]
            self.assertFalse(verdict(frozen_criteria(r, f))["confirmed"])
            control_metrics[name] = saved
        del r["guardrails"]
        self.assertFalse(verdict(frozen_criteria(r, f))["confirmed"])

    def test_unsupported_schema_is_rejected(self):
        f = design()
        f["schema_version"] = 2
        self.assertTrue(any("schema_version" in e for e in validate_design(f)))

    def test_decisions_are_complete_and_splits_disjoint(self):
        f = design()
        self.assertEqual(validate_design(f), [])
        f["splits"]["set_b"][0]["game_id"] = f["splits"]["set_a"][0]["game_id"]
        self.assertTrue(any("overlap" in e for e in validate_design(f)))
        f = design()
        del f["equivalence_bounds"]["cola_mean"]
        f["instruments"]["cola_acceptable_index"] = None
        f["models"]["identities"]["listener1"]["family"] = "generator"
        errors = validate_design(f)
        self.assertTrue(any("cola_mean" in e for e in errors))
        self.assertTrue(any("cola_acceptable_index" in e for e in errors))
        self.assertTrue(any("listener families" in e for e in errors))

    def test_wrong_subtlex_or_domain_choice_fails_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.jsonl"
            f = design()
            f["hashes"] = frozen_hashes(f["n_neutral_candidates"])
            rows = [{"bundle_id": b["bundle_id"], "game_id": b["game_id"], "condition": c,
                     "complete": True, "syntax_available": True, "lemmas_available": True, "subtlex_coverage": 1.}
                    for b in f["splits"]["set_a"] for c in f["conditions"]]
            write_jsonl(path, rows)
            provenance = {**artifact_provenance(f, "set_a"), "mattr_window": 50, "include_domain_terms": True,
                          "subtlex": {"sha256": "wrong", "measure": "FREQcount"},
                          "spacy": {"lang": "en", "model": "core_web_sm", "model_version": "test", "spacy_version": "test"}}
            record_artifact(path, provenance)
            checks = integrity_checks(f, features_path=path, split="set_a")
            result = next(r for r in checks if r["check"] == "feature instruments and lexical definition match freeze")
            self.assertIs(result["passed"], False)

    def test_cola_never_silently_truncates(self):
        scorer = ColaScorer.__new__(ColaScorer)
        scorer._torch = SimpleNamespace(no_grad=nullcontext)
        scorer.batch_size, scorer.max_length = 16, 128
        scorer.tokenizer = lambda *a, **k: {"input_ids": SimpleNamespace(shape=(1, 129))}
        with self.assertRaisesRegex(ValueError, "no text was truncated"):
            scorer.score_sentences(["long sentence"])


class LocalModelTests(unittest.TestCase):
    def test_local_registry_preserves_families_and_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "models.json"
            entries = {"a": {"model": "served-a", "family": "family-a", "base_url": "http://node-a:8000/v1"},
                       "b": {"model": "served-b", "family": "family-b", "base_url": "http://node-b:8000"}}
            path.write_text(json.dumps({"models": entries}))
            with patch.dict(os.environ, {"PROFICIENCY_MODEL_CONFIG": str(path)}):
                a, b = ValidationModel.resolve("a"), ValidationModel.resolve("b")
                self.assertNotEqual(a.family, b.family)
                self.assertEqual(completion_url(a.spec.base_url), "http://node-a:8000/v1/chat/completions")
                response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                    "model": "served-a", "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}]})
                with patch("experiments.proficiency_validation.rewrite.runtime.requests.post", return_value=response) as post:
                    self.assertEqual(a.complete(system="s", user="u").text, "ready")
                    self.assertNotIn("Authorization", post.call_args.kwargs["headers"])
                    with self.assertRaises(JudgeModelError):
                        b.complete(system="s", user="u")

    def test_gemini_compatible_url_is_not_double_versioned(self):
        self.assertEqual(completion_url("https://generativelanguage.googleapis.com/v1beta/openai"),
                         "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")

