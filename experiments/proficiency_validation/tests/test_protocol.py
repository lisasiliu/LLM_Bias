"""Regression tests for scientific failure modes and the local-serving boundary.

All model responses and instruments are fake; no external service is contacted.
"""
import copy
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from components.helpers import CheckResult, write_jsonl
from components.judges.judge_model import Completion, JudgeModelError
from components.stats import cluster_bootstrap, tost_equivalence, proportion_bootstrap, wilson_interval
from components.metrics.grammar import ColaScorer
from experiments.proficiency_validation.code.analysis import (
    listener_analysis, frozen_criteria, verdict, integrity_checks, paired_clusters)
from experiments.proficiency_validation.code.common import PRIMARY_RANGE_METRICS, sha256_text, history_before
from experiments.proficiency_validation.code.english_validator import EnglishFidelityValidator, DIMENSIONS, SEMANTIC_RUBRIC_VERSION
from experiments.proficiency_validation.code.frozen import frozen_hashes
from experiments.proficiency_validation.code.models import ValidationModel, completion_url, model_identity
from experiments.proficiency_validation.code.protocol import validate_design, artifact_sidecar, artifact_provenance, sha256_file
from experiments.proficiency_validation.code.semantic_gate import SemanticGate, generate_with_gate, GateResult
from experiments.proficiency_validation.code.generate import generate
from components.rewriters.language_rewriter import LanguageProficiencyRewriter
from experiments.proficiency_validation.tests.test_offline import source, FakeClient
from components.helpers import build_context


def design():
    """Synthetic decisions for tests only, not proposed study thresholds."""
    metrics = list(PRIMARY_RANGE_METRICS)
    handles = ["generator", "judge", "listener1", "listener2", "listener3"]
    identities = {h: {"handle": h, "model": h, "family": h, "revision": "a" * 40,
        "tokenizer_revision": "b" * 40, "serving_version": "test-version", "precision": "float32",
        "chat_template": "tokenizer-default", "extra_body": {}} for h in handles}
    splits = {stage: [{"bundle_id": f"{stage}_g{i}:P1", "game_id": f"{stage}_g{i}", "speaker": "P1",
                       "source_ids": [f"{stage}_s{i}"], "window": [150, 250]} for i in range(6 if stage == "pilot" else 4)]
              for stage in ("pilot", "set_a", "set_b")}
    return {"schema_version": 3, "validation_scope": "automated", "conditions": ["original", "neutral", "low", "high"],
        "primary_range_metrics": metrics, "range_effect_thresholds": {c: {m: .1 for m in metrics}
            for c in ("low_vs_neutral", "high_vs_neutral")},
        "equivalence_bounds": {m: .1 for m in metrics + ["grammar_errors_per_100_words", "cola_mean"]},
        "bound_rationale": {m: "synthetic test rationale, not a study decision" for m in metrics + ["grammar_errors_per_100_words", "cola_mean"]},
        "min_games": 4, "min_bundles": 4, "min_valid_fraction": 1., "min_subtlex_coverage": .9,
        "listener_equivalence_margin": .1, "neutral_edit_distance_tolerance": .1, "neutral_match_min_fraction": .8,
        "mattr_window": 50, "max_bundles_per_game": 1, "n_neutral_candidates": 2, "max_corrective_retries": 2,
        "generation_temperature": 0., "generation_max_tokens": 400, "judge_max_tokens": 800, "probe_max_tokens": 400,
        "bundle_window_words": [150, 250], "power_rationale": "test data are not an empirical power calculation",
        "listener_margin_rationale": "test margin, not a recommended empirical choice",
        "absolute_quality_rationale": "test limits, not recommended empirical thresholds",
        "absolute_quality": {"grammar_errors_max": .5, "cola_mean_min": .8},
        "statistics": {"n_boot": 1000, "n_perm": 1000, "seed": 101},
        "instruments": {"spacy_model": "en_core_web_sm", "spacy_model_version": "test", "spacy_version": "test",
            "languagetool_version": "LanguageTool test", "languagetool_language": "en-US", "cola_model": "test-cola",
            "cola_commit_hash": "c"*40, "cola_acceptable_index": 1, "cola_max_length": 512,
            "cola_label_rationale": "test metadata with an explicitly verified label", "subtlex_sha256": "d"*64,
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
        criterion = next(c for c in criteria if c["criterion"] == "Listener listener1: low vs Neutral, both orders")
        self.assertIsNone(criterion["passed"])

    def test_unanimous_two_games_does_not_imply_certainty(self):
        ci = proportion_bootstrap({"a": [1.], "b": [1.]}, n_boot=30)
        self.assertLess(ci["ci_low"], .5)
        self.assertEqual(ci["ci_high"], 1.)

    def test_wide_control_ci_is_not_equivalence(self):
        f = design()
        row = {"n_games": 75, "n_bundles": 200, "equivalence_ci90": [.25, .8],
               "by_order": {o: {"n_games": 75, "n_bundles": 200, "ci90": [.25, .8]} for o in ("ab", "ba")}}
        report = {"listener": {"neutral:original": {"listeners": {"listener1": row}}}}
        c = next(c for c in frozen_criteria(report, f) if c["criterion"].startswith("Listener listener1: Neutral"))
        self.assertIs(c["passed"], False)

    def test_high_is_required_and_perception_direction_reversed(self):
        f = design()
        c = frozen_criteria({}, f)
        self.assertTrue(any(x["criterion"].startswith("High vs Neutral") and x["passed"] is None for x in c))
        row = {"n_games": 75, "n_bundles": 200, "cluster_bootstrap_ci": [.05, .2],
               "by_order": {o: {"n_games": 75, "n_bundles": 200, "ci95": [.05, .2]} for o in ("ab", "ba")}}
        report = {"listener": {"high:neutral": {"listeners": {"listener1": row}}}}
        c = next(c for c in frozen_criteria(report, f) if c["criterion"] == "Listener listener1: high vs Neutral, both orders")
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
        result = gate.evaluate(src, "P7 is a werewolf. P2 is a villager.", "low")
        self.assertTrue(result.passed)

    def test_neutral_uses_gated_low_and_high_in_runtime(self):
        src = source()
        texts = ["P7 is a werewolf. He pushed the vote onto P2. He gave no read.",
                 "P7 is a werewolf because he redirected the vote toward P2 without offering his own read."]
        neutral = "P7 is a werewolf since he pushed the vote onto P2 without giving any read of his own."
        client = FakeClient(texts + [json.dumps([neutral])])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="neutral", neutral_anchor_conditions=["low", "high"])
        gate = SemanticGate(require_parser=False, require_validator=False)
        result = generate_with_gate(rw, gate, build_context(src), src)
        self.assertEqual(result.status, "ok")
        self.assertEqual(set(result.anchor_outcomes), {"low", "high"})
        self.assertTrue(all(r["attempts"][0]["gate"]["ok"] for r in result.anchor_outcomes.values()))
        self.assertEqual(len(client.calls), 3)


class FrozenProtocolTests(unittest.TestCase):
    def test_automated_protocol_can_pass_without_human_data(self):
        f = design()
        self.assertNotIn("human_validation", f)
        self.assertNotIn("calibration", f)
        self.assertEqual(validate_design(f), [])
        sample = {"n_games": 4, "n_bundles": 4}
        metric = {**sample, "reject_fdr_05": True, "direction_confirmed": True,
                  "cohens_dz": .5, "equivalence": {"equivalent": True}}
        r = {"expected_bundles": 4, "range": {}, "guardrails": {}, "listener": {},
             "neutral_matching_by_condition": {}, "absolute_quality": {},
             "integrity": [{"check": "synthetic fixture only", "passed": True}]}
        for treatment in ("low", "high"):
            contrast = f"{treatment}_vs_neutral"
            r["range"][contrast] = {"metrics": {m: dict(metric) for m in f["primary_range_metrics"]}}
            r["guardrails"][contrast] = {"metrics": {m: dict(metric) for m in ("grammar_errors_per_100_words", "cola_mean")}}
            r["neutral_matching_by_condition"][treatment] = {"fraction_within_tolerance": 1.}
            interval = [.7, .9] if treatment == "low" else [.1, .3]
            row = {**sample, "cluster_bootstrap_ci": interval,
                   "by_order": {o: {**sample, "ci95": interval} for o in ("ab", "ba")}}
            r["listener"][f"{treatment}:neutral"] = {"listeners": {h: row for h in f["models"]["listeners"] + ["all_models"]}}
        r["range"]["neutral_vs_original"] = {"metrics": {m: dict(metric) for m in f["primary_range_metrics"]}}
        control = {**sample, "equivalence_ci90": [.47, .53],
                   "by_order": {o: {**sample, "ci90": [.47, .53]} for o in ("ab", "ba")}}
        r["listener"]["neutral:original"] = {"listeners": {h: control for h in f["models"]["listeners"] + ["all_models"]}}
        r["absolute_quality"] = {c: {**sample, "grammar_mean": .1, "cola_mean": .9} for c in f["conditions"]}
        criteria = frozen_criteria(r, f)
        self.assertTrue(verdict(criteria)["confirmed"])
        self.assertFalse(any(c["criterion"].startswith("Human") for c in criteria))
        r["human_quality_criteria"] = [{"criterion": "optional evidence", "passed": False}]
        self.assertEqual(criteria, frozen_criteria(r, f))
        del r["guardrails"]
        self.assertFalse(verdict(frozen_criteria(r, f))["confirmed"])

    def test_old_human_required_schema_is_not_silently_reinterpreted(self):
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
            artifact_sidecar(path, provenance)
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
                with patch("experiments.proficiency_validation.code.models.requests.post", return_value=response) as post:
                    self.assertEqual(a.complete(system="s", user="u").text, "ready")
                    self.assertNotIn("Authorization", post.call_args.kwargs["headers"])
                    with self.assertRaises(JudgeModelError):
                        b.complete(system="s", user="u")

    def test_gemini_compatible_url_is_not_double_versioned(self):
        self.assertEqual(completion_url("https://generativelanguage.googleapis.com/v1beta/openai"),
                         "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")


if __name__ == "__main__":
    unittest.main()
