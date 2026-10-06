import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import requests
from components.helpers import build_context
from components.judges.judge_model import ModelSpec
from components.rewriters.language_rewriter import LanguageProficiencyRewriter
from experiments.proficiency_validation.validation.analysis import contrast_analysis, listener_analysis, frozen_criteria
from experiments.proficiency_validation.validation.sources import build_bundles, assemble_condition_texts
from experiments.proficiency_validation.rewrite.runtime import ValidationModel, ValidationTransportError
from experiments.proficiency_validation.validation.protocol import validate_design
from experiments.proficiency_validation.rewrite.gate import SemanticGate, GateResult, generate_with_gate, generation_qc
from experiments.proficiency_validation.tests.test_offline import source, FakeClient
from experiments.proficiency_validation.tests.test_protocol import design


class PassageTests(unittest.TestCase):
    def records(self):
        return [replace(source(text=" ".join(["word"] * 20)), source_id=f"g:s{i}", speech_id=f"s{i}",
                        game_id="g", speaker_id="P1") for i in range(1, 10)]

    def test_whole_game_retains_late_turns_and_does_not_salvage_incomplete_text(self):
        sources = self.records()
        bundles = build_bundles(sources, selection="whole_game", window_min=50, window_max=80)["bundles"]
        self.assertEqual(bundles[0]["source_ids"], [s.source_id for s in sources])
        rows = [{"source_id": s.source_id, "condition": "low", "original_text": s.original_nl,
                 "status": "ok", "rewritten_text": "simple words"} for s in sources[:-1]]
        result = assemble_condition_texts(bundles, rows, ["low"])[0]
        self.assertFalse(result["complete"])
        self.assertIsNone(result["text"])
        self.assertEqual(result["missing_source_ids"], ["g:s9"])

    def test_stratified_sample_contains_all_thirds_and_is_deterministic(self):
        sources = self.records()
        kwargs = dict(selection="phase_stratified", window_min=50, window_max=80, seed=37)
        a = build_bundles(sources, **kwargs)["bundles"][0]
        b = build_bundles(list(reversed(sources)), **kwargs)["bundles"][0]
        self.assertEqual(a, b)
        indices = [int(s.split("s")[-1]) - 1 for s in a["source_ids"]]
        self.assertEqual(indices, sorted(indices))
        self.assertEqual({i // 3 for i in indices}, {0, 1, 2})

    def test_stratified_window_cannot_silently_drop_late_thirds(self):
        result = build_bundles(self.records(), selection="phase_stratified", window_min=20, window_max=30)
        self.assertEqual(result["bundles"], [])


class DecisionTests(unittest.TestCase):
    def test_games_receive_equal_weight_despite_different_speaker_counts(self):
        rows, choices = [], []
        for game, n, diff in (("large", 9, 1), ("small", 1, -1)):
            for i in range(n):
                for condition, value in (("low", diff), ("neutral", 0)):
                    rows.append({"bundle_id": f"{game}:{i}", "game_id": game, "condition": condition, "mattr": value})
                for order in ("ab", "ba"):
                    choices.append({"bundle_id": f"{game}:{i}", "game_id": game, "pair": "low:neutral",
                        "listener": "m", "order": order, "ok": True, "treatment": "low", "comparator": "neutral",
                        "chosen_condition": "low" if diff == 1 else "neutral", "chose_treatment": diff == 1})
        effect = contrast_analysis(rows, ["mattr"], frozen=None, n_boot=30, seed=1, family="range")
        self.assertEqual(effect["low_vs_neutral"]["metrics"]["mattr"]["estimate"], 0)
        perceived = listener_analysis(choices, n_boot=30, seed=1)["low:neutral"]["listeners"]["m"]
        self.assertEqual(perceived["share_treatment_more_limited"], .5)
        self.assertEqual(perceived["share_treatment_raw_votes"], .9)

    def test_small_perception_effect_does_not_pass_meaningful_separation(self):
        f = design()
        row = {"n_games": 1000, "n_bundles": 1000, "signed_preference": .04, "signed_ci95": [.02, .06],
               "by_order": {o: {"n_games": 1000, "n_bundles": 1000, "signed": .04, "ci95": [.02, .06]} for o in ("ab", "ba")}}
        report = {"listener": {"low:neutral": {"listeners": {"listener1": row}}}}
        c = next(c for c in frozen_criteria(report, f) if c["criterion"] == "Listener listener1: low vs Neutral, both orders")
        self.assertIs(c["passed"], False)

    def test_independent_treatments_and_judge_are_required(self):
        f = design()
        self.assertEqual(validate_design(f), [])
        f["conditions"].append("high")
        self.assertTrue(validate_design(f))
        f = design()
        f["models"]["identities"]["judge"]["family"] = "generator"
        self.assertTrue(any("semantic judge" in e for e in validate_design(f)))

    def test_neutral_outcome_does_not_depend_on_other_conditions(self):
        src = source()
        neutral = "P7 is a werewolf since he pushed the vote onto P2 without giving his own read."
        client = FakeClient([json.dumps([neutral])])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="neutral")
        result = generate_with_gate(rw, SemanticGate(require_parser=False, require_validator=False),
                                    build_context(src), src)
        self.assertEqual((result.status, result.text), ("ok", neutral))
        self.assertEqual(len(client.calls), 1)

    def test_failed_attempts_and_infrastructure_losses_are_counted_separately(self):
        rows = [{"condition": "neutral", "status": "infrastructure_unavailable", "n_attempts": 3},
                {"condition": "neutral", "status": "treatment_unavailable", "n_attempts": 2},
                {"condition": "neutral", "status": "ok", "n_attempts": 1}]
        q = generation_qc(rows)["neutral"]
        self.assertEqual((q["infrastructure_unavailable"], q["own_failed_attempts"], q["final_success"]), (1, 2, 1))
        self.assertNotIn("anchor_blocked", q)


class TransportTests(unittest.TestCase):
    def model(self):
        spec = ModelSpec(handle="test", model="test", family="test", provider="local", base_url="http://localhost:8000", api_key_env=())
        return ValidationModel(spec, identity={"model": "test"}, retries=2)

    def response(self, code):
        r = requests.Response()
        r.status_code = code
        r._content = b'{"error": "maximum context length exceeded"}'
        return r

    def test_http400_is_logged_without_transport_retries(self):
        model = self.model()
        with patch("experiments.proficiency_validation.rewrite.runtime.requests.post", return_value=self.response(400)) as post:
            with self.assertRaises(ValidationTransportError) as caught:
                model.complete(system="s", user="some context", max_tokens=800)
        self.assertEqual(post.call_count, 1)
        detail = caught.exception.details
        self.assertEqual(detail["http_status"], 400)
        self.assertIn("maximum context", detail["response_excerpt"])
        self.assertEqual(detail["user_chars"], 12)
        self.assertFalse(detail["retryable"])

    def test_transient_http503_can_recover(self):
        model = self.model()
        ok = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "model": "test", "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}]})
        with patch("experiments.proficiency_validation.rewrite.runtime.requests.post", side_effect=[self.response(503), ok]) as post, patch.object(model, "_sleep"):
            self.assertEqual(model.complete(system="s", user="u").text, "ready")
        self.assertEqual(post.call_count, 2)
        self.assertIsNone(model.last_error)

    def test_gate_does_not_rejudge_or_regenerate_after_permanent_error(self):
        src = source()
        client = FakeClient(["P7 is a werewolf since he pushed the vote onto P2 without giving his own read."])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="low")
        gate = SimpleNamespace(evaluate=lambda *a: GateResult(ok=None, parser={"transport": {"retryable": False}}))
        with patch.object(gate, "evaluate", wraps=gate.evaluate) as evaluate:
            result = generate_with_gate(rw, gate, build_context(src), src)
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(result.status, "infrastructure_unavailable")
