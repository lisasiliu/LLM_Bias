import unittest
from types import SimpleNamespace

from components.helpers import CheckResult
from components.judges.inverse_predicate_parser import ParseResult
from components.judges.judge_model import Completion
from experiments.proficiency_validation.validation.analysis import listener_analysis, passage_retention, frozen_criteria
from experiments.proficiency_validation.rewrite.fidelity import DIMENSIONS, NEUTRAL_DIMENSIONS, EnglishFidelityValidator, alignment_check
from experiments.proficiency_validation.rewrite.fidelity import proficiency_check
from experiments.proficiency_validation.validation.analysis import cluster_sign_flip_test
from experiments.proficiency_validation.validation.calibration import build_challenge_set, run_challenge
from experiments.proficiency_validation.validation.perception import prompt, prompt_sha
from experiments.proficiency_validation.validation.protocol import validate_design
from experiments.proficiency_validation.rewrite.gate import SemanticGate
from experiments.proficiency_validation.tests.test_offline import source
from experiments.proficiency_validation.tests.test_protocol import design


class Model:
    def __init__(self, result):
        self.result = result
        self.calls = []
    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.result, Completion("", "test", "test")


def judgment(original, rewritten, **dimensions):
    return {**{key: True for key in DIMENSIONS}, "violations": [], "uncertain": False, **dimensions,
            "alignments": [{"original": original, "rewrite": rewritten,
                            "equivalent": True, "reason": "test response"}]}


class PropositionEvidenceTests(unittest.TestCase):
    def test_boolean_only_answer_is_not_a_pass(self):
        result = EnglishFidelityValidator(Model({key: True for key in DIMENSIONS})).validate(source(), "candidate")
        self.assertIsNone(result.ok)
        self.assertIn("missing_proposition_alignments", result.reasons[0])

    def test_missing_qualification_cannot_be_hidden_by_incomplete_quotes(self):
        original, rewritten = "P7 is a seer claimant.", "P7 is a seer."
        rows = [{"original": "P7 is a seer", "rewrite": "P7 is a seer.", "equivalent": True}]
        missing, _ = alignment_check(rows, original, rewritten)
        self.assertIn("incomplete_alignment_coverage:original", missing)

    def test_invented_quote_does_not_count_as_evidence(self):
        missing, _ = alignment_check([{"original": "P7 is a villager.", "rewrite": "P7 is a wolf.",
            "equivalent": True}], "P7 is a wolf.", "P7 is a wolf.")
        self.assertTrue(any(x.startswith("ungrounded_alignment_quote") for x in missing))

    def test_non_equivalence_overrides_global_true_flags(self):
        src = source(text="P7 is a seer claimant.")
        rewritten = "P7 is a seer."
        response = judgment(src.original_nl, rewritten)
        response["alignments"][0].update(equivalent=False, reason="claim promoted to fact")
        result = EnglishFidelityValidator(Model(response)).validate(src, rewritten)
        self.assertIs(result.ok, False)
        self.assertTrue(any(x.startswith("proposition_changed") for x in result.reasons))

    def test_preserved_plain_synonym_with_full_evidence_passes(self):
        src = source("claim_role(D1, P3, seer)", "I can hand the table a verified alignment.")
        rewritten = "I can give the table a verified alignment."
        model = Model(judgment(src.original_nl, rewritten))
        result = EnglishFidelityValidator(model).validate(src, rewritten)
        self.assertTrue(result.ok)
        self.assertIn("Hand is a metaphor", model.calls[0]["system"])

    def test_neutral_requires_its_own_range_checks(self):
        src, rewritten = source(), "candidate"
        response = judgment(src.original_nl, rewritten)
        model = Model(response)
        validator = EnglishFidelityValidator(model)
        self.assertIsNone(validator.validate(src, rewritten, condition="neutral").ok)
        response.update({key: True for key in NEUTRAL_DIMENSIONS})
        response["neutral_repetition_preserved"] = False
        result = validator.validate(src, rewritten, condition="neutral")
        self.assertIs(result.ok, False)
        self.assertIn("not_preserved:neutral_repetition_preserved", result.reasons)


class SurfacePolicyTests(unittest.TestCase):
    def test_courtesy_addition_fails_before_a_permissive_model(self):
        src = source("recommend_action(D1, P3, P2, vote_execute, P7)", "P2, vote to execute P7.")
        validator = SimpleNamespace(validate=lambda *a, **k: CheckResult(ok=True))
        gate = SemanticGate(validator=validator, require_parser=False)
        result = gate.evaluate(src, "P2, please vote to execute P7.", "low")
        self.assertFalse(result.ok)
        self.assertTrue(any("courtesy_marker_changed" in r for r in result.reasons))
        self.assertEqual(result.semantic, {})

    def test_equivalent_stance_needs_semantic_adjudication(self):
        src = source(text="I read P7 as a wolf.")
        text = "I think P7 is a wolf."
        surface = proficiency_check(src, text)
        self.assertTrue(surface.ok)
        self.assertTrue(surface.details["requires_semantic_adjudication"])
        gate = SemanticGate(require_parser=False, require_validator=False)
        self.assertIsNone(gate.evaluate(src, text, "low").ok)
        gate.validator = EnglishFidelityValidator(Model(judgment(src.original_nl, text)))
        self.assertTrue(gate.evaluate(src, text, "low").ok)

    def test_new_hedge_without_existing_stance_still_fails(self):
        result = proficiency_check(source(text="P7 is a wolf."), "Maybe P7 is a wolf.")
        self.assertFalse(result.ok)

    def test_action_keyword_does_not_bypass_semantic_adjudication(self):
        src = source("address_role(D1, P3, village, reveal)",
                     "Village, if anyone else claims Seer, their claim is false.")
        rewritten = "Village, if anyone else says they are the Seer, that is false."
        result = proficiency_check(src, rewritten)
        self.assertNotIn("lost_action_slot:reveal", result.reasons)
        self.assertIn("lost_action_slot:reveal", result.details["semantic_signals"])
        self.assertIsNone(SemanticGate(require_parser=False, require_validator=False).evaluate(src, rewritten, "low").ok)

    def test_numeric_loss_is_an_explicit_judge_signal(self):
        src = source("recommend_action(D1, P3, P2, vote_execute, P7)", "P2, use your 1.5 vote to execute P7.")
        rewritten = "P2, use your vote to execute P7."
        model = Model({})
        _, user = EnglishFidelityValidator(model).build_prompts(src, rewritten)
        self.assertIn("numeric_expression_changed", user)


class ParserBaselineTests(unittest.TestCase):
    def parser(self, role="werewolf"):
        class Parser:
            model = None
            calls = 0
            def parse(self, text, **kwargs):
                self.calls += 1
                return ParseResult(ok=True, predicate_name="estimate", recoverable=True,
                    arguments={"Phase": "D1", "Player": "P3", "Target": "P2", "Role": role})
        return Parser()

    def test_matching_secondary_act_requires_semantics_and_records_baseline(self):
        src = source("disagree(D1, P3, s1)", "P2 is wrong, and P2 is a wolf.")
        rewritten = "P2 is wrong. P2 is a wolf."
        parser = self.parser()
        gate = SemanticGate(parser, require_validator=False)
        result = gate.evaluate(src, rewritten, "low")
        self.assertIsNone(result.ok)
        self.assertEqual(result.parser["comparison_basis"], "original_parse")
        self.assertFalse(result.parser["source_predicate_match"])
        gate.validator = EnglishFidelityValidator(Model(judgment(src.original_nl, rewritten)))
        self.assertTrue(gate.evaluate(src, rewritten, "low").ok)
        self.assertEqual(parser.calls, 3)  # one cached original, two rewritten parses

    def test_changed_target_cannot_pass_by_same_predicate_name(self):
        src = source("disagree(D1, P3, s1)", "P2 is wrong, and P2 is a wolf.")
        parser = self.parser()
        original_parse = parser.parse
        def parse(text, **kwargs):
            result = original_parse(text, **kwargs)
            if text != src.original_nl:
                result.arguments["Target"] = "P7"
            return result
        parser.parse = parse
        candidate = "P2 is wrong. P7 is a wolf."
        gate = SemanticGate(parser, EnglishFidelityValidator(Model(judgment(src.original_nl, candidate))))
        self.assertFalse(gate.evaluate(src, candidate, "low").ok)

    def test_missing_recoverability_is_not_a_parser_pass(self):
        src = source("estimate(D1, P3, P2, werewolf)", "P2 is a wolf.")
        parser = self.parser()
        base_parse = parser.parse
        def parse(text, **kwargs):
            result = base_parse(text, **kwargs)
            result.recoverable = None
            return result
        parser.parse = parse
        gate = SemanticGate(parser, require_validator=False)
        self.assertIsNone(gate.evaluate(src, "P2 is a werewolf.", "low").ok)


class InferenceAndPerceptionTests(unittest.TestCase):
    def test_three_game_sign_flip_is_exact_and_seed_independent(self):
        clusters = {"a": [1.], "b": [2.], "c": [3.]}
        a = cluster_sign_flip_test(clusters, seed=1)
        self.assertEqual(a, cluster_sign_flip_test(clusters, seed=2, n_perm=1000))
        self.assertEqual((a["p_value"], a["n_perm"], a["method"]), (.25, 8, "exact"))
        self.assertEqual(cluster_sign_flip_test({"a": [0], "b": [0]})["p_value"], 1.)
        self.assertIsNone(cluster_sign_flip_test({"a": [1]})["p_value"])

    def test_constructs_are_distinct_and_cannot_be_pooled(self):
        self.assertNotEqual(prompt_sha("range"), prompt_sha("perceived_proficiency"))
        self.assertIn("weaker command of English", prompt("perceived_proficiency"))
        with self.assertRaisesRegex(ValueError, "separately"):
            listener_analysis([{"construct": "range"}, {"construct": "perceived_proficiency"}], n_boot=20, seed=1)
        f = design()
        f["perception_constructs"].append("perceived_proficiency")
        self.assertEqual(validate_design(f), [])
        criteria = [c for c in frozen_criteria({}, f) if c["criterion"].startswith("Perceived proficiency:")]
        self.assertTrue(criteria)
        self.assertTrue(all(c["passed"] is None for c in criteria))

    def test_choice_reversal_is_reported_even_when_mean_is_half(self):
        rows = [{"bundle_id": "g:p", "game_id": "g", "pair": "low:neutral", "listener": "m",
                 "treatment": "low", "comparator": "neutral", "order": order, "ok": True,
                 "chosen_condition": "neutral" if order == "ab" else "low", "chose_treatment": order == "ba"}
                for order in ("ab", "ba")]
        r = listener_analysis(rows, n_boot=20, seed=1)["low:neutral"]["listeners"]["m"]
        self.assertEqual(r["share_treatment_more_limited"], .5)
        self.assertEqual(r["order_consistency"]["same_condition"], 0)
        self.assertEqual(r["order_consistency"]["always_b"], 1)
        self.assertEqual(r["position_b_raw_share"], 1.)

    def test_retention_keeps_missing_long_passages_in_denominator(self):
        rows = [{"bundle_id": b, "game_id": b, "condition": c, "complete": c == "original" or b == "short",
                 "word_count": words, "n_utterances": n}
                for b, words, n in (("short", 150, 3), ("long", 600, 12))
                for c in ("original", "low", "neutral")]
        r = passage_retention(rows)["low"]
        self.assertEqual(r["retained_fraction"], .5)
        self.assertEqual(r["excluded"]["mean_original_words"], 600)


class CalibrationTests(unittest.TestCase):
    def test_constructed_cases_include_positive_paraphrases_and_observed_failure_types(self):
        items = build_challenge_set([source()])
        kinds = {r["challenge_type"] for r in items}
        self.assertTrue({"positive_simplification", "claim_attribution_shift", "courtesy_addition",
                         "numeric_qualification_loss", "invented_divination_result"} <= kinds)
        self.assertEqual(len({r["item_id"] for r in items}), len(items))

    def test_calibration_resume_reuses_finished_items(self):
        item = next(r for r in build_challenge_set([source()]) if r["challenge_type"] == "positive_identity")
        gate = SemanticGate(require_parser=False, require_validator=False)
        first = run_challenge([item], gate)
        self.assertEqual(run_challenge([item], None, completed_rows=first), first)
