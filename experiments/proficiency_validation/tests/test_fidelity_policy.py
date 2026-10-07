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

    def test_archived_alignment_audit_checks_the_selected_accepted_text(self):
        from experiments.proficiency_validation.rewrite.fidelity import audit_accepted_alignments
        original, candidate = "P5 made a claim.", "P5's claim is clear."
        alignment = [{"original": original, "rewrite": "P5", "equivalent": True},
                     {"original": original, "rewrite": candidate, "equivalent": True}]
        row = {"source_id": "g:s1", "condition": "high", "status": "ok", "original_text": original,
               "rewritten_text": candidate, "attempts": [{"text": candidate,
               "gate": {"ok": True, "semantic": {"dimensions": {"alignments": alignment}}}}]}
        audit = audit_accepted_alignments([row, {**row, "status": "treatment_unavailable"}])
        self.assertEqual(audit["checked"], 1)
        self.assertEqual(audit["issues"][0]["errors"], ["ungrounded_alignment_quote:0:rewrite"])
        alignment[0]["rewrite"] = candidate
        self.assertEqual(audit_accepted_alignments([row])["issues"], [])

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
        response.update({key: "same" for key in NEUTRAL_DIMENSIONS})
        response["neutral_repetition_level"] = "more_advanced"
        result = validator.validate(src, rewritten, condition="neutral")
        self.assertIs(result.ok, False)
        self.assertIn("not_preserved:neutral_repetition_level:more_advanced", result.reasons)
        response["neutral_repetition_level"] = "simpler"
        self.assertIn("not_preserved:neutral_repetition_level:simpler", validator.validate(src, rewritten, condition="neutral").reasons)
        response["neutral_repetition_level"] = True                      # a boolean is not a direction
        self.assertIsNone(validator.validate(src, rewritten, condition="neutral").ok)
        response.update({key: "same" for key in NEUTRAL_DIMENSIONS})
        self.assertTrue(validator.validate(src, rewritten, condition="neutral").ok)


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


class Scripted:
    def __init__(self, results):
        self.results, self.calls = list(results), []
    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.results.pop(0), Completion("", "test", "test")


class JudgeFormatRobustnessTests(unittest.TestCase):
    """Formatting slips of the judge are repaired or tolerated; its content verdicts are not."""

    def test_quote_matching_ignores_punctuation_but_not_words(self):
        from experiments.proficiency_validation.rewrite.fidelity import _quote_pattern, _TYPOGRAPHIC
        import re
        raw = 'P6\u2019s case against P5 is only "he pushed me first," and P6 is attacking the messenger.'
        text = raw.translate(_TYPOGRAPHIC)        # the checker normalizes typographic marks before matching
        for quote in ('only "he pushed me first"', "P6's case against P5 is only \"he pushed me first\"", "he pushed me first."):
            self.assertTrue(re.search(_quote_pattern(quote), text), quote)
        self.assertFalse(re.search(_quote_pattern("he pushed me second"), text))
        self.assertFalse(re.search(_quote_pattern("pushed me"), "he pushed meat first"))
        self.assertFalse(re.search(_quote_pattern("he pushed"), "she pushed"))
        for quote, text in (("not able", "notable"), ("now here", "nowhere"),
                            ("can", "can't"), ("able", "not-able"), ("not able", "not-able")):
            self.assertIsNone(re.search(_quote_pattern(quote), text), (quote, text))
        self.assertIsNotNone(re.search(_quote_pattern("not able"), "'not, able'"))
        rows = [{"original": 'P6\'s case against P5 is only "he pushed me first"', "rewrite": "P6's case is weak", "equivalent": True},
                {"original": "and P6 is attacking the messenger", "rewrite": "P6's case is weak", "equivalent": True}]
        missing, _ = alignment_check(rows, raw, "P6's case is weak.")
        self.assertEqual(missing, [])      # punctuation moved inside the quotation mark; a shared rewrite excerpt is repeated

    def test_repair_hints_name_the_referenced_statement_and_the_broken_alignment(self):
        from experiments.proficiency_validation.rewrite.fidelity import repair_hints
        src = SimpleNamespace(referenced_speech={"speech_id": "s20", "speaker": "P4"})
        hints = repair_hints(["reference_preserved", "invalid_alignment_quote:1:rewrite", "ungrounded_alignment_quote:0:original",
                              "incomplete_alignment_coverage:rewrite", "violations"], src)
        self.assertEqual(len(hints), 5)
        self.assertIn("s20 by P4", hints[0]); self.assertIn("n/a is only", hints[0])
        self.assertIn("Alignment 1", hints[1]); self.assertIn("repeat that excerpt", hints[1])
        self.assertIn("Alignment 0", hints[2])
        self.assertEqual(repair_hints(["reference_preserved"], SimpleNamespace(referenced_speech=None))[0][:36],
                         "The original responds to an earlier ")

    def test_validator_prompt_requires_repeated_excerpts_for_merged_clauses(self):
        system, _ = EnglishFidelityValidator(Model({})).build_prompts(source(), "candidate")
        self.assertIn("repeat that excerpt in each alignment", system)
        self.assertIn("english_fidelity_v7", __import__("experiments.proficiency_validation.rewrite.fidelity", fromlist=["x"]).SEMANTIC_RUBRIC_VERSION)

    def test_inventory_evidence_lifted_from_history_earns_one_repair(self):
        from experiments.proficiency_validation.rewrite.fidelity import extract_inventory
        history = ["[s1|P3] estimate(D1, P3, P7, werewolf): P7 is pushing hardest against P8."]
        text = "P8 is village, and P7 is a wolf."
        from_history = {"complete": True, "acts": [{"predicate": "estimate(D1, P3, P7, werewolf)", "evidence": "P7 is pushing hardest against P8"}]}
        grounded = {"complete": True, "acts": [{"predicate": "estimate(D1, P3, P8, villager)", "evidence": "P8 is village"},
                                                {"predicate": "estimate(D1, P3, P7, werewolf)", "evidence": "P7 is a wolf"}]}
        model = Scripted([from_history, grounded])
        record = extract_inventory(model, text, phase="D1", speaker="P3", history=history)
        self.assertTrue(record["ok"]); self.assertEqual(len(record["signatures"]), 2)
        self.assertEqual(len(model.calls), 2)
        self.assertIn("repair_extraction", model.calls[1]["user"])
        self.assertEqual(record["attempts"][0]["error"], "evidence_not_in_utterance:0")
        again = extract_inventory(Scripted([from_history, from_history]), text, phase="D1", speaker="P3", history=history)
        self.assertFalse(again["ok"]); self.assertEqual(len(again["attempts"]), 2)
        abstained = Scripted([{"complete": False, "acts": []}])
        self.assertFalse(extract_inventory(abstained, text, phase="D1", speaker="P3", history=history)["ok"])
        self.assertEqual(len(abstained.calls), 1)      # the model's own abstention is not argued with


class ParserAbstentionTests(unittest.TestCase):
    """A parser that recovers nothing from the candidate abstains; the fidelity judge then decides."""

    class AbstainingParser:
        model = None
        def __init__(self):
            self.calls = 0
        def parse(self, text, *, phase, speaker_id, public_history):
            from components.judges.inverse_predicate_parser import ParseResult
            self.calls += 1
            if self.calls == 1:      # the original parses
                return ParseResult(ok=True, predicate_name="estimate",
                                   arguments={"Phase": "D1", "Player": "P3", "Target": "P7", "Role": "werewolf"})
            return ParseResult(ok=True, predicate_name=None, arguments={})

    def test_deferred_abstention_is_decided_and_recorded_by_the_judge(self):
        src, candidate = source(), "P7 is a wolf. He pushed the vote onto P2 and never gave his own read."
        gate = SemanticGate(self.AbstainingParser(), EnglishFidelityValidator(Model(judgment(src.original_nl, candidate))))
        self.assertEqual(gate.parser_abstention, "defer")
        result = gate.evaluate(src, candidate, "low")
        self.assertTrue(result.ok); self.assertIn("parser_abstained", result.skipped)
        self.assertTrue(result.parser["abstained"]); self.assertEqual(result.reasons, [])
        self.assertEqual(gate.provenance()["parser_abstention"], "defer")

    def test_deferred_abstention_still_needs_a_judge_verdict_and_keeps_rejections(self):
        src, candidate = source(), "P7 is a wolf."
        rejected = judgment(src.original_nl, candidate, content_preserved=False)
        gate = SemanticGate(self.AbstainingParser(), EnglishFidelityValidator(Model(rejected)))
        self.assertIs(gate.evaluate(src, candidate, "low").ok, False)
        incomplete = {key: True for key in DIMENSIONS}            # no alignments: the judge has no verdict
        gate = SemanticGate(self.AbstainingParser(), EnglishFidelityValidator(Model(incomplete)))
        result = gate.evaluate(src, candidate, "low")
        self.assertIsNone(result.ok); self.assertIn("semantic_adjudication_required", result.reasons)

    def test_no_verdict_policy_and_qc_count(self):
        from experiments.proficiency_validation.rewrite.gate import generation_qc
        src, candidate = source(), "P7 is a wolf. He pushed the vote onto P2 and never gave his own read."
        gate = SemanticGate(self.AbstainingParser(), EnglishFidelityValidator(Model(judgment(src.original_nl, candidate))),
                            parser_abstention="no_verdict")
        result = gate.evaluate(src, candidate, "low")
        self.assertIsNone(result.ok); self.assertIn("parser_no_verdict:predicate_not_recoverable", result.reasons)
        with self.assertRaises(ValueError):
            SemanticGate(parser_abstention="sometimes")
        rows = [{"condition": "low", "status": "ok", "attempts": [{"gate": {"skipped": ["parser_abstained"]}}]},
                {"condition": "low", "status": "ok", "attempts": [{"gate": {"skipped": []}}]},
                {"condition": "low", "status": "treatment_unavailable", "n_attempts": 1, "attempts": [{"gate": {"skipped": ["parser_abstained"]}}]}]
        self.assertEqual(generation_qc(rows)["low"]["accepted_with_parser_abstention"], 1)

    def test_neutral_prompt_carries_no_edit_distance_target(self):
        from components.rewriters.language_rewriter import LanguageProficiencyRewriter
        from lurkingwolf.rewriter import RewriteContext
        rw = LanguageProficiencyRewriter(llm_client=None, condition="neutral")
        ctx = RewriteContext(predicate=source().predicate_text, speaker_id="P3", original_nl=source().original_nl,
                             phase="day_speech", public_history=[], speaker_role_claimed=None)
        system, user = rw.build_candidate_prompts(ctx, 5, feedback={"previous_candidates": ["an earlier wording"]})
        for text in (system, user):
            self.assertNotIn("edit-distance", text.lower()); self.assertNotIn("edit distance", text.lower())
        self.assertIn("Candidates already considered", user); self.assertIn("an earlier wording", user)
        self.assertIn("Reword every clause", system)


class SecondOpinionTests(unittest.TestCase):
    """Parser three-state rule: recovered mismatch fails; abstention goes to the judge, and when a second
    judge is configured it must agree; conflict or silence leaves no verdict."""

    def gate(self, verdict, second=None, **kwargs):
        from experiments.proficiency_validation.tests.test_fidelity_policy import ParserAbstentionTests as P
        validator = EnglishFidelityValidator(Model(verdict))
        opinion = None if second is None else EnglishFidelityValidator(Model(second))
        if opinion is not None:
            opinion.model.spec = SimpleNamespace(handle="second", family="other")
            opinion.model.identity_record = {"handle": "second", "family": "other"}
        return SemanticGate(P.AbstainingParser(), validator, second_opinion=opinion, **kwargs)

    def test_statuses_and_second_opinion_outcomes(self):
        src, candidate = source(), "P7 is a wolf. He pushed the vote onto P2 and never gave his own read."
        good = judgment(src.original_nl, candidate)
        agree = self.gate(good, good).evaluate(src, candidate, "low")
        self.assertTrue(agree.ok); self.assertEqual(agree.parser_status, "abstained")
        self.assertEqual((agree.second_opinion["ok"], agree.second_opinion["handle"]), (True, "second"))
        conflict = self.gate(good, judgment(src.original_nl, candidate, content_preserved=False)).evaluate(src, candidate, "low")
        self.assertIsNone(conflict.ok); self.assertIn("second_opinion_conflict", conflict.reasons)
        silent = self.gate(good, {key: True for key in DIMENSIONS}).evaluate(src, candidate, "low")
        self.assertIsNone(silent.ok); self.assertIn("second_opinion_unavailable", silent.reasons)
        alone = self.gate(good).evaluate(src, candidate, "low")
        self.assertTrue(alone.ok); self.assertEqual(alone.second_opinion, {})
        rejected = self.gate(judgment(src.original_nl, candidate, content_preserved=False), good).evaluate(src, candidate, "low")
        self.assertIs(rejected.ok, False); self.assertEqual(rejected.second_opinion, {})   # never consulted after a rejection
        self.assertEqual(self.gate(good, good).provenance()["second_opinion"], {"handle": "second", "family": "other"})
        self.assertEqual(agree.to_dict()["parser_status"], "abstained")

    def test_calibration_reports_strict_and_deferred_pipelines(self):
        from experiments.proficiency_validation.validation.calibration import run_challenge, report_challenge
        from experiments.proficiency_validation.tests.test_offline import source as src_fn
        base = src_fn()
        class StubGate:
            def __init__(self, parser, validator_ok):
                self.parser, self.validator_ok = parser, validator_ok
            def judge_all(self, source, rewritten):
                return {"structural": {"ok": True, "reasons": [], "details": {}},
                        "treatment": {"ok": True, "reasons": []}, "parser": self.parser,
                        "semantic": {"ok": self.validator_ok, "reasons": []}}
        item = {"item_id": "x", "challenge_type": "polarity_flip", "expected": "broken", "source_id": base.source_id,
                "predicate": base.predicate_text, "predicate_name": base.predicate_name, "original_text": base.original_nl,
                "candidate_text": "P7 is not a werewolf.", "speaker": base.speaker_id, "phase": base.phase,
                "game_id": base.game_id, "speech_id": base.speech_id, "public_history": list(base.public_history),
                "referenced_speech": None, "provenance": "challenge|prior_history"}
        abstained = {"ok": False, "error": "predicate_not_recoverable", "structural_match": False, "comparison_basis": "source_predicate"}
        # flags are "flagged as broken": parser abstains, judge rejects -> flagged under both policies
        rows = run_challenge([item], StubGate(abstained, False))
        self.assertEqual((rows[0]["flags"]["pipeline"], rows[0]["flags"]["pipeline_strict"]), (True, True))
        # parser abstains, judge passes -> deferred policy passes it (not flagged); strict policy has no verdict
        rows = run_challenge([item], StubGate(abstained, True))
        self.assertEqual((rows[0]["flags"]["pipeline"], rows[0]["flags"]["pipeline_strict"]), (False, None))
        # recovered mismatch is flagged under both policies even when the judge passes
        rows = run_challenge([item], StubGate({"ok": True, "structural_match": False, "comparison_basis": "source_predicate"}, True))
        self.assertEqual((rows[0]["flags"]["pipeline"], rows[0]["flags"]["pipeline_strict"]), (True, True))
        report = report_challenge(rows)
        self.assertIn("pipeline_strict", report["overall"])


class ThreeWayListenerTests(unittest.TestCase):
    """Listeners may answer A, B or same; the signed preference and a position score are game-level data."""

    def rows(self, answers, pair=("low", "neutral"), listener="m", game="g"):
        out = []
        for i, (ab, ba) in enumerate(answers):
            for order, pos in (("ab", ab), ("ba", ba)):
                left, right = (pair if order == "ab" else pair[::-1])
                chosen = None if pos == "same" else (left if pos == "A" else right)
                out.append({"bundle_id": f"{game}:P{i}", "game_id": game, "pair": ":".join(pair), "treatment": pair[0],
                            "comparator": pair[1], "listener": listener, "order": order, "ok": True, "choice": pos,
                            "chosen_condition": chosen, "chose_treatment": None if chosen is None else chosen == pair[0]})
        return out

    def test_parse_three_way_reads_same_and_letters(self):
        from experiments.proficiency_validation.validation.perception import parse_three_way, prompt, VERSIONS
        for raw, want in (("A", "A"), ("passage B", "B"), ("same", "same"), ("No meaningful difference", "same"),
                          ("neither", "same"), ("same.", "same"), ("", None), ("C", None),
                          ("cannot judge", None), ("not the same", None), ("none", None),
                          ("same vocabulary but different syntax", None)):
            self.assertEqual(parse_three_way(raw), want, raw)
        for construct in VERSIONS:
            self.assertIn('"same"', prompt(construct))

    def test_all_same_answers_give_zero_signed_and_zero_position_scores(self):
        rows = self.rows([("same", "same")] * 3, game="g1") + self.rows([("same", "same")] * 3, game="g2")
        r = listener_analysis(rows, n_boot=50, seed=1)["low:neutral"]["listeners"]["m"]
        self.assertEqual((r["signed_preference"], r["tie_rate"], r["position_bias"]["score"]), (0., 1., 0.))
        self.assertIsNone(r["share_treatment_more_limited"]); self.assertIsNone(r["bradley_terry"])
        self.assertEqual(r["order_consistency"]["same_condition"], 3 if False else 6)
        self.assertLess(r["signed_ci90"][0], -.2)
        self.assertGreater(r["signed_ci90"][1], .2)
        self.assertIn("Hoeffding", r["signed_interval_method"])
        for order in ("ab", "ba"):
            self.assertLess(r["by_order"][order]["ci90"][0], -.2)
        self.assertLess(r["position_bias"]["ci90"][0], -.2)

    def test_unanimous_games_have_uncertainty_and_failed_rows_have_no_score(self):
        from experiments.proficiency_validation.validation.analysis import signed_value, signed_bootstrap
        self.assertIsNone(signed_value({"signed": 1, "ok": False}))
        for sign in (-1, 0, 1):
            result = signed_bootstrap({f"g{i}": [sign] for i in range(3)}, n_boot=50, seed=1)
            self.assertEqual(result["estimate"], sign)
            self.assertLess(result["ci_low"], result["ci_high"])
            self.assertLessEqual(result["ci_low"], 0)
            self.assertGreaterEqual(result["ci_high"], 0)
        single = signed_bootstrap({"g1": [0]}, n_boot=50, seed=1)
        self.assertIsNone(single["ci_low"])

    def test_always_choosing_slot_a_cancels_in_the_signed_score_but_shows_as_position_bias(self):
        rows = self.rows([("A", "A")] * 4, game="g1") + self.rows([("A", "A")] * 4, game="g2")
        r = listener_analysis(rows, n_boot=50, seed=1)["low:neutral"]["listeners"]["m"]
        self.assertEqual(r["signed_preference"], 0.)
        self.assertEqual(r["position_bias"]["score"], 1.)
        self.assertEqual(r["order_consistency"]["always_a"], 8)
        self.assertEqual(r["by_order"]["ab"]["signed"], 1.); self.assertEqual(r["by_order"]["ba"]["signed"], -1.)

    def test_consistent_treatment_choice_with_some_ties(self):
        rows = self.rows([("A", "B"), ("A", "B"), ("same", "B")], game="g1") + self.rows([("A", "B"), ("A", "same")], game="g2")
        r = listener_analysis(rows, n_boot=50, seed=1)["low:neutral"]["listeners"]["m"]
        self.assertGreater(r["signed_preference"], .7); self.assertLess(r["signed_preference"], 1.)
        self.assertEqual(r["n_ties"], 2); self.assertEqual(r["share_treatment_more_limited"], 1.)
        self.assertEqual(r["order_consistency"]["same_condition"], 3)      # ties break order consistency
        self.assertLess(abs(r["position_bias"]["score"]), .1)         # ties fall unevenly on the two slots

    def test_criteria_use_the_signed_scale_and_position_margin(self):
        f = design()
        row = {"n_games": 10, "n_bundles": 20, "signed_preference": .6, "signed_ci95": [.4, .8],
               "by_order": {o: {"n_games": 10, "n_bundles": 20, "signed": .6, "ci95": [.4, .8]} for o in ("ab", "ba")},
               "position_bias": {"score": .5, "ci90": [.3, .7]}}
        report = {"listener": {"low:neutral": {"listeners": {"listener1": row, "all_models": row}}}}
        crit = {c["criterion"]: c for c in frozen_criteria(report, f)}
        self.assertIs(crit["Listener listener1: A vs B, both orders"]["passed"], True)
        self.assertIs(crit["Listener position bias within margin: low:neutral"]["passed"], False)
        self.assertIsNone(crit["Listener position bias within margin: neutral:original"]["passed"])


class GuardrailNonInferiorityTests(unittest.TestCase):
    def rows(self, low_minus_neutral, n_games=8):
        out = []
        for g in range(n_games):
            for b in range(2):
                for c, value in (("neutral", .5), ("low", .5 + low_minus_neutral + (g % 2) * .01)):
                    out.append({"bundle_id": f"g{g}:P{b}", "game_id": f"g{g}", "condition": c, "complete": True,
                                "cola_mean": value, "grammar_errors_per_100_words": .3 if c == "neutral" else .3 - low_minus_neutral + (g % 2) * .01})
        return out

    def test_constant_game_differences_leave_noninferiority_unevaluated(self):
        from experiments.proficiency_validation.validation.analysis import contrast_analysis
        rows = self.rows(0, n_games=3)
        for row in rows:
            row["cola_mean"] = .5
            row["grammar_errors_per_100_words"] = 0.
        margins = {"cola_mean": .05, "grammar_errors_per_100_words": .05}
        result = contrast_analysis(rows, list(margins), frozen={"noninferiority_margins": margins},
                                   n_boot=100, seed=1, family="guardrail")
        for metric in margins:
            ni = result["low_vs_neutral"]["metrics"][metric]["noninferiority"]
            self.assertIsNone(ni["noninferior"])
            self.assertIsNone(ni["one_sided_95_bound"])

    def test_low_better_than_neutral_is_non_inferior_and_worse_than_margin_is_not(self):
        from experiments.proficiency_validation.validation.analysis import contrast_analysis
        frozen = {"noninferiority_margins": {"cola_mean": .05, "grammar_errors_per_100_words": .05}}
        better = contrast_analysis(self.rows(+.2), ["cola_mean", "grammar_errors_per_100_words"], frozen=frozen, n_boot=200, seed=1, family="guardrail")
        m = better["low_vs_neutral"]["metrics"]["cola_mean"]["noninferiority"]
        self.assertIs(m["noninferior"], True); self.assertEqual(m["harm_direction"], "lower is worse")
        self.assertIs(better["low_vs_neutral"]["metrics"]["grammar_errors_per_100_words"]["noninferiority"]["noninferior"], True)
        worse = contrast_analysis(self.rows(-.2), ["cola_mean", "grammar_errors_per_100_words"], frozen=frozen, n_boot=200, seed=1, family="guardrail")
        self.assertIs(worse["low_vs_neutral"]["metrics"]["cola_mean"]["noninferiority"]["noninferior"], False)
        self.assertIs(worse["low_vs_neutral"]["metrics"]["grammar_errors_per_100_words"]["noninferiority"]["noninferior"], False)
        self.assertNotIn("noninferiority", worse.get("neutral_vs_original", {}).get("metrics", {}).get("cola_mean", {}))   # control stays equivalence
        unfrozen = contrast_analysis(self.rows(+.2), ["cola_mean"], frozen=None, n_boot=200, seed=1, family="guardrail")
        self.assertIsNone(unfrozen["low_vs_neutral"]["metrics"]["cola_mean"]["noninferiority"]["noninferior"])

    def test_criteria_name_non_inferiority_for_the_treatment_and_equivalence_for_the_control(self):
        f = design()
        names = [c["criterion"] for c in frozen_criteria({}, f)]
        self.assertTrue(any(n == "A vs B: cola_mean not meaningfully worse (non-inferiority)" for n in names))
        self.assertTrue(any(n.startswith("B vs Original: mattr equivalent") for n in names))
        self.assertIn("B vs Original: cola_mean equivalent", names)
        self.assertIn("B vs Original: grammar_errors_per_100_words equivalent", names)
        self.assertFalse(any("A vs B: cola_mean equivalent" in n for n in names))
        broken = dict(f); del broken["noninferiority_margins"]
        self.assertTrue(any("non-inferiority margin" in e for e in validate_design(broken)))


class PowerPlanningTests(unittest.TestCase):
    def test_games_required_follow_the_paired_normal_approximation_and_the_sign_flip_floor(self):
        from experiments.proficiency_validation.validation import power
        self.assertEqual(power.sign_flip_minimum_games(.05), 6)
        self.assertEqual(power.games_for_difference(.5, .5, power=.8), 8)          # d_z = 1 -> ~7.85 games
        self.assertEqual(power.games_for_difference(2., .1, power=.8), 6)          # huge effect: still the sign-flip floor
        self.assertIsNone(power.games_for_difference(0., .5)); self.assertIsNone(power.games_for_difference(.5, None))
        self.assertEqual(power.games_for_equivalence(.1, .1, power=.8), 9)           # (1.645 + 1.282)^2
        self.assertIsNone(power.games_for_equivalence(.1, .1, true_effect=.1))
        report = {"analysis": {"range": {"low_vs_neutral": {"metrics": {"mattr": {"n_games": 3, "estimate": -.02, "sd_diff": .01, "cohens_dz": -2}}},
                                         "neutral_vs_original": {"metrics": {"mattr": {"n_games": 3, "estimate": 0., "sd_diff": .005}}}},
                               "listener": {"low:neutral": {"listeners": {"all_models": {"n_games": 3, "signed_preference": .9, "sd_games": .1}}}}}}
        result = power.plan(report, equivalence_margins={"mattr": .01}, listener_margin=.2)
        self.assertEqual(result["scope"], "per_measure")
        kinds = {r["quantity"]: r for r in result["rows"]}
        self.assertEqual(kinds["A − B: mattr"]["games_80"], 6)
        self.assertEqual(kinds["B − Original: mattr equivalence"]["games_80"], power.games_for_equivalence(.005, .01))
        self.assertIn("| A − B: mattr |", power.render(result))

    def test_invalid_probabilities_fail_before_planning(self):
        from experiments.proficiency_validation.validation import power
        for alpha in (0, 1, -1, float("nan")):
            with self.assertRaises(ValueError):
                power.sign_flip_minimum_games(alpha)
        with self.assertRaises(ValueError):
            power.games_for_difference(.5, .5, power=1.)


class BooleanDimensionWordingTests(unittest.TestCase):
    def test_unambiguous_words_count_as_booleans_and_vague_ones_stay_missing(self):
        from experiments.proficiency_validation.rewrite.fidelity import response_check
        src, candidate = source(), "P7 is a wolf; he pushed the vote onto P2 and gave no read of his own."
        response = judgment(src.original_nl, candidate)
        response["interpersonal_register_preserved"] = "same"
        response["politeness_preserved"] = "preserved"
        missing, broken = response_check(response, src, candidate)
        self.assertEqual((missing, broken), ([], []))
        response["interpersonal_register_preserved"] = "changed"
        self.assertIn("not_preserved:interpersonal_register_preserved", response_check(response, src, candidate)[1])
        response["interpersonal_register_preserved"] = "partially"
        self.assertIn("interpersonal_register_preserved", response_check(response, src, candidate)[0])
