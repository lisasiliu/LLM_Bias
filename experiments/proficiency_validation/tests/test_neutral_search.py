"""The Neutral control is generated independently of Low: candidates, the gate, and a neutrality
rule that never reads an edit-distance target. Edit distance is recorded as exposure only."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from collections import Counter

from components.helpers import build_context
from components.metrics.edit_distance import normalized_token_edit_distance as distance
from components.rewriters.language_rewriter import LanguageProficiencyRewriter, select_candidate, candidate_profile
from experiments.proficiency_validation.validation.analysis import rewrite_exposure, neutral_search_diagnostics, exposure_lines
from experiments.proficiency_validation.rewrite.gate import GateResult, generate_with_gate
from experiments.proficiency_validation.tests.test_offline import FakeClient, source


def _constant_frequency(text):
    return 0.0

class RecordedGate:
    def __init__(self, verdicts=None):
        self.verdicts = verdicts or {}
        self.calls = Counter()

    def evaluate(self, src, text, condition, cheap_only=False):
        if text == src.original_nl:
            return GateResult(False, reasons=["identical_to_original"])
        if cheap_only:
            return GateResult(None, skipped=["parser", "semantic"])
        self.calls[text] += 1
        result = self.verdicts.get(text, GateResult(True, semantic={"ok": True, "dimensions": {"evidence": text}}))
        if isinstance(result, list):
            result = result[min(self.calls[text] - 1, len(result) - 1)]
        return copy.deepcopy(result)


class NeutralIndependenceTests(unittest.TestCase):
    def setUp(self):
        self.src = source(text="P7 is a werewolf because he pushed the vote onto P2 without ever giving a read of his own.")
        self.ctx = build_context(self.src)
        self.light = self.src.original_nl.replace("pushed", "pressed")
        self.heavier = self.light.replace("ever giving", "once offering").replace("a read of his own", "his own read")
        self.split = "P7 is a werewolf. He pressed the vote onto P2 and never once offered his own read."

    def run_search(self, rounds, gate=None):
        client = FakeClient([json.dumps(candidates) for candidates in rounds])
        rewriter = LanguageProficiencyRewriter(llm_client=client, condition="neutral")
        gate = gate or RecordedGate()
        gate.neutral_frequency = getattr(gate, "neutral_frequency", None) or _constant_frequency
        outcome = generate_with_gate(rewriter, gate, self.ctx, self.src, max_corrective_retries=len(rounds) - 1)
        return outcome, client, gate

    def test_prompt_never_mentions_low_or_an_edit_distance_target(self):
        rw = LanguageProficiencyRewriter(condition="neutral")
        system, user = rw.build_candidate_prompts(self.ctx)
        for text in (system, user):
            lowered = text.lower()
            for forbidden in ("edit distance", "edit-distance", "token edits", "low rewrite", "anchor"):
                self.assertNotIn(forbidden, lowered)
        self.assertIn("Reword every clause", system)
        _, retry = rw.build_candidate_prompts(self.ctx, feedback={"reasons": ["not_preserved:content_preserved"],
                                                                  "previous": self.light, "previous_candidates": [self.light]})
        self.assertIn("failed these checks", retry)
        self.assertIn("Candidates already considered", retry)
        self.assertNotIn("edit-distance", retry.lower())

    def test_selection_prefers_the_candidate_closest_to_the_original_surface_profile(self):
        selection = select_candidate([self.split, self.heavier, self.light], self.src.original_nl, seed=7, frequency=_constant_frequency)
        self.assertEqual(selection.candidates, [self.split, self.heavier, self.light])
        self.assertEqual(selection.chosen, self.light)                   # same sentence count, fewest word changes
        self.assertEqual(selection.rule, "closest_profile")
        self.assertEqual(len(selection.distances), 3)
        self.assertEqual(selection.profiles[0]["sentence_delta"], 1)       # the split candidate ranks last
        self.assertEqual(candidate_profile(self.light, self.src.original_nl, _constant_frequency), {"frequency_delta": 0.0, "sentence_delta": 0, "word_delta_ratio": 0.})

    def test_exact_profile_ties_break_by_the_fixed_seed_only(self):
        a = self.src.original_nl.replace("pushed", "pressed")
        b = self.src.original_nl.replace("pushed", "shoved")
        first = select_candidate([a, b], self.src.original_nl, seed=3, frequency=_constant_frequency).chosen
        self.assertEqual(first, select_candidate([a, b], self.src.original_nl, seed=3, frequency=_constant_frequency).chosen)
        self.assertEqual(first, select_candidate([b, a], self.src.original_nl, seed=3, frequency=_constant_frequency).chosen)   # order-independent
        self.assertEqual({select_candidate([a, b], self.src.original_nl, seed=s, frequency=_constant_frequency).chosen for s in range(40)}, {a, b})

    def test_first_round_with_a_fidelity_passer_is_accepted_whatever_its_distance(self):
        result, client, gate = self.run_search([[self.light, self.heavier]])
        self.assertEqual((result.status, result.text), ("ok", self.light))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(sum(gate.calls.values()), 2)
        selection = result.attempts[0].neutral_selection
        self.assertEqual(selection["n_passed_gate"], 2)
        self.assertAlmostEqual(selection["chosen_distance"], distance(self.src.original_nl, self.light))

    def test_semantic_rejection_is_fed_back_and_a_later_passer_is_accepted(self):
        gate = RecordedGate({self.light: GateResult(False, ["proposition_changed:0"])})
        result, client, gate = self.run_search([[self.light], [self.heavier]], gate=gate)
        self.assertEqual((result.status, result.text), ("ok", self.heavier))
        self.assertIn("proposition_changed:0", result.attempts[1].feedback_reasons)
        retry_prompt = client.calls[1]["messages"][1]["content"]
        self.assertIn(self.light, retry_prompt)                              # already-tried wording is listed
        self.assertEqual(gate.calls[self.light], 1)

    def test_all_rounds_rejected_is_a_fidelity_failure_with_complete_records(self):
        gate = RecordedGate({self.light: GateResult(False, ["not_preserved:commitment_preserved"])})
        result, client, gate = self.run_search([[self.light]] * 3, gate=gate)
        self.assertEqual(result.status, "treatment_unavailable")
        self.assertEqual(result.to_dict()["failure_kind"], "fidelity_failure")
        self.assertIsNone(result.text)
        self.assertEqual(gate.calls[self.light], 1)                          # the repeated candidate is judged once
        self.assertEqual([a.neutral_selection["n_reused_verdicts"] for a in result.attempts], [0, 1, 1])
        self.assertEqual(len(client.calls), 3)

    def test_only_copies_of_the_original_is_a_real_failure(self):
        result, _, _ = self.run_search([[self.src.original_nl, self.src.original_nl]] * 2)
        self.assertEqual(result.status, "treatment_unavailable")
        self.assertIn("identical_to_original", result.attempts[1].feedback_reasons)
        self.assertEqual(result.attempts[0].neutral_selection["rule"], "fallback_all_dropped")

    def test_no_verdict_is_not_cached_and_the_round_is_retried(self):
        gate = RecordedGate({self.light: [GateResult(None), GateResult(None), GateResult(True)]})
        result, _, gate = self.run_search([[self.light]] * 2, gate=gate)
        self.assertEqual(result.status, "ok")
        self.assertEqual(gate.calls[self.light], 3)
        self.assertEqual(result.attempts[-1].neutral_selection["n_reused_verdicts"], 0)

    def test_cache_does_not_cross_sources_or_generation_calls(self):
        gate = RecordedGate()
        self.run_search([[self.light]], gate=gate)
        self.run_search([[self.light]], gate=gate)
        self.assertEqual(gate.calls[self.light], 2)

    def test_casing_change_is_judged_separately(self):
        changed = self.light.replace("because", "BECAUSE")
        gate = RecordedGate({self.light: GateResult(False, ["proposition_changed:0"])})
        _, _, gate = self.run_search([[self.light], [changed]], gate=gate)
        self.assertEqual(dict(gate.calls), {self.light: 1, changed: 1})

    def test_exposure_report_describes_distances_without_a_matching_rule(self):
        rows = [{"source_id": sid, "condition": c, "status": "ok", "edit_distance_to_original": d}
                for sid, c, d in (("a", "low", .4), ("a", "neutral", .1), ("b", "low", .3), ("b", "neutral", .2),
                                  ("c", "low", .5), ("c", "neutral", .3), ("d", "low", .2))]
        rows.append({"source_id": "d", "condition": "neutral", "status": "treatment_unavailable", "original_text": "x y z",
                     "attempts": [{"neutral_selection": {"candidate_gates": [{"text": "x y", "ok": False}]}}]})
        exposure = rewrite_exposure(rows)
        self.assertEqual(exposure["conditions"]["low"]["n"], 4)
        pair = exposure["pairs"]["low_vs_neutral"]
        self.assertEqual(pair["n"], 3)
        self.assertAlmostEqual(pair["gap"]["mean"], .2)
        self.assertAlmostEqual(pair["neutral_at_least_half_of_treatment"], 2 / 3)
        self.assertTrue(0 <= pair["distribution_overlap"] <= 1)
        self.assertNotIn("fraction_within_tolerance", json.dumps(exposure))
        self.assertEqual(exposure["search"]["failed_without_fidelity_passer"], 1)
        text = "\n".join(exposure_lines(exposure))
        self.assertIn("describes rewrite exposure", text)
        self.assertIn("| A | 4 |", text)

    def test_search_diagnostics_count_repetition_and_rounds(self):
        row = {"source_id": "one", "condition": "neutral", "status": "ok", "original_text": self.src.original_nl,
               "attempts": [{"text": self.light, "neutral_selection": {"candidate_gates": [{"text": self.light, "ok": False}]}},
                            {"text": self.light, "neutral_selection": {"candidate_gates": [{"text": self.light, "ok": True, "reused_verdict": True}]}}]}
        snapshot = copy.deepcopy(row)
        audit = neutral_search_diagnostics([row])
        self.assertEqual((audit["repeated_candidate_occurrences"], audit["reused_verdicts"], audit["accepted_after_retry"]), (1, 1, 1))
        self.assertEqual(row, snapshot)


class ExposureFloorTests(unittest.TestCase):
    def setUp(self):
        self.src = source(text="P7 is a werewolf because he pushed the vote onto P2 without ever giving a read of his own.")
        self.ctx = build_context(self.src)
        self.near_copy = self.src.original_nl.replace("pushed", "pressed")                      # 1 of 18 tokens
        self.thorough = "P7 is a werewolf since he forced the vote onto P2 without once offering his own read."

    def run_search(self, rounds, floor):
        client = FakeClient([json.dumps(c) for c in rounds])
        gate = RecordedGate(); gate.neutral_exposure_floor = floor; gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(LanguageProficiencyRewriter(llm_client=client, condition="neutral"), gate,
                                     self.ctx, self.src, max_corrective_retries=len(rounds) - 1)

        return outcome, client

    def test_near_copies_are_rejected_with_feedback_and_a_thorough_passer_is_chosen(self):
        outcome, client = self.run_search([[self.near_copy], [self.near_copy, self.thorough]], floor=.10)
        self.assertEqual((outcome.status, outcome.text), ("ok", self.thorough))
        first = outcome.attempts[0]
        self.assertFalse(first.gate.ok); self.assertTrue(first.gate.reasons[0].startswith("below_exposure_floor"))
        self.assertEqual(first.neutral_selection["n_below_floor"], 1)
        self.assertIn("below_exposure_floor", client.calls[1]["messages"][1]["content"])
        self.assertEqual(outcome.attempts[1].neutral_selection["n_passed_gate"], 1)     # the near-copy is never judged
        self.assertEqual(outcome.attempts[1].neutral_selection["n_below_floor"], 1)

    def test_only_near_copies_is_a_recorded_failure_not_an_acceptance(self):
        from experiments.proficiency_validation.rewrite.gate import failure_cause
        outcome, _ = self.run_search([[self.near_copy]] * 2, floor=.10)
        self.assertEqual(outcome.status, "treatment_unavailable")
        self.assertEqual(failure_cause(outcome.to_dict()), "near_copy_only")
        self.assertEqual(sum(1 for a in outcome.attempts for g in a.neutral_selection["candidate_gates"] if g.get("semantic")), 0)
        accepted, _ = self.run_search([[self.near_copy]], floor=0.)
        self.assertEqual(accepted.status, "ok")                       # the floor is the only thing that changed


class FrequencySelectionTests(unittest.TestCase):
    def test_missing_frequency_fails_before_generation_even_with_original_fallback(self):
        from experiments.proficiency_validation.rewrite.gate import GatedRewriter, SemanticGate
        client = FakeClient([])
        with patch.dict(os.environ, {}, clear=True), tempfile.TemporaryDirectory() as tmp, \
             patch("components.judges.judge_model._dotenv_values", return_value={}):
            with self.assertRaisesRegex(ValueError, "SUBTLEX"):
                GatedRewriter(base="neutral", gate=SemanticGate(), llm_client=client,
                              on_unavailable="original", log_path=str(Path(tmp) / "rewrites.jsonl"))
            rw = LanguageProficiencyRewriter(condition="neutral", llm_client=client)
            src = source()
            with self.assertRaisesRegex(ValueError, "before generation"):
                generate_with_gate(rw, RecordedGate(), build_context(src), src)
            self.assertEqual(client.calls, [])
            self.assertFalse((Path(tmp) / "rewrites.jsonl").exists())

    def test_environment_frequency_runs_neutral_and_configuration_round_trips(self):
        from experiments.proficiency_validation.rewrite.gate import GatedRewriter, SemanticGate
        from lurkingwolf.rewriter import Rewriter
        src = source()
        candidate = "P7 is a werewolf because he pressed the vote onto P2 without once offering his own read."
        client = FakeClient([json.dumps([candidate])])
        gate = SemanticGate(judge_handle="deepseek-chat")
        with patch.dict(os.environ, {"SUBTLEX": "fixture.tsv"}), \
             patch("experiments.proficiency_validation.rewrite.__main__.word_frequency_scorer", return_value=_constant_frequency) as loader:
            rw = GatedRewriter(base="neutral", gate=gate, llm_client=client, spacy_model="fixture_spacy")
            loader.assert_called_once_with("fixture.tsv", "fixture_spacy")
            with patch.object(gate, "evaluate", return_value=GateResult(True)):
                self.assertEqual(rw.rewrite(build_context(src)), candidate)
            restored = Rewriter.from_config(rw.to_config(), llm_client=FakeClient([]))
            self.assertEqual(restored.to_config(), rw.to_config())
            self.assertEqual(restored.gate.neutral_spacy_model, "fixture_spacy")
        self.assertEqual(rw.log[-1]["status"], "ok")
        self.assertFalse(rw.log[-1].get("fallback_to_original", False))

    def test_missing_spacy_stops_frequency_initialization(self):
        from experiments.proficiency_validation.rewrite.__main__ import word_frequency_scorer
        with patch("components.metrics.lexical.SubtlexNorms"), \
             patch("experiments.proficiency_validation.validation.measurements.load_nlp_or_none", return_value=None):
            with self.assertRaisesRegex(ValueError, "spaCy"):
                word_frequency_scorer("fixture.tsv")

    def test_selection_prefers_the_candidate_closest_in_word_frequency(self):
        from components.rewriters.language_rewriter import select_candidate, NEUTRAL_SELECTION_RULE
        original = "P5 is lying about the seer check, so vote P5 out today."
        plain = "P5 is not telling the truth about the seer check, so vote P5 out today."
        fancy = "P5 is fabricating the seer check, so vote P5 out today."
        scores = {original: 3.0, plain: 3.1, fancy: 2.6}
        chosen = select_candidate([fancy, plain], original, seed=1, frequency=scores.get)
        self.assertEqual(chosen.chosen, plain)
        self.assertEqual(chosen.rule, NEUTRAL_SELECTION_RULE)
        self.assertEqual([round(p["frequency_delta"], 2) for p in chosen.profiles], [0.4, 0.1])
        with self.assertRaises(ValueError):
            select_candidate([fancy, plain], original, frequency=None)

    def test_gate_records_the_subtlex_file(self):
        from experiments.proficiency_validation.rewrite.gate import SemanticGate
        gate = SemanticGate(None, None, require_parser=False, require_validator=False, judge_handle="deepseek-chat")
        gate.neutral_frequency = _constant_frequency
        self.assertIsNone(gate.to_config()["subtlex_path"])
        with self.assertRaises(ValueError):
            gate.set_neutral_frequency(None)
