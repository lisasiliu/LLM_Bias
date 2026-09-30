import copy
import json
import unittest
from collections import Counter

from components.helpers import build_context
from components.metrics.edit_distance import normalized_token_edit_distance as distance
from components.rewriters.language_rewriter import LanguageProficiencyRewriter
from experiments.proficiency_validation.validation.analysis import neutral_matching, neutral_search_diagnostics
from experiments.proficiency_validation.rewrite.gate import GateResult, generate_with_gate
from experiments.proficiency_validation.tests.test_offline import FakeClient, source


class RecordedGate:
    neutral_edit_tolerance = .10

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


class NeutralSearchTests(unittest.TestCase):
    def setUp(self):
        self.src = source(text="P7 is a werewolf because he pushed the vote onto P2 without ever giving a read of his own.")
        self.ctx = build_context(self.src)
        self.weak = self.src.original_nl.replace("pushed", "pressed")
        self.better = self.weak.replace("ever giving", "once offering")
        self.matched = self.better.replace("a read of his own", "his own read")

    def run_search(self, rounds, gate=None, target=.40):
        client = FakeClient([json.dumps(candidates) for candidates in rounds])
        rewriter = LanguageProficiencyRewriter(llm_client=client, condition="neutral")
        gate = gate or RecordedGate()
        outcome = generate_with_gate(rewriter, gate, self.ctx, self.src,
            target_edit_distance=target, max_corrective_retries=len(rounds) - 1)
        return outcome, client, gate

    def test_first_prompt_has_matching_guidance_without_invented_failure(self):
        rw = LanguageProficiencyRewriter(condition="neutral")
        _, user = rw.build_candidate_prompts(self.ctx, feedback={"target_edit_distance": .4, "edit_tolerance": .1})
        self.assertNotIn("previous rewrite", user.lower())
        self.assertNotIn("unspecified", user)
        self.assertIn("0.300000 to 0.500000", user)
        self.assertIn("substitutions, insertions and deletions", user)
        self.assertIn("roughly 7.6 token edits", user)

    def test_retry_feedback_uses_actual_distance_and_direction(self):
        rw = LanguageProficiencyRewriter(condition="neutral")
        for target, direction in ((.4, "Increase"), (0., "Reduce")):
            _, user = rw.build_candidate_prompts(self.ctx, feedback={"target_edit_distance": target,
                "previous": self.better, "edit_tolerance": .1,
                "reasons": ["neutral_edit_distance_mismatch:gap=0.25,tolerance=0.1"]})
            self.assertIn(f"Measured candidate distance: {distance(self.src.original_nl, self.better):.6f}", user)
            self.assertIn(direction + " the amount of rewording", user)
            self.assertIn("passed the fidelity checks but missed", user)
            self.assertNotIn("failed these semantic checks", user)

    def test_repeated_exact_candidate_is_judged_once_and_still_fails_matching(self):
        result, client, gate = self.run_search([[self.weak]] * 3)
        self.assertEqual(result.status, "treatment_unavailable")
        self.assertEqual(result.to_dict()["failure_kind"], "control_matching_failure")
        self.assertIsNone(result.text)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(gate.calls[self.weak], 1)
        selections = [a.neutral_selection for a in result.attempts]
        self.assertEqual([s["n_reused_verdicts"] for s in selections], [0, 1, 1])
        self.assertEqual(selections[-1]["candidate_gates"][0]["semantic"]["dimensions"]["evidence"], self.weak)
        self.assertIn("Candidates already considered", client.calls[-1]["messages"][1]["content"])

    def test_best_fidelity_passer_survives_worse_round_and_copy(self):
        result, client, _ = self.run_search([[self.better], [self.weak], [self.src.original_nl]])
        self.assertEqual([a.text for a in result.attempts], [self.better] * 3)
        self.assertIsNone(result.text)  # retained is not accepted outside the band
        self.assertEqual(result.attempts[-1].neutral_selection["carried_candidate"]["text"], self.better)
        self.assertIn("Closest candidate that passed fidelity checks so far: " + self.better,
                      client.calls[-1]["messages"][1]["content"])

    def test_retry_accepts_new_in_band_candidate(self):
        target = distance(self.src.original_nl, self.matched)
        self.assertGreater(target - distance(self.src.original_nl, self.weak), .1)
        result, _, gate = self.run_search([[self.weak], [self.matched]], target=target)
        self.assertEqual((result.status, result.text), ("ok", self.matched))
        self.assertEqual(sum(gate.calls.values()), 2)

    def test_distance_match_cannot_override_semantic_rejection(self):
        gate = RecordedGate({self.matched: GateResult(False, ["proposition_changed"])})
        result, _, _ = self.run_search([[self.matched]] * 3, gate=gate,
                                      target=distance(self.src.original_nl, self.matched))
        self.assertEqual(result.status, "treatment_unavailable")
        self.assertEqual(gate.calls[self.matched], 1)
        self.assertEqual(result.attempts[-1].gate.reasons, ["proposition_changed"])

    def test_no_verdict_is_not_cached_as_rejection_or_acceptance(self):
        gate = RecordedGate({self.matched: [GateResult(None), GateResult(None), GateResult(True)]})
        result, _, _ = self.run_search([[self.matched]] * 2, gate=gate,
                                      target=distance(self.src.original_nl, self.matched))
        self.assertEqual(result.status, "ok")
        self.assertEqual(gate.calls[self.matched], 3)
        self.assertEqual(result.attempts[-1].neutral_selection["n_reused_verdicts"], 0)

    def test_cache_does_not_cross_sources_or_generation_calls(self):
        gate = RecordedGate()
        self.run_search([[self.weak]], gate=gate)
        self.run_search([[self.weak]], gate=gate)
        self.assertEqual(gate.calls[self.weak], 2)

    def test_casing_change_is_judged_separately_across_rounds(self):
        changed = self.weak.replace("because", "BECAUSE")
        _, _, gate = self.run_search([[self.weak], [changed]])
        self.assertEqual(dict(gate.calls), {self.weak: 1, changed: 1})

    def test_tolerance_boundary_ignores_floating_point_roundoff(self):
        target = distance(self.src.original_nl, self.better) + .1
        result, _, _ = self.run_search([[self.better]], target=target)
        self.assertEqual(result.status, "ok")
        rows = [{"source_id": "boundary", "condition": c, "status": "ok", "edit_distance_to_original": d}
                for c, d in (("low", .4), ("neutral", .3))]
        self.assertEqual(neutral_matching(rows, tolerance=.1)["fraction_within_tolerance"], 1.)

    def test_archive_diagnostic_counts_repetition_and_lost_best_candidate(self):
        row = {"source_id": "one", "condition": "neutral", "status": "treatment_unavailable",
               "original_text": self.src.original_nl, "target_edit_distance": .4,
               "gate_provenance": {"neutral_edit_tolerance": .1},
               "attempts": [{"text": text, "neutral_selection": {"candidate_gates": [{"text": text, "ok": True}]}}
                            for text in (self.better, self.weak, self.weak)]}
        snapshot = copy.deepcopy(row)
        audit = neutral_search_diagnostics([row])
        self.assertEqual(audit["repeated_candidate_occurrences"], 1)
        self.assertEqual(audit["failed_final_selection_worse_than_best_passer"], 1)
        self.assertEqual(audit["best_passer_below_band"], 1)
        self.assertEqual(audit["failed_with_eligible_recorded_candidate"], 0)
        self.assertEqual(row, snapshot)

    def test_archive_diagnostic_exposes_missed_eligible_candidate(self):
        row = {"condition": "neutral", "status": "treatment_unavailable", "original_text": self.src.original_nl,
               "target_edit_distance": distance(self.src.original_nl, self.matched),
               "gate_provenance": {"neutral_edit_tolerance": .1},
               "attempts": [{"neutral_selection": {"candidate_gates": [{"text": self.matched, "ok": True}]}}]}
        self.assertEqual(neutral_search_diagnostics([row])["failed_with_eligible_recorded_candidate"], 1)

    def test_matching_report_keeps_excluded_anchor_distances_visible(self):
        rows = [{"source_id": sid, "condition": "low", "status": "ok", "edit_distance_to_original": d}
                for sid, d in (("accepted", .1), ("failed", .4), ("not_attempted", .7))]
        rows += [{"source_id": "accepted", "condition": "neutral", "status": "ok", "edit_distance_to_original": .1},
                 {"source_id": "failed", "condition": "neutral", "status": "treatment_unavailable", "attempts": [{}]}]
        result = neutral_matching(rows, tolerance=.1)
        self.assertEqual(result["paired_means"]["treatment"], .1)
        self.assertEqual(result["failed_neutral_attempts"], {"n": 1, "mean_treatment_distance": .4})
