"""Offline checks for the flip-replay analysis (synthetic phase results, no engine)."""
import unittest

from experiments.proficiency_flip.code import analyze as fa


def phase(votes_received, executed=None, wolf=None, seer=None, trust=None, votes=None, alive=None):
    trust = trust or {}
    return {"votes_received": votes_received, "executed": executed, "wolf_target": wolf, "seer_target": seer,
            "reliability": trust, "votes": votes or {}, "alive_at_start": alive or ["P1", "P2", "P3"]}


class RunIdTests(unittest.TestCase):
    def test_run_ids_from_the_runner_parse(self):
        self.assertEqual(fa.parse_run_id("nl__d2"), {"mode": "nl", "condition": None, "self": None, "day": 2})
        self.assertEqual(fa.parse_run_id("flip_self__low__judge__generator__p3__d1"),
                         {"mode": "flip_self", "condition": "low", "self": "P3", "day": 1})
        self.assertEqual(fa.parse_run_id("flip_others__neutral__judge__generator__p12__d3")["self"], "P12")
        self.assertIsNone(fa.parse_run_id("pred_only__d1"))


class OutcomeTests(unittest.TestCase):
    def test_speaker_outcomes_use_other_players_trust_only(self):
        r = phase({"P1": 2}, executed="P1", wolf="P2", seer="P1",
                  trust={"P1": {"P1": 9, "P2": 1}, "P2": {"P1": 3}, "P3": {"P1": 5}})
        o = fa.speaker_outcomes(r, "P1")
        self.assertEqual((o["votes_received"], o["executed"], o["wolf_target"], o["seer_target"]), (2, 1, 0, 1))
        self.assertEqual(o["trust_in_speaker"], 4)
        self.assertIsNone(fa.speaker_outcomes(phase({}), "P9")["trust_in_speaker"])

    def test_paired_differences_are_grouped_by_game_and_use_identity_as_control(self):
        identity = phase({"P1": 1, "P2": 0}, votes={"P1": "P2", "P2": "P1"}, trust={"P2": {"P1": 4}, "P1": {"P2": 6}})
        low_p1 = phase({"P1": 3, "P2": 0}, executed="P1", trust={"P2": {"P1": 2}})
        neutral_p1 = phase({"P1": 1, "P2": 0}, trust={"P2": {"P1": 4}})
        others_low_p1 = phase({"P1": 1}, votes={"P1": "P3"}, trust={"P1": {"P2": 3}})
        rec = {"seed": "201", "day": 1, "baseline": {"role_dist": {"P1": "villager", "P2": "werewolf", "P3": "seer"}, **identity},
               "replays": {("nl", None, None): identity, ("flip_self", "low", "P1"): low_p1,
                           ("flip_self", "neutral", "P1"): neutral_p1, ("flip_others", "low", "P1"): others_low_p1},
               "expected": {}, "gate": {}}
        d = fa.paired_differences([rec])
        self.assertEqual(d["speaker"][("low", "identity")]["votes_received"]["201"], [2])
        self.assertEqual(d["speaker"][("low", "identity")]["executed"]["201"], [1])
        self.assertEqual(d["speaker"][("low", "identity")]["trust_in_speaker"]["201"], [-2])
        self.assertEqual(d["speaker"][("low", "neutral")]["votes_received"]["201"], [2])
        self.assertEqual(d["by_role"][("low", "identity")]["village"]["votes_received"]["201"], [2])
        self.assertEqual(d["own"]["low_vs_identity"]["changed_vote"]["201"], [1])
        self.assertEqual(d["own"]["low_vs_identity"]["trust_toward_others"]["201"], [-3])
        self.assertEqual(d["n_pairs"][("flip_self", "low", "identity")], 1)

    def test_game_weighted_summary_is_exact_for_few_games(self):
        e = fa.game_weighted({"a": [1.0, 3.0], "b": [2.0], "c": [2.0]})
        self.assertEqual((e["n_pairs"], e["n_games"], e["estimate"]), (4, 3, 2.0))
        self.assertEqual(e["p_sign_flip"], 0.25)
        self.assertEqual(fa.game_weighted({})["n_pairs"], 0)

    def test_render_handles_empty_blocks(self):
        s = fa.summarize([])
        text = fa.render(s)
        self.assertIn("Replay availability", text)
        self.assertIn("Pilot", text)


if __name__ == "__main__":
    unittest.main()
