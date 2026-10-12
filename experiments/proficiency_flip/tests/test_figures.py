import json
import tempfile
import unittest
from pathlib import Path

from experiments.proficiency_flip.code import figures as ff
from experiments.proficiency_flip.code import export_passages as ep


def phase(votes, reliability, wolf="P3", executed="P2"):
    return {"votes": dict(votes), "reliability": {a: dict(t) for a, t in reliability.items()},
            "wolf_target": wolf, "seer_target": "P4", "guard_target": None, "executed": executed}


def est(x, lo, hi, n=4):
    return {"estimate": x, "ci_low": lo, "ci_high": hi, "n_pairs": n}


class FigureTests(unittest.TestCase):
    def records(self):
        records = []
        for seed in ("201", "202", "203"):
            identity = phase({"P1": "P2", "P2": "P1"}, {"P1": {"P2": 5}, "P2": {"P1": 6}})
            replays = {("nl", None, None): identity, ("nl", "repeat1", None): phase({"P1": "P2", "P2": "P3"}, {"P1": {"P2": 5}, "P2": {"P1": 6}})}
            for cond, flip in (("low", 2), ("neutral", 1), ("high", 0)):
                for speaker in ("P1", "P2"):
                    changed = phase({"P1": "P2", "P2": "P3" if flip else "P1"},
                                    {"P1": {"P2": 5 - flip}, "P2": {"P1": 6}}, wolf="P1" if flip > 1 else "P3")
                    replays[("flip_self", cond, speaker)] = changed
                    replays[("flip_others", cond, speaker)] = identity
            records.append({"seed": seed, "day": 1, "baseline": {**identity, "role_dist": {}},
                            "replays": replays, "expected": {}, "gate": {}, "exposure": {}})
        return records

    def test_change_rates_and_the_report_aggregation(self):
        rows = ff.phase_rows(self.records())
        low = [r for r in rows if r["mode"] == "flip_self" and r["condition"] == "low"]
        self.assertEqual(len(low), 6)
        self.assertEqual((low[0]["vote"], low[0]["skill"], low[0]["reliability"]), (0.5, 0.5, 0.5))
        self.assertEqual([r["vote"] for r in rows if r["condition"] == "baseline"], [0.0, 0.0, 0.0])
        self.assertEqual([r["vote"] for r in rows if r["condition"] == "repeat"], [0.5, 0.5, 0.5])
        s = ff.rate_stats(rows, "flip_self", "low", "vote")
        self.assertEqual((s["n_games"], s["estimate"]), (3, 0.5))
        d = ff.paired_rate_stats(rows, "flip_self", "low", "high", "skill")
        self.assertEqual((d["n_games"], d["estimate"]), (3, 0.5))
        self.assertIn("p_sign_flip", d)
        # a player seen on more days does not weigh more than one seen once
        values = {"201": {"P1": [1.0, 1.0, 1.0], "P2": [0.0]}}
        self.assertEqual(ff.game_level(values)["estimate"], 0.5)

    def utterances(self):
        return [{"seed": "201", "day": 1, "mode": "flip_self", "self": "P1", "speaker": "P1", "condition": c,
                 "replay": f"flip_self__{c}__p1__d1", "fallback": c == "high",
                 "source": "I think P2 is the wolf. We should vote.",
                 "delivered": "P2 is wolf. Vote P2." if c == "low" else "I think P2 is the wolf. We should vote."}
                for c in ("low", "neutral", "high")]

    def test_all_figures_are_written(self):
        summary = {"games": ["201", "202", "203"],
                   "flip_self": {k: {"trust_in_speaker": est(-0.1, -0.2, 0.0), "votes_received": est(0.3, -0.1, 0.9)}
                                 for k, _ in ff.CONTRASTS},
                   "flip_self_by_role": {k: {side: {"trust_in_speaker": est(-0.05, -0.2, 0.1)}
                                             for side in ("werewolf", "village")} for k, _ in ff.CONTRASTS},
                   "flip_others": {k: {"changed_vote": est(0.2, 0.1, 0.4), "trust_toward_others": est(0.04, 0.0, 0.1)}
                                   for k, _ in ff.CONTRASTS}}
        scorer = lambda text: {"n_words": len(text.split()), "n_sentences": text.count(".") or 1, "subtlex_mean": 3.0}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            stats = ff.make_all(self.records(), summary, out, utterances=self.utterances(), scorer=scorer)
            names = {p.name for p in out.iterdir()}
            for stem in ("change_rates", "scope_contrasts", "speaker_outcomes", "self_response", "manipulation_check"):
                self.assertIn(f"{stem}.png", names)
                self.assertIn(f"{stem}.pdf", names)
            self.assertEqual(stats["manipulation_check"]["low:delivered:words"]["n_obs"], 1)
            self.assertEqual(stats["manipulation_check"]["low:delivered:words"]["estimate"], 5)
            self.assertIn("identity:repeat:reliability", stats["change_rates"])
            self.assertAlmostEqual(json.loads((out / "figure_stats.json").read_text())
                                   ["change_rates"]["flip_self:low:vote"]["estimate"], 0.5)

    def test_passages_pair_delivered_speech_with_its_source(self):
        rows = ep.passages(self.utterances())
        self.assertEqual(len(rows), 6)
        low = {r["condition"]: r for r in rows if r["bundle_id"].endswith(":low")}
        self.assertEqual(low["low"]["text"], "P2 is wolf. Vote P2.")
        self.assertEqual(low["original"]["text"], "I think P2 is the wolf. We should vote.")
        self.assertTrue(low["low"]["complete"] and low["original"]["complete"])
        self.assertEqual({r["bundle_id"] for r in rows if r["condition"] == "high"} ,
                         {"201:d1:flip_self:p1:P1:high"})
        self.assertEqual([r["n_fallbacks"] for r in rows if r["condition"] == "high"], [1])


if __name__ == "__main__":
    unittest.main()
