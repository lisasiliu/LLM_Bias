"""Offline checks for the politeness flip analysis (synthetic phase results, no engine)."""
import unittest

from experiments.politeness_flip.code import analyze as pa


def phase(votes_received, executed=None, trust=None, votes=None, roles=None):
    return {"votes_received": votes_received, "executed": executed, "wolf_target": None, "seer_target": None,
            "reliability": trust or {}, "votes": votes or {}, "alive_at_start": ["P1", "P2", "P3"],
            "role_dist": roles or {"P1": "villager", "P2": "villager", "P3": "villager"}}


class DesignTests(unittest.TestCase):
    def test_the_four_strategies_parse_out_of_the_runner_s_run_ids(self):
        for strategy in pa.CONDITIONS:
            meta = pa.flip.parse_run_id(f"flip_self__{strategy}__p1__d1", conditions=pa.CONDITIONS)
            self.assertEqual((meta["mode"], meta["condition"], meta["self"], meta["day"]),
                             ("flip_self", strategy, "P1", 1))
        self.assertIsNone(pa.flip.parse_run_id("flip_self__low__p1__d1", conditions=pa.CONDITIONS)["condition"])
        self.assertEqual(pa.flip.parse_run_id("nl__repeat1__d2", conditions=pa.CONDITIONS)["condition"], "repeat1")

    def test_contrasts_cover_the_untreated_replay_and_the_unredressed_strategy(self):
        self.assertEqual(len(pa.CONTRASTS), 7)
        self.assertEqual(sum(1 for _, c in pa.CONTRASTS if c == "identity"), 4)
        self.assertEqual({t for t, c in pa.CONTRASTS if c == pa.REFERENCE},
                         set(pa.CONDITIONS) - {pa.REFERENCE})
        self.assertNotIn((pa.REFERENCE, pa.REFERENCE), pa.CONTRASTS)

    def test_the_config_the_runner_reads_matches_this_design(self):
        from lurkingwolf.config import load_config
        cfg = load_config(pa.EXPERIMENT_DIR / "config.yaml")
        comp = cfg.comparisons[pa.COMPARISON]
        strategies = {s.rewriter_params["strategy"] for s in comp.replays if s.rewriter_class}
        self.assertEqual(strategies, set(pa.CONDITIONS))
        self.assertEqual(len([s for s in comp.replays if s.mode == "nl"]), 2)


class SummaryTests(unittest.TestCase):
    def records(self):
        records = []
        for seed in ("201", "202", "203"):
            identity = phase({"P1": 1}, executed="P2", trust={"P2": {"P1": 5}, "P3": {"P1": 5}}, votes={"P1": "P2"})
            replays = {("nl", None, None): identity,
                       ("nl", "repeat1", None): phase({"P1": 1}, executed="P2",
                                                      trust={"P2": {"P1": 5}, "P3": {"P1": 6}}, votes={"P1": "P2"})}
            for i, strategy in enumerate(pa.CONDITIONS):
                treated = phase({"P1": 1 + i}, executed="P2", trust={"P2": {"P1": 5 + i}, "P3": {"P1": 5}},
                                votes={"P1": "P3" if i else "P2"})
                for speaker in ("P1",):
                    replays[("flip_self", strategy, speaker)] = treated
                    replays[("flip_others", strategy, speaker)] = treated
            records.append({"seed": seed, "day": 1, "baseline": identity, "replays": replays,
                            "expected": {}, "gate": {}, "exposure": {}})
        return records

    def test_every_contrast_is_estimated_and_holm_covers_the_strategy_family(self):
        s = pa.summarize(self.records())
        self.assertEqual((len(s["games"]), s["n_checkpoints"]), (3, 3))
        for treatment, control in pa.CONTRASTS:
            self.assertIn(f"{treatment}_vs_{control}", s["flip_self"])
        # positive_politeness is the second strategy: votes_received +1 against bald_on_record
        self.assertEqual(s["flip_self"]["positive_politeness_vs_bald_on_record"]["votes_received"]["estimate"], 1)
        self.assertEqual(s["flip_self"]["off_record_vs_bald_on_record"]["votes_received"]["estimate"], 3)
        self.assertEqual(s["flip_self"]["bald_on_record_vs_identity"]["votes_received"]["estimate"], 0)
        family = s["holm_vs_reference"]["votes_received"]
        self.assertEqual(set(family), {f"{c}_vs_{pa.REFERENCE}" for c in pa.CONDITIONS if c != pa.REFERENCE})
        for adj in family.values():
            self.assertGreaterEqual(adj["p_holm"], adj["p"])
        # P2 was executed, so only P3 scores the speaker: 5 in the identity replay, 6 in its repeat
        self.assertEqual(s["identity_vs_repeat"]["trust_in_speaker"]["estimate"], -1.0)

    def test_availability_counts_expected_and_present_under_the_same_label(self):
        records = self.records()
        for rec in records:
            rec["expected"] = {f"flip_self__{c}__p1__d1": 0 for c in pa.CONDITIONS}
            rec["expected"]["nl__d1"] = 0
        a = pa.summarize(records)["availability"]["replays"]
        for c in pa.CONDITIONS:
            self.assertEqual((a[f"flip_self:{c}"]["expected"], a[f"flip_self:{c}"]["present"]), (3, 3))
        self.assertNotIn("expected", a.get("flip_self", {}))

    def test_render_names_the_strategies_and_survives_an_empty_cohort(self):
        text = pa.render(pa.summarize(self.records()))
        self.assertIn("Politeness flip replays", text)
        for strategy in pa.CONDITIONS:
            self.assertIn(strategy.replace("_", " "), text)
        self.assertIn("identity repeat", text)
        empty = pa.render(pa.summarize([]))
        self.assertIn("Replay availability", empty)


if __name__ == "__main__":
    unittest.main()
