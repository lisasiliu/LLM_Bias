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


class TrustAndNoiseTests(unittest.TestCase):
    def test_repeat_identity_run_id(self):
        self.assertEqual(fa.parse_run_id("nl__repeat1__d2"), {"mode": "nl", "condition": "repeat1", "self": None, "day": 2})
        self.assertEqual(fa.parse_run_id("nl__d2")["condition"], None)

    def test_trust_counts_village_assessors_alive_after_the_night_only(self):
        r = {"role_dist": {"P1": "villager", "P2": "werewolf", "P3": "seer", "P4": "villager", "P5": "villager"},
             "alive_at_start": ["P1", "P2", "P3", "P4", "P5"], "executed": "P4", "night_killed": "P5",
             "reliability": {"P2": {"P1": 10}, "P3": {"P1": 6}, "P4": {"P1": 2}, "P5": {"P1": 2}, "P1": {"P1": 9}}}
        self.assertEqual(fa.assessors(r, "P1"), ["P3"])
        self.assertEqual(fa.speaker_outcomes(r, "P1")["trust_in_speaker"], 6)
        self.assertIsNone(fa.own_behaviour(r, "P2", r, "werewolf")["trust_toward_others"])
        r["reliability"]["P3"] = {"P1": 6, "P2": 3, "P4": 9}
        self.assertEqual(fa.own_behaviour(r, "P3", r, "seer")["trust_toward_others"], 4.5)

    def test_holm_family_is_the_prespecified_set(self):
        self.assertEqual(fa.holm({"trust": None, "votes": .03})["votes"]["p_holm"], .06)

    def test_identity_repeat_gives_the_noise_floor(self):
        identity = phase({"P1": 1}, wolf="P1", trust={"P2": {"P1": 7}, "P3": {"P1": 5}}, votes={"P1": "P2"})
        repeat = phase({"P1": 0}, wolf="P3", trust={"P2": {"P1": 6}, "P3": {"P1": 5}}, votes={"P1": "P3"})
        rec = {"seed": "201", "day": 1, "baseline": {**identity, "role_dist": {"P1": "villager", "P2": "villager", "P3": "villager"}},
               "replays": {("nl", None, None): identity, ("nl", "repeat1", None): repeat}, "expected": {}, "gate": {}, "exposure": {}}
        d = fa.paired_differences([rec])
        self.assertEqual(d["repeat_noise"]["votes_received"]["201"]["P1"], [1])
        self.assertEqual(d["repeat_noise"]["trust_in_speaker"]["201"]["P1"], [0.5])
        self.assertEqual(d["repeat_noise"]["changed_vote"]["201"]["P1"], [1])
        self.assertNotIn("trust_in_speaker", fa.summarize([rec])["identity_vs_baseline"])
        self.assertIn("identity repeat", fa.render(fa.summarize([rec])))


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
        self.assertEqual(d["speaker"][("low", "identity")]["votes_received"]["201"]["P1"], [2])
        self.assertEqual(d["speaker"][("low", "identity")]["executed"]["201"]["P1"], [1])
        self.assertEqual(d["speaker"][("low", "identity")]["trust_in_speaker"]["201"]["P1"], [-2])
        self.assertEqual(d["speaker"][("low", "neutral")]["votes_received"]["201"]["P1"], [2])
        self.assertEqual(d["by_role"][("low", "identity")]["village"]["votes_received"]["201"]["P1"], [2])
        self.assertEqual(d["own"]["low_vs_identity"]["changed_vote"]["201"]["P1"], [1])
        self.assertEqual(d["own"]["low_vs_identity"]["trust_toward_others"]["201"]["P1"], [-3])
        self.assertEqual(d["n_pairs"][("flip_self", "low", "identity")], 1)

    def test_game_weighted_summary_is_exact_for_few_games(self):
        e = fa.game_weighted({"a": {"P1": [1.0], "P2": [3.0]}, "b": {"P1": [2.0]}, "c": {"P1": [2.0]}})
        self.assertEqual((e["n_pairs"], e["n_games"], e["estimate"]), (4, 3, 2.0))
        self.assertEqual(e["p_sign_flip"], 0.25)
        self.assertEqual(fa.game_weighted({})["n_pairs"], 0)

    def test_day_counts_do_not_change_speaker_weights(self):
        records = []
        for day, alive in ((1, ["P1", "P2"]), (2, ["P1"])):
            identity = phase({p: 0 for p in alive}, alive=alive)
            replays = {("nl", None, None): identity}
            for speaker in alive:
                replays[("flip_self", "neutral", speaker)] = identity
                replays[("flip_self", "low", speaker)] = phase(
                    {speaker: 2 if speaker == "P2" else 0}, alive=alive)
            records.append({"seed": "201", "day": day, "baseline": {**identity,
                            "role_dist": {"P1": "villager", "P2": "villager"}},
                            "replays": replays, "expected": {}, "gate": {}})
        result = fa.summarize(records)
        effect = result["flip_self"]["low_vs_neutral"]["votes_received"]
        self.assertEqual(effect["estimate"], 1.0)
        self.assertEqual((effect["n_pairs"], effect["n_speakers"]), (3, 2))
        self.assertAlmostEqual(effect["pooled_mean"], 2 / 3)
        self.assertEqual(result["flip_self_per_protocol"]["low_vs_neutral"]["votes_received"]["n_pairs"], 0)

    def test_missing_empty_partial_and_failed_logs_are_not_complete_exposure(self):
        key = ("flip_self", "low", "P1")
        meta = {"mode": "flip_self", "condition": "low", "self": "P1"}
        result = {"speeches": [{"speaker": "P1", "nl": [{"text": "first"}, {"text": "second"}]}]}
        good = [{"speaker": "P1", "condition": "low", "status": "ok", "rewritten_text": text}
                for text in ("first", "second")]
        self.assertFalse(fa.clean_exposure({}, key))
        for rows in ([], good[:1], good[:1] + good[:1], good + [{**good[0], "speaker": "P2"}],
                     [{**r, "status": "infrastructure_unavailable"} for r in good],
                     [{**r, "condition": "neutral"} for r in good],
                     [{**r, "rewritten_text": "different"} for r in good]):
            with self.subTest(rows=rows):
                exposure = fa.exposure_summary(rows, result, meta)
                self.assertFalse(fa.clean_exposure({"exposure": {key: exposure}}, key))
        valid = fa.exposure_summary(good, result, meta)
        self.assertTrue(fa.clean_exposure({"exposure": {key: valid}}, key))
        fallback = [{**r, "status": "treatment_unavailable", "fallback_to_original": True,
                     "original_text": r["rewritten_text"]} for r in good]
        exposure = fa.exposure_summary(fallback, result, meta)
        self.assertTrue(exposure["verified"])
        self.assertFalse(fa.clean_exposure({"exposure": {key: exposure}}, key))

    def test_last_words_rows_count_as_exposure_without_a_recorded_speech(self):
        key = ("flip_others", "low", "P1")
        meta = {"mode": "flip_others", "condition": "low", "self": "P1"}
        result = {"speeches": [{"speaker": p, "nl": [{"text": p}]} for p in ("P1", "P2", "P3")]}
        rows = [{"speaker": p, "condition": "low", "status": "ok", "rewritten_text": p, "phase": "day_speech"}
                for p in ("P2", "P3")]
        rows.append({"speaker": "P3", "condition": "low", "status": "ok", "rewritten_text": "farewell",
                     "phase": "last_speech", "speech_id": None})
        exposure = fa.exposure_summary(rows, result, meta)
        self.assertEqual((exposure["rewritten"], exposure["last_words"], exposure["expected"]), (3, 1, 2))
        self.assertTrue(exposure["verified"])
        self.assertTrue(fa.clean_exposure({"replays": {key: result}, "gate": {}, "expected": {},
                                           "exposure": {key: exposure}}, key))
        exposure = fa.exposure_summary(rows[:1] + rows[2:], result, meta)
        self.assertFalse(exposure["verified"])

    def test_rows_of_an_interrupted_attempt_precede_the_completed_attempt(self):
        key = ("flip_self", "low", "P1")
        meta = {"mode": "flip_self", "condition": "low", "self": "P1"}
        result = {"speeches": [{"speaker": "P1", "nl": [{"text": "first"}, {"text": "second"}]}]}
        complete = [{"speaker": "P1", "condition": "low", "status": "ok", "rewritten_text": text,
                     "phase": "day_speech"} for text in ("first", "second")]
        interrupted = [{**complete[0], "status": "treatment_unavailable", "fallback_to_original": True,
                        "original_text": "first", "rewritten_text": None}]
        exposure = fa.exposure_summary(interrupted + complete, result, meta)
        self.assertEqual((exposure["interrupted_rows"], exposure["rewritten"], exposure["fallback"]), (1, 2, 0))
        self.assertTrue(exposure["verified"])
        self.assertTrue(fa.clean_exposure({"replays": {key: result}, "gate": {}, "expected": {},
                                           "exposure": {key: exposure}}, key))
        exposure = fa.exposure_summary(interrupted + complete, None, meta)
        self.assertEqual((exposure["interrupted_rows"], exposure["fallback"], exposure["verified"]), (0, 1, False))

    def test_surrounding_exposure_matches_other_speakers_and_counts_unknown_logs(self):
        meta = {"mode": "flip_others", "condition": "neutral", "self": "P1"}
        result = {"speeches": [{"speaker": p, "nl": [{"text": p}]} for p in ("P1", "P2", "P3")]}
        rows = [{"speaker": p, "condition": "neutral", "status": "ok", "rewritten_text": p} for p in ("P2", "P3")]
        key = ("flip_others", "neutral", "P1")
        exposure = fa.exposure_summary(rows, result, meta)
        rec = {"replays": {key: result}, "gate": {}, "expected": {}, "exposure": {key: exposure}}
        self.assertTrue(fa.clean_exposure(rec, key))
        self.assertEqual(fa.availability([rec])["exposure"]["flip_others:neutral"]["clean_replays"], 1)
        rec["exposure"] = {}
        summary = fa.availability([rec])["exposure"]["flip_others:neutral"]
        self.assertEqual((summary["clean_replays"], summary["unverified_replays"]), (0, 1))

    def test_render_handles_empty_blocks(self):
        s = fa.summarize([])
        text = fa.render(s)
        self.assertIn("Replay availability", text)
        self.assertIn("Proficiency flip replays", text)


if __name__ == "__main__":
    unittest.main()
