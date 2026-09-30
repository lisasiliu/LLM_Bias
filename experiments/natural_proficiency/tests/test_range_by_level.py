"""Offline checks for the prompt-level range analysis (no spaCy, no model)."""
import unittest

from experiments.natural_proficiency.code import range_by_level as rbl


def utterance(level, game, speaker, sid, text):
    return {"id": f"{level}/{game}:{sid}", "level": level, "game": game, "speech_id": sid,
            "phase": "day_1_discussion", "speaker": speaker, "role": "villager",
            "predicate": "estimate", "predicate_text": "estimate(D1, P1, P2, werewolf)", "text": text}


class PassageTests(unittest.TestCase):
    def test_passages_follow_speech_order_and_eligibility_uses_the_pilot_minimum(self):
        rows = [utterance("L1", "game01", "P1", "s10", "second " * 100),
                utterance("L1", "game01", "P1", "s2", "first " * 100),
                utterance("L1", "game01", "P2", "s3", "short text.")]
        passages = rbl.assemble_passages(rows)
        self.assertEqual([p["speaker"] for p in passages], ["P1", "P2"])
        self.assertTrue(passages[0]["text"].startswith("first"))
        self.assertEqual(passages[0]["speech_ids"], ["s2", "s10"])
        self.assertTrue(passages[0]["eligible"])
        self.assertFalse(passages[1]["eligible"])
        measured = rbl.measure(passages, nlp=None)
        self.assertIsNotNone(measured[0]["mattr"])
        self.assertNotIn("mattr", measured[1])          # ineligible passages carry no metrics

    def test_contrast_pairs_games_by_seed_and_weights_games_equally(self):
        features = []
        for game, (l1, l5) in {"game01": (0.5, 0.7), "game02": (0.4, 0.6), "game03": (0.6, 0.8)}.items():
            features.append({"level": "L1", "game": game, "speaker": "P1", "eligible": True, "mattr": l1})
            features.append({"level": "L1", "game": game, "speaker": "P2", "eligible": True, "mattr": l1})
            features.append({"level": "L5", "game": game, "speaker": "P1", "eligible": True, "mattr": l5})
        features.append({"level": "L5", "game": "game99", "speaker": "P1", "eligible": True, "mattr": 0.1})  # unpaired
        contrast = rbl.paired_contrast(features, "L1", "L5", "mattr")
        self.assertEqual(contrast["n_games"], 3)
        self.assertAlmostEqual(contrast["estimate"], -0.2)
        self.assertEqual(contrast["sign_flip_method"], "exact")
        self.assertEqual(rbl.level_estimate(features, "L5", "mattr")["n_games"], 4)
        self.assertAlmostEqual(rbl.level_estimate(features, "L1", "mattr")["estimate"], 0.5)

    def test_unknown_level_is_rejected_before_any_parsing(self):
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "utterances.jsonl"
            path.write_text(json.dumps(utterance("L9", "game01", "P1", "s1", "x " * 200)) + "\n")
            with self.assertRaisesRegex(ValueError, "L9"):
                rbl.analyze(path)


if __name__ == "__main__":
    unittest.main()
