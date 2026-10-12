"""Offline checks for the politeness manipulation check (no model calls)."""
import json
import tempfile
import unittest
from pathlib import Path

from experiments.politeness_flip.code import analyze as pa
from experiments.politeness_flip.code import manipulation_check as mc


def write_replay(root, seed, day, mode, strategy, player, rows):
    d = root / f"run_qwen_politeness_flip_{seed}" / pa.COMPARISON / f"replays/day_{day}" / \
        f"{mode}__{strategy}__{player.lower()}__d{day}"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "rewrites.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def row(text, speaker="P2", phase="day_speech", status="ok"):
    return {"speaker": speaker, "phase": phase, "predicate": "estimate(...)", "status": status,
            "original_text": "原", "rewritten_text": text}


class SamplingTests(unittest.TestCase):
    def tree(self, tmp):
        """Two games; `flip_others` yields five times as many utterances as `flip_self`,
        as it does in the real replays."""
        root = Path(tmp)
        for seed in ("201", "202"):
            for strategy in pa.CONDITIONS:
                write_replay(root, seed, 1, "flip_self", strategy, "P1",
                             [row(f"{seed}-{strategy}-self-{i}") for i in range(4)]
                             + [row("last words", phase="last_speech"), row("", status="treatment_unavailable")])
                write_replay(root, seed, 1, "flip_others", strategy, "P1",
                             [row(f"{seed}-{strategy}-others-{i}") for i in range(20)])
        return root

    def test_only_delivered_discussion_utterances_are_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            us = mc.delivered_utterances(self.tree(tmp))
        self.assertEqual(len(us), 2 * len(pa.CONDITIONS) * 24)
        self.assertTrue(all(u["phase"] == "day_speech" for u in us))
        self.assertEqual({u["requested"] for u in us}, set(pa.CONDITIONS))
        self.assertEqual(len({u["item_id"] for u in us}), len(us))

    def test_the_sample_balances_strategies_scopes_and_games_and_is_deterministic(self):
        from collections import Counter
        with tempfile.TemporaryDirectory() as tmp:
            us = mc.delivered_utterances(self.tree(tmp))
        picked = mc.sample(us, per_condition=8, seed=7)
        self.assertEqual(Counter(u["requested"] for u in picked), Counter({c: 8 for c in pa.CONDITIONS}))
        # the scopes are balanced although flip_others has five times the material
        self.assertEqual(Counter(u["mode"] for u in picked), Counter({"flip_self": 16, "flip_others": 16}))
        self.assertEqual(Counter(u["game_id"] for u in picked), Counter({"201": 16, "202": 16}))
        self.assertEqual([u["item_id"] for u in picked], [u["item_id"] for u in mc.sample(us, 8, 7)])
        self.assertEqual(len(mc.sample(us, per_condition=1000, seed=7)), len(us))


class ClassifierResolutionTests(unittest.TestCase):
    def test_a_handle_from_the_model_manifest_resolves_to_the_served_model(self):
        import json, os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({"models": {"classifier_qwen": {
                "model": "Qwen/Qwen3.5-27B", "family": "qwen", "base_url_env": "GEN_URL",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}}}))
            with patch.dict(os.environ, {"PROFICIENCY_MODEL_CONFIG": str(manifest),
                                         "GEN_URL": "http://127.0.0.1:8000"}):
                model = mc.ValidationModel.resolve("classifier_qwen", purpose="politeness_classifier")
        self.assertEqual((model.spec.model, model.spec.base_url),
                         ("Qwen/Qwen3.5-27B", "http://127.0.0.1:8000"))
        self.assertEqual(model.extra_body, {"chat_template_kwargs": {"enable_thinking": False}})


class ReportTests(unittest.TestCase):
    def passes(self):
        rows = []
        # bald_on_record is labelled correctly 3/3; off_record is read as negative politeness
        for item, requested, labels in (
                ("a", "bald_on_record", ["bald_on_record"] * 3),
                ("b", "bald_on_record", ["bald_on_record", "bald_on_record", "ambiguous_or_mixed"]),
                ("c", "off_record", ["negative_politeness"] * 3),
                ("d", "positive_politeness", ["positive_politeness", "off_record", "positive_politeness"])):
            for p, label in enumerate(labels, 1):
                rows.append({"item_id": item, "pass": p, "classifier": "judge", "requested": requested,
                             "game_id": "201", "mode": "flip_self", "label": label, "ok": True})
        return rows

    def test_majority_vote_agreement_and_confusion(self):
        t = mc.tally(self.passes())["judge"]
        self.assertEqual(t["n_items"], 4)
        self.assertEqual(t["agreement"]["bald_on_record"], 1.0)
        self.assertEqual(t["agreement"]["off_record"], 0.0)
        # the confusion counts utterances, not passes: this one item was read as negative politeness
        self.assertEqual(t["confusion"]["off_record"], {"negative_politeness": 1})
        self.assertAlmostEqual(t["overall_agreement"], 0.75)

    def test_a_failed_pass_does_not_become_a_label_and_is_counted(self):
        rows = [{"item_id": "e", "pass": p, "classifier": "judge", "requested": "off_record",
                 "game_id": "201", "mode": "flip_self", "label": None, "ok": False} for p in (1, 2, 3)]
        t = mc.tally(rows)["judge"]
        self.assertEqual(t["agreement"]["off_record"], 0.0)
        self.assertEqual(t["confusion"]["off_record"], {"failed": 1})
        self.assertEqual((t["failed_passes"], t["n_passes"], t["failed_pass_rate"]), (3, 3, 1.0))

    def test_every_item_appears_in_its_confusion_row(self):
        rows = self.passes() + [{"item_id": "f", "pass": p, "classifier": "judge", "requested": "off_record",
                                 "game_id": "202", "mode": "flip_others",
                                 "label": l, "ok": True}
                                for p, l in enumerate(["bald_on_record", "off_record", "positive_politeness"], 1)]
        t = mc.tally(rows)["judge"]
        counted = sum(sum(c.values()) for c in t["confusion"].values())
        self.assertEqual(counted, t["n_items"])
        self.assertEqual(t["confusion"]["off_record"].get("no_majority"), 1)
        self.assertEqual(set(t["by_scope"]), {"flip_self", "flip_others"})

    def test_two_classifiers_are_reported_side_by_side(self):
        rows = self.passes() + [{**r, "classifier": "other"} for r in self.passes()]
        summary = {"created_at": "now", "classifiers": mc.tally(rows)}
        self.assertEqual(set(summary["classifiers"]), {"judge", "other"})
        text = mc.render(summary)
        self.assertIn("Classifier `judge`", text)
        self.assertIn("bald on record", text)


if __name__ == "__main__":
    unittest.main()
