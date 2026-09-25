"""End-to-end provenance, source isolation, missing-data and resume regressions."""
import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from components.helpers import SourceRecord, read_jsonl, write_jsonl, parse_history_entry
from components.judges.judge_model import Completion
from components.stats import tost_equivalence
from experiments.proficiency_validation.code import cli, report
from experiments.proficiency_validation.code.analysis import listener_analysis
from experiments.proficiency_validation.code.checkpoint import Checkpoint
from experiments.proficiency_validation.code.common import DATA_DIR
from experiments.proficiency_validation.code.freeze import main_freeze
from experiments.proficiency_validation.code.frozen import enforce_frozen
from experiments.proficiency_validation.code.generate import generate
from experiments.proficiency_validation.code.human_validation import select_items
from experiments.proficiency_validation.code.listener import import_human_worksheet
from experiments.proficiency_validation.code.semantic_gate import SemanticGate
from experiments.proficiency_validation.tests.test_offline import source
from experiments.proficiency_validation.tests.test_protocol import design


class SourceReportTests(unittest.TestCase):
    def test_report_uses_untreated_games_and_reproduces_the_committed_document(self):
        result = report.summarize(DATA_DIR)
        self.assertEqual(result["corpus"]["n_utterances"], 280)
        self.assertEqual(result["corpus"]["n_games"], 3)
        self.assertEqual(result["corpus"]["predicates"]["claim_divine"], 20)
        self.assertEqual(result["n_bundles"], 24)
        self.assertEqual(result["validation_status"], "NOT YET VALIDATED")
        self.assertEqual(result["provenance"]["validation_scope"],
                         "automated linguistic measures and model perception")
        self.assertIn("The validation scope is measured linguistic range and model perception",
                      report.render(result))
        self.assertEqual(report.render(result), (DATA_DIR.parent / "VALIDATION.md").read_text())
        for row in read_jsonl(DATA_DIR / "sources.jsonl"):
            self.assertNotIn("rewritten_text", row)
            self.assertNotIn("requested_strategy", row)
            self.assertTrue(row["provenance"].startswith("game_run:experiments/validation/"))
            own = int(row["speech_id"][1:])
            for entry in row["public_history"]:
                sid = parse_history_entry(entry).get("sid")
                if sid and sid.startswith("s") and sid[1:].isdigit():
                    self.assertLess(int(sid[1:]), own)

    def test_source_text_tampering_is_detected_before_reporting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (*report.ARTIFACTS, "provenance.json"):
                (root / name).write_bytes((DATA_DIR / name).read_bytes())
            with (root / "sources.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                report.summarize(root)

    def test_old_politeness_benchmark_is_not_a_proficiency_command(self):
        actions = cli.build_parser()._subparsers._group_actions[0].choices
        self.assertNotIn("benchmark-judges", actions)
        self.assertIn("prepare-data", actions)
        self.assertFalse((DATA_DIR / "judge_benchmark_human.json").exists())
        self.assertFalse((DATA_DIR / "challenge_set.jsonl").exists())


class MissingEvidenceTests(unittest.TestCase):
    def test_zero_variance_differences_do_not_prove_equivalence(self):
        self.assertIsNone(tost_equivalence({"g1": [0.], "g2": [0.]}, .1, n_boot=30)["equivalent"])

    def test_blank_human_choices_are_retained(self):
        key = {"item_001": {"bundle_id": "g:p", "game_id": "g", "pair": "low:neutral",
                            "treatment": "low", "comparator": "neutral", "left_condition": "low",
                            "right_condition": "neutral"}}
        rows = import_human_worksheet("### item_001\nmore_limited:\n", key, "r1")
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["ok"])
        stat = listener_analysis(rows, n_boot=20, seed=1)["low:neutral"]["listeners"]["all_humans"]
        self.assertEqual(stat["n_abstained"], 1)
        self.assertEqual(stat["abstention_rate"], 1)

    def test_model_requires_both_orders_for_each_comparison(self):
        row = {"bundle_id": "g:p", "game_id": "g", "pair": "low:neutral", "listener": "model",
               "listener_family": "fake", "treatment": "low", "comparator": "neutral", "order": "ab",
               "ok": True, "chosen_condition": "low", "chose_treatment": True}
        self.assertEqual(listener_analysis([row], n_boot=20, seed=1), {})

    def test_calibration_selection_does_not_depend_on_gate_verdict(self):
        rows = [{"source_id": f"g{i}:s1", "game_id": f"g{i}", "condition": c,
                 "original_text": "original", "rewritten_text": "rewrite", "status": "ok",
                 "attempts": [{"gate": {"ok": i % 2 == 0}}]}
                for i in range(8) for c in ("low", "high")]
        first = select_items(rows, "calibration", 6, 101)
        changed = copy.deepcopy(rows)
        for row in changed:
            row["attempts"][0]["gate"]["ok"] = not row["attempts"][0]["gate"]["ok"]
        second = select_items(changed, "calibration", 6, 101)
        selection = lambda items: [(r["source_id"], r["condition"]) for r in items.values()]
        self.assertEqual(selection(first), selection(second))
        self.assertEqual(len({r["game_id"] for r in first.values()}), 6)

    def test_frozen_analysis_cannot_pass_without_empirical_evidence(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            f = design()
            for split, bundles in f["splits"].items():
                write_jsonl(root / (split + ".jsonl"), bundles)
                write_jsonl(root / (split + "_sources.jsonl"), [{"source_id": b["source_ids"][0]} for b in bundles])
            (root / "decisions.json").write_text(json.dumps(f))
            (root / "subtlex.txt").write_text("Word\tLg10CD\nword\t1.0\n")
            args = SimpleNamespace(decisions=root / "decisions.json",
                pilot_bundles=root / "pilot.jsonl", set_a_bundles=root / "set_a.jsonl", set_b_bundles=root / "set_b.jsonl",
                pilot_sources=root / "pilot_sources.jsonl", set_a_sources=root / "set_a_sources.jsonl",
                set_b_sources=root / "set_b_sources.jsonl", subtlex=root / "subtlex.txt", out=root / "frozen.json")
            with patch("experiments.proficiency_validation.code.freeze.model_identity",
                       side_effect=lambda h: f["models"]["identities"][h]):
                self.assertEqual(main_freeze(args), 0)
            self.assertEqual(enforce_frozen(args.out)["schema_version"], 3)
            self.assertNotIn("human_validation", enforce_frozen(args.out))
            self.assertNotIn("calibration", enforce_frozen(args.out))
            code = cli.main(["analyze", "--frozen", str(args.out), "--split", "set_a",
                             "--json", str(root / "result.json"), "--out", str(root / "result.md")])
            self.assertEqual(code, 1)
            result = json.loads((root / "result.json").read_text())
            self.assertFalse(result["verdict"]["confirmed"])
            self.assertGreater(result["verdict"]["not_evaluated"], 0)
            self.assertTrue(any("High vs Neutral" in c["criterion"] for c in result["criteria"]))


class CheckpointTests(unittest.TestCase):
    def test_resume_preserves_failures_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rows.jsonl"
            first = Checkpoint(output, {"model": "fake"})
            first.append({"source_id": "s1", "status": "infrastructure_unavailable"})
            first.close()
            with self.assertRaisesRegex(ValueError, "settings differ"):
                Checkpoint(output, {"model": "changed"}, resume=True)
            resumed = Checkpoint(output, {"model": "fake"}, resume=True)
            self.assertEqual(resumed.rows[0]["status"], "infrastructure_unavailable")
            resumed.append({"source_id": "s2", "status": "ok"})
            resumed.finish()
            resumed.close()
            self.assertEqual(len(read_jsonl(output)), 2)
            self.assertFalse(Path(str(output) + ".partial").exists())

    def test_generation_resume_reuses_the_gated_anchor(self):
        src = source()
        class Model:
            spec = SimpleNamespace(model="fake", handle="fake", provider="local")
            model_served = "fake"
            calls = 0
            def complete(self, **kwargs):
                answers = [
                    "P7 is a werewolf because he redirected the vote toward P2 without offering his own read.",
                    json.dumps(["P7 is a werewolf since he pushed the vote onto P2 without giving his own read."])]
                answer = answers[self.calls]
                self.calls += 1
                return Completion(answer, "fake", "fake")
            def provenance(self):
                return {"identity": {"model": "fake"}}
        model = Model()
        previous = {"source_id": src.source_id, "condition": "low", "status": "ok", "attempts": [],
                    "rewritten_text": "P7 is a werewolf. He pushed the vote onto P2. He gave no read."}
        gate = SemanticGate(require_parser=False, require_validator=False)
        rows = generate([src], conditions=["low", "neutral", "high"], generator=model, gate=gate,
                        completed_rows=[previous], log=lambda *a: None)
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], previous)
        self.assertEqual(rows[-1]["status"], "ok")
        self.assertEqual(set(rows[-1]["anchor_outcomes"]), {"low", "high"})


if __name__ == "__main__":
    unittest.main()
