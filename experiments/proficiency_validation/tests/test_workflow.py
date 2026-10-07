import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from components.helpers import read_jsonl, write_jsonl, parse_history_entry, write_back_record
from components.judges.judge_model import Completion
from components.stats import tost_equivalence
import experiments.proficiency_validation.validation.__main__ as cli
import experiments.proficiency_validation.validation.report as report
from experiments.proficiency_validation.validation.analysis import listener_analysis
from experiments.proficiency_validation.rewrite.runtime import Checkpoint, configure_run
from experiments.proficiency_validation.rewrite.runtime import DATA_DIR, RUNS_DIR, EXPERIMENT_DIR, read_metadata, update_metadata
from experiments.proficiency_validation.rewrite.runtime import artifact_provenance, record_artifact
from experiments.proficiency_validation.validation.measurements import verify_guardrail_materials
from experiments.proficiency_validation.validation.export import _rows as export_rows
from experiments.proficiency_validation.validation.protocol import main_freeze
from experiments.proficiency_validation.validation.protocol import enforce_frozen
from experiments.proficiency_validation.rewrite.__main__ import generate
from experiments.proficiency_validation.rewrite.gate import SemanticGate
from experiments.proficiency_validation.tests.test_offline import source
from experiments.proficiency_validation.tests.test_protocol import design


def _constant_frequency(text):
    return 0.0

class SourceReportTests(unittest.TestCase):
    def test_report_uses_untreated_games_and_reproduces_the_committed_document(self):
        result = report.summarize(DATA_DIR, RUNS_DIR / "pilot_qwen35_27b")
        self.assertEqual(result["corpus"]["n_utterances"], 280)
        self.assertEqual(result["corpus"]["n_games"], 3)
        self.assertEqual(result["corpus"]["predicates"]["claim_divine"], 20)
        self.assertEqual(result["n_bundles"], 24)
        self.assertEqual(result["validation_status"], "pilot")
        self.assertEqual(result["provenance"]["validation_scope"],
                         "automated linguistic measures and model perception")
        self.assertEqual(report.render(result), (EXPERIMENT_DIR / "VALIDATION.md").read_text())
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
            for name in (*report.ARTIFACTS, "manifest.json"):
                (root / name).write_bytes((DATA_DIR / name).read_bytes())
            with (root / "sources.jsonl").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "provenance"):
                report.summarize(root)

    def test_old_politeness_benchmark_is_not_a_proficiency_command(self):
        actions = cli.build_parser()._subparsers._group_actions[0].choices
        self.assertNotIn("benchmark-judges", actions)
        self.assertIn("prepare-data", actions)
        self.assertTrue(set(actions).isdisjoint({
            "build-human-qc", "import-human-qc", "calibrate-english-gate",
            "build-human-worksheet", "import-human-worksheet"}))
        options = actions["analyze"]._option_string_actions
        self.assertNotIn("--human-qc", options)
        self.assertNotIn("--human-qc-key", options)
        self.assertFalse((DATA_DIR / "judge_benchmark_human.json").exists())
        self.assertFalse((DATA_DIR / "challenge_set.jsonl").exists())


class GuardrailProvenanceTests(unittest.TestCase):
    def materials(self, measured):
        assembled = measured / "linguistic_features_assembled.jsonl"
        guardrails = measured / "grammar_guardrails.jsonl"
        rows = [{"bundle_id": "g:P3", "game_id": "g", "speaker": "P3", "condition": c,
                 "complete": True, "text": "P7 is a werewolf."} for c in ("original", "low", "neutral")]
        write_jsonl(assembled, rows)
        record_artifact(assembled, artifact_provenance(None, "pilot"))
        write_jsonl(guardrails, [{k: v for k, v in r.items() if k != "text"} |
                                {"grammar_errors_per_100_words": 0., "cola_mean": .9} for r in rows])
        record_artifact(guardrails, artifact_provenance(None, "pilot", inputs={"assembled": assembled}))
        write_jsonl(measured / "linguistic_features.jsonl", rows)
        return assembled, guardrails

    def test_valid_materials_are_verified_and_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            measured = Path(tmp) / "validation"
            assembled, guardrails = self.materials(measured)
            verify_guardrail_materials(guardrails, assembled)
            rows = export_rows(measured, "primary")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["grammar_errors_per_100_words"] == 0. and r["cola_mean"] == .9 for r in rows))

    def test_tampered_guardrails_stop_both_report_cohorts_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            for cohort in ("validation", "validation/whole_game"):
                with self.subTest(cohort=cohort):
                    root = Path(tmp) / cohort.replace("/", "_")
                    measured = root / cohort
                    _, guardrails = self.materials(measured)
                    rows = read_jsonl(guardrails)
                    rows[0]["grammar_errors_per_100_words"] = 99.
                    write_jsonl(guardrails, rows)
                    with self.assertRaisesRegex(ValueError, "guardrail provenance"):
                        report.summarize_pilot(root / "sources", root)
                    with self.assertRaisesRegex(ValueError, "guardrail provenance"):
                        export_rows(measured, cohort)

    def test_missing_and_changed_passage_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            measured = Path(tmp) / "validation"
            for change in ("missing_guardrail_metadata", "missing_passage_metadata", "changed_passage", "new_passage_hash"):
                with self.subTest(change=change):
                    assembled, guardrails = self.materials(measured)
                    if change == "missing_guardrail_metadata":
                        update_metadata(guardrails, "provenance", {})
                    elif change == "missing_passage_metadata":
                        update_metadata(assembled, "provenance", {})
                    else:
                        rows = read_jsonl(assembled)
                        rows[0]["text"] = "P7 is a villager."
                        write_jsonl(assembled, rows)
                        if change == "new_passage_hash":
                            record_artifact(assembled, artifact_provenance(None, "pilot"))
                    with self.assertRaisesRegex(ValueError, "guardrail provenance"):
                        verify_guardrail_materials(guardrails, assembled)

    def test_passage_identity_and_coverage_must_match_even_with_valid_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            measured = Path(tmp) / "validation"
            for change in ("speaker", "game_id", "complete", "missing", "duplicate"):
                with self.subTest(change=change):
                    assembled, guardrails = self.materials(measured)
                    rows = read_jsonl(guardrails)
                    if change == "missing":
                        rows.pop()
                    elif change == "duplicate":
                        rows.append(dict(rows[0]))
                    else:
                        rows[0][change] = False if change == "complete" else "other"
                    write_jsonl(guardrails, rows)
                    record_artifact(guardrails, artifact_provenance(None, "pilot", inputs={"assembled": assembled}))
                    with self.assertRaisesRegex(ValueError, "identities|duplicate"):
                        verify_guardrail_materials(guardrails, assembled)


class WriteBackTests(unittest.TestCase):
    def test_engine_round_trip_preserves_predicate_reference_and_treatment(self):
        src = source("agree(D1, P3, s3)", "I agree with P1 that P7 is a werewolf.")
        src.referenced_speech = {"speech_id": "s3", "speaker": "P1",
                                 "predicate": "estimate(D1, P1, P7, werewolf)", "text": "P7 is a werewolf."}
        text = "P1 is right. P7 is a werewolf."
        record = {"validated": True, "source_id": src.source_id, "strategy": "low",
                  "predicate_text": src.predicate_text, "rewritten_nl": text}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = write_back_record(record, src, root, treatment_kind="proficiency")
            self.assertTrue(result.ok, result.reasons)
            self.assertEqual(result.speech_id, "s4")
            self.assertEqual(result.predicate_after, src.predicate_text)
            self.assertEqual(result.args_after, src.predicate_args)
            self.assertEqual(result.text_after, text)
            history = json.loads((root / "public_history.json").read_text())["phases"][src.phase]
            self.assertTrue(any(entry.startswith("[s3|P1]") for entry in history))
            turn = json.loads((root / "game_nl.json").read_text())[0]
            self.assertEqual(turn["treatment"], {"kind": "proficiency", "strategy": "low", "source_id": src.source_id})


class MissingEvidenceTests(unittest.TestCase):
    def test_zero_variance_differences_do_not_prove_equivalence(self):
        self.assertIsNone(tost_equivalence({"g1": [0.], "g2": [0.]}, .1, n_boot=30)["equivalent"])


    def test_model_requires_both_orders_for_each_comparison(self):
        row = {"bundle_id": "g:p", "game_id": "g", "pair": "low:neutral", "listener": "model",
               "listener_family": "fake", "treatment": "low", "comparator": "neutral", "order": "ab",
               "ok": True, "chosen_condition": "low", "chose_treatment": True}
        self.assertEqual(listener_analysis([row], n_boot=20, seed=1), {})
        for failed in (
                {**row, "order": "ba", "ok": False, "chosen_condition": None},
                {**row, "order": "ba", "chosen_condition": None}):
            self.assertEqual(listener_analysis([row, failed], n_boot=20, seed=1), {})


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
            with patch("experiments.proficiency_validation.validation.protocol.model_identity",
                       side_effect=lambda h: f["models"]["identities"][h]), \
                 patch("experiments.proficiency_validation.validation.protocol.working_tree_clean", return_value=True), \
                 patch("experiments.proficiency_validation.validation.protocol.git_commit", return_value="a" * 40):
                self.assertEqual(main_freeze(args), 0)
                with patch("experiments.proficiency_validation.validation.protocol.working_tree_clean", return_value=False):
                    with self.assertRaisesRegex(ValueError, "uncommitted"):
                        main_freeze(SimpleNamespace(**{**vars(args), "out": root / "frozen2.json"}))
            with self.assertRaises(SystemExit):              # not at the frozen commit: refused
                enforce_frozen(args.out)
            with patch("experiments.proficiency_validation.validation.protocol.git_commit", return_value="a" * 40):
                self.assertEqual(enforce_frozen(args.out)["schema_version"], 7)
                self.assertEqual(enforce_frozen(args.out)["validation_scope"], "automated")
                code = cli.main(["analyze", "--frozen", str(args.out), "--split", "set_a",
                                 "--json", str(root / "result.json"), "--out", str(root / "result.md")])
            self.assertEqual(code, 1)
            result = json.loads((root / "result.json").read_text())
            self.assertFalse(result["verdict"]["confirmed"])
            self.assertGreater(result["verdict"]["not_evaluated"], 0)
            self.assertTrue(any("A vs B" in c["criterion"] for c in result["criteria"]))


class CheckpointTests(unittest.TestCase):
    def test_shared_manifest_preserves_parallel_workers_and_model_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            models, settings = {"generator": {"model": "fixture"}}, {"workers": 12}
            configure_run(manifest, models, settings)
            paths = [root / "rewrite" / "shards" / f"worker_{i}.jsonl" for i in range(12)]
            with ThreadPoolExecutor(max_workers=12) as workers:
                list(workers.map(lambda p: update_metadata(p, "qc", {"name": p.name}), paths))
            configure_run(manifest, models, settings)
            for path in paths:
                self.assertEqual(read_metadata(path, "qc"), {"name": path.name})
            self.assertEqual(json.loads(manifest.read_text())["models"], models)
            self.assertEqual(list(root.rglob("*.json")), [manifest])
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                configure_run(manifest, models, {"workers": 6})
            self.assertEqual(json.loads(manifest.read_text())["configuration"], settings)

    def test_checkpoint_keeps_exclusive_lock_across_finish_and_allows_other_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "rewrite" / "rows.jsonl"
            first = Checkpoint(output, {"model": "fake"})
            try:
                for finished in (False, True):
                    if finished:
                        first.append({"id": 1})
                        first.finish()
                    with self.assertRaisesRegex(ValueError, "another job"):
                        Checkpoint(output, {"model": "fake"}, resume=True)
                second = Checkpoint(root / "validation" / "rows.jsonl", {"model": "fake"})
                second.finish()
                second.close()
            finally:
                first.close()
            resumed = Checkpoint(output, {"model": "fake"}, resume=True)
            self.assertEqual(resumed.rows, [{"id": 1}])
            resumed.close()
            self.assertEqual(sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()),
                             ["manifest.json", "rewrite/rows.jsonl", "validation/rows.jsonl"])

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

    def test_resume_allows_report_changes_and_rejects_model_output_code_changes(self):
        from experiments.proficiency_validation.rewrite.runtime import configure_run
        prefix = "experiments.proficiency_validation."
        def settings(**changes):
            hashes = {prefix + name: "1" for name in ("validation.analysis", "validation.cefr", "validation.perception",
                                                      "validation.calibration", "rewrite.fidelity", "rewrite.gate")}
            hashes.update({prefix + name: value for name, value in changes.items()})
            return {"model": "fake", "hashes": {"rubric": "r1", "module_source_sha256": hashes}}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rows.jsonl"
            first = Checkpoint(output, settings())
            first.append({"id": 1}); first.finish(); first.close()
            resumed = Checkpoint(output, settings(**{"validation.analysis": "2"}), resume=True)
            self.assertEqual(resumed.rows, [{"id": 1}]); resumed.close()
            for module in ("validation.cefr", "validation.perception", "validation.calibration", "rewrite.fidelity", "rewrite.gate", "unknown"):
                with self.subTest(module=module), self.assertRaisesRegex(ValueError, "settings differ"):
                    Checkpoint(output, settings(**{module: "2"}), resume=True)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                Checkpoint(output, {**settings(), "model": "changed"}, resume=True)
            manifest = Path(tmp) / "run" / "manifest.json"
            configure_run(manifest, {"judge": {"model": "m"}}, settings())
            configure_run(manifest, {"judge": {"model": "m"}}, settings(**{"validation.analysis": "2"}))
            with self.assertRaisesRegex(ValueError, "select a new RUN"):
                configure_run(manifest, {"judge": {"model": "m"}}, settings(**{"validation.cefr": "2"}))

    def test_resume_restarts_an_attempt_that_wrote_nothing_but_protects_written_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run" / "validation" / "judged.jsonl"
            empty = Checkpoint(output, {"inputs": "items-v1"}); empty.close()
            self.assertTrue(empty.partial.exists())
            restarted = Checkpoint(output, {"inputs": "items-v2"}, resume=True)   # nothing was lost
            self.assertEqual(restarted.settings["settings"], {"inputs": "items-v2"})
            restarted.append({"id": 1}); restarted.close()
            with self.assertRaisesRegex(ValueError, "settings differ"):
                Checkpoint(output, {"inputs": "items-v3"}, resume=True)
            resumed = Checkpoint(output, {"inputs": "items-v2"}, resume=True)
            self.assertEqual(resumed.rows, [{"id": 1}]); resumed.close()

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
        gate.neutral_frequency = _constant_frequency
        rows = generate([src], conditions=["low", "neutral", "high"], generator=model, gate=gate,
                        completed_rows=[previous], log=lambda *a: None)
        self.assertEqual(model.calls, 2)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], previous)
        self.assertEqual(rows[-1]["status"], "ok")
        self.assertNotIn("anchor_outcomes", rows[-1])


class ChallengeShardTests(unittest.TestCase):
    def test_judged_shards_merge_once_they_cover_every_item_under_one_setting(self):
        from experiments.proficiency_validation.validation.calibration import merge_challenge
        from experiments.proficiency_validation.rewrite.runtime import read_metadata, record_artifact
        from components.helpers import write_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            cal = Path(tmp) / "run" / "validation" / "calibration"
            items = [{"item_id": f"i{k}"} for k in range(5)]
            (cal / "shards").mkdir(parents=True)
            write_jsonl(cal / "items.jsonl", items)
            shards = []
            for index in range(2):
                path = cal / "shards" / f"judged_{index:02}.jsonl"
                settings = {"judge": {"model": "fake"}, "hashes": {"rubric": "r1", "module_source_sha256": {
                                "experiments.proficiency_validation.validation.analysis": str(index)}},
                            "shard": [index, 2]}
                ck = Checkpoint(path, settings)
                for item in items[index::2]:
                    ck.append({**item, "flags": {"pipeline": True}})
                ck.finish(); ck.close()
                record_artifact(path, {"gate": {"judge_handle": "judge"}, "n_items": len(items[index::2])})
                shards.append(path)
            rows = merge_challenge(cal / "items.jsonl", shards, cal / "judged.jsonl")
            self.assertEqual([r["item_id"] for r in rows], ["i0", "i2", "i4", "i1", "i3"])
            self.assertFalse(any(p.exists() for p in shards)); self.assertFalse((cal / "shards").exists())
            self.assertEqual(read_metadata(cal / "judged.jsonl")["n_items"], 5)
            self.assertEqual(len(read_metadata(cal / "judged.jsonl")["merged_shards"]), 2)
            self.assertFalse(read_metadata(shards[0], "checkpoint"))      # the shard records are gone
            with self.assertRaisesRegex(ValueError, "already exists"):
                merge_challenge(cal / "items.jsonl", shards, cal / "judged.jsonl")

    def test_shards_with_different_judges_or_missing_items_do_not_merge(self):
        from experiments.proficiency_validation.validation.calibration import merge_challenge
        from components.helpers import write_jsonl
        with tempfile.TemporaryDirectory() as tmp:
            cal = Path(tmp) / "run" / "validation" / "calibration"; cal.mkdir(parents=True)
            items = [{"item_id": f"i{k}"} for k in range(4)]
            write_jsonl(cal / "items.jsonl", items)
            def shard(name, judge, rows):
                path = cal / name
                ck = Checkpoint(path, {"judge": {"model": judge}, "shard": [0, 2]})
                for r in rows: ck.append(r)
                ck.finish(); ck.close(); return path
            a = shard("a.jsonl", "fake", items[0::2]); b = shard("b.jsonl", "other", items[1::2])
            with self.assertRaisesRegex(ValueError, "different settings"):
                merge_challenge(cal / "items.jsonl", [a, b], cal / "judged.jsonl")
            with self.assertRaisesRegex(ValueError, "complete item coverage"):
                merge_challenge(cal / "items.jsonl", [a], cal / "judged.jsonl")
            self.assertTrue(a.exists())        # a refused merge removes nothing


class ExportTests(unittest.TestCase):
    def test_export_keeps_each_cefr_rater_and_only_complete_listener_orders(self):
        from experiments.proficiency_validation.validation.export import _rows
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature = {"bundle_id": "b", "game_id": "g", "condition": "low", "complete": True}
            write_jsonl(root / "linguistic_features.jsonl", [feature])
            cefr = [{**feature, "rater": rater, "rater_family": rater, "level": band, "ok": True, "status": "rated"}
                    for rater, band in (("r1", "A"), ("r2", "C"))]
            write_jsonl(root / "cefr_auxiliary.jsonl", cefr)
            base = {**feature, "pair": "low:neutral", "treatment": "low", "comparator": "neutral",
                    "construct": "range", "ok": True}
            listeners = [{**base, "listener": "complete", "order": order, "signed": 1} for order in ("ab", "ba")]
            listeners += [{**base, "listener": "incomplete", "order": "ab", "signed": -1},
                          {**base, "listener": "incomplete", "order": "ba", "signed": -1, "ok": False}]

            write_jsonl(root / "range_listener.jsonl", listeners)
            row, = _rows(root, "test")
            self.assertEqual({r: v["band"] for r, v in row["cefr_auxiliary_ratings"].items()}, {"r1": "A", "r2": "C"})
            self.assertEqual(row["listener_signed_range_low_vs_neutral"], 1.)
            self.assertEqual(set(row["listener_ratings"]["low:neutral"]["range"]), {"complete"})
            listeners[-1]["ok"] = True
            write_jsonl(root / "range_listener.jsonl", listeners)
            row, = _rows(root, "test")
            self.assertEqual(row["listener_signed_range_low_vs_neutral"], 0.)
            cefr.append(dict(cefr[0]))
            write_jsonl(root / "cefr_auxiliary.jsonl", cefr)
            with self.assertRaises(ValueError):
                _rows(root, "test")
