import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from components.helpers import build_context, read_jsonl, write_jsonl
from components.judges.inverse_predicate_parser import ParseResult
from components.judges.judge_model import Completion
from components.rewriters.language_rewriter import LanguageProficiencyRewriter
import experiments.proficiency_validation.validation.cefr as cefr
from experiments.proficiency_validation.validation.sources import build_bundles
from experiments.proficiency_validation.rewrite.fidelity import EnglishFidelityValidator, response_check
from experiments.proficiency_validation.rewrite.runtime import record_artifact, digest, read_metadata, update_metadata
from experiments.proficiency_validation.validation.protocol import validate_design
from experiments.proficiency_validation.rewrite.__main__ import check_generation_file, audit_generation, merge
from experiments.proficiency_validation.rewrite.gate import SemanticGate, generate_with_gate, failure_cause
from experiments.proficiency_validation.tests.test_fidelity_policy import judgment
from experiments.proficiency_validation.tests.test_offline import source, FakeClient
from experiments.proficiency_validation.tests.test_protocol import design


class ScriptedModel:
    family = "fixture"
    identity_record = {"handle": "rater", "family": "fixture", "revision": "fixture-only"}

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        if self.responses:
            result = self.responses.pop(0)
        else:
            text = json.loads(kwargs["user"])["ordered_turns"][0]
            result = {"level": "B", "insufficient_evidence": False, "reason": "Fixture only",
                "evidence": [{"dimension": d, "quote": text, "explanation": "Fixture evidence"}
                             for d in ("range", "control", "coherence")]}
        return result, Completion("", "fixture", "fixture")

    def preflight(self):
        return True, "fixture"

    def provenance(self):
        return {"identity": self.identity_record}


class EvidenceRepairTests(unittest.TestCase):
    def test_missing_connector_is_repaired_on_the_unchanged_text(self):
        src = source(text="P7 is a wolf and P2 is a villager.")
        candidate = "P7 is a wolf. P2 is a villager."
        incomplete = judgment(src.original_nl, candidate)
        incomplete["alignments"] = [
            {"original": "P7 is a wolf", "rewrite": "P7 is a wolf", "equivalent": True},
            {"original": "P2 is a villager", "rewrite": "P2 is a villager", "equivalent": True}]
        model = ScriptedModel([incomplete, judgment(src.original_nl, candidate)])
        result = EnglishFidelityValidator(model).validate(src, candidate)
        self.assertTrue(result.ok)
        self.assertEqual(len(model.calls), 2)
        repair = json.loads(model.calls[1]["user"].split("\n\n")[-1])["repair_evaluation"]
        self.assertEqual([r["text"] for r in repair["uncovered"]["original"]], ["and"])
        self.assertEqual(len(result.details["evaluation_attempts"]), 2)

    def test_explicit_negative_evidence_overrides_true_flags_without_repair(self):
        src = source(text="P2 offered an unverifiable clear of P7.")
        candidate = "P2 gave a clear of P7."
        result = judgment(src.original_nl, candidate)
        result["violations"] = [{"dimension": "content_preserved", "original": "unverifiable", "rewrite": "",
                                 "reason": "The stated limitation was removed."}]
        model = ScriptedModel([result])
        self.assertFalse(EnglishFidelityValidator(model).validate(src, candidate).ok)
        self.assertEqual(len(model.calls), 1)

    def test_explanatory_contradiction_and_malformed_evidence_do_not_pass(self):
        src = source()
        response = judgment(src.original_nl, "candidate")
        response["notes"] = "content_preserved should be false"
        model = ScriptedModel([response, response])
        self.assertIsNone(EnglishFidelityValidator(model).validate(src, "candidate").ok)
        self.assertIn("contradictory_verdict_explanation", model.calls[1]["user"])
        response["alignments"] = None
        missing, _ = response_check(response, src, "candidate")
        self.assertIn("missing_proposition_alignments", missing)

    def test_invalid_evaluation_does_not_trigger_identical_regeneration(self):
        src = source()
        candidate = "P7 is a werewolf. He pushed the vote onto P2 without giving a read of his own."
        model = ScriptedModel([{}, {}])
        client = FakeClient([candidate])
        gate = SemanticGate(validator=EnglishFidelityValidator(model), require_parser=False)
        rw = LanguageProficiencyRewriter(llm_client=client, condition="low")
        result = generate_with_gate(rw, gate, build_context(src), src)
        self.assertIsNone(result.text)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(failure_cause(result.to_dict()), "evaluation_output_failure")

    def test_neutral_matching_is_an_acceptance_requirement(self):
        src = source()
        candidate = src.original_nl.replace("because", "since")
        client = FakeClient([json.dumps([candidate])])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="neutral")
        gate = SemanticGate(require_parser=False, require_validator=False)
        gate.neutral_edit_tolerance = .05
        result = generate_with_gate(rw, gate, build_context(src), src, target_edit_distance=.5,
                                    max_corrective_retries=0)
        self.assertEqual(failure_cause(result.to_dict()), "control_matching_failure")
        self.assertIn("0.5000", client.calls[0]["messages"][1]["content"])
        self.assertIsNone(result.text)


class InventoryTests(unittest.TestCase):
    def test_ambiguous_single_label_uses_all_acts_and_keeps_full_semantics(self):
        src = source("disagree(D1, P3, s1)", "P1 is wrong and P2 is a wolf.")
        candidate = "P1 is wrong. P2 is a wolf."
        def inventory(text, target="P2"):
            return {"complete": True, "acts": [
                {"predicate": "disagree(D1, P3, s1)", "evidence": "P1 is wrong"},
                {"predicate": f"estimate(D1, P3, {target}, werewolf)", "evidence": f"{target} is a wolf"}]}
        model = ScriptedModel([inventory(src.original_nl), inventory(candidate), inventory("", "P7")])
        class Parser:
            max_tokens = 3072
            def parse(self, text, **kwargs):
                if text == src.original_nl:
                    return ParseResult(ok=True, recoverable=True, predicate_name="disagree",
                        arguments={"Phase": "D1", "Player": "P3", "RefID": "s1"})
                return ParseResult(ok=True, recoverable=True, predicate_name="estimate",
                    arguments={"Phase": "D1", "Player": "P3", "Target": "P2", "Role": "werewolf"})
        parser = Parser()
        parser.model = model
        gate = SemanticGate(parser=parser, validator=EnglishFidelityValidator(
            ScriptedModel([judgment(src.original_nl, candidate)])))
        verdict = gate.evaluate(src, candidate, "low")
        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.parser["comparison_basis"], "act_inventory")
        for call in model.calls:
            self.assertEqual(set(json.loads(call["user"])), {"utterance", "phase", "speaker", "prior_public_history"})
        changed = gate._run_parser(src, "P1 is wrong. P7 is a wolf.")
        self.assertFalse(changed["structural_match"])
        self.assertEqual(len(model.calls), 3)  # Original inventory reused.


class CEFRTests(unittest.TestCase):
    def test_invalid_quote_diagnostics_keep_the_rating_missing(self):
        text = "I suspect P7 because he changed his vote."
        response = {"level": "B", "insufficient_evidence": False, "reason": "Connected explanation.",
                    "evidence": [{"dimension": d, "quote": text, "explanation": "Observed language."}
                                 for d in ("range", "control", "coherence")]}
        self.assertEqual(cefr.rating_errors(response, text), [])
        response["evidence"][0]["quote"] = "I suspect P7... he changed his vote."
        result = cefr.CEFRRater(ScriptedModel([response])).rate([text])
        self.assertFalse(result["ok"])
        self.assertIsNone(result["level"])
        self.assertEqual(result["error_codes"], ["ungrounded_quote:0:range"])

    def test_rater_accepts_only_broad_bands(self):
        text = "I think P2 is a wolf."
        for level in ("A", "B", "C", "A1", "A2", "B1", "B2", "C1", "C2"):
            with self.subTest(level=level):
                response = {"level": level, "insufficient_evidence": False, "reason": "Fixture",
                    "evidence": [{"dimension": d, "quote": text, "explanation": "Fixture"}
                                 for d in ("range", "control", "coherence")]}
                result = cefr.CEFRRater(ScriptedModel([response])).rate([text])
                self.assertEqual(result["ok"], level in {"A", "B", "C"})
                self.assertEqual(result["level"], level if result["ok"] else None)

    def test_external_labels_map_to_bands_without_losing_the_reference_label(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "references.jsonl"
            labels = ["A1", "A2", "B1", "B2", "C1", "C2", "A", "B", "C"]
            rows = [{"item_id": str(i), "text": f"Distinct fixture text {i}.", "level": label,
                     "source": "fixture", "task": "learner_production", "split": "calibration"}
                    for i, label in enumerate(labels)]
            write_jsonl(path, rows)
            mapped = cefr.calibration_items(path)
            self.assertEqual([r["level"] for r in mapped], [s[0] for s in labels])
            self.assertEqual([r["reference_level"] for r in mapped], labels)
            self.assertEqual(read_jsonl(path), rows)

    def test_six_level_rating_outputs_cannot_be_analyzed_as_bands(self):
        with self.assertRaisesRegex(ValueError, "A/B/C"):
            cefr.ordinal_analysis([{"bundle_id": "g:P1", "condition": "low", "rater": "r",
                                    "ok": True, "level": "B1"}])

    def test_prompt_is_sample_blind_and_requires_grounded_dimensions(self):
        model = ScriptedModel()
        rater = cefr.CEFRRater(model)
        self.assertTrue(rater.rate(["I think P2 is a wolf.", "His vote changed."])["ok"])
        payload = json.loads(model.calls[0]["user"])
        self.assertEqual(set(payload), {"ordered_turns"})
        self.assertEqual(len(payload["ordered_turns"]), 2)
        bad = {"level": "C", "insufficient_evidence": False, "reason": "Fixture",
               "evidence": [{"dimension": "range", "quote": "fabricated words", "explanation": "Fixture"}]}
        self.assertFalse(cefr.CEFRRater(ScriptedModel([bad])).rate(["Actual words"])["ok"])
        insufficient = {"level": None, "insufficient_evidence": True, "reason": "Too little evidence"}
        result = cefr.CEFRRater(ScriptedModel([insufficient])).rate(["Yes."])
        self.assertFalse(result["ok"])
        self.assertIsNone(result["level"])

    def test_direction_uses_equal_game_weight_and_retains_missing_pairs(self):
        rows = []
        for game, speaker, low in (("g1", "P1", "A"), ("g1", "P2", "A"),
                                   ("g2", "P1", "C"), ("g3", "P1", None)):
            for c, level in (("original", "B"), ("neutral", "B"), ("low", low)):
                rows.append({"bundle_id": f"{game}:{speaker}", "game_id": game, "condition": c,
                    "rater": "fixture", "level": level, "ok": level is not None,
                    "status": "rated" if level else "incomplete_passage"})
        report = cefr.ordinal_analysis(rows, n_boot=30)
        block = report["raters"]["fixture"]["contrasts"]["low:neutral"]
        self.assertEqual((block["eligible"], block["paired"], block["unpaired"]), (4, 3, 1))
        self.assertEqual(block["game_weighted_shares"]["lower"]["estimate"], .5)
        self.assertEqual(block["transitions"]["B -> A"], 2)
        self.assertEqual(report["scale"], ["A", "B", "C"])
        self.assertEqual(report["level_distributions"]["fixture"]["low"], {"A": 2, "C": 1})
        rendered = "\n".join(cefr.render_cefr(report))
        self.assertIn("| A | B | C |", rendered)
        self.assertIn("same-band", rendered)
        self.assertNotIn("| A1 |", rendered)
        self.assertFalse(report["used_for_selection"])

    def test_calibration_missing_responses_count_against_accuracy(self):
        rows = [{"rater": "r", "expected_level": "A", "ok": True, "level": "A"},
                {"rater": "r", "expected_level": "B", "ok": False, "level": None}]
        report = cefr.calibration_report(rows)["r"]
        self.assertEqual(report["accuracy_all_items"], .5)
        self.assertEqual(report["valid_fraction"], .5)
        self.assertEqual(set(report["confusion"]), {"A", "B", "C"})
        self.assertNotIn("within_one_valid", report)

    def test_auxiliary_direction_cannot_change_primary_criteria(self):
        from experiments.proficiency_validation.validation.analysis import criteria
        f = design()
        first = criteria({}, f)
        self.assertEqual(first, criteria({"cefr": {"low": "C", "neutral": "A"}}, f))

    def test_enabled_cefr_requires_calibration_and_independent_model(self):
        f = design()
        f["cefr"] = {"enabled": True, "raters": ["generator"], "max_tokens": 1536}
        errors = validate_design(f)
        self.assertTrue(any("calibration" in e for e in errors))
        self.assertTrue(any("CEFR" in e and "generator" in e for e in errors))

    def test_frozen_calibration_coverage_counts_three_bands(self):
        spec = {"raters": ["r"], "min_calibration_accuracy": .8,
                "min_calibration_valid_fraction": .9, "min_calibration_per_level": 2}
        report = {"r": {"accuracy_all_items": 1., "valid_fraction": 1.,
                        "items_per_level": {"A": 2, "B": 2, "C": 2}}}
        self.assertTrue(cefr.calibration_meets(report, spec))
        report["r"]["items_per_level"].pop("C")
        self.assertFalse(cefr.calibration_meets(report, spec))

    def test_prompt_only_pilot_runs_and_analyzes_without_calibration(self):
        from experiments.proficiency_validation.validation.__main__ import main
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = source()
            bundles, rewrites, output = [root / n for n in ("bundles.jsonl", "rewrites.jsonl", "cefr.jsonl")]
            write_jsonl(bundles, build_bundles([src], window_min=1, selection="whole_game")["bundles"])
            write_jsonl(rewrites, [{"source_id": src.source_id, "condition": c, "original_text": src.original_nl,
                "rewritten_text": src.original_nl, "status": "ok", "edit_distance_to_original": 0.}
                for c in ("low", "neutral")])
            command = ["cefr-rate", "--bundles", str(bundles), "--rewrites", str(rewrites),
                       "--raters", "rater", "--out", str(output)]
            model = ScriptedModel()
            with patch("experiments.proficiency_validation.rewrite.runtime.ValidationModel.resolve", return_value=model):
                self.assertEqual(main(command), 0)
                self.assertEqual(len(model.calls), 3)
                self.assertEqual(main(command + ["--resume"]), 0)
                self.assertEqual(len(model.calls), 3)
            prov, errors = cefr.verify_ratings(output, rewrites_path=rewrites)
            self.assertEqual(errors, [])
            self.assertEqual(prov["calibration_status"], "not_measured")
            self.assertIsNone(prov["calibration"])
            self.assertNotIn("calibration", prov["input_sha256"])
            report = root / "report.md"
            self.assertEqual(main(["analyze", "--no-frozen", "--rewrites", str(rewrites),
                "--cefr", str(output), "--out", str(report), "--n-boot", "30"]), 0)
            text = report.read_text()
            self.assertIn("External calibration is unmeasured", text)
            self.assertIn("| A | B | C |", text)
            # A legacy scale or an invented calibration claim cannot masquerade as this run.
            for overrides in ({"prompt_version": "cefr_text_v1"}, {"calibration_status": "measured"}):
                record_artifact(output, {**prov, **overrides})
                self.assertTrue(cefr.verify_ratings(output, rewrites_path=rewrites)[1])

    def test_frozen_scoring_still_requires_declared_calibration(self):
        from experiments.proficiency_validation.validation.__main__ import main
        frozen = {"cefr": {"enabled": True, "raters": ["rater"], "max_tokens": 1536}}
        with patch("experiments.proficiency_validation.validation.protocol.enforce_frozen", return_value=frozen):
            with self.assertRaisesRegex(ValueError, "declared calibration"):
                main(["cefr-rate", "--bundles", "unused", "--rewrites", "unused", "--raters", "rater",
                      "--frozen", "unused", "--out", "unused"])

    def test_prompt_examples_cannot_overlap_an_uncalibrated_pilot(self):
        from experiments.proficiency_validation.validation.__main__ import main
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = source()
            write_jsonl(root / "bundles.jsonl", build_bundles([src], window_min=1, selection="whole_game")["bundles"])
            write_jsonl(root / "rewrites.jsonl", [{"source_id": src.source_id, "condition": "low",
                "original_text": src.original_nl, "rewritten_text": src.original_nl, "status": "ok"}])
            write_jsonl(root / "examples.jsonl", [{"item_id": "i", "text": src.original_nl, "level": "B2",
                "task": "learner_production", "source": "fixture", "split": "example"}])
            with self.assertRaisesRegex(ValueError, "overlap"):
                main(["cefr-rate", "--bundles", str(root / "bundles.jsonl"), "--rewrites", str(root / "rewrites.jsonl"),
                      "--examples", str(root / "examples.jsonl"), "--raters", "rater", "--out", str(root / "unused")])

    def test_calibration_and_rating_cli_preserve_cohort_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            refs, calibration, bundles, rewrites, output = [root / n for n in
                ("refs.jsonl", "calibration.jsonl", "bundles.jsonl", "rewrites.jsonl", "cefr.jsonl")]
            write_jsonl(refs, [{"item_id": "external1", "text": "My class starts at nine.", "level": "A2",
                "source": "fixture dataset only", "task": "learner_production", "split": "calibration"}])
            src = source()
            write_jsonl(bundles, build_bundles([src], window_min=1, selection="whole_game")["bundles"])
            write_jsonl(rewrites, [{"source_id": src.source_id, "condition": c, "original_text": src.original_nl,
                "rewritten_text": src.original_nl if c == "neutral" else None,
                "status": "ok" if c == "neutral" else "treatment_unavailable"} for c in ("low", "neutral")])
            common = dict(raters="rater", examples=None, max_tokens=1536, resume=False)
            model = ScriptedModel()
            with patch("experiments.proficiency_validation.rewrite.runtime.ValidationModel.resolve", return_value=model):
                cefr.main(SimpleNamespace(**common, cefr_command="cefr-calibrate", items=refs, out=calibration))
                args = SimpleNamespace(**common, cefr_command="cefr-rate", bundles=bundles, rewrites=rewrites,
                    calibration=calibration, frozen=None, split="pilot", out=output)
                cefr.main(args)
                calls = len(model.calls)
                args.resume = True
                cefr.main(args)
                self.assertEqual(len(model.calls), calls)
            rows = read_jsonl(output)
            self.assertEqual(read_jsonl(calibration)[0]["expected_level"], "A")
            self.assertEqual(read_jsonl(calibration)[0]["reference_level"], "A2")
            self.assertEqual(len(rows), 3)
            self.assertEqual(next(r for r in rows if r["condition"] == "low")["status"], "incomplete_passage")
            self.assertEqual(cefr.verify_ratings(output, rewrites_path=rewrites)[1], [])
            from experiments.proficiency_validation.rewrite.runtime import read_metadata
            summary = read_metadata(output, "summary")
            self.assertEqual(summary["calibration_status"], "measured")
            self.assertIn("Exact band agreement", "\n".join(cefr.render_cefr(summary)))
            rows.pop()
            write_jsonl(output, rows)
            self.assertTrue(cefr.verify_ratings(output, rewrites_path=rewrites)[1])

    def test_examples_and_calibration_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            item = {"item_id": "i", "text": "I live here.", "level": "A1", "source": "fixture", "task": "learner_production"}
            write_jsonl(root / "items.jsonl", [{**item, "split": "calibration"}])
            write_jsonl(root / "examples.jsonl", [{**item, "split": "example"}])
            with self.assertRaisesRegex(ValueError, "overlap"):
                cefr.main(SimpleNamespace(raters="rater", examples=root / "examples.jsonl", max_tokens=1536,
                    cefr_command="cefr-calibrate", items=root / "items.jsonl"))


class GenerationRecordTests(unittest.TestCase):
    def shard(self, path, *, version="fixture_v1", recorded_version=None):
        settings = {"artifact": {}, "conditions": ["low"], "source_ids": [path.stem],
                    "hashes": {"rewriter_prompt_version": recorded_version or version,
                        "rewriter_prompt_sha256": {"low": "prompt"}, "semantic_rubric_version": "rubric",
                        "semantic_rubric_sha256": "rubricsha"}}
        row = {"source_id": path.stem, "condition": "low", "original_text": "text", "prompt_version": version,
            "prompt_template_sha256": "prompt", "gate_provenance": {
                "semantic_rubric_version": "rubric", "semantic_rubric_sha256": "rubricsha"}}
        write_jsonl(path, [row])
        update_metadata(path, "checkpoint", {"settings": settings, "settings_sha256": digest(settings)})
        record_artifact(path, {})
        return row

    def test_hash_correct_output_cannot_hide_stale_settings(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "shard.jsonl"
            self.shard(path, recorded_version="old")
            self.assertTrue(any("disagree" in e for e in check_generation_file(path)))

    def test_merge_rejects_mixed_generation_policies(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shards = [root / "a.jsonl", root / "b.jsonl"]
            self.shard(shards[0])
            self.shard(shards[1], version="fixture_v2")
            with self.assertRaisesRegex(ValueError, "different policies"):
                merge(SimpleNamespace(shards=shards, sources=root / "unused", out=root / "merged.jsonl"))
            self.assertFalse((root / "merged.jsonl").exists())

    def test_compaction_preserves_bytes_and_detects_tampered_shard_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [root / "rewrite" / "shards" / f"{key}.jsonl" for key in ("a", "b")]
            rows = [self.shard(path) for path in paths]
            original = b"".join(path.read_bytes() for path in paths)
            sources = root / "sources.jsonl"
            write_jsonl(sources, [{"source_id": row["source_id"], "original_nl": row["original_text"]} for row in rows])
            output = root / "rewrite" / "rewrites.jsonl"
            self.assertEqual(merge(SimpleNamespace(shards=paths, sources=sources, out=output)), 0)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(audit_generation(output), [])
            self.assertTrue(all(not path.exists() for path in paths))
            metadata = read_metadata(output)
            metadata["compacted_shards"][0]["checkpoint"]["settings"]["source_ids"] = ["changed"]
            update_metadata(output, "provenance", metadata)
            self.assertTrue(any("settings digest mismatch" in error for error in audit_generation(output)))
