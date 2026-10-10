"""The run manifest's frozen artifacts: one hash per strategy, and drift is reported."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from components.rewriters.politeness_rewriter import POLITENESS_STRATEGIES, PolitenessRewriter
from experiments.politeness_flip.code.provenance import (
    CANONICAL_CONTEXT, SOURCE_FILES, frozen_hashes, verify)

VALIDATION_PROVENANCE = Path("experiments/politeness_validation/data/provenance.json")


def manifest_with(hashes: dict) -> dict:
    return {"configuration": {"experiment": "politeness_flip", "hashes": hashes}}


class TestFrozenHashes(unittest.TestCase):
    def setUp(self):
        self.hashes = frozen_hashes()

    def test_one_template_hash_per_strategy(self):
        templates = self.hashes["rewriter_prompt_sha256"]
        self.assertEqual(sorted(templates), sorted(POLITENESS_STRATEGIES))
        self.assertEqual(len(set(templates.values())), len(POLITENESS_STRATEGIES))

    def test_template_hash_is_the_same_for_every_predicate(self):
        """The definition slot is a placeholder, so the hash a rewrite row carries
        does not depend on which predicate was rewritten."""
        rewriter = PolitenessRewriter(llm_client=None, strategy="negative_politeness")
        other = CANONICAL_CONTEXT.__class__(
            predicate="recommend_action(D2, P4, vote, P1, werewolf)",
            original_nl="Vote P1 today.", phase="day_speech", speaker_id="P4",
            speaker_role_claimed="villager", public_history=[])
        self.assertNotEqual(rewriter.prompt_sha256(CANONICAL_CONTEXT), rewriter.prompt_sha256(other))
        self.assertEqual(rewriter.template_sha256(),
                         self.hashes["rewriter_prompt_sha256"]["negative_politeness"])

    def test_rewrite_rows_carry_the_template_hash(self):
        with TemporaryDirectory() as tmp:
            rewriter = PolitenessRewriter(llm_client=None, strategy="off_record")
            rewriter.log_path = Path(tmp) / "rewrites.jsonl"
            rewriter._record(CANONICAL_CONTEXT, "Funny how quiet P7 went after that vote.")
            row = json.loads(rewriter.log_path.read_text(encoding="utf-8"))
        self.assertEqual(row["prompt_sha256"], self.hashes["rewriter_prompt_sha256"]["off_record"])

    def test_canonical_prompts_are_the_validated_ones(self):
        """The flip run's prompts are the prompts the politeness validation froze."""
        recorded = json.loads(VALIDATION_PROVENANCE.read_text(encoding="utf-8"))["frozen_artifacts"]
        self.assertEqual(self.hashes["rewriter_canonical_prompt_sha256"],
                         recorded["generation_prompt_sha256"])
        self.assertEqual(self.hashes["politeness_classifier_prompt_sha256"],
                         recorded["politeness_classifier_prompt_sha256"])

    def test_sources_cover_the_rewriter_the_instruments_and_the_engine(self):
        self.assertEqual(sorted(self.hashes["source_sha256"]), sorted(SOURCE_FILES))

    def test_verify_reports_only_what_changed(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest_with(self.hashes)), encoding="utf-8")
            self.assertEqual(verify(path), [])

            tampered = json.loads(json.dumps(self.hashes))
            tampered["rewriter_prompt_sha256"]["off_record"] = "0" * 64
            tampered["semantic_rubric_version"] = "v0"
            path.write_text(json.dumps(manifest_with(tampered)), encoding="utf-8")
            self.assertEqual(sorted(verify(path)),
                             ["rewriter_prompt_sha256.off_record", "semantic_rubric_version"])


if __name__ == "__main__":
    unittest.main()
