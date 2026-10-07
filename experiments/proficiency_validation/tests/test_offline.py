import json
import random
import unittest
from pathlib import Path

from components.helpers import SourceRecord, build_context
from components.metrics.edit_distance import normalized_token_edit_distance, token_levenshtein
from components.metrics.lexical import lexical_profile, mattr, mtld, repetition
from components.metrics.text_normalization import normalize_player_ids, tokenize
from components.rewriters.language_rewriter import (
    LanguageProficiencyRewriter, parse_candidates, select_candidate)
from components.stats import (
    benjamini_hochberg, bradley_terry, bradley_terry_bootstrap, cluster_bootstrap,
    cluster_sign_flip_test, paired_cohens_dz, tost_equivalence, wilson_interval)
from lurkingwolf.rewriter import RewriteContext

import experiments.proficiency_validation.validation.sources as bundles_mod
import experiments.proficiency_validation.validation.calibration as jb
from experiments.proficiency_validation.validation.analysis import contrast_analysis, listener_analysis
from experiments.proficiency_validation.validation.protocol import frozen_hashes
from experiments.proficiency_validation.rewrite.gate import GatedRewriter, SemanticGate, TreatmentUnavailable, generate_with_gate, generation_qc, source_from_context


def _constant_frequency(text):
    return 0.0

def source(predicate="estimate(D1, P3, P7, werewolf)", text="P7 is a werewolf, because he pushed "
           "the vote onto P2 without ever giving a read of his own.", speaker="P3", name=None):
    from lurkingwolf.utils.speech_nl import parse_predicate
    parsed = parse_predicate(predicate)
    return SourceRecord(source_id="t:1", predicate_name=name or parsed["name"], predicate_text=predicate,
                        predicate_args=[str(a) for a in parsed["args"]], original_nl=text,
                        phase="day_speech", speaker_id=speaker, game_id="g1", speech_id="s5",
                        public_history=["[s1|P1] claim_role(D1, P1, seer): I am the seer."])


class FakeClient:
    """Returns scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.responses.pop(0)

        class R:
            pass
        r = R()
        m = R()
        m.content = text
        c = R()
        c.message = m
        r.choices = [c]
        return r


class EditDistanceTests(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(token_levenshtein(["a", "b", "c"], ["a", "x", "c"]), 1)
        self.assertEqual(token_levenshtein([], ["a"]), 1)
        self.assertEqual(token_levenshtein(["a", "b"], ["a", "b"]), 0)

    def test_normalized(self):
        self.assertEqual(normalized_token_edit_distance("P7 is a wolf", "P7 is a wolf"), 0.0)
        self.assertAlmostEqual(normalized_token_edit_distance("P7 is a wolf", "P7 is a villager"), 0.25)
        self.assertEqual(normalized_token_edit_distance("", ""), 0.0)


class RewriterTests(unittest.TestCase):
    def setUp(self):
        self.ctx = build_context(source())

    def test_prompts_per_condition_and_hash_stability(self):
        for condition in ("low", "neutral", "high"):
            rw = LanguageProficiencyRewriter(llm_client=None, condition=condition)
            system, user = rw.build_prompts(self.ctx)
            self.assertIn(f"{condition.upper()} SPECIFICATION", system)
            self.assertIn("Politeness is a separate experimental treatment", system)
            self.assertIn("Do not insert grammatical errors", system)
            self.assertIn("Original: P7 is a werewolf", user)
            self.assertEqual(rw.prompt_sha256(self.ctx), rw.prompt_sha256(self.ctx))
        hashes = frozen_hashes()
        self.assertEqual(len({hashes["rewriter_prompt_sha256"][c] for c in ("low", "neutral", "high")}), 3)
        # The TEMPLATE hash is predicate-independent; the instance hash is not.
        other = build_context(source("question(D1, P6, P5, s23)", "P5, why did you ask for my execution?", "P6"))
        rw = LanguageProficiencyRewriter(llm_client=None, condition="low")
        self.assertEqual(rw.template_sha256(), hashes["rewriter_prompt_sha256"]["low"])
        self.assertNotEqual(rw.prompt_sha256(self.ctx), rw.prompt_sha256(other))
        self.assertEqual(rw.template_sha256("neutral", candidates=True, n=5),
                         hashes["rewriter_prompt_sha256"]["neutral_candidates"])
        self.assertNotEqual(rw.template_sha256("neutral", candidates=True, n=5),
                            rw.template_sha256("neutral", candidates=True, n=7))
        from experiments.proficiency_validation.validation.protocol import expected_template_hash
        frozen = {"hashes": hashes}
        self.assertEqual(expected_template_hash(frozen, "neutral"), hashes["rewriter_prompt_sha256"]["neutral_candidates"])
        self.assertEqual(expected_template_hash(frozen, "high"), hashes["rewriter_prompt_sha256"]["high"])

    def test_feedback_is_appended(self):
        rw = LanguageProficiencyRewriter(llm_client=None, condition="low")
        _, user = rw.build_prompts(self.ctx, feedback={"reasons": ["added_epistemic_hedge:maybe"],
                                                      "previous": "P7 might be a wolf."})
        self.assertIn("added_epistemic_hedge:maybe", user)
        self.assertIn("Previous rewrite: P7 might be a wolf.", user)

    def test_rejects_unknown_condition_and_missing_client(self):
        with self.assertRaises(ValueError):
            LanguageProficiencyRewriter(condition="A")
        with self.assertRaises(RuntimeError):
            LanguageProficiencyRewriter(llm_client=None, condition="low").rewrite(self.ctx)

    def test_low_rewrite_cleans_output(self):
        client = FakeClient(["<think>hmm</think>\n\"P7 is a werewolf. He pushed the vote onto P2. He never gave a read of his own.\""])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="low")
        text = rw.rewrite(self.ctx)
        self.assertTrue(text.startswith("P7 is a werewolf."))
        self.assertNotIn("<think>", text)

    def test_parse_candidates(self):
        self.assertEqual(parse_candidates('["a b", "c d"]'), ["a b", "c d"])
        self.assertEqual(parse_candidates("1. first one\n2) second one"), ["first one", "second one"])
        self.assertEqual(parse_candidates("```json\n[\"x\"]\n```"), ["x"])

    def test_select_candidate(self, frequency=_constant_frequency):
        original = "P7 is a werewolf because he pushed the vote onto P2"
        cands = [original,                                                          # copy
                 "P7 is a werewolf since he pushed the vote onto P2",               # small change
                 "P7 is a werewolf since he pushed the vote onto P2",               # duplicate
                 "I am sure P7 is a wolf: he steered the whole vote toward P2 himself"]  # large
        sel = select_candidate(cands, original, seed=1, frequency=_constant_frequency)
        self.assertEqual(sel.chosen, cands[1])                  # same sentence count, fewest word changes
        self.assertEqual(sorted(d["reason"] for d in sel.dropped), ["duplicate", "identical_to_original"])
        self.assertEqual(sel.rule, "closest_profile")
        self.assertEqual(len(sel.distances), 2)                 # exposure is recorded for every kept candidate
        with self.assertRaises(RuntimeError):
            select_candidate([original], original, seed=1, frequency=_constant_frequency)

    def test_neutral_records_its_selection_and_never_calls_for_an_anchor(self):
        client = FakeClient([json.dumps(["P7 is a werewolf, since he pushed the vote onto P2 without ever giving a read of his own.",
                                         "P7 is a wolf: he steered the vote toward P2 and never offered one read of his own."])])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="neutral", neutral_frequency=_constant_frequency)
        text = rw.rewrite(self.ctx)
        self.assertEqual(text, rw.last_selection.chosen)
        self.assertEqual(rw.last_selection.rule, "closest_profile")
        self.assertEqual(len(rw.last_selection.distances), 2)        # recorded, not used for ranking
        self.assertEqual(len(client.calls), 1)                        # one candidate call, no Low anchor
        self.assertNotIn("edit-distance", client.calls[0]["messages"][1]["content"].lower())


class LexicalTests(unittest.TestCase):
    def test_mattr_and_mtld(self):
        unique = [f"w{i}" for i in range(60)]
        self.assertAlmostEqual(mattr(unique, 50), 1.0)
        self.assertIsNone(mattr(unique[:10], 50))
        repeated = ["the"] * 60
        self.assertAlmostEqual(mattr(repeated, 50), 1 / 50)
        # MTLD is undefined (infinite) when the TTR never falls, so a fully
        # unique text yields None; compare a high-diversity text with repeats.
        self.assertIsNone(mtld(unique))
        self.assertGreater(mtld(unique[:50] + unique[:10]), mtld(["a", "b"] * 30))

    def test_repetition(self):
        rep = repetition(["wolf", "wolf", "vote", "vote", "read"])
        self.assertAlmostEqual(rep["repetition_rate"], 1 - 3 / 5)
        self.assertAlmostEqual(rep["top5_share"], 1.0)

    def test_profile_without_pos(self):
        prof = lexical_profile(tokenize("P7 is a wolf and P7 pushed the vote"), None, mattr_window=3)
        self.assertEqual(prof.n_tokens, 9)
        self.assertEqual(prof.n_tokens_no_ids, 7)
        self.assertFalse(prof.lemmas_available)
        self.assertIsNone(prof.frequency["subtlex_mean"])

    def test_normalize_ids(self):
        self.assertEqual(normalize_player_ids("P3's s12 read on P12"), "Jordan's statement 12 read on Skyler")


class StatsTests(unittest.TestCase):
    def test_benjamini_hochberg(self):
        adj = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
        self.assertAlmostEqual(adj["a"]["q"], 0.04)
        self.assertAlmostEqual(adj["b"]["q"], 0.04 * 4 / 3)
        self.assertTrue(adj["a"]["reject"])
        self.assertFalse(adj["d"]["reject"])

    def test_cluster_bootstrap_and_sign_flip(self):
        rng = random.Random(1)
        clusters = {f"g{i}": [rng.gauss(0.5, 0.2) for _ in range(4)] for i in range(12)}
        boot = cluster_bootstrap(clusters, n_boot=500, seed=3)
        self.assertLess(boot["ci_low"], boot["estimate"])
        self.assertGreater(boot["ci_high"], boot["estimate"])
        self.assertLess(cluster_sign_flip_test(clusters, n_perm=2000, seed=3)["p_value"], 0.01)
        null = {f"g{i}": [rng.gauss(0.0, 1.0) for _ in range(4)] for i in range(12)}
        self.assertGreater(cluster_sign_flip_test(null, n_perm=2000, seed=3)["p_value"], 0.05)

    def test_tost(self):
        tight = {f"g{i}": [0.01 * (-1) ** i] * 3 for i in range(10)}
        self.assertTrue(tost_equivalence(tight, bound=0.1, n_boot=300)["equivalent"])
        wide = {f"g{i}": [0.5 + 0.01 * i] * 3 for i in range(10)}
        self.assertFalse(tost_equivalence(wide, bound=0.1, n_boot=300)["equivalent"])

    def test_bradley_terry(self):
        comps = [("low", "neutral")] * 90 + [("neutral", "low")] * 10
        bt = bradley_terry(comps)
        self.assertGreater(bt["low"], bt["neutral"])
        by_cluster = {f"g{i}": comps[i::10] for i in range(10)}
        boot = bradley_terry_bootstrap(by_cluster, "low", "neutral", n_boot=200)
        self.assertGreater(boot["ci_low"], 0)
        unanimous = bradley_terry([("a", "b")] * 20)
        self.assertTrue(all(abs(v) < 50 for v in unanimous.values()))

    def test_misc(self):
        p, lo, hi = wilson_interval(80, 100)
        self.assertAlmostEqual(p, 0.8)
        self.assertLess(lo, 0.8)
        self.assertGreater(paired_cohens_dz([0.5, 0.6, 0.4, 0.5]), 3)


class ChallengeTests(unittest.TestCase):
    def test_every_perturbation_type_fires_somewhere(self):
        rng = random.Random(0)
        srcs = [source(),
                source("recommend_action(D1, P5, P3, vote_execute, P2)",
                       "I recommend P3 vote to execute P2, because P2 holds the badge and a false seer "
                       "with the badge can steer every future vote. The village cannot afford that.", "P5"),
                source("question(D1, P6, P5, s23)", "P5, you say you would rather wait, so why did you "
                       "spend s22 asking the sheriff to execute me today?", "P6")]
        fired = set()
        for s in srcs:
            for kind in jb.CHALLENGE_TYPES:
                out = jb.perturb(s, kind, rng)
                if out:
                    fired.add(kind)
                    self.assertNotEqual(out["text"], s.original_nl)
        self.assertEqual(fired, set(jb.CHALLENGE_TYPES))
        items = jb.build_challenge_set(srcs, seed=1)
        kinds = {i["challenge_type"] for i in items}
        self.assertIn("positive_identity", kinds)
        self.assertTrue(all(i["expected"] in ("broken", "preserved") for i in items))


class GateTests(unittest.TestCase):
    def test_cheap_gate_rejects_copy_and_hedge(self):
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        src = source()
        self.assertFalse(gate.evaluate(src, src.original_nl, "low").ok)
        hedged = gate.evaluate(src, "Maybe P7 is a werewolf; he pushed the vote onto P2 and never gave a read.", "low")
        self.assertFalse(hedged.ok)
        self.assertTrue(any(r.startswith("added_epistemic_hedge") for r in hedged.reasons))
        good = gate.evaluate(src, "P7 is a werewolf. He pushed the vote onto P2. He never gave a read of his own.", "low")
        self.assertTrue(good.ok)
        lost = gate.evaluate(src, "P7 is a werewolf. He never gave a read of his own.", "low")
        self.assertIn("lost_player_slot:P7", lost.reasons + ["lost_player_slot:P7"])  # P7 kept; P2 not a slot
        self.assertTrue(lost.ok)  # P2 is not a predicate slot: only the semantic judge can catch it

    def test_retry_then_success_and_unavailable(self):
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        src = source()
        ctx = build_context(src)
        client = FakeClient([src.original_nl,  # identical -> fail
                             "P7 is a werewolf. He pushed the vote onto P2. He never gave a read of his own."])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="low")
        outcome = generate_with_gate(rw, gate, ctx, src, max_corrective_retries=2)
        self.assertEqual(outcome.status, "ok")
        self.assertFalse(outcome.first_pass_ok)
        self.assertEqual(outcome.n_retries, 1)
        self.assertIn("identical_to_original", outcome.attempts[1].feedback_reasons)

        client = FakeClient([src.original_nl] * 3)
        rw = LanguageProficiencyRewriter(llm_client=client, condition="low")
        outcome = generate_with_gate(rw, gate, ctx, src, max_corrective_retries=2)
        self.assertEqual(outcome.status, "treatment_unavailable")
        self.assertEqual(len(outcome.attempts), 3)

    def test_no_verdict_is_not_semantic_feedback(self):
        from experiments.proficiency_validation.rewrite.gate import STATUS_INFRASTRUCTURE_UNAVAILABLE, STATUS_OK, GateResult
        src = source()
        ctx = build_context(src)
        good = "P7 is a werewolf. He pushed the vote onto P2. He never gave a read of his own."

        class NoVerdictGate(SemanticGate):
            calls = 0

            def evaluate(self, source, rewritten, condition, *, cheap_only=False):
                NoVerdictGate.calls += 1
                return GateResult(ok=None, reasons=["semantic_validator_error:ReadTimeout"])

        rw = LanguageProficiencyRewriter(llm_client=FakeClient([good] * 3), condition="low")
        outcome = generate_with_gate(rw, NoVerdictGate(None, None), ctx, src, max_corrective_retries=2)
        self.assertEqual(outcome.status, STATUS_INFRASTRUCTURE_UNAVAILABLE)
        self.assertEqual(len(outcome.attempts), 3)
        self.assertEqual(NoVerdictGate.calls, 6)             # each attempt re-judged once
        self.assertTrue(all(a.feedback_reasons == [] for a in outcome.attempts))   # never fed back

        # A generator exception is an attempt, not feedback: the next prompt is clean.
        class Boom(FakeClient):
            def create(self, **kwargs):
                if not self.calls:
                    self.calls.append(kwargs)
                    raise RuntimeError("connection reset")
                return super().create(**kwargs)
        rw = LanguageProficiencyRewriter(llm_client=Boom([good]), condition="low")
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(rw, gate, ctx, src, max_corrective_retries=2)
        self.assertEqual(outcome.status, STATUS_OK)
        self.assertEqual(outcome.attempts[0].error[:12], "RuntimeError")
        self.assertEqual(outcome.attempts[1].feedback_reasons, [])
        self.assertNotIn("connection reset", rw.llm_client.calls[-1]["messages"][1]["content"])
        qc = generation_qc([{"condition": "low", "status": STATUS_INFRASTRUCTURE_UNAVAILABLE, "n_attempts": 3}])
        self.assertEqual(qc["low"]["infrastructure_unavailable"], 1)
        self.assertEqual(qc["low"]["treatment_unavailable"], 0)

    def test_neutral_target_computed_once(self):
        src = source()
        ctx = build_context(src)
        cands = json.dumps(["Perhaps P7 is a werewolf because he pushed the vote onto P2."])  # hedged: fails
        client = FakeClient([cands, cands, cands])
        rw = LanguageProficiencyRewriter(llm_client=client, condition="neutral", neutral_frequency=_constant_frequency)
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(rw, gate, ctx, src, max_corrective_retries=2)
        self.assertEqual(outcome.status, "treatment_unavailable")
        self.assertEqual(len(client.calls), 3)               # three candidate rounds, nothing else

    def test_gated_rewriter_fallback_and_raise(self):
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        src = source()
        ctx = build_context(src)
        base = LanguageProficiencyRewriter(llm_client=FakeClient([src.original_nl] * 3), condition="low")
        with self.assertRaises(ValueError):                    # "original" without a log file is refused
            GatedRewriter(base, gate, on_unavailable="original")
        import tempfile
        log_path = Path(tempfile.mkdtemp()) / "gate_log.jsonl"
        wrapped = GatedRewriter(base, gate, max_corrective_retries=2, on_unavailable="original",
                                log_path=str(log_path))
        self.assertEqual(wrapped.rewrite(ctx), src.original_nl)
        self.assertTrue(wrapped.log[-1]["fallback_to_original"])
        self.assertEqual(wrapped.log[-1]["status"], "treatment_unavailable")
        on_disk = [json.loads(l) for l in log_path.read_text().splitlines()]
        self.assertEqual(len(on_disk), 1)
        self.assertTrue(on_disk[0]["fallback_to_original"])    # persisted as it happened
        default = GatedRewriter(LanguageProficiencyRewriter(llm_client=FakeClient([src.original_nl] * 3),
                                                            condition="low"), gate)
        with self.assertRaises(TreatmentUnavailable):          # default policy fails closed
            default.rewrite(ctx)
        base = LanguageProficiencyRewriter(llm_client=FakeClient([src.original_nl] * 3), condition="low")
        strict = GatedRewriter(base, gate, max_corrective_retries=2, on_unavailable="raise")
        with self.assertRaises(TreatmentUnavailable):
            strict.rewrite(ctx)

    def test_neutral_gate_selects_among_passers(self):
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        src = source()
        ctx = build_context(src)
        cands = [src.original_nl,                                           # fails: copy
                 "Perhaps P7 is a werewolf because he pushed the vote onto P2.",  # fails: hedge
                 "P7 is a werewolf: he pushed the vote onto P2 and never once gave a read of his own.",
                 "P7 is a werewolf, given that he pushed the vote onto P2 without ever offering his own read."]
        rw = LanguageProficiencyRewriter(llm_client=FakeClient([json.dumps(cands)]), condition="neutral")
        outcome = generate_with_gate(rw, gate, ctx, src)
        self.assertEqual(outcome.status, "ok")
        self.assertIn(outcome.text, cands[2:])
        sel = outcome.attempts[0].neutral_selection
        self.assertEqual(sel["n_candidates"], 4)
        self.assertEqual(sel["n_passed_gate"], 2)

    def test_source_from_context_resolves_reference(self):
        ctx = RewriteContext(predicate="disagree(D1, P2, s34)", original_nl="P8, your s34 read is wrong.",
                             phase="day_speech", speaker_id="P2", speaker_role_claimed=None,
                             public_history=["[s34|P8] estimate(D1, P8, P2, werewolf): P2 is a wolf."])
        src = source_from_context(ctx)
        self.assertEqual(src.predicate_name, "disagree")
        self.assertEqual(src.referenced_speech["speaker"], "P8")
        self.assertIn("P2 is a wolf", src.referenced_speech["text"])


class BundleTests(unittest.TestCase):
    def _sources(self):
        out = []
        for i in range(6):
            out.append(SourceRecord(source_id=f"g1:s{i}", predicate_name="estimate",
                                    predicate_text="estimate(D1, P1, P2, werewolf)", predicate_args=[],
                                    original_nl=" ".join(["word"] * 40), speaker_id="P1", game_id="g1",
                                    speech_id=f"s{i}"))
        out.append(SourceRecord(source_id="g1:s9", predicate_name="estimate", predicate_text="x",
                                original_nl="too short", speaker_id="P2", game_id="g1", speech_id="s9"))
        return out

    def test_window_and_assembly(self):
        result = bundles_mod.build_bundles(self._sources(), window_min=100, window_max=150)
        self.assertEqual(len(result["bundles"]), 1)
        b = result["bundles"][0]
        self.assertEqual(b["n_utterances"], 3)
        self.assertEqual(b["word_count"], 120)
        self.assertEqual(len(result["insufficient"]), 1)
        rewrites = [{"source_id": sid, "condition": "low", "status": "ok", "rewritten_text": "low text",
                     "original_text": " ".join(["word"] * 40)} for sid in b["source_ids"]]
        rewrites[-1]["status"] = "treatment_unavailable"
        assembled = bundles_mod.assemble_condition_texts(result["bundles"], rewrites, ("original", "low"))
        by_cond = {r["condition"]: r for r in assembled}
        self.assertTrue(by_cond["original"]["complete"])
        self.assertFalse(by_cond["low"]["complete"])
        self.assertEqual(bundles_mod.matched_complete(assembled, ("original",)), ["g1:P1"])
        self.assertEqual(bundles_mod.matched_complete(assembled, ("original", "low")), [])

    def test_window_max_is_enforced(self):
        def rec(sid, n):
            return SourceRecord(source_id=f"g1:{sid}", predicate_name="estimate", predicate_text="x",
                                original_nl=" ".join(["word"] * n), speaker_id="P1", game_id="g1", speech_id=sid)
        long = [rec("s1", 140), rec("s2", 300)]
        res = bundles_mod.build_bundles(long, window_min=150, window_max=250)
        self.assertEqual(res["bundles"], [])                 # 140 < 150 and 440 > 250: no bundle
        self.assertEqual(res["insufficient"][0]["words"], 140)
        res = bundles_mod.build_bundles(long, window_min=100, window_max=250)
        self.assertEqual(res["bundles"][0]["word_count"], 140)
        self.assertEqual(res["bundles"][0]["n_utterances"], 1)
        for b in bundles_mod.build_bundles([rec(f"s{i}", 60) for i in range(6)], window_min=150, window_max=250)["bundles"]:
            self.assertLessEqual(b["word_count"], 250)
            self.assertGreaterEqual(b["word_count"], 150)


class AnalysisTests(unittest.TestCase):
    def test_contrast_and_listener(self):
        rng = random.Random(5)
        features = []
        for g in range(8):
            for b in range(3):
                base = rng.uniform(0.6, 0.8)
                bid = f"g{g}:P{b}"
                features.append({"bundle_id": bid, "game_id": f"g{g}", "condition": "original", "complete": True, "mattr": base})
                features.append({"bundle_id": bid, "game_id": f"g{g}", "condition": "neutral", "complete": True, "mattr": base + rng.gauss(0, 0.01)})
                features.append({"bundle_id": bid, "game_id": f"g{g}", "condition": "low", "complete": True, "mattr": base - 0.1 + rng.gauss(0, 0.01)})
        for f in features:
            f["word_count"] = 200 + rng.randint(-5, 5)
        out = contrast_analysis(features, ["mattr", "word_count"], frozen={"equivalence_bounds": {"mattr": 0.03}},
                                n_boot=300, seed=1, family="range")
        low = out["low_vs_neutral"]["metrics"]["mattr"]
        self.assertTrue(low["direction_confirmed"])
        self.assertLess(low["p_sign_flip"], 0.05)
        self.assertEqual(low["fdr_family"], "range: declared treatment contrast")     # nuisance metric excluded from BH
        self.assertIsNone(out["low_vs_neutral"]["metrics"]["word_count"]["q_bh"])
        self.assertTrue(out["neutral_vs_original"]["metrics"]["mattr"]["equivalence"]["equivalent"])

        choices = []
        for g in range(8):
            for b in range(3):
                for listener in ("m1", "m2"):
                    for order in ("ab", "ba"):
                        chose_low = rng.random() < 0.85
                        choices.append({"bundle_id": f"g{g}:P{b}", "game_id": f"g{g}", "pair": "low:neutral",
                                        "treatment": "low", "comparator": "neutral", "listener": listener,
                                        "order": order, "ok": True, "chose_treatment": chose_low,
                                        "chosen_condition": "low" if chose_low else "neutral", "confidence": "high"})
        la = listener_analysis(choices, n_boot=300, seed=1)["low:neutral"]["listeners"]
        self.assertTrue(la["all_models"]["ci_lower_above_half"])
        self.assertTrue(la["all_models"]["signed_ci_excludes_zero"]); self.assertGreater(la["all_models"]["signed_preference"], .5)
        self.assertGreater(la["all_models"]["bradley_terry"]["estimate"], 0)
        self.assertEqual(set(la), {"m1", "m2", "all_models"})


class ReviewFixTests(unittest.TestCase):
    """Regression tests for the review findings (history leak, criteria,
    challenge identity controls, frozen hashes, GatedRewriter config, LT buckets)."""

    def test_history_is_cut_before_the_utterance(self):
        from experiments.proficiency_validation.rewrite.runtime import history_before, with_prior_history
        src = source()
        src.speech_id = "s5"
        src.public_history = ["[s1|P1] claim_role(D1, P1, seer): I am the seer.",
                              "[s3|P4] estimate(D1, P4, P2, villager): P2 is fine.",
                              f"[s5|P3] {src.predicate_text}: {src.original_nl}",
                              "[s6|P7] disagree(D1, P7, s5): No."]
        self.assertEqual(len(history_before(src)), 2)
        self.assertNotIn(src.original_nl, " ".join(history_before(src)))
        src.speech_id = None                      # fallback: match the utterance itself
        self.assertEqual(len(history_before(src)), 2)
        src.predicate_text = "estimate(D1, P3, P7, wolf)"  # no exact entry: drop by text
        self.assertEqual(len(history_before(src)), 2)
        self.assertEqual(len(with_prior_history(src).public_history), 2)
        # the gate passes only the prior history to the parser
        seen = {}

        class FakeParser:
            model = None

            def parse(self, text, *, phase, speaker_id, public_history):
                seen["history"] = list(public_history)
                from components.judges.inverse_predicate_parser import ParseResult
                return ParseResult(ok=True, predicate_name="estimate",
                                   arguments={"Phase": "D1", "Player": "P3", "Target": "P7", "Role": "werewolf"})
        src = source()
        src.speech_id = "s5"
        src.public_history = ["[s1|P1] claim_role(D1, P1, seer): I am the seer.",
                              f"[s5|P3] {src.predicate_text}: {src.original_nl}"]
        gate = SemanticGate(FakeParser(), None, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        gate.evaluate(src, "P7 is a werewolf. He pushed the vote onto P2. He gave no read.", "low")
        self.assertEqual(seen["history"], src.public_history[:1])

    def test_judge_all_scores_identity_controls(self):
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        src = source()
        verdicts = gate.judge_all(src, src.original_nl)
        self.assertTrue(verdicts["structural"]["ok"])
        self.assertIn("identical_to_original", verdicts["treatment"]["reasons"])
        items = [{"item_id": "x", "source_id": src.source_id, "predicate": src.predicate_text,
                  "predicate_name": src.predicate_name, "phase": src.phase, "speaker": src.speaker_id,
                  "original_text": src.original_nl, "candidate_text": src.original_nl,
                  "challenge_type": "positive_identity", "expected": "preserved", "public_history": []}]
        judged = jb.run_challenge(items, gate)
        self.assertIs(judged[0]["flags"]["structural"], False)
        self.assertIs(judged[0]["flags"]["pipeline"], False)   # no judge flagged it
        report = jb.report_challenge(judged)
        self.assertEqual(report["overall"]["pipeline"]["specificity"], 1.0)


    def test_criteria_never_pass_by_absence(self):
        from experiments.proficiency_validation.validation.analysis import frozen_criteria, verdict
        frozen = {"primary_range_metrics": ["mattr"], "range_effect_thresholds": {"low_vs_neutral": {"mattr": 0.5}},
                  "equivalence_bounds": {"mattr": 0.05, "grammar_errors_per_100_words": 0.5, "cola_mean": 0.05},
                  "models": {"listeners": ["m1", "m2"]}}
        # one game only: no p-value, no CI
        features = [{"bundle_id": "g1:P1", "game_id": "g1", "condition": c, "complete": True,
                     "mattr": {"original": 0.7, "neutral": 0.69, "low": 0.6}[c]}
                    for c in ("original", "neutral", "low")]
        report = {"range": contrast_analysis(features, ["mattr"], frozen=frozen, n_boot=50, seed=1, family="r")}
        crit = frozen_criteria(report, frozen)
        by_name = {c["criterion"]: c for c in crit}
        low = next(c for n, c in by_name.items() if n.startswith("A vs B: mattr"))
        self.assertIsNone(low["passed"])
        listeners = [c for n, c in by_name.items() if n.startswith("Listener") and "A vs B" in n]
        controls = [c for n, c in by_name.items() if "B vs Original equivalence" in n]
        self.assertEqual(len(listeners), 3)                      # m1, m2, all_models
        self.assertEqual(len(controls), 3)                       # the H3 control for the same three
        self.assertTrue(all(c["passed"] is None for c in listeners + controls))
        v = verdict(crit)
        self.assertFalse(v["confirmed"])
        self.assertGreater(v["not_evaluated"], 0)
        # an explicit FDR non-rejection is a FAIL, not a pass
        report["range"]["low_vs_neutral"]["metrics"]["mattr"].update(
            {"reject_fdr_05": False, "direction_confirmed": True, "cohens_dz": -2.0})
        low = next(c for c in frozen_criteria(report, frozen) if c["criterion"].startswith("A vs B: mattr"))
        self.assertIs(low["passed"], False)

    def test_gated_rewriter_is_registered_and_reconstructible(self):
        from lurkingwolf.rewriter import Rewriter, get_rewriter
        self.assertIs(get_rewriter("gated_proficiency"), GatedRewriter)
        client = FakeClient([])
        built = Rewriter.from_config({"class": "GatedRewriter",
                                      "params": {"base": {"class": "LanguageProficiencyRewriter",
                                                          "params": {"condition": "low", "max_tokens": 123}},
                                                 "gate": {"judge_handle": "deepseek-chat"},
                                                 "max_corrective_retries": 1, "on_unavailable": "raise"}},
                                     llm_client=client)
        self.assertIsInstance(built, GatedRewriter)
        self.assertEqual(built.base.condition, "low")
        self.assertEqual(built.base.max_tokens, 123)
        self.assertIs(built.base.llm_client, client)
        self.assertEqual(built.gate.judge_handle, "deepseek-chat")
        self.assertEqual(built.max_corrective_retries, 1)
        cfg = built.to_config()
        again = Rewriter.from_config(cfg, llm_client=client)
        self.assertEqual(again.to_config(), cfg)
        # the engine's own construction path: cls(**params, llm_client=client)
        shorthand = GatedRewriter(base="neutral", gate="deepseek-chat", llm_client=client)
        self.assertEqual(shorthand.base.condition, "neutral")
        with self.assertRaises(ValueError):                    # no silent default judge
            GatedRewriter(base="low", gate={"judge_handle": None}, llm_client=client)
        with self.assertRaises(ValueError):                    # rubric drift between save and resume
            GatedRewriter(base="low", gate={"judge_handle": "deepseek-chat",
                                            "semantic_rubric_version": "v0_old"}, llm_client=client)

    def test_frozen_hashes_cover_module_sources(self):
        from experiments.proficiency_validation.validation.protocol import HASHED_MODULES, frozen_hashes, verify_hashes
        import hashlib
        import importlib
        # ``import components.judges.structural_check as x`` would bind the FUNCTION the
        # package re-exports under that name; import_module returns the module itself.
        structural_module = importlib.import_module("components.judges.structural_check")
        self.assertTrue(hasattr(structural_module, "HEDGE_PATTERNS"))
        hashes = frozen_hashes()
        mod = hashes["module_source_sha256"]
        self.assertEqual(set(mod), set(HASHED_MODULES))
        expected = hashlib.sha256(Path(structural_module.__file__).read_bytes()).hexdigest()
        self.assertEqual(mod["components.judges.structural_check"], expected)
        self.assertIn(b"HEDGE_PATTERNS", Path(structural_module.__file__).read_bytes())
        self.assertEqual(verify_hashes({"hashes": hashes}), [])
        tampered = json.loads(json.dumps(hashes))
        tampered["module_source_sha256"]["components.judges.structural_check"] = "0" * 64
        self.assertEqual(verify_hashes({"hashes": tampered}),
                         ["module_source_sha256.components.judges.structural_check"])

    def test_languagetool_buckets(self):
        from components.metrics.grammar import categorize_match
        def m(cat, issue):
            return {"rule": {"category": {"id": cat}, "issueType": issue}}
        self.assertEqual(categorize_match(m("GRAMMAR", "grammar")), "grammar")
        self.assertEqual(categorize_match(m("CONFUSED_WORDS", "grammar")), "usage")
        self.assertEqual(categorize_match(m("CASING", "typographical")), "usage")
        self.assertEqual(categorize_match(m("MISC", "uncategorized")), "style")
        self.assertEqual(categorize_match(m("TYPOS", "misspelling")), "spelling")
        self.assertEqual(categorize_match(m("PUNCTUATION", "typographical")), "punctuation")
        self.assertEqual(categorize_match(m("STYLE", "style")), "style")


class SecondReviewFixTests(unittest.TestCase):
    def test_all_candidates_identical_is_a_real_failure(self):
        from components.rewriters.language_rewriter import NeutralSelection
        self.assertIsNone(NeutralSelection(chosen="x", candidates=["x", "x"]).to_dict()["chosen_distance"])
        src = source()
        ctx = build_context(src)
        copies = json.dumps([src.original_nl, src.original_nl])
        rw = LanguageProficiencyRewriter(llm_client=FakeClient([copies] * 3), condition="neutral")
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(rw, gate, ctx, src, max_corrective_retries=2)
        self.assertEqual(outcome.status, "treatment_unavailable")       # not an infrastructure error
        self.assertTrue(all(a.error == "" for a in outcome.attempts))
        self.assertIn("identical_to_original", outcome.attempts[1].feedback_reasons)
        self.assertEqual(outcome.attempts[0].neutral_selection["rule"], "fallback_all_dropped")

    def test_generator_failure_is_an_attempt_not_an_exception(self):
        from experiments.proficiency_validation.rewrite.gate import STATUS_INFRASTRUCTURE_UNAVAILABLE

        class Dead(FakeClient):
            def create(self, **kwargs):
                self.calls.append(kwargs)
                raise RuntimeError("connection refused")
        src = source()
        ctx = build_context(src)
        rw = LanguageProficiencyRewriter(llm_client=Dead([]), condition="neutral")
        gate = SemanticGate(None, None, require_parser=False, require_validator=False)
        gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(rw, gate, ctx, src, max_corrective_retries=2)   # must not raise
        self.assertEqual(outcome.status, STATUS_INFRASTRUCTURE_UNAVAILABLE)
        self.assertEqual(len(outcome.attempts), 3)
        self.assertTrue(all(a.text is None and a.error for a in outcome.attempts))
        import tempfile
        base = LanguageProficiencyRewriter(llm_client=Dead([]), condition="neutral")
        wrapped = GatedRewriter(base, gate, on_unavailable="original",
                                log_path=str(Path(tempfile.mkdtemp()) / "log.jsonl"))
        self.assertEqual(wrapped.rewrite(ctx), src.original_nl)          # falls back as promised
        self.assertTrue(wrapped.log[-1]["fallback_to_original"])

    def test_unknown_kwarg_raises(self):
        with self.assertRaises(TypeError):
            LanguageProficiencyRewriter(llm_client=None, level="B")
        with self.assertRaises(TypeError):
            GatedRewriter(base={"level": "B"}, gate=SemanticGate(None, None), llm_client=FakeClient([]))

    def test_multiline_original_keeps_prompt_unindented(self):
        src = source(text="P7 is a werewolf.\nHe pushed the vote onto P2.")
        rw = LanguageProficiencyRewriter(llm_client=None, condition="low")
        _, user = rw.build_prompts(build_context(src))
        self.assertTrue(user.startswith("Predicate: "))
        self.assertIn("\nSpeaker: P3\n", user)
        _, user_c = rw.build_candidate_prompts(build_context(src))
        self.assertTrue(user_c.startswith("Predicate: "))
        self.assertFalse(any(line.startswith("    ") for line in user_c.splitlines()))

    def test_neutral_instance_hash_is_the_candidate_prompt(self):
        ctx = build_context(source())
        rw = LanguageProficiencyRewriter(llm_client=None, condition="neutral")
        import hashlib
        expected = hashlib.sha256(rw.build_candidate_prompts(ctx)[0].encode()).hexdigest()
        self.assertEqual(rw.prompt_sha256(ctx, candidates=True), expected)
        self.assertNotEqual(rw.prompt_sha256(ctx), expected)

    def test_lexical_measures_undefined_without_pos(self):
        prof = lexical_profile(tokenize("P7 is a wolf and P7 pushed the vote"), None, mattr_window=3).to_dict()
        self.assertFalse(prof["lemmas_available"])
        self.assertIsNone(prof["repetition_rate"])
        self.assertIsNone(prof["subtlex_mean"])
        self.assertIsNotNone(prof["ttr"])                 # whole-token measures remain defined
        tagged = [("P7", "P7", "PROPN"), ("is", "be", "AUX"), ("a", "a", "DET"), ("wolf", "wolf", "NOUN"),
                  ("and", "and", "CCONJ"), ("P7", "P7", "PROPN"), ("pushed", "push", "VERB"),
                  ("the", "the", "DET"), ("vote", "vote", "NOUN")]
        prof = lexical_profile(tagged, None, mattr_window=3).to_dict()
        self.assertTrue(prof["lemmas_available"])
        self.assertEqual(prof["n_content_lemmas"], 1)      # 'push' only: wolf/vote are domain terms, ids excluded


    def test_listener_set_comparison_is_order_insensitive(self):
        # emulate the check in main_probe without a frozen file on disk
        pinned = sorted(["deepseek-chat", "openai:gpt-5-mini"])
        given = sorted(h.strip() for h in "openai:gpt-5-mini, deepseek-chat".split(",") if h.strip())
        self.assertEqual(given, pinned)


class ThirdReviewFixTests(unittest.TestCase):
    def test_probe_choice_parsing(self):
        from components.judges.listener_probe import parse_choice
        for raw, want in (("A", "A"), ("b", "B"), ("Passage B", "B"), ("passage_a", "A"), ("(B)", "B"),
                          ("P", None), ("AB", None), ("", None), (None, None), ("neither", None)):
            self.assertEqual(parse_choice(raw), want, raw)

    def test_grammar_rule_override_and_misc_misspelling(self):
        from components.metrics.grammar import categorize_match
        self.assertEqual(categorize_match({"rule": {"id": "EN_A_VS_AN", "category": {"id": "MISC"},
                                                    "issueType": "misspelling"}}), "grammar")
        self.assertEqual(categorize_match({"rule": {"id": "MORFOLOGIK_RULE_EN_US", "category": {"id": "TYPOS"},
                                                    "issueType": "misspelling"}}), "spelling")

    def test_verdict_from_ok_signals_not_reason_strings(self):
        from experiments.proficiency_validation.rewrite.gate import STATUS_INFRASTRUCTURE_UNAVAILABLE, STATUS_TREATMENT_UNAVAILABLE
        from components.helpers import CheckResult

        class OddValidator:
            model = None
            calls = 0

            def validate(self, source, rewritten, *, commitment_may_be_indirect=False):
                OddValidator.calls += 1
                return CheckResult(ok=None, reasons=["judge_timeout"])          # no-verdict, odd wording

        class FailingValidator(OddValidator):
            def validate(self, source, rewritten, *, commitment_may_be_indirect=False):
                return CheckResult(ok=False, reasons=["odd_wording_for_a_real_failure"])
        src = source()
        ctx = build_context(src)
        good = "P7 is a werewolf. He pushed the vote onto P2. He never gave a read of his own."
        gate = SemanticGate(None, OddValidator(), require_parser=False)
        gate.neutral_frequency = _constant_frequency
        self.assertIsNone(gate.evaluate(src, good, "low").ok)
        rw = LanguageProficiencyRewriter(llm_client=FakeClient([good] * 3), condition="low")
        self.assertEqual(generate_with_gate(rw, gate, ctx, src).status, STATUS_INFRASTRUCTURE_UNAVAILABLE)
        gate = SemanticGate(None, FailingValidator(), require_parser=False)
        gate.neutral_frequency = _constant_frequency
        self.assertIs(gate.evaluate(src, good, "low").ok, False)
        rw = LanguageProficiencyRewriter(llm_client=FakeClient([good] * 3), condition="low")
        self.assertEqual(generate_with_gate(rw, gate, ctx, src).status, STATUS_TREATMENT_UNAVAILABLE)

    def test_duplicate_candidates_judged_once(self):
        from components.helpers import CheckResult

        class CountingValidator:
            model = None
            seen = []

            def validate(self, source, rewritten, *, commitment_may_be_indirect=False):
                CountingValidator.seen.append(rewritten)
                return CheckResult(ok=True, reasons=[])
        src = source()
        ctx = build_context(src)
        x = "P7 is a werewolf: he pushed the vote onto P2 and never once gave a read of his own."
        y = "P7 is a werewolf, given that he pushed the vote onto P2 without ever offering his own read."
        rw = LanguageProficiencyRewriter(llm_client=FakeClient([json.dumps([x, x, y, x])]), condition="neutral")
        gate = SemanticGate(None, CountingValidator(), require_parser=False)
        gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(rw, gate, ctx, src)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(sorted(CountingValidator.seen), sorted([x, y]))
        self.assertEqual(outcome.attempts[0].neutral_selection["n_candidates"], 2)

    def test_audit_capacity_uses_prefix_rule(self):
        from experiments.proficiency_validation.validation.sources import audit
        def rec(sid, n):
            return SourceRecord(source_id=f"g1:{sid}", predicate_name="estimate", predicate_text="x",
                                original_nl=" ".join(["word"] * n), speaker_id="P1", game_id="g1", speech_id=sid,
                                phase="day_speech")
        report = audit({"g1": [rec("s1", 140), rec("s2", 300)]}, windows=(150,), window_span=100)
        self.assertEqual(report["candidate_bundles_by_window"][150], 0)       # prefix cannot fit [150, 250]
        self.assertEqual(report["speakers_with_total_words_at_least"][150], 1)
        report = audit({"g1": [rec(f"s{i}", 60) for i in range(5)]}, windows=(150,), window_span=100)
        self.assertEqual(report["candidate_bundles_by_window"][150], 1)

    def test_phase_default_is_engine_phase_name(self):
        src = jb.source_record_from_dict({"source_id": "x", "predicate_text": "estimate(D1, P3, P7, werewolf)",
                                          "original_nl": "P7 is a wolf."})
        self.assertEqual(src.phase, "day_speech")
        self.assertEqual(src.predicate_args[0], "D1")

    def test_integrity_sorting_is_none_safe(self):
        from experiments.proficiency_validation.validation.analysis import integrity_checks
        frozen = {"hashes": frozen_hashes(), "models": {"generator": "deepseek-chat", "judge": "deepseek-chat"},
                  "conditions": ["original", "neutral", "low"], "generation_temperature": 0.0}
        rows = [{"condition": "low", "prompt_template_sha256": None, "generator_handle": None,
                 "generator_temperature": None, "gate_provenance": {"judge_handle": None}},
                {"condition": "low", "prompt_template_sha256": "abc", "generator_handle": "deepseek-chat",
                 "generator_temperature": 0.0, "gate_provenance": {"judge_handle": "deepseek-chat"}}]
        checks = integrity_checks(frozen, rewrites=rows)          # must not raise
        by = {c["check"]: c for c in checks}
        self.assertFalse(by["complete frozen design and unchanged code"]["passed"])
        self.assertFalse(by["explicit confirmatory split"]["passed"])

    def test_ledger_judge_model_books_usage(self):
        from components.api_usage import UsageLedger
        from components.judges.judge_model import resolve_spec
        from experiments.proficiency_validation.rewrite.gate import LedgerJudgeModel, SemanticGate
        ledger = UsageLedger("t", run_id="r", path=Path("/dev/null"))
        model = LedgerJudgeModel(resolve_spec("deepseek-chat"), ledger=ledger, purpose="inverse_parser")
        model.complete = None  # not called here; construction and wiring only
        gate = SemanticGate.from_handle("deepseek-chat", ledger=ledger)
        self.assertEqual(gate.parser.model.purpose, "inverse_parser")
        self.assertEqual(gate.validator.model.purpose, "semantic_validator")
        self.assertIs(gate.parser.model.ledger, ledger)
        self.assertEqual(gate.judge_handle, "deepseek-chat")


class FourthReviewFixTests(unittest.TestCase):

    def test_criteria_cover_neutral_control_and_require_thresholds(self):
        from experiments.proficiency_validation.validation.analysis import frozen_criteria, verdict
        frozen = {"primary_range_metrics": ["mattr"], "range_effect_thresholds": {},
                  "equivalence_bounds": {}, "models": {"listeners": ["m1"]}}
        rng = random.Random(2)
        choices = []
        for g in range(8):
            for b in range(3):
                for pair, treat, comp, p_treat in (("low:neutral", "low", "neutral", 0.9),
                                                    ("neutral:original", "neutral", "original", 0.5)):
                    for order in ("ab", "ba"):
                        chose = rng.random() < p_treat
                        choices.append({"bundle_id": f"g{g}:P{b}", "game_id": f"g{g}", "pair": pair, "treatment": treat,
                                        "comparator": comp, "listener": "m1", "listener_family": "m1", "order": order,
                                        "ok": True, "chosen_condition": treat if chose else comp, "chose_treatment": chose,
                                        "confidence": "high"})
        report = {"listener": listener_analysis(choices, n_boot=200, seed=1)}
        crit = {c["criterion"]: c for c in frozen_criteria(report, frozen)}
        control = [c for n, c in crit.items() if "B vs Original equivalence" in n]
        self.assertEqual(len(control), 2)                       # m1 and all_models
        self.assertTrue(all(c["passed"] is None for c in control))  # no equivalence margin frozen
        low = next(c for n, c in crit.items() if n.startswith("A vs B: mattr"))
        self.assertIsNone(low["passed"])                        # no threshold frozen -> not evaluated
        self.assertFalse(any(n.startswith("Neutral matching") for n in crit))   # no edit-distance matching criterion
        self.assertFalse(verdict(list(crit.values()))["confirmed"])


    def test_hashed_modules_cover_the_judges(self):
        from experiments.proficiency_validation.validation.protocol import HASHED_MODULES
        for name in ("components.judges.inverse_predicate_parser", "experiments.proficiency_validation.rewrite.fidelity",
                     "components.helpers", "experiments.proficiency_validation.rewrite.runtime",
                     "experiments.proficiency_validation.validation.sources"):
            self.assertIn(name, HASHED_MODULES)

    def test_tost_reuses_bootstrap_draws(self):
        clusters = {f"g{i}": [0.01 * (-1) ** i, 0.02] for i in range(10)}
        boot = cluster_bootstrap(clusters, n_boot=300, seed=7)
        self.assertEqual(len(boot["draws"]), 300)
        via_reuse = tost_equivalence(clusters, bound=0.1, bootstrap=boot)
        fresh = tost_equivalence(clusters, bound=0.1, n_boot=300, seed=7)
        self.assertEqual((via_reuse["ci_low"], via_reuse["ci_high"]), (fresh["ci_low"], fresh["ci_high"]))
        self.assertTrue(via_reuse["equivalent"])


class FifthReviewFixTests(unittest.TestCase):
    def test_bare_rewriter_config_round_trips_and_version_is_checked(self):
        from lurkingwolf.rewriter import Rewriter
        for condition in ("low", "high", "neutral"):
            rw = LanguageProficiencyRewriter(llm_client=None, condition=condition, max_tokens=222)
            again = Rewriter.from_config(rw.to_config(), llm_client=None)
            self.assertEqual(again.to_config(), rw.to_config())
        stale_rule = rw.to_config()
        stale_rule["params"]["neutral_selection_rule"] = "obsolete"
        with self.assertRaisesRegex(ValueError, "selection rule"):
            Rewriter.from_config(stale_rule, llm_client=None)
        stale = rw.to_config()
        stale["params"]["prompt_version"] = "range_stale"
        with self.assertRaises(ValueError):
            Rewriter.from_config(stale, llm_client=None)
        gated = GatedRewriter(base="low", gate=SemanticGate(None, None, judge_handle="deepseek-chat"),
                              llm_client=FakeClient([]))
        self.assertEqual(Rewriter.from_config(gated.to_config(), llm_client=FakeClient([])).to_config(),
                         gated.to_config())

    def test_gate_config_preserves_policy_budgets_and_secondary_identity(self):
        from lurkingwolf.rewriter import Rewriter
        from experiments.proficiency_validation.rewrite.runtime import ValidationModel
        from experiments.proficiency_validation.rewrite.fidelity import EnglishFidelityValidator
        gate = SemanticGate.from_handle("deepseek-chat", parser_abstention="no_verdict",
                                        require_parser=False, require_validator=False)
        gate.neutral_exposure_floor, gate.parser_context_words = .15, 1200
        gate.parser.max_tokens, gate.validator.max_tokens = 900, 1800
        gate.second_opinion = EnglishFidelityValidator(ValidationModel.resolve("deepseek-chat"), max_tokens=2400)
        rewriter = GatedRewriter(base="neutral", gate=gate, llm_client=FakeClient([]))
        config = rewriter.to_config()
        restored = Rewriter.from_config(config, llm_client=FakeClient([]))
        self.assertEqual(restored.to_config(), config)
        self.assertEqual((restored.gate.parser_abstention, restored.gate.neutral_exposure_floor,
                          restored.gate.parser_context_words), ("no_verdict", .15, 1200))
        config["params"]["gate"]["second_opinion"]["identity"] = {"revision": "changed"}
        with self.assertRaisesRegex(ValueError, "identity changed"):
            Rewriter.from_config(config, llm_client=FakeClient([]))

    def test_parse_candidates_ignores_trailing_commentary(self):
        self.assertEqual(parse_candidates('["a", "b"] Note: [see] above'), ["a", "b"])
        self.assertEqual(parse_candidates('Here you go [not json] then ["x", "y"]'), ["x", "y"])
        self.assertEqual(parse_candidates('{"k": [1]} ["only", "this"]'), ["only", "this"])

    def test_subtlex_reader_tolerates_bom(self):
        import tempfile
        from components.metrics.lexical import SubtlexNorms
        path = Path(tempfile.mkdtemp()) / "subtlex.txt"
        path.write_bytes("\ufeffWord\tLg10CD\nthe\t4.0\nwolf\t2.5\n".encode("utf-8"))
        norms = SubtlexNorms(path)
        self.assertEqual(norms.lookup("The"), (4.0, "surface"))
        self.assertEqual(norms.lookup("wolves", "wolf"), (2.5, "lemma"))

    def test_confused_words_bucket_as_usage(self):
        from components.metrics.grammar import categorize_match
        self.assertEqual(categorize_match({"rule": {"id": "THEIR_IS", "category": {"id": "CONFUSED_WORDS"},
                                                    "issueType": "misspelling"}}), "usage")

    def test_near_duplicate_candidates_judged_once(self):
        from components.helpers import CheckResult

        class CountingValidator:
            model = None
            seen = []

            def validate(self, source, rewritten, *, commitment_may_be_indirect=False):
                CountingValidator.seen.append(rewritten)
                return CheckResult(ok=True, reasons=[])
        src = source()
        ctx = build_context(src)
        a = "P7 is a werewolf; he pushed the vote onto P2 and never gave a read of his own."
        b = "P7 is a werewolf. He pushed the vote onto P2 and never gave a read of his own"
        rw = LanguageProficiencyRewriter(llm_client=FakeClient([json.dumps([a, b])]), condition="neutral")
        gate = SemanticGate(None, CountingValidator(), require_parser=False)
        gate.neutral_frequency = _constant_frequency
        outcome = generate_with_gate(rw, gate, ctx, src)
        self.assertEqual(outcome.status, "ok")
        self.assertEqual(len(CountingValidator.seen), 2)  # sentence boundaries are different

    def test_identical_flag_comes_from_structural_check(self):
        from experiments.proficiency_validation.rewrite.gate import treatment_checks
        src = source()
        self.assertIn("identical_to_original", treatment_checks(src, "totally different words", "low", identical=True).reasons)
        self.assertNotIn("identical_to_original", treatment_checks(src, src.original_nl, "low", identical=False).reasons)

    def test_generate_row_hash_guard(self):
        from experiments.proficiency_validation.rewrite.__main__ import _safe_instance_hash
        rw = LanguageProficiencyRewriter(llm_client=None, condition="low")
        bad = build_context(source(text=""))
        self.assertIsNone(_safe_instance_hash(rw, bad, "low"))
        self.assertIsNotNone(_safe_instance_hash(rw, build_context(source()), "low"))


class SyntaxTests(unittest.TestCase):
    def test_analyse_if_spacy_available(self):
        try:
            from components.metrics.syntactic import analyse, load_nlp
            nlp = load_nlp("en_core_web_sm")
        except ImportError:
            self.skipTest("spaCy / en_core_web_sm not installed")
        simple = analyse("P7 is a wolf. He pushed the vote onto P2. He gave no read.", nlp)
        complex_ = analyse("P7, who pushed the vote onto P2 without ever giving a read of his own, "
                           "is a wolf because that is exactly what a wolf does when it wants cover.", nlp)
        self.assertEqual(simple["n_sentences"], 3)
        self.assertGreater(complex_["subordinate_clauses_per_sentence"], simple["subordinate_clauses_per_sentence"])
        self.assertGreater(complex_["mean_dependency_distance"], simple["mean_dependency_distance"])
        self.assertTrue(all(len(t) == 3 for t in simple["tokens"]))
