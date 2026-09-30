"""Synthetic fidelity challenge set: build perturbations and minimal pairs, judge them, score sensitivity and specificity."""
from __future__ import annotations
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple
from components.helpers import SourceRecord
from components.metrics.text_normalization import PLAYER_ID_RE as PLAYER_RE
from components.stats import wilson_interval
from experiments.proficiency_validation.rewrite.runtime import fmt, now_iso
import sys
from pathlib import Path
from components.helpers import read_jsonl, write_jsonl
import experiments.proficiency_validation.rewrite.__main__ as generate
from experiments.proficiency_validation.rewrite.runtime import dump_json
from experiments.proficiency_validation.rewrite.runtime import source_record_from_dict


# ── scoring ──

JUDGES = ("structural", "parser", "validator", "pipeline")


def confusion(rows: Sequence[Tuple[bool, Optional[bool]]]) -> dict:
    """``rows`` = (gold_broken, judge_flagged); ``None`` flags are no-verdicts."""
    tp = sum(1 for g, f in rows if g and f is True)
    fn = sum(1 for g, f in rows if g and f is False)
    fp = sum(1 for g, f in rows if not g and f is True)
    tn = sum(1 for g, f in rows if not g and f is False)
    nv = sum(1 for _, f in rows if f is None)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    broken_total = sum(g for g, _ in rows)
    preserved_total = len(rows) - broken_total
    return {
        "n": len(rows), "n_broken": tp + fn, "n_preserved": tn + fp, "no_verdict": nv,
        "n_broken_all": broken_total, "n_preserved_all": preserved_total,
        "valid_fraction": (len(rows) - nv) / len(rows) if rows else None,
        "broken_detection_all": tp / broken_total if broken_total else None,
        "preserved_acceptance_all": tn / preserved_total if preserved_total else None,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens, "sensitivity_ci95": wilson_interval(tp, tp + fn)[1:],
        "specificity": spec, "specificity_ci95": wilson_interval(tn, tn + fp)[1:],
        "precision_of_flag": tp / (tp + fp) if tp + fp else None,
        "npv_of_pass": tn / (tn + fn) if tn + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def _flag(value: Optional[bool]) -> Optional[bool]:
    """A judge 'ok' value -> flagged (True), passed (False) or no verdict (None)."""
    if value is True:
        return False
    if value is False:
        return True
    return None


# ── synthetic challenge set ──

ROLE_FLIPS = [
    (r"\bwerewolf\b", "villager"), (r"\bwerewolves\b", "villagers"),
    (r"\ba wolf\b", "a villager"), (r"\bwolves\b", "villagers"), (r"\bwolf\b", "villager"),
    (r"\bvillager\b", "wolf"), (r"\bvillagers\b", "wolves"),
    (r"\bvillage[- ]side\b", "wolf-side"), (r"\bwolf[- ]side\b", "village-side"),
    (r"\bvillager_camp\b", "wolf_camp"), (r"\bwolf_camp\b", "villager_camp"),
    (r"\bseer\b", "witch"),
]
POLARITY_RULES = [
    (r"\bis not\b", "is"), (r"\bisn't\b", "is"), (r"\bare not\b", "are"), (r"\baren't\b", "are"),
    (r"\bcannot\b", "can"), (r"\bcan't\b", "can"), (r"\bdid not\b", "did"), (r"\bdidn't\b", "did"),
    (r"\bnever\b", "always"), (r"\bno longer\b", "still"),
    (r"\bis a\b", "is not a"), (r"\bis my\b", "is not my"), (r"\bis the\b", "is not the"),
    (r"\bare\b", "are not"), (r"\bis\b", "is not"), (r"\bhas\b", "has not"), (r"\bhave\b", "have not"),
]
CERTAINTY_RULES = [
    (r"\bmight be\b", "is"), (r"\bcould be\b", "is"), (r"\bmay be\b", "is"),
    (r"\bprobably\b", "definitely"), (r"\bI think\b", "I know"), (r"\bI suspect\b", "I know"),
    (r"\bis definitely\b", "might be"), (r"\bis clearly\b", "might be"),
    (r"\bis a\b", "might be a"), (r"\bis my\b", "might be my"), (r"\bis the\b", "might be the"),
    (r"\bis\b", "might be"),
]
# Ordered: the rules that target the act's own verb come first, so a challenge
# in a question is weakened at "why did you" rather than at an incidental verb.
DIRECTIVE_RULES = [
    (r"\bwhy did you\b", "I wonder if you"), (r"\bexplain why\b", "feel free to say why"),
    (r"\bI recommend\b", "I wonder whether"), (r"\bvote to execute\b", "keep an eye on"),
    (r"\bvote out\b", "keep an eye on"), (r"\bvote for\b", "think about"),
    (r"\bpasses the badge to\b", "could consider the badge for"),
    (r"\bthe badge goes to\b", "the badge could maybe go to"),
    (r"\bshould\b", "could perhaps"), (r"\bmust\b", "might"), (r"\bexecute\b", "watch"),
]
REASON_SPLIT = re.compile(r"(?:\s+because\s+|\s+since\s+|:\s+|\s+—\s+|\s+-\s+|,\s+which\s+|,\s+so\s+)", re.IGNORECASE)
REASON_BANK = [
    "because they voted with the confirmed wolf on the first day",
    "because they went completely silent right after the seer claim",
    "because they changed their vote at the last second without a reason",
    "because their reads have matched the dead wolves every single time",
]
CHALLENGE_TYPES = ("speaker_target_swap", "role_flip", "polarity_flip", "action_target_change",
                   "reason_deletion", "reason_substitution", "proposition_deletion",
                   "certainty_shift", "directive_force_shift")

# Small constructed pairs supplement perturbations of real utterances. Their
# labels follow the stated transformation, not a model's vote on a pilot rewrite.
CONTROLLED_CASES = (
    ("association_identity_shift", "broken", "fluff(D1, P3)",
     "P7 says he is not allied with P2.", "P7 says he is not P2."),
    ("evidential_qualification_loss", "broken", "fluff(D1, P3)",
     "P7 gave an unverifiable clear on P2.", "P7 gave a clear on P2."),
    ("evidential_strength_loss", "broken", "fluff(D1, P3)",
     "I have publicly confirmed P2 as village-side.", "I said P2 is village-side."),
    ("action_scope_shift", "broken", "fluff(D1, P3)",
     "I can control the speaking order.", "I can control who speaks."),
    ("positive_association_paraphrase", "preserved", "fluff(D1, P3)",
     "P7 dissociates from P2.", "P7 says he is not allied with P2."),
    ("positive_simplification", "preserved", "claim_role(D1, P3, seer)",
     "I am the Seer, and I can hand the table a verified alignment every day.",
     "I am the Seer, and I can give the table a verified alignment every day."),
    ("positive_simplification", "preserved", "estimate(D1, P3, P7, werewolf)",
     "I read P7 as a wolf because P7 changed their vote without a reason.",
     "I think P7 is a wolf because P7 changed their vote and gave no reason."),
    ("positive_clause_split", "preserved", "estimate(D1, P3, P7, werewolf)",
     "P7 changed their vote without a reason, which makes P7 suspicious.",
     "P7 changed their vote and gave no reason. That makes P7 suspicious."),
    ("claim_attribution_shift", "broken", "fluff(D1, P3)",
     "P7 is a seer claimant.", "P7 is the seer."),
    ("claim_attribution_shift", "broken", "fluff(D1, P3)",
     "P7 claims to be the seer and says I am a villager.", "P7 says I am the seer."),
    ("invented_divination_result", "broken", "fluff(D1, P3)",
     "P7 said their divination named P2.", "P7 said their divination showed P2 is a villager."),
    ("numeric_qualification_loss", "broken", "recommend_action(D1, P3, P2, vote_execute, P7)",
     "P2, use your 1.5 vote to execute P7.", "P2, use your vote to execute P7."),
    ("courtesy_addition", "broken", "recommend_action(D1, P3, P2, vote_execute, P7)",
     "P2, vote to execute P7.", "P2, please vote to execute P7."),
    ("courtesy_deletion", "broken", "recommend_action(D1, P3, P2, vote_execute, P7)",
     "P2, please vote to execute P7.", "P2, vote to execute P7."),
    ("positive_multiple_acts", "preserved", "disagree(D1, P3, s1)",
     "P2's claim that I am a wolf is false, and P2 is a wolf.",
     "P2's claim that I am a wolf is false. P2 is a wolf."),
)


def _first_rule(text: str, rules) -> Optional[Tuple[str, str]]:
    for pattern, replacement in rules:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE), pattern
    return None


def _segments(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\s+—\s+|\s+-\s+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def perturb(source: SourceRecord, kind: str, rng: random.Random) -> Optional[dict]:
    """One perturbed text, or ``None`` when the rule does not apply to this text."""
    text = source.original_nl
    name = source.predicate_name
    args = list(source.predicate_args)
    if kind == "speaker_target_swap":
        ids = sorted({m.group(0) for m in PLAYER_RE.finditer(text)})
        if len(ids) < 2:
            return None
        a, b = ids[0], ids[1]
        swapped = re.sub(rf"\b({a}|{b})\b", lambda m: b if m.group(1) == a else a, text)
        return {"text": swapped, "rule": f"swap {a}<->{b}"}
    if kind == "role_flip":
        out = _first_rule(text, ROLE_FLIPS)
        return {"text": out[0], "rule": out[1]} if out else None
    if kind == "polarity_flip":
        out = _first_rule(text, POLARITY_RULES)
        return {"text": out[0], "rule": out[1]} if out else None
    if kind == "action_target_change":
        if name != "recommend_action" or len(args) < 5:
            return None
        target = args[4].strip()
        if not PLAYER_RE.fullmatch(target) or target not in text:
            return None
        mentioned = {m.group(0) for m in PLAYER_RE.finditer(text)} | {source.speaker_id}
        candidates = [f"P{i}" for i in range(1, 13) if f"P{i}" not in mentioned]
        if not candidates:
            return None
        new = rng.choice(candidates)
        return {"text": re.sub(rf"\b{target}\b", new, text), "rule": f"target {target}->{new}"}
    if kind == "reason_deletion":
        parts = REASON_SPLIT.split(text, maxsplit=1)
        if len(parts) < 2 or len(parts[0].split()) < 4:
            return None
        head = parts[0].rstrip(" ,;:—-")
        return {"text": head + ("." if not head.endswith((".", "?", "!")) else ""),
                "rule": "drop clause after first reason connective"}
    if kind == "reason_substitution":
        parts = REASON_SPLIT.split(text, maxsplit=1)
        if len(parts) < 2 or len(parts[0].split()) < 4:
            return None
        head = parts[0].rstrip(" ,;:—-")
        return {"text": f"{head} {rng.choice(REASON_BANK)}.", "rule": "replace reason clause"}
    if kind == "proposition_deletion":
        segs = _segments(text)
        if len(segs) < 2:
            return None
        drop = rng.randrange(1, len(segs))
        kept = [s for i, s in enumerate(segs) if i != drop]
        return {"text": " ".join(kept), "rule": f"drop segment {drop} of {len(segs)}"}
    if kind == "certainty_shift":
        out = _first_rule(text, CERTAINTY_RULES)
        return {"text": out[0], "rule": out[1]} if out else None
    if kind == "directive_force_shift":
        if name not in ("recommend_action", "question"):
            return None
        out = _first_rule(text, DIRECTIVE_RULES)
        return {"text": out[0], "rule": out[1]} if out else None
    raise ValueError(f"unknown challenge type {kind!r}")


def build_challenge_set(sources: Sequence[SourceRecord], *, seed: int = 101) -> List[dict]:
    """Perturbed negatives for every applicable rule, plus positive controls."""
    rng = random.Random(seed)
    items: List[dict] = []
    for src in sources:
        base = {"source_id": src.source_id, "game_id": src.game_id, "speech_id": src.speech_id,
                "predicate": src.predicate_text, "predicate_name": src.predicate_name,
                "phase": src.phase, "speaker": src.speaker_id, "original_text": src.original_nl,
                "referenced_speech": src.referenced_speech,
                "public_history": list(src.public_history)}
        items.append({**base, "item_id": f"pos_identity_{len(items):03d}",
                      "challenge_type": "positive_identity", "expected": "preserved",
                      "candidate_text": src.original_nl, "rule": "original vs original"})
        for kind in CHALLENGE_TYPES:
            out = perturb(src, kind, rng)
            if out is None or out["text"].strip() == src.original_nl.strip():
                continue
            items.append({**base, "item_id": f"neg_{kind}_{len(items):03d}",
                          "challenge_type": kind, "expected": "broken",
                          "candidate_text": out["text"], "rule": out["rule"]})
    for i, (kind, expected, predicate, original, candidate) in enumerate(CONTROLLED_CASES):
        src = source_record_from_dict({"source_id": f"constructed:{i}", "game_id": "constructed",
            "speech_id": "s2", "predicate": predicate, "speaker": "P3", "original_text": original,
            "referenced_speech": {"speech_id": "s1", "speaker": "P2", "text": "P3 is a wolf."}
                                 if kind == "positive_multiple_acts" else None,
            "public_history": ["[s1|P2] estimate(D1, P2, P3, werewolf): P3 is a wolf."]})
        items.append({**src.to_dict(), "item_id": f"constructed_{i:03d}", "predicate": predicate,
            "speaker": "P3", "original_text": original, "candidate_text": candidate,
            "challenge_type": kind, "expected": expected, "rule": kind,
            "construction": "controlled minimal pair"})
    return items


def run_challenge(items: Sequence[dict], gate, *, completed_rows=None, on_row=None) -> List[dict]:
    """Every judge on every item, independently (calls the judge models).

    Uses ``gate.judge_all`` rather than ``gate.evaluate``: the production gate
    short-circuits after a cheap failure and rejects a copy of the original,
    which would leave the parser and validator unscored on exactly the items a
    benchmark needs and would fail the identity positive controls by design.
    """
    allowed = set(CHALLENGE_TYPES) | {"positive_identity"} | {r[0] for r in CONTROLLED_CASES}
    if any(i.get("challenge_type") not in allowed for i in items):
        raise ValueError("unsupported synthetic challenge type")
    from experiments.proficiency_validation.rewrite.runtime import assert_unique
    assert_unique(items, ("item_id",))
    prior = list(completed_rows or [])
    assert_unique(prior, ("item_id",))
    done = {r["item_id"]: r for r in prior}
    if not set(done) <= {r["item_id"] for r in items}:
        raise ValueError("unexpected challenge checkpoint rows")
    out = []
    for item in items:
        if item["item_id"] in done:
            out.append(done[item["item_id"]])
            continue
        src = source_record_from_dict({**item, "original_nl": item["original_text"],
                                       "predicate_text": item["predicate"]})
        verdicts = gate.judge_all(src, item["candidate_text"])
        parser = verdicts.get("parser")
        semantic = verdicts.get("semantic")
        flags = {
            "structural": _flag(verdicts["structural"]["ok"]),
            "parser": (_flag(parser["structural_match"]) if parser and parser["ok"] else None),
            "validator": _flag(semantic["ok"]) if semantic else None,
        }
        applicable = [j for j in ("structural", "parser", "validator")
                      if (j != "parser" or parser is not None) and (j != "validator" or semantic is not None)]
        known = [flags[j] for j in applicable if flags[j] is not None]
        flags["pipeline"] = (True if any(known)
                             else (False if len(known) == len(applicable) else None))
        if (not flags["pipeline"] and verdicts["structural"]["details"].get("requires_semantic_adjudication")
                and (semantic or {}).get("ok") is not True):
            flags["pipeline"] = None
        if (flags["pipeline"] is False and (parser or {}).get("comparison_basis") in ("original_parse", "act_inventory")
                and (semantic or {}).get("ok") is not True):
            flags["pipeline"] = None
        out.append({**item, "judges": verdicts, "flags": flags, "judged_at": now_iso()})
        if on_row:
            on_row(out[-1])
    return out


def report_challenge(items: Sequence[dict]) -> dict:
    types = sorted({i["challenge_type"] for i in items})
    per_type: Dict[str, dict] = {}
    for kind in types:
        rows = [i for i in items if i["challenge_type"] == kind and "flags" in i]
        per_type[kind] = {
            "n": len(rows), "expected": rows[0]["expected"] if rows else None,
            **{judge: confusion([(i["expected"] == "broken", i["flags"].get(judge)) for i in rows])
               for judge in JUDGES}}
    overall = {judge: confusion([(i["expected"] == "broken", i["flags"].get(judge))
                                 for i in items if "flags" in i]) for judge in JUDGES}
    return {"n_items": len(items), "per_type": per_type, "overall": overall}


def render_challenge(report: dict) -> str:
    lines = ["# Synthetic semantic challenge set", "",
             f"Items judged: {report['n_items']}.", "",
             "Scores describe the constructed cases. Fixed minimal pairs are development regressions, "
             "including examples represented in the rubric. They do not provide held-out accuracy on natural rewrites.", "",
             "| challenge type | expected | n | structural flags | parser flags | validator flags | pipeline flags |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind, row in report["per_type"].items():
        def hits(judge):
            c = row[judge]
            return f"{c['tp'] + c['fp']}/{c['n'] - c['no_verdict']}"
        lines.append(f"| {kind} | {row['expected']} | {row['n']} | {hits('structural')} | "
                     f"{hits('parser')} | {hits('validator')} | {hits('pipeline')} |")
    lines += ["", "Sensitivity and specificity below condition on a valid verdict. Coverage and all-item rates include missing judgments in their denominators.", "",
              "| judge | sensitivity, valid verdicts | specificity, valid verdicts | valid fraction | broken flagged, all items | preserved passed, all items |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for judge, c in report["overall"].items():
        lines.append(f"| {judge} | {fmt(c['sensitivity'])} "
                     f"[{fmt(c['sensitivity_ci95'][0], 2)}, {fmt(c['sensitivity_ci95'][1], 2)}] | "
                     f"{fmt(c['specificity'])} "
                     f"[{fmt(c['specificity_ci95'][0], 2)}, {fmt(c['specificity_ci95'][1], 2)}] | "
                     f"{fmt(c['valid_fraction'])} | {fmt(c['broken_detection_all'])} | {fmt(c['preserved_acceptance_all'])} |")
    return "\n".join(lines) + "\n"


def cmd_build_challenge(args) -> int:
    sources = generate.load_source_records(args.sources)
    items = build_challenge_set(sources, seed=args.seed)
    write_jsonl(args.out, items)
    counts = {}
    for item in items:
        counts[item["challenge_type"]] = counts.get(item["challenge_type"], 0) + 1
    print(f"{len(items)} challenge items -> {args.out}")
    for kind, n in sorted(counts.items()):
        print(f"  {kind:28s} {n}")
    return 0


def cmd_run_challenge(args) -> int:
    from experiments.proficiency_validation.rewrite.gate import SemanticGate
    items = read_jsonl(args.items)
    if args.limit:
        items = items[:args.limit]
    gate = SemanticGate.from_handle(args.judge, max_tokens=args.judge_max_tokens)
    ok, msg = gate.judge_model.preflight()
    if not ok:
        print(f"judge preflight failed: {msg}", file=sys.stderr)
        return 1
    from experiments.proficiency_validation.rewrite.runtime import Checkpoint
    from experiments.proficiency_validation.validation.protocol import frozen_hashes
    from experiments.proficiency_validation.rewrite.runtime import artifact_provenance, record_artifact
    provenance = artifact_provenance(None, "pilot", inputs={"challenge_items": args.items})
    settings = {**provenance, "hashes": frozen_hashes(), "judge": gate.judge_model.identity_record,
                "judge_max_tokens": args.judge_max_tokens, "limit": args.limit}
    checkpoint = Checkpoint(args.out, settings, resume=args.resume)
    try:
        judged = run_challenge(items, gate, completed_rows=checkpoint.rows, on_row=checkpoint.append)
        checkpoint.finish()
    finally:
        checkpoint.close()
    record_artifact(args.out, {**provenance, "created_at": now_iso(), "gate": gate.provenance(),
                               "n_items": len(judged), "code_hashes": settings["hashes"]})
    print(f"{len(judged)} judged items -> {args.out}")
    return 0


def cmd_report_challenge(args) -> int:
    items = read_jsonl(args.items)
    report = report_challenge(items)
    text = render_challenge(report)
    if args.out:
        dump_json(args.out, report)
        Path(args.out).with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"wrote {args.out} and {Path(args.out).with_suffix('.md')}")
    print(text)
    return 0
