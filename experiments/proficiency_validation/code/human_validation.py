"""Optional blind English worksheets and supplementary gate calibration.

Worksheets collect externally supplied labels for supplementary analyses.
Unknown or missing labels remain unresolved. Calibration selects at most one
candidate per game to use games as the independent sampling units for binomial
confidence intervals.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

from components.helpers import read_jsonl, write_jsonl
from components.stats import wilson_interval, cohen_kappa
from .common import dump_json, sha256_text, sha256_file
from .english_validator import SEMANTIC_RUBRIC_VERSION
from .protocol import assert_unique, digest

FIELDS = ("meaning", "stance", "politeness_register", "grammatical", "natural", "understandable")
FIDELITY = FIELDS[:3]


def select_items(rewrites, purpose, n, seed):
    """Selection is fixed before human labels and never stratified on gate verdicts."""
    if n < 1:
        raise ValueError("sample size must be positive")
    items = candidate_items(rewrites, purpose == "calibration")
    rng = random.Random(seed)
    groups = defaultdict(list)
    for item in items:
        groups[item["game_id"] if purpose == "calibration" else item["condition"]].append(item)
    if purpose == "calibration":
        # Sample games uniformly, then one final candidate within each game.
        games = sorted(groups)
        rng.shuffle(games)
        selected = [rng.choice(groups[g]) for g in games[:n]]
    else:
        selected = []
        for condition in sorted(groups):
            rng.shuffle(groups[condition])
            selected.extend(groups[condition][:n])
    rng.shuffle(selected)
    return {f"item_{i:04d}": item for i, item in enumerate(selected, 1)}


def candidate_items(rewrites, calibration=False):
    out = []
    for r in rewrites:
        attempts = r.get("attempts", [])
        if not attempts:
            continue
        attempt = attempts[-1]
        text = r.get("rewritten_text") if r.get("status") == "ok" else attempt.get("text")
        gate = attempt.get("gate") or {}
        if not text or (not calibration and r.get("status") != "ok"):
            continue
        out.append({"source_id": r["source_id"], "game_id": r["game_id"], "condition": r["condition"],
                    "original": r["original_text"], "candidate": text, "gate_ok": gate.get("ok"),
                    "referenced_speech": r.get("referenced_speech"),
                    "gate_provenance": r.get("gate_provenance"), "status": r.get("status"),
                    "text_sha256": sha256_text(text)})
    return out


def worksheet(items):
    lines = ["# English fidelity and quality", "",
             "Compare each candidate with its original. The experimental condition and model are hidden.",
             "Use yes / no / unsure for every field. Do not assume that either text is grammatical.",
             "meaning: same claims, reasons, targets and reference; no added or deleted information.",
             "stance: same certainty, speaker/addressee, directness and speech-act force.",
             "politeness_register: same politeness, warmth, deference, aggression and formality; no added rapport.",
             "grammatical: the candidate has correct English grammar.",
             "natural: the candidate is natural spoken dialogue in this game.",
             "understandable: its intended message can be understood without guessing or repairing the text.", ""]
    for key, item in items.items():
        lines += [f"### {key}", "", "Original: " + item["original"], "",
                  "Candidate: " + item["candidate"], ""]
        if item.get("referenced_speech"):
            lines += ["Referenced public statement: " + item["referenced_speech"].get("text", ""), ""]
        lines += [f"{field}:" for field in FIELDS] + ["note:", ""]
    return "\n".join(lines)


def import_sheet(text, key, annotator):
    rows, current, answers = [], None, {}
    def finish():
        if current in key["items"]:
            row = {"item_id": current, "annotator": annotator, "key_sha256": digest(key)}
            row.update({f: answers.get(f, "missing").lower() for f in FIELDS})
            if any(row[f] not in ("yes", "no", "unsure", "missing") for f in FIELDS):
                raise ValueError(f"unrecognized answer at {current}")
            rows.append(row)
    for line in text.splitlines():
        match = re.match(r"^###\s+(item_\d+)\s*$", line)
        if match:
            finish()
            current, answers = match[1], {}
        elif ":" in line:
            k, v = line.split(":", 1)
            if k in FIELDS and v.strip():
                answers[k] = v.strip().strip("`")
    finish()
    return rows


def summarize(key, annotations, min_raters=2):
    assert_unique(annotations, ("item_id", "annotator"))
    by_item = defaultdict(list)
    for row in annotations:
        if row.get("key_sha256") != digest(key) or row["item_id"] not in key["items"]:
            raise ValueError("human annotation does not belong to this worksheet/key")
        by_item[row["item_id"]].append(row)
    results = []
    for item_id, item in key["items"].items():
        ratings = by_item[item_id]
        enough = len({r["annotator"] for r in ratings}) >= min_raters
        fidelity_labels = [all(r[f] == "yes" for f in FIDELITY) for r in ratings]
        negative_labels = [any(r[f] == "no" for f in FIDELITY) for r in ratings]
        gold = (True if enough and all(fidelity_labels) else
                False if enough and all(negative_labels) else None)
        results.append({**item, "item_id": item_id, "n_raters": len(ratings), "gold_fidelity": gold,
                        "quality_pass": enough and all(all(r[f] == "yes" for f in FIELDS) for r in ratings)})
    raters = sorted({r["annotator"] for r in annotations})
    agreement = {}
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            for field in FIELDS:
                pairs = []
                for ratings in by_item.values():
                    d = {r["annotator"]: r[field] for r in ratings}
                    if a in d and b in d:
                        pairs.append((d[a], d[b]))
                agreement[f"{a}/{b}/{field}"] = {"n": len(pairs), "kappa": cohen_kappa(pairs)}
    return {"items": results, "agreement": agreement}


def calibration_report(key, annotations, rules):
    from .protocol import number
    for k in ("min_raters", "min_broken", "min_preserved"):
        if not number(rules.get(k), 2, integer=True):
            raise ValueError(f"calibration rule {k} must be at least two")
    for k in ("sensitivity_lower_min", "specificity_lower_min"):
        if not (number(rules.get(k), 0, 1) and rules[k] > 0):
            raise ValueError(f"calibration rule {k} must be a probability greater than 0")
    if not (number(rules.get("max_unresolved_fraction"), 0, 1) and rules["max_unresolved_fraction"] < 1):
        raise ValueError("calibration rule max_unresolved_fraction must be a probability below 1")
    if key.get("purpose") != "calibration":
        raise ValueError("calibration needs a calibration worksheet, including accepted and rejected candidates")
    if len(str(rules.get("rationale") or "").strip()) < 20:
        raise ValueError("calibration thresholds and sample size need a substantive rationale")
    report = summarize(key, annotations, rules["min_raters"])
    items = report["items"]
    if len({r["game_id"] for r in items}) != len(items):
        raise ValueError("calibration CIs require one independent item per game")
    provenance = [r.get("gate_provenance") or {} for r in items]
    versions = {p.get("semantic_rubric_version") for p in provenance}
    identities = [p.get("judge_model", {}).get("identity") for p in provenance]
    if versions != {SEMANTIC_RUBRIC_VERSION} or not identities or any(i != identities[0] for i in identities) or not identities[0]:
        raise ValueError("calibration candidates must use one identified English judge/rubric")
    # Missing judge verdicts count against the instrument, not as omitted hard cases.
    negatives = [r for r in items if r["gold_fidelity"] is False]
    positives = [r for r in items if r["gold_fidelity"] is True]
    sens = wilson_interval(sum(r["gate_ok"] is False for r in negatives), len(negatives))
    spec = wilson_interval(sum(r["gate_ok"] is True for r in positives), len(positives))
    report.update(rubric_version=SEMANTIC_RUBRIC_VERSION, judge_identity=identities[0],
                  selection=key.get("selection"),
                  game_ids=sorted(r["game_id"] for r in items), rules=rules,
                  n_broken=len(negatives), n_preserved=len(positives),
                  n_unresolved=len(items)-len(negatives)-len(positives), sensitivity=sens, specificity=spec)
    report["passed"] = bool(len(negatives) >= rules["min_broken"] and len(positives) >= rules["min_preserved"]
        and sens[1] is not None and sens[1] >= rules["sensitivity_lower_min"]
        and spec[1] is not None and spec[1] >= rules["specificity_lower_min"]
        and report["n_unresolved"] / max(1, len(items)) <= rules["max_unresolved_fraction"])
    report["passed"] = report["passed"] and calibration_is_valid(report)
    return report


def calibration_is_valid(report):
    """Recompute acceptance from the archived item-level calibration evidence."""
    try:
        rules, items = report["rules"], report["items"]
        negatives = [r for r in items if r["gold_fidelity"] is False]
        positives = [r for r in items if r["gold_fidelity"] is True]
        sens = wilson_interval(sum(r["gate_ok"] is False for r in negatives), len(negatives))
        spec = wilson_interval(sum(r["gate_ok"] is True for r in positives), len(positives))
        return bool(report.get("passed") is True and report.get("selection") == "uniform games then one candidate"
            and len(str(rules.get("rationale") or "").strip()) >= 20
            and rules["min_raters"] >= 2 and all(r["n_raters"] >= rules["min_raters"] for r in items)
            and rules["min_broken"] >= 2 and rules["min_preserved"] >= 2
            and len(negatives) >= rules["min_broken"] and len(positives) >= rules["min_preserved"]
            and report["n_broken"] == len(negatives) and report["n_preserved"] == len(positives)
            and list(sens) == list(report["sensitivity"]) and list(spec) == list(report["specificity"])
            and 0 < rules["sensitivity_lower_min"] <= sens[1]
            and 0 < rules["specificity_lower_min"] <= spec[1]
            and (len(items)-len(negatives)-len(positives))/len(items) <= rules["max_unresolved_fraction"] < 1
            and len(set(report["game_ids"])) == len(items)
            and sorted(report["game_ids"]) == sorted(r["game_id"] for r in items))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def quality_criteria(key, annotations, frozen, rewrites):
    if not frozen.get("human_validation"):
        raise ValueError("optional human QC needs its own declared sampling settings; it is not an automated criterion")
    if key.get("purpose") != "quality":
        raise ValueError("confirmatory human QC requires a quality worksheet")
    expected = select_items(rewrites, "quality", frozen["human_validation"]["min_items_per_condition"],
                            frozen["statistics"]["seed"])
    if key.get("items") != expected:
        raise ValueError("human quality sample differs from the frozen selection rule")
    report = summarize(key, annotations, frozen["human_validation"]["min_raters"])
    actual = {(r["source_id"], r["condition"]): r for r in rewrites}
    for item in report["items"]:
        row = actual.get((item["source_id"], item["condition"]))
        if not row or sha256_text(row.get("rewritten_text") or "") != item["text_sha256"]:
            raise ValueError("human quality checks do not match the analyzed accepted rewrites")
    criteria = []
    rules = frozen["human_validation"]
    for c in frozen["conditions"]:
        if c == "original":
            continue
        rows = [r for r in report["items"] if r["condition"] == c]
        n, games = len(rows), len({r["game_id"] for r in rows})
        rate = sum(r["quality_pass"] for r in rows) / n if n else None
        criteria.append({"criterion": f"Human English fidelity/quality: {c}",
                         "passed": bool(n >= rules["min_items_per_condition"] and games >= rules["min_games_per_condition"]
                                        and rate is not None and rate >= rules["min_pass_rate"]),
                         "detail": {"n": n, "games": games, "pass_rate": rate}, "note": "unsure/missing/disagreement do not pass"})
    return criteria, report


def add_parser(sub):
    b = sub.add_parser("build-human-qc", help="optional English fidelity/quality worksheet")
    b.add_argument("--rewrites", type=Path, required=True)
    b.add_argument("--purpose", choices=["calibration", "quality"], required=True)
    b.add_argument("--n-per-stratum", type=int, default=None,
                   help="calibration: total games; quality: items per condition; pinned for frozen quality")
    b.add_argument("--seed", type=int, default=101)
    b.add_argument("--frozen", type=Path)
    b.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--key", type=Path, required=True)
    b.set_defaults(func=main_build)
    i = sub.add_parser("import-human-qc")
    i.add_argument("--worksheet", type=Path, required=True)
    i.add_argument("--key", type=Path, required=True)
    i.add_argument("--annotator", required=True)
    i.add_argument("--out", type=Path, required=True)
    i.set_defaults(func=main_import)
    c = sub.add_parser("calibrate-english-gate")
    c.add_argument("--key", type=Path, required=True)
    c.add_argument("--annotations", type=Path, nargs="+", required=True)
    c.add_argument("--rules", type=Path, required=True, help="predeclared sensitivity/specificity and sample requirements")
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=main_calibrate)


def main_build(args):
    if args.frozen:
        from .frozen import enforce_frozen
        from .protocol import verify_artifact
        frozen = enforce_frozen(args.frozen)
        if not frozen.get("human_validation"):
            raise ValueError("this automated freeze has no optional human sampling settings; use an unfrozen supplementary worksheet")
        if args.purpose != "quality":
            raise ValueError("calibration is completed before freeze")
        _, errors = verify_artifact(args.rewrites, frozen, args.split)
        if errors:
            raise ValueError(errors)
        args.n_per_stratum = frozen["human_validation"]["min_items_per_condition"]
        args.seed = frozen["statistics"]["seed"]
    if args.n_per_stratum is None:
        raise ValueError("--n-per-stratum is required outside frozen quality validation")
    selected = select_items(read_jsonl(args.rewrites), args.purpose, args.n_per_stratum, args.seed)
    key = {"purpose": args.purpose, "seed": args.seed, "rewrites_sha256": sha256_file(args.rewrites),
           "selection": "uniform games then one candidate" if args.purpose == "calibration" else "uniform candidates within condition",
           "items": selected}
    if not selected:
        raise ValueError("no eligible candidates")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(worksheet(key["items"]), encoding="utf-8")
    dump_json(args.key, key)
    print(f"{len(selected)} blind items; keep {args.key} away from raters")
    return 0


def main_import(args):
    key = json.loads(args.key.read_text())
    write_jsonl(args.out, import_sheet(args.worksheet.read_text(), key, args.annotator))
    return 0


def main_calibrate(args):
    key, rules = json.loads(args.key.read_text()), json.loads(args.rules.read_text())
    report = calibration_report(key, [r for p in args.annotations for r in read_jsonl(p)], rules)
    report["artifact_hashes"] = {"key": sha256_file(args.key), "rules": sha256_file(args.rules),
                                "annotations": [sha256_file(p) for p in args.annotations]}
    dump_json(args.out, report)
    print("calibration passed" if report["passed"] else "calibration NOT passed")
    return 0 if report["passed"] else 1
