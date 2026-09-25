"""Blind English fidelity/quality worksheets and held-out gate calibration.

Labels are never filled by the program. Unknown/missing human labels cannot
become a pass. Calibration selects at most one candidate per game so the
binomial confidence intervals have independent sampling units.
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
                  "Candidate: " + item["candidate"], ""] + [f"{field}:" for field in FIELDS] + ["note:", ""]
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
    if key.get("purpose") != "calibration":
        raise ValueError("calibration needs a calibration worksheet, including accepted and rejected candidates")
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
                  game_ids=sorted(r["game_id"] for r in items), rules=rules,
                  n_broken=len(negatives), n_preserved=len(positives),
                  n_unresolved=len(items)-len(negatives)-len(positives), sensitivity=sens, specificity=spec)
    report["passed"] = bool(len(negatives) >= rules["min_broken"] and len(positives) >= rules["min_preserved"]
        and sens[1] is not None and sens[1] >= rules["sensitivity_lower_min"]
        and spec[1] is not None and spec[1] >= rules["specificity_lower_min"]
        and report["n_unresolved"] / max(1, len(items)) <= rules["max_unresolved_fraction"])
    return report


def quality_criteria(key, annotations, frozen, rewrites):
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
    b = sub.add_parser("build-human-qc", help="blind English fidelity/quality worksheet")
    b.add_argument("--rewrites", type=Path, required=True)
    b.add_argument("--purpose", choices=["calibration", "quality"], required=True)
    b.add_argument("--n-per-stratum", type=int, required=True)
    b.add_argument("--seed", type=int, default=101)
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
    items = candidate_items(read_jsonl(args.rewrites), args.purpose == "calibration")
    rng = random.Random(args.seed)
    rng.shuffle(items)
    grouped = defaultdict(list)
    for item in items:
        grouped[(item["condition"], item["gate_ok"] if args.purpose == "calibration" else True)].append(item)
    selected, games = [], set()
    # Round robin avoids filling all games with the first condition/status.
    counts = defaultdict(int)
    while any(grouped.values()):
        for group in sorted(grouped, key=str):
            if not grouped[group]:
                continue
            item = grouped[group].pop()
            if counts[group] >= args.n_per_stratum or (args.purpose == "calibration" and item["game_id"] in games):
                continue
            selected.append(item)
            games.add(item["game_id"])
            counts[group] += 1
    rng.shuffle(selected)
    key = {"purpose": args.purpose, "seed": args.seed, "rewrites_sha256": sha256_file(args.rewrites),
           "items": {f"item_{i:04d}": r for i, r in enumerate(selected, 1)}}
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
