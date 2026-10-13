"""Did the replays' delivered speech realize the strategy it was asked for?

    python -m experiments.politeness_flip.code.manipulation_check run [--classifier HANDLE]
    python -m experiments.politeness_flip.code.manipulation_check report

`run` samples the utterances the other players actually heard, strata balanced over
strategy and game, and labels each one with the blind politeness classifier of
`experiments/politeness_validation/` (three independent passes, majority vote). The
classifier is never told which strategy was requested. Every pass is appended to
`data/manipulation_passes.jsonl` and nothing is overwritten, so a second classifier
adds its own rows and `report` reads them side by side.

`report` makes no model calls: it takes each item's majority label and writes the
agreement with the requested strategy and the confusion between them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from components.helpers import read_jsonl
from components.judges.blind_politeness_classifier import (
    LABELS, PROMPT_VERSION, BlindPolitenessClassifier, system_prompt)
from components.judges.judge_model import FAILED, NO_MAJORITY, majority
from experiments.politeness_flip.code import analyze as pa
from experiments.language_flip.code import analyze as flip
from experiments.language_flip.code.analyze import _logged_text
from experiments.language_validation.rewrite.runtime import ValidationModel, fmt, now_iso

DATA = pa.RUN_DIR
PASSES = DATA / "manipulation_passes.jsonl"
REFERENCE_PASSES = DATA / "classifier_reference_passes.jsonl"
REPORT_JSON = DATA / "manipulation_check.json"
REPORT_MD = pa.EXPERIMENT_DIR / "manipulation_check.md"
N_PASSES = 3
PER_CONDITION = 250
SEED = 101


def delivered_utterances(games_dir: Path = pa.GAMES_DIR) -> List[dict]:
    """Every discussion utterance the other players heard in a treated replay."""
    out: List[dict] = []
    for root in sorted(games_dir.glob(f"run_*/{pa.COMPARISON}")):
        seed = root.parent.name.rsplit("_", 1)[-1]
        for run_dir in sorted(root.glob("replays/day_*/*")):
            meta = pa.flip.parse_run_id(run_dir.name, conditions=pa.CONDITIONS)
            log = run_dir / "rewrites.jsonl"
            if meta is None or meta["condition"] not in pa.CONDITIONS or not log.exists():
                continue
            result_path = run_dir / "phase_result.json"
            result = (json.loads(result_path.read_text())
                      if result_path.exists() and result_path.stat().st_size else None)
            rows = read_jsonl(log)
            start = flip.exposure_summary(rows, result, meta)["interrupted_rows"]
            for i, row in enumerate(rows[start:], start):
                if row.get("phase", "day_speech") != "day_speech":
                    continue
                category, text = _logged_text(row)
                if category != "rewritten" or not text:
                    continue
                out.append({"item_id": f"{seed}:{run_dir.name}:{i}", "game_id": seed, "day": meta["day"],
                            "mode": meta["mode"], "self_player": meta["self"], "speaker": row.get("speaker"),
                            "requested": meta["condition"], "predicate": row.get("predicate"),
                            "phase": row.get("phase", "day_speech"), "text": text,
                            "original_text": row.get("original_text") or ""})
    return out


SCOPES = ("flip_self", "flip_others")


def sample(utterances: List[dict], per_condition: int = PER_CONDITION, seed: int = SEED) -> List[dict]:
    """``per_condition`` utterances per strategy, split evenly between the two scopes and
    spread over the games. `flip_others` produces about five times as many utterances as
    `flip_self`, and the treated speakers of `flip_self` carry the primary analysis, so the
    scopes are balanced rather than sampled in proportion."""
    rng = random.Random(seed)
    chosen: List[dict] = []
    for condition in pa.CONDITIONS:
        for i, scope in enumerate(SCOPES):
            quota = per_condition // len(SCOPES) + (per_condition % len(SCOPES) if i == 0 else 0)
            by_game: Dict[str, List[dict]] = defaultdict(list)
            for u in utterances:
                if u["requested"] == condition and u["mode"] == scope:
                    by_game[u["game_id"]].append(u)
            for rows in by_game.values():
                rng.shuffle(rows)
            games = sorted(by_game)
            picked: List[dict] = []
            while games and len(picked) < quota:
                for game in list(games):
                    if not by_game[game]:
                        games.remove(game)
                        continue
                    picked.append(by_game[game].pop())
                    if len(picked) >= quota:
                        break
            chosen.extend(picked)
    return chosen


REFERENCE_SETS = {
    # Lisa's two sets with known answers: the classifier's own accuracy is measured here, not the rewriter's.
    "werewolf_rewrites": ("experiments/politeness_validation/data/manual_validation.jsonl",
                          "rewritten_text", "requested_strategy"),
    "zhao_hawkins": ("experiments/politeness_validation/data/external_validation.jsonl",
                     "text", "mapped_strategy"),
}


def reference_items() -> List[dict]:
    """The validated reference items, in the shape the classifier run expects."""
    root = Path(__file__).resolve().parents[3]
    items = []
    for dataset, (path, text_field, label_field) in REFERENCE_SETS.items():
        for row in read_jsonl(root / path):
            label = (row.get(label_field) or "").strip().lower()
            if not row.get(text_field) or label not in pa.CONDITIONS:
                continue
            items.append({"item_id": f"{dataset}:{row['item_id']}", "game_id": dataset, "day": 0,
                          "mode": dataset, "self_player": None, "speaker": row.get("speaker", ""),
                          "requested": label, "predicate": row.get("predicate"),
                          "phase": row.get("phase", ""), "text": row[text_field], "original_text": ""})
    return items


def cmd_run(args) -> int:
    # Handles named in PROFICIENCY_MODEL_CONFIG resolve to the served model; anything else
    # falls through to the judge registry, so a hosted classifier works the same way.
    model = ValidationModel.resolve(args.classifier, purpose="politeness_classifier")
    ok, msg = model.preflight()
    if not ok:
        print(f"preflight failed: {msg}")
        return 1
    clf = BlindPolitenessClassifier(model)
    sha = hashlib.sha256(system_prompt(PROMPT_VERSION).encode()).hexdigest()
    if args.reference:
        items = reference_items()
        print(f"reference sets: {Counter(i['game_id'] for i in items)}")
    else:
        items = sample(delivered_utterances(args.games), args.per_condition, args.seed)
    by_id = {it["item_id"]: it for it in items}
    out_path = REFERENCE_PASSES if args.reference else PASSES
    stored = read_jsonl(out_path) if out_path.exists() else []
    done = {(r["item_id"], r["pass"], r["classifier"]) for r in stored
            if r.get("ok") or not args.retry_failed}
    todo = [(it, p) for it in items for p in range(1, N_PASSES + 1)
            if (it["item_id"], p, args.classifier) not in done]
    print(f"{len(items)} utterances, {len(todo)} calls to make ({len(done)} passes already stored)")

    def one(job):
        it, p = job
        res = clf.classify(it["text"], phase=it["phase"], speaker_id=it["speaker"])
        return {"item_id": it["item_id"], "pass": p, "classifier": args.classifier,
                "requested": it["requested"], "game_id": it["game_id"], "mode": it["mode"],
                "label": res.label, "ok": res.ok, "error": res.error, "confidence": res.confidence,
                "speech_act": res.speech_act, "act_left_to_inference": res.act_left_to_inference,
                "inconsistent": res.inconsistent, "evidence": res.evidence, "notes": res.notes,
                "model_served": res.provenance.get("model_served", ""),
                "prompt": PROMPT_VERSION, "prompt_sha256": sha, "created_at": now_iso()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a", encoding="utf-8") as out:
        for i, row in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if i % 100 == 0:
                out.flush()
                print(f"  {i}/{len(todo)}")
    print(f"{written} passes appended -> {out_path}")
    if not args.reference:
        (DATA / "manipulation_items.jsonl").write_text(
            "".join(json.dumps(by_id[i], ensure_ascii=False) + "\n" for i in sorted(by_id)))
    return 0


#: outcomes of the majority vote that are not one of the classifier's labels
VOTE_OUTCOMES = (NO_MAJORITY, FAILED)
COLUMNS = LABELS + VOTE_OUTCOMES


def tally(rows: List[dict], classifier: Optional[str] = None) -> dict:
    """Majority label per item, then agreement and confusion against the requested strategy."""
    passes: Dict[tuple, List[Optional[str]]] = defaultdict(list)
    meta: Dict[tuple, dict] = {}
    failed_passes: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        if classifier and r["classifier"] != classifier:
            continue
        key = (r["classifier"], r["item_id"])
        passes[key].append(r["label"] if r.get("ok") else None)
        meta[key] = {"requested": r["requested"], "game": r["game_id"], "mode": r.get("mode", "")}
        counts = failed_passes[r["classifier"]]
        counts[0] += int(not r.get("ok"))
        counts[1] += 1
    per_classifier: Dict[str, dict] = {}
    for (handle, item_id), labels in passes.items():
        label, _ = majority(labels, passes=N_PASSES)
        block = per_classifier.setdefault(handle, {"items": 0, "confusion": defaultdict(Counter),
                                                   "by_game": defaultdict(lambda: [0, 0]),
                                                   "by_scope": defaultdict(lambda: [0, 0])})
        want = meta[(handle, item_id)]["requested"]
        block["items"] += 1
        block["confusion"][want][label] += 1
        for field, key in (("by_game", meta[(handle, item_id)]["game"]),
                           ("by_scope", meta[(handle, item_id)]["mode"])):
            block[field][key][0] += int(label == want)
            block[field][key][1] += 1
    out = {}
    for handle, block in per_classifier.items():
        confusion = {want: dict(counts) for want, counts in block["confusion"].items()}
        agree = {want: counts.get(want, 0) / sum(counts.values()) for want, counts in confusion.items()
                 if sum(counts.values())}
        per_game = {g: hit / n for g, (hit, n) in block["by_game"].items() if n}
        overall = sum(c.get(w, 0) for w, c in confusion.items()) / max(1, block["items"])
        failed, total = failed_passes[handle]
        decided = {want: {l: n for l, n in counts.items() if l not in VOTE_OUTCOMES}
                   for want, counts in confusion.items()}
        agree_decided = {want: counts.get(want, 0) / sum(counts.values())
                         for want, counts in decided.items() if sum(counts.values())}
        out[handle] = {"n_items": block["items"], "agreement": agree,
                       "agreement_of_decided": agree_decided, "overall_agreement": overall,
                       "confusion": confusion, "n_games": len(per_game),
                       "by_scope": {m: {"n_items": n, "agreement": hit / n}
                                    for m, (hit, n) in sorted(block["by_scope"].items()) if n},
                       "n_passes": total, "failed_passes": failed,
                       "failed_pass_rate": failed / total if total else None,
                       "game_agreement_min": min(per_game.values()) if per_game else None,
                       "game_agreement_max": max(per_game.values()) if per_game else None}
    return out


def render(summary: dict) -> str:
    reference = summary.get("source", "").startswith("reference")
    lines = (["# How well the classifier reads the four strategies", "",
              "The items below come from the two sets with known answers in "
              "`experiments/politeness_validation/`: the Werewolf rewrites two researchers labelled by hand, "
              "and the held-out Zhao & Hawkins items with published labels. Agreement here measures the "
              "classifier, not the rewriter.", ""]
             if reference else
             ["# Politeness manipulation check on the replays", "",
              "Every utterance below is one the other players heard in a treated replay. The blind "
              "politeness classifier of `experiments/politeness_validation/` labelled it without being "
              "told which strategy was requested, three independent passes per utterance, majority vote. "
              "The sample is balanced over the four strategies, over the two rewrite scopes and over the games.", ""])
    for handle, block in sorted(summary["classifiers"].items()):
        lines += [f"## Classifier `{handle}`", "",
                  (f"{block['n_items']} items from {block['n_games']} reference sets; the known answer "
                   f"was the majority label for {fmt(block['overall_agreement'], 3)} of them."
                   if reference else
                   f"{block['n_items']} utterances over {block['n_games']} games; the requested strategy "
                   f"was the majority label for {fmt(block['overall_agreement'], 3)} of them "
                   f"(per game {fmt(block['game_agreement_min'], 2)} to {fmt(block['game_agreement_max'], 2)})."), "",
                  "| requested | " + " | ".join(l.replace("_", " ") for l in COLUMNS) + " | agreement |",
                  "|---|" + "--:|" * (len(COLUMNS) + 1)]
        for want in pa.CONDITIONS:
            counts = block["confusion"].get(want, {})
            row = " | ".join(str(counts.get(l, 0)) for l in COLUMNS)
            lines.append(f"| {want.replace('_', ' ')} | {row} | {fmt(block['agreement'].get(want), 3)} |")
        lines += ["", f"| {'reference set' if reference else 'rewrite scope'} | items | agreement |", "|---|--:|--:|"]
        for mode, stat in block.get("by_scope", {}).items():
            lines.append(f"| {mode.replace('_', ' ')} | {stat['n_items']} | {fmt(stat['agreement'], 3)} |")
        lines += ["", f"{block['failed_passes']} of {block['n_passes']} passes did not return a usable label "
                  f"({fmt(block['failed_pass_rate'], 3)}).", ""]
    lines += ["Rows are the strategy the rewriter was asked for, columns the label the classifier gave. "
              "`ambiguous_or_mixed` and `failed_manipulation` are the classifier's own categories for an "
              "utterance that realizes no single strategy; `no_majority` and `failed` are votes where the "
              "three passes did not agree or returned nothing.", "",
              f"Generated {summary['created_at']} from "
              f"`{(REFERENCE_PASSES if reference else PASSES).relative_to(pa.EXPERIMENT_DIR)}`.", ""]
    return "\n".join(lines)


def cmd_report(args) -> int:
    path = REFERENCE_PASSES if args.reference else PASSES
    rows = read_jsonl(path) if path.exists() else []
    if not rows:
        print(f"no passes in {path}; run the check first")
        return 1
    summary = {"created_at": now_iso(), "n_passes": N_PASSES, "prompt": PROMPT_VERSION,
               "source": "reference sets with known answers" if args.reference else "replay speech",
               "classifiers": tally(rows)}
    out_json = DATA / ("classifier_reference.json" if args.reference else "manipulation_check.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=1))
    # The tables go into the experiment's one summary; `analyze` reads this file back.
    for handle, block in sorted(summary["classifiers"].items()):
        print(f"{handle}: {block['n_items']} utterances, agreement {block['overall_agreement']:.3f}")
    print(f"-> {out_json}; run `analyze` to put the tables in summary.md")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="label a balanced sample of the delivered utterances")
    r.add_argument("--games", type=Path, default=pa.GAMES_DIR)
    r.add_argument("--classifier", default="judge", help="JudgeModel handle, e.g. the served game model")
    r.add_argument("--per-condition", type=int, default=PER_CONDITION)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--reference", action="store_true",
                   help="label the validated reference sets instead of the replay speech, which measures "
                        "the classifier's own accuracy against known answers")
    r.add_argument("--retry-failed", action="store_true",
                   help="ask again for passes whose earlier call returned no usable label")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report", help="majority vote, agreement and confusion; no model calls")
    p.add_argument("--reference", action="store_true", help="report the reference-set passes instead")
    p.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
