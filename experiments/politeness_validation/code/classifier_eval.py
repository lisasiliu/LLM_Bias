"""Run the politeness classifier (gpt-5.6-luna, 3-pass majority) on both reference sets.

    python -m experiments.politeness_validation.code.classifier_eval run
    python -m experiments.politeness_validation.code.classifier_eval apply

`run` calls the model: 3 independent passes per item, over the 48 Werewolf
rewrites and the 135 Zhao & Hawkins held-out items. Every pass is appended to
`data/classifier_passes.jsonl` (resumable; nothing is overwritten), so the
per-item evidence behind every reported number is kept.

`apply` makes no model calls. It takes the 3-pass majority of each item and
writes it into the `automatic_*` fields of `manual_validation.jsonl` and
`external_validation.jsonl`.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

from components.helpers import read_jsonl, write_jsonl
from components.judges.blind_politeness_classifier import (
    PROMPT_VERSION, BlindPolitenessClassifier, system_prompt)
from components.judges.judge_model import JudgeModel
from components.judges.majority_vote import majority

DATA = Path(__file__).resolve().parents[1] / "data"
MANUAL = DATA / "manual_validation.jsonl"
EXTERNAL = DATA / "external_validation.jsonl"
PASSES = DATA / "classifier_passes.jsonl"
MODEL = "gpt-5.6-luna"
N_PASSES = 3


def _items() -> List[dict]:
    items = []
    for r in read_jsonl(MANUAL):
        items.append({"dataset": "werewolf_rewrites", "item_id": r["item_id"],
                      "text": r["rewritten_text"], "phase": r.get("phase", ""),
                      "speaker": r.get("speaker", "")})
    for r in read_jsonl(EXTERNAL):
        items.append({"dataset": "zhao_hawkins", "item_id": r["item_id"],
                      "text": r["text"], "phase": "", "speaker": ""})
    return items


def cmd_run(args) -> int:
    import hashlib
    model = JudgeModel.resolve(MODEL)
    ok, msg = model.preflight()
    if not ok:
        print(f"preflight failed: {msg}")
        return 1
    clf = BlindPolitenessClassifier(model)
    sha = hashlib.sha256(system_prompt(PROMPT_VERSION).encode()).hexdigest()
    done = {(r["dataset"], r["item_id"], r["pass"])
            for r in (read_jsonl(PASSES) if PASSES.exists() else [])}
    todo = [(it, p) for it in _items() for p in range(1, N_PASSES + 1)
            if (it["dataset"], it["item_id"], p) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} calls to make ({len(done)} already stored)")

    def one(job):
        it, p = job
        res = clf.classify(it["text"], phase=it["phase"], speaker_id=it["speaker"])
        return {"dataset": it["dataset"], "item_id": it["item_id"], "pass": p,
                "label": res.label, "ok": res.ok, "error": res.error,
                "confidence": res.confidence, "speech_act": res.speech_act,
                "act_left_to_inference": res.act_left_to_inference,
                "evidence": res.evidence, "notes": res.notes,
                "model": MODEL, "model_served": res.provenance.get("model_served", ""),
                "prompt": PROMPT_VERSION, "prompt_sha256": sha}

    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            PASSES.open("a", encoding="utf-8") as out:
        for i, row in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
    print("done")
    return 0


def _votes() -> Dict[tuple, dict]:
    by: Dict[tuple, List[dict]] = {}
    for r in read_jsonl(PASSES):
        by.setdefault((r["dataset"], r["item_id"]), []).append(r)
    out = {}
    for key, rows in by.items():
        rows.sort(key=lambda r: r["pass"])
        label, n = majority([r["label"] for r in rows], passes=N_PASSES)
        first = next((r for r in rows if r["label"] == label), rows[0])
        out[key] = {"label": label, "votes": n, "passes": [r["label"] for r in rows],
                    "first": first, "model_served": rows[0]["model_served"]}
    return out


def cmd_apply(args) -> int:
    votes = _votes()
    manual = read_jsonl(MANUAL)
    for r in manual:
        v = votes[("werewolf_rewrites", r["item_id"])]
        r["automatic_politeness_strategy"] = v["label"]
        r["automatic_politeness_votes"] = v["passes"]
        r["automatic_politeness_confidence"] = v["first"]["confidence"]
        r["politeness_classifier_model"] = MODEL
        r["politeness_classifier_prompt"] = PROMPT_VERSION
    write_jsonl(MANUAL, manual)
    external = read_jsonl(EXTERNAL)
    for r in external:
        v = votes[("zhao_hawkins", r["item_id"])]
        f = v["first"]
        r["automatic_strategy"] = v["label"]
        r["automatic_votes"] = v["passes"]
        r["automatic_usable"] = v["label"] != "failed"
        r["automatic_identified_act"] = f["speech_act"]
        r["automatic_confidence"] = f["confidence"]
        r["automatic_notes"] = f["notes"]
        r["classifier_model"] = MODEL
        r["classifier_prompt"] = PROMPT_VERSION
        for stale in ("classifier_prompt_version", "classifier_model_served"):
            r.pop(stale, None)
    write_jsonl(EXTERNAL, external)
    print(f"updated {len(manual)} + {len(external)} records")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--limit", type=int, default=0, help="stop after N calls (smoke test)")
    sub.add_parser("apply")
    args = ap.parse_args(argv)
    return {"run": cmd_run, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
