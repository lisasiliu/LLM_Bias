"""Relabel the 48 human-annotated rewrites with the marker-guided Luna prompt.

Three passes per item (majority = primary label); reference = rater_1's strategy,
which equals the requested strategy on all 48. Writes the passes to
data/classifier_passes_marker_guided.jsonl and prints agreement against the
baseline prompt (`act_then_strategy`, stored on each record).

    python -m experiments.politeness_validation.code.relabel_markers
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from components.judges.blind_politeness_classifier import (
    BlindPolitenessClassifier, STRATEGY_LABELS)
from components.judges.judge_model import JudgeModel

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL, PROMPT, PASSES = "gpt-5.6-luna", "marker_guided", 3


def main() -> int:
    items = [json.loads(l) for l in open(DATA / "original_48" / "manual_validation.jsonl")]
    clf = BlindPolitenessClassifier(JudgeModel.resolve(MODEL), max_tokens=1200,
                                    prompt_version=PROMPT)
    jobs = [(it, p) for it in items for p in range(1, PASSES + 1)]

    def run(job):
        it, p = job
        return it, p, clf.classify(it["rewritten_text"], phase=it.get("phase") or "",
                                   speaker_id=it.get("speaker") or "")

    out = DATA / "classifier_passes_marker_guided.jsonl"
    with ThreadPoolExecutor(8) as pool, open(out, "w") as f:
        results = list(pool.map(run, jobs))
        for it, p, r in results:
            f.write(json.dumps({"item_id": it["item_id"], "pass": p, "ok": r.ok,
                                "requested": it["requested_strategy"], "label": r.label,
                                "ranking": r.ranking, "markers": r.markers,
                                "speech_act": r.speech_act, "confidence": r.confidence,
                                "evidence": r.evidence, "notes": r.notes, "error": r.error,
                                "model": MODEL, "prompt": PROMPT}) + "\n")

    votes = {}
    for it, p, r in results:
        votes.setdefault(it["item_id"], []).append(r)
    rows = []
    for it in items:
        rs = [r for r in votes[it["item_id"]] if r.ok]
        c = Counter(r.label for r in rs).most_common()
        top = c[0][0] if c and (len(c) == 1 or c[0][1] > c[1][1]) else None
        rows.append((it, top, rs))
    strategies = list(STRATEGY_LABELS)
    n = len(rows)
    new = sum(1 for it, t, _ in rows if t == it["requested_strategy"])
    old = sum(1 for it, _, _ in rows
              if it["automatic_politeness_strategy"] == it["requested_strategy"])
    single = [r.label == it["requested_strategy"] for it, _, rs in rows for r in rs]
    print(f"3-pass majority: marker_guided {new}/{n}   baseline {old}/{n}")
    print(f"single pass:     marker_guided {sum(single)}/{len(single)}")
    print("per requested strategy (marker_guided / baseline):")
    for s in strategies:
        sub = [(it, t) for it, t, _ in rows if it["requested_strategy"] == s]
        print(f"  {s:20s} {sum(t == s for _, t in sub)}/{len(sub)}  /  "
              f"{sum(it['automatic_politeness_strategy'] == s for it, _ in sub)}/{len(sub)}")
    print("confusion (rows=human, cols=marker_guided):")
    cols = strategies + ["ambiguous_or_mixed", "failed_manipulation", "none"]
    for s in strategies:
        cnt = Counter((t or "none") for it, t, _ in rows if it["requested_strategy"] == s)
        print(f"  {s:20s}", {c: cnt[c] for c in cols if cnt[c]})
    second = [(it, t, rs) for it, t, rs in rows
              if rs and rs[0].ranking and len(rs[0].ranking) > 1]
    hit = sum(1 for it, _, rs in rows
              if any(it["requested_strategy"] in r.ranking for r in rs[:1]))
    multi = sum(1 for _, _, rs in rows if rs and len(rs[0].ranking) > 1)
    print(f"human label anywhere in pass-1 ranking: {hit}/{n}; multi-strategy rankings: {multi}/{n}")
    print("errors (majority != human):")
    for it, t, rs in rows:
        if t != it["requested_strategy"]:
            print(f"  {it['item_id']} human={it['requested_strategy']} new={t} "
                  f"old={it['automatic_politeness_strategy']} rank={[r.ranking for r in rs]}")
    print("failed calls:", sum(1 for _, _, r in results if not r.ok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
