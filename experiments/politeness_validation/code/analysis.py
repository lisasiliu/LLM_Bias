"""Recompute every number in the README from data/.

    python -m experiments.politeness_validation.code.analysis      # prints markdown; writes results.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
STRATEGIES = ["bald_on_record", "positive_politeness", "negative_politeness", "off_record"]
RATERS = ["lisa", "rq"]


def kappa(a: list, b: list) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    chance = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (observed - chance) / (1 - chance) if chance < 1 else 1.0


def confusion(rows: list, truth: list, pred: list, labels: list) -> list:
    counts = Counter(zip(truth, pred))
    cols = labels + sorted({p for _, p in counts if p not in labels})
    out = ["| human \\ judge | " + " | ".join(cols) + " |", "|---|" + "--:|" * len(cols)]
    for t in labels:
        out.append(f"| {t} | " + " | ".join(str(counts[(t, c)]) for c in cols) + " |")
    return out


def main() -> None:
    items = [json.loads(l) for l in open(DATA / "items.jsonl", encoding="utf-8")]
    n = len(items)
    out = ["# Results", ""]

    out += [f"## Rewrites rated by both researchers: {n}", "",
            "| requested strategy | n | Researcher 1 recognised | Researcher 2 recognised | judge recognised |", "|---|--:|--:|--:|--:|"]
    for s in STRATEGIES:
        rows = [i for i in items if i["requested_strategy"] == s]
        hits = [sum(i["labels"][r]["strategy"] == s for i in rows) for r in RATERS]
        out.append(f"| {s} | {len(rows)} | {hits[0]} | {hits[1]} | {sum(i['judge_label'] == s for i in rows)} |")
    totals = [sum(i["labels"][r]["strategy"] == i["requested_strategy"] for i in items) for r in RATERS]
    out.append(f"| **all** | {n} | {totals[0]} ({totals[0] / n:.0%}) | {totals[1]} ({totals[1] / n:.0%}) | "
               f"{sum(i['judge_label'] == i['requested_strategy'] for i in items)} |")
    out += ["", f"Items labelled by a researcher as `ambiguous`: {sum(i['labels'][r]['strategy'] == 'ambiguous' for i in items for r in RATERS)} labels.", ""]

    a = [i["labels"]["lisa"]["strategy"] for i in items]
    b = [i["labels"]["rq"]["strategy"] for i in items]
    both = [i for i in items if i["labels"]["lisa"]["strategy"] == i["labels"]["rq"]["strategy"]]
    out += ["## Agreement between the two researchers", "",
            f"Strategy: {len(both)}/{n} identical ({len(both) / n:.0%}), Cohen's kappa {kappa(a, b):.2f}.", ""]
    out += ["Disagreements (requested strategy; Researcher 1 / Researcher 2 / judge):", ""]
    for i in items:
        la, lb = i["labels"]["lisa"]["strategy"], i["labels"]["rq"]["strategy"]
        if la != lb:
            out.append(f"- {i['item_id']}: {i['requested_strategy']}; {la} / {lb} / {i['judge_label']}")
    ma = [i["labels"]["lisa"]["meaning"] for i in items]
    mb = [i["labels"]["rq"]["meaning"] for i in items]
    out += ["", "## Meaning preserved", "", "| | yes | no | unsure |", "|---|--:|--:|--:|"]
    for name, m in (("Researcher 1", ma), ("Researcher 2", mb)):
        out.append(f"| {name} | {m.count('yes')} | {m.count('no')} | {m.count('unsure')} |")
    agree = sum(x == y for x, y in zip(ma, mb))
    out += ["", f"Same meaning label: {agree}/{n} ({agree / n:.0%}), kappa {kappa(ma, mb):.2f}. "
            f"Both say yes: {sum(x == y == 'yes' for x, y in zip(ma, mb))}; both say no: {sum(x == y == 'no' for x, y in zip(ma, mb))}; "
            f"either says no: {sum('no' in (x, y) for x, y in zip(ma, mb))}.", "",
            "Items either researcher marked as changing the meaning, by predicate and requested strategy:", ""]
    lost = Counter((i["predicate"], i["requested_strategy"]) for i in items if "no" in (i["labels"]["lisa"]["meaning"], i["labels"]["rq"]["meaning"]))
    out += [f"- {p} / {s}: {c}" for (p, s), c in sorted(lost.items())]

    out += ["", "## Judge against the researchers", ""]
    for r, name in zip(RATERS, ("Researcher 1", "Researcher 2")):
        k = sum(i["judge_label"] == i["labels"][r]["strategy"] for i in items)
        out.append(f"- {name}: {k}/{n} ({k / n:.0%}), kappa {kappa([i['labels'][r]['strategy'] for i in items], [i['judge_label'] for i in items]):.2f}")
    cons = [i for i in both]
    k = sum(i["judge_label"] == i["labels"]["lisa"]["strategy"] for i in cons)
    out += [f"- items where the researchers agree: {k}/{len(cons)} ({k / len(cons):.0%})", "",
            "On the items where the researchers agree (rows = their label):", ""]
    out += confusion(cons, [i["labels"]["lisa"]["strategy"] for i in cons], [i["judge_label"] for i in cons], STRATEGIES)

    z = [json.loads(l) for l in open(DATA / "zhao-hawkins_judge.jsonl", encoding="utf-8")]
    hit = sum(x["mapped_strategy"] == x["automatic_strategy"] for x in z)
    out += ["", f"## Judge on Zhao & Hawkins (2025): {len(z)} items", "",
            f"Four-way agreement {hit}/{len(z)} ({hit / len(z):.1%}), kappa {kappa([x['mapped_strategy'] for x in z], [x['automatic_strategy'] for x in z]):.2f}.", ""]
    out += confusion(z, [x["mapped_strategy"] for x in z], [x["automatic_strategy"] for x in z], STRATEGIES)
    pn = [x for x in z if x["mapped_strategy"] in STRATEGIES[1:3]]
    placed = [x for x in pn if x["automatic_strategy"] in STRATEGIES[1:3]]
    same = sum(x["mapped_strategy"] == x["automatic_strategy"] for x in placed)
    out += ["", f"Of {len(pn)} positive or negative items the judge placed {len(placed)} in that pair, and {same} of those match the published label."]

    text = "\n".join(out) + "\n"
    (HERE / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
