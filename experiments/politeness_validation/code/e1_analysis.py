"""E1 / E2 analysis of the filled rating sheets.

    python -m experiments.politeness_validation.code.e1_analysis judge
    python -m experiments.politeness_validation.code.e1_analysis analyze \\
        --rater lisa sheet_lisa.md sheet_addon_lisa.md --rater runqi sheet_runqi.md sheet_addon_runqi.md

`judge` runs gpt-5.6-luna (3 passes, majority) on the 100 delivered rewrites, once, before any rating.
`analyze` reads the filled sheets (a rater's main and add-on file together) and the hidden key, and writes
data/e1_results.md and e1_results.json. Every number is recomputed from the sheets, the key and the stored
judge labels; nothing is typed in.

Definitions
- Human label for an item: the label both raters gave; otherwise "disputed" (listed, excluded, bounded).
- Recognised: the human label equals the requested strategy. Reported per rater and for both raters.
- Meaning kept: `meaning: yes`. Reported per rater and for both raters.
- Judge precision (E2): of the judge-non-bald naturals, the share whose human label is non-bald
  (positive, negative or off-record); the strict version also requires the same strategy as the judge.
- Judge miss rate (E2): of ALL judge-bald naturals (main + add-on), the share whose human label is non-bald.
- `ambiguous` as a consensus label is neither bald nor non-bald and is excluded from both rates (counted).
- Corrected non-bald share = w_nb * precision + w_b * miss rate, with w from the judge's labels on the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
E1 = ROOT / "experiments" / "politeness_validation" / "data"
MODEL = "gpt-5.6-luna"
STRATS = ("bald_on_record", "positive_politeness", "negative_politeness", "off_record")
ALIASES = {"bald": "bald_on_record", "positive": "positive_politeness", "negative": "negative_politeness",
           "off-record": "off_record", "offrecord": "off_record", "off_record": "off_record",
           "ambiguous": "ambiguous", "ambiguous_or_mixed": "ambiguous"}
W_NONBALD, W_BALD = 0.125, 0.875          # judge's label shares on the 1,429 corpus utterances


# ── statistics ──
def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided interval for a binomial proportion, by bisection on the binomial tails."""
    if n == 0:
        return (float("nan"), float("nan"))

    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - _binom_cdf(k - 1, n, p) < alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: _binom_cdf(k, n, p) > alpha / 2)
    return lower, upper


def frac(k: int, n: int) -> str:
    if n == 0:
        return "n/a (0)"
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}]"


def kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None
    n = len(pairs)
    labels = {x for p in pairs for x in p}
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == l for a, _ in pairs) / n) * (sum(b == l for _, b in pairs) / n) for l in labels)
    return None if pe == 1 else (po - pe) / (1 - pe)


# ── sheets ──
def norm_label(raw: str) -> str:
    v = re.sub(r"[`*]", "", raw).strip().lower()
    return ALIASES.get(v, v)


def parse_sheet(text: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    parts = re.split(r"^### (e1_\d+)\s*$", text, flags=re.M)
    for iid, body in zip(parts[1::2], parts[2::2]):
        fields = {}
        for name in ("strategy", "confidence", "meaning"):
            m = re.search(rf"^{name}:[ \t]*(.*)$", body, flags=re.M)
            fields[name] = m.group(1).strip() if m else ""
        fields["strategy"] = norm_label(fields["strategy"])
        fields["meaning"] = re.sub(r"[`*]", "", fields["meaning"]).strip().lower()
        out[iid] = fields
    return out


def load_rater(paths: Sequence[str]) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for p in paths:
        merged.update(parse_sheet(Path(p).read_text(encoding="utf-8")))
    return merged


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── judge on the 100 rewrites ──
def cmd_judge(_args) -> int:
    from concurrent.futures import ThreadPoolExecutor
    from components.judges.blind_politeness_classifier import BlindPolitenessClassifier
    from components.judges.judge_model import JudgeModel, majority
    key = json.loads((E1 / "key.json").read_text(encoding="utf-8"))
    rewrites = {(r["source_id"], r["strategy"]): r for r in read_jsonl(E1 / "rewrites.jsonl")}
    sources = {r["source_id"]: r for r in read_jsonl(E1 / "sources.jsonl")}
    model = JudgeModel.resolve(MODEL)
    ok, msg = model.preflight()
    if not ok:
        print("preflight failed:", msg)
        return 1
    clf = BlindPolitenessClassifier(model)
    jobs = []
    for iid, k in key.items():
        if k["source"] != "rewrite":
            continue
        r = rewrites[(k["source_id"], k["strategy"])]
        s = sources[k["source_id"]]
        jobs += [(iid, p, r["delivered_nl"], s["phase"], s["speaker_id"]) for p in (1, 2, 3)]

    def one(job):
        iid, p, text, phase, speaker = job
        res = clf.classify(text, phase=phase, speaker_id=speaker)
        return {"item": iid, "pass": p, "label": res.label, "ok": res.ok, "error": res.error,
                "model": MODEL}

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, jobs))
    by = defaultdict(list)
    for r in rows:
        by[r["item"]].append(r["label"] if r["ok"] else None)
    out = [{"item": i, "label": majority(v)[0], "votes": majority(v)[1], "passes": v} for i, v in by.items()]
    (E1 / "rewrite_judge.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    (E1 / "rewrite_judge_passes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(len(out), "items judged;", Counter(o["label"] for o in out))
    return 0


# ── analysis ──
def human_label(a: Dict[str, str], b: Dict[str, str]) -> str:
    la, lb = a.get("strategy", ""), b.get("strategy", "")
    return la if la and la == lb else "disputed"


def analyse(raters: Dict[str, Dict[str, Dict[str, str]]], key: dict, judge: Dict[str, str],
            *, w_nb: float = W_NONBALD, w_b: float = W_BALD, boot: int = 20000, seed: int = 7) -> dict:
    (n1, r1), (n2, r2) = list(raters.items())
    res: dict = {"raters": [n1, n2]}
    missing = [i for i in key if i not in r1 or not r1[i]["strategy"] or i not in r2 or not r2[i]["strategy"]]
    res["unrated_items"] = missing
    rated = [i for i in key if i not in missing]
    human = {i: human_label(r1[i], r2[i]) for i in rated}

    res["kappa_strategy"] = kappa([(r1[i]["strategy"], r2[i]["strategy"]) for i in rated])
    res["n_rated"] = len(rated)
    res["disputed"] = sorted(i for i in rated if human[i] == "disputed")

    rewrites = [i for i in rated if key[i]["source"] == "rewrite"]
    mp = [(r1[i]["meaning"], r2[i]["meaning"]) for i in rewrites if r1[i]["meaning"] and r2[i]["meaning"]]
    res["kappa_meaning"] = kappa(mp)

    # E1 per requested strategy
    e1 = {}
    for st in STRATS:
        items = [i for i in rewrites if key[i]["strategy"] == st]
        n = len(items)
        row = {"n": n}
        for nm, r in ((n1, r1), (n2, r2)):
            row[f"recognised_{nm}"] = frac(sum(r[i]["strategy"] == st for i in items), n)
            row[f"meaning_{nm}"] = frac(sum(r[i]["meaning"] == "yes" for i in items), n)
        row["recognised_both"] = frac(sum(human[i] == st for i in items), n)
        row["meaning_both"] = frac(sum(r1[i]["meaning"] == "yes" and r2[i]["meaning"] == "yes" for i in items), n)
        k_rec = [sum(r[i]["strategy"] == st for i in items) / n for r in (r1, r2)]
        k_mean = [sum(r[i]["meaning"] == "yes" for i in items) / n for r in (r1, r2)]
        row["accepted"] = bool(min(k_rec) >= 0.80 and min(k_mean) >= 0.90)
        blocked = [i for i in items if key[i].get("bald_blocked_deleted_content")]
        if blocked:
            row["blocked_original_delivered"] = len(blocked)
            row["recognised_both_excluding_blocked"] = frac(
                sum(human[i] == st for i in items if i not in blocked), n - len(blocked))
        e1[st] = row
    res["e1"] = e1

    # judge vs human on the rewrites
    jr = [i for i in rewrites if i in judge and human[i] != "disputed"]
    conf: Dict[str, Counter] = defaultdict(Counter)
    for i in jr:
        conf[human[i]][judge[i]] += 1
    res["judge_vs_human_rewrites"] = {
        "agree": frac(sum(human[i] == judge[i] for i in jr), len(jr)),
        "n_disputed_excluded": sum(1 for i in rewrites if human[i] == "disputed"),
        "confusion_human_rows_judge_cols": {h: dict(c) for h, c in conf.items()},
        "per_human_class": {st: frac(sum(human[i] == judge[i] for i in jr if human[i] == st),
                                     sum(human[i] == st for i in jr)) for st in STRATS},
        "judge_vs_requested": frac(sum(judge[i] == key[i]["strategy"] for i in rewrites if i in judge),
                                   sum(i in judge for i in rewrites))}

    # E2
    nat = [i for i in rated if key[i]["source"] == "natural"]
    nb = [i for i in nat if key[i]["judge_label"] != "bald_on_record"]
    jb = [i for i in nat if key[i]["judge_label"] == "bald_on_record"]

    def split(items, want_nonbald):
        pos = [i for i in items if human[i] in STRATS and ((human[i] != "bald_on_record") == want_nonbald)]
        den = [i for i in items if human[i] in STRATS]
        return pos, den

    p_pos, p_den = split(nb, True)
    m_pos, m_den = split(jb, True)
    strict = [i for i in p_den if human[i] == key[i]["judge_label"]]
    k_p, n_p, k_m, n_m = len(p_pos), len(p_den), len(m_pos), len(m_den)
    prec, miss = (k_p / n_p if n_p else float("nan")), (k_m / n_m if n_m else float("nan"))
    lo_p, hi_p = clopper_pearson(k_p, n_p)
    lo_m, hi_m = clopper_pearson(k_m, n_m)
    rng = random.Random(seed)
    draws = sorted(w_nb * sum(rng.random() < prec for _ in range(n_p)) / n_p +
                   w_b * sum(rng.random() < miss for _ in range(n_m)) / n_m for _ in range(boot)) \
        if n_p and n_m else []
    res["e2"] = {
        "judge_non_bald_items": len(nb), "judge_bald_items": len(jb),
        "precision_non_bald": frac(k_p, n_p), "precision_non_bald_strict_same_strategy": frac(len(strict), n_p),
        "excluded_precision": {"disputed": sum(human[i] == "disputed" for i in nb),
                               "ambiguous": sum(human[i] == "ambiguous" for i in nb)},
        "miss_rate_bald": frac(k_m, n_m),
        "excluded_miss": {"disputed": sum(human[i] == "disputed" for i in jb),
                          "ambiguous": sum(human[i] == "ambiguous" for i in jb)},
        "weights": {"non_bald": w_nb, "bald": w_b},
        "corrected_non_bald_share": None if not draws else {
            "point": w_nb * prec + w_b * miss,
            "bootstrap_95": [draws[int(0.025 * boot)], draws[int(0.975 * boot)]],
            "conservative_95_from_exact_bounds": [w_nb * lo_p + w_b * lo_m, w_nb * hi_p + w_b * hi_m]},
        "rule": "precision below 50% means the natural trend is not cited as a finding",
        "precision_rule_met": (prec >= 0.5) if n_p else None,
    }
    return res


def render(res: dict) -> str:
    L = [f"# E1 / E2 results", "", f"Raters: {res['raters'][0]}, {res['raters'][1]}. "
         f"Items rated by both: {res['n_rated']}; unrated: {len(res['unrated_items'])}. "
         f"Disputed (raters differ): {len(res['disputed'])}.", "",
         f"Cohen's kappa, strategy: {res['kappa_strategy']}. Meaning: {res['kappa_meaning']}.", "",
         "## E1 per requested strategy (25 items each; exact 95% intervals)", ""]
    L += ["| strategy | recognised, both | meaning kept, both | accepted by the rule |", "| --- | --- | --- | --- |"]
    for st, r in res["e1"].items():
        L.append(f"| {st} | {r['recognised_both']} | {r['meaning_both']} | {'yes' if r['accepted'] else 'no'} |")
    L += ["", "Per rater:", ""]
    for st, r in res["e1"].items():
        for nm in res["raters"]:
            L.append(f"- {st}, {nm}: recognised {r['recognised_' + nm]}; meaning {r['meaning_' + nm]}")
        if "blocked_original_delivered" in r:
            L.append(f"- {st}: {r['blocked_original_delivered']} blocked, original delivered; recognised "
                     f"without them {r['recognised_both_excluding_blocked']}")
    j = res["judge_vs_human_rewrites"]
    L += ["", "## Judge against humans on the 100 rewrites", "",
          f"Agreement with the human label: {j['agree']} ({j['n_disputed_excluded']} disputed excluded). "
          f"Judge against the requested strategy: {j['judge_vs_requested']}.", ""]
    for st, v in j["per_human_class"].items():
        L.append(f"- human {st}: judge agrees {v}")
    L += ["", f"Confusion (rows human, columns judge): {json.dumps(j['confusion_human_rows_judge_cols'])}", "",
          "## E2 judge as verifier on natural speech", ""]
    e = res["e2"]
    L += [f"- Judge non-bald items: {e['judge_non_bald_items']}; judge bald items: {e['judge_bald_items']}",
          f"- Precision of judge non-bald labels (human also non-bald): {e['precision_non_bald']}",
          f"- Same, requiring the same strategy: {e['precision_non_bald_strict_same_strategy']}",
          f"- Miss rate on judge-bald labels (human non-bald): {e['miss_rate_bald']}",
          f"- Excluded from precision {e['excluded_precision']}, from miss rate {e['excluded_miss']}"]
    c = e["corrected_non_bald_share"]
    if c:
        L.append(f"- Corrected non-bald share = {e['weights']['non_bald']} x precision + {e['weights']['bald']} x "
                 f"miss rate = {c['point']:.1%}; bootstrap 95% [{c['bootstrap_95'][0]:.1%}, "
                 f"{c['bootstrap_95'][1]:.1%}]; conservative bounds [{c['conservative_95_from_exact_bounds'][0]:.1%}, "
                 f"{c['conservative_95_from_exact_bounds'][1]:.1%}]")
    L.append(f"- Precision rule ({e['rule']}): {'met' if e['precision_rule_met'] else 'NOT met'}")
    if res["disputed"]:
        L += ["", "Disputed items: " + ", ".join(res["disputed"])]
    return "\n".join(L) + "\n"


def cmd_analyze(args) -> int:
    raters = {name: load_rater(files) for name, *files in args.rater}
    if len(raters) != 2:
        raise SystemExit("give exactly two --rater NAME FILE [FILE...]")
    key = json.loads((E1 / "key.json").read_text(encoding="utf-8"))
    jpath = E1 / "rewrite_judge.jsonl"
    judge = {r["item"]: r["label"] for r in read_jsonl(jpath)} if jpath.exists() else {}
    res = analyse(raters, key, judge)
    (E1 / "e1_results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (E1 / "e1_results.md").write_text(render(res), encoding="utf-8")
    print(render(res))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("judge").set_defaults(fn=cmd_judge)
    a = sub.add_parser("analyze")
    a.add_argument("--rater", nargs="+", action="append", required=True, metavar="NAME_THEN_FILES")
    a.set_defaults(fn=cmd_analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
