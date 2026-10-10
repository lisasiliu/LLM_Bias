"""Analyse the in-game politeness experiment: report, classify, figures, audit.

    python -m experiments.politeness_experiment.code.analysis classify   # blind judge (gpt-5.6-luna) on delivered text
    python -m experiments.politeness_experiment.code.analysis report     # results.md and data/results.json
    python -m experiments.politeness_experiment.code.analysis figures    # figures/*.svg and figures/index.html
    python -m experiments.politeness_experiment.code.analysis audit      # re-run the admission checks with gpt-5.6-luna

The games were played with `deepseek-chat` running the inverse parser and the semantic
validator on its own rewrites. `audit` rebuilds each delivered rewrite's source record
from `touched.jsonl` and the saved public history, re-runs the admission gate with an
independent model, and appends to `data/luna_reaudit.jsonl`. It is resumable and stops
without writing if the judge is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from components.helpers import read_jsonl
from components.judges.judge_model import majority
from components.helpers import SourceRecord
from components.judges.structural_check import AdmissionGate
from components.judges.judge_model import JudgeModel
from components.rewriters.politeness_rewriter import predicate_definition, predicate_entry
from components.rewriters.trust_politeness_policy import referenced_statement
from experiments.politeness_experiment.code.run import DEFAULT_GAMES
from lurkingwolf.utils.speech_nl import parse_predicate


HERE = Path(__file__).resolve().parents[1]
PASSES = HERE / "data" / "manipulation_passes.jsonl"
RESULTS_JSON = HERE / "data" / "results.json"
RESULTS_MD = HERE / "results.md"
WOLF = {"werewolf", "white_wolf_king"}
ARMS = ["all_bald", "trust_positive", "trust_bald", "all_positive"]
COMPARISONS = [("trust_positive", "all_bald"), ("trust_bald", "all_bald"), ("trust_positive", "trust_bald"),
               ("all_positive", "all_bald")]
SHORT = {"bald_on_record": "bald", "positive_politeness": "positive", "negative_politeness": "negative"}


# ── loading ────────────────────────────────────────────────────────────────

def load(root: Path) -> Dict[tuple, dict]:
    games = {}
    for manifest in sorted(root.glob("pe*/*/manifest.json")):
        d = manifest.parent
        seed, arm = d.parent.name, d.name
        rec = json.loads((d / "game_record.json").read_text())
        hidden = rec["hidden_info"]
        memory = {p.stem: json.loads(p.read_text()) for p in (d / "agent_memory").glob("P*.json")}
        touched = read_jsonl(d / "touched.jsonl") if (d / "touched.jsonl").exists() else []
        nl = json.loads((d / "game_nl.json").read_text()) if (d / "game_nl.json").exists() else []
        games[(seed, arm)] = {
            "seed": seed, "arm": arm, "dir": d, "roles": hidden["role_distribution"],
            "days": hidden.get("day_actions", []), "memory": memory, "touched": touched, "nl": nl,
            "winner": json.loads(manifest.read_text()).get("game_state_status"),
        }
    return games


def rating_after(game: dict, row: dict) -> Optional[int]:
    """The addressee's rating of the speaker at the first rating round after the utterance."""
    a, s = row["addressee"], row["speaker"]
    hist = (game["memory"].get(a) or {}).get("reasoning_history", [])
    n = row["addressee_ratings_so_far"]
    if len(hist) <= n:
        return None
    info = hist[n]["reasoning_result"].get(s)
    if not info:
        return None
    conf = int(info.get("confidence", 5))
    suspected = str(info.get("role", "villager")).strip().lower().replace(" ", "_")
    return 11 - conf if suspected in WOLF else conf


def rating_change(game: dict, row: dict) -> Optional[int]:
    before = row.get("addressee_rating_of_speaker_before")
    after = rating_after(game, row)
    return None if before is None or after is None else after - before


# ── statistics (pure python) ───────────────────────────────────────────────

def signflip_p(diffs: List[float]) -> float:
    """Exact two-sided sign-flip permutation p-value for paired differences."""
    diffs = [d for d in diffs if d != 0]
    if not diffs:
        return 1.0
    obs = abs(sum(diffs))
    sums = [0.0]
    for d in diffs:
        sums = [s + d for s in sums] + [s - d for s in sums]
    return sum(1 for s in sums if abs(s) >= obs - 1e-12) / len(sums)


def wilcoxon_p(diffs: List[float]) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value (zeros dropped, mid-ranks)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    doubled = [int(round(2 * r)) for r in ranks]
    total = sum(doubled)
    dist = {0: 1}
    for r in doubled:
        new = dict(dist)
        for s, c in dist.items():
            new[s + r] = new.get(s + r, 0) + c
        dist = new
    w_plus = sum(r for r, x in zip(doubled, d) if x > 0)
    obs = min(w_plus, total - w_plus)
    tail = sum(c for s, c in dist.items() if min(s, total - s) <= obs)
    return min(1.0, tail / 2 ** n)


def paired(per_seed: Dict[str, Dict[str, float]], a: str, b: str) -> dict:
    pairs = [(v[a], v[b]) for v in per_seed.values() if a in v and b in v]
    diffs = [x - y for x, y in pairs]
    return {"n_pairs": len(diffs),
            "mean_diff": statistics.mean(diffs) if diffs else None,
            "signflip_p": signflip_p(diffs) if diffs else None,
            "wilcoxon_p": wilcoxon_p(diffs) if diffs else None}


# ── manipulation check ─────────────────────────────────────────────────────

def _uid(seed: str, arm: str, i: int) -> str:
    return f"{seed}|{arm}|{i}"


def cmd_classify(args) -> int:
    from components.judges.blind_politeness_classifier import (
        PROMPT_VERSION, BlindPolitenessClassifier, system_prompt)
    from components.judges.judge_model import JudgeModel
    games = load(Path(args.root))
    jobs = []
    for (seed, arm), g in games.items():
        for i, row in enumerate(g["touched"]):
            jobs += [(_uid(seed, arm, i), row, p) for p in (1, 2, 3)]
    done = {(r["id"], r["pass"]) for r in read_jsonl(PASSES)}
    todo = [(u, r, p) for u, r, p in jobs if (u, p) not in done]
    print(f"{len(todo)} judge calls ({len(done)} already stored)")
    if not todo:
        return 0
    model = JudgeModel.resolve("gpt-5.6-luna")
    ok, msg = model.preflight()
    if not ok:
        print("preflight failed:", msg)
        return 1
    clf = BlindPolitenessClassifier(model)
    sha = hashlib.sha256(system_prompt(PROMPT_VERSION).encode()).hexdigest()

    def one(job):
        uid, row, p = job
        res = clf.classify(row["final_text"], phase=row["phase"], speaker_id=row["speaker"])
        return {"id": uid, "pass": p, "label": res.label, "ok": res.ok, "error": res.error,
                "model": "gpt-5.6-luna", "prompt": PROMPT_VERSION, "prompt_sha256": sha}

    PASSES.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=10) as pool, PASSES.open("a", encoding="utf-8") as out:
        for i, r in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps(r) + "\n")
            out.flush()
    return 0


def judged_labels() -> Dict[str, str]:
    by = defaultdict(list)
    for r in read_jsonl(PASSES):
        by[r["id"]].append(r["label"])
    return {u: majority(ls, passes=3)[0] for u, ls in by.items() if len(ls) >= 3}


# ── report ─────────────────────────────────────────────────────────────────

def pct(k: int, n: int) -> str:
    return f"{k} of {n} ({k / n:.0%})" if n else "no cases"


def md_table(header: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def seeds_of(games, arm) -> List[str]:
    return sorted(s for (s, a) in games if a == arm)


def cmd_report(args) -> int:
    games = load(Path(args.root))
    labels = judged_labels()
    res: dict = {}
    md: List[str] = ["# Results", ""]

    # 1. games completed
    all_seeds = sorted({s for s, _ in games})
    arms_all = ["natural", "natural_replay"] + ARMS + (["bl_written"] if any(a == "bl_written" for _, a in games) else [])
    rows = [[s] + [("done" if (s, a) in games else "-") for a in arms_all] for s in all_seeds]
    md += ["## Games", "", f"{len(all_seeds)} seeds with at least one finished game.", "",
           md_table(["Seed"] + arms_all, rows), ""]
    res["seeds"] = all_seeds

    # 2. fallback and delivery
    md += ["## Fallbacks and delivery", ""]
    fb_rows, delivered = [], {}
    for arm in ARMS:
        rows_a = [r for (s, a), g in games.items() if a == arm for r in g["touched"]]
        n = len(rows_a)
        c = Counter(r["delivered_source"] for r in rows_a)
        delivered[arm] = c
        fb_rows.append([arm, n, c["rewrite"], c["bald_fallback"], c["original_fallback"],
                        f"{(n - c['rewrite']) / n:.0%}" if n else "-"])
    md += [md_table(["Arm", "Touched utterances", "Assigned rewrite delivered", "Bald fallback",
                     "Original text", "Any fallback"], fb_rows), ""]
    act_rows = []
    for arm in ARMS:
        for act in ("question", "disagree", "recommend_action"):
            rs = [r for (s, a), g in games.items() if a == arm for r in g["touched"] if r["speech_act"] == act]
            if rs:
                act_rows.append([arm, act, len(rs), f"{sum(r['delivered_source'] != 'rewrite' for r in rs) / len(rs):.0%}"])
    md += ["Fallback rate by speech act:", "", md_table(["Arm", "Speech act", "Utterances", "Any fallback"], act_rows), ""]
    pos = {a: sum(1 for (s, ar), g in games.items() if ar == a for r in g["touched"]
                  if r["assigned_strategy"] == "positive_politeness") for a in ("trust_positive", "trust_bald")}
    pos_ok = {a: sum(1 for (s, ar), g in games.items() if ar == a for r in g["touched"]
                     if r["assigned_strategy"] == "positive_politeness" and r["delivered_source"] == "rewrite")
              for a in ("trust_positive", "trust_bald")}
    md += [f"Positive rewrites assigned and delivered: `trust_positive` {pos_ok['trust_positive']} of {pos['trust_positive']}, "
           f"`trust_bald` {pos_ok['trust_bald']} of {pos['trust_bald']}.", ""]
    res["fallback"] = {a: dict(c) for a, c in delivered.items()}

    # 3. manipulation check
    md += ["## Manipulation check", ""]
    if labels:
        rows_m = []
        for arm in ARMS:
            for group in ("trusted", "distrusted", "middle", "unrated"):
                tot = hit = 0
                for (s, a), g in games.items():
                    if a != arm:
                        continue
                    for i, r in enumerate(g["touched"]):
                        if r["rank_group"] == group and _uid(s, a, i) in labels:
                            tot += 1
                            hit += labels[_uid(s, a, i)] == r["assigned_strategy"]
                if tot:
                    rows_m.append([arm, group, tot, pct(hit, tot)])
        md += ["The blind judge's label for the delivered text matches the assigned strategy:", "",
               md_table(["Arm", "Rank group", "Utterances", "Match"], rows_m), ""]
    else:
        md += ["Not yet run.", ""]

    # 4. primary outcome
    md += ["## Primary outcome: change in the addressee's rating of the speaker", "",
           "Change from the rating in force when the speaker spoke to the addressee's next rating round, "
           "for village-side addressees. Averaged per seed, then compared between arms across seeds.", ""]
    per_seed: Dict[str, Dict[str, float]] = defaultdict(dict)
    per_seed_group: Dict[str, Dict[str, Dict[str, float]]] = {g: defaultdict(dict) for g in ("trusted", "distrusted")}
    counts = Counter()
    wolf_vals = defaultdict(list)
    for (seed, arm), g in games.items():
        if arm not in ARMS:
            continue
        vals, by_group = [], defaultdict(list)
        for r in g["touched"]:
            counts[(arm, "touched")] += 1
            ch = rating_change(g, r)
            village = g["roles"].get(r["addressee"]) not in WOLF
            if not village:
                if ch is not None:
                    wolf_vals[arm].append(ch)
                continue
            if ch is None:
                counts[(arm, "missing")] += 1
                if r.get("addressee_rating_of_speaker_before") is None:
                    counts[(arm, "no_before")] += 1
                else:
                    counts[(arm, "no_after")] += 1
                continue
            vals.append(ch)
            by_group[r["rank_group"]].append(ch)
        if vals:
            per_seed[seed][arm] = statistics.mean(vals)
        for grp in ("trusted", "distrusted"):
            if by_group[grp]:
                per_seed_group[grp][seed][arm] = statistics.mean(by_group[grp])
    rows_p = []
    for arm in ARMS:
        vs = [v[arm] for v in per_seed.values() if arm in v]
        rows_p.append([arm, len(vs), f"{statistics.mean(vs):+.3f}" if vs else "-",
                       counts[(arm, "touched")], counts[(arm, "missing")]])
    md += [md_table(["Arm", "Seeds", "Mean change per seed", "Touched utterances", "No usable change (village addressee)"],
                    rows_p), "",
           "No usable change means the addressee had not yet rated the speaker when the speaker spoke "
           f"({sum(counts[(a, 'no_before')] for a in ARMS)} utterances in total) or no later rating round "
           f"exists because the addressee died or the game ended ({sum(counts[(a, 'no_after')] for a in ARMS)}).", ""]
    comp_rows = []
    res["primary"] = {}
    for a, b in COMPARISONS:
        c = paired(per_seed, a, b)
        res["primary"][f"{a}_vs_{b}"] = c
        comp_rows.append([f"{a} vs {b}", c["n_pairs"],
                          f"{c['mean_diff']:+.3f}" if c["mean_diff"] is not None else "-",
                          f"{c['signflip_p']:.3f}" if c["signflip_p"] is not None else "-",
                          f"{c['wilcoxon_p']:.3f}" if c["wilcoxon_p"] is not None else "-"])
    md += ["Paired by seed:", "", md_table(["Comparison", "Seed pairs", "Mean difference", "Permutation p", "Wilcoxon p"], comp_rows), ""]
    for grp in ("trusted", "distrusted"):
        rows_g = []
        for a, b in COMPARISONS:
            c = paired(per_seed_group[grp], a, b)
            rows_g.append([f"{a} vs {b}", c["n_pairs"],
                           f"{c['mean_diff']:+.3f}" if c["mean_diff"] is not None else "-",
                           f"{c['signflip_p']:.3f}" if c["signflip_p"] is not None else "-"])
        md += [f"Addressees the speaker ranked **{grp}**:", "",
               md_table(["Comparison", "Seed pairs", "Mean difference", "Permutation p"], rows_g), ""]
    md += ["Werewolf addressees (their score is not trust), descriptive:", "",
           md_table(["Arm", "Utterances", "Mean change"],
                    [[a, len(wolf_vals[a]), f"{statistics.mean(wolf_vals[a]):+.3f}" if wolf_vals[a] else "-"] for a in ARMS]), ""]

    # 5. secondary
    md += ["## Secondary outcomes (descriptive)", ""]
    sec_rows = []
    for arm in ["natural"] + ARMS:
        gs = [g for (s, a), g in games.items() if a == arm]
        vill = sum(1 for g in gs if g["winner"] == "villagers_win")
        ex = hit = 0
        for g in gs:
            for d in g["days"]:
                if d.get("executed"):
                    ex += 1
                    hit += g["roles"].get(d["executed"]) in WOLF
        fol = tot = 0
        for g in gs:
            for r in g["touched"]:
                if r["speech_act"] != "recommend_action":
                    continue
                parts = [x.strip() for x in r["predicate"].strip(")").split("(")[1].split(",")]
                if len(parts) >= 5 and parts[3] == "vote_execute" and parts[0].startswith("D"):
                    day = next((d for d in g["days"] if d["day"] == parts[0]), None)
                    if day:
                        tot += 1
                        fol += r["addressee"] in day["votes"].get(parts[4], [])
        sec_rows.append([arm, len(gs), pct(vill, len(gs)), pct(hit, ex), pct(fol, tot) if tot else "-"])
    md += [md_table(["Arm", "Games", "Village wins", "Day executions that hit a werewolf",
                     "vote_execute recommendations followed"], sec_rows), ""]

    # 6. drift
    md += ["## Drift check (natural replay against the baseline)", ""]
    drift_rows = []
    for seed in all_seeds:
        base, rep = games.get((seed, "natural")), games.get((seed, "natural_replay"))
        if not (base and rep):
            continue
        flat = lambda g: [(t["phase"], t["speaker"], it["pred"], it["text"]) for t in g["nl"] for it in t["speech"]]
        fb, fr = flat(base), flat(rep)
        same = sum(1 for x, y in zip(fb, fr) if x == y)
        first = next((i for i, (x, y) in enumerate(zip(fb, fr)) if x != y), None)
        ex = lambda g: [d.get("executed") for d in g["days"]]
        drift_rows.append([seed, f"{same} of {max(len(fb), len(fr))}",
                           "none" if first is None and len(fb) == len(fr) else f"utterance {first}",
                           "yes" if ex(base) == ex(rep) else "no",
                           "yes" if base["winner"] == rep["winner"] else "no"])
    md += [md_table(["Seed", "Utterances identical", "First difference", "Same executions", "Same winner"], drift_rows)
           if drift_rows else "Not run.", ""]
    for seed in all_seeds:
        a, b = games.get((seed, "natural_replay")), games.get((seed, "natural_replay_b"))
        if a and b:
            flat = lambda g: [(t["phase"], t["speaker"], it["pred"], it["text"]) for t in g["nl"] for it in t["speech"]]
            fa, fb = flat(a), flat(b)
            same = sum(1 for x, y in zip(fa, fb) if x == y)
            md += [f"A second natural replay of {seed} from the same checkpoint (`natural_replay_b`) has "
                   f"{same} of {max(len(fa), len(fb))} utterances identical to the first replay, so replays "
                   "differ from one another about as much as from the baseline: the variation comes from the "
                   "model, not from restoring the checkpoint.", ""]

    md += ["## Limitations", "",
           "- One model family (`deepseek-chat`) plays every role, writes the rewrites and runs two of the three checks.",
           "- Twenty seeds are few; the secondary outcomes are descriptive only.",
           "- Reliability ratings are the agents' own LLM-generated numbers, mostly between 4 and 6.",
           "- The blind judge is weak on off-record, which no arm uses; its other labels were validated on a small set.",
           "- Replays from a checkpoint do not reproduce the baseline game exactly (see the drift check), so arms are compared with each other, not with the baseline's later play."]
    RESULTS_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    RESULTS_JSON.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_MD}")
    return 0




FIG = HERE / "figures"
COLORS = {"bald": "#52514e", "positive": "#2a78d6", "negative": "#eb6834", "other": "#b9b7ae",
          "rewrite": "#2a78d6", "bald fallback": "#1baf7a", "original text": "#b9b7ae"}
STYLE = ("<style>.f{font-family:system-ui,-apple-system,Segoe UI,sans-serif;font-size:12px}"
         ".t{fill:#0b0b0b}.t2{fill:#52514e}.g{stroke:#e6e5e1}"
         "@media (prefers-color-scheme:dark){.t{fill:#fff}.t2{fill:#c3c2b7}.g{stroke:#34332f}}</style>")
KEY = ["Settings (how the speaker addresses each listener):",
       "all_bald: bald on-record to everyone",
       "trust_positive: positive politeness to trusted listeners, bald to the rest",
       "trust_bald: positive politeness to distrusted listeners, bald to the rest",
       "Trusted / distrusted = top / bottom half of the speaker's own reliability ratings."]
VOTE = re.compile(r"voted\(D(\d+), (P\d+), (P\d+)\)")


# ── data ───────────────────────────────────────────────────────────────────

def votes_of(game: dict) -> Dict[Tuple[int, str], str]:
    """(day, voter) -> target, from the saved public history."""
    import json
    out = {}
    phases = json.loads((game["dir"] / "public_history.json").read_text())["phases"]
    for lines in phases.values():
        for line in lines if isinstance(lines, list) else []:
            m = VOTE.match(line) if isinstance(line, str) else None
            if m:
                out[(int(m[1]), m[2])] = m[3]
    return out


def units(games: dict, labels: Dict[str, str]) -> List[dict]:
    rows = []
    for (seed, arm), g in games.items():
        if arm not in ARMS:
            continue
        votes = votes_of(g)
        for i, r in enumerate(g["touched"]):
            if g["roles"].get(r["addressee"]) in WOLF or g["roles"].get(r["speaker"]) in WOLF:
                continue                      # village speaker to village addressee only
            label = SHORT.get(labels.get(f"{seed}|{arm}|{i}", ""), "other")
            voted = votes.get((r["day"], r["addressee"]))
            rows.append({"game": f"{seed}|{arm}", "arm": arm, "group": r["rank_group"],
                         "label": label, "act": r["speech_act"],
                         "change": rating_change(g, r),
                         "vote_against": None if voted is None else int(voted == r["speaker"])})
    return rows


def boot(rows: List[dict], key: str, n: int = 1000) -> Optional[Tuple[float, float, float, int]]:
    vals = [(r["game"], r[key]) for r in rows if r[key] is not None]
    if len(vals) < 5:
        return None
    by = defaultdict(list)
    for g, v in vals:
        by[g].append(v)
    games, rng = list(by), random.Random(101)
    means = []
    for _ in range(n):
        pool = [v for g in rng.choices(games, k=len(games)) for v in by[g]]
        means.append(sum(pool) / len(pool))
    means.sort()
    allv = [v for _, v in vals]
    return sum(allv) / len(allv), means[int(.025 * n)], means[int(.975 * n)], len(allv)


# ── svg ────────────────────────────────────────────────────────────────────

def wrap(w: int, h: int, title: str, body: List[str], sub: str = "", key: bool = True) -> str:
    if key:
        body = body + [f'<text class="t2" x="16" y="{h + 6 + 16 * i}"'
                       f'{" style=font-weight:600" if i == 0 else ""}>{html.escape(t)}</text>'
                       for i, t in enumerate(KEY)]
        h += 6 + 16 * len(KEY)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="f" viewBox="0 0 {w} {h}" width="{w}" '
            f'role="img" aria-label="{html.escape(title)}">{STYLE}<rect width="{w}" height="{h}" fill="none"/>'
            f'<text class="t" x="16" y="24" style="font-size:15px;font-weight:600">{html.escape(title)}</text>'
            f'<text class="t2" x="16" y="42">{html.escape(sub)}</text>' + "".join(body) + "</svg>")


def stacked(title: str, sub: str, rows: List[Tuple[str, Dict[str, int]]], order: List[str]) -> str:
    left, width, top, bh = 190, 420, 62, 26
    body = []
    for k, (name, counts) in enumerate(rows):
        y, total = top + k * (bh + 10), sum(counts.values()) or 1
        body.append(f'<text class="t" x="{left - 8}" y="{y + 17}" text-anchor="end">{html.escape(name)}</text>')
        x = left
        for cat in order:
            w = width * counts.get(cat, 0) / total
            if w:
                body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bh}" fill="{COLORS[cat]}"/>')
                if w > 26:
                    body.append(f'<text x="{x + w / 2:.1f}" y="{y + 17}" fill="#fff" text-anchor="middle">'
                                f'{100 * counts[cat] / total:.0f}%</text>')
                x += w
        body.append(f'<text class="t2" x="{left + width + 8}" y="{y + 17}">n={sum(counts.values())}</text>')
    ly = top + len(rows) * (bh + 10) + 8
    for j, cat in enumerate(order):
        body.append(f'<rect x="{left + j * 120}" y="{ly}" width="12" height="12" fill="{COLORS[cat]}"/>'
                    f'<text class="t2" x="{left + j * 120 + 18}" y="{ly + 11}">{cat}</text>')
    return wrap(left + width + 70, ly + 30, title, body, sub)


def dots(title: str, sub: str, rows: List[Tuple[str, Optional[tuple]]], lo: float, hi: float,
         ref: Optional[float] = None, fmt: str = "{:+.2f}") -> str:
    left, width, top, step = 230, 360, 64, 30
    sx = lambda v: left + width * (v - lo) / (hi - lo)
    body = []
    if ref is not None:
        body.append(f'<line class="g" x1="{sx(ref)}" x2="{sx(ref)}" y1="{top - 10}" y2="{top + len(rows) * step}" stroke-dasharray="4 3"/>')
    for k, (name, est) in enumerate(rows):
        y = top + k * step
        body.append(f'<text class="t" x="{left - 10}" y="{y + 4}" text-anchor="end">{html.escape(name)}</text>')
        if est is None:
            body.append(f'<text class="t2" x="{left + 6}" y="{y + 4}">too few data</text>')
            continue
        m, a, b, n = est
        body.append(f'<line x1="{sx(max(a, lo)):.1f}" x2="{sx(min(b, hi)):.1f}" y1="{y}" y2="{y}" stroke="#2a78d6" stroke-width="2.5"/>'
                    f'<circle cx="{sx(m):.1f}" cy="{y}" r="5" fill="#2a78d6"/>'
                    f'<text class="t2" x="{left + width + 12}" y="{y + 4}">{fmt.format(m)}  (n={n})</text>')
    ay = top + len(rows) * step
    for v in (lo, (lo + hi) / 2, hi):
        body.append(f'<text class="t2" x="{sx(v):.1f}" y="{ay + 6}" text-anchor="middle">{fmt.format(v)}</text>')
    return wrap(left + width + 150, ay + 24, title, body, sub)


# ── figures ────────────────────────────────────────────────────────────────

def cmd_figures(args=None) -> None:
    games = load(DEFAULT_GAMES)
    labels = judged_labels()
    rows = units(games, labels)
    FIG.mkdir(exist_ok=True)
    out: Dict[str, str] = {}

    delivery = []
    for arm in ARMS:
        c = Counter()
        for (seed, a), g in games.items():
            if a == arm:
                for r in g["touched"]:
                    c[{"rewrite": "rewrite", "bald_fallback": "bald fallback"}.get(r["delivered_source"], "original text")] += 1
        delivery.append((arm, dict(c)))
    out["1_delivery"] = stacked("What was delivered, by setting",
                                "assigned rewrite vs bald fallback vs original text (all touched utterances)",
                                delivery, ["rewrite", "bald fallback", "original text"])

    comp = []
    for arm in ARMS:
        for grp in ("trusted", "distrusted", "middle"):
            c = Counter(r["label"] for r in rows if r["arm"] == arm and r["group"] == grp)
            comp.append((f"{arm} / {grp}", dict(c)))
    out["2_judged_strategy"] = stacked("Strategy the blind judge found, by setting and addressee rank",
                                       "gpt-5.6-luna majority label of the delivered text (village-to-village)",
                                       comp, ["bald", "positive", "negative", "other"])

    cats = [("all utterances", lambda r: True)]
    cats += [(f"delivered {s}", (lambda s: lambda r: r["label"] == s)(s)) for s in ("bald", "positive")]
    cats += [(f"setting: {a}", (lambda a: lambda r: r["arm"] == a)(a)) for a in ARMS]
    cats += [(f"{g} addressee", (lambda g: lambda r: r["group"] == g)(g)) for g in ("trusted", "distrusted")]
    sub = "addressee's rating of speaker at next round minus rating when spoken; 95% game bootstrap"
    out["3_rating_change"] = dots("Change in the addressee's trust in the speaker", sub,
                                  [(n, boot([r for r in rows if f(r)], "change")) for n, f in cats],
                                  -1.0, 1.5, ref=0)
    out["4_vote_against_speaker"] = dots(
        "Did the addressee vote against the speaker that day?",
        "share of village addressees whose same-day vote targeted the speaker; 95% game bootstrap",
        [(n, boot([r for r in rows if f(r)], "vote_against")) for n, f in cats],
        0.0, 0.5, fmt="{:.2f}")

    by_act = [(a, lambda r, a=a: r["act"] == a) for a in ("recommend_action", "question")]
    out["5_vote_against_by_act"] = dots(
        "Votes against the speaker, by speech act and strategy",
        "village-to-village utterances; 95% game bootstrap",
        [(f"{a} / {s}", boot([r for r in rows if f(r) and r["label"] == s], "vote_against"))
         for a, f in by_act for s in ("bald", "positive")], 0.0, 0.6, fmt="{:.2f}")

    for name, svg in out.items():
        (FIG / f"{name}.svg").write_text(svg)
    page = "".join(f'<h3>{n}</h3>{s}' for n, s in out.items())
    (FIG / "index.html").write_text(f"<!doctype html><meta charset=utf-8><title>Pilot figures</title>"
                                    f"<body style='font-family:system-ui;margin:24px'>{page}</body>")
    print(f"{len(rows)} village-to-village units; wrote {len(out)} figures to {FIG}")



AUDIT_OUT = Path(__file__).resolve().parents[1] / "data" / "luna_reaudit.jsonl"


def history_before(game_dir: Path, record: dict) -> list:
    """Public-history lines preceding the utterance, or [] if it cannot be located."""
    phases = json.loads((game_dir / "public_history.json").read_text())["phases"]
    flat = [line for lines in phases.values() if isinstance(lines, list)
            for line in lines if isinstance(line, str)]
    suffix = f"{record['predicate']}: {record['final_text']}"
    for i, line in enumerate(flat):
        if line.endswith(suffix) and f"|{record['speaker']}]" in line:
            return flat[:i]
    return []


def source_for(game_dir: Path, r: dict) -> SourceRecord | None:
    history = history_before(game_dir, r)
    if not history:
        return None
    parsed = parse_predicate(r["predicate"])
    name, args = parsed["name"], list(parsed["args"])
    ref = referenced_statement(history, args[-1].strip()) if name in ("disagree", "question", "agree") and args else None
    return SourceRecord(
        source_id=f"{r['seed']}:{r['arm']}:{r['speaker']}:{len(history)}",
        predicate_name=name, predicate_text=r["predicate"], predicate_args=args,
        predicate_definition=predicate_definition(name),
        nl_hint=predicate_entry(name).get("nl_hint", ""), original_nl=r["original_text"],
        phase=r["phase"], speaker_id=r["speaker"], speaker_role_claimed=r["speaker_role"],
        game_id=r["seed"], public_history=history, referenced_speech=ref)


def cmd_audit(args) -> None:

    done = set()
    if AUDIT_OUT.exists():
        done = {json.loads(l)["key"] for l in AUDIT_OUT.read_text().splitlines() if l.strip()}
    gate = AdmissionGate(JudgeModel.resolve("gpt-5.6-luna"))
    n = 0
    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_OUT.open("a") as fh:
        for path in sorted(args.root.glob("pe*/*/touched.jsonl")):
            for i, line in enumerate(path.read_text().splitlines()):
                r = json.loads(line)
                if r["delivered_source"] == "original_fallback" or not r["final_text"]:
                    continue
                key = f"{r['seed']}|{r['arm']}|{i}"
                if key in done:
                    continue
                source = source_for(path.parent, r)
                row = {"key": key, "seed": r["seed"], "arm": r["arm"],
                       "speech_act": r["speech_act"], "delivered_source": r["delivered_source"],
                       "assigned_strategy": r["assigned_strategy"], "rank_group": r["rank_group"]}
                if source is None:
                    row["error"] = "history not located"
                else:
                    attempt = next((a for a in r["attempts"] if a["admitted"]), None)
                    row["deepseek_admitted"] = bool(attempt)
                    try:
                        res = gate.check(source, r["final_text"],
                                         commitment_may_be_indirect=False)
                        if res["semantic_status"] == "no_verdict" or res["parser_error"]:
                            raise SystemExit(f"Luna unavailable ({res['parser_error'] or res['semantic_reasons']}); "
                                             f"stopped after {n} rows, rerun to resume")
                        row.update(luna_admitted=res["admitted"], parser_status=res["parser_status"],
                                   semantic_status=res["semantic_status"],
                                   semantic_reasons=res["semantic_reasons"],
                                   parser_mismatched_slots=res["parser_mismatched_slots"],
                                   structural_ok=res["structural_ok"])
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"[:300]
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                if args.limit and n >= args.limit:
                    return


# ── flip-self / flip-others ──

FLIP_ROOT = Path("games_fresh/politeness_flip")
WOLVES = {"werewolf", "white_wolf_king"}


def cmd_flipcheck(args) -> int:
    """Manipulation check on the flip games: blind Luna (3-pass majority) on a random sample of delivered
    texts per setting, plus the untouched original text of the same utterances as a natural reference."""
    from components.judges.blind_politeness_classifier import BlindPolitenessClassifier
    rng = random.Random(0)
    rows = defaultdict(list)
    for f in sorted(Path(args.flip_root).glob("*/day*/*/touched.jsonl")):
        for l in f.read_text().splitlines():
            r = json.loads(l)
            rows[r["arm"]].append(r)
    jobs = []
    for arm, rs in sorted(rows.items()):
        for r in rng.sample(rs, min(args.n, len(rs))):
            jobs.append((arm, "delivered", r["delivered_source"], r["final_text"], r))
            jobs.append((arm, "original", r["delivered_source"], r["original_text"], r))
    model = JudgeModel.resolve("gpt-5.6-luna")
    ok, msg = model.preflight()
    if not ok:
        print("preflight failed:", msg)
        return 1
    clf = BlindPolitenessClassifier(model)

    def one(job):
        arm, kind, src, text, r = job
        labels = [clf.classify(text, phase=r["phase"], speaker_id=r["speaker"]).label for _ in range(3)]
        return {"setting": arm, "kind": kind, "delivered_source": src, "speech_act": r["speech_act"],
                "speaker_camp": "wolf" if r["speaker_role"] in WOLVES else "village",
                "assigned": r["assigned_strategy"], "labels": labels, "label": majority(labels, passes=3)[0]}
    out = Path(args.out)
    with ThreadPoolExecutor(max_workers=10) as pool, out.open("w", encoding="utf-8") as fh:
        for res in pool.map(one, jobs):
            fh.write(json.dumps(res) + "\n")
    tab = defaultdict(Counter)
    for r in read_jsonl(out):
        tab[(r["setting"], r["kind"])][r["label"]] += 1
    for (arm, kind), c in sorted(tab.items()):
        print(arm, kind, dict(c))
    return 0


def cmd_flipall(args) -> int:
    """Blind Luna on every delivered text in the flip games (adaptive 2+1 vote, identical to a 3-pass
    majority). Resumable; one row per touched utterance."""
    from components.judges.blind_politeness_classifier import BlindPolitenessClassifier
    out = Path(args.out_all)
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    jobs = []
    for f in sorted(Path(args.flip_root).glob("*/day*/*/touched.jsonl")):
        for i, l in enumerate(f.read_text().splitlines()):
            r = json.loads(l)
            uid = f"{f.parent.parent.parent.name}|{f.parent.parent.name}|{f.parent.name}|{i}"
            if uid not in done:
                jobs.append((uid, r))
    print(f"{len(jobs)} utterances to judge ({len(done)} done)")
    model = JudgeModel.resolve("gpt-5.6-luna")
    ok, msg = model.preflight()
    if not ok:
        print("preflight failed:", msg)
        return 1
    clf = BlindPolitenessClassifier(model)

    def one(job):
        uid, r = job
        lab = lambda: clf.classify(r["final_text"], phase=r["phase"], speaker_id=r["speaker"]).label
        labels = [lab(), lab()]
        if labels[0] != labels[1]:
            labels.append(lab())
        return {"id": uid, "setting": r["arm"], "delivered_source": r["delivered_source"],
                "speech_act": r["speech_act"], "assigned": r["assigned_strategy"],
                "speaker_camp": "wolf" if r["speaker_role"] in WOLVES else "village",
                "labels": labels, "label": majority(labels, passes=len(labels))[0]}
    with ThreadPoolExecutor(max_workers=10) as pool, out.open("a", encoding="utf-8") as fh:
        for res in pool.map(one, jobs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    return 0


def cmd_flipmetrics(args) -> int:
    """The language experiment's skill and vote change rates against the baseline phase
    (compute_phase_delta), plus the DIRECTION of vote change. Reliability is held back (see Q37).
    Differences between settings are paired by seed (sign-flip, two-sided); every setting is also
    tested against zero change from the baseline."""
    from types import SimpleNamespace
    from lurkingwolf.eval.metrics import compute_phase_delta
    from lurkingwolf.eval.replay import extract_baseline_phase
    root, games = Path(args.flip_root), Path(DEFAULT_GAMES)
    base, rows = {}, []
    for f in sorted(root.glob("*/day*/*/result.json")):
        r = json.loads(f.read_text())
        if r["mode"] == "control":
            continue
        key = (r["seed"], r["day"])
        if key not in base:
            base[key] = extract_baseline_phase(games / r["seed"] / "natural", r["day"])
        b = base[key]
        d = compute_phase_delta(b, SimpleNamespace(**{k: r.get(k) for k in (
            "wolf_target", "seer_target", "guard_target", "votes", "votes_received", "reliability",
            "executed", "config_name", "self_player", "experiment_type")}))["aggregates"]
        me, roles = r["player"], r["role_dist"]
        wolves = {p for p, ro in roles.items() if ro in WOLVES}

        def wolf_share(votes):
            cast = [v for v in votes.values() if v]
            return sum(v in wolves for v in cast) / len(cast) if cast else None
        ws_r, ws_b = wolf_share(r["votes"]), wolf_share(b.votes)
        others = [p for p in r["alive_at_start"] if p != me]
        rows.append({
            "seed": r["seed"], "mode": r["mode"], "setting": r["setting"],
            "self_camp": "wolf" if me in wolves else "village",
            "skill_change_rate": d["skill_change_rate"], "vote_change_rate": d["vote_change_rate"],
            "votes_on_self": r["votes_received"].get(me, 0) - b.votes_received.get(me, 0),
            "votes_on_others": (sum(r["votes_received"].get(p, 0) - b.votes_received.get(p, 0) for p in others) / len(others)) if others else None,
            "wolf_vote_share": None if ws_r is None or ws_b is None else ws_r - ws_b})
    metrics = [("skill_change_rate", "skill targets changed (%)", 100), ("vote_change_rate", "vote targets changed (%)", 100),
               ("votes_on_self", "votes on the self player, replay minus baseline", 1),
               ("votes_on_others", "votes per other player, replay minus baseline", 1),
               ("wolf_vote_share", "share of votes cast on wolves, replay minus baseline (pp)", 100)]
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m, _, _ in metrics:
            if r[m] is not None:
                per[(r["mode"], r["setting"], "all", m)][r["seed"]].append(r[m])
                per[(r["mode"], r["setting"], r["self_camp"], m)][r["seed"]].append(r[m])
    sm = {k: {sd: sum(v) / len(v) for sd, v in d.items()} for k, d in per.items()}
    settings = ["all_bald", "all_positive", "trust_positive", "trust_bald"]
    lines = ["# Flip games: skill and vote change against the baseline game", "",
             f"Seeds: {len({r['seed'] for r in rows})}. Replays: {len(rows)}. Values are means over seeds. "
             "'vs baseline' tests each setting against zero change (sign-flip over seeds); differences between settings are paired by seed.", ""]
    for mode in ("flip_self", "flip_others"):
        lines += [f"## {mode}", "", "| outcome | self camp | " + " | ".join(settings) + " |", "|---|---|" + "--:|" * len(settings)]
        for m, label, sc in metrics:
            if mode == "flip_others" and m == "votes_on_self":
                label = "votes on the untreated self player, replay minus baseline"
            for camp in ("all", "village", "wolf"):
                cells = []
                for s in settings:
                    d = sm.get((mode, s, camp, m), {})
                    if len(d) < 3:
                        cells.append("")
                        continue
                    v = list(d.values())
                    zero = 0 if m in ("skill_change_rate", "vote_change_rate") else 0
                    cells.append(f"{sc*sum(v)/len(v):+.2f} (p={signflip_p(v):.2f})" if m not in ("skill_change_rate", "vote_change_rate") else f"{sc*sum(v)/len(v):.1f}")
                lines.append(f"| {label} | {camp} | " + " | ".join(cells) + " |")
        lines += ["", f"### {mode}: setting differences (paired by seed)", "", "| outcome | self camp | contrast | seeds | difference | p |", "|---|---|---|--:|--:|--:|"]
        for m, label, sc in metrics:
            for camp in ("all", "village", "wolf"):
                for a, b in (("all_positive", "all_bald"), ("trust_positive", "all_bald"), ("trust_bald", "all_bald"), ("trust_positive", "trust_bald")):
                    A, B = sm.get((mode, a, camp, m), {}), sm.get((mode, b, camp, m), {})
                    seeds = sorted(set(A) & set(B))
                    if len(seeds) < 3:
                        continue
                    dd = [A[x] - B[x] for x in seeds]
                    lines.append(f"| {label} | {camp} | {a} - {b} | {len(seeds)} | {sc*sum(dd)/len(dd):+.2f} | {signflip_p(dd):.2f} |")
        lines.append("")
    out = Path(args.out_paper)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("classify", "report", "figures", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=DEFAULT_GAMES)
        if name == "audit":
            p.add_argument("--limit", type=int, default=0)
    fc = sub.add_parser("flipcheck")
    fc.add_argument("--flip-root", default=str(FLIP_ROOT))
    fc.add_argument("--n", type=int, default=150)
    fc.add_argument("--out", default="experiments/politeness_experiment/data/flip_manipulation_check.jsonl")
    for nm in ("flipall",):
        q = sub.add_parser(nm)
        q.add_argument("--flip-root", default=str(FLIP_ROOT))
        q.add_argument("--out-all", default="experiments/politeness_experiment/data/flip_delivered_judged.jsonl")
    pp = sub.add_parser("flipmetrics")
    pp.add_argument("--flip-root", default=str(FLIP_ROOT))
    pp.add_argument("--out-paper", default="experiments/politeness_experiment/results_flip.md")
    args = ap.parse_args(argv)
    if args.cmd in ("classify", "report"):
        args.root = str(args.root)
    return {"classify": cmd_classify, "report": cmd_report, "figures": cmd_figures,
            "audit": cmd_audit, "flipcheck": cmd_flipcheck, "flipall": cmd_flipall, "flipmetrics": cmd_flipmetrics}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
