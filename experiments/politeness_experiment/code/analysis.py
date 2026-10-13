"""Analyse the flip replays: manipulation check, judge every delivered text, skill and vote change.

    python -m experiments.politeness_experiment.code.analysis check     # blind judge on a sample of delivered texts
    python -m experiments.politeness_experiment.code.analysis judge     # blind judge on every delivered text
    python -m experiments.politeness_experiment.code.analysis metrics   # change against the baseline phase

The blind judge is gpt-5.6-luna with three passes, not the model that wrote the rewrites. The control
replay is a column next to the four settings: it measures the variation that starts at the changed line.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from components.helpers import read_jsonl
from components.judges.judge_model import LUNA, JudgeModel, majority
from experiments.politeness_experiment.code.run import DEFAULT_GAMES, FLIP_ROOT

WOLVES = {"werewolf", "white_wolf_king"}


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




def cmd_check(args) -> int:
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
    model = JudgeModel.resolve(LUNA)
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


def cmd_judge(args) -> int:
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
    model = JudgeModel.resolve(LUNA)
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


def cmd_metrics(args) -> int:
    """The language experiment's skill and vote change rates against the baseline phase
    (compute_phase_delta), plus the DIRECTION of vote change. 
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
            r["mode"], r["setting"] = "flip_self", "control"
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
    settings = ["control", "all_bald", "all_positive", "trust_positive", "trust_bald"]
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
                    cells.append(f"{sc*sum(v)/len(v):+.2f} (p={signflip_p(v):.2f})" if m not in ("skill_change_rate", "vote_change_rate") else f"{sc*sum(v)/len(v):.1f}")
                lines.append(f"| {label} | {camp} | " + " | ".join(cells) + " |")
        lines += ["", f"### {mode}: setting differences (paired by seed)", "", "| outcome | self camp | contrast | seeds | difference | p |", "|---|---|---|--:|--:|--:|"]
        for m, label, sc in metrics:
            for camp in ("all", "village", "wolf"):
                for a, b in (("all_positive", "all_bald"), ("trust_positive", "all_bald"), ("trust_bald", "all_bald"), ("trust_positive", "trust_bald"),
                             ("all_bald", "control"), ("all_positive", "control"), ("trust_positive", "control"), ("trust_bald", "control")):
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


def reliability_detail(baseline, result: dict) -> list:
    """Size of the reliability-score changes between the baseline phase and a replay (scores are 1-10).
    Returns [mean absolute point change over all scores, share of scores that moved by 2 or more points,
    mean signed change in the scores other players give the self player, the same for village assessors only].
    A wolf's score for another player is its role-guess confidence, not trust, so the village-only figure is the
    trust reading."""
    deltas, self_all, self_village = [], [], []
    me, roles = result.get("self_player") or result.get("player"), result.get("role_dist", {})
    for assessor, scores in (result.get("reliability") or {}).items():
        for target, new in scores.items():
            old = (baseline.reliability.get(assessor) or {}).get(target)
            if old is None or new is None:
                continue
            deltas.append(new - old)
            if target == me and assessor != me:
                self_all.append(new - old)
                if roles.get(assessor) not in ("werewolf", "white_wolf_king"):
                    self_village.append(new - old)
    mean = lambda v: sum(v) / len(v) if v else 0.0
    return [mean([abs(d) for d in deltas]), mean([1.0 if abs(d) >= 2 else 0.0 for d in deltas]), mean(self_all), mean(self_village)]


def cmd_e3(args) -> int:
    """E3: for each setting, the difference from the control (natural text, branching at the same line) in the share of
    skill targets, vote targets and reliability scores that differ from the baseline phase. Paired by player-day
    (seed, day, self player; repetitions averaged), pooled over player-days, 95% interval from a seed-clustered
    bootstrap. Reported with all seeds and without pe01."""
    from types import SimpleNamespace
    from lurkingwolf.eval.metrics import compute_phase_delta
    from lurkingwolf.eval.replay import extract_baseline_phase
    root, games = Path(args.flip_root), Path(DEFAULT_GAMES)
    metrics = ["skill_change_rate", "vote_change_rate", "reliability_change_rate"]
    base, cells = {}, defaultdict(list)                   # (seed, day, player, setting) -> [per-replay metrics]
    for f in sorted(root.glob("pe*/day*/*/result.json")):
        r = json.loads(f.read_text())
        if int(r["seed"][2:]) > args.max_seed:
            continue
        key = (r["seed"], r["day"])
        if key not in base:
            base[key] = extract_baseline_phase(games / r["seed"] / "natural", r["day"])
        agg = compute_phase_delta(base[key], SimpleNamespace(**{k: r.get(k) for k in (
            "wolf_target", "seer_target", "guard_target", "votes", "votes_received", "reliability",
            "executed", "config_name", "self_player", "experiment_type")}))["aggregates"]
        setting = "control" if r["mode"] == "control" else r["setting"]
        extra = reliability_detail(base[key], r)
        cells[(r["seed"], r["day"], r["player"], setting)].append([agg[m] for m in metrics] + extra)
    mean = lambda rows: [sum(c) / len(c) for c in zip(*rows)]
    units = defaultdict(dict)                             # (seed, day, player) -> setting -> mean metrics
    for (seed, day, player, setting), rows in cells.items():
        units[(seed, day, player)][setting] = mean(rows)
    settings = ["all_bald", "all_positive", "trust_positive", "trust_bald"]
    rng = random.Random(12345)

    def estimate(unit_keys, setting, j):
        d = [units[k][setting][j] - units[k]["control"][j] for k in unit_keys if setting in units[k] and "control" in units[k]]
        return sum(d) / len(d) if d else None, len(d)

    def interval(unit_keys, setting, j, B=2000):
        by = defaultdict(list)
        for k in unit_keys:
            by[k[0]].append(k)
        seeds = sorted(by)
        out = []
        for _ in range(B):
            keys = [k for sd in (rng.choice(seeds) for _ in seeds) for k in by[sd]]
            v, _n = estimate(keys, setting, j)
            if v is not None:
                out.append(v)
        out.sort()
        return (out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]) if out else (None, None)

    lines = ["# E3: setting minus control (no change)", "",
             "Share of skill targets, vote targets and reliability scores that differ from the baseline phase, setting minus the "
             "control replay. Paired by player-day (repetitions averaged), pooled over player-days; 95% intervals resample seeds "
             "(2000 draws). Values in percentage points.", ""]
    for label, keep in (("all seeds", lambda s: True), ("without pe01", lambda s: s != "pe01")):
        keys = [k for k in units if keep(k[0]) and "control" in units[k]]
        lines += [f"## {label}: {len({k[0] for k in keys})} seeds, {len(keys)} player-days", "",
                  "| setting | skill targets | vote targets | reliability scores |", "|---|--:|--:|--:|"]
        for st in settings:
            cells_md = []
            for j in range(3):
                v, n = estimate(keys, st, j)
                lo, hi = interval(keys, st, j)
                cells_md.append("" if v is None else f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] (n={n})")
            lines.append(f"| {st} | " + " | ".join(cells_md) + " |")
        ctrl = [sum(units[k]["control"][j] for k in keys) / len(keys) for j in range(3)]
        lines += ["", "Control level (share differing from the baseline): " + ", ".join(
            f"{m.split('_')[0]} {100 * c:.1f}%" for m, c in zip(metrics, ctrl)), ""]
        lines += ["Reliability churn, setting minus control (score points on the 1-10 scale; the 2+ share in percentage points):", "",
                  "| setting | mean absolute point change | share of scores moving 2+ points | change in scores others give the self player (all assessors) | same, village assessors only |",
                  "|---|--:|--:|--:|--:|"]
        for st in settings:
            cells_md = []
            for j, scale in ((3, 1), (4, 100), (5, 1), (6, 1)):
                v, n = estimate(keys, st, j)
                lo, hi = interval(keys, st, j)
                cells_md.append("" if v is None else f"{scale * v:+.2f} [{scale * lo:+.2f}, {scale * hi:+.2f}]")
            lines.append(f"| {st} | " + " | ".join(cells_md) + " |")
        ctrl = [sum(units[k]["control"][j] for k in keys) / len(keys) for j in range(3, 7)]
        lines += ["", f"Control level: mean absolute change {ctrl[0]:.2f} points, {100 * ctrl[1]:.1f}% of scores move 2+ points, "
                      f"scores others give the self player change {ctrl[2]:+.2f} (village assessors {ctrl[3]:+.2f}).", ""]
    out = Path(args.out_e3)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--n", type=int, default=150)
    c.add_argument("--out", default="experiments/politeness_experiment/data/flip_manipulation_check.jsonl")
    j = sub.add_parser("judge")
    j.add_argument("--out-all", default="experiments/politeness_experiment/data/flip_delivered_judged.jsonl")
    m = sub.add_parser("metrics")
    m.add_argument("--out-paper", default="experiments/politeness_experiment/results_flip.md")
    e = sub.add_parser("e3")
    e.add_argument("--out-e3", default="experiments/politeness_experiment/results.md")
    e.add_argument("--max-seed", type=int, default=99, help="use seeds pe01..pe<max-seed> only")
    for q in (c, j, m, e):
        q.add_argument("--flip-root", default=str(FLIP_ROOT))
    args = ap.parse_args(argv)
    return {"check": cmd_check, "judge": cmd_judge, "metrics": cmd_metrics, "e3": cmd_e3}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
