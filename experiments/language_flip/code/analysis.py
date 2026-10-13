"""Analysis of the language flip_self replays.

    python -m experiments.language_flip.code.analysis export    # replays in games/ -> data/*.csv
    python -m experiments.language_flip.code.analysis effects   # data/player_days.csv -> results.md

Levels: A = restricted range, B = neutral paraphrase at the original range, C = expanded range. The control replay keeps the
natural text and branches at the same line, so each level is reported as its difference from the control; A and C are also
compared with B, which isolates the change in range from the effect of being paraphrased.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from components.stats import paired_contrast, reliability_detail

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
DEFAULT_GAMES = Path("games/flip_experiments")
WOLVES = {"werewolf", "white_wolf_king"}
LEVELS = ["A", "B", "C"]
MEASURES = ["skill_change_rate", "vote_change_rate", "reliability_change_rate", "mean_abs_point_change", "share_scores_moved_2plus",
            "self_trust_change_all_assessors", "self_trust_change_village_assessors"]


def write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.name}: {len(rows)} rows")


def cmd_export(args) -> int:
    from lurkingwolf.eval.metrics import compute_phase_delta
    from lurkingwolf.eval.replay import extract_baseline_phase
    games, baselines = Path(args.games), {}
    replays, rewrites = [], []
    for f in sorted((games / "language").glob("pe*/day*/*/result.json")):
        seed = f.parts[-4]
        r = json.loads(f.read_text())
        if (seed, r["day"]) not in baselines:
            baselines[(seed, r["day"])] = extract_baseline_phase(games / seed / "natural", r["day"])
        base = baselines[(seed, r["day"])]
        agg = compute_phase_delta(base, SimpleNamespace(**{k: r.get(k) for k in (
            "wolf_target", "seer_target", "guard_target", "votes", "votes_received", "reliability",
            "executed", "config_name", "self_player", "experiment_type")}))["aggregates"]
        cost = json.loads((f.parent / "cost.json").read_text())
        role = r["role_dist"].get(r["player"])
        replays.append({
            "seed": seed, "day": r["day"], "self_player": r["player"], "self_role": role,
            "self_camp": "wolf" if role in WOLVES else "village", "condition": "control" if r["mode"] == "control" else r["setting"],
            "rep": r["rep"], "skill_change_rate": agg["skill_change_rate"], "vote_change_rate": agg["vote_change_rate"],
            "reliability_change_rate": agg["reliability_change_rate"], "mean_reliability_delta": agg["mean_reliability_delta"],
            **dict(zip(MEASURES[3:], reliability_detail(base, r))),
            "votes_on_self_replay": r["votes_received"].get(r["player"], 0), "votes_on_self_baseline": base.votes_received.get(r["player"], 0),
            "executed_replay": r["executed"], "executed_baseline": base.executed,
            "rewrites": r["rewrites"], "original_delivered": r["fallbacks"],
            "cost_usd_ledger": cost["total_cost_usd"], "elapsed_seconds": r["elapsed_seconds"]})
        path = f.parent / "rewrites.jsonl"
        for line in (open(path) if path.exists() else []):
            e = json.loads(line)
            attempts = e.get("attempts") or []
            ok_at = next((i + 1 for i, a in enumerate(attempts) if (a.get("gate") or {}).get("ok") is True), None)
            reasons = lambda a: "|".join((a.get("gate") or {}).get("reasons") or [])
            fallback = bool(e.get("fallback_to_original"))
            rewrites.append({
                "seed": seed, "day": r["day"], "self_player": r["player"], "level": r["setting"], "speaker": e.get("speaker"),
                "predicate": e.get("predicate"), "status": e.get("status"), "original_delivered": fallback,
                "n_attempts": len(attempts), "delivered_attempt": ok_at,
                "failure_attempt1": reasons(attempts[0]) if attempts else "",
                "failure_attempt2": reasons(attempts[1]) if len(attempts) > 1 else "",
                "original_text": e.get("original_nl"), "delivered_text": e.get("original_nl") if fallback else e.get("rewritten_text")})
    write_csv(DATA / "replays.csv", replays)
    write_csv(DATA / "rewrites.csv", rewrites)
    seed_cost, seed_replays = Counter(), Counter()
    for r in replays:
        seed_cost[r["seed"]] += r["cost_usd_ledger"]
        seed_replays[r["seed"]] += 1
    summary = []
    for seed in sorted({r["seed"] for r in rewrites}):
        for level in LEVELS:
            u = [x for x in rewrites if x["seed"] == seed and x["level"] == level]
            summary.append({"seed": seed, "level": level, "rewrites": len(u),
                            "failed_first_attempt": sum(bool(x["failure_attempt1"]) for x in u),
                            "rescued_by_retry": sum(x["delivered_attempt"] == 2 for x in u),
                            "original_delivered": sum(x["original_delivered"] for x in u),
                            "seed_replays": seed_replays[seed], "seed_cost_usd_ledger": round(seed_cost[seed], 4)})
    write_csv(DATA / "seed_level_summary.csv", summary)
    cells, meta = defaultdict(lambda: defaultdict(list)), {}
    for r in replays:
        key = (r["seed"], r["day"], r["self_player"])
        cells[key][r["condition"]].append(r)
        meta[key] = (r["self_role"], r["self_camp"])
    player_days = []
    for key, by_condition in sorted(cells.items()):
        if "control" not in by_condition:
            continue
        row = {"seed": key[0], "day": key[1], "self_player": key[2], "self_role": meta[key][0], "self_camp": meta[key][1],
               "n_control_reps": len(by_condition["control"])}
        for condition in ["control"] + LEVELS:
            reps = by_condition.get(condition)
            for m in MEASURES:
                row[f"{condition}__{m}"] = round(sum(x[m] for x in reps) / len(reps), 6) if reps else ""
        player_days.append(row)
    write_csv(DATA / "player_days.csv", player_days)
    return 0


def cmd_effects(args) -> int:
    rows = list(csv.DictReader(open(DATA / "player_days.csv", encoding="utf-8")))
    scale = [100, 100, 100, 1, 100, 1, 1]
    heads = ["skill targets (pp)", "vote targets (pp)", "reliability scores moved (pp)", "mean absolute point change (pts)",
             "scores moving 2+ pts (pp)", "scores others give the self player, all assessors (pts)", "same, village assessors only (pts)"]
    lines = ["# Language levels minus control, and A and C minus B", "",
             "Share of skill targets, vote targets and reliability scores that differ from the baseline phase, and the size of the reliability "
             "changes. Paired by player-day (repetitions averaged) and pooled over player-days; 95% intervals resample whole seeds (2000 draws).", ""]
    for label, keep in (("all seeds", lambda s: True), ("without pe01", lambda s: s != "pe01")):
        subset = [r for r in rows if keep(r["seed"])]
        lines += [f"## {label}: {len({r['seed'] for r in subset})} seeds, {len(subset)} player-days", "",
                  "| contrast | " + " | ".join(heads) + " |", "|---|" + "--:|" * len(heads)]
        for a, b in (("A", "control"), ("B", "control"), ("C", "control"), ("A", "B"), ("C", "B")):
            cells = []
            for m, k in zip(MEASURES, scale):
                c = paired_contrast(subset, a, b, m)
                cells.append("" if c["estimate"] is None else f"{k * c['estimate']:+.2f} [{k * c['ci_low']:+.2f}, {k * c['ci_high']:+.2f}]")
            lines.append(f"| {a} - {b} | " + " | ".join(cells) + " |")
        lines.append("")
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("export")
    export.add_argument("--games", default=str(DEFAULT_GAMES))
    sub.add_parser("effects")
    args = parser.parse_args(argv)
    return {"export": cmd_export, "effects": cmd_effects}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
