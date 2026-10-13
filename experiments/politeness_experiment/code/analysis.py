"""Analysis of the politeness flip replays.

    python -m experiments.politeness_experiment.code.analysis export    # replays in games/ -> data/*.csv
    python -m experiments.politeness_experiment.code.analysis effects   # data/player_days.csv -> results.md
    python -m experiments.politeness_experiment.code.analysis judge     # blind judge on delivered texts (manipulation check)

`effects` needs only the checked-in CSVs. The control replay (natural text, branching at the same line as a treated replay)
measures the variation a replay shows on its own, so each setting is reported as its difference from the control.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from components.judges.judge_model import LUNA, JudgeModel, majority
from components.stats import paired_contrast
from experiments.politeness_experiment.code.run import DEFAULT_GAMES

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data"
WOLVES = {"werewolf", "white_wolf_king"}
SETTINGS = ["all_bald", "all_positive", "trust_positive", "trust_bald"]
MEASURES = ["skill_change_rate", "vote_change_rate", "reliability_change_rate", "mean_abs_point_change",
            "share_scores_moved_2plus", "self_trust_change_all_assessors", "self_trust_change_village_assessors"]


def reliability_detail(baseline, result: dict) -> list:
    """Size of the reliability-score changes between the baseline phase and a replay (scores are 1-10).

    Returns [mean absolute point change over all scores, share of scores that moved by 2 or more points, mean signed
    change in the scores the other players give the self player, the same for village assessors only]. A wolf's score for
    another player is its role-guess confidence, not trust, so the village-only figure is the trust reading."""
    deltas, to_self, to_self_village = [], [], []
    me, roles = result.get("self_player") or result.get("player"), result.get("role_dist", {})
    for assessor, scores in (result.get("reliability") or {}).items():
        for target, new in scores.items():
            old = (baseline.reliability.get(assessor) or {}).get(target)
            if old is None or new is None:
                continue
            deltas.append(new - old)
            if target == me and assessor != me:
                to_self.append(new - old)
                if roles.get(assessor) not in WOLVES:
                    to_self_village.append(new - old)
    mean = lambda v: sum(v) / len(v) if v else 0.0
    return [mean([abs(d) for d in deltas]), mean([1.0 if abs(d) >= 2 else 0.0 for d in deltas]), mean(to_self), mean(to_self_village)]


def write_csv(path: Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path.relative_to(HERE.parents[1])}: {len(rows)} rows")


# ── export ──

def cmd_export(args) -> int:
    from lurkingwolf.eval.metrics import compute_phase_delta
    from lurkingwolf.eval.replay import extract_baseline_phase
    root, baselines = Path(args.games), {}
    replays, utterances = [], []
    for f in sorted(root.glob("pe*/day*/*/result.json")):
        seed = f.parts[-4]
        r = json.loads(f.read_text())
        if (seed, r["day"]) not in baselines:
            baselines[(seed, r["day"])] = extract_baseline_phase(root / seed / "natural", r["day"])
        base = baselines[(seed, r["day"])]
        agg = compute_phase_delta(base, SimpleNamespace(**{k: r.get(k) for k in (
            "wolf_target", "seer_target", "guard_target", "votes", "votes_received", "reliability",
            "executed", "config_name", "self_player", "experiment_type")}))["aggregates"]
        cost = json.loads((f.parent / "cost.json").read_text())
        touched_path = f.parent / "touched.jsonl"
        touched = [json.loads(l) for l in open(touched_path)] if touched_path.exists() else []
        role = r["role_dist"].get(r["player"])
        condition = "control" if r["mode"] == "control" else r["setting"]
        replays.append({
            "seed": seed, "day": r["day"], "self_player": r["player"], "self_role": role,
            "self_camp": "wolf" if role in WOLVES else "village", "condition": condition, "rep": r["rep"],
            "players_alive_at_start": len(r["alive_at_start"]),
            "skill_change_rate": agg["skill_change_rate"], "vote_change_rate": agg["vote_change_rate"],
            "reliability_change_rate": agg["reliability_change_rate"], "mean_reliability_delta": agg["mean_reliability_delta"],
            **dict(zip(MEASURES[3:], reliability_detail(base, r))),
            "votes_on_self_replay": r["votes_received"].get(r["player"], 0), "votes_on_self_baseline": base.votes_received.get(r["player"], 0),
            "executed_replay": r["executed"], "executed_baseline": base.executed,
            "treated_utterances": len(touched), "original_delivered": sum(t["delivered"] == "original" for t in touched),
            "cost_usd_ledger": cost["total_cost_usd"], "elapsed_seconds": r["elapsed_seconds"]})
        for t in touched:
            attempts = t["attempts"]
            reasons = t["gate_failure_reasons"]
            utterances.append({
                "seed": seed, "day": r["day"], "self_player": r["player"], "rep": r["rep"], "setting": t["arm"],
                "speaker": t["speaker"], "speaker_role": t["speaker_role"], "speaker_camp": "wolf" if t["speaker_role"] in WOLVES else "village",
                "speech_act": t["speech_act"], "addressee": t["addressee"], "addressee_role": t["addressee_role"],
                "rank_group": t["rank_group"], "rank_group_by_reliability": t.get("rank_group_by_reliability"),
                "rank_group_by_consensus": t.get("rank_group_by_consensus"), "assigned_strategy": t["assigned_strategy"],
                "delivered": t["delivered"], "delivered_attempt": t["delivered_attempt"], "n_attempts": len(attempts),
                "failure_attempt1": "|".join(reasons[0] or []) if reasons else "",
                "failure_attempt2": "|".join(reasons[1] or []) if len(reasons) > 1 else "",
                "reliability_for_addressee": t.get("reliability_for_addressee"),
                "town_consensus_for_addressee": t.get("town_consensus_for_addressee"),
                "original_text": t["original_text"], "attempt1_text": attempts[0].get("text") if attempts else "",
                "delivered_text": t["final_text"]})
    DATA.mkdir(exist_ok=True)
    write_csv(DATA / "replays.csv", replays)
    write_csv(DATA / "treated_utterances.csv", utterances)
    # per seed and setting: how many rewrites the gate let through
    seed_cost, seed_replays = Counter(), Counter()
    for r in replays:
        seed_cost[r["seed"]] += r["cost_usd_ledger"]
        seed_replays[r["seed"]] += 1
    summary = []
    for seed in sorted(seed_replays):
        for setting in SETTINGS:
            u = [x for x in utterances if x["seed"] == seed and x["setting"] == setting]
            summary.append({"seed": seed, "setting": setting, "treated": len(u),
                            "failed_first_attempt": sum(bool(x["failure_attempt1"]) for x in u),
                            "rescued_by_retry": sum(x["delivered_attempt"] == 2 for x in u),
                            "original_delivered": sum(x["delivered"] == "original" for x in u),
                            "seed_replays": seed_replays[seed], "seed_cost_usd_ledger": round(seed_cost[seed], 4)})
    write_csv(DATA / "seed_setting_summary.csv", summary)
    # one row per (seed, day, self player) with the control and each setting averaged over repetitions
    cells = defaultdict(lambda: defaultdict(list))
    meta = {}
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
        for condition in ["control"] + SETTINGS:
            reps = by_condition.get(condition)
            for m in MEASURES:
                row[f"{condition}__{m}"] = round(sum(x[m] for x in reps) / len(reps), 6) if reps else ""
        player_days.append(row)
    write_csv(DATA / "player_days.csv", player_days)
    return 0


# ── effects ──

def cmd_effects(args) -> int:
    rows = list(csv.DictReader(open(DATA / "player_days.csv", encoding="utf-8")))
    scale = [100, 100, 100, 1, 100, 1, 1]
    heads = ["skill targets (pp)", "vote targets (pp)", "reliability scores moved (pp)", "mean absolute point change (pts)",
             "scores moving 2+ pts (pp)", "scores others give the self player, all assessors (pts)", "same, village assessors only (pts)"]
    lines = ["# Politeness settings minus control", "",
             "Share of skill targets, vote targets and reliability scores that differ from the baseline phase, and the size of the reliability "
             "changes, for each setting minus the control replay. Paired by player-day (repetitions averaged) and pooled over player-days; "
             "95% intervals resample whole seeds (2000 draws).", ""]
    for label, keep in (("all seeds", lambda s: True), ("without pe01", lambda s: s != "pe01")):
        subset = [r for r in rows if keep(r["seed"])]
        lines += [f"## {label}: {len({r['seed'] for r in subset})} seeds, {len(subset)} player-days", "",
                  "| setting | " + " | ".join(heads) + " |", "|---|" + "--:|" * len(heads)]
        for setting in SETTINGS:
            cells = []
            for m, k in zip(MEASURES, scale):
                c = paired_contrast(subset, setting, "control", m)
                cells.append("" if c["estimate"] is None else f"{k * c['estimate']:+.2f} [{k * c['ci_low']:+.2f}, {k * c['ci_high']:+.2f}]")
            lines.append(f"| {setting} | " + " | ".join(cells) + " |")
        level = []
        for m, k in zip(MEASURES, scale):
            vals = [float(r[f"control__{m}"]) for r in subset if r[f"control__{m}"] != ""]
            level.append(f"{k * sum(vals) / len(vals):.2f}")
        lines += ["", "Control level: " + ", ".join(f"{h.split(' (')[0]} {v}" for h, v in zip(heads, level)), ""]
    (HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── manipulation check ──

def cmd_judge(args) -> int:
    """Blind strategy judge (3-pass majority, not the model that wrote the rewrites) on a random sample of delivered rewrites per
    setting, and on the original wording of the same utterances."""
    from components.judges.blind_politeness_classifier import BlindPolitenessClassifier
    rng = random.Random(0)
    by_setting = defaultdict(list)
    for r in csv.DictReader(open(DATA / "treated_utterances.csv", encoding="utf-8")):
        if r["delivered"] == "rewrite":
            by_setting[r["setting"]].append(r)
    jobs = [(s, kind, r) for s, rs in sorted(by_setting.items()) for r in rng.sample(rs, min(args.n, len(rs)))
            for kind in ("delivered", "original")]
    model = JudgeModel.resolve(LUNA)
    ok, message = model.preflight()
    if not ok:
        print("preflight failed:", message)
        return 1
    judge = BlindPolitenessClassifier(model)

    def one(job):
        setting, kind, r = job
        text = r["delivered_text"] if kind == "delivered" else r["original_text"]
        labels = [judge.classify(text, speaker_id=r["speaker"]).label for _ in range(3)]
        return {"setting": setting, "kind": kind, "assigned": r["assigned_strategy"], "speech_act": r["speech_act"],
                "label": majority(labels, passes=3)[0], "labels": labels}
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(one, jobs))
    with open(DATA / "manipulation_check.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in results)
    table = defaultdict(Counter)
    for r in results:
        table[(r["setting"], r["kind"])][r["label"]] += 1
    for (setting, kind), counts in sorted(table.items()):
        print(setting, kind, dict(counts))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    export = sub.add_parser("export")
    export.add_argument("--games", default=str(DEFAULT_GAMES))
    sub.add_parser("effects")
    judge = sub.add_parser("judge")
    judge.add_argument("--n", type=int, default=150, help="rewrites sampled per setting")
    args = parser.parse_args(argv)
    return {"export": cmd_export, "effects": cmd_effects, "judge": cmd_judge}[args.cmd](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
