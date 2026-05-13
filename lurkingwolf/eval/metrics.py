"""Per-phase metrics: compute deltas between baseline and replay results.

Three signal types, matching eval/ naming:
  - s1 (skill / night actions): did wolf/seer/guard change target?
  - s2 (votes): did each voter change their vote? did vote-received change?
  - s3 (reliability): how did each player's trust scores shift?

Grouping dimensions: config, self_player, self_role, day, voter_role,
target_role.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .replay import BaselinePhaseData, PhaseResult, extract_baseline_phase, count_baseline_days


# ─── Per-phase delta ───────────────────────────────────────────────────

SKILL_SLOTS = [
    ("wolf_target", "Werewolf"),
    ("seer_target", "Seer"),
    ("guard_target", "Guard"),
]


def compute_phase_delta(
    baseline: BaselinePhaseData,
    replay: PhaseResult,
) -> dict:
    """Compute deltas between baseline and replay for one phase.

    Returns a dict with delta_skill, delta_vote, delta_reliability, plus
    summary aggregates.
    """
    role_dist = baseline.role_dist

    # ── s1: skill deltas ──
    delta_skill: List[dict] = []
    for attr, role in SKILL_SLOTS:
        b_target = getattr(baseline, attr)
        r_target = getattr(replay, attr)
        if b_target is None and r_target is None:
            continue
        delta_skill.append({
            "slot": attr,
            "role": role,
            "baseline": b_target,
            "replay": r_target,
            "changed": int(b_target != r_target),
        })

    # ── s2: vote deltas ──
    delta_vote: List[dict] = []
    all_voters = sorted(set(baseline.votes) | set(replay.votes))
    for voter in all_voters:
        b_target = baseline.votes.get(voter)
        r_target = replay.votes.get(voter)
        voter_role = role_dist.get(voter, "unknown")
        delta_vote.append({
            "voter": voter,
            "baseline": b_target,
            "replay": r_target,
            "changed": int(b_target != r_target),
        })

    # Vote-received deltas (per target)
    delta_votes_received: List[dict] = []
    all_targets = sorted(
        set(baseline.votes_received) | set(replay.votes_received)
    )
    for target in all_targets:
        b_count = baseline.votes_received.get(target, 0)
        r_count = replay.votes_received.get(target, 0)
        target_role = role_dist.get(target, "unknown")
        delta_votes_received.append({
            "target": target,
            "baseline": b_count,
            "replay": r_count,
            "delta": r_count - b_count,
        })

    # ── s3: reliability deltas ──
    delta_reliability: List[dict] = []
    all_assessors = sorted(set(baseline.reliability) | set(replay.reliability))
    for assessor in all_assessors:
        b_scores = baseline.reliability.get(assessor, {})
        r_scores = replay.reliability.get(assessor, {})
        all_targets_r = sorted(set(b_scores) | set(r_scores))
        for target in all_targets_r:
            b_val = b_scores.get(target)
            r_val = r_scores.get(target)
            if b_val is None and r_val is None:
                continue
            delta_reliability.append({
                "assessor": assessor,
                "target": target,
                "baseline": b_val,
                "replay": r_val,
                "delta": (r_val - b_val) if (b_val is not None and r_val is not None) else None,
            })

    # Execution delta
    execution_changed = int(baseline.executed != replay.executed)

    n_skill = len(delta_skill)
    n_vote = len(delta_vote)
    n_rel = len(delta_reliability)

    return {
        "day": baseline.day,
        "config_name": replay.config_name,
        "self_player": replay.self_player,
        "self_role": role_dist.get(replay.self_player, "n/a") if replay.self_player else "n/a",
        "experiment_type": replay.experiment_type,
        "role_dist": dict(sorted(role_dist.items())),
        "delta_skill": delta_skill,
        "delta_vote": delta_vote,
        "delta_votes_received": delta_votes_received,
        "delta_reliability": delta_reliability,
        "execution_changed": execution_changed,
        "baseline_executed": baseline.executed,
        "replay_executed": replay.executed,
        "aggregates": {
            "skill_change_rate": sum(r["changed"] for r in delta_skill) / n_skill if n_skill else 0.0,
            "vote_change_rate": sum(r["changed"] for r in delta_vote) / n_vote if n_vote else 0.0,
            "reliability_change_rate": sum(
                1 for r in delta_reliability if r.get("delta") is not None and r["delta"] != 0
            ) / n_rel if n_rel else 0.0,
            "mean_reliability_delta": statistics.mean(
                [r["delta"] for r in delta_reliability if r["delta"] is not None]
            ) if any(r["delta"] is not None for r in delta_reliability) else 0.0,
        },
    }


# ─── Sweep-level aggregation ──────────────────────────────────────────

def aggregate_phase_deltas(output_root: Path, baseline_dir: Optional[Path] = None) -> dict:
    """Walk output_root/replays/ and compute deltas vs baseline.

    Returns {
        "per_phase": [...],      # one delta dict per replay
        "by_config": {...},      # grouped stats
        "by_self_role": {...},
        "by_day": {...},
        "by_config_role": {...},
    }
    """
    output_root = Path(output_root)
    if baseline_dir is None:
        baseline_dir = output_root / "baseline"
    baseline_dir = Path(baseline_dir)

    if not baseline_dir.exists():
        return {"error": f"No baseline directory found at {baseline_dir}"}

    num_days = count_baseline_days(baseline_dir)
    baseline_cache: Dict[int, BaselinePhaseData] = {}
    for day in range(1, num_days + 1):
        baseline_cache[day] = extract_baseline_phase(baseline_dir, day)

    replays_dir = output_root / "replays"
    if not replays_dir.exists():
        return {"error": "No replays directory found"}

    per_phase: List[dict] = []
    for day_dir in sorted(replays_dir.iterdir()):
        if not day_dir.is_dir() or not day_dir.name.startswith("day_"):
            continue
        try:
            day = int(day_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if day not in baseline_cache:
            continue
        baseline = baseline_cache[day]

        for run_dir in sorted(day_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            result_path = run_dir / "phase_result.json"
            if not result_path.exists():
                continue
            try:
                result_data = json.loads(result_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            replay = PhaseResult(
                day=result_data["day"],
                config_name=result_data["config_name"],
                self_player=result_data.get("self_player"),
                experiment_type=result_data["experiment_type"],
                role_dist=result_data["role_dist"],
                votes=result_data["votes"],
                vote_counts=result_data.get("vote_counts", {}),
                executed=result_data.get("executed"),
                votes_received=result_data.get("votes_received", {}),
                wolf_target=result_data.get("wolf_target"),
                seer_target=result_data.get("seer_target"),
                guard_target=result_data.get("guard_target"),
                witch_save=result_data.get("witch_save", False),
                witch_poison=result_data.get("witch_poison"),
                night_killed=result_data.get("night_killed"),
                reliability=result_data.get("reliability", {}),
                speeches=result_data.get("speeches", []),
            )
            delta = compute_phase_delta(baseline, replay)
            per_phase.append(delta)

    # Group by various dimensions
    by_config = _group_deltas(per_phase, "config_name")
    by_self_role = _group_deltas(per_phase, "self_role")
    by_day = _group_deltas(per_phase, "day")
    by_config_role = {}
    for delta in per_phase:
        key = f"{delta['config_name']}|{delta['self_role']}"
        by_config_role.setdefault(key, []).append(delta)
    by_config_role = {k: _summarize_group(v) for k, v in by_config_role.items()}

    return {
        "num_baseline_days": num_days,
        "num_replays": len(per_phase),
        "per_phase": per_phase,
        "by_config": by_config,
        "by_self_role": by_self_role,
        "by_day": by_day,
        "by_config_role": by_config_role,
    }


def _group_deltas(deltas: List[dict], key: str) -> Dict[str, dict]:
    groups: Dict[str, List[dict]] = {}
    for d in deltas:
        val = str(d.get(key, "unknown"))
        groups.setdefault(val, []).append(d)
    return {k: _summarize_group(v) for k, v in groups.items()}


def _summarize_group(deltas: List[dict]) -> dict:
    """Aggregate rates across a group of phase deltas."""
    n = len(deltas)
    if n == 0:
        return {"n": 0}

    skill_rates = [d["aggregates"]["skill_change_rate"] for d in deltas]
    vote_rates = [d["aggregates"]["vote_change_rate"] for d in deltas]
    rel_rates = [d["aggregates"]["reliability_change_rate"] for d in deltas]
    exec_changes = [d["execution_changed"] for d in deltas]

    return {
        "n": n,
        "mean_skill_change_rate": statistics.mean(skill_rates),
        "mean_vote_change_rate": statistics.mean(vote_rates),
        "mean_reliability_change_rate": statistics.mean(rel_rates),
        "execution_change_rate": sum(exec_changes) / n,
        "stdev_skill": statistics.stdev(skill_rates) if n > 1 else 0.0,
        "stdev_vote": statistics.stdev(vote_rates) if n > 1 else 0.0,
        "stdev_reliability": statistics.stdev(rel_rates) if n > 1 else 0.0,
    }


# ─── Compact serialization ───────────────────────────────────────────

def _compact_metrics_json(data: dict) -> str:
    """Compact phase_metrics.json — per_phase entries on minimal lines."""
    lines = ["{"]
    top_keys = list(data.keys())
    for ki, key in enumerate(top_keys):
        comma = "," if ki < len(top_keys) - 1 else ""
        val = data[key]
        if key == "per_phase":
            lines.append('  "per_phase": [')
            for pi, phase in enumerate(val):
                pc = "," if pi < len(val) - 1 else ""
                # Each per_phase entry: scalars on first line, lists inline
                parts = ["{"]
                pkeys = list(phase.keys())
                for pki, pk in enumerate(pkeys):
                    pkc = "," if pki < len(pkeys) - 1 else ""
                    pv = phase[pk]
                    if pk in ("delta_skill", "delta_vote", "delta_votes_received", "delta_reliability"):
                        inner = ", ".join(json.dumps(r, ensure_ascii=False) for r in pv)
                        parts.append(f"    {json.dumps(pk)}: [{inner}]{pkc}")
                    elif isinstance(pv, dict):
                        parts.append(f"    {json.dumps(pk)}: {json.dumps(pv, ensure_ascii=False)}{pkc}")
                    else:
                        parts.append(f"    {json.dumps(pk)}: {json.dumps(pv, ensure_ascii=False)}{pkc}")
                parts.append(f"  }}{pc}")
                lines.append("  " + parts[0])
                lines.extend(parts[1:])
            lines.append(f"  ]{comma}")
        elif isinstance(val, dict):
            inner = json.dumps(val, ensure_ascii=False)
            lines.append(f"  {json.dumps(key)}: {inner}{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(val, ensure_ascii=False)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ─── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute per-phase metrics from eval output."
    )
    parser.add_argument("output_root",
                        help="eval output root (contains baseline/ + replays/)")
    parser.add_argument("--baseline-from", default=None,
                        help="Use baseline from this directory instead of <output_root>/baseline/")
    parser.add_argument("--out", default=None,
                        help="Output JSON (default: <output_root>/phase_metrics.json)")
    args = parser.parse_args()

    baseline = Path(args.baseline_from) if args.baseline_from else None
    result = aggregate_phase_deltas(Path(args.output_root), baseline_dir=baseline)
    out_path = Path(args.out) if args.out else Path(args.output_root) / "phase_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_compact_metrics_json(result))
    print(f"Wrote {out_path} ({result.get('num_replays', 0)} replays)")


if __name__ == "__main__":
    main()
