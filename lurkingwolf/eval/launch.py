"""CLI for the per-phase evaluation.

  pred_vs_nl:  Does surface form matter?
    Baseline = pred-only game.  Replays = NL baseline (1 per day).

Examples:

    # predicate-only vs NL
    python -m lurkingwolf.eval --mode pred_vs_nl \\
        --vllm --vllm-url http://localhost:8080/ \\
        --vllm-model Qwen/Qwen3.5-27B

    # List replay plan without running
    python -m lurkingwolf.eval --mode pred_vs_nl --list
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .configs import BASELINE_EXPERIMENT, configs_for_mode, expand_phase_runs
from .runner import (
    PROJECT_ROOT, SubprocessArgs,
    run_baseline, run_replays,
)
from .replay import count_baseline_days, list_day_checkpoints


def _parse_self_players(spec: str, num_players: int) -> List[str]:
    spec = spec.strip()
    if spec.lower() == "all":
        return [f"P{i}" for i in range(1, num_players + 1)]
    return [tok.strip() for tok in spec.split(",") if tok.strip()]


def _model_short(model: Optional[str]) -> str:
    if not model:
        return "default"
    return model.split("/")[-1].replace(":", "_").lower()


def _normalize_mode(mode: str) -> str:
    return mode


def _default_output_root(mode: str, seed: str, model: Optional[str]) -> Path:
    tag = mode
    ts = datetime.now().strftime("%m%d_%H%M")
    return PROJECT_ROOT / "lurkingwolf" / "games" / f"eval_{_model_short(model)}_{tag}_{seed}_{ts}"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-phase evaluation: baseline + phase replays.",
    )
    parser.add_argument("--mode", required=True,
                        choices=("pred_vs_nl",),
                        help="pred_vs_nl")
    parser.add_argument("--self", default="all", dest="self_players",
                        help="Self player(s): 'all', 'P1', or comma list 'P1,P3'")
    parser.add_argument("--players", type=int, default=7,
                        help="Number of players per game (default 7)")
    parser.add_argument("--seed", default="101",
                        help="Fixed random seed (default 101)")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="Max in-flight replay subprocesses")
    parser.add_argument("--list", action="store_true",
                        help="Print the replay plan and exit")
    parser.add_argument("--days", default=None,
                        help="Which days to replay: 'all' (default), '1', '1,2'")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline if already run (reuse existing)")
    parser.add_argument("--baseline-from", default=None,
                        help="Reuse baseline from this directory instead of running one")

    # Engine knobs
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")

    # Backend
    parser.add_argument("--url", default=None,
                        help="Model server URL (default: $VLLM_BASE_URL or http://localhost:8000)")
    parser.add_argument("--api-key", default=None,
                        help="Model server API key (default: $VLLM_API_KEY or EMPTY)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")

    # Subset selection
    parser.add_argument("--config", default=None, nargs="+",
                        help="Only run configs whose name contains any of these substrings.")

    args = parser.parse_args(argv)

    self_players = _parse_self_players(args.self_players, args.players)
    valid_players = {f"P{i}" for i in range(1, args.players + 1)}
    bad = [p for p in self_players if p not in valid_players]
    if bad:
        print(f"error: invalid self player(s) {bad}", file=sys.stderr)
        return 2

    mode = _normalize_mode(args.mode)
    configs = configs_for_mode(mode)
    if args.config:
        subs = list(args.config)
        configs = [c for c in configs if any(sub in c.name for sub in subs)]
        if not configs:
            print(f"error: no configs match --config {subs!r}", file=sys.stderr)
            return 2

    output_root = (
        Path(args.output_root) if args.output_root
        else _default_output_root(mode, args.seed, args.model)
    )

    sub = SubprocessArgs(
        url=args.url,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_rounds=args.max_rounds,
        num_players=args.players,
        token_budget=args.token_budget,
        enable_thinking=args.enable_thinking,
        seed=args.seed,
    )

    # ── Step 1: Baseline ──────────────────────────────────────────
    baseline_experiment = BASELINE_EXPERIMENT[mode]
    baseline_dir = output_root / "baseline"
    if args.baseline_from:
        baseline_dir = Path(args.baseline_from)
        if not (baseline_dir / "game_record.json").exists():
            print(f"error: no game_record.json in --baseline-from {baseline_dir}",
                  file=sys.stderr)
            return 1
        print(f"Reusing baseline from: {baseline_dir}/")
    elif args.skip_baseline and baseline_dir.exists() and (baseline_dir / "game_record.json").exists():
        print(f"Reusing existing baseline: {baseline_dir}/")
    elif not args.list:
        print(f"Output root: {output_root}/")
        print(f"Baseline experiment: {baseline_experiment}\n")
        baseline_dir = run_baseline(output_root, sub,
                                    experiment_type=baseline_experiment)
    else:
        print("(baseline would be run here)")

    # Determine available days
    if args.list:
        days_list = [1, 2, 3]  # placeholder for --list
    else:
        num_days = count_baseline_days(baseline_dir)
        checkpoints = list_day_checkpoints(baseline_dir)
        available_days = []
        for ckpt in checkpoints:
            name = ckpt.stem  # e.g. "day_2_start"
            parts = name.split("_")
            try:
                available_days.append(int(parts[1]))
            except (IndexError, ValueError):
                continue

        if not available_days:
            print("error: no day-start checkpoints found. "
                  "Ensure run_game.py saves them.", file=sys.stderr)
            return 1

        if args.days and args.days.lower() != "all":
            requested = [int(d.strip()) for d in args.days.split(",")]
            days_list = [d for d in requested if d in available_days]
        else:
            days_list = available_days

        print(f"Baseline has {num_days} days, {len(available_days)} checkpoints available.")
        print(f"Replaying days: {days_list}")

    # ── Step 2: Expand replay runs ────────────────────────────────
    runs = expand_phase_runs(configs, self_players, days_list)

    if args.list:
        print(f"\n# {len(runs)} replay(s) — mode={mode}, self={self_players}, days={days_list}")
        for s in runs:
            print(
                f"  {s.run_id:<56}  exp={s.config.experiment_type:<11}  "
                f"self={s.self_player or '-'}  day={s.day}"
            )
        return 0

    print(f"\n{len(runs)} phase replays across {len(configs)} configs × "
          f"{len(self_players)} selfs × {len(days_list)} days\n")

    results = run_replays(
        runs=runs,
        baseline_dir=baseline_dir,
        output_root=output_root,
        sub=sub,
        max_concurrency=args.max_concurrency,
    )

    failed = [r for r in results if r.returncode != 0 or r.error]

    # ── Post-processing: metrics + plots ──────────────────────────
    succeeded = [r for r in results if r.returncode == 0 and r.error is None]
    if succeeded:
        print("\nComputing per-phase metrics…")
        try:
            from .metrics import aggregate_phase_deltas, _compact_metrics_json
            metrics_baseline = Path(args.baseline_from) if args.baseline_from else None
            metrics = aggregate_phase_deltas(output_root, baseline_dir=metrics_baseline)
            metrics_path = output_root / "phase_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                f.write(_compact_metrics_json(metrics))
            print(f"  phase_metrics.json — {metrics.get('num_replays', 0)} replays")
        except Exception as exc:
            print(f"  metrics failed: {exc}", file=sys.stderr)

        print("Rendering figures…")
        try:
            from .plots import render
            pngs = render(output_root)
            for p in pngs:
                print(f"  {p.name}")
            print(f"  {len(pngs)} figure(s) → {output_root}/graphs/")
        except Exception as exc:
            print(f"  plots failed: {exc}", file=sys.stderr)

    if failed:
        print(f"\n{len(failed)} replay(s) failed:")
        for r in failed[:10]:
            print(f"  {r.run_id}: {r.error or f'rc={r.returncode}'}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
