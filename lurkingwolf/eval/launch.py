"""CLI for the per-phase evaluation.

Two entry paths:
  1. python -m lurkingwolf.eval --config examples/formality/config.yaml --seed 101
  2. python -m lurkingwolf.eval --mode pred_vs_nl  (quick single-comparison)

Examples:

    # Full experiment from YAML config
    python -m lurkingwolf.eval --config examples/formality/config.yaml --seed 101

    # Quick pred_vs_nl comparison
    python -m lurkingwolf.eval --mode pred_vs_nl --seed 101
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from ..config import ExperimentConfig, ComparisonSpec, ReplaySpec, load_config
from .runner import run_experiment


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-phase evaluation: baseline + phase replays.",
    )
    parser.add_argument("--config", default=None,
                        help="Path to YAML experiment config.")
    parser.add_argument("--mode", default=None,
                        choices=("pred_vs_nl",),
                        help="Quick single-comparison mode (alternative to --config)")
    parser.add_argument("--seed", default="101",
                        help="Random seed (default 101)")
    parser.add_argument("--players", type=int, default=7,
                        help="Number of players (default 7)")
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="Max in-flight replay subprocesses")

    # Engine knobs
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--token-budget", type=int, default=None)

    # Backend
    parser.add_argument("--url", default=None,
                        help="Model server URL (default: $LLM_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="Model server API key (default: $LLM_API_KEY)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")

    args = parser.parse_args(argv)

    if not args.config and not args.mode:
        print("error: provide --config <path.yaml> or --mode pred_vs_nl",
              file=sys.stderr)
        return 2

    output_root = None
    if args.config:
        cfg = load_config(args.config)
        config_dir = Path(args.config).resolve().parent
        model_slug = cfg.model.rsplit("/", 1)[-1].replace("@", "").replace(" ", "_").lower()
        output_root = config_dir / "games" / f"run_{model_slug}_{cfg.name}_{args.seed}"
    elif args.mode == "pred_vs_nl":
        cfg = ExperimentConfig(
            name="pred_vs_nl",
            n_players=args.players,
            seeds=[args.seed],
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_rounds=args.max_rounds,
            comparisons=OrderedDict([
                ("pred_vs_nl", ComparisonSpec(
                    baseline="pred_only",
                    replays=[ReplaySpec(mode="nl")],
                )),
            ]),
        )
    else:
        print(f"error: unknown mode {args.mode!r}", file=sys.stderr)
        return 2

    try:
        run_experiment(
            cfg,
            seed=args.seed,
            url=args.url,
            api_key=args.api_key,
            max_concurrency=args.max_concurrency,
            token_budget=args.token_budget,
            output_root=output_root,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
