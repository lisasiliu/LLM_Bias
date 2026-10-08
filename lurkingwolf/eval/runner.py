"""Per-phase evaluation runner.

Two-step process per comparison:
  1. Run a baseline game (pred-only or NL) with ``--save-checkpoints``.
  2. For each replay spec × self_player × day, replay the phase.

``run_experiment()`` orchestrates all comparisons from a single config.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config import ExperimentConfig, ReplaySpec


_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent

def _python() -> str:
    return sys.executable


# ─── Subprocess args (shared across baseline + replays) ────────────────

@dataclass
class SubprocessArgs:
    url: Optional[str] = None
    api_key: Optional[str] = None
    model: str = "Qwen/Qwen3.5-27B"
    temperature: float = 0.0
    max_tokens: int = 300
    max_rounds: int = 20
    num_players: int = 7
    token_budget: Optional[int] = None
    enable_thinking: bool = False
    seed: str = "101"
    extra_env: Optional[Dict[str, str]] = None


# ─── Step 1: Run baseline game ────────────────────────────────────────

def run_baseline(
    output_root: Path,
    sub: SubprocessArgs,
    *,
    experiment_type: str = "nl",
    rewriter_module: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    baseline_dir = output_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    if (baseline_dir / "game_record.json").exists():
        if verbose:
            print(f"  Reusing existing baseline: {baseline_dir}/")
        return baseline_dir

    cmd: List[str] = [
        _python(), "-u", "-m", "lurkingwolf.run_game",
        "--experiment", experiment_type,
        "--players", str(sub.num_players),
        "--games", "1",
        "--game-id", "g",
        "--seed", sub.seed,
        "--temperature", str(sub.temperature),
        "--max-tokens", str(sub.max_tokens),
        "--max-rounds", str(sub.max_rounds),
        "--output", str(baseline_dir.resolve()),
        "--save-checkpoints",
        "--verbose",
    ]
    if sub.token_budget is not None:
        cmd += ["--token-budget", str(sub.token_budget)]
    cmd += ["--model", sub.model]
    if sub.url:
        cmd += ["--url", sub.url]
    if sub.api_key:
        cmd += ["--api-key", sub.api_key]
    if rewriter_module:
        cmd += ["--rewriter-module", rewriter_module]

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if sub.extra_env:
        env.update(sub.extra_env)

    log_path = baseline_dir / "subprocess.log"
    if verbose:
        print(f"  Running baseline ({experiment_type}) → {baseline_dir}/")
        print(f"    cmd: {' '.join(cmd[:6])} ...")

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=logf, stderr=subprocess.STDOUT,
            env=env, check=False,
        )
    elapsed = time.time() - t0

    if verbose:
        status = "ok" if proc.returncode == 0 else "FAIL"
        print(f"  Baseline {status} ({elapsed:.1f}s, rc={proc.returncode})")

    if proc.returncode != 0:
        raise RuntimeError(
            f"Baseline game failed (rc={proc.returncode}). "
            f"See {log_path}"
        )
    return baseline_dir


# ─── Step 2: Phase replays ────────────────────────────────────────────

@dataclass(frozen=True)
class _RunSpec:
    replay: ReplaySpec
    self_player: Optional[str]
    day: int
    run_id: str


def _alive_at_day_start(baseline_dir: Path, day: int) -> Optional[set]:
    """Players alive when the day-start checkpoint was taken; None when the checkpoint cannot be read."""
    import pickle
    path = baseline_dir / "checkpoints" / f"day_{day}_start.pkl"
    try:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        return set(ckpt.state.get_alive())
    except Exception:
        return None


def _skip_dead_self_players(runs: List["_RunSpec"], baseline_dir: Path) -> List["_RunSpec"]:
    """Drop self-dependent replays whose self player is dead at the day start: with no speech to
    rewrite they repeat the identity replay."""
    alive = {day: _alive_at_day_start(baseline_dir, day) for day in {r.day for r in runs}}
    kept = [r for r in runs if r.self_player is None or alive[r.day] is None or r.self_player in alive[r.day]]
    skipped = len(runs) - len(kept)
    if skipped:
        print(f"  Skipping {skipped} replays whose self player is dead at the day start")
    return kept


def _expand_runs(
    replays: List[ReplaySpec],
    n_players: int,
    days: List[int],
) -> List[_RunSpec]:
    runs: List[_RunSpec] = []
    players = [f"P{i}" for i in range(1, n_players + 1)]
    for replay in replays:
        if replay.self_dependent:
            for p in players:
                for d in days:
                    run_id = f"{replay.run_tag}__{p.lower()}__d{d}"
                    runs.append(_RunSpec(replay=replay, self_player=p, day=d, run_id=run_id))
        else:
            for d in days:
                run_id = f"{replay.run_tag}__d{d}"
                runs.append(_RunSpec(replay=replay, self_player=None, day=d, run_id=run_id))
    return runs


def _build_replay_command_from_spec(
    spec: _RunSpec,
    checkpoint_path: Path,
    out_path: Path,
    sub: SubprocessArgs,
    *,
    rewriter_module: Optional[str] = None,
) -> List[str]:
    cmd: List[str] = [
        _python(), "-u", "-m", "lurkingwolf.eval.replay_cli",
        "--checkpoint", str(checkpoint_path.resolve()),
        "--experiment", "nl" if spec.replay.mode != "pred_only" else "pred_only",
        "--config-name", spec.run_id,
        "--output", str(out_path.resolve()),
    ]
    if spec.self_player:
        cmd += ["--self-player", spec.self_player]
    if sub.url:
        cmd += ["--vllm-url", sub.url]
    if sub.api_key:
        cmd += ["--vllm-key", sub.api_key]
    if sub.model:
        cmd += ["--vllm-model", sub.model]
    if spec.replay.rewriter_class:
        cmd += ["--rewriter", spec.replay.rewriter_class]
        if spec.replay.rewriter_params:
            cmd += ["--rewriter-params", json.dumps(spec.replay.rewriter_params)]
        cmd += ["--rewriter-scope", spec.replay.scope]
    if rewriter_module:
        cmd += ["--rewriter-module", rewriter_module]
    return cmd


@dataclass
class ReplayResult:
    run_id: str
    self_player: Optional[str]
    day: int
    run_dir: Path
    returncode: int
    elapsed_seconds: float
    error: Optional[str] = None


def _run_one(
    spec: _RunSpec,
    baseline_dir: Path,
    comp_root: Path,
    sub: SubprocessArgs,
    *,
    rewriter_module: Optional[str] = None,
) -> ReplayResult:
    run_dir = comp_root / "replays" / f"day_{spec.day}" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "phase_result.json"
    if result_path.exists() and result_path.stat().st_size > 0:
        return ReplayResult(
            run_id=spec.run_id, self_player=spec.self_player,
            day=spec.day, run_dir=run_dir, returncode=0,
            elapsed_seconds=0.0, error=None,
        )

    ckpt_path = baseline_dir / "checkpoints" / f"day_{spec.day}_start.pkl"
    if not ckpt_path.exists():
        return ReplayResult(
            run_id=spec.run_id, self_player=spec.self_player,
            day=spec.day, run_dir=run_dir, returncode=-1,
            elapsed_seconds=0.0, error=f"Checkpoint not found: {ckpt_path}",
        )

    out_path = run_dir / "phase_result.json"
    cmd = _build_replay_command_from_spec(
        spec, ckpt_path, out_path, sub, rewriter_module=rewriter_module,
    )

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if sub.extra_env:
        env.update(sub.extra_env)

    log_path = run_dir / "subprocess.log"
    t0 = time.time()
    error: Optional[str] = None
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write("$ " + " ".join(cmd) + "\n\n")
            logf.flush()
            proc = subprocess.run(
                cmd, cwd=str(PROJECT_ROOT),
                stdout=logf, stderr=subprocess.STDOUT,
                env=env, check=False,
            )
        returncode = proc.returncode
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        returncode = -1

    elapsed = time.time() - t0
    return ReplayResult(
        run_id=spec.run_id, self_player=spec.self_player,
        day=spec.day, run_dir=run_dir, returncode=returncode,
        elapsed_seconds=elapsed, error=error,
    )


def _run_replays(
    runs: List[_RunSpec],
    baseline_dir: Path,
    comp_root: Path,
    sub: SubprocessArgs,
    *,
    rewriter_module: Optional[str] = None,
    max_concurrency: int = 4,
) -> List[ReplayResult]:
    results: List[ReplayResult] = []
    total = len(runs)
    completed = 0

    print(f"  {total} phase replays (concurrency={max_concurrency})...")

    with cf.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_spec = {
            pool.submit(
                _run_one, spec, baseline_dir, comp_root, sub,
                rewriter_module=rewriter_module,
            ): spec
            for spec in runs
        }
        for fut in cf.as_completed(future_to_spec):
            spec = future_to_spec[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = ReplayResult(
                    run_id=spec.run_id, self_player=spec.self_player,
                    day=spec.day,
                    run_dir=comp_root / "replays" / f"day_{spec.day}" / spec.run_id,
                    returncode=-1, elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(res)
            completed += 1
            if res.returncode == 0 and res.error is None and res.elapsed_seconds == 0.0:
                status = "skip"
            elif res.returncode == 0 and res.error is None:
                status = "ok"
            else:
                status = "FAIL"
            print(
                f"    [{completed:>3}/{total}] {status:<4} "
                f"{res.run_id:<48} "
                f"({res.elapsed_seconds:5.1f}s)",
                flush=True,
            )

    return results


# ─── Unified experiment orchestrator ─────────────────────────────────

def run_experiment(
    cfg: ExperimentConfig,
    *,
    seed: Optional[str] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_concurrency: int = 4,
    token_budget: Optional[int] = None,
    output_root: Optional[Path] = None,
) -> Path:
    """Run all comparisons in a config for one seed.

    Returns the experiment root directory.
    """
    from .replay import list_day_checkpoints

    seed = seed or cfg.seeds[0]
    if output_root is None:
        model_slug = cfg.model.rsplit("/", 1)[-1].replace("@", "").replace(" ", "_").lower()
        output_root = PROJECT_ROOT / "experiments" / f"run_{model_slug}_{cfg.name}_{seed}"
    root = output_root
    root.mkdir(parents=True, exist_ok=True)

    sub = SubprocessArgs(
        url=url,
        api_key=api_key,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        max_rounds=cfg.max_rounds,
        num_players=cfg.n_players,
        token_budget=token_budget,
        seed=seed,
    )

    all_results: List[ReplayResult] = []

    for comp_name, comp in cfg.comparisons.items():
        comp_root = root / comp_name
        comp_root.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Comparison: {comp_name}  (baseline={comp.baseline})")
        print(f"{'='*60}")

        # Step 1: baseline game
        baseline_dir = run_baseline(
            comp_root, sub,
            experiment_type=comp.baseline,
            rewriter_module=cfg.rewriter_module,
        )

        # Determine available days
        checkpoints = list_day_checkpoints(baseline_dir)
        days: List[int] = []
        for ckpt in checkpoints:
            parts = ckpt.stem.split("_")
            try:
                days.append(int(parts[1]))
            except (IndexError, ValueError):
                continue

        if not days:
            print(f"  WARNING: no day-start checkpoints found in {baseline_dir}")
            continue

        print(f"  Checkpoints: days {days}")

        # Step 2: expand + run replays
        if not comp.replays:
            print(f"  No replays for {comp_name}")
            continue

        runs = _skip_dead_self_players(_expand_runs(comp.replays, cfg.n_players, days), baseline_dir)
        results = _run_replays(
            runs, baseline_dir, comp_root, sub,
            rewriter_module=cfg.rewriter_module,
            max_concurrency=max_concurrency,
        )
        all_results.extend(results)

        # Per-comparison summary
        ok = sum(1 for r in results if r.returncode == 0 and r.error is None)
        fail = sum(1 for r in results if r.returncode != 0 or r.error is not None)
        print(f"  {comp_name}: {ok}/{len(results)} ok, {fail} failed")

    # Overall summary
    summary = {
        "config": cfg.name,
        "seed": seed,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "comparisons": list(cfg.comparisons.keys()),
        "total_replays": len(all_results),
        "succeeded": sum(1 for r in all_results if r.returncode == 0 and r.error is None),
        "failed": sum(1 for r in all_results if r.returncode != 0 or r.error is not None),
        "results": [
            {
                "run_id": r.run_id,
                "self": r.self_player,
                "day": r.day,
                "rc": r.returncode,
                "elapsed_seconds": r.elapsed_seconds,
                "error": r.error,
            }
            for r in all_results
        ],
    }
    with open(root / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    total_fail = summary["failed"]
    print(f"\nExperiment complete → {root}/")
    print(f"  {summary['succeeded']}/{summary['total_replays']} replays ok"
          + (f", {total_fail} failed" if total_fail else ""))

    return root
