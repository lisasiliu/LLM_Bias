"""Per-phase evaluation runner.

Two-step process:
  1. Run a baseline game (pred-only or NL) with ``--save-checkpoints``.
     This produces day-start checkpoints at each day.
  2. For each (config, self_player, day), load the checkpoint and replay
     the phase with modified settings (e.g. a different rewriter).
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

from .configs import PhaseConfig, PhaseRunSpec, configs_for_mode, expand_phase_runs


_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent

_WEREWOLF_ENV = Path("/scratch/ll5937/envs/werewolf/bin/python")


def _python() -> str:
    if _WEREWOLF_ENV.exists():
        return str(_WEREWOLF_ENV)
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
    verbose: bool = True,
) -> Path:
    """Run a single game with checkpoints enabled.

    Returns the baseline game directory.
    """
    baseline_dir = output_root / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

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

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if sub.extra_env:
        env.update(sub.extra_env)

    log_path = baseline_dir / "subprocess.log"
    if verbose:
        print(f"Running baseline game → {baseline_dir}/")
        print(f"  cmd: {' '.join(cmd[:6])} ...")

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

def _build_replay_command(
    spec: PhaseRunSpec,
    checkpoint_path: Path,
    out_path: Path,
    sub: SubprocessArgs,
) -> List[str]:
    """Build the subprocess command for one phase replay."""
    cmd: List[str] = [
        _python(), "-u", "-m", "lurkingwolf.eval.replay_cli",
        "--checkpoint", str(checkpoint_path.resolve()),
        "--experiment", spec.config.experiment_type,
        "--config-name", spec.config.name,
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
    if spec.config.rewriter_factory is not None:
        cmd += ["--rewriter", spec.config.rewriter_factory.name]
        if spec.config.rewriter_params:
            cmd += ["--rewriter-params", json.dumps(spec.config.rewriter_params)]
        cmd += ["--rewriter-scope", spec.config.scope]
    return cmd


@dataclass
class ReplayResult:
    run_id: str
    config_name: str
    self_player: Optional[str]
    day: int
    run_dir: Path
    returncode: int
    elapsed_seconds: float
    error: Optional[str] = None


def _run_one_replay(
    spec: PhaseRunSpec,
    baseline_dir: Path,
    output_root: Path,
    sub: SubprocessArgs,
) -> ReplayResult:
    """Execute one phase replay as a subprocess."""
    run_dir = output_root / "replays" / f"day_{spec.day}" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    result_path = run_dir / "phase_result.json"
    if result_path.exists() and result_path.stat().st_size > 0:
        return ReplayResult(
            run_id=spec.run_id,
            config_name=spec.config.name,
            self_player=spec.self_player,
            day=spec.day,
            run_dir=run_dir,
            returncode=0,
            elapsed_seconds=0.0,
            error=None,
        )

    ckpt_path = baseline_dir / "checkpoints" / f"day_{spec.day}_start.pkl"
    if not ckpt_path.exists():
        return ReplayResult(
            run_id=spec.run_id,
            config_name=spec.config.name,
            self_player=spec.self_player,
            day=spec.day,
            run_dir=run_dir,
            returncode=-1,
            elapsed_seconds=0.0,
            error=f"Checkpoint not found: {ckpt_path}",
        )

    out_path = run_dir / "phase_result.json"
    cmd = _build_replay_command(spec, ckpt_path, out_path, sub)

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
        run_id=spec.run_id,
        config_name=spec.config.name,
        self_player=spec.self_player,
        day=spec.day,
        run_dir=run_dir,
        returncode=returncode,
        elapsed_seconds=elapsed,
        error=error,
    )


def run_replays(
    *,
    runs: List[PhaseRunSpec],
    baseline_dir: Path,
    output_root: Path,
    sub: SubprocessArgs,
    max_concurrency: int = 4,
    progress: bool = True,
) -> List[ReplayResult]:
    """Run all phase replays in parallel."""
    results: List[ReplayResult] = []
    total = len(runs)
    completed = 0
    started = time.time()

    if progress:
        print(f"\nRunning {total} phase replays (concurrency={max_concurrency})...")

    with cf.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_spec = {
            pool.submit(_run_one_replay, spec, baseline_dir, output_root, sub): spec
            for spec in runs
        }
        for fut in cf.as_completed(future_to_spec):
            spec = future_to_spec[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = ReplayResult(
                    run_id=spec.run_id,
                    config_name=spec.config.name,
                    self_player=spec.self_player,
                    day=spec.day,
                    run_dir=output_root / "replays" / f"day_{spec.day}" / spec.run_id,
                    returncode=-1,
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(res)
            completed += 1
            if progress:
                if res.returncode == 0 and res.error is None and res.elapsed_seconds == 0.0:
                    status = "skip"
                elif res.returncode == 0 and res.error is None:
                    status = "ok"
                else:
                    status = "FAIL"
                print(
                    f"  [{completed:>3}/{total}] {status:<4} "
                    f"{res.run_id:<48} "
                    f"({res.elapsed_seconds:5.1f}s)",
                    flush=True,
                )

    summary = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "total_replays": total,
        "succeeded": sum(1 for r in results if r.returncode == 0 and r.error is None),
        "failed": sum(1 for r in results if r.returncode != 0 or r.error is not None),
        "wall_seconds": time.time() - started,
        "results": [
            {
                "run_id": r.run_id,
                "config": r.config_name,
                "self": r.self_player,
                "day": r.day,
                "rc": r.returncode,
                "elapsed_seconds": r.elapsed_seconds,
                "error": r.error,
            }
            for r in results
        ],
    }
    with open(output_root / "replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if progress:
        print(
            f"\nReplays complete: {summary['succeeded']}/{total} ok, "
            f"{summary['failed']} failed, wall {summary['wall_seconds']:.1f}s."
        )
    return results
