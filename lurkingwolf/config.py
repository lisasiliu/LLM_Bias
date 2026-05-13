"""YAML experiment configuration loader.

Provides ``ExperimentConfig`` and ``ReplaySpec`` dataclasses for defining
reproducible experiments with optional rewriter interventions.

Usage:
    from config import load_config
    cfg = load_config("examples/formality.yaml")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ReplaySpec:
    rewriter_class: str
    rewriter_params: Dict[str, Any] = field(default_factory=dict)
    scope: str = "all"
    inject_at: str = "day_start"
    rotate_self: bool = False


@dataclass
class ExperimentConfig:
    name: str
    output_dir: str
    n_players: int = 7
    n_games: int = 1
    seeds: List[str] = field(default_factory=lambda: ["101"])
    model: str = "Qwen/Qwen3.5-9B"
    temperature: float = 0.0
    max_tokens: int = 300
    max_rounds: int = 20
    baseline_rewriter: Optional[Dict[str, Any]] = None
    replays: List[ReplaySpec] = field(default_factory=list)
    metrics: List[str] = field(default_factory=lambda: ["skill", "vote", "reliability"])


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate_raw(raw, p)

    replays = []
    for r in raw.get("replays", []):
        replays.append(ReplaySpec(
            rewriter_class=r["rewriter_class"],
            rewriter_params=r.get("rewriter_params", {}),
            scope=r.get("scope", "all"),
            inject_at=r.get("inject_at", "day_start"),
            rotate_self=r.get("rotate_self", False),
        ))

    seeds = raw.get("seeds", ["101"])
    if isinstance(seeds, (int, str)):
        seeds = [str(seeds)]
    else:
        seeds = [str(s) for s in seeds]

    return ExperimentConfig(
        name=raw["name"],
        output_dir=raw.get("output_dir", f"games/{raw['name']}"),
        n_players=raw.get("n_players", 7),
        n_games=raw.get("n_games", 1),
        seeds=seeds,
        model=raw.get("model", "Qwen/Qwen3.5-9B"),
        temperature=raw.get("temperature", 0.0),
        max_tokens=raw.get("max_tokens", 300),
        max_rounds=raw.get("max_rounds", 20),
        baseline_rewriter=raw.get("baseline_rewriter"),
        replays=replays,
        metrics=raw.get("metrics", ["skill", "vote", "reliability"]),
    )


def _validate_raw(raw: dict, path: Path) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")
    if "name" not in raw:
        raise ValueError(f"Config missing required 'name' field: {path}")

    n_players = raw.get("n_players", 7)
    if not (7 <= n_players <= 12):
        raise ValueError(f"n_players must be 7-12, got {n_players}")

    valid_inject = {"day_start", "election_start"}
    for i, r in enumerate(raw.get("replays", [])):
        if "rewriter_class" not in r:
            raise ValueError(f"Replay #{i} missing 'rewriter_class'")
        inject = r.get("inject_at", "day_start")
        if inject not in valid_inject:
            raise ValueError(
                f"Replay #{i} inject_at={inject!r} not in {valid_inject}"
            )
