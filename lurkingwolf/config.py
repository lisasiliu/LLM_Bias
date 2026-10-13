"""YAML experiment configuration loader.

Provides ``ExperimentConfig`` with named ``ComparisonSpec`` blocks.
Each comparison has its own baseline experiment type and replay list.

Usage:
    from lurkingwolf.config import load_config
    cfg = load_config("examples/formality/config.yaml")
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


VALID_MODES = {"pred_only", "nl", "flip_self", "flip_others"}
VALID_INJECT_AT = {"day_start", "election_start"}
VALID_BASELINES = {"pred_only", "nl"}
FLIP_SELF_METRICS = ["skill", "vote", "reliability"]
FLIP_OTHERS_METRICS = ["fairness"]


@dataclass
class ReplaySpec:
    mode: str
    rewriter_class: Optional[str] = None
    rewriter_params: Dict[str, Any] = field(default_factory=dict)
    inject_at: str = "day_start"
    self_role: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    repeat_index: int = 0          # >0: another untreated replay of the same checkpoint

    def __post_init__(self):
        if not self.metrics:
            if self.mode == "flip_self":
                self.metrics = list(FLIP_SELF_METRICS)
            elif self.mode == "flip_others":
                self.metrics = list(FLIP_OTHERS_METRICS)

    @property
    def self_dependent(self) -> bool:
        return self.mode in ("flip_self", "flip_others")

    @property
    def scope(self) -> str:
        if self.mode == "flip_self":
            return "self"
        elif self.mode == "flip_others":
            return "others"
        return "all"

    @property
    def run_tag(self) -> str:
        parts = [self.mode]
        if self.repeat_index:
            parts.append(f"repeat{self.repeat_index}")
        if self.rewriter_class:
            for k, v in sorted(self.rewriter_params.items()):
                parts.append(str(v))
        return "__".join(parts)


@dataclass
class ComparisonSpec:
    baseline: str
    replays: List[ReplaySpec] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    name: str
    n_players: int = 7
    n_games: int = 1
    seeds: List[str] = field(default_factory=lambda: ["101"])
    model: str = "Qwen/Qwen3.5-27B"
    temperature: float = 0.0
    max_tokens: int = 300
    max_rounds: int = 20
    comparisons: "OrderedDict[str, ComparisonSpec]" = field(default_factory=OrderedDict)
    metrics: List[str] = field(
        default_factory=lambda: ["skill", "vote", "reliability"]
    )
    rewriter_module: Optional[str] = None


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    _validate_raw(raw, p)

    comparisons = OrderedDict()
    for comp_name, comp_raw in raw.get("comparisons", {}).items():
        replays = []
        for r in comp_raw.get("replays", []):
            replays.append(ReplaySpec(
                mode=r["mode"],
                rewriter_class=r.get("rewriter"),
                rewriter_params=r.get("rewriter_params", {}),
                inject_at=r.get("inject_at", "day_start"),
                self_role=r.get("self_role"),
                metrics=r.get("metrics", []),
                repeat_index=int(r.get("repeat_index", 0)),
            ))
        comparisons[comp_name] = ComparisonSpec(
            baseline=comp_raw["baseline"],
            replays=replays,
        )

    seeds = raw.get("seeds", ["101"])
    if isinstance(seeds, (int, str)):
        seeds = [str(seeds)]
    else:
        seeds = [str(s) for s in seeds]

    return ExperimentConfig(
        name=raw["name"],
        n_players=raw.get("n_players", 7),
        n_games=raw.get("n_games", 1),
        seeds=seeds,
        model=raw.get("model", "Qwen/Qwen3.5-27B"),
        temperature=raw.get("temperature", 0.0),
        max_tokens=raw.get("max_tokens", 300),
        max_rounds=raw.get("max_rounds", 20),
        comparisons=comparisons,
        metrics=raw.get("metrics", ["skill", "vote", "reliability"]),
        rewriter_module=raw.get("rewriter_module"),
    )


def _validate_raw(raw: dict, path: Path) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")
    if "name" not in raw:
        raise ValueError(f"Config missing required 'name' field: {path}")

    n_players = raw.get("n_players", 7)
    if not (7 <= n_players <= 12):
        raise ValueError(f"n_players must be 7-12, got {n_players}")

    if "comparisons" not in raw:
        raise ValueError(f"Config missing required 'comparisons' field: {path}")

    for comp_name, comp in raw.get("comparisons", {}).items():
        if not isinstance(comp, dict):
            raise ValueError(f"Comparison '{comp_name}' must be a mapping")

        baseline = comp.get("baseline")
        if baseline is None:
            raise ValueError(
                f"Comparison '{comp_name}' missing required 'baseline' field"
            )
        if baseline not in VALID_BASELINES:
            raise ValueError(
                f"Comparison '{comp_name}' baseline={baseline!r} "
                f"not in {VALID_BASELINES}"
            )

        for i, r in enumerate(comp.get("replays", [])):
            repeat = r.get("repeat_index", 0)
            if not isinstance(repeat, int) or repeat < 0 or (repeat and r.get("mode") != "nl"):
                raise ValueError(
                    f"{path}: comparison '{comp_name}' replay #{i}: repeat_index must be a "
                    "non-negative integer and is only used on an 'nl' replay"
                )
            mode = r.get("mode")
            if mode is None:
                raise ValueError(
                    f"Comparison '{comp_name}' replay #{i} missing required 'mode'"
                )
            if mode not in VALID_MODES:
                raise ValueError(
                    f"Comparison '{comp_name}' replay #{i} mode={mode!r} "
                    f"not in {VALID_MODES}"
                )
            if mode in ("flip_self", "flip_others") and not r.get("rewriter"):
                raise ValueError(
                    f"Comparison '{comp_name}' replay #{i} mode={mode!r} "
                    f"requires a 'rewriter' field"
                )
            inject = r.get("inject_at", "day_start")
            if inject not in VALID_INJECT_AT:
                raise ValueError(
                    f"Comparison '{comp_name}' replay #{i} inject_at={inject!r} "
                    f"not in {VALID_INJECT_AT}"
                )
