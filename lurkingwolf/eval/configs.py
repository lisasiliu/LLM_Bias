"""Phase-level experiment configs for eval.

Experiment 1  (--mode pred_vs_nl):  Does language matter?
  Baseline: pred-only game.  Replay: NL baseline (1 per day).

Custom rewriter experiments can be built with ``rewriter_configs()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


BASELINE_EXPERIMENT: Dict[str, str] = {
    "pred_vs_nl": "pred_only",
}


def _player_ids(num_players: int) -> List[str]:
    return [f"P{i}" for i in range(1, num_players + 1)]


@dataclass(frozen=True)
class PhaseConfig:
    """One experimental manipulation applied during a phase replay."""
    name: str
    family: str
    experiment_type: str
    description: str
    self_dependent: bool
    rewriter_factory: Optional[Callable[..., object]] = None
    rewriter_params: Optional[Dict] = None
    scope: str = "all"

    def make_rewriter(self, llm_client=None):
        if self.rewriter_factory is None:
            return None
        params = dict(self.rewriter_params or {})
        return self.rewriter_factory(**params, llm_client=llm_client)


def pred_vs_nl_configs() -> List[PhaseConfig]:
    return [PhaseConfig(
        name="nl_baseline_replay",
        family="pred_vs_nl",
        experiment_type="nl",
        description="Replay pred-only checkpoint with NL baseline.",
        self_dependent=False,
    )]


def rewriter_configs(
    rewriter_class,
    params_variants: List[Dict],
    *,
    family: str = "rewriter",
    scopes: Optional[List[str]] = None,
) -> List[PhaseConfig]:
    """Build PhaseConfig list for a rewriter with multiple param variants.

    Args:
        rewriter_class: A Rewriter subclass (has .name attribute).
        params_variants: List of param dicts, e.g. [{"level": "formal"}, {"level": "casual"}].
        family: Config family name.
        scopes: List of scopes ("all", "self", "others"). Default: ["all"].

    Returns:
        List of PhaseConfig, one per (variant, scope) combination.
        Scope "self" configs are self_dependent=True.
    """
    if scopes is None:
        scopes = ["all"]

    configs: List[PhaseConfig] = []
    for params in params_variants:
        tag = "_".join(f"{k}={v}" for k, v in sorted(params.items()))
        for scope in scopes:
            name = f"{rewriter_class.name}__{tag}__{scope}"
            configs.append(PhaseConfig(
                name=name,
                family=family,
                experiment_type="nl",
                description=f"{rewriter_class.name} ({tag}, scope={scope})",
                self_dependent=(scope == "self"),
                rewriter_factory=rewriter_class,
                rewriter_params=params,
                scope=scope,
            ))
    return configs


def configs_for_mode(mode: str) -> List[PhaseConfig]:
    if mode == "pred_vs_nl":
        return pred_vs_nl_configs()
    raise ValueError(f"mode must be 'pred_vs_nl', got {mode!r}")


@dataclass(frozen=True)
class PhaseRunSpec:
    """One replay = (config, self_player, day)."""
    config: PhaseConfig
    self_player: Optional[str]
    day: int

    @property
    def run_id(self) -> str:
        parts = [self.config.name]
        if self.self_player is not None:
            parts.append(self.self_player.lower())
        parts.append(f"d{self.day}")
        return "__".join(parts)


def expand_phase_runs(
    configs: List[PhaseConfig],
    self_players: List[str],
    days: List[int],
) -> List[PhaseRunSpec]:
    """Cross configs × self_players × days, deduplicating self-independent configs."""
    runs: List[PhaseRunSpec] = []
    seen_independent: set = set()

    for cfg in configs:
        if not cfg.self_dependent:
            for day in days:
                key = (cfg.name, day)
                if key in seen_independent:
                    continue
                seen_independent.add(key)
                runs.append(PhaseRunSpec(config=cfg, self_player=None, day=day))

    for cfg in configs:
        if cfg.self_dependent:
            for self_player in self_players:
                for day in days:
                    runs.append(PhaseRunSpec(
                        config=cfg, self_player=self_player, day=day,
                    ))
    return runs
