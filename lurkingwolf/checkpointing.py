"""
checkpointing.py -- Save and restore game state at phase boundaries.

Used by eval to load day-start checkpoints and replay phases with
modified settings. Agents hold non-picklable references (LLM client,
prompts dict, token tracker) so we serialize their mutable state dicts
and rebuild fresh LLMAgent instances at load time.
"""

from __future__ import annotations

import copy
import pickle
import random as _random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import LLMAgent
from .engine import GameState
from .utils import TokenTracker


# ── Fields on LLMAgent that carry game-relevant mutable state ───────────────
# Everything NOT in this list is reconstructed at load time (LLM client, prompts,
# token_tracker) or is derived from __init__ args (player_id, role, model, …).

_AGENT_STATE_FIELDS = (
    "fact",
    "potential_truth",
    "potential_deception",
    "reliability",
    "town_consensus",
    "reasoning_history",
    "relationship_estimates",
    "action_history",
    "statement_history",
    "thinking_traces",
    "last_reasoning",
    "last_speech",
    "call_log",
    "temperature",
    "max_tokens",
    "speech_mode",
)

# __init__ args needed to rebuild an LLMAgent.
_AGENT_INIT_FIELDS = (
    "player_id", "role", "game_id", "model", "players",
)


def _dump_agent(agent: LLMAgent) -> Dict[str, Any]:
    """Serialize one LLMAgent to a plain dict (no non-picklable refs)."""
    data: Dict[str, Any] = {}
    for f in _AGENT_INIT_FIELDS:
        data[f] = copy.deepcopy(getattr(agent, f))
    for f in _AGENT_STATE_FIELDS:
        data[f] = copy.deepcopy(getattr(agent, f))
    if agent.rewriter is not None:
        data["rewriter_config"] = agent.rewriter.to_config()
    return data


def _rebuild_agent(
    agent_data: Dict[str, Any],
    prompts: dict,
    client,
    token_tracker: Optional[TokenTracker],
) -> LLMAgent:
    """Rebuild an LLMAgent from serialized state + fresh external deps."""
    agent = LLMAgent(
        player_id=agent_data["player_id"],
        role=agent_data["role"],
        game_id=agent_data["game_id"],
        model=agent_data["model"],
        client=client,
        prompts=prompts,
        players=list(agent_data["players"]),
        temperature=agent_data.get("temperature", 0),
        max_tokens=agent_data.get("max_tokens", 300),
        token_tracker=token_tracker,
        speech_mode=agent_data.get("speech_mode", "predicate"),
    )
    for f in _AGENT_STATE_FIELDS:
        if f in agent_data:
            setattr(agent, f, copy.deepcopy(agent_data[f]))
    if "rewriter_config" in agent_data:
        from .rewriter import Rewriter
        agent.rewriter = Rewriter.from_config(
            agent_data["rewriter_config"], llm_client=client,
        )
    return agent


# ── Checkpoint record ───────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """Self-contained snapshot of game progress at a phase boundary.

    Everything needed to continue the game deterministically from this point,
    minus shared external deps (LLM client/prompts/tracker) which the caller
    supplies at load time.
    """

    phase_name: str                          # e.g. "before_day_1_discussion"
    state: GameState
    agents: Dict[str, Dict[str, Any]]        # player_id -> serialized agent state
    rng_state: tuple                         # random.getstate()
    tracker_state: Dict[str, Any]            # subset of TokenTracker fields

    # Game-loop bookkeeping carried alongside state
    timeline_order: List[str] = field(default_factory=list)
    group_headers: Dict[str, dict] = field(default_factory=dict)
    per_phase_predicates: Dict[str, List[str]] = field(default_factory=dict)
    reasoning_timeline: List[dict] = field(default_factory=list)
    ph_mark: int = 0                          # len(state.public_history) at last snap

    # Run-level metadata so eval can rebuild agents without re-parsing CLI
    meta: Dict[str, Any] = field(default_factory=dict)
    # meta includes: model, experiment_type, seed, game_id, num_players


def _dump_tracker(tracker: Optional[TokenTracker]) -> Dict[str, Any]:
    if tracker is None:
        return {}
    return {
        "model": tracker.model,
        "max_tokens": tracker.max_tokens,
        "total_input_tokens": tracker.total_input_tokens,
        "total_output_tokens": tracker.total_output_tokens,
        "call_count": tracker.call_count,
    }


def _rebuild_tracker(tracker_state: Dict[str, Any]) -> Optional[TokenTracker]:
    if not tracker_state:
        return None
    t = TokenTracker(tracker_state["model"], tracker_state.get("max_tokens"))
    t.total_input_tokens = tracker_state.get("total_input_tokens", 0)
    t.total_output_tokens = tracker_state.get("total_output_tokens", 0)
    t.call_count = tracker_state.get("call_count", 0)
    return t


# ── Public API ──────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    phase_name: str,
    state: GameState,
    agents: Dict[str, LLMAgent],
    tracker: Optional[TokenTracker],
    *,
    timeline_order: Optional[List[str]] = None,
    group_headers: Optional[Dict[str, dict]] = None,
    per_phase_predicates: Optional[Dict[str, List[str]]] = None,
    reasoning_timeline: Optional[List[dict]] = None,
    ph_mark: int = 0,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Pickle a full game snapshot to *path*."""
    ckpt = Checkpoint(
        phase_name=phase_name,
        state=copy.deepcopy(state),
        agents={pid: _dump_agent(a) for pid, a in agents.items()},
        rng_state=_random.getstate(),
        tracker_state=_dump_tracker(tracker),
        timeline_order=list(timeline_order or []),
        group_headers=copy.deepcopy(group_headers or {}),
        per_phase_predicates=copy.deepcopy(per_phase_predicates or {}),
        reasoning_timeline=copy.deepcopy(reasoning_timeline or []),
        ph_mark=ph_mark,
        meta=dict(meta or {}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(ckpt, f)


def load_checkpoint(
    path: Path,
    prompts: dict,
    client,
) -> tuple:
    """Load a checkpoint and rebuild agents with fresh deps.

    Returns (checkpoint, agents_dict, tracker). The returned Checkpoint still
    holds the raw serialized agent dicts in its .agents field — agents_dict is
    the rebuilt dict of live LLMAgent instances ready to act().
    Also restores random module state as a side effect.
    """
    with open(path, "rb") as f:
        ckpt: Checkpoint = pickle.load(f)

    tracker = _rebuild_tracker(ckpt.tracker_state)
    agents = {
        pid: _rebuild_agent(data, prompts, client, tracker)
        for pid, data in ckpt.agents.items()
    }
    _random.setstate(ckpt.rng_state)
    return ckpt, agents, tracker


# ── Phase naming convention ─────────────────────────────────────────────────

def phase_checkpoint_name(phase: str) -> str:
    """Standardized checkpoint filename (without extension).

    Convention: `{phase}_start` — the snapshot represents the state at the
    START of that phase, before any of the phase's predicates are emitted.
    """
    return f"{phase}_start"


def list_checkpoints(game_dir: Path) -> List[Path]:
    """Return sorted list of checkpoint .pkl files in a game directory."""
    ckpt_dir = game_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("*.pkl"))
