"""Phase replay: load a day-start checkpoint, modify agent settings,
re-run discussion → vote → night, collect per-phase results.

This is the core of eval. Each call to ``replay_phase`` produces a
``PhaseResult`` with the votes, night actions, and post-phase reliability
for one (config, self_player, day) combination.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..engine import GameState
from ..agent import LLMAgent, extract_vote_target
from ..checkpointing import load_checkpoint
from ..run_game import (
    NL_EXPERIMENT_TYPES,
    broadcast_speech_nl,
    run_night_phase,
    resolve_wolf_votes,
    check_wwk_explosion,
    process_death_chain,
    broadcast_fact,
)


# ─── PhaseResult ───────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    """Results from replaying one day phase (discussion → vote → night)."""

    day: int
    config_name: str
    self_player: Optional[str]
    experiment_type: str
    role_dist: Dict[str, str]

    # Vote data (day N) — first round only
    votes: Dict[str, Optional[str]]       # voter -> target (None=abstain)
    vote_counts: Dict[str, float]          # target -> weighted count
    executed: Optional[str]
    votes_received: Dict[str, int]         # target -> raw unweighted count

    # Night data (night N)
    wolf_target: Optional[str]
    seer_target: Optional[str]
    guard_target: Optional[str]
    witch_save: bool
    witch_poison: Optional[str]
    night_killed: Optional[str]

    # Post-phase reliability (from memory update after discussion+vote+night)
    reliability: Dict[str, Dict[str, int]]  # player -> {other: score}

    # State snapshot
    alive_at_start: List[str] = field(default_factory=list)

    # Speeches (for debugging / NL analysis)
    speeches: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "config_name": self.config_name,
            "self_player": self.self_player,
            "experiment_type": self.experiment_type,
            "role_dist": self.role_dist,
            "alive_at_start": self.alive_at_start,
            "votes": self.votes,
            "vote_counts": self.vote_counts,
            "executed": self.executed,
            "votes_received": self.votes_received,
            "wolf_target": self.wolf_target,
            "seer_target": self.seer_target,
            "guard_target": self.guard_target,
            "witch_save": self.witch_save,
            "witch_poison": self.witch_poison,
            "night_killed": self.night_killed,
            "reliability": self.reliability,
            "speeches": self.speeches,
        }


# ─── Agent setting overrides ──────────────────────────────────────────

def _apply_overrides(
    agents: Dict[str, LLMAgent],
    experiment_type: str,
    *,
    rewriter=None,
    scope: str = "all",
    self_player: Optional[str] = None,
) -> None:
    """Modify agent speech_mode and rewriter in place for the replay."""
    speech_mode = "natural" if experiment_type in NL_EXPERIMENT_TYPES else "predicate"
    for pid, agent in agents.items():
        agent.speech_mode = speech_mode
        if rewriter is not None:
            if scope == "all":
                agent.rewriter = rewriter
            elif scope == "self" and pid == self_player:
                agent.rewriter = rewriter
            elif scope == "others" and pid != self_player:
                agent.rewriter = rewriter


# ─── Discussion replay ────────────────────────────────────────────────

def _run_discussion(
    state: GameState,
    agents: Dict[str, LLMAgent],
    order: List[str],
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Re-run discussion phase. Returns speech records."""
    speeches: List[Dict[str, Any]] = []
    for speaker_idx, speaker in enumerate(order):
        if not state.is_alive(speaker):
            continue
        if check_wwk_explosion(state, agents, verbose, dry_run):
            break
        remaining = [p for p in order[speaker_idx + 1:] if state.is_alive(p)]
        speech_extra = (
            f"Players who have not yet spoken this turn: {', '.join(remaining)}"
            if remaining else ""
        )
        alive_others = [p for p in state.get_alive() if p != speaker]
        preds = agents[speaker].act(
            state, "day_speech", alive_others, extra_context=speech_extra, dry_run=dry_run,
        )
        flat = state.record_speech(speaker, preds, speech=agents[speaker].last_speech)
        agents[speaker].record_own_statement(
            f"day_{state.day}_discussion", flat,
        )
        speaker_nl = " ".join(
            (item.get("text") or "").strip()
            for item in (agents[speaker].last_speech or [])
        ).strip()
        if speaker_nl:
            broadcast_speech_nl(agents, state, speaker, speaker_nl, dry_run=dry_run)
        speech_items = [
            {"pred": it.get("pred", ""), "text": it.get("text", "")}
            for it in (agents[speaker].last_speech or [])
        ]
        speeches.append({"speaker": speaker, "predicates": flat, "nl": speech_items})
        if verbose:
            for line in flat:
                print(f"  {line}")
    state.flush_turn_buffer()
    return speeches


# ─── Vote replay ──────────────────────────────────────────────────────

def _run_vote(
    state: GameState,
    agents: Dict[str, LLMAgent],
    discussion_preds: List[str],
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """Run day vote. Returns {votes, vote_counts, executed}."""
    state.current_turn_buffer = list(discussion_preds)
    state.start_voting()

    votes: Dict[str, Optional[str]] = {}
    for voter in state.get_alive():
        allowed = [p for p in state.get_alive() if p != voter]
        preds = agents[voter].act(state, "day_vote", allowed, dry_run=dry_run)
        target = extract_vote_target(preds[0]) if preds else None
        if not target or target.lower() in ("none", "null", "abstain") or target not in allowed:
            votes[voter] = None
        else:
            votes[voter] = target

    day_result = state.resolve_votes(votes)
    state.current_turn_buffer.clear()

    first_round_votes = dict(votes)
    first_round_counts = dict(day_result.vote_counts)
    final_executed = day_result.executed

    if verbose:
        print(f"  Votes: {votes}")
        print(f"  Vote counts: {first_round_counts}")
        if day_result.tied_candidates:
            print(f"  Tied: {day_result.tied_candidates}; "
                  f"random tiebreak → {day_result.executed}")
        else:
            print(f"  Executed: {day_result.executed}")

    broadcast_fact(agents, state, f"executed(D{state.day}, {final_executed})")
    if final_executed in agents:
        agents[final_executed].receive_fact(
            f"executed(D{state.day}, {final_executed})"
        )
    process_death_chain(
        state, agents, [final_executed],
        is_night_reveal=False,
        night=state.night, day=state.day,
        verbose=verbose, dry_run=dry_run,
    )

    # Compute unweighted votes-received from first-round votes only
    votes_received: Dict[str, int] = {}
    for target in first_round_votes.values():
        if target is not None:
            votes_received[target] = votes_received.get(target, 0) + 1

    return {
        "votes": first_round_votes,
        "vote_counts": first_round_counts,
        "executed": final_executed,
        "votes_received": votes_received,
    }


# ─── Core replay function ─────────────────────────────────────────────

def replay_phase(
    checkpoint_path: Path,
    prompts: dict,
    client,  # LLMClient | None
    *,
    agent_settings: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    experiment_type: str,
    config_name: str = "",
    self_player: Optional[str] = None,
    rewriter=None,
    rewriter_scope: str = "all",
    dry_run: bool = False,
    verbose: bool = False,
) -> PhaseResult:
    """Load a day-start checkpoint and replay one full phase.

    Steps:
      1. Load checkpoint (state + agents at day start)
      2. Apply agent setting overrides
      3. Run discussion → listeners parse modified speeches
      4. Run vote → collect votes
      5. Run night → collect skills
      6. Run memory update → collect post-phase reliability
         (the day's discussion is shown to the raters)

    Returns a PhaseResult with all collected data.
    """
    ckpt, agents, tracker = load_checkpoint(checkpoint_path, prompts, client)
    state = ckpt.state
    day = state.day

    role_dist = {pid: state.roles[pid] for pid in state.players}
    alive_at_start = list(state.get_alive())

    order = ckpt.meta.get("speech_order", list(state.get_alive()))

    _apply_overrides(
        agents, experiment_type,
        rewriter=rewriter, scope=rewriter_scope, self_player=self_player,
    )

    if rewriter is not None and hasattr(rewriter, "bind"):
        rewriter.bind(agents)                    # policies that read the live agents' ratings

    if verbose:
        print(f"\n  == Replay Day {day} Discussion ({config_name}) ==")

    # Discussion
    speeches = _run_discussion(
        state, agents, order, dry_run=dry_run, verbose=verbose,
    )

    # Collect discussion predicates for vote context
    discussion_preds = list(state.public_history[ckpt.ph_mark:])

    # Vote
    if verbose:
        print(f"\n  == Replay Day {day} Vote ==")
    vote_data = _run_vote(
        state, agents, discussion_preds, dry_run=dry_run, verbose=verbose,
    )

    # Night
    timeline_order = list(ckpt.timeline_order)
    group_headers = dict(ckpt.group_headers)
    if verbose:
        print(f"\n  == Replay Night {day} ==")

    winner = state.check_win()
    night_result = None
    pending_night_deaths = []
    if not winner:
        night_result, pending_night_deaths = run_night_phase(
            state, agents, timeline_order, group_headers, verbose, dry_run,
        )

    # Post-phase memory update for reliability scores
    time_str = f"day_{day}_post_replay"
    state.current_turn_buffer = list(discussion_preds)    # raters see what was said today
    for p in state.get_alive():
        if p in agents:
            agents[p].update_memory(state, time_str, dry_run=dry_run)
    state.current_turn_buffer.clear()

    reliability = {}
    for pid, agent in agents.items():
        reliability[pid] = dict(agent.reliability)

    result = PhaseResult(
        day=day,
        config_name=config_name,
        self_player=self_player,
        experiment_type=experiment_type,
        role_dist=role_dist,
        votes=vote_data["votes"],
        vote_counts=vote_data["vote_counts"],
        executed=vote_data["executed"],
        votes_received=vote_data["votes_received"],
        wolf_target=night_result.wolf_target if night_result else None,
        seer_target=night_result.seer_target if night_result else None,
        guard_target=night_result.guard_target if night_result else None,
        witch_save=bool(night_result and night_result.witch_save_target),
        witch_poison=night_result.witch_poison_target if night_result else None,
        night_killed=night_result.killed if night_result else None,
        reliability=reliability,
        alive_at_start=alive_at_start,
        speeches=speeches,
    )
    return result, agents, state


# ─── Baseline data extraction ─────────────────────────────────────────

@dataclass
class BaselinePhaseData:
    """Baseline data for one day phase, extracted from a full game run."""
    day: int
    role_dist: Dict[str, str]

    votes: Dict[str, Optional[str]]
    vote_counts: Dict[str, float]
    executed: Optional[str]
    votes_received: Dict[str, int]

    wolf_target: Optional[str]
    seer_target: Optional[str]
    guard_target: Optional[str]
    witch_save: bool
    witch_poison: Optional[str]
    night_killed: Optional[str]

    reliability: Dict[str, Dict[str, int]]

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "role_dist": self.role_dist,
            "votes": self.votes,
            "vote_counts": self.vote_counts,
            "executed": self.executed,
            "votes_received": self.votes_received,
            "wolf_target": self.wolf_target,
            "seer_target": self.seer_target,
            "guard_target": self.guard_target,
            "witch_save": self.witch_save,
            "witch_poison": self.witch_poison,
            "night_killed": self.night_killed,
            "reliability": self.reliability,
        }


def extract_baseline_phase(game_dir: Path, day: int) -> BaselinePhaseData:
    """Extract per-phase baseline data from a completed game's output files."""
    gr = json.loads((game_dir / "game_record.json").read_text(encoding="utf-8"))
    role_dist = gr.get("hidden_info", {}).get("role_distribution", {})

    day_actions = gr.get("hidden_info", {}).get("day_actions", [])
    night_actions = gr.get("hidden_info", {}).get("night_actions", [])

    # Day vote data (1-indexed; day_actions[0] = D1)
    da = day_actions[day - 1] if day - 1 < len(day_actions) else {}
    raw_votes = da.get("votes", {})
    votes: Dict[str, Optional[str]] = {}
    votes_received: Dict[str, int] = {}
    for target, voters in raw_votes.items():
        votes_received[target] = len(voters or [])
        for v in (voters or []):
            votes[v] = target

    vote_counts: Dict[str, float] = {}
    sheriff = da.get("sheriff")
    for target, voters in raw_votes.items():
        count = 0.0
        for v in (voters or []):
            count += 1.5 if v == sheriff else 1.0
        vote_counts[target] = count

    # Night data (night index = day index, since N0 is before D1)
    na = night_actions[day] if day < len(night_actions) else {}

    # Post-phase reliability from agent memory (day+1 reasoning)
    reliability: Dict[str, Dict[str, int]] = {}
    mem_dir = game_dir / "agent_memory"
    if mem_dir.exists():
        for p_path in sorted(mem_dir.glob("P*.json")):
            pid = p_path.stem
            mem = json.loads(p_path.read_text(encoding="utf-8"))
            for entry in mem.get("reasoning_history", []):
                if entry.get("timeline") == f"day_{day + 1}":
                    rr = entry.get("reasoning_result", {})
                    reliability[pid] = {}
                    for target_id, info in rr.items():
                        conf = info.get("confidence", 5)
                        role_guess = (info.get("role") or "").lower()
                        own_role = role_dist.get(pid, "").lower()
                        wolf_roles = {"werewolf", "white_wolf_king"}
                        if role_guess in wolf_roles and own_role not in wolf_roles:
                            reliability[pid][target_id] = 11 - conf
                        else:
                            reliability[pid][target_id] = conf
                    break
            # Fallback: use the day N reliability if day+1 doesn't exist
            if pid not in reliability:
                for entry in mem.get("reasoning_history", []):
                    if entry.get("timeline") == f"day_{day}":
                        rr = entry.get("reasoning_result", {})
                        reliability[pid] = {}
                        for target_id, info in rr.items():
                            conf = info.get("confidence", 5)
                            reliability[pid][target_id] = conf
                        break

    return BaselinePhaseData(
        day=day,
        role_dist=role_dist,
        votes=votes,
        vote_counts=vote_counts,
        executed=da.get("executed"),
        votes_received=votes_received,
        wolf_target=na.get("wolf_target"),
        seer_target=na.get("seer_target"),
        guard_target=na.get("guard_target"),
        witch_save=bool(na.get("witch_save_target")),
        witch_poison=na.get("witch_poison_target"),
        night_killed=na.get("killed") if isinstance(na.get("killed"), str)
                     else (na.get("killed", [None]) or [None])[0] if isinstance(na.get("killed"), list)
                     else None,
        reliability=reliability,
    )


def count_baseline_days(game_dir: Path) -> int:
    """Return the number of day phases in the baseline game."""
    gr = json.loads((game_dir / "game_record.json").read_text(encoding="utf-8"))
    return len(gr.get("hidden_info", {}).get("day_actions", []))


def list_day_checkpoints(game_dir: Path) -> List[Path]:
    """Return sorted list of day-start checkpoint files."""
    ckpt_dir = game_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("day_*_start.pkl"))
