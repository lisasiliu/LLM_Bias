"""
schemas.py -- Build JSON records for the werewolf output schemas.

Stripped-down versions following approved mockup structure:
  - public_history: {num_players, sheriff, phases}
  - current_turn: built at save time from per_phase_predicates + group headers
  - agent_memory: {agent_id, role, fact, potential_truth, potential_deception,
                   reliability, reasoning_history, relationship_estimates,
                   action_history, statement_history}
  - game_record: {game_id, experiment_type, model, num_players, hidden_info}
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from .engine import GameState


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


# ═══════════════════════════════════════════════════════════════
# 1. Public History (single merged file for the whole game)
# ═══════════════════════════════════════════════════════════════

def build_public_history(state: GameState,
                         per_phase_predicates: Dict[str, List[str]]) -> dict:
    """Build public history: num_players, sheriff, phases."""
    return {
        "num_players": len(state.players),
        "sheriff": {
            "player_id": state.sheriff_history if state.sheriff_history else None,
            "vote_weight": state.sheriff_vote_weight,
        },
        "phases": dict(per_phase_predicates),
    }


# ═══════════════════════════════════════════════════════════════
# 2. Current Turn (built at save time from per_phase_predicates)
# ═══════════════════════════════════════════════════════════════

def capture_group_header(state: GameState) -> dict:
    """Capture alive/dead/sheriff for a current_turn group header."""
    return {
        "alive_players": state.get_alive(),
        "dead_players": state.get_dead(),
        "sheriff": state.sheriff,
    }


# ═══════════════════════════════════════════════════════════════
# 3. Agent Memory (one per player)
# ═══════════════════════════════════════════════════════════════

def build_agent_memory(
    agent_id: str,
    role: str,
    *,
    fact: List[str],
    potential_truth: List[dict],
    potential_deception: List[dict],
    reliability: Dict[str, int],
    reasoning_history: List[dict],
    relationship_estimates: Dict[str, list],
    action_history: List[str],
    statement_history: List[dict],
    thinking_traces: Optional[List[dict]] = None,
) -> dict:
    """Build agent memory for one player."""
    result = {
        "agent_id": agent_id,
        "role": role,
        "fact": list(fact),
        "potential_truth": list(potential_truth),
        "potential_deception": list(potential_deception),
        "reliability": dict(reliability),
        "reasoning_history": list(reasoning_history),
        "relationship_estimates": dict(relationship_estimates),
        "action_history": list(action_history),
        "statement_history": list(statement_history),
    }
    if thinking_traces:
        result["thinking_traces"] = list(thinking_traces)
    return result


# ═══════════════════════════════════════════════════════════════
# 4. Game Record (aggregated at end of game)
# ═══════════════════════════════════════════════════════════════

def build_game_record(
    state: GameState,
    experiment_type: str,
    model: str,
    *,
    seed: Optional[int] = None,
) -> dict:
    """Build game record: game_id, experiment_type, model, num_players, hidden_info."""
    # Check which roles are in the game to omit unused fields
    has_witch = "witch" in state.roles.values()

    hidden_info = {
        "role_distribution": dict(state.roles),
        "night_actions": [],
        "day_actions": [],
    }
    for nr in state.night_log:
        night_entry = {
            "night": f"N{nr.night}",
            "wolf_target": nr.wolf_target,
            "seer_target": nr.seer_target,
            "seer_result": nr.seer_result,
            "guard_target": nr.guard_target,
        }
        if has_witch:
            night_entry["witch_save_target"] = nr.witch_save_target
            night_entry["witch_poison_target"] = nr.witch_poison_target
        # killed as a list (multiple deaths possible: wolf + witch poison, hunter, etc.)
        killed_list = []
        if nr.killed:
            killed_list.append(nr.killed)
        if has_witch and nr.witch_poison_target and nr.witch_poison_target != nr.killed:
            if nr.witch_poison_target in state.dead:
                killed_list.append(nr.witch_poison_target)
        night_entry["killed"] = killed_list
        hidden_info["night_actions"].append(night_entry)

    for dr in state.vote_log:
        # Build {target: [voters]} format
        vote_dict: Dict[str, list] = {}
        for voter, target in dr.votes.items():
            if target is None:
                continue  # skip abstentions
            vote_dict.setdefault(target, []).append(voter)
        hidden_info["day_actions"].append({
            "day": f"D{dr.day}",
            "sheriff": dr.sheriff,
            "votes": vote_dict,
            "executed": dr.executed,
        })

    record: dict = {}
    if seed is not None:
        record["random_seed"] = seed
    record["game_id"] = state.game_id
    record["experiment_type"] = experiment_type
    record["model"] = model
    record["num_players"] = len(state.players)
    record["hidden_info"] = hidden_info
    return record
