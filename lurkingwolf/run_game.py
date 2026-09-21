"""
run_game.py -- Run a Werewolf game with LLM agents (predicate or NL output).

Game flow:
  N0 → D1 (sheriff election → announce N0 deaths → last_words → ability_window
        → discussion → vote → optional PK speech+revote → execution
        chain) → N1 → D2 (announce → last_words → ability_window → discussion
        → vote → ...) → ...
Supports 7-12 players (see engine.ROLE_CONFIGS).

Predicate format: predicate(phase_label, speaker_or_target, args...)
  phase_label = N0, D1, N1, D2, ...  (no game_id)

Output structure per game:
  game_<id>/
    manifest.json        -- file map + metadata
    public_history.json  -- per-phase predicate strings
    current_turn/
      <phase>.json       -- shared game snapshot per phase
    agent_memory/
      P1.json ... Pn.json
    game_record.json     -- hidden info (roles + night actions)
    game_log.jsonl       -- raw LLM calls
    resources.json       -- tokens + timing
    game.out             -- captured terminal output
"""

import io
import os
import re
import sys
import json
import argparse
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv, find_dotenv

from .engine import GameState, WOLF_ROLES, randomize_roles
from .agent import (
    LLMAgent, extract_vote_target, extract_attack_target,
    extract_night_target, extract_sheriff_direction, extract_successor,
    extract_run_sheriff, extract_withdraw_sheriff, extract_witch_save, extract_witch_poison_target,
    extract_hunter_target, extract_wwk_explode,
)
from .schemas import (
    build_public_history, build_agent_memory, build_game_record,
    capture_group_header,
)
from .utils import TokenTracker, TokenBudgetExceeded
from .utils import LLMClient
from .checkpointing import save_checkpoint, phase_checkpoint_name

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_FILE = SCRIPT_DIR / "prompts" / "prompt.json"
EXPERIMENT_TYPES = {
    "pred_only":   "predicate_only",
    "nl":          "natural_language",
}

NL_EXPERIMENT_TYPES = {"nl"}


def _fmt_vote_counts(counts: dict) -> str:
    """Format vote counts as multiline 'voted: count', sorted descending."""
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    lines = [f"    {target}: {int(ct) if ct == int(ct) else ct}" for target, ct in sorted_items]
    return "  Vote counts: {\n" + "\n".join(lines) + "\n  }"


def load_prompts() -> dict:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_model(prompts: dict, model_str: str) -> str:
    return prompts.get("models", {}).get(model_str, model_str)



# ═══════════════════════════════════════════════════════════════
# Agent memory helpers
# ═══════════════════════════════════════════════════════════════

def init_agent_facts(agents: Dict[str, LLMAgent], state: GameState) -> None:
    """Seed agents with initial facts they would know at game start."""
    for p, agent in agents.items():
        agent.receive_fact(f"role({p}, {agent.role})")
        if agent.role in WOLF_ROLES:
            for other in state.players:
                if other != p and state.roles[other] in WOLF_ROLES:
                    agent.receive_fact(f"role({other}, {state.roles[other]})")


def broadcast_fact(agents: Dict[str, LLMAgent], state: GameState,
                   fact_str: str) -> None:
    """Send a fact to all alive agents."""
    for p in state.get_alive():
        if p in agents:
            agents[p].receive_fact(fact_str)


def broadcast_speech(agents: Dict[str, LLMAgent], state: GameState,
                     speaker: str, predicates: List[str]) -> None:
    """Legacy: broadcast speaker-supplied predicates to listeners. Use
    broadcast_speech_nl for the rigorous listener-side parse path."""
    for p in state.get_alive():
        if p != speaker and p in agents:
            speaker_reliability = agents[p].reliability.get(speaker)
            is_suspect = speaker_reliability is None or speaker_reliability < 7.5
            for pred in predicates:
                agents[p].classify_claim(speaker, pred,
                                         suspected_deception=is_suspect)


def broadcast_speech_nl(agents: Dict[str, LLMAgent], state: GameState,
                        speaker: str, nl_text: str,
                        dry_run: bool = False) -> None:
    """Each alive listener (other than the speaker) parses the speaker's
    natural-language speech into their own predicates and classifies each
    as truth or deception. One LLM call per listener.
    """
    if not nl_text or not nl_text.strip():
        return
    for p in state.get_alive():
        if p == speaker or p not in agents:
            continue
        agents[p].classify_speech_nl(state, speaker, nl_text, dry_run=dry_run)


def resolve_wolf_votes(votes: Dict[str, Optional[str]]) -> tuple:
    tally: Dict[str, int] = {}
    for target in votes.values():
        if not target or target == "abstain":
            continue
        tally[target] = tally.get(target, 0) + 1
    if not tally:
        return "abstain", tally
    max_votes = max(tally.values())
    leaders = [target for target, count in tally.items() if count == max_votes]
    if len(leaders) != 1:
        return "tie", tally
    return leaders[0], tally


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


# ═══════════════════════════════════════════════════════════════
# WWK explosion check helper
# ═══════════════════════════════════════════════════════════════

def check_wwk_explosion(
    state: GameState,
    agents: Dict[str, LLMAgent],
    verbose: bool,
    dry_run: bool,
) -> bool:
    """Check if the White Wolf King wants to explode. Returns True if explosion happened."""
    wwks = [p for p in state.get_alive() if state.roles.get(p) == "white_wolf_king"]
    if not wwks:
        return False
    wwk = wwks[0]
    preds = agents[wwk].act(state, "wwk_explode_check", state.get_alive(), dry_run=dry_run)
    if not preds:
        return False
    target = extract_wwk_explode(preds[0])
    if target is None:
        return False  # wwk_pass or invalid
    # Explode!
    killed_target = state.handle_wwk_explosion(wwk, target)
    if verbose:
        print(f"  !! WWK {wwk} EXPLODES, killing {target}!")
    # Broadcast deaths
    broadcast_fact(agents, state, f"wwk_explode(D{state.day}, {wwk}, {target})")
    # Hunter chain: if target was hunter, they may shoot
    if killed_target and state.roles.get(killed_target) == "hunter" and state.hunter_can_shoot:
        if verbose:
            print(f"  Hunter {killed_target} was killed by explosion! May shoot.")
        hp = agents[killed_target].act(state, "hunter_shoot", state.get_alive(), dry_run=dry_run)
        shot = extract_hunter_target(hp[0]) if hp else None
        if shot and shot.lower() in ("none", "null"):
            shot = None
        if shot:
            state.handle_hunter_death(killed_target, shot)
            broadcast_fact(agents, state, f"hunter_shot(D{state.day}, {killed_target}, {shot})")
            if verbose:
                print(f"  >> Hunter {killed_target} shoots {shot}!")
    return True


def _run_last_words(
    state: GameState,
    agents: Dict[str, LLMAgent],
    killed: str,
    *,
    timeline: str,
    verbose: bool,
    dry_run: bool,
    nl_log: Optional[List[dict]] = None,
) -> None:
    """Run last_speech for one player. Listener-side parse broadcasts the
    natural-language text to surviving LLM listeners.
    """
    if killed not in agents:
        return
    preds = agents[killed].act(state, "last_speech", dry_run=dry_run)
    flat = state.record_speech(killed, preds, speech=agents[killed].last_speech)
    agents[killed].record_own_statement(timeline, flat)
    speaker_nl = " ".join(
        (item.get("text") or "").strip()
        for item in (agents[killed].last_speech or [])
    ).strip()
    if speaker_nl:
        broadcast_speech_nl(agents, state, killed, speaker_nl, dry_run=dry_run)
    if nl_log is not None and agents[killed].last_speech:
        nl_log.append({
            "phase": timeline,
            "speaker": killed,
            "speech": [{"pred": it.get("pred", ""), "text": it.get("text", "")}
                       for it in agents[killed].last_speech],
        })
    if verbose:
        for line in flat:
            print(f"  {line}")
    state.flush_turn_buffer()


def _run_ability_window(
    state: GameState,
    agents: Dict[str, LLMAgent],
    killed: str,
    *,
    day: int,
    verbose: bool,
    dry_run: bool,
) -> Optional[str]:
    """Process the ability window for one death:
      - sheriff badge succession (if killed was sheriff)
      - hunter shoot (if killed was hunter and can shoot)
      - white wolf king self-destruct (if killed was WWK)
    Returns any chained death (target killed by hunter/wwk), or None.
    """
    if state.sheriff == killed:
        if verbose:
            print(f"  Sheriff {killed} died. Choosing badge successor.")
        successor = None
        if killed in agents and not dry_run:
            preds = agents[killed].act(
                state, "sheriff_successor", state.get_alive(), dry_run=dry_run
            )
            successor = extract_successor(preds[0]) if preds else None
        state.handle_sheriff_death(successor)
        if state.sheriff:
            broadcast_fact(agents, state, f"sheriff_set(D{day}, {state.sheriff})")
        if verbose:
            print(f"  New sheriff: {state.sheriff or '(abolished)'}")

    if state.roles.get(killed) == "hunter" and state.hunter_can_shoot:
        if verbose:
            print(f"  Hunter {killed} may shoot.")
        preds = agents[killed].act(
            state, "hunter_shoot", state.get_alive(), dry_run=dry_run
        )
        shot_target = extract_hunter_target(preds[0]) if preds else None
        if shot_target and shot_target.lower() in ("none", "null"):
            shot_target = None
        if shot_target:
            killed_target = state.handle_hunter_death(killed, shot_target)
            broadcast_fact(agents, state, f"hunter_shot(D{day}, {killed}, {shot_target})")
            if verbose:
                print(f"  >> Hunter {killed} shoots {shot_target}!")
            return killed_target
        if verbose:
            print(f"  >> Hunter {killed} chooses not to shoot.")
        return None

    if state.roles.get(killed) == "white_wolf_king":
        if verbose:
            print(f"  White Wolf King {killed} may self-destruct.")
        preds = agents[killed].act(
            state, "wwk_explode_check", state.get_alive(), dry_run=dry_run
        )
        target = extract_wwk_explode(preds[0]) if preds else None
        if target:
            killed_target = state.handle_wwk_explosion(killed, target)
            broadcast_fact(agents, state, f"wwk_explode(D{day}, {killed}, {target})")
            if verbose:
                print(f"  >> White Wolf King {killed} explodes, killing {target}!")
            return killed_target
        if verbose:
            print(f"  >> White Wolf King {killed} does not explode.")
        return None

    return None


def process_death_chain(
    state: GameState,
    agents: Dict[str, LLMAgent],
    primary_deaths: List[str],
    *,
    is_night_reveal: bool,
    night: int,
    day: int,
    verbose: bool,
    dry_run: bool,
    nl_log: Optional[List[dict]] = None,
) -> List[str]:
    """Reveal primary deaths and run last_words + ability_window for every
    death in the chain. Used for both night-reveal and day-vote-execution.

    For night reveal: emits "attacked" + "announcement(...death)" predicates,
    moves players from alive→dead. (Day-vote execution side already does this
    in engine.resolve_votes, so callers there pass the executed player as the
    primary death.)

    Last words run for primary deaths only. Chained deaths (hunter shot,
    WWK explosion target) get the ability_window but no last_words.

    Returns the full ordered list of all deaths processed.
    """
    primary = dedupe_keep_order(primary_deaths)

    if is_night_reveal:
        if verbose:
            if primary:
                print(f"  [Announcement] Night deaths: {primary}")
            else:
                print("  [Announcement] No one died last night.")
        if not primary:
            state.public_history.append(
                f"announcement(N{night}, none, death): No one died last night."
            )
            return []
        for killed in primary:
            state.alive.discard(killed)
            state.dead.add(killed)
            state.public_history.append(f"attacked(N{night}, {killed})")
            state.public_history.append(
                f"announcement(N{night}, {killed}, death): "
                f"{killed} was killed last night."
            )
            broadcast_fact(agents, state, f"attacked(N{night}, {killed})")
            if killed in agents:
                agents[killed].receive_fact(f"attacked(N{night}, {killed})")
    else:
        # Day-vote execution path: caller is responsible for engine state
        # changes (resolve_votes already moves alive→dead and emits the
        # execution announcement). Just chain off the primary death.
        if not primary:
            return []

    # Last words only for night-0 deaths or day-vote executions (not night >= 1)
    do_lw = (not is_night_reveal) or (night == 0)
    queue: List[tuple] = [(p, do_lw) for p in primary]  # (killed, do_last_words)
    seen = set(primary)
    all_deaths = list(primary)

    timeline_prefix = f"night_{night}" if is_night_reveal else f"day_{day}_vote"

    while queue:
        killed, do_last_words = queue.pop(0)
        if do_last_words:
            _run_last_words(
                state, agents, killed,
                timeline=f"{timeline_prefix}_last_words_{killed}",
                verbose=verbose, dry_run=dry_run,
                nl_log=nl_log,
            )
        chained = _run_ability_window(
            state, agents, killed,
            day=day, verbose=verbose, dry_run=dry_run,
        )
        if chained and chained not in seen:
            seen.add(chained)
            all_deaths.append(chained)
            # Chained death triggers ability_window only (no last_words)
            queue.append((chained, False))

    return all_deaths


def reveal_pending_night_deaths(
    state: GameState,
    agents: Dict[str, LLMAgent],
    pending_deaths: List[str],
    night: int,
    day: int,
    verbose: bool,
    dry_run: bool,
    nl_log: Optional[List[dict]] = None,
) -> List[str]:
    """Backwards-compatible wrapper around process_death_chain for night
    reveal. Returns the list of all deaths (initial + chained)."""
    return process_death_chain(
        state, agents, pending_deaths,
        is_night_reveal=True,
        night=night, day=day,
        verbose=verbose, dry_run=dry_run,
        nl_log=nl_log,
    )


# ═══════════════════════════════════════════════════════════════
# Night phase helper
# ═══════════════════════════════════════════════════════════════

def run_night_phase(
    state: GameState,
    agents: Dict[str, LLMAgent],
    timeline_order: List[str],
    group_headers: Dict[str, dict],
    verbose: bool,
    dry_run: bool,
) -> tuple:
    """Execute one night phase. Returns (NightResult, pending_deaths)."""
    night = state.start_night()
    timeline = f"night_{night}"
    timeline_order.append(timeline)

    if verbose:
        print(f"\n--- Night {night} ---")
        print(f"  Alive: {state.get_alive()}")

    # Night 0: agents must select targets randomly (no evidence yet)
    n0_hint = ""
    if night == 0:
        n0_hint = (
            "\n\nIMPORTANT: It is the first night and you have NO evidence about "
            "any player. You MUST select your target RANDOMLY from the valid targets. "
            "Do not apply strategy — just pick one at random."
        )

    wolf_target = None
    seer_target = None
    guard_target = None

    # Werewolf attack — wolves keep voting until they reach consensus
    # (or they unanimously abstain). All alive players are valid targets
    # including self and wolf teammates. A safety cap prevents pathological loops.
    wolves = [p for p in state.get_alive() if state.roles[p] in WOLF_ROLES]
    WOLF_VOTE_SAFETY_CAP = 50
    if wolves:
        targets = [p for p in state.get_alive() if state.roles[p] not in WOLF_ROLES]
        random.shuffle(targets)
        # Each wolf may also vote for themselves
        valid_targets = list(targets) + [w for w in wolves]
        wolf_votes: Dict[str, str] = {}
        wolf_tally: Dict[str, int] = {}
        result = "abstain"
        for vote_round in range(1, WOLF_VOTE_SAFETY_CAP + 1):
            prev_round_votes = dict(wolf_votes)
            wolf_votes = {}
            round_context = n0_hint
            if vote_round > 1:
                tally_str = ", ".join(
                    f"{w}->{t}" for w, t in sorted(prev_round_votes.items())
                )
                round_context += (
                    f"\n\nThe wolf vote did not reach consensus (round {vote_round}). "
                    f"Previous round: {tally_str}. "
                    f"Coordinate with your teammates and vote again. Choose the target "
                    f"that best helps the wolf camp, or vote 'none' for no kill."
                )
            for wolf in wolves:
                preds = agents[wolf].act(state, "night_wolf", valid_targets,
                                         dry_run=dry_run, extra_context=round_context)
                vote_target = "abstain"
                if preds:
                    if "none" in preds[0].lower():
                        vote_target = "abstain"
                    else:
                        vote_target = extract_attack_target(preds[0]) or "abstain"
                if vote_target != "abstain" and vote_target not in valid_targets:
                    vote_target = "abstain"
                wolf_votes[wolf] = vote_target
            result, wolf_tally = resolve_wolf_votes(wolf_votes)
            if result != "tie":
                break
            if verbose:
                print(f"  Wolf vote tied (round {vote_round}); revoting. Tally: {wolf_tally}")
        wolf_target = None if result in ("abstain", "tie") else result
        for wolf, vote_target in wolf_votes.items():
            agents[wolf].record_own_action(f"attacked(N{night}, {vote_target})")
        if verbose:
            print(f"  Wolf votes (final): {wolf_votes}")
            print(f"  Wolf vote tally: {wolf_tally}")
            print(f"  Wolves attack: {wolf_target or '(no kill)'}")

    # Seer divine
    seers = state.get_alive_with_role("seer")
    if seers:
        seer = seers[0]
        targets = [p for p in state.get_alive() if p != seer]
        random.shuffle(targets)
        preds = agents[seer].act(state, "night_seer", targets, dry_run=dry_run,
                                 extra_context=n0_hint)
        if preds:
            seer_target = extract_night_target(preds[0])
        if not seer_target and targets:
            seer_target = random.choice(targets)
        if verbose:
            print(f"  Seer {seer} divines: {seer_target}")
        agents[seer].record_own_action(f"divine(N{night}, {seer}, {seer_target})")

    # Guard protect (cannot protect same player as last night)
    guards = state.get_alive_with_role("guard")
    if guards:
        guard = guards[0]
        targets = list(state.get_valid_guard_targets())
        random.shuffle(targets)
        preds = agents[guard].act(state, "night_guard", targets, dry_run=dry_run,
                                  extra_context=n0_hint)
        if preds:
            guard_target = extract_night_target(preds[0])
        if guard_target and guard_target == state.last_guard_target:
            guard_target = None
        if not guard_target and targets:
            guard_target = random.choice(targets)
        if verbose:
            print(f"  Guard {guard} protects: {guard_target}")
        agents[guard].record_own_action(f"guard(N{night}, {guard}, {guard_target})")

    # Witch action (after wolf/seer/guard, witch sees who was attacked)
    witch_save = False
    witch_poison = None
    witches = state.get_alive_with_role("witch")
    if witches:
        witch = witches[0]
        targets = [p for p in state.get_alive() if p != witch]
        random.shuffle(targets)
        # Tell witch who was attacked (extra context)
        extra = f"\nTonight's victim: {wolf_target or 'no one'}."
        if state.witch_antidote_used:
            extra += "\nYour antidote has already been used."
        else:
            extra += "\nYou still have your antidote."
        if state.witch_poison_used:
            extra += "\nYour poison has already been used."
        else:
            extra += "\nYou still have your poison."
        preds = agents[witch].act(state, "night_witch", targets, dry_run=dry_run,
                                  extra_context=n0_hint + extra)
        for pred in (preds or []):
            if extract_witch_save(pred) and not state.witch_antidote_used:
                witch_save = True
            pt = extract_witch_poison_target(pred)
            if pt and not state.witch_poison_used:
                witch_poison = pt
        if verbose:
            if witch_save:
                print(f"  Witch {witch} saves the victim!")
            if witch_poison:
                print(f"  Witch {witch} poisons: {witch_poison}")
            if not witch_save and not witch_poison:
                print(f"  Witch {witch} does nothing.")
        if witch_save:
            agents[witch].record_own_action(f"antidote(N{night}, {witch})")
        if witch_poison:
            agents[witch].record_own_action(f"poison(N{night}, {witch}, {witch_poison})")

    # Resolve night. Engine records hidden results and mutates deaths, then the
    # experiment restores pending deaths so morning reveal follows platform flow.
    history_start = len(state.public_history)
    night_result = state.resolve_night(
        wolf_target, seer_target, guard_target,
        witch_save=witch_save, witch_poison=witch_poison,
    )
    if night_result.killed:
        print(f"  >> {night_result.killed} killed!")
    elif wolf_target:
        print(f"  >> {wolf_target} was saved!")
    if witch_poison and witch_poison in state.dead:
        print(f"  >> {witch_poison} poisoned by witch!")
    if verbose and night_result.seer_result:
        print(f"  >> Seer learns: {seer_target} is {night_result.seer_result}")

    # Seed seer's divination as a private fact
    if night_result.seer_result and seers:
        seer = seers[0]
        agents[seer].receive_fact(
            f"divine(N{night}, {seer}, {seer_target}, {night_result.seer_result})"
        )

    night_dead = []
    if night_result.killed:
        night_dead.append(night_result.killed)
    if night_result.witch_poison_target and night_result.witch_poison_target in state.dead:
        if night_result.witch_poison_target != night_result.killed:
            night_dead.append(night_result.witch_poison_target)
    night_dead = dedupe_keep_order(night_dead)

    del state.public_history[history_start:]
    for killed in night_dead:
        state.alive.add(killed)
        state.dead.discard(killed)
    state.pending_night_deaths = set(night_dead)

    # Capture group header for night file (alive/dead/sheriff at start of night)
    group_headers[timeline] = capture_group_header(state)

    return night_result, night_dead


# ═══════════════════════════════════════════════════════════════
# Game runner
# ═══════════════════════════════════════════════════════════════

def run_single_game(
    game_id: str,
    model: str,
    client,  # LLMClient | None
    prompts: dict,
    *,
    num_players: int = 7,
    temperature: float = 0,
    max_tokens: int = 300,
    max_rounds: int = 20,
    verbose: bool = False,
    dry_run: bool = False,
    experiment_type: str = "pred_only",
    seed: Optional[str] = None,
    token_budget: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    rewriter=None,
) -> dict:

    if seed is not None:
        random.seed(seed)  # random.seed accepts strings

    t_start = time.time()

    # speech_mode: "natural" enables Stage 2 NL generation
    speech_mode = "natural" if experiment_type in NL_EXPERIMENT_TYPES else "predicate"
    experiment_full = EXPERIMENT_TYPES.get(experiment_type, experiment_type)

    players = [f"P{i}" for i in range(1, num_players + 1)]
    roles = randomize_roles(players)
    state = GameState(game_id, players, roles)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Game {game_id}")
        print(f"Seed: {seed}")
        print(f"Roles: {roles}")
        print(f"Model: {model}  Experiment: {experiment_full}")
        if token_budget:
            print(f"Token budget: {token_budget:,}")
        print(f"{'='*60}")

    tracker = TokenTracker(model, max_tokens=token_budget)

    agents: Dict[str, LLMAgent] = {}
    for p in players:
        agents[p] = LLMAgent(
            player_id=p, role=roles[p], game_id=game_id,
            model=model, client=client, prompts=prompts,
            players=players,
            temperature=temperature, max_tokens=max_tokens,
            token_tracker=tracker,
            speech_mode=speech_mode,
        )

    if rewriter is not None:
        for agent in agents.values():
            agent.rewriter = rewriter

    # Seed initial facts
    init_agent_facts(agents, state)

    timeline_order: List[str] = []
    group_headers: Dict[str, dict] = {}  # group_key -> {alive_players, dead_players, sheriff}
    reasoning_timeline: List[dict] = []

    per_phase_predicates: Dict[str, List[str]] = {}
    _ph_mark = [0]
    nl_log: List[dict] = []  # NL speech records for game_nl.json

    def _snap_phase(tl: str):
        """Capture new predicates added since last snapshot."""
        new = state.public_history[_ph_mark[0]:]
        per_phase_predicates[tl] = list(new)
        _ph_mark[0] = len(state.public_history)

    def _log_nl(phase: str, speaker: str, agent: "LLMAgent") -> None:
        """Record NL speech items from agent.last_speech into nl_log."""
        items = agent.last_speech or []
        if not items:
            return
        nl_log.append({
            "phase": phase,
            "speaker": speaker,
            "speech": [{"pred": it.get("pred", ""), "text": it.get("text", "")}
                       for it in items],
        })


    ckpt_meta = {
        "model": model,
        "experiment_type": experiment_type,
        "seed": seed,
        "game_id": game_id,
        "num_players": num_players,
        "token_budget": token_budget,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    def _save_ckpt(phase_key: str, extra_meta: dict | None = None):
        if checkpoint_dir is None:
            return
        meta = {**ckpt_meta, **(extra_meta or {})}
        path = Path(checkpoint_dir) / f"{phase_checkpoint_name(phase_key)}.pkl"
        save_checkpoint(
            path=path,
            phase_name=phase_key,
            state=state,
            agents=agents,
            tracker=tracker,
            timeline_order=timeline_order,
            group_headers=group_headers,
            per_phase_predicates=per_phase_predicates,
            reasoning_timeline=reasoning_timeline,
            ph_mark=_ph_mark[0],
            meta=meta,
        )

    # ════════════ NIGHT 0 ════════════
    _, pending_night_deaths = run_night_phase(
        state, agents, timeline_order, group_headers, verbose, dry_run
    )
    pending_night = state.night
    winner = None

    _snap_phase("night_0")

    # ── Game loop: D1 -> N1 -> D2 -> N2 -> ... ──
    for round_num in range(1, max_rounds + 1):
        if winner:
            break

        # ════════════ DAY PHASE ════════════
        day = state.start_day()

        # ── Sheriff election on Day 1 (before night-death announcements) ──
        if day == 1:
            el_timeline = "sheriff_election"
            timeline_order.append(el_timeline)
            group_headers[el_timeline] = capture_group_header(state)
            if verbose:
                print(f"\n--- Sheriff Election ---")

            candidates: List[str] = []
            for p in state.get_alive():
                preds = agents[p].act(state, "sheriff_nominate", dry_run=dry_run)
                if preds and extract_run_sheriff(preds[0]):
                    candidates.append(p)
            if verbose:
                print(f"  Candidates: {candidates}")
            for c in candidates:
                state.public_history.append(f"run_sheriff(D{day}, {c})")

            election_abandoned = False
            elected_pid: Optional[str] = None

            if not candidates:
                if verbose:
                    print("  No candidates; no sheriff this game.")
            elif len(candidates) == 1:
                # Single-candidate fast path: auto-elect, no speeches/vote.
                result = state.resolve_sheriff_election(candidates, {})
                elected_pid = result.elected
                if elected_pid:
                    broadcast_fact(agents, state, f"sheriff_set(D{day}, {elected_pid})")
                if verbose:
                    print(f"  Sole candidate {elected_pid} auto-elected.")
            else:
                # Campaign speeches; WWK may self-explode before any speaker.
                # Wolf self-destruct during election → abandon, no badge.
                for c in candidates:
                    if check_wwk_explosion(state, agents, verbose, dry_run):
                        state.abandon_sheriff_election("wolf_explosion")
                        election_abandoned = True
                        break
                    if not state.is_alive(c):
                        continue
                    agents[c].update_memory(state, f"day_{state.day}", dry_run=dry_run)
                    running_now = [x for x in candidates if state.is_alive(x)]
                    extra = f"Players currently running for Sheriff (cannot vote): {', '.join(running_now)}"
                    c_idx = candidates.index(c)
                    remaining_el = [x for x in candidates[c_idx + 1:] if state.is_alive(x)]
                    if remaining_el:
                        extra += f"\nCandidates who have not yet given their speech: {', '.join(remaining_el)}"
                    preds = agents[c].act(state, "sheriff_election_speech",
                                          extra_context=extra, dry_run=dry_run)
                    flat = state.record_speech(c, preds, speech=agents[c].last_speech)
                    agents[c].record_own_statement(el_timeline, flat)
                    _log_nl(el_timeline, c, agents[c])
                    speaker_nl = " ".join(
                        (item.get("text") or "").strip()
                        for item in (agents[c].last_speech or [])
                    ).strip()
                    if speaker_nl:
                        broadcast_speech_nl(agents, state, c, speaker_nl,
                                            dry_run=dry_run)
                    if verbose:
                        for line in flat:
                            print(f"  {line}")
                state.flush_turn_buffer()

                if not election_abandoned:
                    # Withdrawal window: candidates may withdraw before the vote
                    withdrawn: set = set()
                    for c in list(candidates):
                        if not state.is_alive(c):
                            continue
                        wd_preds = agents[c].act(state, "sheriff_withdraw", dry_run=dry_run)
                        if any(extract_withdraw_sheriff(p) for p in wd_preds):
                            withdrawn.add(c)
                            candidates.remove(c)
                            state.public_history.append(f"withdraw_sheriff(D{day}, {c})")
                            if verbose:
                                print(f"  {c} withdrew from the election.")
                    voting_candidates = [c for c in candidates if state.is_alive(c)]
                    election_votes: Dict[str, str] = {}
                    for voter in state.get_alive():
                        if voter in voting_candidates or voter in withdrawn:
                            continue  # active candidates and withdrawn candidates do not vote
                        preds = agents[voter].act(
                            state, "sheriff_election_vote",
                            voting_candidates, dry_run=dry_run,
                        )
                        target = extract_vote_target(preds[0]) if preds else None
                        if (not target or target not in voting_candidates) and voting_candidates:
                            target = random.choice(voting_candidates)
                        elif not target or target not in voting_candidates:
                            target = "abstain"
                        election_votes[voter] = target

                    print(f"  Election votes: {election_votes}")

                    result = state.resolve_sheriff_election(
                        voting_candidates, election_votes
                    )
                    print(_fmt_vote_counts(result.vote_counts))

                    # Tie: pick a random winner from tied candidates, no PK round
                    if result.tied_candidates and not election_abandoned:
                        winner = random.choice(list(result.tied_candidates))
                        print(f"  Sheriff vote tied: {result.tied_candidates}; random winner: {winner}.")
                        state.sheriff = winner
                        state.sheriff_history.append(winner)
                        state.public_history.append(f"sheriff_set(D{state.day}, {winner})")
                        state.public_history.append(
                            f"announcement(D{state.day}, {winner}, sheriff): "
                            f"{winner} is now sheriff (random tie-break)."
                        )
                        result.elected = winner
                    # # PK round on first-round tie
                    # if result.tied_candidates and not election_abandoned:
                    #     if verbose:
                    #         print(f"  Sheriff vote tied: {result.tied_candidates}; PK.")
                    #     pk_speakers = list(result.tied_candidates)
                    #     for c in pk_speakers:
                    #         if check_wwk_explosion(state, agents, verbose, dry_run):
                    #             state.abandon_sheriff_election("wolf_explosion")
                    #             election_abandoned = True
                    #             break
                    #         if not state.is_alive(c):
                    #             continue
                    #         agents[c].update_memory(state, f"day_{state.day}", dry_run=dry_run)
                    #         running_now = [x for x in pk_speakers if state.is_alive(x)]
                    #         extra = f"Players currently running for Sheriff (cannot vote): {', '.join(running_now)}"
                    #         preds = agents[c].act(state, "sheriff_election_speech",
                    #                               extra_context=extra, dry_run=dry_run)
                    #         flat = state.record_speech(c, preds,
                    #                                    speech=agents[c].last_speech)
                    #         agents[c].record_own_statement(el_timeline, flat)
                    #         _log_nl(f"{el_timeline}_pk", c, agents[c])
                    #         speaker_nl = " ".join(
                    #             (item.get("text") or "").strip()
                    #             for item in (agents[c].last_speech or [])
                    #         ).strip()
                    #         if speaker_nl:
                    #             broadcast_speech_nl(agents, state, c, speaker_nl,
                    #                                 dry_run=dry_run)
                    #         if verbose:
                    #             for line in flat:
                    #                 print(f"  {line}")
                    #
                    #     if not election_abandoned:
                    #         state.flush_turn_buffer()
                    #         pk_candidates = [c for c in pk_speakers if state.is_alive(c)]
                    #         pk_votes: Dict[str, str] = {}
                    #         for voter in state.get_alive():
                    #             if voter in pk_candidates:
                    #                 continue
                    #             preds = agents[voter].act(
                    #                 state, "sheriff_election_vote",
                    #                 pk_candidates, dry_run=dry_run,
                    #             )
                    #             target = extract_vote_target(preds[0]) if preds else None
                    #             if not target or target not in pk_candidates:
                    #                 target = "abstain"
                    #             pk_votes[voter] = target
                    #         print(f"  PK election votes: {pk_votes}")
                    #         result = state.resolve_sheriff_election(
                    #             pk_candidates, pk_votes, is_pk_round=True,
                    #         )
                    elected_pid = result.elected
                    if elected_pid:
                        broadcast_fact(agents, state, f"sheriff_set(D{day}, {elected_pid})")
                    if elected_pid:
                        print(f"  Elected sheriff: {elected_pid}")
                    else:
                        print("  No sheriff (election failed).")

            _snap_phase(el_timeline)
            group_headers[el_timeline]["sheriff"] = state.sheriff
            _save_ckpt("sheriff_election")

        # Morning reveal. Day 1 runs sheriff election before these deaths become public.
        announce_timeline = f"day_{day}_announce"
        timeline_order.append(announce_timeline)
        # Keep original primary deaths before chain (for anchor calculation below)
        _primary_night_deaths = list(pending_night_deaths)
        reveal_pending_night_deaths(
            state, agents, pending_night_deaths, pending_night,
            day, verbose, dry_run,
            nl_log=nl_log,
        )
        state.pending_night_deaths.clear()
        winner = state.check_win()
        if verbose:
            if winner:
                print(f"  [Announcement] Game over — {winner} wins!")
            else:
                print(f"  [Announcement] Game continues.")
        _snap_phase(announce_timeline)

        # Capture day group header after public night deaths are applied.
        group_headers[f"day_{day}"] = capture_group_header(state)
        if verbose:
            print(f"  Alive: {state.get_alive()}")

        if winner:
            break

        # (N0 last_words now handled inside process_death_chain → reveal_pending_night_deaths.)

        # ── Update agent memory BEFORE discussion ──
        time_str = f"day_{day}"
        phase_snapshots = {}
        for p in state.get_alive():
            snapshot = agents[p].update_memory(state, time_str, dry_run=dry_run)
            phase_snapshots[p] = snapshot
        reasoning_timeline.append({
            "timeline": time_str,
            "agents": phase_snapshots,
        })

        # ── Sheriff decides speech direction (left/right) ──
        direction = None
        if state.sheriff and state.is_alive(state.sheriff):
            preds = agents[state.sheriff].act(state, "sheriff_order", dry_run=dry_run)
            if preds:
                direction = extract_sheriff_direction(preds[0])
                state.public_history.append(preds[0])
            if verbose and direction:
                print(f"  Sheriff {state.sheriff} orders: {direction}")

        # Compute anchor from last night's primary deaths + sheriff's direction choice.
        # Exactly 1 death: anchor = first alive neighbor of the dead seat in that direction.
        # 0 or 2+ deaths: no fixed anchor (engine starts from first alive seat).
        anchor = None
        if direction and len(_primary_night_deaths) == 1:
            dead_seat = _primary_night_deaths[0]
            all_seats = state.players  # full seat-order list (alive + dead)
            if dead_seat in all_seats:
                dead_idx = all_seats.index(dead_seat)
                step = 1 if direction == "right" else -1
                n = len(all_seats)
                for k in range(1, n + 1):
                    candidate = all_seats[(dead_idx + step * k) % n]
                    if state.is_alive(candidate):
                        anchor = candidate
                        break

        order = state.get_speech_order(direction, anchor)

        _save_ckpt(f"day_{day}", extra_meta={
            "speech_order": list(order),
            "speech_direction": direction,
        })

        # ── Discussion ── sheriff is included and speaks last (engine
        # already orders the speech list with sheriff at the end).
        discussion_timeline = f"day_{day}_discussion"
        timeline_order.append(discussion_timeline)
        if verbose:
            print(f"\n--- Day {day} ---")
            print(f"\n  == Day {day} Discussion ==")
        wwk_exploded = False
        for speaker_idx, speaker in enumerate(order):
            if not state.is_alive(speaker):
                continue
            # WWK explosion check before each speaker
            if check_wwk_explosion(state, agents, verbose, dry_run):
                wwk_exploded = True
                break
            remaining = [p for p in order[speaker_idx + 1:] if state.is_alive(p)]
            speech_extra = (f"Players who have not yet spoken this turn: {', '.join(remaining)}"
                            if remaining else "")
            alive_others = [p for p in state.get_alive() if p != speaker]
            preds = agents[speaker].act(state, "day_speech", alive_others, extra_context=speech_extra,
                                        dry_run=dry_run)
            flat = state.record_speech(speaker, preds,
                                       speech=agents[speaker].last_speech)
            agents[speaker].record_own_statement(discussion_timeline, flat)
            _log_nl(discussion_timeline, speaker, agents[speaker])
            # Listener-side parse: each other LLM listener parses speaker's
            # natural-language speech into their own predicates.
            speaker_nl = " ".join(
                (item.get("text") or "").strip()
                for item in (agents[speaker].last_speech or [])
            ).strip()
            if speaker_nl:
                broadcast_speech_nl(agents, state, speaker, speaker_nl, dry_run=dry_run)
            if verbose:
                for line in flat:
                    print(f"  {line}")

        state.flush_turn_buffer()
        _snap_phase(discussion_timeline)

        # Restore discussion into turn buffer so voters can read it during day_vote
        state.current_turn_buffer = list(per_phase_predicates[discussion_timeline])

        # If WWK exploded during discussion, skip rest of day → go to night
        if wwk_exploded:
            state.current_turn_buffer.clear()
            winner = state.check_win()
            if verbose:
                if winner:
                    print(f"  [Announcement] Game over — {winner} wins!")
                else:
                    print(f"  [Announcement] WWK exploded! Skipping to night.")
            if winner:
                _snap_phase(f"day_{day}_vote")  # snap empty vote phase
                break
            # Skip directly to night
            _, pending_night_deaths = run_night_phase(
                state, agents, timeline_order, group_headers, verbose, dry_run
            )
            pending_night = state.night
            _snap_phase(f"night_{state.night}")
            continue  # skip to next day in the game loop

        # (sheriff_summary removed: the sheriff is now the last speaker
        # of the discussion phase per platform rules.)

        # ── VOTING ──
        # (WWK can only self-explode during the speech/discussion phase;
        #  no explosion check here.)
        vote_timeline = f"day_{day}_vote"
        timeline_order.append(vote_timeline)
        state.start_voting()
        if verbose:
            print(f"\n  == Day {day} Vote ==")

        def _collect_day_votes(eligible_voters: List[str],
                               valid_target_pool: List[str]) -> Dict[str, Optional[str]]:
            collected: Dict[str, Optional[str]] = {}
            for voter in eligible_voters:
                allowed = [p for p in valid_target_pool if p != voter]
                preds = agents[voter].act(
                    state, "day_vote", allowed, dry_run=dry_run
                )
                target = extract_vote_target(preds[0]) if preds else None
                if not target or target.lower() in ("none", "null", "abstain") \
                        or target not in allowed:
                    collected[voter] = None
                else:
                    collected[voter] = target
            return collected

        # Round 1: every alive player may vote on every other alive player.
        votes = _collect_day_votes(
            list(state.get_alive()), list(state.get_alive())
        )
        day_result = state.resolve_votes(votes)
        print(f"  Votes: {votes}")
        print(_fmt_vote_counts(day_result.vote_counts))
        if day_result.tied_candidates:
            print(f"  Tied: {day_result.tied_candidates}; "
                  f"random tiebreak → {day_result.executed}")
        else:
            print(f"  Executed: {day_result.executed}")

        state.current_turn_buffer.clear()

        executed = day_result.executed
        broadcast_fact(agents, state, f"executed(D{day}, {executed})")
        if executed in agents:
            agents[executed].receive_fact(f"executed(D{day}, {executed})")
        process_death_chain(
            state, agents, [executed],
            is_night_reveal=False,
            night=state.night, day=day,
            verbose=verbose, dry_run=dry_run,
            nl_log=nl_log,
        )

        winner = state.check_win()
        if verbose:
            if winner:
                print(f"  [Announcement] Game over — {winner} wins!")
            else:
                print(f"  [Announcement] Game continues.")

        _snap_phase(vote_timeline)

        if winner:
            break

        # ════════════ NIGHT PHASE ════════════
        _, pending_night_deaths = run_night_phase(
            state, agents, timeline_order, group_headers, verbose, dry_run
        )
        pending_night = state.night
        _snap_phase(f"night_{state.night}")

    else:
        if not winner:
            if verbose:
                print(f"\n** MAX ROUNDS ({max_rounds}) reached -- cancelled **")
            state.winner = "cancelled"

    # ── Build final outputs ──
    public_history = build_public_history(state, per_phase_predicates)

    by_agent_memory = {}
    for p in players:
        by_agent_memory[p] = build_agent_memory(
            p, roles[p],
            fact=agents[p].fact,
            potential_truth=agents[p].potential_truth,
            potential_deception=agents[p].potential_deception,
            reliability=agents[p].reliability,
            reasoning_history=agents[p].reasoning_history,
            relationship_estimates=agents[p].relationship_estimates,
            action_history=agents[p].action_history,
            statement_history=agents[p].statement_history,
            thinking_traces=agents[p].thinking_traces,
        )

    game_record = build_game_record(
        state, experiment_full, model, seed=seed
    )

    elapsed = time.time() - t_start

    return {
        "state": state,
        "public_history": public_history,
        "per_phase_predicates": per_phase_predicates,
        "group_headers": group_headers,
        "game_record": game_record,
        "by_agent_memory": by_agent_memory,
        "agents": agents,
        "token_tracker": tracker,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "experiment_type": experiment_full,
        "model": model,
        "timeline_order": timeline_order,
        "reasoning_timeline": reasoning_timeline,
        "nl_log": nl_log,
        # Treatment provenance: which rewriter (if any) produced the NL in this run.
        # Downstream source collection needs this to avoid reusing TREATED speech
        # as untreated source material.
        "rewriter": rewriter.to_config() if rewriter is not None else None,
    }


# ═══════════════════════════════════════════════════════════════
# Output saving
# ═══════════════════════════════════════════════════════════════

_COMPACT_FIELDS = {"potential_truth", "potential_deception"}

def _dump_agent_memory(memory: dict) -> str:
    """Serialize agent memory with potential_truth/deception items on one line each."""
    s = json.dumps(memory, ensure_ascii=False, indent=2)
    for field in _COMPACT_FIELDS:
        def _compact(m, _field=field):
            inner = m.group(1)
            if not inner.strip():
                return m.group(0)
            items = re.findall(r'\{[^{}]+\}', inner, re.DOTALL)
            lines = ["    " + json.dumps(json.loads(it), ensure_ascii=False)
                     for it in items]
            return f'"{_field}": [\n' + ",\n".join(lines) + "\n  ]"
        s = re.sub(
            rf'"{re.escape(field)}":\s*\[(.*?)\]',
            _compact, s, flags=re.DOTALL,
        )
    return s

def _compact_json(data: dict) -> str:
    """Serialize any game JSON with compact player lists (one line each).

    Collapses ["P1", "P2", ...] arrays onto a single line.
    Used for public_history, current_turn, and game_record.
    """
    s = json.dumps(data, ensure_ascii=False, indent=2)
    def _compact_player_array(m):
        items = re.findall(r'"[^"]+"', m.group(0))
        return "[" + ", ".join(items) + "]"
    s = re.sub(
        r'\[\s*"P\d+"(?:\s*,\s*"P\d+")*\s*\]',
        _compact_player_array, s, flags=re.DOTALL,
    )
    return s


def _build_current_turn_files(
    per_phase_predicates: dict,
    group_headers: dict,
    timeline_order: list,
) -> dict:
    """Build current_turn file contents from per_phase_predicates + group headers.

    Groups phases into files:
      - sheriff_election -> sheriff_election.json (standalone)
      - night_N_last_words, day_N_* -> day_N.json
      - night_N -> night_N.json

    Each file = {alive_players, dead_players, sheriff, phase1: [...], phase2: [...]}
    where phase arrays are flat predicate lists identical to public_history.
    """
    # Map phase_key -> group file key
    def _group_for(phase_key: str) -> str:
        if phase_key == "sheriff_election":
            return "sheriff_election"
        if phase_key.startswith("night_"):
            parts = phase_key.split("_")
            # "night_0_last_words" -> group "day_1" (it's announced on day 1)
            if len(parts) > 2:
                # night_N_last_words belongs to day_(N+1)
                night_num = int(parts[1])
                return f"day_{night_num + 1}"
            return f"night_{parts[1]}"
        if phase_key.startswith("day_"):
            parts = phase_key.split("_")
            return f"day_{parts[1]}"
        return phase_key

    groups = {}  # group_key -> {header + phase arrays}
    for phase_key in timeline_order:
        if phase_key not in per_phase_predicates:
            continue
        group_key = _group_for(phase_key)
        if group_key not in groups:
            # Find header: use group_key if available, else use phase_key
            header = group_headers.get(group_key, group_headers.get(phase_key, {}))
            groups[group_key] = dict(header)  # copy alive/dead/sheriff
        groups[group_key][phase_key] = per_phase_predicates[phase_key]
    return groups

def save_game_output(result: dict, out_dir: Path,
                     terminal_output: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = result.get("seed")

    # Public history
    with open(out_dir / "public_history.json", "w", encoding="utf-8") as f:
        f.write(_compact_json(result["public_history"]))

    # Current turn — one file per day/night/election with phase groupings
    ct_dir = out_dir / "current_turn"
    ct_dir.mkdir(exist_ok=True)
    ct_files = _build_current_turn_files(
        result.get("per_phase_predicates", {}),
        result.get("group_headers", {}),
        result.get("timeline_order", []),
    )
    for group_key, file_data in ct_files.items():
        with open(ct_dir / f"{group_key}.json", "w", encoding="utf-8") as f:
            f.write(_compact_json(file_data))

    # Agent memory
    mem_dir = out_dir / "agent_memory"
    mem_dir.mkdir(exist_ok=True)
    for player, memory in result["by_agent_memory"].items():
        with open(mem_dir / f"{player}.json", "w", encoding="utf-8") as f:
            f.write(_dump_agent_memory(memory))

    with open(out_dir / "game_record.json", "w", encoding="utf-8") as f:
        f.write(_compact_json(result["game_record"]))

    # Game log (raw LLM calls) — seed is written as first line
    agents = result.get("agents", {})
    with open(out_dir / "game_log.jsonl", "w", encoding="utf-8") as f:
        if seed is not None:
            f.write(json.dumps({"random_seed": seed}, ensure_ascii=False) + "\n")
        if agents:
            for player, agent in agents.items():
                for entry in agent.call_log:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Resources (tokens + time)
    resources = {}
    tracker = result.get("token_tracker")
    if tracker:
        resources["tokens"] = tracker.summary()
    elapsed = result.get("elapsed_seconds")
    if elapsed is not None:
        resources["elapsed_seconds"] = elapsed
    if seed is not None:
        resources["random_seed"] = seed
    if resources:
        with open(out_dir / "resources.json", "w", encoding="utf-8") as f:
            json.dump(resources, f, ensure_ascii=False, indent=2)

    # Natural-language speech log (only written when there is NL content)
    nl_log = result.get("nl_log") or []
    if nl_log:
        with open(out_dir / "game_nl.json", "w", encoding="utf-8") as f:
            f.write(_compact_json(nl_log))

    # Terminal output (.out file)
    if terminal_output:
        with open(out_dir / "game.out", "w", encoding="utf-8") as f:
            f.write(terminal_output)

    # Manifest
    gr = result["game_record"]
    state = result["state"]
    manifest = {
        "game_id": gr["game_id"],
        "experiment_type": gr["experiment_type"],
        "model": gr["model"],
        "num_players": gr["num_players"],
        "game_state_status": state.winner or "cancelled",
        "files": {
            "game_record": "game_record.json",
            "game_log": "game_log.jsonl",
            "resources": "resources.json",
            "public_history": "public_history.json",
            "current_turn": {
                gk: f"current_turn/{gk}.json"
                for gk in _build_current_turn_files(
                    result.get("per_phase_predicates", {}),
                    result.get("group_headers", {}),
                    result.get("timeline_order", []),
                )
            },
            "agent_memory": {
                p: f"agent_memory/{p}.json"
                for p in result["by_agent_memory"]
            },
        },
    }
    if seed is not None:
        manifest["random_seed"] = seed
    if nl_log:
        manifest["files"]["game_nl"] = "game_nl.json"
    # Null when the run is untreated; a config dict when a rewriter rewrote the NL.
    manifest["rewriter"] = result.get("rewriter")
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════════
# Stdout Tee for capturing terminal output
# ═══════════════════════════════════════════════════════════════

class _Tee:
    """Write to multiple streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(
        description="Run predicate-only Werewolf games with LLM agents.",
    )
    parser.add_argument("--model", default=None,
                        help="Model name (default: $LLM_MODEL or Qwen/Qwen3.5-27B)")
    parser.add_argument("--players", type=int, default=7, choices=range(7, 13),
                        help="Number of players (7-12, default: 7)")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--game-id", default="g")
    parser.add_argument("--output", "-o")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiment",
                        choices=list(EXPERIMENT_TYPES.keys()),
                        default="pred_only",
                        help="pred_only (predicate only, default) or nl (natural language)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Random seed string (default: auto-generated MMDDHHMM)")
    parser.add_argument("--token-budget", type=int, default=None,
                        help="Max total tokens per game (aborts if exceeded)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking on thinking models (faster; sets DISABLE_THINKING=1)")
    parser.add_argument("--url", default=None,
                        help="Model server URL (default: $LLM_BASE_URL or http://localhost:8000)")
    parser.add_argument("--api-key", default=None,
                        help="Model server API key (default: $LLM_API_KEY or EMPTY)")
    parser.add_argument("--save-checkpoints", action="store_true",
                        help="Save state snapshots to <game_dir>/checkpoints/ at each "
                             "phase boundary. Required for eval replays.")
    parser.add_argument("--rewriter", default=None,
                        help="Rewriter name from the registry (e.g. 'formality')")
    parser.add_argument("--rewriter-params", default="{}",
                        help="JSON string of rewriter parameters")
    parser.add_argument("--rewriter-module", default=None,
                        help="Python module to import for rewriter registration "
                             "(e.g. 'examples.formality.rewriter')")
    parser.add_argument("--config", default=None,
                        help="Path to YAML experiment config (overrides CLI flags)")

    args = parser.parse_args()

    if args.config:
        from .config import load_config as _load_config
        from .eval.runner import run_experiment
        _cfg = _load_config(args.config)
        config_dir = Path(args.config).resolve().parent
        _seed = args.seed or _cfg.seeds[0]
        output_root = Path(args.output) if args.output else (
            config_dir / "games" / f"{_cfg.name}_{_seed}"
        )
        run_experiment(
            _cfg,
            seed=_seed,
            url=args.url,
            api_key=args.api_key,
            token_budget=args.token_budget,
            output_root=output_root,
        )
        return

    if args.no_thinking:
        os.environ["DISABLE_THINKING"] = "1"

    prompts = load_prompts()

    model = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen3.5-27B")
    client = None
    if not args.dry_run:
        client = LLMClient(base_url=args.url, api_key=args.api_key)

    model_slug = model.rsplit("/", 1)[-1].replace("@", "").replace(" ", "_").lower()
    exp_slug = args.experiment
    base_seed = args.seed if args.seed is not None else datetime.now().strftime("%m%d%H%M")
    run_suffix = base_seed
    base_dir = (Path(args.output) if args.output
                else SCRIPT_DIR / "games" / f"run_{model_slug}_{exp_slug}_{run_suffix}")

    print(f"Running {args.games} game(s) with model: {model}")
    print(f"Experiment: {EXPERIMENT_TYPES[args.experiment]}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"Output: {base_dir}/")
    print()

    rewriter_instance = None
    if args.rewriter:
        import json as _json
        import importlib as _importlib
        if args.rewriter_module:
            _importlib.import_module(args.rewriter_module)
        from .rewriter import get_rewriter
        rewriter_cls = get_rewriter(args.rewriter)
        rewriter_params = _json.loads(args.rewriter_params)
        rewriter_instance = rewriter_cls(**rewriter_params, llm_client=client)
        print(f"Rewriter: {args.rewriter} ({rewriter_params})")

    _old_stdout = sys.stdout

    try:
        for i in range(args.games):
            game_id = f"{args.game_id}{i + 1}"
            game_dir = base_dir / game_id if args.games > 1 else base_dir

            game_seed = f"{base_seed}_{i}" if args.games > 1 else base_seed
            ckpt_dir = game_dir / "checkpoints" if args.save_checkpoints else None
            game_dir.mkdir(parents=True, exist_ok=True)

            out_path = game_dir / "game.out"
            with open(out_path, "w", encoding="utf-8", buffering=1) as out_f:
                sys.stdout = _Tee(_old_stdout, out_f)
                try:
                    result = run_single_game(
                        game_id=game_id,
                        model=model,
                        client=client,
                        prompts=prompts,
                        num_players=args.players,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_rounds=args.max_rounds,
                        verbose=args.verbose,
                        dry_run=args.dry_run,
                        experiment_type=args.experiment,
                        seed=game_seed,
                        token_budget=args.token_budget,
                        checkpoint_dir=ckpt_dir,
                        rewriter=rewriter_instance,
                    )
                    sys.stdout.flush()
                finally:
                    sys.stdout = _old_stdout

            save_game_output(result, game_dir, terminal_output="")
            winner = result["state"].winner or "unknown"
            print(f"Game {game_id}: winner={winner} -> {game_dir}/")
            result["token_tracker"].print_summary()

        print(f"\nAll games saved to: {base_dir}/")

    finally:
        sys.stdout = _old_stdout


if __name__ == "__main__":
    main()
