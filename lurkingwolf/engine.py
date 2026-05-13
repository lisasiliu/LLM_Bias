"""
engine.py -- Werewolf game state machine and phase logic.

Pure game logic, no LLM dependency. Manages players, roles, phases,
voting, night actions, sheriff mechanics, and win conditions.

Game flow:
  N0 → D1 (sheriff election → announce → last_words → ability_window
        → discussion → vote → execution) → N1 → D2 → ...
Supports 7-12 players. Guard cannot protect the same player on consecutive nights.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# Role distributions by player count (werewolves, seer, guard, villagers)
ROLE_CONFIGS = {
    7:  {"werewolf": 2, "seer": 1, "guard": 1, "villager": 3},
    8:  {"werewolf": 3, "seer": 1, "witch": 1, "hunter": 1, "villager": 2},
    9:  {"werewolf": 3, "seer": 1, "witch": 1, "hunter": 1, "villager": 3},
    10: {"werewolf": 3, "seer": 1, "witch": 1, "hunter": 1, "guard": 1, "villager": 3},
    11: {"werewolf": 4, "seer": 1, "witch": 1, "hunter": 1, "guard": 1, "villager": 3},
    12: {"werewolf": 3, "white_wolf_king": 1, "seer": 1, "witch": 1, "hunter": 1, "guard": 1, "villager": 4},
}
VALID_ROLES = {"werewolf", "seer", "guard", "villager", "witch", "hunter", "white_wolf_king"}
WOLF_ROLES = {"werewolf", "white_wolf_king"}

SPEECH_PREDICATES = {
    "roleclaim", "estimate", "divined", "guarded",
    "question", "answer", "agree", "disagree",
    "allied", "unallied", "dissociate", "recommend_action", "address_role", "fluff",
}

# Sentinel for "no sheriff" election/succession outcome
NO_SHERIFF = "NO_SHERIFF"


def randomize_roles(players: List[str]) -> Dict[str, str]:
    """Assign roles randomly based on player count."""
    n = len(players)
    if n not in ROLE_CONFIGS:
        raise ValueError(f"Unsupported player count {n}. Supported: {sorted(ROLE_CONFIGS.keys())}")
    pool = []
    for role, count in ROLE_CONFIGS[n].items():
        pool.extend([role] * count)
    random.shuffle(pool)
    shuffled = list(players)
    random.shuffle(shuffled)
    return dict(zip(shuffled, pool))


@dataclass
class SheriffElectionResult:
    elected: Optional[str] = None
    tied_candidates: List[str] = field(default_factory=list)
    pk_round: int = 1
    vote_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class NightResult:
    night: int = 0          # night index (0 = N0, 1 = N1, ...)
    wolf_target: Optional[str] = None
    seer_target: Optional[str] = None
    seer_result: Optional[str] = None
    guard_target: Optional[str] = None
    witch_save_target: Optional[str] = None
    witch_poison_target: Optional[str] = None
    killed: Optional[str] = None


@dataclass
class DayResult:
    day: int = 0             # day index (1 = D1, 2 = D2, ...)
    sheriff: Optional[str] = None    # who was sheriff during this day
    votes: Dict[str, str] = field(default_factory=dict)
    executed: Optional[str] = None
    vote_counts: Dict[str, float] = field(default_factory=dict)
    tied_candidates: List[str] = field(default_factory=list)
    pk_round: int = 1        # 1 = first round, 2 = PK revote


class GameState:
    def __init__(self, game_id: str, players: List[str], roles: Dict[str, str]):
        self.game_id = game_id
        self.players = list(players)
        self.roles = dict(roles)
        self.alive: Set[str] = set(players)
        self.dead: Set[str] = set()
        self.pending_night_deaths: Set[str] = set()  # killed but not yet officially revealed
        self.night = -1          # incremented to 0 on first start_night()
        self.day = 0             # incremented to 1 on first start_day()
        self.phase = "setup"
        self.sheriff: Optional[str] = None
        self.sheriff_vote_weight: float = 1.5
        self.sheriff_history: List[str] = []  # ordered list of all players who held sheriff
        self.winner: Optional[str] = None  # "villagers_win", "werewolves_win", or "cancelled"

        # Guard: cannot protect the same player two nights in a row
        self.last_guard_target: Optional[str] = None

        # Witch: one-use potions
        self.witch_antidote_used: bool = False
        self.witch_poison_used: bool = False
        # Hunter: can shoot on death unless poisoned by witch
        self.hunter_can_shoot: bool = True

        # Public history — single growing list, shared by all
        self.public_history: List[str] = []
        # Current turn buffer — speeches accumulate here during a phase,
        # then get flushed to public_history when the phase ends
        self.current_turn_buffer: List[str] = []
        # Vote and night logs
        self.vote_log: List[DayResult] = []
        self.night_log: List[NightResult] = []

        # Speech act IDs -- every predicate in a speech gets a unique ID
        self.speech_id_counter: int = 0
        # Registry: maps speech ID (e.g. "s23") -> {speaker, predicate, text}
        self.speech_registry: Dict[str, dict] = {}

        # Private per-player knowledge (seer results, wolf teammates, own role)
        self.private_knowledge: Dict[str, List[str]] = {p: [] for p in players}

        # Initialize private knowledge (no game_id in predicates)
        wolves = [p for p, r in roles.items() if r in ("werewolf", "white_wolf_king")]
        for w in wolves:
            for other_w in wolves:
                if other_w != w:
                    self.private_knowledge[w].append(
                        f"teammate({w}, {other_w})"
                    )
        for p in players:
            self.private_knowledge[p].append(
                f"identity({p}, {roles[p]})"
            )

    # ── Speech IDs ──

    def next_speech_id(self) -> str:
        """Return the next unique speech act ID (s1, s2, ...)."""
        self.speech_id_counter += 1
        return f"s{self.speech_id_counter}"

    # ── Queries ──

    def get_alive(self) -> List[str]:
        return [p for p in self.players if p in self.alive]

    def get_dead(self) -> List[str]:
        return [p for p in self.players if p in self.dead]

    def get_alive_with_role(self, role: str) -> List[str]:
        return [p for p in self.alive if self.roles[p] == role]

    def is_alive(self, player: str) -> bool:
        return player in self.alive

    # ── Sheriff election (Day 1) ──

    def resolve_sheriff_election(
        self, candidates: List[str], votes: Dict[str, str],
        is_pk_round: bool = False,
    ) -> SheriffElectionResult:
        """Tally sheriff election votes.

        Returns a SheriffElectionResult. On a first-round tie, sets
        tied_candidates and leaves self.sheriff untouched so the caller
        can run a PK round. On is_pk_round=True, a tie collapses to no
        sheriff and emits a no_sheriff announcement.

        Single-candidate fast path: if exactly one candidate runs, they
        are elected without a vote (caller passes an empty votes dict).
        """
        result = SheriffElectionResult(pk_round=2 if is_pk_round else 1)
        if not candidates:
            return result

        if not is_pk_round and len(candidates) == 1:
            elected = candidates[0]
            if elected in self.alive:
                self.sheriff = elected
                self.sheriff_history.append(elected)
                self.public_history.append(f"sheriff_set(D{self.day}, {elected})")
                self.public_history.append(
                    f"announcement(D{self.day}, {elected}, sheriff): "
                    f"{elected} is now sheriff (sole candidate)."
                )
                result.elected = elected
            return result

        counts: Dict[str, int] = {}
        for voter, target in votes.items():
            self.public_history.append(
                f"voted(D{self.day}, {voter}, {target})"
            )
            if target and target != "abstain":
                counts[target] = counts.get(target, 0) + 1
        result.vote_counts = counts

        vote_dict: Dict[str, list] = {}
        for voter, target in votes.items():
            if target and target != "abstain":
                vote_dict.setdefault(target, []).append(voter)
        if vote_dict:
            self.public_history.append(vote_dict)

        if not counts:
            self.public_history.append(
                f"announcement(D{self.day}, none, no_sheriff): "
                f"Sheriff election failed (no votes)."
            )
            return result

        max_votes = max(counts.values())
        top = sorted([p for p, c in counts.items() if c == max_votes])

        if len(top) > 1:
            if is_pk_round:
                self.public_history.append(
                    f"announcement(D{self.day}, none, no_sheriff): "
                    f"Sheriff election PK still tied; no sheriff this game."
                )
            result.tied_candidates = top
            return result

        elected = top[0]
        if elected in self.alive:
            self.sheriff = elected
            self.sheriff_history.append(elected)
            self.public_history.append(f"sheriff_set(D{self.day}, {elected})")
            self.public_history.append(
                f"announcement(D{self.day}, {elected}, sheriff): "
                f"{elected} is now sheriff."
            )
            result.elected = elected
        return result

    def abandon_sheriff_election(self, reason: str = "wolf_explosion") -> None:
        """Cancel sheriff election (e.g. wolf self-explodes during election).
        Emits a no_sheriff announcement and leaves self.sheriff = None.
        """
        self.public_history.append(
            f"announcement(D{self.day}, none, no_sheriff): "
            f"Sheriff election abandoned ({reason}); no sheriff this game."
        )

    # ── Day phase ──

    def start_day(self) -> int:
        """Begin a new day. Returns the day number."""
        self.day += 1
        self.phase = "day_speech"
        return self.day

    def get_speech_order(self, direction: Optional[str] = None,
                         anchor: Optional[str] = None) -> List[str]:
        """Return discussion speaking order. Sheriff (if alive) speaks LAST.

        anchor: starting seat (e.g. dead-neighbour or sheriff). Defaults to sheriff.
        direction: 'left' (counterclockwise) or 'right' (clockwise) from anchor.
        Sheriff is appended at the end if alive, regardless of seat position.
        Anchor itself is the first speaker; if anchor is the sheriff, it still
        speaks last.
        """
        alive = self.get_alive()
        if not alive:
            return []

        if anchor is None:
            anchor = self.sheriff if self.sheriff and self.sheriff in self.alive else None

        if not anchor or not direction:
            order = list(alive)
            random.shuffle(order)
        else:
            anchor_idx = alive.index(anchor) if anchor in alive else 0
            order = []
            step = 1 if direction == "right" else -1
            i = anchor_idx
            for _ in range(len(alive)):
                order.append(alive[i])
                i = (i + step) % len(alive)

        # Sheriff always speaks last (move to end if alive and present)
        if self.sheriff and self.sheriff in self.alive and self.sheriff in order:
            order = [p for p in order if p != self.sheriff] + [self.sheriff]

        return order

    def record_speech(self, speaker: str, predicates: List[str],
                      speech: Optional[List[dict]] = None) -> List[str]:
        """Record speech into current turn buffer as flat strings.

        Speeches accumulate in current_turn_buffer during a phase.
        Call flush_turn_buffer() after the phase to move them to public_history.

        For predicate-only mode, each predicate becomes: [sN|speaker] predicate
        For natural language mode, pass speech=[{pred, text}, ...] to get:
            [sN|speaker] predicate: Natural language text

        Returns the list of flat strings added to current_turn_buffer.
        """
        # Build predicate -> text lookup from speech list
        pred_to_text: Dict[str, str] = {}
        if speech:
            for item in speech:
                pred = item.get("pred", item.get("predicate", ""))
                text = item.get("text", "")
                if pred and text:
                    pred_to_text[pred] = text

        flat: List[str] = []
        for pred in predicates:
            sid = self.next_speech_id()
            text = pred_to_text.get(pred, "")
            if text:
                entry = f"[{sid}|{speaker}] {pred}: {text}"
            else:
                entry = f"[{sid}|{speaker}] {pred}"
            self.current_turn_buffer.append(entry)
            flat.append(entry)
            # Register speech ID for cross-reference (agree/disagree/answer)
            self.speech_registry[sid] = {
                "speaker": speaker,
                "predicate": pred,
                "text": text,
            }
        return flat

    def flush_turn_buffer(self) -> None:
        """Move current turn speeches to public history and clear the buffer."""
        self.public_history.extend(self.current_turn_buffer)
        self.current_turn_buffer.clear()

    def start_voting(self) -> None:
        self.phase = "day_vote"

    def resolve_votes(self, votes: Dict[str, Optional[str]]) -> DayResult:
        """Tally daytime execution votes (sheriff = 1.5x weight).

        Outcomes:
          - single top → executed
          - tie (including all-abstain) → random tiebreak

        Voters may pass None or 'abstain' as target.
        Someone is always executed.
        """
        result = DayResult(
            day=self.day, sheriff=self.sheriff, votes=dict(votes),
            pk_round=1,
        )

        counts: Dict[str, float] = {}
        for voter, target in votes.items():
            if target is None or target == "abstain":
                continue
            weight = self.sheriff_vote_weight if voter == self.sheriff else 1.0
            counts[target] = counts.get(target, 0.0) + weight
        result.vote_counts = counts

        if not counts:
            eligible = sorted(self.alive)
            result.tied_candidates = eligible
            result.executed = random.choice(eligible)
        else:
            max_votes = max(counts.values())
            top = sorted([p for p, c in counts.items() if c == max_votes])
            if len(top) > 1:
                result.tied_candidates = top
                result.executed = random.choice(top)
            else:
                result.executed = top[0]

        for voter, target in votes.items():
            self.public_history.append(
                f"voted(D{self.day}, {voter}, {target})"
            )
        vote_dict: Dict[str, list] = {}
        for voter, target in votes.items():
            if target is not None and target != "abstain":
                vote_dict.setdefault(target, []).append(voter)
        if vote_dict:
            self.public_history.append(vote_dict)

        self.alive.discard(result.executed)
        self.dead.add(result.executed)
        self.public_history.append(
            f"executed(D{self.day}, {result.executed})"
        )
        self.public_history.append(
            f"announcement(D{self.day}, {result.executed}, execution): "
            f"{result.executed} was executed."
        )

        self.vote_log.append(result)
        return result

    # ── Night phase ──

    def start_night(self) -> int:
        """Begin a new night. Returns the night number."""
        self.night += 1
        self.phase = "night"
        return self.night

    def get_valid_guard_targets(self) -> List[str]:
        """Guard targets: all alive players EXCEPT the one guarded last night."""
        targets = list(self.alive)
        if self.last_guard_target and self.last_guard_target in targets:
            targets.remove(self.last_guard_target)
        return targets

    def resolve_night(self, wolf_target: Optional[str],
                      seer_target: Optional[str],
                      guard_target: Optional[str],
                      witch_save: bool = False,
                      witch_poison: Optional[str] = None) -> NightResult:
        result = NightResult(
            night=self.night,
            wolf_target=wolf_target,
            seer_target=seer_target,
            guard_target=guard_target,
            witch_save_target=wolf_target if witch_save else None,
            witch_poison_target=witch_poison,
        )

        # Seer learns target's alignment (private knowledge only -- no public log)
        if seer_target and seer_target in self.alive:
            result.seer_result = "wolf_camp" if self.roles[seer_target] in WOLF_ROLES else "villager_camp"
            seers = self.get_alive_with_role("seer")
            if seers:
                pred = f"divined(N{self.night}, {seers[0]}, {seer_target}, {result.seer_result})"
                self.private_knowledge[seers[0]].append(pred)

        # Wolf attack — blocked if guard protects same target OR witch saves
        if wolf_target and wolf_target in self.alive:
            if guard_target == wolf_target:
                result.killed = None
            elif witch_save and not self.witch_antidote_used:
                result.killed = None
                self.witch_antidote_used = True
            else:
                result.killed = wolf_target
                self.alive.discard(wolf_target)
                self.dead.add(wolf_target)
                self.public_history.append(
                    f"attacked(N{self.night}, {wolf_target})"
                )

        # Witch poison — separate from wolf kill
        if witch_poison and witch_poison in self.alive and not self.witch_poison_used:
            self.witch_poison_used = True
            self.alive.discard(witch_poison)
            self.dead.add(witch_poison)
            self.public_history.append(
                f"attacked(N{self.night}, {witch_poison})"
            )
            # If hunter is poisoned, they cannot shoot
            if self.roles.get(witch_poison) == "hunter":
                self.hunter_can_shoot = False

        # Announcements — list all deaths generically
        killed_players = []
        if result.killed:
            killed_players.append(result.killed)
        if witch_poison and witch_poison in self.dead and witch_poison != result.killed:
            killed_players.append(witch_poison)

        if killed_players:
            for kp in killed_players:
                self.public_history.append(
                    f"announcement(N{self.night}, {kp}, death): {kp} was killed last night."
                )
        else:
            self.public_history.append(
                f"announcement(N{self.night}, none, death): No one died last night."
            )

        # Track guard target for consecutive-night rule
        self.last_guard_target = guard_target

        self.night_log.append(result)
        return result

    # ── Sheriff ──

    def handle_sheriff_death(self, successor: Optional[str] = None) -> Optional[str]:
        """Handle sheriff death. successor can be a player, NO_SHERIFF, or None."""
        if self.sheriff and self.sheriff not in self.alive:
            old_sheriff = self.sheriff
            phase_label = f"N{self.night}" if self.phase == "night" else f"D{self.day}"
            if successor == NO_SHERIFF:
                self.sheriff = None
                self.public_history.append(
                    f"sheriff_successor({phase_label}, {old_sheriff}, NO_SHERIFF)"
                )
            elif successor and successor in self.alive:
                self.sheriff = successor
                self.sheriff_history.append(self.sheriff)
                self.public_history.append(
                    f"sheriff_successor({phase_label}, {old_sheriff}, {self.sheriff})"
                )
                self.public_history.append(
                    f"sheriff_set({phase_label}, {self.sheriff})"
                )
                self.public_history.append(
                    f"announcement({phase_label}, {self.sheriff}, sheriff): {self.sheriff} is now sheriff."
                )
            else:
                alive = self.get_alive()
                self.sheriff = random.choice(alive) if alive else None
                if self.sheriff:
                    self.sheriff_history.append(self.sheriff)
                    self.public_history.append(
                        f"sheriff_successor({phase_label}, {old_sheriff}, {self.sheriff})"
                    )
                    self.public_history.append(
                        f"sheriff_set({phase_label}, {self.sheriff})"
                    )
                    self.public_history.append(
                        f"announcement({phase_label}, {self.sheriff}, sheriff): {self.sheriff} is now sheriff."
                    )
            return self.sheriff
        return self.sheriff

    def handle_hunter_death(self, hunter: str, target: str) -> Optional[str]:
        """Hunter shoots a player on death. Returns the shot target or None."""
        if target and target in self.alive:
            self.alive.discard(target)
            self.dead.add(target)
            phase_label = f"N{self.night}" if self.phase == "night" else f"D{self.day}"
            self.public_history.append(
                f"hunter_shot({phase_label}, {hunter}, {target})"
            )
            self.public_history.append(
                f"announcement({phase_label}, {target}, death): {target} was killed by the hunter."
            )
            return target
        return None

    def handle_wwk_explosion(self, wwk: str, target: str) -> Optional[str]:
        """White Wolf King self-explodes, killing itself and one target.
        Returns the target killed, or None if target invalid."""
        phase_label = f"D{self.day}"
        # Kill WWK
        if wwk in self.alive:
            self.alive.discard(wwk)
            self.dead.add(wwk)
        # Kill target
        killed_target = None
        if target and target in self.alive:
            self.alive.discard(target)
            self.dead.add(target)
            killed_target = target
        self.public_history.append(
            f"wwk_explode({phase_label}, {wwk}, {target})"
        )
        self.public_history.append(
            f"announcement({phase_label}, {wwk}, death): {wwk} (White Wolf King) self-destructed."
        )
        if killed_target:
            self.public_history.append(
                f"announcement({phase_label}, {killed_target}, death): {killed_target} was killed by the explosion."
            )
        return killed_target

    # ── Win check ──

    def check_win(self) -> Optional[str]:
        """Check win conditions. Returns 'village', 'werewolf', or None."""
        wolves_alive = len([p for p in self.alive if self.roles[p] in ("werewolf", "white_wolf_king")])
        non_wolves_alive = len(self.alive) - wolves_alive

        phase_label = f"N{self.night}" if self.phase == "night" else f"D{self.day}"

        if wolves_alive == 0:
            self.winner = "villagers_win"
            self.phase = "finished"
            self.public_history.append(
                f"game_status({phase_label}, villagers_win)"
            )
            return "villagers_win"
        if wolves_alive >= non_wolves_alive:
            self.winner = "werewolves_win"
            self.phase = "finished"
            self.public_history.append(
                f"game_status({phase_label}, werewolves_win)"
            )
            return "werewolves_win"

        # Game continues
        self.public_history.append(
            f"game_status({phase_label}, continue)"
        )
        return None
