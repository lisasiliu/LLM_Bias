"""Game rules for each supported player count, keyed by player count integer."""

# ── Shared sections (identical across all player counts) ──────────────────────

_GAME_FLOW = """\
Game Flow
The game alternates between night and day. Night 0 is the first night. After Night 0, a Sheriff Election takes place as a separate phase. Day 1 begins after the election concludes.

Each day phase follows this order:
1. Announce who died during the previous night
2. Ability window for night deaths
3. Daytime discussion (the Sheriff speaks last if one is alive)
4. Daytime vote
5. If a player is executed, their last words and ability window follow
6. Move to the next night

The ability window runs when a player has died. It covers death-triggered abilities and sheriff badge transfer if the dead player held it. Last words are given only by players eliminated through daytime voting, not by night kill victims. On a peaceful night or a day where no one is executed, the corresponding steps do not run."""

_PLAYER_ID = """\
Player Identification
Each player has a seat number such as P1, P2, P3. Refer to other players by their P-number or nickname."""

_SHERIFF_ELECTION = """\
Sheriff Election
The Sheriff Election takes place once, after Night 0 and before Day 1.
- Every living player may run or abstain.
- If only one player runs, that player becomes Sheriff without a vote.
- If multiple players run, candidates give speeches and may withdraw. Non-candidates vote. If the vote is tied, one tied candidate is chosen at random.
- If nobody runs, the game continues without a Sheriff."""

_SHERIFF_POWERS = """\
Sheriff Powers
The Sheriff decides the daytime speaking order, speaks last during daytime discussion, and has 1.5 votes during daytime elimination voting. When the Sheriff dies or is eliminated, the badge may be passed to any living player or destroyed."""

_SPEAKING_ORDER = """\
Speaking Order
If the Sheriff is alive, the Sheriff chooses the speaking direction. If exactly one player died at night, discussion starts from the seat adjacent to the dead seat in the direction the Sheriff picks. If the night was peaceful or two or more players died, the Sheriff picks the direction freely. The Sheriff always speaks last.

If there is no living Sheriff, the system picks a random starting player and direction."""

_DISCUSSION = """\
Discussion
During speaking phases, players may send multiple public messages. Each speaker has a countdown timer and may finish early."""

_VOTING = """\
Voting
Players vote to eliminate one player. Abstaining is allowed. The player with the most votes is eliminated. If the top vote is tied, the tied players enter a PK speech round and a revote follows. If the revote is also tied or all voters abstain, no one is executed. The Sheriff's vote counts as 1.5 votes."""

_ELIMINATION = """\
Elimination and Role Reveal
When a player is eliminated or killed at night, their role is revealed. Eliminated players may give last words before leaving."""

_WIN_CONDITIONS = """\
Win Conditions
- Village Team wins when all Werewolf Team members are eliminated.
- Werewolf Team wins when the number of living Werewolf Team members equals or exceeds the number of living Village Team players."""

_HUNTER_ABILITY = """\
Hunter Special Ability
When the Hunter is eliminated by daytime voting or by a Werewolf attack at night, the Hunter may immediately shoot one living player. The shot target is also eliminated. The Hunter cannot shoot if killed by the Witch's Poison. This ability triggers once, at the moment of elimination."""

_WWK_ABILITY = """\
White Wolf King Special Ability
The White Wolf King may self-destruct at the start of any day or election phase to kill itself and one chosen living player. The current phase ends immediately with no result.

When the White Wolf King is eliminated by daytime voting, the White Wolf King may self-destruct in the ability window to kill one additional living player.

The White Wolf King cannot self-destruct if killed by the Witch's Poison at night. The White Wolf King may choose to forgo self-destruct in any situation."""


# ── Shared role entries ───────────────────────────────────────────────────────

_R_WOLF = "- Werewolf (Werewolf Team): knows all Werewolf Team members. Each night, all wolves choose one target together and must agree. The target may be any living player, including themselves or a fellow wolf. May choose no kill. Works to appear trustworthy during the day."
_R_WWK = "- White Wolf King (Werewolf Team): knows all Werewolf Team members. Votes with the Werewolves each night. May self-destruct during any day or election phase to kill itself and one chosen player, ending the phase with no result. When eliminated by voting, may self-destruct in the ability window to kill one additional player. Cannot self-destruct if killed by the Witch's Poison."
_R_SEER = "- Seer (Village Team): each night, checks one living player's team alignment. Learns whether that player is Werewolf Team or Village Team, not their specific role. Results are private."
_R_WITCH = "- Witch (Village Team): is told who the Werewolves targeted before deciding each night. Has one antidote and one poison, each usable once per game. May use at most one potion per night. Cannot use the antidote to save herself. May choose not to act."
_R_HUNTER = "- Hunter (Village Team): when eliminated by voting or wolf attack, may immediately shoot one player. Cannot shoot if killed by the Witch's Poison."
_R_GUARD = "- Guard (Village Team): each night, protects one player from Werewolf elimination. Self-protection is allowed. Cannot protect the same player on two consecutive nights."
_R_VILLAGER = "- Villager (Village Team): no special ability."

# ── Shared info-visibility lines ──────────────────────────────────────────────

_IV_BASE = """\
- Each player sees only their own role.
- Public speeches and vote results are visible to everyone.
- Werewolf Team members know who all their teammates are.
- Seer results are private."""
_IV_WITCH = "- The Witch knows who the Werewolves targeted each night before deciding whether to act."
_IV_GUARD = "- Guard choices are private."

# ── Builder ───────────────────────────────────────────────────────────────────

def _build(*sections: str) -> str:
    return "\n\n".join(s.strip() for s in sections if s)


# ── Guides ────────────────────────────────────────────────────────────────────

RULES_GUIDES: dict = {

    7: _build(
        "Werewolf Game Rules — 7-Player Setup",
        """\
Player Setup
7 players total.
- 2 Werewolves (Werewolf Team)
- 3 Villagers (Village Team)
- 1 Seer (Village Team)
- 1 Guard (Village Team)

Teams
- Werewolf Team: Werewolves
- Village Team: Villagers, Seer, Guard""",
        _WIN_CONDITIONS,
        _GAME_FLOW,
        _PLAYER_ID,
        """\
Night Order
Each night, roles act in this order:
1. Guard
2. Werewolves
3. Seer""",
        _SHERIFF_ELECTION,
        _SHERIFF_POWERS,
        _SPEAKING_ORDER,
        _DISCUSSION,
        _VOTING,
        _ELIMINATION,
        "Role Guide\n" + "\n".join([_R_WOLF, _R_SEER, _R_GUARD, _R_VILLAGER]),
        "Information Visibility\n" + "\n".join([_IV_BASE, _IV_GUARD]),
        "Role Claims and Identity\nEach special role (Seer, Guard) exists only once in this game. If two players claim the same role, one of them is lying.",
    ),

    8: _build(
        "Werewolf Game Rules — 8-Player Setup",
        """\
Player Setup
8 players total.
- 3 Werewolves (Werewolf Team)
- 2 Villagers (Village Team)
- 1 Seer (Village Team)
- 1 Witch (Village Team)
- 1 Hunter (Village Team)

Teams
- Werewolf Team: Werewolves
- Village Team: Villagers, Seer, Witch, Hunter""",
        _WIN_CONDITIONS,
        _GAME_FLOW,
        _PLAYER_ID,
        """\
Night Order
Each night, roles act in this order:
1. Werewolves
2. Witch
3. Seer""",
        _SHERIFF_ELECTION,
        _SHERIFF_POWERS,
        _SPEAKING_ORDER,
        _DISCUSSION,
        _VOTING,
        _ELIMINATION,
        _HUNTER_ABILITY,
        "Role Guide\n" + "\n".join([_R_WOLF, _R_SEER, _R_WITCH, _R_HUNTER, _R_VILLAGER]),
        "Information Visibility\n" + "\n".join([_IV_BASE, _IV_WITCH]),
        "Role Claims and Identity\nEach special role (Seer, Witch, Hunter) exists only once in this game. If two players claim the same role, one of them is lying.",
    ),

    9: _build(
        "Werewolf Game Rules — 9-Player Setup",
        """\
Player Setup
9 players total.
- 3 Werewolves (Werewolf Team)
- 3 Villagers (Village Team)
- 1 Seer (Village Team)
- 1 Witch (Village Team)
- 1 Hunter (Village Team)

Teams
- Werewolf Team: Werewolves
- Village Team: Villagers, Seer, Witch, Hunter""",
        _WIN_CONDITIONS,
        _GAME_FLOW,
        _PLAYER_ID,
        """\
Night Order
Each night, roles act in this order:
1. Werewolves
2. Witch
3. Seer""",
        _SHERIFF_ELECTION,
        _SHERIFF_POWERS,
        _SPEAKING_ORDER,
        _DISCUSSION,
        _VOTING,
        _ELIMINATION,
        _HUNTER_ABILITY,
        "Role Guide\n" + "\n".join([_R_WOLF, _R_SEER, _R_WITCH, _R_HUNTER, _R_VILLAGER]),
        "Information Visibility\n" + "\n".join([_IV_BASE, _IV_WITCH]),
        "Role Claims and Identity\nEach special role (Seer, Witch, Hunter) exists only once in this game. If two players claim the same role, one of them is lying.",
    ),

    10: _build(
        "Werewolf Game Rules — 10-Player Setup",
        """\
Player Setup
10 players total.
- 3 Werewolves (Werewolf Team)
- 3 Villagers (Village Team)
- 1 Seer (Village Team)
- 1 Witch (Village Team)
- 1 Hunter (Village Team)
- 1 Guard (Village Team)

Teams
- Werewolf Team: Werewolves
- Village Team: Villagers, Seer, Witch, Hunter, Guard""",
        _WIN_CONDITIONS,
        _GAME_FLOW,
        _PLAYER_ID,
        """\
Night Order
Each night, roles act in this order:
1. Guard
2. Werewolves
3. Witch
4. Seer""",
        _SHERIFF_ELECTION,
        _SHERIFF_POWERS,
        _SPEAKING_ORDER,
        _DISCUSSION,
        _VOTING,
        _ELIMINATION,
        _HUNTER_ABILITY,
        "Role Guide\n" + "\n".join([_R_WOLF, _R_SEER, _R_WITCH, _R_HUNTER, _R_GUARD, _R_VILLAGER]),
        "Information Visibility\n" + "\n".join([_IV_BASE, _IV_WITCH, _IV_GUARD]),
        "Role Claims and Identity\nEach special role (Seer, Guard, Witch, Hunter) exists only once in this game. If two players claim the same role, one of them is lying.",
    ),

    11: _build(
        "Werewolf Game Rules — 11-Player Setup",
        """\
Player Setup
11 players total.
- 4 Werewolves (Werewolf Team)
- 3 Villagers (Village Team)
- 1 Seer (Village Team)
- 1 Witch (Village Team)
- 1 Hunter (Village Team)
- 1 Guard (Village Team)

Teams
- Werewolf Team: Werewolves
- Village Team: Villagers, Seer, Witch, Hunter, Guard""",
        _WIN_CONDITIONS,
        _GAME_FLOW,
        _PLAYER_ID,
        """\
Night Order
Each night, roles act in this order:
1. Guard
2. Werewolves
3. Witch
4. Seer""",
        _SHERIFF_ELECTION,
        _SHERIFF_POWERS,
        _SPEAKING_ORDER,
        _DISCUSSION,
        _VOTING,
        _ELIMINATION,
        _HUNTER_ABILITY,
        "Role Guide\n" + "\n".join([_R_WOLF, _R_SEER, _R_WITCH, _R_HUNTER, _R_GUARD, _R_VILLAGER]),
        "Information Visibility\n" + "\n".join([_IV_BASE, _IV_WITCH, _IV_GUARD]),
        "Role Claims and Identity\nEach special role (Seer, Guard, Witch, Hunter) exists only once in this game. If two players claim the same role, one of them is lying.",
    ),

    12: _build(
        "Werewolf Game Rules — 12-Player Setup",
        """\
Player Setup
12 players total.
- 3 Werewolves (Werewolf Team)
- 1 White Wolf King (Werewolf Team)
- 4 Villagers (Village Team)
- 1 Seer (Village Team)
- 1 Witch (Village Team)
- 1 Hunter (Village Team)
- 1 Guard (Village Team)

Teams
- Werewolf Team: Werewolves, White Wolf King
- Village Team: Villagers, Seer, Witch, Hunter, Guard""",
        _WIN_CONDITIONS,
        _GAME_FLOW,
        _PLAYER_ID,
        """\
Night Order
Each night, roles act in this order:
1. Guard
2. Werewolves and White Wolf King
3. Witch
4. Seer""",
        _SHERIFF_ELECTION,
        _SHERIFF_POWERS,
        _SPEAKING_ORDER,
        _DISCUSSION,
        _VOTING,
        _ELIMINATION,
        _HUNTER_ABILITY,
        _WWK_ABILITY,
        "Role Guide\n" + "\n".join([_R_WOLF, _R_WWK, _R_SEER, _R_WITCH, _R_HUNTER, _R_GUARD, _R_VILLAGER]),
        "Information Visibility\n" + "\n".join([_IV_BASE, _IV_WITCH, _IV_GUARD]),
        "Role Claims and Identity\nEach special role (Seer, Guard, Witch, Hunter) exists only once in this game. If two players claim the same role, one of them is lying.",
    ),

}
