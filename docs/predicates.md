# Predicate Reference

Predicates are the structured representation of player actions and game events in LurkingWolf. Every speech, action, and system event is encoded as a predicate with typed arguments. The game engine, eval system, and rewriters all operate on this shared vocabulary.

## Conventions

- **Phase labels**: `D1` = day 1, `D2` = day 2, `N0` = night 0, `N1` = night 1, etc.
- **Speech IDs**: `s1`, `s2`, `s3`, ... assigned sequentially within a turn.
- **Player IDs**: `P1`, `P2`, ..., `P12`.
- **Alignment**: `wolf_camp` or `villager_camp` (used in divination results).

---

## Speech predicates

Used during discussion phases (day speech, election speech, last speech).

| Predicate | Signature | Description |
|-----------|-----------|-------------|
| `roleclaim` | `(Phase, Player, ClaimedRole)` | Player claims to be a role. May be false. |
| `estimate` | `(Phase, Player, Target, Role)` | Player states their read on Target's role. Speaker is not Target. |
| `divined` | `(Phase, Player, Target, Alignment)` | Player claims they divined Target. Alignment: `wolf_camp` or `villager_camp`. May be a lie. |
| `guarded` | `(Phase, Player, Target)` | Player claims they protected Target last night as guard. May be a lie. |
| `question` | `(Phase, Player, Target, RefID)` | Player challenges Target about a specific statement by RefID. |
| `answer` | `(Phase, Player, RefID)` | Player responds to a question. RefID must reference a `question` predicate. |
| `agree` | `(Phase, Player, RefID)` | Player publicly supports a statement by speech ID. |
| `disagree` | `(Phase, Player, RefID)` | Player publicly opposes a statement by speech ID. |
| `allied` | `(Phase, Player, T1, T2)` | Player claims T1 and T2 are on the same team. About T1 and T2, not the speaker. |
| `unallied` | `(Phase, Player, T1, T2)` | Player claims T1 and T2 are on different teams. About T1 and T2, not the speaker. |
| `dissociate` | `(Phase, Player, Target)` | Player states they do not share the same side as Target, based on Target's behavior or statements. |
| `recommend_action` | `(Phase, Speaker, Actor, Action, Target)` | Speaker recommends that Actor take Action toward Target. See action table below. |
| `address_role` | `(Phase, Speaker, TargetRole, Advice)` | Speaker addresses all players of a role group with advice. See tables below. |
| `fluff` | `(Phase, Player)` | Player says current evidence is insufficient and identifies what would help them judge. |
| `witch_save` | `(Phase, Player)` | Player claims they used the witch's antidote. May be a lie. |
| `witch_poison` | `(Phase, Player, Target)` | Player claims they used the witch's poison on Target. May be a lie. |

### Examples

```
roleclaim(D1, P3, seer)
estimate(D1, P2, P5, werewolf)
divined(D1, P3, P5, wolf_camp)
question(D1, P1, P4, s6)
agree(D1, P2, s3)
recommend_action(D1, P3, P1, vote_execute, P5)
address_role(D1, P3, seer, reveal)
fluff(D1, P1)
```

### `recommend_action` -- valid Actions

| Action | Meaning |
|--------|---------|
| `vote_sheriff` | Vote for a player in the sheriff election |
| `vote_execute` | Vote to execute a player during the day |
| `divine` | Seer should divine a target |
| `protect` | Guard should protect a target |
| `use_antidote` | Witch should use antidote |
| `use_poison` | Witch should use poison on a target |
| `shoot` | Hunter should shoot a target |
| `pass_badge` | Transfer the sheriff badge to a specific player |
| `tear_badge` | Destroy the sheriff badge so no one holds it |
| `question` | Question a specific player |
| `pressure` | Apply pressure to a specific player |
| `run_sheriff` | Nominate oneself for sheriff |
| `withdraw_sheriff` | Withdraw from the sheriff election |

### `address_role` -- valid TargetRole values

`seer`, `guard`, `witch`, `hunter`, `villager`, `werewolf`, `village`

### `address_role` -- valid Advice values

| Advice | Meaning |
|--------|---------|
| `hide` | Do not reveal your role |
| `reveal` | Disclose your role publicly |
| `vote_execute` | Cast your vote to eliminate a player |
| `hold` | Refrain from using your ability |
| `act_now` | Use your ability now |

---

## Action predicates

Used in specific game phases (night actions, sheriff election, death triggers).

| Predicate | Signature | Phase | Description |
|-----------|-----------|-------|-------------|
| `run_sheriff` | `(Phase, Player)` | Sheriff nomination | Player nominates themselves for sheriff. |
| `sheriff_order` | `(Phase, Sheriff, left\|right)` | Day start | Sheriff picks speaking direction. `left` = descending ID, `right` = ascending ID. |
| `sheriff_successor` | `(Phase, Sheriff, Successor)` | On death | Dying sheriff names successor or `NO_SHERIFF` to tear the badge. |
| `hunter_shot` | `(Phase, Hunter, Target)` | On elimination | Hunter shoots Target (or `none` to skip). |
| `wwk_explode` | `(Phase, WWK, Target)` | Day phase | White Wolf King self-destructs, killing Target. |
| `wwk_pass` | `(Phase, WWK)` | Day phase | White Wolf King chooses not to self-destruct. |
| `attacked` | `(Phase, Player)` | Wolf night action | Wolves choose a target to kill (or `none`). |
| `divined` | `(Phase, Player, Target, Alignment)` | Seer night action | Seer checks a target's alignment. |
| `guarded` | `(Phase, Player, Target)` | Guard night action | Guard protects a target. |
| `witch_save` | `(Phase, Witch)` | Witch night action | Witch uses antidote to save tonight's victim. |
| `witch_poison` | `(Phase, Witch, Target)` | Witch night action | Witch uses poison on Target. |

---

## Result predicates

System-generated, read-only. Players cannot emit these; they appear in the game log and public history.

| Predicate | Signature | Description |
|-----------|-----------|-------------|
| `executed` | `(Phase, Player)` | Player was voted out during the day. |
| `attacked` | `(Phase, Player)` | Player was killed by werewolves at night. |
| `voted` | `(Phase, Voter, Target)` | Vote cast during day or election. |
| `sheriff_set` | `(Phase, Player)` | Player is now sheriff. |
| `game_status` | `(Phase, Status)` | Game state: `continue`, `villagers_win`, or `werewolves_win`. |
| `announcement` | `(Phase, Player\|none, Event)` | System announcement. Events: `death`, `execution`, `sheriff`. |
| `hunter_shot` | `(Phase, Hunter, Target)` | Hunter shot Target on death (result record). |
| `wwk_explode` | `(Phase, WWK, Target)` | White Wolf King self-destructed, killing Target (result record). |

---

## Roles

| Role | Team | Special ability |
|------|------|----------------|
| `werewolf` | Wolf | Votes to kill one player each night. |
| `white_wolf_king` | Wolf | Can self-destruct during the day to kill one player. |
| `seer` | Village | Divines one player's alignment each night. |
| `guard` | Village | Protects one player from wolf attack each night (cannot repeat). |
| `witch` | Village | One antidote (save victim) and one poison (kill target), each once per game. |
| `hunter` | Village | Shoots one player on elimination. |
| `villager` | Village | No special ability. |
