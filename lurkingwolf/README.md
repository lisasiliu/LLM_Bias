# LurkingWolf Engine

Core game engine, agents, and evaluation framework for studying how natural language surface form affects LLM decision-making in Werewolf.

## Quick start

```bash
# Full experiment from a YAML config (one command, one seed, one folder)
lurk run --config examples/formality/config.yaml --seed 101

# Single NL game
python -m lurkingwolf.run_game --experiment nl --seed 101 --verbose

# Dry run (no LLM calls)
python -m lurkingwolf.run_game --dry-run --verbose
```

## CLI commands

| Command | Description |
|---------|-------------|
| `lurk run --config <yaml> --seed <s>` | Run a full experiment (all comparisons, baselines + replays) |
| `lurk eval --config <yaml> --seed <s>` | Same as `run --config` but via the eval entry point |
| `lurk eval --mode pred_vs_nl --seed <s>` | Quick single-comparison (pred-only vs NL) |
| `lurk check-determinism --model <name>` | Test NL generation determinism across all predicates |

## 3-stage pipeline

1. **Stage 1** — Predicate generation (game engine, `agent.py:act()`)
2. **Stage 2** — Deterministic NL realization (seed=101, `utils/speech_nl.py`)
3. **Stage 3** — Pluggable rewriter (`rewriter.py`, applied in `agent.py`)

## Experiment modes

| Mode | Description |
|------|-------------|
| `pred_only` | Predicates only, no natural language. Used as baseline for RQ1 (does surface form matter?). |
| `nl` | Full pipeline: predicates + NL sentences. Used as replay for RQ1 and baseline for RQ2. |
| `flip_self` | Rewriter applied to one player's speech only, rotated through all players. Metrics: skill, vote, reliability. |
| `flip_others` | Rewriter applied to everyone except one player, rotated. Metrics: fairness. |

## YAML config

Experiments are defined as named **comparisons**, each with a baseline game type and a list of replays:

```yaml
name: formality
model: Qwen/Qwen3.5-27B
seeds: [101]
rewriter_module: examples.formality.rewriter

comparisons:
  pred_vs_nl:
    baseline: pred_only
    replays:
      - mode: nl

  nl_vs_formality:
    baseline: nl
    replays:
      - mode: flip_self
        rewriter: FormalityRewriter
        rewriter_params: {level: formal}
      - mode: flip_others
        rewriter: FormalityRewriter
        rewriter_params: {level: casual}
```

Output: `examples/formality/games/eval_qwen3.5-27b_101/{comparison}/baseline/` + `replays/`

## Backend

Any OpenAI-compatible server (vLLM, Ollama, LiteLLM, TGI, OpenRouter, etc.). Set `LLM_BASE_URL` in `.env` or pass `--url`. Default model: `Qwen/Qwen3.5-27B`.

## Output structure

### Single game
```
games/run_qwen3.5-27b_nl_101/
  game.out                  Human-readable game log
  public_history.json       Per-phase predicates + announcements
  game_record.json          Hidden info (true roles, night actions, votes)
  game_log.jsonl            Raw LLM calls
  resources.json            Token usage + timing
  manifest.json             File map + metadata
  current_turn/             One JSON file per day/night
  agent_memory/             Per-player memory (P1.json ... P7.json)
  checkpoints/              State snapshots (if --save-checkpoints)
```

### Config-driven experiment
```
examples/formality/games/eval_qwen3.5-27b_101/
  pred_vs_nl/
    baseline/               pred_only game + checkpoints
    replays/
      day_1/
        nl__d1/             phase_result.json + agent_memory/ + game.out
  nl_vs_formality/
    baseline/               nl game + checkpoints
    replays/
      day_1/
        flip_self__formal__p1__d1/
        flip_self__casual__p1__d1/
        flip_others__formal__p1__d1/
        ...
  experiment_summary.json
```

## Predicate grammar

All predicates are defined in `prompts/prompt.json`. See `docs/predicates.md` for the full reference.

### Speech predicates (day discussion claims)

| Predicate | Signature | Description |
|-----------|-----------|-------------|
| `claim_role` | `(Phase, Player, Role)` | Claim a role (may be false) |
| `estimate` | `(Phase, Player, Target, Role)` | Assess another player's role |
| `claim_divine` | `(Phase, Player, Target, Alignment)` | Claim a divination result (may be a lie) |
| `claim_guard` | `(Phase, Player, Target)` | Claim guard protection on Target (may be a lie) |
| `claim_antidote` | `(Phase, Player)` | Claim use of witch's antidote (may be a lie) |
| `claim_poison` | `(Phase, Player, Target)` | Claim use of witch's poison (may be a lie) |
| `question` | `(Phase, Player, Target, RefID)` | Challenge Target about a specific statement |
| `answer` | `(Phase, Player, RefID)` | Respond to a question directed at you |
| `agree` | `(Phase, Player, RefID)` | Support another player's statement |
| `disagree` | `(Phase, Player, RefID)` | Oppose another player's statement |
| `allied` | `(Phase, Player, T1, T2)` | Claim T1 and T2 are on the same team |
| `unallied` | `(Phase, Player, T1, T2)` | Claim T1 and T2 are on different teams |
| `dissociate` | `(Phase, Player, Target)` | Deny sharing the same side as Target |
| `recommend_action` | `(Phase, Speaker, Actor, Action, Target)` | Recommend Actor take Action toward Target |
| `address_role` | `(Phase, Speaker, TargetRole, Advice)` | Address all players of a role group |
| `fluff` | `(Phase, Player)` | Insufficient evidence; identify what would help |

### Night action predicates

| Predicate | Signature | Phase | Description |
|-----------|-----------|-------|-------------|
| `divine` | `(Phase, Seer, Target)` | `night_seer` | Seer checks a target's alignment |
| `guard` | `(Phase, Guard, Target)` | `night_guard` | Guard protects a target |
| `antidote` | `(Phase, Witch)` | `night_witch` | Witch uses antidote to save victim |
| `poison` | `(Phase, Witch, Target)` | `night_witch` | Witch uses poison on Target |
| `shoot` | `(Phase, Hunter, Target)` | `hunter_shoot` | Hunter shoots on death (or `none`) |
| `explode` | `(Phase, WWK, Target)` | `wwk_explode_check` | WWK self-destructs, killing Target |
| `pass` | `(Phase, WWK)` | `wwk_explode_check` | WWK chooses not to self-destruct |

### Other action predicates

| Predicate | Signature | Description |
|-----------|-----------|-------------|
| `run_sheriff` | `(Phase, Player)` | Nominate self for sheriff |
| `sheriff_order` | `(Phase, Sheriff, left\|right)` | Sheriff picks speaking direction |
| `sheriff_successor` | `(Phase, Sheriff, Successor)` | Dying sheriff names successor |

### Result predicates (system-generated)

| Predicate | Signature | Description |
|-----------|-----------|-------------|
| `executed` | `(Phase, Player)` | Voted out during day |
| `attacked` | `(Phase, Player)` | Killed by werewolf at night |
| `voted` | `(Phase, Voter, Target)` | Vote cast during day or election |
| `sheriff_set` | `(Phase, Player)` | Player is now sheriff |
| `game_status` | `(Phase, Status)` | continue, villagers_win, or werewolves_win |
| `announcement` | `(Phase, Player\|none, Event)` | System announcement (death, execution, sheriff) |
| `hunter_shot` | `(Phase, Hunter, Target)` | Hunter shot on death |
| `wwk_explode` | `(Phase, WWK, Target)` | White Wolf King self-destructed |
