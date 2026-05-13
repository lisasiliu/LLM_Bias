# LurkingWolf Engine

Core game engine, agents, and evaluation framework.

## Running games

```bash
# Basic NL game
python -m lurkingwolf.run_game --experiment nl --seed 101 --verbose

# Dry run (no LLM calls)
python -m lurkingwolf.run_game --dry-run --verbose

# 9 players (adds witch + hunter)
python -m lurkingwolf.run_game --players 9 --experiment nl --verbose

# 12 players (adds white wolf king)
python -m lurkingwolf.run_game --players 12 --experiment nl --verbose

# With a rewriter
python -m lurkingwolf.run_game --experiment nl --rewriter formality \
    --rewriter-params '{"level": "formal"}'

# From a YAML config
python -m lurkingwolf.run_game --config examples/formality/config.yaml
```

## Experiment types

| Type | Description |
|------|-------------|
| `pred_only` | Predicates only, no natural language (default) |
| `nl` | Two-stage: predicates then NL sentences |

## Backend

The engine calls LLMs via an OpenAI-compatible server (e.g. vLLM). Set `VLLM_BASE_URL` in `.env` or pass `--url`:

```bash
python -m lurkingwolf.run_game --url http://localhost:8000/ \
    --model Qwen/Qwen3.5-27B --verbose
```

Default model: `Qwen/Qwen3.5-27B`.

## Output structure

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

## Predicate grammar

See `docs/predicates.md` for the full reference. All predicates are defined in `lurkingwolf/prompts/prompt.json`.

| Predicate | Example | Meaning |
|-----------|---------|---------|
| `roleclaim` | `roleclaim(D1, P3, seer)` | Claim a role |
| `estimate` | `estimate(D1, P3, P4, villager)` | Assess another player's role |
| `divined` | `divined(D1, P1, P4, villager_camp)` | Claim a seer divination result |
| `agree` | `agree(D1, P3, s5)` | Support another statement |
| `disagree` | `disagree(D1, P3, s5)` | Oppose another statement |
| `recommend_action` | `recommend_action(D1, P3, P5, vote_execute, P2)` | Recommend an action |

Result predicates (system-generated): `voted`, `executed`, `attacked`, `sheriff_set`, `game_status`, `announcement`.
