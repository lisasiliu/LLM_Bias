# User Guide

This guide walks you through running games, writing rewriters, and running experiments with LurkingWolf.

## Prerequisites

- Python 3.10+
- An OpenAI-compatible model server (e.g. vLLM, Ollama, TGI). Set `VLLM_BASE_URL` in your `.env` if not using the default `http://localhost:8000`.
- Dependencies: `pip install -r requirements.txt`

## 1. Running your first game

```bash
python -m lurkingwolf.run_game --experiment nl --seed 101 --verbose
```

This runs a 7-player game with natural language. Each player is an LLM agent with a hidden role (werewolf, seer, villager, etc.). You'll see discussion, voting, and night actions printed to the terminal and saved to `games/run_qwen3.5-27b_nl_101/`.

### CLI options

```
--experiment TYPE    pred_only (default) or nl (natural language)
--players N          7 to 12 players (default: 7)
--seed STRING        Random seed for reproducibility
--url URL            Model server URL (default: http://localhost:8000)
--model MODEL        Model name (default: Qwen/Qwen3.5-27B)
--rewriter NAME      Apply a registered rewriter to all NL output
--rewriter-params    JSON string of rewriter constructor args
--config PATH        Load settings from a YAML file (overrides CLI)
--save-checkpoints   Save game state at each phase (needed for eval replays)
--dry-run            See prompts without calling the LLM
--verbose            Print game progress to terminal
```

### Role distribution

| Players | Werewolves | WWK | Seer | Witch | Hunter | Guard | Villagers |
|---------|-----------|-----|------|-------|--------|-------|-----------|
| 7       | 2         | --  | 1    | --    | --     | 1     | 3         |
| 8       | 3         | --  | 1    | 1     | 1      | --    | 2         |
| 9       | 3         | --  | 1    | 1     | 1      | --    | 3         |
| 10      | 3         | --  | 1    | 1     | 1      | 1     | 3         |
| 11      | 4         | --  | 1    | 1     | 1      | 1     | 3         |
| 12      | 3         | 1   | 1    | 1     | 1      | 1     | 4         |

## 2. The 3-stage pipeline

Every speech in the game goes through three stages:

**Stage 1: Predicate generation.** The LLM produces structured actions:
```
estimate(D1, P3, P4, villager)
recommend_action(D1, P3, P3, vote_execute, P5)
```

**Stage 2: NL realization.** Each predicate becomes a natural language sentence:
```
"I think P4 is a villager based on their behavior."
"I recommend we vote to execute P5 -- their statements don't add up."
```

**Stage 3: Rewriter (optional).** Your rewriter transforms the Stage 2 text before other agents see it. The predicates stay the same -- only the surface text changes.

This separation is the key insight: by rewriting only Stage 3 while keeping Stages 1-2 identical, you can isolate the effect of surface-level text on game outcomes.

See `docs/predicates.md` for the full predicate reference.

## 3. Writing a rewriter

Create a Python file with your rewriter class:

```python
# my_rewriter.py
from lurkingwolf.rewriter import Rewriter, RewriteContext, register

@register
class VerboseRewriter(Rewriter):
    name = "verbose"

    def __init__(self, factor=2, llm_client=None, **kwargs):
        self.factor = factor
        self.llm_client = llm_client

    def rewrite(self, ctx: RewriteContext) -> str:
        if self.llm_client is None:
            return ctx.original_nl

        prompt = (
            f"Expand this game speech to be roughly {self.factor}x longer "
            f"while preserving all factual claims exactly:\n\n"
            f"{ctx.original_nl}"
        )
        response = self.llm_client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    def to_config(self):
        return {"class": "VerboseRewriter", "params": {"factor": self.factor}}
```

The `@register` decorator adds it to the global registry. The `llm_client` is the same vLLM client used by the game engine -- you can use it for LLM-based rewrites at no extra setup cost.

### What you get in `RewriteContext`

| Field | Description |
|-------|-------------|
| `original_nl` | The Stage 2 text to transform |
| `predicate` | Source predicate, e.g. `"estimate(D1, P3, P4, villager)"` |
| `phase` | Game phase: `"day_speech"`, `"sheriff_summary"`, `"last_speech"`, etc. |
| `speaker_id` | Player ID, e.g. `"P1"` |
| `speaker_role_claimed` | The role this agent believes it has |
| `public_history` | All predicate strings broadcast so far in the game |

### Filtering which predicates to rewrite

By default, your rewriter runs on every predicate. Override `applies_to_predicates` to filter:

```python
@property
def applies_to_predicates(self):
    return {"estimate", "roleclaim", "dissociate"}
```

### Running it

Import your rewriter before starting the game so `@register` fires:

```bash
python -c "import my_rewriter" && \
python -m lurkingwolf.run_game --experiment nl --rewriter verbose \
    --rewriter-params '{"factor": 3}'
```

See `examples/formality/rewriter.py` for a complete working example (formal vs. casual speech).

## 4. YAML experiment configs

For reproducible experiments, define everything in a YAML file with named comparisons:

```yaml
name: verbosity_experiment
n_players: 7
seeds: [101, 102, 103]
model: Qwen/Qwen3.5-27B
rewriter_module: examples.verbosity.rewriter

comparisons:
  pred_vs_nl:
    baseline: pred_only
    replays:
      - mode: nl

  nl_vs_verbose:
    baseline: nl
    replays:
      - mode: flip_self
        rewriter: VerboseRewriter
        rewriter_params: {factor: 2}
        inject_at: day_start

      - mode: flip_others
        rewriter: VerboseRewriter
        rewriter_params: {factor: 2}
        inject_at: day_start

metrics: [skill, vote, reliability, fairness]
```

Run it:
```bash
python -m lurkingwolf.run_game --config path/to/experiment.yaml --seed 101
```

Output: `lurkingwolf/games/verbosity_experiment_101/`

### Config fields

| Field | Default | Description |
|-------|---------|-------------|
| `name` | (required) | Experiment name |
| `comparisons` | (required) | Named comparison blocks |
| `n_players` | 7 | 7 to 12 |
| `seeds` | `[101]` | Random seeds for reproducibility |
| `model` | `Qwen/Qwen3.5-27B` | Model name |
| `temperature` | 0.0 | LLM sampling temperature |
| `rewriter_module` | none | Python module to import for rewriter registration |

### Comparison fields

| Field | Description |
|-------|-------------|
| `baseline` | `pred_only` or `nl` — the baseline game type |
| `replays` | List of replay specs to run against the baseline |

### Replay spec fields

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | (required) | `pred_only`, `nl`, `flip_self`, or `flip_others` |
| `rewriter` | none | Rewriter class name (required for flip modes) |
| `rewriter_params` | `{}` | Parameters passed to the rewriter |
| `inject_at` | `day_start` | `day_start` or `election_start` |
| `self_role` | none | Only rotate through players with this role |

## 5. How it works

One command runs the full experiment:

1. For each comparison, run the baseline game with checkpoints
2. Expand replays: `flip_self`/`flip_others` modes rotate through all players x days
3. Run replays in parallel (4 workers by default)
4. Output tree:
```
lurkingwolf/games/{name}_{seed}/
  {comparison}/
    baseline/          # full game output + checkpoints
    replays/
      day_1/
        {run_id}/      # phase_result.json + agent_memory/ + game.out
      day_2/
        ...
```

This loads each day's checkpoint and replays with different settings, measuring what changed.

### What gets measured

- **Skill delta**: Did the seer divine a different target? Did wolves attack someone different?
- **Vote delta**: Did any player change their vote?
- **Reliability delta**: Did trust scores between players shift?

### Custom rewriter evaluation

You can build eval configs programmatically:

```python
from lurkingwolf.eval.configs import rewriter_configs
from my_rewriter import VerboseRewriter

configs = rewriter_configs(
    VerboseRewriter,
    params_variants=[{"factor": 2}, {"factor": 3}],
    scopes=["all", "self"],
)
```

## 6. Output files

Each game produces:

```
games/run_qwen3.5-27b_nl_101/
  game.out                  Human-readable game log
  public_history.json       Per-phase predicate strings
  game_record.json          Hidden info (true roles, night actions, votes)
  game_log.jsonl            Raw LLM calls
  resources.json            Token usage and timing
  manifest.json             File map and metadata
  current_turn/             One JSON per day/night
  agent_memory/             Per-player memory snapshots (P1.json ... P7.json)
  checkpoints/              State snapshots (if --save-checkpoints)
```
