# LurkingWolf

A Werewolf (Mafia) game engine where every player is an LLM agent. Built for studying how surface-level text changes (formality, verbosity, hedging) influence strategic decisions in multi-agent games.

## Quickstart

```bash
# 1. Start any OpenAI-compatible server (vLLM, Ollama, LiteLLM, TGI, etc.)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --port 8000

# 2. Run a game (predicate-only, no NL)
python -m lurkingwolf.run_game --verbose

# 3. Run with natural language generation
python -m lurkingwolf.run_game --experiment nl --verbose

# 4. Dry run (no LLM calls, see prompts and game flow)
python -m lurkingwolf.run_game --dry-run --verbose
```

Output goes to `lurkingwolf/games/run_{model}_{experiment}_{seed}/`.

## How it works

Games have 7-12 LLM players with hidden roles (werewolves, seer, villager, etc.). Each day: players discuss, vote to execute someone. Each night: werewolves attack, special roles act.

Speech goes through a 3-stage pipeline:

1. **Predicates** -- the LLM generates structured actions like `estimate(D1, P3, P4, villager)`
2. **NL realization** -- predicates are converted to natural language sentences (deterministic, seed=101)
3. **Rewriter** (optional) -- a pluggable module transforms the text (e.g. make it formal/casual)

Stage 3 is where you inject experimental manipulations. The game engine and other agents only see the rewritten text, so you can measure how surface-level changes affect downstream decisions.

## Models

Works with **any OpenAI-compatible server** -- not just vLLM. Ollama, LiteLLM, TGI, OpenRouter, and the OpenAI API itself all work out of the box.

```bash
# Environment variables (or pass --url / --model on the CLI)
export LLM_BASE_URL=http://localhost:8000
export LLM_API_KEY=EMPTY
export LLM_MODEL=Qwen/Qwen3.5-27B
```

| Model | Use case |
|-------|----------|
| `Qwen/Qwen3.5-9B` | Fast iteration, development |
| `Qwen/Qwen3.5-27B` | Full experiments |

## Experiment modes

The framework supports 4 experimental conditions (scope templates):

| Mode | What happens | Default metrics |
|------|-------------|-----------------|
| `pred_only` | Predicate-only baseline, no NL | -- |
| `nl` | NL baseline, no rewriter | -- |
| `flip_self` | Rewriter on self player only, rotate through all players | skill, vote, reliability |
| `flip_others` | Rewriter on everyone except self, rotate through all players | fairness |

Define experiments in YAML:

```yaml
name: formality_study
n_players: 7
seeds: [101, 102, 103]
model: Qwen/Qwen3.5-27B

replays:
  - mode: pred_only
  - mode: nl

  - mode: flip_self
    rewriter: FormalityRewriter
    rewriter_params: {level: formal}
    inject_at: day_start

  - mode: flip_others
    rewriter: FormalityRewriter
    rewriter_params: {level: casual}
    inject_at: day_start

metrics: [skill, vote, reliability, fairness]
```

Run with: `python -m lurkingwolf.run_game --config examples/formality/config.yaml`

## Writing a custom rewriter

See the [User Guide](docs/guide.md) for the full walkthrough on adding your own text manipulation and running experiments.

Quick version:

```python
from lurkingwolf.rewriter import Rewriter, RewriteContext, register

@register
class MyRewriter(Rewriter):
    name = "my_rewriter"

    def __init__(self, level="default", llm_client=None, **kwargs):
        self.level = level
        self.llm_client = llm_client

    def rewrite(self, ctx: RewriteContext) -> str:
        # transform ctx.original_nl however you want
        return ctx.original_nl.upper()  # example: shout everything

    def to_config(self):
        return {"class": "MyRewriter", "params": {"level": self.level}}
```

Then run:
```bash
python -m lurkingwolf.run_game --experiment nl --rewriter my_rewriter
```

## Project structure

```
lurkingwolf/          Core engine, agents, eval framework
  run_game.py         CLI entry point
  agent.py            LLM agent (prompting, parsing, memory)
  engine.py           Game state machine (roles, phases, win conditions)
  rewriter.py         Rewriter ABC + registry
  config.py           YAML experiment config loader
  schemas.py          Output record builders
  checkpointing.py    State serialization for replays
  utils/              Helpers (LLM client, token tracker, NL parsing, gen config)
  prompts/            Prompt templates (prompt.json, templates.py, strategy_guides.py)
  eval/               Per-phase replay evaluation system
examples/             Demo rewriters + experiment configs
docs/                 User guide + predicate reference
scripts/              Job submission templates (adapt to your cluster)
```

## Evaluation

The replay system lets you run counterfactual experiments: take a completed game, load a checkpoint at day N, and replay with different text manipulations. This measures:

- **Skill delta** -- did night actions (wolf attack, seer divine, etc.) change?
- **Vote delta** -- did voting targets change?
- **Reliability delta** -- did trust scores shift?
- **Fairness** -- is a player biased by how others sound?

```bash
# Run pred_only vs NL comparison
python -m lurkingwolf.eval --mode pred_vs_nl \
    --url http://localhost:8000/ --model Qwen/Qwen3.5-9B
```

## License

Research use.
