# LurkingWolf

A Werewolf (Mafia) game engine where every player is an LLM agent. Built for studying how surface-level text changes (formality, verbosity, hedging) influence strategic decisions in multi-agent games.

## Quickstart

```bash
# 1. Start a vLLM server (in a separate terminal / SLURM job)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --port 8000

# 2. Run a game (predicate-only, no NL)
cd lurkingwolf
python run_game.py --vllm --verbose

# 3. Run with natural language generation
python run_game.py --vllm --experiment nl --verbose

# 4. Dry run (no LLM calls, see prompts and game flow)
python run_game.py --dry-run --verbose
```

Output goes to `lurkingwolf/games/run_{model}_{experiment}_{seed}/`.

## How it works

Games have 7-12 LLM players with hidden roles (werewolves, seer, villager, etc.). Each day: players discuss, vote to execute someone. Each night: werewolves attack, special roles act.

Speech goes through a 3-stage pipeline:

1. **Predicates** -- the LLM generates structured actions like `estimate(D1, P3, P4, villager)`
2. **NL realization** -- predicates are converted to natural language sentences
3. **Rewriter** (optional) -- a pluggable module transforms the text (e.g. make it formal/casual)

Stage 3 is where you inject experimental manipulations. The game engine and other agents only see the rewritten text, so you can measure how surface-level changes affect downstream decisions.

## Models

We use Qwen 3.5 served locally via [vLLM](https://docs.vllm.ai/):

| Model | Alias | Use case |
|-------|-------|----------|
| `Qwen/Qwen3.5-9B` | `qwen3.5-9b` | Fast iteration, development |
| `Qwen/Qwen3.5-27B` | `qwen3.5-27b` | Full experiments |

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
python run_game.py --vllm --experiment nl --rewriter my_rewriter
```

## Project structure

```
lurkingwolf/          Core engine, agents, eval framework
  run_game.py         CLI entry point
  agent.py            LLM agent (prompting, parsing, memory)
  engine.py           Game state machine (roles, phases, win conditions)
  rewriter.py         Rewriter ABC + registry
  config.py           YAML experiment config loader
  eval_v2/            Per-phase replay evaluation system
llm_prompts/          Prompt templates (prompt.json, prompt_templates.py)
examples/             Demo rewriters + experiment configs
docs/                 User guide
slurm_scripts/        SLURM templates (adapt to your cluster)
```

## Evaluation (eval_v2)

The replay system lets you run counterfactual experiments: take a completed game, load a checkpoint at day N, and replay with different text manipulations. This measures:

- **Skill delta** -- did night actions (wolf attack, seer divine, etc.) change?
- **Vote delta** -- did voting targets change?
- **Reliability delta** -- did trust scores shift?

```bash
# Run pred_only vs NL comparison
cd lurkingwolf
python -m eval_v2 --mode pred_vs_nl \
    --vllm --vllm-url http://localhost:8000/ --vllm-model Qwen/Qwen3.5-9B
```

## SLURM

The `slurm_scripts/` directory has job submission templates. These are cluster-specific -- copy `slurm_conf_template.yaml` to `slurm_conf.yaml` and edit it for your cluster (account name, conda env path, GPU type).

## License

Research use.
