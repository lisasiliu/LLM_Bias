# Proficiency-cue validation (Low / Neutral / Original, optional High)

Validation of the linguistic-proficiency-cue rewriter for LURK Werewolf: does the Low
condition realize a restricted linguistic range that blind listeners perceive, while staying
grammatical, semantically faithful and pragmatically unchanged?

Read **[PLAN.md](PLAN.md)** for the design, the hypotheses, the frozen-threshold procedure and
the list of decisions/deviations from the working plan.

## Layout

| path | contents |
| --- | --- |
| `code/cli.py` | single entry point, `python -m experiments.proficiency_validation.code.cli <command>` |
| `code/semantic_gate.py` | the hard gate, corrective retries, `treatment_unavailable` / fallback policy, `GatedRewriter` |
| `code/generate.py` | Low → Neutral (matched candidates) → High generation with full attempt logs |
| `code/judge_benchmark.py` | judge calibration: human benchmark and synthetic challenge set |
| `code/audit.py`, `code/bundles.py` | untreated-game audit; matched source bundles |
| `code/features.py`, `code/guardrails.py` | Family A features; Family B instruments + politeness drift |
| `code/listener.py` | Family C model probe, human worksheet build/import |
| `code/analysis.py` | contrasts, equivalence, listener aggregation, confirmatory criteria, report |
| `code/frozen.py`, `code/freeze.py` | frozen hashes and their enforcement (`--frozen`); `suggest-bounds`, `freeze`, `verify-frozen` |
| `tests/test_offline.py` | offline test suite (`python -m unittest experiments.proficiency_validation.tests.test_offline`), no model calls |
| `data/judge_benchmark_human.{json,md}` | semantic judges scored against the 48 human meaning labels |
| `data/challenge_set.jsonl` | 129 synthetic challenge items (95 known-broken, 46 positive controls), not yet judged |

Reusable components live outside this directory:

- `components/rewriters/language_rewriter.py` — `LanguageProficiencyRewriter` (`low` / `neutral` / `high`)
- `components/judges/listener_probe.py` — blind pairwise proficiency probe
- `components/metrics/` — tokenization and id normalization, edit distance, lexical and
  syntactic measures, LanguageTool / CoLA clients
- `components/stats.py` — cluster bootstrap, sign-flip tests, TOST, BH, Bradley–Terry, kappa

## Setup

```bash
pip install -r requirements.txt -r requirements-proficiency.txt
python -m spacy download en_core_web_sm
```

Model access uses the same `JudgeModel` handles as the politeness validation
(`deepseek-chat`, or `<provider>:<model>` with the provider's API key in the environment or
`.env`). Every command that calls a model runs a preflight and appends usage to
`data/api_usage.jsonl`.

## Commands

Offline (no model calls):

```bash
python -m experiments.proficiency_validation.code.cli benchmark-judges --out data/judge_benchmark_human.json
python -m experiments.proficiency_validation.code.cli build-challenge --out data/challenge_set.jsonl
python -m experiments.proficiency_validation.code.cli audit games_fresh/run01 games_fresh/run02
python -m experiments.proficiency_validation.code.cli bundles games_fresh/run* --sources-out sources.jsonl --out bundles.jsonl
python -m experiments.proficiency_validation.code.cli generate --sources sources.jsonl --conditions low,neutral --out /dev/null --dry-run
python -m experiments.proficiency_validation.code.cli features --bundles bundles.jsonl --rewrites rewrites.jsonl --subtlex SUBTLEX-US.txt --out features.jsonl
python -m experiments.proficiency_validation.code.cli analyze --rewrites … --features … --guardrails … --listener … --politeness … --out report.md
python -m experiments.proficiency_validation.code.cli suggest-bounds --features features.jsonl
python -m experiments.proficiency_validation.code.cli freeze --neutral-tolerance 0.10 --neutral-match-min-fraction 0.8 \
    --range-threshold subtlex_mean=… --range-threshold mattr=… --range-threshold clause_pattern_entropy=… \
    --bound grammar_errors_per_100_words=… --bound cola_mean=… --bound subtlex_mean=… --bound mattr=… --bound clause_pattern_entropy=… \
    --listeners a,b,c --spacy-model-version … --languagetool-version … --cola-model … --cola-commit-hash … --subtlex SUBTLEX-US.txt
python -m experiments.proficiency_validation.code.cli verify-frozen
```

Model calls:

```bash
python -m experiments.proficiency_validation.code.cli run-challenge --items data/challenge_set.jsonl --judge deepseek-chat --out challenge_judged.jsonl
python -m experiments.proficiency_validation.code.cli report-challenge --items challenge_judged.jsonl
python -m experiments.proficiency_validation.code.cli generate --sources sources.jsonl --conditions low,neutral,high --generator deepseek-chat --judge deepseek-chat --out rewrites.jsonl
python -m experiments.proficiency_validation.code.cli guardrails --assembled features_assembled.jsonl --languagetool-url http://localhost:8081 --cola-model textattack/roberta-base-CoLA --rewrites rewrites.jsonl --politeness-judge deepseek-chat --out guardrails.jsonl
python -m experiments.proficiency_validation.code.cli probe --assembled features_assembled.jsonl --listeners deepseek-chat,openai:gpt-5-mini,gemini:gemini-2.5-flash --out listener.jsonl
```

Human pairwise judgement (blind):

```bash
python -m experiments.proficiency_validation.code.cli build-human-worksheet --assembled features_assembled.jsonl --pair low:neutral --n 48 --out worksheet.md --key key.json
python -m experiments.proficiency_validation.code.cli import-human-worksheet --worksheet worksheet_filled.md --key key.json --annotator rater_1 --out human_choices.jsonl
```

The worksheet builder refuses to write a sheet that leaks a condition; keep the key away from
the rater. Human rows are passed to `analyze --listener` alongside the model rows; a
`cannot_tell` answer is kept as an abstention and reported, not dropped.

## Confirmatory runs

After the pilot, `freeze` writes `data/frozen_config.json`. From then on pass `--frozen
data/frozen_config.json` to `generate`, `features`, `guardrails`, `probe` and `analyze`: each
command recomputes the code hashes, refuses to run if anything drifted, and pins its
parameters from the config. `analyze --frozen` also checks that the data files themselves were
produced under the frozen prompts, models and instruments, and prints a verdict in which a
criterion that could not be evaluated never counts as passed.

## Using the rewriter in a replay

```python
from components.rewriters.language_rewriter import LanguageProficiencyRewriter
from experiments.proficiency_validation.code.semantic_gate import GatedRewriter, SemanticGate

base = LanguageProficiencyRewriter(llm_client=client, condition="low", model=model_id)
rewriter = GatedRewriter(base, SemanticGate.from_handle("deepseek-chat"),
                         max_corrective_retries=2)           # default: fail closed (raise)
# for a live replay that must keep running, opt in explicitly AND log every fallback to disk:
rewriter = GatedRewriter(base, SemanticGate.from_handle("deepseek-chat"),
                         on_unavailable="original", log_path="runs/low_p3_gate_log.jsonl")
# assign `rewriter` to the focal speaker only (flip_self)
```

The gated rewriter is registered as `gated_proficiency`, so the engine's own paths build it
too (import the module first so the registration runs):

```bash
python -m lurkingwolf.run_game --experiment nl \
    --rewriter-module experiments.proficiency_validation.code.semantic_gate \
    --rewriter gated_proficiency \
    --rewriter-params '{"base": {"condition": "low", "model": "MODEL_ID"}, "gate": "deepseek-chat", "on_unavailable": "original", "log_path": "runs/gate_log.jsonl"}'
```

In YAML use `rewriter: GatedRewriter` with the same params. The bare `proficiency` rewriter
has no gate and no fallback log; do not use it for treated runs.

`on_unavailable="original"` keeps the game running and writes `fallback_to_original=True` to
the log file as it happens; PLAN.md D6 says how such branches are treated in the analysis.
Without it the wrapper raises, so no untreated text can be recorded under a treatment label.
