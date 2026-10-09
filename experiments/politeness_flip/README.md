# Politeness flip replays

RQ2 of the paper: when one designated speaker addresses the table with one Brown & Levinson
politeness strategy all day, do the other agents change how they vote against, target and
trust that speaker? This folder runs that question through the LURK replay framework with the
politeness rewriter validated in `experiments/politeness_validation/`.

| Path | Contents |
| --- | --- |
| `config.yaml` | The experiment: 30 seeds of 7-player NL baselines, two identity replays, `flip_self` and `flip_others` replays for the four strategies |
| `code/analyze.py` | Paired analysis of the treated speaker's outcomes, games as independent units |
| `summary.md` | The result |
| `code/manipulation_check.py` | Blind labelling of the delivered speech, and the report it writes |
| `manipulation_check.md` | Agreement with the requested strategy, and the confusion between them |
| `data/runs/qwen35_27b/` | This run: `manifest.json`, `politeness_summary.json` (every number in `summary.md`), the labelled sample and every classifier pass |
| `games/` | Baselines and replays |

## How a replay works

The baseline game and its day-start checkpoints are the ones the proficiency flip plays for
the same seed, so both experiments treat the same days; the cluster script copies them in
before the replays. A replay loads one day-start checkpoint and plays that day again. In
`flip_self` the rewriter sits on one player for the whole day, so every public utterance of
that player is rewritten with the assigned strategy before other agents read it. In
`flip_others` it sits on every other player. The `nl` replay repeats the day with no
rewriter and is the control each treated replay is paired with; a second `nl` replay of the
same checkpoint measures the model's run-to-run variation.

Each rewrite is logged to `rewrites.jsonl` in the replay directory. The politeness rewriter
fails closed: an unusable generation raises rather than returning untreated text under a
treatment label, so the replay fails and is re-run instead of quietly losing its treatment.
Unlike the proficiency rewriter it runs no semantic gate during the replay, so whether the
rewrites preserved the game content is checked afterwards on the recorded utterances.

## Conditions and contrasts

The four strategies are `bald_on_record`, `positive_politeness`, `negative_politeness` and
`off_record`. Every one of them rewrites the utterance, so a contrast between two strategies
isolates politeness. `bald_on_record` carries no redress and is the reference the other three
are read against. A strategy against the identity replay also carries the effect of being
rewritten at all, which the second identity replay measures on its own.

Outcomes, pairing and aggregation are the proficiency flip's: for a treated speaker S in
`flip_self`, per (game, day), the votes S received, whether S was executed, whether the
wolves or the seer targeted S that night, and the mean reliability score the village players
alive after the night assign to S. Differences are averaged over days within each speaker,
speakers within each game, and games, with equal weights at each step. Intervals are game
bootstraps and p-values sign-flip tests over games. Holm corrects the three strategies against
the reference within each of two outcomes, the trust in the speaker and the votes against them;
every other outcome and the contrasts against the identity replay are descriptive. Which family
RQ2 treats as confirmatory is the politeness experiment's decision to record before the numbers
are read.

## Did the speech realize the strategy?

The rewriter was validated on a separate sample in `experiments/politeness_validation/`. The
replays get their own check: the blind politeness classifier labels the utterances the other
players actually heard, three passes each, never told which strategy was requested.

```bash
export PROFICIENCY_MODEL_CONFIG=...   # the manifest that names the classifier handle
python -m experiments.politeness_flip.code.manipulation_check run --classifier classifier_qwen
python -m experiments.politeness_flip.code.manipulation_check report
```

The sample is balanced over the four strategies, over the two rewrite scopes and over the
games. `flip_others` rewrites about five times as many utterances as `flip_self`, so sampling
in proportion would leave the treated speakers who carry the primary analysis thinly covered. Passes are
appended and never overwritten, so a second classifier adds its own rows and the report shows
them side by side. On the cluster,
`experiments/proficiency_validation/hpc/politeness_check_qwen35_27b.slurm` runs both steps
with the served model.

## Run

Thirty seeds (201–230). On the cluster,
`experiments/proficiency_validation/hpc/politeness_flip_qwen35_27b.slurm` is a 30-task array
submitted with `--dependency=aftercorr:<proficiency array job id>` so each seed starts once
its baseline exists. After the last seed:

```bash
python -m experiments.politeness_flip.code.analyze
```
