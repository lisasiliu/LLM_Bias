# Language flip experiment

## Question

When one player's speech is delivered at a restricted or an expanded linguistic range (flip_self), do the other players change
who they vote for, what they do at night, and how much they trust the speaker?

## Setup

Seven-player games (2 werewolves, seer, guard, 3 villagers), all `deepseek-chat`, temperature 0. Thirty natural baseline games
(`pe01`-`pe30`, the same baselines as the politeness experiment) are played once with checkpoints saved at each day start. Every
replay starts from a day-start checkpoint and plays that day: discussion, vote, night, then the players' reliability update.
Each living player is the self player in turn.

**Replay.** The replay repeats the baseline's model responses, matched on the exact prompts, up to the first utterance that is
changed, and draws fresh responses from there on. The control replay keeps the natural text but branches at that same line, so its
difference from the baseline is the noise floor (two control replays per player-day, one replay per level).

**Levels.** The self player's speech acts that can be rewritten are delivered in one of three levels:

| level | rewrite |
| --- | --- |
| A | restricted range: simpler vocabulary, shorter and plainer sentences |
| B | neutral paraphrase at the original range (controls for being paraphrased) |
| C | expanded range: richer vocabulary and more complex syntax |

**Gate.** The rewrite keeps every player number of the original (the speaker may become "I", the addressee "you"), role labels,
speech IDs, the action and target, polarity and the original's reasons (two yes/no questions go to gpt-5.6-luna). Level B must
also keep the original's vocabulary, repetition and syntax range. A rewrite that fails is regenerated once at temperature 0.7, and
if it fails again the speaker's original text is delivered and logged.

## Running

```bash
python -m experiments.language_flip.code.run flip --first 1 --last 2 --workers 6
python -m experiments.language_flip.code.analysis export     # games/ -> data/*.csv
python -m experiments.language_flip.code.analysis effects    # data/player_days.csv -> results.md
```

Replays are written to `games/flip_experiments/language/<seed>/day<N>/<mode>__<level>__<player>/`.

## Results

Thirty seeds, 1,445 replays, 289 player-days with controls. Each level is reported as its difference from the control replay, and
A and C also against B, in percentage points of skill targets, vote targets and reliability scores that differ from the baseline
phase, with 95% intervals that resample whole seeds. `results.md` has every column, with and without `pe01`.

| contrast | skill targets (pp) | vote targets (pp) | scores others give the self player (village assessors, pts) |
| --- | --: | --: | --: |
| A - control | +1.7 [-1.0, +4.4] | +2.1 [-0.7, +4.8] | +0.03 [-0.02, +0.09] |
| B - control | -1.0 [-3.6, +1.7] | -2.1 [-4.5, +0.2] | +0.04 [-0.02, +0.08] |
| C - control | +2.5 [+0.2, +5.2] | +1.8 [-0.8, +4.6] | +0.02 [-0.03, +0.07] |
| A - B | +2.7 [-0.4, +6.0] | +4.2 [+1.3, +7.3] | -0.01 [-0.05, +0.03] |
| C - B | +3.4 [+0.6, +6.6] | +3.9 [+1.2, +6.5] | -0.01 [-0.07, +0.04] |

Changing the range in either direction, relative to a neutral paraphrase, raises the share of other players' vote targets that
change by about four points; the self player's trust scores do not move.

**Delivery.** Of 3,615 rewrites the original wording was delivered when both attempts failed the checks for 1.2% (A), 0.7% (B) and
1.0% (C); at most 8.7% on any seed and level. `data/seed_level_summary.csv` has the counts.

## Files

| file | contents |
| --- | --- |
| `data/replays.csv` | one row per replay |
| `data/rewrites.csv` | one row per rewrite: level, predicate, attempts, failure reasons, original and delivered text |
| `data/seed_level_summary.csv` | per seed and level: rewrites, first-attempt failures, rescues by the retry, originals delivered, cost |
| `data/player_days.csv` | control and level values of the seven measures per (seed, day, self player) |
| `results.md` | the tables above for all columns |
