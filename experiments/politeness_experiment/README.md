# Politeness flip experiment

## Question

When one player's speech is delivered in a fixed politeness setting (flip_self), or everyone else's is
(flip_others), do the other players change who they vote for, what they do at night, and how much
they trust the speaker?

## Setup

Seven-player games (2 werewolves, seer, guard, 3 villagers), all `deepseek-chat`, temperature 0.
Thirty natural baseline games (`pe01`-`pe30`) are played once with checkpoints saved at each day start.
Every replay starts from a day-start checkpoint and plays that day: discussion, vote, night, then the
players' reliability update.

**Replay.** The replay repeats the baseline's model responses, matched on the exact prompts, up to the
first utterance that is changed, and draws fresh responses from there on. All variance therefore starts
at the changed line. The control replay keeps the natural text but branches at that same line, so its
difference from the baseline is the noise floor.

**Speech acts.** The nine acts whose wording can carry politeness: `estimate`, `question`, `agree`,
`disagree`, `recommend_action`, `address_role`, `allied`, `unallied`, `dissociate`. Role claims,
`answer` and `fluff` stay as the speaker wrote them. Wolves are treated like everyone else.

**Settings.**

| setting | trusted listener | distrusted listener |
| --- | --- | --- |
| `all_bald` | bald | bald |
| `all_positive` | positive | positive |
| `trust_positive` | positive | bald |
| `trust_bald` | bald | positive |

The listener is the player the utterance is addressed to or about: the target of `estimate`,
`question` and `dissociate`, the author of the statement answered by `agree` and `disagree`, the actor
of `recommend_action`. `address_role`, `allied` and `unallied` have no single listener and are
delivered bald in the trust settings. A speaker ranks the living players by their own current
reliability scores; the top half, including the middle player of an odd count, is trusted and the
rest is distrusted.

**Rewrites.** A rewrite is delivered only if it keeps what the speech act needs: every player number in the original
(the speaker may become "I" and the addressee "you"), role labels, referenced speech IDs, the action and its target, polarity,
and any reasons the original gave. Most of this is checked by rule; two yes/no questions go to gpt-5.6-luna (is the
action the same toward the same target; are the original's reasons still given). A rewrite that fails is regenerated once with
the same prompt at temperature 0.7, and if it fails again the speaker's original text is delivered. Both attempts are logged.

## Outcomes

Skill targets, vote targets and executions compared with the baseline phase, as in the language
experiment. Reliability scores and the engine's town-consensus score are both recorded before and
after the phase. The blind judge (gpt-5.6-luna) labels every delivered text to check the setting.

## Running

```bash
python -m experiments.politeness_experiment.code.run play --seed pe21     # a new baseline
python -m experiments.politeness_experiment.code.run flip --first 1 --last 2 --workers 6
python -m experiments.politeness_experiment.code.analysis export    # games/ -> data/*.csv
python -m experiments.politeness_experiment.code.analysis effects   # data/player_days.csv -> results.md
python -m experiments.politeness_experiment.code.analysis judge     # blind judge on delivered texts
```

Each replay writes `result.json`, `cost.json` and `touched.jsonl` (one row per treated utterance:
listener, rank, assigned and delivered strategy, original and final text) to
`games/flip_experiments/<seed>/day<N>/<mode>__<setting>__<player>/`, next to the baseline in `<seed>/natural/`.

## Results

Thirty seeds, 2,890 replays, 289 player-days with a control replay (`data/replays.csv`, `data/player_days.csv`).
Each setting is reported as its difference from the control replay in percentage points (skill and vote targets, scores moved)
or rating points, with a 95% interval that resamples whole seeds. `results.md` has every column, with and without `pe01`.

| setting | skill targets (pp) | vote targets (pp) | scores others give the self player (village assessors, pts) |
| --- | --: | --: | --: |
| `all_bald` | +2.3 [+0.4, +4.0] | +1.0 [-0.9, +2.8] | 0.00 [-0.05, +0.05] |
| `all_positive` | +1.5 [-1.3, +4.5] | -1.4 [-3.2, +0.4] | -0.08 [-0.14, -0.02] |
| `trust_positive` | +1.4 [-0.7, +3.6] | -1.0 [-2.6, +0.5] | -0.03 [-0.08, +0.02] |
| `trust_bald` | +1.7 [-0.6, +4.3] | -0.7 [-3.2, +1.8] | -0.01 [-0.05, +0.04] |

No setting moves vote targets or the share of reliability scores that change. Skill targets rise by about two points under
`all_bald` only. The one reliable shift in how others rate the speaker is a drop of 0.08 points under `all_positive`.

**Delivery.** Of 7,785 treated utterances, the speaker's original wording was delivered when both attempts failed the
checks in 11.0% of `all_bald`, 1.3% of `all_positive`, 6.0% of `trust_positive` and 7.2% of `trust_bald` utterances; the
bald rewrites fail mostly because a justification was trimmed. A fallback dilutes a setting toward the control.
`data/seed_setting_summary.csv` has the counts per seed.
