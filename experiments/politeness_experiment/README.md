# Politeness flip experiment

## Question

When one player's speech is delivered in a fixed politeness setting (flip_self), or everyone else's is
(flip_others), do the other players change who they vote for, what they do at night, and how much
they trust the speaker?

## Setup

Seven-player games (2 werewolves, seer, guard, 3 villagers), all `deepseek-chat`, temperature 0.
Twenty natural baseline games (`pe01`-`pe20`) are played once with checkpoints saved at each day start.
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

**Rewrites.** Each rewrite is read back by a model that did not write it (gpt-5.6-luna). An inverse
parser sees only the rewrite and the public record and must recover the original predicate, and a
semantic validator checks commitment, act force, evidence, the referenced statement and the speaker's
perspective, and flags invented or deleted content. Both must agree. If either fails, the speaker's
original text is delivered and logged.

## Outcomes

Skill targets, vote targets and executions compared with the baseline phase, as in the language
experiment. Reliability scores and the engine's town-consensus score are both recorded before and
after the phase. The blind judge (gpt-5.6-luna) labels every delivered text to check the setting.

## Running

```bash
python -m experiments.politeness_experiment.code.run play --seed pe21     # a new baseline
python -m experiments.politeness_experiment.code.run flip --first 1 --last 2 --workers 6
python -m experiments.politeness_experiment.code.analysis check|judge|metrics
```

Each replay writes `result.json`, `cost.json` and `touched.jsonl` (one row per treated utterance:
listener, rank, assigned and delivered strategy, original and final text) to
`games/flip_experiments/<seed>/day<N>/<mode>__<setting>__<player>/`, next to the baseline in `<seed>/natural/`.
