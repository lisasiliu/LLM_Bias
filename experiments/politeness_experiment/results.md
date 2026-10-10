# Results

## Games

20 seeds with at least one finished game.

| Seed | natural | natural_replay | all_bald | liking | flipped | all_positive | bl_written |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pe01 | done | done | done | done | done | - | done |
| pe02 | done | done | done | done | done | - | done |
| pe03 | done | done | done | done | done | - | done |
| pe04 | done | - | done | done | done | - | done |
| pe05 | done | - | done | done | done | - | done |
| pe06 | done | - | done | done | done | - | done |
| pe07 | done | - | done | done | done | - | done |
| pe08 | done | - | done | done | done | - | done |
| pe09 | done | - | done | done | done | - | done |
| pe10 | done | - | done | done | done | - | done |
| pe11 | done | - | done | done | done | - | done |
| pe12 | done | - | done | done | done | - | done |
| pe13 | done | - | done | done | done | - | done |
| pe14 | done | - | done | done | done | - | done |
| pe15 | done | - | done | done | done | - | done |
| pe16 | done | - | done | done | done | - | done |
| pe17 | done | - | done | done | done | - | done |
| pe18 | done | - | done | done | done | - | done |
| pe19 | done | - | done | done | done | - | done |
| pe20 | done | - | done | done | done | - | done |

## Fallbacks and delivery

| Arm | Touched utterances | Assigned rewrite delivered | Bald fallback | Original text | Any fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_bald | 239 | 214 | 0 | 25 | 10% |
| liking | 256 | 198 | 28 | 30 | 23% |
| flipped | 275 | 249 | 14 | 12 | 9% |
| all_positive | 0 | 0 | 0 | 0 | - |

Fallback rate by speech act:

| Arm | Speech act | Utterances | Any fallback |
| --- | ---: | ---: | ---: |
| all_bald | question | 60 | 18% |
| all_bald | disagree | 1 | 100% |
| all_bald | recommend_action | 178 | 7% |
| liking | question | 75 | 24% |
| liking | disagree | 1 | 100% |
| liking | recommend_action | 180 | 22% |
| flipped | question | 79 | 4% |
| flipped | disagree | 1 | 0% |
| flipped | recommend_action | 195 | 12% |

Positive rewrites assigned and delivered: `trust_positive` 94 of 126, `trust_bald` 75 of 90.

## Manipulation check

The blind judge's label for the delivered text matches the assigned strategy:

| Arm | Rank group | Utterances | Match |
| --- | ---: | ---: | ---: |
| all_bald | trusted | 116 | 108 of 116 (93%) |
| all_bald | distrusted | 69 | 68 of 69 (99%) |
| all_bald | middle | 36 | 35 of 36 (97%) |
| all_bald | unrated | 18 | 14 of 18 (78%) |
| liking | trusted | 126 | 95 of 126 (75%) |
| liking | distrusted | 86 | 86 of 86 (100%) |
| liking | middle | 31 | 31 of 31 (100%) |
| liking | unrated | 13 | 13 of 13 (100%) |
| flipped | trusted | 135 | 127 of 135 (94%) |
| flipped | distrusted | 90 | 77 of 90 (86%) |
| flipped | middle | 37 | 35 of 37 (95%) |
| flipped | unrated | 13 | 11 of 13 (85%) |

## Primary outcome: change in the addressee's rating of the speaker

Change from the rating in force when the speaker spoke to the addressee's next rating round, for village-side addressees. Averaged per seed, then compared between arms across seeds.

| Arm | Seeds | Mean change per seed | Touched utterances | No usable change (village addressee) |
| --- | ---: | ---: | ---: | ---: |
| all_bald | 14 | +0.269 | 239 | 127 |
| liking | 10 | +0.240 | 256 | 135 |
| flipped | 14 | +0.229 | 275 | 147 |
| all_positive | 0 | - | 0 | 0 |

No usable change means the addressee had not yet rated the speaker when the speaker spoke (105 utterances in total) or no later rating round exists because the addressee died or the game ended (304).

Paired by seed:

| Comparison | Seed pairs | Mean difference | Permutation p | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: |
| liking vs all_bald | 8 | -0.154 | 0.547 | 0.625 |
| flipped vs all_bald | 12 | -0.175 | 0.234 | 0.273 |
| liking vs flipped | 9 | +0.046 | 0.812 | 0.688 |
| all_positive vs all_bald | 0 | - | - | - |

Addressees the speaker ranked **trusted**:

| Comparison | Seed pairs | Mean difference | Permutation p |
| --- | ---: | ---: | ---: |
| liking vs all_bald | 6 | -0.278 | 0.500 |
| flipped vs all_bald | 10 | +0.075 | 0.875 |
| liking vs flipped | 7 | -0.131 | 0.688 |
| all_positive vs all_bald | 0 | - | - |

Addressees the speaker ranked **distrusted**:

| Comparison | Seed pairs | Mean difference | Permutation p |
| --- | ---: | ---: | ---: |
| liking vs all_bald | 2 | -0.500 | 1.000 |
| flipped vs all_bald | 2 | -0.250 | 1.000 |
| liking vs flipped | 4 | +0.125 | 1.000 |
| all_positive vs all_bald | 0 | - | - |

Werewolf addressees (their score is not trust), descriptive:

| Arm | Utterances | Mean change |
| --- | ---: | ---: |
| all_bald | 15 | +0.267 |
| liking | 24 | +0.333 |
| flipped | 25 | +0.520 |
| all_positive | 0 | - |

## Secondary outcomes (descriptive)

| Arm | Games | Village wins | Day executions that hit a werewolf | vote_execute recommendations followed |
| --- | ---: | ---: | ---: | ---: |
| natural | 20 | 11 of 20 (55%) | 23 of 37 (62%) | - |
| all_bald | 20 | 9 of 20 (45%) | 20 of 40 (50%) | 75 of 142 (53%) |
| liking | 20 | 6 of 20 (30%) | 20 of 41 (49%) | 71 of 141 (50%) |
| flipped | 20 | 10 of 20 (50%) | 22 of 39 (56%) | 90 of 157 (57%) |
| all_positive | 0 | no cases | no cases | - |

## Drift check (natural replay against the baseline)

| Seed | Utterances identical | First difference | Same executions | Same winner |
| --- | ---: | ---: | ---: | ---: |
| pe01 | 1 of 83 | utterance 1 | no | yes |
| pe02 | 0 of 83 | utterance 0 | no | yes |
| pe03 | 0 of 73 | utterance 0 | no | yes |

A second natural replay of pe01 from the same checkpoint (`natural_replay_b`) has 0 of 83 utterances identical to the first replay, so replays differ from one another about as much as from the baseline: the variation comes from the model, not from restoring the checkpoint.

## Limitations

- One model family (`deepseek-chat`) plays every role, writes the rewrites and runs two of the three checks.
- Twenty seeds are few; the secondary outcomes are descriptive only.
- Reliability ratings are the agents' own LLM-generated numbers, mostly between 4 and 6.
- The blind judge is weak on off-record, which no arm uses; its other labels were validated on a small set.
- Replays from a checkpoint do not reproduce the baseline game exactly (see the drift check), so arms are compared with each other, not with the baseline's later play.
