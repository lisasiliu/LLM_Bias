# Politeness manipulation check on the replays

Every utterance below is one the other players heard in a treated replay. The blind politeness classifier of `experiments/politeness_validation/` labelled it without being told which strategy was requested, three independent passes per utterance, majority vote. The sample is balanced over the four strategies, over the two rewrite scopes and over the games.

## Classifier `classifier_qwen`

1000 utterances over 30 games; the requested strategy was the majority label for 0.503 of them (per game 0.41 to 0.60).

| requested | bald on record | positive politeness | negative politeness | off record | ambiguous or mixed | failed manipulation | no majority | failed | agreement |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bald on record | 250 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.000 |
| positive politeness | 187 | 38 | 25 | 0 | 0 | 0 | 0 | 0 | 0.152 |
| negative politeness | 35 | 0 | 215 | 0 | 0 | 0 | 0 | 0 | 0.860 |
| off record | 220 | 6 | 24 | 0 | 0 | 0 | 0 | 0 | 0.000 |

| rewrite scope | items | agreement |
|---|--:|--:|
| flip others | 500 | 0.516 |
| flip self | 500 | 0.490 |

0 of 3000 passes did not return a usable label (0.000).

## Classifier `openai:gpt-5.6-luna`

1000 utterances over 30 games; the requested strategy was the majority label for 0.602 of them (per game 0.47 to 0.72).

| requested | bald on record | positive politeness | negative politeness | off record | ambiguous or mixed | failed manipulation | no majority | failed | agreement |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bald on record | 242 | 4 | 0 | 0 | 0 | 0 | 4 | 0 | 0.968 |
| positive politeness | 85 | 108 | 17 | 0 | 0 | 0 | 38 | 2 | 0.432 |
| negative politeness | 7 | 0 | 227 | 0 | 0 | 0 | 13 | 3 | 0.908 |
| off record | 39 | 8 | 13 | 25 | 0 | 0 | 78 | 87 | 0.100 |

| rewrite scope | items | agreement |
|---|--:|--:|
| flip others | 500 | 0.600 |
| flip self | 500 | 0.604 |

652 of 3000 passes did not return a usable label (0.217).

Rows are the strategy the rewriter was asked for, columns the label the classifier gave. `ambiguous_or_mixed` and `failed_manipulation` are the classifier's own categories for an utterance that realizes no single strategy; `no_majority` and `failed` are votes where the three passes did not agree or returned nothing.

Generated 2026-10-09T23:04:12+00:00 from `data/runs/qwen35_27b/manipulation_passes.jsonl`.
