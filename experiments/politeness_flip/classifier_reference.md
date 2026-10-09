# How well the classifier reads the four strategies

The items below come from the two sets with known answers in `experiments/politeness_validation/`: the Werewolf rewrites two researchers labelled by hand, and the held-out Zhao & Hawkins items with published labels. Agreement here measures the classifier, not the rewriter.

## Classifier `openai:gpt-5.6-luna`

183 items from 2 reference sets; the known answer was the majority label for 0.656 of them.

| requested | bald on record | positive politeness | negative politeness | off record | ambiguous or mixed | failed manipulation | no majority | failed | agreement |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| bald on record | 41 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0.953 |
| positive politeness | 5 | 40 | 1 | 1 | 0 | 0 | 7 | 3 | 0.702 |
| negative politeness | 11 | 1 | 35 | 1 | 0 | 0 | 6 | 3 | 0.614 |
| off record | 12 | 0 | 0 | 4 | 0 | 0 | 6 | 4 | 0.154 |

| reference set | items | agreement |
|---|--:|--:|
| werewolf rewrites | 48 | 0.729 |
| zhao hawkins | 135 | 0.630 |

97 of 549 passes did not return a usable label (0.177).

Rows are the strategy the rewriter was asked for, columns the label the classifier gave. `ambiguous_or_mixed` and `failed_manipulation` are the classifier's own categories for an utterance that realizes no single strategy; `no_majority` and `failed` are votes where the three passes did not agree or returned nothing.

Generated 2026-10-09T23:04:11+00:00 from `data/runs/qwen35_27b/classifier_reference_passes.jsonl`.
