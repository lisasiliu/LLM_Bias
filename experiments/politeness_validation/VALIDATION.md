# Politeness validation

We rewrite natural Werewolf utterances into four politeness strategies and check
that each rewrite (1) realizes the requested strategy and (2) keeps the game
meaning. Every number below is recomputed from `data/` by
`python -m experiments.politeness_validation.code.validation analyze`.

## Method

**Strategies.** Brown & Levinson (1987): bald on-record, positive politeness,
negative politeness, off-record. Positive and negative politeness both state the
act and soften it (for solidarity or for the listener's autonomy). Off-record
leaves the act itself for the listener to infer.

**Source utterances.** Untreated speech from LURK games in which every player is
`deepseek-chat` (DeepSeek's non-reasoning alias of `deepseek-flash`) and no
rewriter is applied. The validation items use 12 utterances, three each of
`estimate`, `disagree`, `question` and `recommend_action`, taken from two
9-player games.

**Rewriter.** `deepseek-chat`, temperature 0. One short prompt per strategy,
written for this study from Brown & Levinson's definitions. All prompts forbid
changing the speaker, target, polarity or reason. Positive politeness may add
rapport but not invented praise or evidence; negative politeness softens the act,
not the truth claim; off-record keeps the target and proposition recoverable. A
failed or empty rewrite is an error, never the source text.

**Meaning checks.** Deterministic checks (players, referenced statements, action,
polarity), an inverse parser that reconstructs the predicate from the rewrite
without seeing it, and a semantic validator over dimensions such as commitment,
act force and evidence. Disagreement between them goes to a human.

**Human annotation.** Two researchers independently labelled all 48 rewrites (12
sources x 4 strategies). They did not see the requested strategy or any
automatic output. They saw the predicate, the original utterance and the rewrite,
and judged the dominant strategy and whether the meaning was preserved.

**LLM politeness judge.** `gpt-5.6-luna` (OpenAI; no dated version is published).
The prompt has two steps: say what the speaker is doing, then decide how that act
is realized. Stated bluntly is bald; stated with solidarity is positive; stated
with deference or reduced imposition is negative; left for the listener to infer
is off-record. The judge sees only the utterance, speaker and phase, as a
listening player would, and is never told the requested strategy. The model
cannot be run at temperature 0, so each item is judged three times and the label
at least two passes agree on is kept; if all three differ the item counts as a
miss. All 549 passes are in `data/classifier_passes.jsonl`.

## Results

### Human manipulation check

| Requested strategy | n | Researcher 1 | Researcher 2 |
| --- | ---: | ---: | ---: |
| Bald on-record | 12 | 12/12 | 12/12 |
| Positive politeness | 12 | 12/12 | 12/12 |
| Negative politeness | 12 | 12/12 | 12/12 |
| Off-record | 12 | 12/12 | 12/12 |
| **Overall** | **48** | **48/48** | **48/48** |

The researchers agree on all 48 labels (Cohen's kappa = 1.0). Neither used
`ambiguous` or `failed`.

### Meaning preservation

| | Yes | No | Unsure |
| --- | ---: | ---: | ---: |
| Researcher 1 | 43 | 3 | 2 |
| Researcher 2 | 41 | 5 | 2 |

Agreement is 46/48 (kappa = 0.816). Every judgement of lost meaning is on a
`recommend_action` rewrite.

### LLM politeness judge against the human labels

Agreement with the annotators, averaged over the two: **88/96 (91.7%)**.

| Human label | n | judge agrees |
| --- | ---: | ---: |
| Bald on-record | 12 | 12 |
| Positive politeness | 12 | 12 |
| Negative politeness | 12 | 12 |
| Off-record | 12 | 8 |

The four misses are off-record rewrites that turn a direct challenge into a
statement ("I'm still trying to line up two things you've said..."); the judge
labelled them negative or positive politeness.

### LLM politeness judge on Zhao & Hawkins (non-Werewolf)

As an external check we ran the judge on 135 held-out items from Zhao & Hawkins
(2025), short feedback messages labelled with Brown & Levinson strategies. Their
labels are per politeness marker, made by an LLM and verified by hand, and each
scenario has a hidden 0-3 rating of how much the speaker liked the performance.

Four-way agreement is **70.4% (95/135)**.

| Published label | n | judge agrees |
| --- | ---: | ---: |
| Bald on-record | 31 | 31 |
| Positive politeness | 45 | 32 |
| Negative politeness | 45 | 31 |
| Off-record | 14 | 1 |

Of the 90 positive or negative items, the judge placed 64 in that pair and
labelled 63 of them the same way as the published label.

| Text | Published | Judge |
| --- | --- | --- |
| "It wasn't bad! You looked so good up there!" | positive | positive |
| "It was not good." (hidden rating 0) | off-record | bald |
| "I've seen you performing better." (hidden rating 0) | off-record | off-record |

Their off-record items are mostly plain evaluations from low-rated scenarios, so
the indirectness comes from a hidden opinion a listener cannot see. We read
"It was not good." as a stated evaluation, so this set does not test
off-record; it supports the bald, positive and negative categories.

## Limitations

- 12 source utterances from two games, played with 9 players, so the 48 rewrites
  are 12 matched sets rather than 48 independent items.
- Both annotators are authors of the study. They saw the predicate and original
  when labelling strategy, which a listening player would not; a rewrite-only
  re-annotation has not been done.
- The judge misses a third of off-record rewrites (8/12 on Werewolf, 1/14 on
  Zhao & Hawkins), and the Werewolf set was also used to choose the judge model.
- The inverse parser and semantic validator run on `deepseek-chat`, the model
  that wrote the rewrites.
