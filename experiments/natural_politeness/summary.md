# Politeness of untreated speech across prompt levels

**Main result.** Untreated LURK agents speak bald-on-record at every level of
strategic prompt guidance. In a random sample of 150 primary speech acts (30 per
level, L1-L5), 149 were labelled bald-on-record (99.3%; Wilson 95% CI
96.3-99.9%). The one exception was a `please`-mitigated request labelled negative
politeness. No positive-politeness or off-record act was observed. The amount of
strategy in the system prompt changes what agents argue about. It does not change
how politely they say it.

## Setup

We played 146 nine-player games with `deepseek-chat` at five prompt-guidance
levels. L1 is the bare rules, and L5 is the full default strategy prompt (D018).
The seeds `np01`-`np30` were shared, so every level received the same role deals.
A few L2 and L3 runs are missing, which leaves L1 30, L2 29, L3 27, L4 30 and
L5 30. We took every utterance realizing one of the five primary predicates
(`estimate`, `question`, `agree`, `disagree`, `recommend_action`), 8,678 in all,
and drew 30 per level at random. Each was labelled with one of the four Brown &
Levinson strategies under the `act_first_v2` rubric of the frozen classifier.

## Results

| level | n | bald | positive | negative | off-record |
|---|--:|--:|--:|--:|--:|
| L1 | 30 | 30 | 0 | 0 | 0 |
| L2 | 30 | 30 | 0 | 0 | 0 |
| L3 | 30 | 30 | 0 | 0 | 0 |
| L4 | 30 | 29 | 0 | 1 | 0 |
| L5 | 30 | 30 | 0 | 0 | 0 |
| all | 150 | 149 | 0 | 1 | 0 |

The same result appears in the independent human audit of the earlier 30-game L5
corpus, where 47 of 50 random utterances were bald (Q29). So natural play is a
near-uniform bald baseline, and the politeness rewriter is the only source of
strategy variation. Measuring politeness effects therefore requires a controlled
rewrite treatment. It cannot come from observational variation in the corpus.

The act mix is stable across levels, with one exception: L5 produces about twice
as many `question` acts per game (5.9, against 2.0-2.8 at L1-L4). `agree` is
almost absent everywhere (5 of 8,678), as Q25/Q27 note.

## Limitations

- The labels come from one pass by an LLM (Claude), not from a human annotator,
  and the labeller could see the level and role. This is a diagnostic. It is not
  a validated measurement.
- With n = 30 per level, a level could still have up to about 11% non-bald
  speech (the lower bound of the per-level CI is 88.6%), and this sample would
  not detect it. No between-level contrast was tested.
- The L5 arm was run in a separate session. Its manifests do not record a
  strategy level, so L5 is identified only because it uses the default prompt.

## Files

| file | contents |
|---|---|
| `data/utterances.jsonl` | all 8,678 primary-predicate utterances, with level, game, speaker, role and predicate |
| `data/labelled_sample.jsonl` | the 150 sampled utterances and their labels |
| `code/politeness_by_level.py` | `extract` rebuilds `utterances.jsonl` from `games_fresh/`; `analyze` prints every number above |

The sample was drawn once and is kept as data. Its shuffle was seeded through
Python's per-process `hash()`, so the draw itself cannot be reproduced.
