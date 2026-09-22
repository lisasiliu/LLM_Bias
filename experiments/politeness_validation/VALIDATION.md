# Politeness Validation

Does a controlled rewrite realize the requested Brown & Levinson strategy, and
does it preserve the game-relevant meaning? Every number here is recomputed from
`data/` by `python -m experiments.politeness_validation.code.validation analyze`.

## Setup

- 12 untreated source utterances from natural game play, across the 4 speech-act
  types that occur naturally (`estimate`, `disagree`, `question`,
  `recommend_action`)
- each source rewritten into all 4 politeness strategies — bald on-record,
  positive politeness, negative politeness, off-record — holding the underlying
  structured predicate fixed
- **48 matched rewrites** generated from the same underlying source content
  under four politeness conditions
- both researchers annotated the same 48 independently
- the requested strategy and every automatic judgement were hidden from both

## Manipulation check

| Requested strategy | n | Rater 1 | Rater 2 |
| --- | ---: | ---: | ---: |
| Bald on-record | 12 | 12/12 | 12/12 |
| Positive politeness | 12 | 12/12 | 12/12 |
| Negative politeness | 12 | 12/12 | 12/12 |
| Off-record | 12 | 12/12 | 12/12 |
| **Overall** | **48** | **48/48** | **48/48** |

Inter-rater strategy agreement: **48/48 (100%)**, Cohen's κ = **1.000**
(unweighted; the labels are nominal).

Neither annotator used `ambiguous_or_mixed` or `failed_manipulation` for any
item.

## Meaning preservation

| Rater | Yes | No | Unsure |
| --- | ---: | ---: | ---: |
| Rater 1 | 43 | 3 | 2 |
| Rater 2 | 41 | 5 | 2 |

Exact agreement: **46/48 (95.8%)**, Cohen's κ = **0.816**. `unsure` is kept as
its own category and is never folded into either side.

The two disagreements are both `recommend_action` rewrites, where rater 2 was
the stricter reader:

| item | Rater 1 | Rater 2 | requested |
| --- | --- | --- | --- |
| `item_001` | yes | no | bald on-record |
| `item_028` | yes | no | off-record |

Across both raters, every judgement of lost meaning falls on
`recommend_action`, suggesting that action directives are especially sensitive
to rewriting. Not adjudicated.

## Semantic quality control

Stimulus construction runs each rewrite through deterministic structural checks,
an inverse predicate parser that reconstructs the predicate without being shown
it, and a dimensional semantic validator; anything the parser and the validator
do not both clear goes to human review. The validation result above comes from
the two independent human judgements, not from these automatic checks, whose
per-item outputs are retained in `data/manual_validation.jsonl`.

## Automatic politeness classifier

Agreement with the human annotations was 34/48 against each rater. Recall was
substantially lower than the annotators' for positive-politeness and off-record
realizations, so the classifier is used **only diagnostically and never as an
acceptance criterion**. Its per-item predictions are retained in the JSONL.

## External construct check

Brown & Levinson supplies the theoretical strategy definitions. Zhao & Hawkins
(2025) provides an external construct check, with the caveat that their
annotation unit is the marker span while ours is the whole utterance. Of the 90
items their annotation assigns to positive or negative politeness, our
classifier placed 44 in the same pair; within those 44 it agreed with the
published family on **95.5%**. This is **conditional** construct evidence for
the positive/negative distinction, not a general classifier accuracy.

## Methodological decision

The four-way manipulation is supported by two independent blind annotations that
agree completely. Meaning preservation is quantified by both raters and is high,
with the few losses concentrated in `recommend_action`.
Automatic politeness labels are diagnostic only. The politeness manipulation is
therefore validated. Downstream experiments use only rewrites that pass semantic
quality control or subsequent human adjudication.

**Limitation.** One annotator was an author of the study; the second annotator
labeled the items independently and blind to treatment.
