# Politeness validation

Does the rewriter produce the politeness strategy it is asked for, does the rewrite keep the game meaning, and is the LLM judge
that labels strategies reliable? Natural Werewolf utterances were rewritten into the four Brown & Levinson strategies (bald
on-record, positive politeness, negative politeness, off-record), rated blind by two researchers, and compared with the judge.
The judge was also checked on 135 items from Zhao & Hawkins (2025), an outside set labelled with the same strategies.

Every number below is recomputed from `data/` by

```bash
python -m experiments.politeness_validation.code.analysis      # prints the tables and writes results.md
```

## Method

**Source utterances.** 13 untreated utterances from 11 seven-player games in which every player is `deepseek-chat` and no
rewriter is applied: `estimate` (3), `question` (3), `recommend_action` (4), `disagree` (1), `address_role` (1), `unallied` (1).
Each was rewritten into all four strategies, giving 52 items.

**Rewriter.** `deepseek-chat`, temperature 0, one short prompt per strategy written from Brown & Levinson's definitions.
All prompts forbid changing the speaker, target, polarity or reason. Positive politeness may add rapport but not invented praise or
evidence; negative politeness softens the act, not the truth claim; off-record keeps the target and proposition recoverable.
When a bald rewrite dropped content that the checks require, the original wording was shown instead (3 of the 13 bald items).

**Human annotation.** Two researchers independently labelled all 52 items. They saw the predicate, the original utterance and the
rewrite, not the requested strategy or any automatic label. For each item they gave the dominant strategy (or `ambiguous`), a
confidence, and whether the meaning was preserved (`yes`, `no`, `unsure`).

**LLM judge.** `gpt-5.6-luna`, three passes, label kept when at least two agree. It sees only the utterance, the speaker and the phase, as a
listening player would, and is never told the requested strategy. It first says what act the speaker performs, then how the act is
realized: stated bluntly is bald, stated with solidarity is positive, stated with deference or reduced imposition is negative, left
for the listener to infer is off-record.

## Results

### The requested strategy was recognised

| requested strategy | n | Researcher 1 | Researcher 2 | judge |
| --- | ---: | ---: | ---: | ---: |
| Bald on-record | 13 | 12 | 13 | 13 |
| Positive politeness | 13 | 12 | 13 | 11 |
| Negative politeness | 13 | 12 | 11 | 13 |
| Off-record | 13 | 9 | 13 | 8 |
| **All** | **52** | **45 (87%)** | **50 (96%)** | **45 (87%)** |

The researchers gave the same strategy on 43 of 52 items (83%, Cohen's kappa 0.77). They disagree mostly on off-record
(four items, where Researcher 1 read a positive, negative or bald act) and between positive and negative politeness (four
items). One label was `ambiguous`.

### Meaning preservation

| | yes | no |
| --- | ---: | ---: |
| Researcher 1 | 46 | 6 |
| Researcher 2 | 37 | 15 |

Neither used `unsure`. The researchers agree on 37 of 52 items (kappa 0.14, low because Researcher 2 marks far more losses). Both
say the meaning is kept on 34 items and both say it is lost on 3; 18 items are flagged by at least one. Flags concentrate in
off-record rewrites (12 of 18) and `recommend_action` (8 of 18), where indirectness can leave the action or target unclear.

### Judge against the researchers

The judge matches Researcher 1 on 42 of 52 items (81%, kappa 0.74) and Researcher 2 on 43 (83%, kappa 0.77). On the 43 items
where the researchers agree it matches on 39 (91%):

| researchers' label | n | judge agrees |
| --- | ---: | ---: |
| Bald on-record | 12 | 12 |
| Positive politeness | 12 | 11 |
| Negative politeness | 10 | 10 |
| Off-record | 9 | 6 |

Off-record is the weak class: the judge labels some off-record rewrites as the bald, positive or negative act they imply.

### Judge on Zhao & Hawkins (2025)

Short feedback messages labelled with Brown & Levinson strategies; their labels mark politeness markers, made by an LLM
and verified by hand. Four-way agreement is 95 of 135 (70.4%, kappa 0.59).

| published label | n | judge agrees |
| --- | ---: | ---: |
| Bald on-record | 31 | 31 |
| Positive politeness | 45 | 32 |
| Negative politeness | 45 | 31 |
| Off-record | 14 | 1 |

Of the 90 positive or negative items the judge placed 64 in that pair, and 63 of them match the published label. The published
off-record items are mostly plain evaluations from low-rated scenarios ("It was not good."), where the indirectness comes from a
hidden opinion; we read them as stated evaluations, so this set supports the bald, positive and negative categories rather than
testing off-record.

## Limitations

- 13 source utterances, each in all four strategies, so the 52 items are 13 matched sets rather than independent items.
- Both annotators are authors of the study and saw the predicate and original text, which a listening player would not.
- Researchers disagree on meaning preservation (kappa 0.14); the strategy result does not depend on it.
- The judge misses a third of off-record rewrites on the researchers' labels and almost all of them on Zhao & Hawkins.

## Files

| file | contents |
| --- | --- |
| `data/items.jsonl` | the 52 rated items: original and shown text, requested strategy, judge label, both raters' labels |
| `data/rewrite_lisa.csv`, `data/rewrite_rq.csv` | each rater's strategy, confidence and meaning label per item |
| `data/rewrite_judge.csv` | the judge's label per item |
| `data/key.json` | where each item came from (`peNN:sNN` = seed and speech ID) |
| `data/zhao-hawkins_judge.jsonl` | the 135 outside items: published label, mapped strategy, judge label |
| `code/analysis.py` | recomputes every table above and writes `results.md` |

Code used elsewhere: `components/rewriters/politeness_rewriter.py` (rewriter), `components/judges/blind_politeness_classifier.py`
(judge), `components/judges/gate.py` (checks a rewrite passes before delivery in a game).
