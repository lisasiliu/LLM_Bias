# Politeness validation

Does the rewriter produce the politeness strategy it is asked for, and is the judge that labels strategies reliable?
Werewolf utterances were rewritten into the four Brown & Levinson strategies (bald on-record, positive, negative,
off-record), rated blind by two people for strategy and for whether the meaning survived, and compared with the LLM judge.
The judge was also checked against 135 items from Zhao & Hawkins (2025), an outside set labelled with the same strategies.

## Data (`data/`)

| file | contents |
| --- | --- |
| `items.jsonl` | the 52 rated rewrites: original and shown text, requested strategy, the judge's label, both raters' labels |
| `rewrite_lisa.csv`, `rewrite_rq.csv` | each rater's strategy, confidence and meaning label (`yes`/`no`/`unsure`) per item |
| `rewrite_judge.csv` | the judge's label per item, in the same layout (`gpt-5.6-luna`, majority of 3 passes) |
| `key.json` | where each item came from (`peNN:sNN` = seed and speech ID) and the requested strategy |
| `zhao-hawkins_judge.jsonl` | the 135 outside items: published label, mapped strategy, the judge's label |

## Code

- `components/rewriters/politeness_rewriter.py`: the rewriter and the trust-based setting policy
- `components/judges/blind_politeness_classifier.py`: the strategy judge
- `components/judges/gate.py`: the checks a rewrite must pass before it is delivered in a game
