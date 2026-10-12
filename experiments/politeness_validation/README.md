# Politeness validation

Werewolf utterances rewritten into the four Brown & Levinson politeness strategies (bald on-record,
positive, negative, off-record), rated blind by two people, and compared with the LLM politeness judge.

## Data

All files are in `data/`.

| file | contents |
| --- | --- |
| `items.jsonl` | the 52 rated rewrites, one per line: original and rewritten text, the strategy that was requested, the LLM judge's label, and both raters' labels |
| `werewolf-labels_lisa.csv` | Lisa's strategy, confidence and meaning label for each of the 52 items |
| `werewolf-labels_rq.csv` | Runqi's labels for the same 52 items |
| `werewolf-labels_judge.csv` | the LLM judge's label for each of the 52 items, in the same layout as the two human label files (judge: gpt-5.6-luna, majority of 3 passes) |
| `key.json` | the hidden answer key for the 52 items: where each came from and which strategy was requested |
| `natural_lisa.md`, `natural_rq.md` | the 60 natural utterances for the second rating round (30 the LLM judge called bald, 30 non-bald), one identical sheet per rater, to fill in |
| `natural_key.json` | hidden key for those 60: the judge's label and group for each item; do not open before rating |
| `zhao-hawkins_judge.jsonl` | 135 items from Zhao & Hawkins (2025), used as an outside check of the LLM judge: published label, mapped strategy, judge label |

## Reusable components

- `components/rewriters/politeness_rewriter.py` - the rewriter
- `components/judges/blind_politeness_classifier.py` - the LLM politeness judge prompt and parser
- `components/judges/judge_model.py` - model access and the 3-pass vote
- `components/judges/inverse_predicate_parser.py`, `dimensional_semantic_validator.py`, `structural_check.py` - meaning checks
