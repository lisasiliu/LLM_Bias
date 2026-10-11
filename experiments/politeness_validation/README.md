# Politeness validation

Werewolf utterances rewritten into the four Brown & Levinson politeness strategies (bald on-record,
positive, negative, off-record), rated blind by two people, and compared with the LLM politeness judge.

## Data

All files are in `data/`.

| file | contents |
| --- | --- |
| `items.jsonl` | the 72 rated utterances, one per line: original and rewritten text, the strategy that was requested, the LLM judge's label, and both raters' labels |
| `labels_lisa.csv` | Lisa's strategy, confidence and meaning label for each of the 72 items |
| `labels_rq.csv` | Runqi's labels for the same 72 items |
| `key.json` | the hidden answer key for the 72 items: where each came from and which strategy was requested |
| `zhao_hawkins.jsonl` | 135 items from Zhao & Hawkins (2025), used as an outside check of the LLM judge: published label, mapped strategy, judge label |

## Reusable components

- `components/rewriters/politeness_rewriter.py` - the rewriter
- `components/judges/blind_politeness_classifier.py` - the LLM politeness judge prompt and parser
- `components/judges/judge_model.py` - model access and the 3-pass vote
- `components/judges/inverse_predicate_parser.py`, `dimensional_semantic_validator.py`, `structural_check.py` - meaning checks
