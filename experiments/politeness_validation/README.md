# Politeness validation

Controlled Brown & Levinson politeness rewrites of LURK Werewolf utterances, and
the evidence that they realize the requested strategy and keep the game meaning.
Results and method: [VALIDATION.md](VALIDATION.md).

## Data

| file | contents |
| --- | --- |
| `data/items.jsonl` | the 72 rated items: source, predicate, original and rewritten text, requested strategy, the LLM judge's label, and both raters' labels |
| `data/labels_lisa.csv`, `data/labels_rq.csv` | each rater's strategy, confidence and meaning label per item |
| `data/key.json` | the hidden key: source, requested strategy and delivery for each item |

## Reusable components

- `components/rewriters/politeness_rewriter.py` - the rewriter
- `components/judges/blind_politeness_classifier.py` - the LLM politeness judge prompt and parser
- `components/judges/judge_model.py`, `majority_vote.py` - model access and the 3-pass vote
- `components/judges/inverse_predicate_parser.py`, `dimensional_semantic_validator.py`, `structural_check.py` - meaning checks
