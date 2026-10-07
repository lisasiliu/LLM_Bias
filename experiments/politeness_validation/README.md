# Politeness validation

Controlled Brown & Levinson politeness rewrites of LURK Werewolf utterances, and
the evidence that they realize the requested strategy and keep the game meaning.
Results and method: [VALIDATION.md](VALIDATION.md).

## Data

| file | contents |
| --- | --- |
| `data/manual_validation.jsonl` | one record per rewrite: source, predicate, original, requested strategy, rewrite, both human annotations, the LLM politeness judge's label and the meaning-check outputs |
| `data/external_validation.jsonl` | the 135 Zhao & Hawkins items: published label, mapped strategy, judge label |
| `data/classifier_passes.jsonl` | every judge pass (3 per item) with the evidence it gave |
| `data/provenance.json` | sources, sampling, annotation procedure, models and settings |
| `data/researcher_{1,2}_worksheet.md` | the blind worksheets the annotators filled in |
| `data/researcher_{1,2}_key.json` | the hidden key for each worksheet |

## Commands

Recompute every statistic in VALIDATION.md (offline):

```bash
python -m experiments.politeness_validation.code.validation analyze
python -m experiments.politeness_validation.code.validation verify-provenance
```

Re-run the LLM politeness judge on both reference sets (calls the OpenAI API,
about $0.20), then write its labels into the two JSONL files:

```bash
python -m experiments.politeness_validation.code.classifier_eval run
python -m experiments.politeness_validation.code.classifier_eval apply
```

Build a blind worksheet for a new annotation round, and merge it back:

```bash
python -m experiments.politeness_validation.code.validation build-worksheet \
    --items data/manual_validation.jsonl --out worksheet.md --key worksheet_key.json
python -m experiments.politeness_validation.code.validation import-annotations \
    --worksheet worksheet.md --key worksheet_key.json --annotator rater_3 \
    --out data/manual_validation.jsonl
```

## Reusable components

- `components/rewriters/politeness_rewriter.py` - the rewriter
- `components/judges/blind_politeness_classifier.py` - the LLM politeness judge prompt and parser
- `components/judges/judge_model.py`, `majority_vote.py` - model access and the 3-pass vote
- `components/judges/inverse_predicate_parser.py`, `dimensional_semantic_validator.py`, `structural_check.py` - meaning checks
