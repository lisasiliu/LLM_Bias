# Politeness validation

Validation of controlled Brown & Levinson politeness rewrites of LURK Werewolf
utterances: whether each rewrite realizes the requested strategy, and whether it
preserves game-relevant meaning.

Read **[VALIDATION.md](VALIDATION.md)** for the results and what they mean.

## Data

| file | contents |
| --- | --- |
| `data/manual_validation.jsonl` | one record per rewritten utterance: provenance, predicate, speaker, untreated original, requested strategy, rewrite, the blind manual annotation, every automatic verdict, and the prompt/rubric hashes it was produced under |
| `data/external_validation.jsonl` | one record per external construct-check item: the published label, how it maps onto our strategies, the split, and our automatic prediction |
| `data/provenance.json` | source runs, sampling and annotation procedure, what was hidden from the annotator, models and settings, frozen prompt and rubric hashes, git commit |
| `data/independent_annotation_worksheet.md` | a 20-item blind worksheet prepared for a second, independent annotator; not yet collected |
| `data/independent_annotation_key.json` | the hidden record for that worksheet — do not open it before annotating |

The JSONL files are the scientifically useful artifacts; everything in
`VALIDATION.md` is computed from them.

## Commands

All run offline — none of them calls a model.

Build a blank worksheet for a new blind annotation round:

```bash
python -m experiments.politeness_validation.code.validation build-worksheet \
    --items data/manual_validation.jsonl \
    --out worksheet.md --key worksheet_key.json
```

It shows the predicate, the original, the rewrite and any referenced statement,
hides the requested strategy and all automatic verdicts, shuffles the items, and
refuses to write a worksheet that leaks a hidden verdict. `--compact` asks only
for strategy, confidence, meaning and a note.

Merge a completed worksheet back into the hidden record:

```bash
python -m experiments.politeness_validation.code.validation import-annotations \
    --worksheet worksheet.md --key worksheet_key.json \
    --out data/manual_validation.jsonl
```

### Second-annotator reliability round

`data/independent_annotation_worksheet.md` is ready to send to an independent
annotator. It is a 20-item subsample of the 48, balanced 5 per strategy and 5
per predicate so that all sixteen strategy x predicate cells appear, shuffled
under a different seed from the first round. Once it comes back:

```bash
python -m experiments.politeness_validation.code.validation import-annotations \
    --worksheet independent_annotation_worksheet.md \
    --key data/independent_annotation_key.json \
    --out data/independent_annotation.jsonl
```

Each record keeps the first round's `item_id`, so the two annotations can be
compared item by item.

Reproduce every statistic in `VALIDATION.md`:

```bash
python -m experiments.politeness_validation.code.validation analyze
```

Check that the frozen prompts and the semantic rubric still match what
`provenance.json` records:

```bash
python -m experiments.politeness_validation.code.validation verify-provenance
```

## Reusable components

The rewriter and the judges live outside this directory because the downstream
listener experiment uses them too:

- `components/rewriters/politeness_rewriter.py` — the frozen rewriter
- `components/judges/politeness_classifier.py` — the strategy classifier
  (diagnostic only)
- `components/judges/inverse_parser.py` — the inverse predicate parser
- `components/judges/semantic_dimensions.py` — the dimensional semantic validator
- `components/judges/semantic.py` — deterministic structural checks
- `components/validation/adjudication.py` — the accept/queue policy
- `components/validation/sources.py` — reading untreated sources from a game run
