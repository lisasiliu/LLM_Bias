# Proficiency validation

We rewrite what Werewolf agents say so that one speaker sounds like a less proficient
English speaker, without changing what they say. Before any game experiment uses these
rewrites, this folder checks two things: that each rewrite still means the same as the
original, and that the rewrites really differ in linguistic range (vocabulary and syntax)
the way we intend. Everything here is automated. [VALIDATION.md](VALIDATION.md) is the
current report.

## What is in this folder

| Path | What it is |
| --- | --- |
| `rewrite/` | Generates the rewrites and runs the fidelity gate. Entry point: `python -m experiments.proficiency_validation.rewrite` |
| `validation/` | Builds passages, measures them, runs the model listeners and CEFR ratings, writes the report. Entry point: `python -m experiments.proficiency_validation.validation` |
| `data/corpus/` | The 280 untreated utterances from three games (`run01`–`run03`, 9 players, `deepseek-chat`) and their baseline measurements |
| `data/runs/pilot_qwen35_27b/` | The current pilot: every rewrite attempt, the passages, measurements, listener answers, CEFR ratings and `manifest.json` (models, settings, hashes) |
| `tests/` | Offline tests, no model calls: `python -m pytest experiments/proficiency_validation/tests` |
| `hpc/` | Cluster job scripts (not tracked by git) |

The rewriter itself lives in `components/rewriters/language_rewriter.py`; the version of a
game that uses it goes through `rewrite.gate.GatedRewriter`.

## The three conditions

| Condition | What it is |
| --- | --- |
| Original | The untreated utterance |
| Low | Same content, restricted range: common concrete words, repeated rather than varied vocabulary, less embedded syntax. Grammar stays correct; no errors, no accent |
| Neutral | An independent paraphrase with the same range as the original: new wording, same vocabulary level, clause structure and register. It separates "being rewritten by the model at all" from "being rewritten as Low". It is not matched to Low; whether it is really neutral is tested against Original |
| High | Same content, wider range: more precise and less common vocabulary, more embedded syntax |

The main comparison is Low versus Neutral. Neutral versus Original checks that the control
really is neutral. High versus Neutral is the mirror image for the wider-range cue and is
reported as descriptive: High sits outside the primary comparison family in the pilot.

## Passages

The unit of measurement is a passage: 150 to 250 words of whole utterances by one player in one
game, drawn with a fixed seed from the early, middle and late thirds of that player's timeline
and kept in speech order. The utterance IDs are chosen from the untreated game before any rewrite
exists and are the same in every condition. The three games give 27 player timelines, 24 of which
have enough words. A passage only enters a Low versus Neutral comparison when every one of its
utterances passed the gate in both conditions; we never stitch together the ones that happened to
pass. In the setup table of the report, "Utterances" is the number of utterances in the whole game
and "Bundles" the number of qualifying passages.

As a robustness check the report also measures whole-game transcripts (every public utterance a
player made in one game). One missing rewrite removes a whole transcript, so that cohort is smaller
and moves between runs; it shows whether the cue persists across a whole game.

## How every rewrite is checked

Each rewritten utterance goes through three checks before anything else uses it:

1. **Structural check.** Deterministic: the player IDs, predicate slots and polarity are
   unchanged, no hedge like "maybe" was added, no "please" was added or removed.
2. **Inverse parser.** A model that only sees the rewrite and the public history before it
   parses the rewrite back into a predicate; the slots must match. The original is parsed the
   same way, so a disagreement between the original and its recorded predicate is visible too.
   When the parser recovers no predicate at all from the rewrite, it abstains and the fidelity
   judge decides alone, exactly as for the 74 utterances whose predicates the parser does not
   cover; the report counts these rewrites separately (`--parser-abstention no_verdict` restores
   the stricter rule). When a second judge from another model family is configured
   (`--secondary-judge`), it re-judges every such rewrite and only agreement counts as a pass.
   On the constructed test
   set the parser catches 18% of the broken rewrites and never flags a preserved one, and every
   break it catches the judge catches too; the judge calibration reports both policies.
3. **Fidelity judge.** A model with a fixed rubric must quote aligned spans from both texts
   that cover every word, mark each alignment equivalent or not, and answer yes or no on
   content, certainty, speech-act force, speaker perspective, references, politeness and
   register. One failed alignment or dimension rejects the rewrite. If the judge's answer is
   incomplete (a quote that is not in the text, an empty quote, a dimension answered "n/a"
   although the utterance answers an earlier statement) it gets one repair request that names
   each problem; if the answer is still incomplete there is no verdict. Quotes are matched word
   by word, so a comma that moved inside a quotation mark does not count as a misquote. A
   repair can never turn a detected violation into a pass.

When the parser cannot recover the recorded predicate from the original itself (the corpus
label and the text disagree), both texts go through a blind speech-act inventory instead; the
rewrite passes this step only if both inventories list the same acts. An inventory whose
evidence is lifted from the game history rather than the utterance gets one repair request.

A rewrite that fails gets up to two corrective retries with the failure reasons as feedback.
If it still fails, the utterance is recorded as unavailable. The original text is never used
under a treatment label. Every attempt, candidate and verdict is kept in `rewrite/rewrites.jsonl`.

Neutral is generated on its own, with no knowledge of Low: five candidates in one call, each
through the same gate plus three directional judge questions (is the vocabulary, the repetition
pattern and the syntax simpler, the same, or more advanced than the original? only "same"
passes, so a drift upward is caught as well as a drift down). A candidate that changed fewer
than 5% of the tokens is a near-copy and cannot be selected (the active-paraphrase floor).
Among the remaining passers the one whose surface profile stays closest to the original is kept
(same sentence count first, then the smallest change in word count; exact ties by a fixed seed).
The primary range measures are not used for this choice, so the Neutral-versus-Original test
stays independent. How much each condition rewrote (token edit distance) is recorded and
reported as exposure, not used as a requirement. Earlier pilots matched Neutral to Low's edit
distance; that lost about one Neutral in eight because a paraphrase that keeps the clause
structure cannot move as many tokens as a rewrite that splits sentences.

Two more checks on the judge itself. `--secondary-judge HANDLE` (for example `deepseek-chat`)
re-judges every accepted rewrite with a second model and records whether it agrees; it never
gates and is off unless set. The judge calibration step builds constructed rewrites that break
exactly one thing or preserve the meaning, judges them with the same gate, and reports how
many broken ones were caught and how many preserved ones were let through.

## What we measure on the passages that passed

- **Linguistic range** (spaCy and SUBTLEX-US). The primary family has three measures, one per
  construct: word frequency (SUBTLEX Lg10CD, higher in Low), vocabulary diversity (MATTR, lower
  in Low) and subordinate clauses per 100 words (lower in Low). Only this family is corrected for
  multiple testing. MTLD, repetition, dependency distance, tree depth, noun-phrase length and
  sentence length are secondary diagnostics; the number of distinct clause patterns has no
  expected direction (splitting one long sentence into several raises it, and the untreated
  L1–L5 games show no level trend). A sentence-splitting sensitivity table repeats the primary
  contrasts on the passages where Low changed the sentence count the least.
- **Grammar** (LanguageTool 6.6 grammar rules per 100 words; CoLA acceptability): Low may be
  more acceptable than Neutral but not meaningfully worse (one-sided non-inferiority against a
  frozen margin); Neutral must be equivalent to Original (two-sided, 90% interval). Both must
  clear absolute quality floors.
- **Blind model listeners**: given two passages of the same player (both orders), which one
  shows more limited vocabulary and sentence range, or is there no meaningful difference? A
  second question asks which speaker seems to have the weaker command of English. Answers are
  scored +1 (treatment more limited), 0 (same) or −1, averaged over the two orders, then over
  listeners, then over passages within a game. A position score (+1 for choosing slot A, −1 for
  B) checks that the listener follows the texts, not the presentation order. These are two
  separate measurements.
- **CEFR A/B/C rating** (auxiliary only): a model rates each passage A, B or C with quoted
  evidence (one repair request if a quote is inexact). It is auxiliary.

Games are the independent units: players are averaged within a game, games are averaged with
equal weight, confidence intervals resample games, and p-values come from an exact sign-flip
test over games. With three games the smallest possible p is 0.25; the pilot reports
direction and size.

## Where the pilot stands

Run of 6 October 2026 (prompts `range_v9`, rubric `english_fidelity_v7`, independent Neutral with
the 5% paraphrase floor, chronological-thirds passages primary, three-way listener answers,
SUBTLEX-US, LanguageTool 6.6 and CoLA; Qwen3.5-27B plays every role; 59 minutes on one H200):

- Low passes the gate for 270 of 280 utterances, High for 268, Neutral for 257. Of Neutral's 23
  failures, 18 never produced a candidate above the paraphrase floor and 5 returned the original
  unchanged. 4 acceptances per condition rest on the judge alone because the parser recovered no
  predicate.
- Rewrite exposure (token edit distance): Low changes a mean 0.25 of the tokens, High 0.47,
  Neutral 0.10.
- 17 of 24 player passages have complete Low and Neutral versions (15 for High), from all 3
  games; the whole-game robustness cohort has 6 and 7.
- Low versus Neutral (17 passages): all three primary measures in the predicted direction with
  the bootstrap interval excluding zero: word frequency (SUBTLEX Lg10CD) +0.31, MATTR −0.017,
  subordinate clauses −1.2 per 100 words; MTLD −9.5, repetition +0.069, dependency distance
  −0.61, tree depth −1.33, sentence length −30 words agree. On the 11 passages where Low split
  sentences the least, word frequency and subordination hold; MATTR shrinks to −0.008 with an
  interval that includes zero. Grammar errors per 100 words differ by 0.02 (one-sided bound
  0.04); CoLA acceptability is 0.17 higher for Low.
- Neutral versus Original (21 passages): MATTR −0.002, subordinate clauses −0.03, dependency
  distance +0.03, sentence length +0.9; Neutral uses slightly rarer words (SUBTLEX −0.066,
  interval [−0.096, −0.019]) and repeats slightly less (−0.024). Grammar errors −0.04, CoLA +0.02.
- High versus Neutral (15 passages): rarer and more varied vocabulary (SUBTLEX −0.46, MATTR
  +0.022, MTLD +19); subordinate clauses −0.7 per 100 words with an interval touching zero;
  dependency distance, tree depth and sentence length within noise.
- Blind listeners with a "same" option: on the range question Low is judged more limited than
  Neutral in 17 of 17 passages (signed preference +0.92, interval [0.83, 1.00], 6% ties); High is
  never judged more limited (15 of 15); Neutral and Original are called the same in 95% of
  pairs (signed preference −0.04, interval [−0.13, 0.00]). On the proficiency question the
  listener calls 74% of Low–Neutral pairs the same and otherwise picks Low (+0.25, interval
  [0.17, 0.29]); Neutral and Original are the same in 95%. Position scores stay within ±0.08
  except High–Neutral on the proficiency question (+0.27, 70% ties).
- Judge calibration on 315 constructed rewrites: sensitivity 0.996, specificity 0.978 under the
  deferred parser policy; 0.996 and 0.977 under the strict one.
- CEFR: 7 of 19 Low passages rated B, everything else C; three invalid responses.
- Power: with the pilot's between-game spread, six games give 80% power for each primary
  measure and for the listener signed preference (six is also the exact sign-flip floor).

## Planning, freezing and exporting

- `python -m experiments.proficiency_validation.validation power --report "$RUN/validation/report.json"`
  turns the pilot's game-level effects into the number of games a confirmatory set needs
  (80% and 90% power; the exact sign-flip test needs at least six games). Equivalence margins
  are inputs (`--equivalence-margin mattr=0.01 --listener-margin 0.2`), never read from data.
- `validation freeze` refuses a dirty working tree and records the clean commit; every
  confirmatory command (`--frozen`) must run at that exact commit. Confirmatory runs freeze the
  strict parser policy (`parser_abstention_policy: no_verdict`), the paraphrase floor, the
  parser context budget, the listener margins and the non-inferiority margins.
- `validation export --pilot-dir "$RUN"` writes `data/provenance.json` (models, settings, hashes,
  acceptance, coverage, calibration) and `data/automatic_validation.jsonl` (one row per passage
  and condition with every automatic measurement), the publication-facing files that mirror
  the collaborator's politeness validation. Raw outputs stay under `data/runs/`.

## Reproduce the report

```bash
python -m experiments.proficiency_validation.validation report \
    --pilot-dir experiments/proficiency_validation/data/runs/pilot_qwen35_27b \
    --out experiments/proficiency_validation/VALIDATION.md \
    --json experiments/proficiency_validation/data/runs/pilot_qwen35_27b/validation/report.json
python -m experiments.proficiency_validation.rewrite verify-run \
    --rewrites experiments/proficiency_validation/data/runs/pilot_qwen35_27b/rewrite/rewrites.jsonl
python -m pytest experiments/proficiency_validation/tests
```

Reporting reads stored files only. `rewrite generate --dry-run` prints the prompts without
calling a model. `validation prepare-data <game dirs>` rebuilds `data/corpus/` from raw game
logs (kept locally in `experiments/validation/`, ignored by git).

## Run a pilot

Install with `pip install -e '.[proficiency]'` and `python -m spacy download en_core_web_sm`.
Point `PROFICIENCY_MODEL_CONFIG` at the run's `manifest.json`; its `models` block names each
handle's model, family, endpoint variable and weight revision. Use a new run directory
whenever the prompts, rubric, model or settings change.

```bash
RUN=experiments/proficiency_validation/data/runs/pilot_qwen35_27b
SRC=experiments/proficiency_validation/data/corpus/sources.jsonl
export PROFICIENCY_MODEL_CONFIG="$RUN/manifest.json"

python -m experiments.proficiency_validation.rewrite generate \
    --sources "$SRC" --conditions low,neutral,high --generator generator --judge judge \
    --judge-max-tokens 3072 --parser-abstention defer --out "$RUN/rewrite/rewrites.jsonl"
python -m experiments.proficiency_validation.validation bundles \
    --sources "$SRC" --selection phase_stratified --window 150 250 --out "$RUN/validation/bundles.jsonl"
python -m experiments.proficiency_validation.validation features \
    --bundles "$RUN/validation/bundles.jsonl" --rewrites "$RUN/rewrite/rewrites.jsonl" \
    --conditions original,low,neutral,high --out "$RUN/validation/linguistic_features.jsonl"
python -m experiments.proficiency_validation.validation probe \
    --assembled "$RUN/validation/linguistic_features_assembled.jsonl" --listeners listener_qwen \
    --pairs low:neutral,high:neutral,neutral:original --construct range --out "$RUN/validation/range_listener.jsonl"
python -m experiments.proficiency_validation.validation probe \
    --assembled "$RUN/validation/linguistic_features_assembled.jsonl" --listeners listener_qwen \
    --pairs low:neutral,high:neutral,neutral:original --construct perceived_proficiency \
    --out "$RUN/validation/perceived_proficiency_listener.jsonl"
python -m experiments.proficiency_validation.validation cefr-rate \
    --bundles "$RUN/validation/bundles.jsonl" --rewrites "$RUN/rewrite/rewrites.jsonl" \
    --raters listener_qwen --out "$RUN/validation/cefr_auxiliary.jsonl"
# Robustness cohort: the same four steps with --selection whole_game, written to
# "$RUN/validation/whole_game/"; the report picks that folder up when it exists.
python -m experiments.proficiency_validation.validation report --pilot-dir "$RUN" \
    --out experiments/proficiency_validation/VALIDATION.md --json "$RUN/validation/report.json"
```

Other commands: `features --subtlex PATH` adds lexical frequency; `guardrails` runs
LanguageTool and CoLA on the assembled passages; `build-challenge --max-items 300`,
`run-challenge` and `report-challenge` test the judge on constructed broken and preserved
rewrites (the report picks the result up from `validation/calibration/report.json`);
`cefr-calibrate` scores the CEFR rater on externally labelled learner texts. For parallel
generation, give each worker `--shard-index i --shard-count n`, then `rewrite merge-rewrites`.
`--resume` continues an interrupted run with the same settings.

## Before a formal run

The pilot is for development. A formal run freezes the protocol first
(`validation freeze`, checked by every later command with `--frozen`):

- the exact prompts, rubric, measurement code and model identities (by hash);
- the games: a pilot set, Set A and Set B with no games in common, and the passages chosen
  from Original before any rewriting;
- the thresholds: minimum effect sizes for the primary range measures, equivalence bounds
  for grammar and for Neutral versus Original, the listener margin and minimum acceptance rates
  for Low and Neutral, each with a written reason;
- independent models: the judge from a different family than the generator, at least three
  listener families different from both and from the agents playing the game.

A formal result needs every primary range measure in the predicted direction with a
confidence interval excluding zero after multiplicity correction, grammar equivalence,
Neutral equivalent to Original, every listener above the margin in both presentation
orders, and acceptance rates above the frozen minima.

## Scope

The measurements describe model-generated game dialogue as read by model listeners; the cue
is a property of the displayed text. Whole-game passages depend on how long a player survived
and how often they spoke.
