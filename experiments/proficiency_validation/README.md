# English proficiency cue: validation

We vary the linguistic range of the English spoken by Werewolf agents at three levels while
holding meaning, speech-act force and grammaticality fixed, and we validate the manipulation
before it enters the game experiments. Validation combines automatic language measures, blind
model listeners from two model families and two human annotators on the same passage pairs.
[VALIDATION.md](VALIDATION.md) holds the current results.

## Conditions

| Condition | Definition |
| --- | --- |
| Original | The untreated utterance. |
| A | Restricted range: common, concrete vocabulary, repeated words, short clauses with little embedding. |
| B | A paraphrase at the source's own level: vocabulary level, repetition and clause structure judged unchanged. |
| C | Expanded range: less common and more precise vocabulary, more embedded syntax. |

Every condition is generated independently from the Original. A versus B is the primary
contrast. B versus Original tests that paraphrasing by itself leaves the measured properties
unchanged. C versus B describes the upper direction. The labels follow the CEFR-derived
A/B/C levels used in the automated proficiency literature; the conditions are relative levels
of one speaker's text, and the auxiliary CEFR ratings below are reported separately.

## Materials

Three untreated games (27 players, 280 public utterances) form the development corpus. The unit
of analysis is a speaker passage: 150 to 250 words of whole utterances by one player in one game,
drawn with a fixed seed from the early, middle and late thirds of that player's timeline and
kept in speech order. The utterance set is fixed on the Original before any rewrite exists and is
identical across conditions, which yields 24 passages. Whole-game transcripts (every public
utterance of a player) form a robustness cohort. Games are the independent units throughout:
players are averaged within a game, games receive equal weight, and intervals resample games.

## Generation and fidelity gate

Qwen3.5-27B rewrites each utterance under a condition-specific prompt that sees the public
history before the utterance. Each candidate passes three checks. (1) A deterministic structural
check: player IDs, predicate slots and polarity are unchanged, no hedge is added, "please" is
neither added nor removed. (2) An inverse parser recovers the speech act from the rewrite alone;
its slots must match the recorded predicate. (3) A fidelity judge aligns every span of the two
texts, marks each alignment as equivalent or not, and answers fixed questions on content,
certainty, force, speaker perspective, references, politeness and register; for B it also rates
vocabulary level, repetition and syntax as simpler, same or more advanced, and all three must be
"same". Incomplete judge answers get one repair request that names each problem. A candidate
that fails receives up to two corrective retries with the failure reasons; an utterance whose
candidates all fail is recorded as unavailable, and a passage enters a comparison only when every
one of its utterances is available in both conditions.

B generates five candidates per call. Candidates that change under 5% of the tokens are set
aside, the gate judges the rest, and the passer whose sentence count and length are closest to
the Original is kept. An optional selection rule (`--neutral-selection
closest_surface_and_frequency_v1`) ranks the passers first by the closeness of their mean
SUBTLEX word frequency to the Original's; the run records which rule was used. Judge calibration on 315 constructed rewrites (a broken target, role,
polarity, reason, certainty or directive force, plus preserved paraphrases) measures the gate's
sensitivity and specificity.

## Measures

**Linguistic range** (spaCy, SUBTLEX-US). The primary family is word frequency (SUBTLEX Lg10CD),
vocabulary diversity (MATTR, 50-token window) and subordinate clauses per 100 words, with
Benjamini–Hochberg correction within the family. MTLD, content-lemma repetition, dependency
distance, tree depth, noun-phrase length and sentence length are secondary diagnostics. A
sentence-splitting sensitivity analysis repeats the primary contrasts on the passages where A
changed the sentence count least.

**Grammar guardrails.** LanguageTool grammar errors per 100 words and CoLA acceptability. A must
be non-inferior to B against a fixed margin, and B must be equivalent to Original.

**Blind model listeners.** Two listeners from different model families (Qwen3.5-27B and
OpenAI's GPT luna) read two passages of the same player in both presentation orders and answer which
passage shows the more limited range of English, or "same". A second question asks which speaker
seems to have the weaker command of English. Answers are scored +1 (treatment judged more
limited), 0 (same) or −1, averaged over orders, listeners and passages within a game. A position
score (+1 for slot A, −1 for slot B) measures presentation-order preference. The pairs are A–B,
C–B, A–C and B–Original.

**Human annotation.** Two researchers independently label every complete pair of the four
comparisons (62 pairs in the current run) with the same three-way question, in a randomized
order with randomized sides. The analysis reports percent agreement and Cohen's κ between the
researchers, the direction and signed preference per comparison, consensus labels after
adjudication, each model listener's agreement with the consensus, and each model's consistency
across presentation orders.

**Auxiliary CEFR ratings.** A model assigns a CEFR band (A, B or C) with quoted evidence to each
passage; band distributions and paired transitions describe where the conditions fall.

**Inference.** Cluster bootstrap over games, an exact sign-flip test over games, BH within the
primary family, one-sided non-inferiority for the guardrails (A versus B) and TOST equivalence
for B versus Original. With three games the smallest attainable two-sided p is 0.25, so the
pilot reports estimates with intervals and the confirmatory study fixes its sample size from
the pilot's between-game variance.

## Pilot results

Run of 7 October 2026 (Qwen3.5-27B as generator, judge, first listener and CEFR rater;
GPT luna as second listener; prompts `range_v9`, rubric `english_fidelity_v7`). Full tables are
in [VALIDATION.md](VALIDATION.md).

- Gate acceptance: A 269 of 280 utterances, C 269, B 255 (20 B sources produced only near-copies
  below the 5% paraphrase floor and 5 returned the original unchanged). Primary cohort: 15
  complete A–B passages, 13 C–B, 15 A–C and 19 B–Original across all three games; the whole-game
  robustness cohort has 7 A–B and 7 C–B transcripts.
- Rewrite exposure (token edit distance to the original): A changes 0.25 of the tokens, C 0.47,
  B 0.10.
- A versus B (15 passages): all three primary measures in the predicted direction with intervals
  excluding zero (SUBTLEX +0.30, MATTR −0.022, subordinate clauses −1.2 per 100 words); MTLD
  −11.7, repetition +0.059, dependency distance −0.53, tree depth −1.18 and sentence length −27
  words agree. On the 9 passages where A split sentences least, all three primary measures hold
  (MATTR −0.013, interval [−0.029, −0.004]).
- B versus Original (19 passages): MATTR +0.001, subordinate clauses −0.17 (interval
  [−0.45, 0.11]), sentence length +0.1; B uses slightly rarer words (SUBTLEX −0.066, interval
  [−0.103, −0.007]) and repeats less (−0.025). Grammar errors −0.05 per 100 words, CoLA +0.02.
- C versus B (13 passages): rarer and more varied vocabulary (SUBTLEX −0.43, MATTR +0.013, MTLD
  +14); subordinate clauses −0.17 (interval [−1.36, 0.90]); dependency distance, tree depth and
  sentence length within noise.
- Guardrails for A versus B: grammar errors +0.02 per 100 words (one-sided bound 0.05), CoLA
  acceptability +0.17.
- Listeners, range question: both model families judge A more limited than B and than C in
  every passage, and C less limited than B in every passage, in both presentation orders (Qwen
  answers "same" once in 30 A–B comparisons). On B versus Original the Qwen listener answers
  "same" in 92% of pairs (signed preference −0.07); luna answers "same" in 55% and otherwise calls
  the Original more limited (−0.43, interval [−0.64, −0.30]).
- Listeners, proficiency question: Qwen calls most pairs the same (60% of A–B, 81% of C–B) and
  otherwise picks the expected side (A–B +0.42, A–C +0.71); luna agrees with the expected side on
  C–B (−0.56) and answers A–B inconsistently across orders (+0.03; 8 of 15 passages receive the
  same answer in both orders).
- Judge calibration on 315 constructed rewrites: sensitivity 0.996, specificity 0.978 under
  both parser policies.
- CEFR: 6 of 19 A passages receive band B and 13 band C; every rated B, C and Original passage
  receives band C.
- Power: six games per primary measure and for the range-listener score at 80% power for the
  A–B contrast.

## Files

| Path | Content |
| --- | --- |
| `rewrite/` | Rewriting and the fidelity gate (`python -m experiments.proficiency_validation.rewrite`) |
| `validation/` | Passages, measures, listeners, CEFR ratings, human annotation, power, export and the report (`python -m experiments.proficiency_validation.validation`) |
| `data/corpus/` | The untreated utterances and their baseline measures |
| `data/runs/pilot_qwen35_27b/` | The current run: every rewrite attempt, passages, measures, listener answers, CEFR ratings and `manifest.json` |
| `data/researcher_1_worksheet.{md,csv}`, `data/researcher_2_worksheet.{md,csv}` | Blind annotation sheets; the `_key.json` files hold the hidden condition assignment |
| `data/manual_validation.jsonl` | Imported human labels and consensus (written by `import-annotations`) |
| `data/provenance.json`, `data/automatic_validation.jsonl` | Publication export: models, settings, hashes, and one row per passage and condition |
| `tests/` | Offline tests (`python -m pytest experiments/proficiency_validation/tests`) |
| `hpc/` | Cluster job scripts (untracked) |

The rewriter is `components/rewriters/language_rewriter.py`.

## Commands

The cluster script `hpc/pilot_qwen35_27b.slurm` runs the whole pipeline for one run directory:
generation, features, guardrails, both listener probes, CEFR ratings, judge calibration, report,
power and export. Pipeline changes are first exercised end to end with DeepSeek flash as
generator, judge and rater through the same commands (a run directory outside version control,
with `thinking` disabled in the model entry), and the Qwen run follows once the change is settled.
The second listener runs locally against the Qwen run's passages:

```bash
RUN=experiments/proficiency_validation/data/runs/pilot_qwen35_27b
export PROFICIENCY_MODEL_CONFIG=$RUN/manifest.json
for construct in range perceived_proficiency; do
  python -m experiments.proficiency_validation.validation probe \
      --assembled $RUN/validation/linguistic_features_assembled.jsonl --listeners openai:gpt-5.6-luna \
      --pairs low:neutral,high:neutral,low:high,neutral:original --construct $construct \
      --max-tokens 8192 --concurrency 8 --out $RUN/validation/${construct}_listener_openai.jsonl
done
python -m experiments.proficiency_validation.validation report --pilot-dir $RUN \
    --out experiments/proficiency_validation/VALIDATION.md --json $RUN/validation/report.json
python -m experiments.proficiency_validation.validation power --report $RUN/validation/report.json --out $RUN/validation/power.json
python -m experiments.proficiency_validation.validation export --pilot-dir $RUN
```

Human annotation:

```bash
DATA=experiments/proficiency_validation/data
python -m experiments.proficiency_validation.validation build-worksheets \
    --assembled $RUN/validation/linguistic_features_assembled.jsonl --out-dir $DATA
# each researcher fills in more_limited (A / B / same) in their own worksheet, then:
python -m experiments.proficiency_validation.validation import-annotations \
    --worksheet $DATA/researcher_1_worksheet.md --key $DATA/researcher_1_key.json --rater researcher_1
python -m experiments.proficiency_validation.validation import-annotations \
    --worksheet $DATA/researcher_2_worksheet.md --key $DATA/researcher_2_key.json --rater researcher_2
python -m experiments.proficiency_validation.validation consensus-worksheet \
    --assembled $RUN/validation/linguistic_features_assembled.jsonl        # disagreements only
python -m experiments.proficiency_validation.validation import-consensus --worksheet $DATA/consensus_worksheet.md
python -m experiments.proficiency_validation.validation annotation-summary \
    --listener $RUN/validation/range_listener.jsonl $RUN/validation/range_listener_openai.jsonl
```

The report includes the human section once `data/manual_validation.jsonl` exists. Internal
condition identifiers are `low`, `neutral`, `high` and `original`; the report and the export
display them as A, B, C and Original.

## Confirmatory protocol

A formal run freezes the protocol first (`validation freeze` on a clean commit): prompts, rubric,
measurement code and model identities by hash; disjoint game cohorts with passages chosen from
the Original before rewriting; effect-size thresholds for the primary measures, guardrail
non-inferiority margins, equivalence bounds for B versus Original, the listener margin and
minimum acceptance rates. Every confirmatory command runs with `--frozen` at that commit, with
the strict parser policy in which parser abstention leaves a candidate without a verdict.
Confirmation requires every primary measure to meet its direction, BH-adjusted significance and
effect-size criteria, the guardrails to pass, B to be equivalent to Original in range, grammar
and listener judgments, each listener to meet the signed-effect criterion in both orders, and the
position score to stay within its margin.
