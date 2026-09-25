# Linguistic-Proficiency-Cue Validation Plan

**Conditions:** Low / Neutral paraphrase / Original, with an optional High extension.
**Status:** pre-registration draft. Pilot not yet run. Everything marked *frozen at pilot* is
decided by the pilot and written to `data/frozen_config.json` before Confirmatory Set A.

This document is the English, code-facing version of the working plan
(`ref/Linguistic_Proficiency_Cue_Validation_Plan.docx`). Section 11 lists every
place where it deviates from that plan and why.

## 1. Construct and conditions

The treatment is a **restricted linguistic-range cue** carried by one speaker across all of
their messages in a replay branch: basic, repetitive vocabulary and narrow, simple syntax,
while the text stays grammatical, natural and fully understandable. It is *not* a CEFR level,
*not* an error rate, and *not* an accent or first-language stereotype. Grammaticality is a
guardrail (Family B), so that Low differs from Neutral in range, not in correctness.

| condition | role | linguistic realization | held fixed |
| --- | --- | --- | --- |
| Original | untreated Stage-2 baseline | the Stage-2 utterance as generated | everything |
| Neutral | paraphrase control | new surface wording at the same range, register, politeness, certainty; edit distance matched to Low | predicate, reasons, conclusion, speaker/target, act force, certainty, politeness |
| Low | treatment | basic and repetitive vocabulary, narrow simple syntax, direct structure; grammatical | same as Neutral |
| High | extension (pilot-gated) | precise vocabulary, richer syntactic choice, mature cohesion | same as Neutral |

The primary causal contrast is **Low − Neutral**: both sides have been rewritten by the same
model under the same semantic contract with matched surface change, so the contrast isolates
the range cue from "having been rewritten at all". Neutral − Original is the control check;
Low − Original is descriptive; High − Neutral is exploratory.

Prompts: `components/rewriters/language_rewriter.py` (`PROMPT_VERSION = range_v1`). The
shared semantic contract forbids changes to certainty, speech act, politeness and game content
and forbids manufactured errors; the condition specifications are quoted in the paper.

## 2. Hypotheses and confirmatory criteria

H1 (Family A, linguistic range). Relative to Neutral, Low has higher mean SUBTLEX `Lg10CD`
(more common words), lower MATTR and MTLD, higher content-lemma repetition, and lower
clause-pattern entropy, fewer subordinate clauses per sentence, shorter dependency distance
and shallower trees. Primary metrics: `subtlex_mean`, `mattr`, `clause_pattern_entropy`
(one per sub-family); the rest are supporting. Criterion: expected direction, cluster-bootstrap
95% CI excluding 0, BH q < .05 within the family, and |d_z| at or above the threshold
*frozen at pilot*.

H2 (Family B, guardrail). Low and Neutral are **equivalent** in `grammar_errors_per_100_words`
(LanguageTool, GRAMMAR category) and in mean CoLA acceptability: the 90% cluster-bootstrap CI
of the paired difference lies within ±bound, bound *frozen at pilot* (TOST).

H3 (Family C, perception). Blind listener models choose Low as the more limited speaker in
Low-vs-Neutral pairs: the cluster-bootstrap 95% CI lower bound of the share exceeds 0.50,
for each listener family and pooled, in Set A and again in Set B. Neutral-vs-Original
should sit at 0.50 (CI covers 0.50).

H0-control. Neutral and Original are equivalent on the primary range metrics (TOST, bounds
*frozen at pilot*), and |edit distance(Low) − edit distance(Neutral)| is within the frozen
tolerance for the large majority of sources.

Pragmatic guardrail (diagnostic, not a criterion): the blind politeness classifier assigns
the same Brown & Levinson label to the rewrite as to the original at similar rates in Low and
Neutral. This classifier agreed with human raters on only 34/48 items in the politeness
validation, so it cannot gate anything; the semantic gate's `commitment_preserved`,
`act_force_preserved` and `speaker_perspective_preserved` dimensions are the hard checks.

## 3. Units, sampling and stages

Unit of analysis: a **source bundle** = one speaker's ordered utterances in one game, taken in
speech order until the passage reaches the window minimum and never beyond the window maximum
(pilot starts at 150–250 words; a speaker whose utterances cannot fit is "insufficient" and
excluded). Every condition is assembled from exactly the same utterances, so every comparison
is matched. One bundle per speaker per game; a small cap per game spreads the sample across
games. The **game** is the independent unit for inference.

| stage | untreated games | bundles | purpose |
| --- | ---: | ---: | --- |
| Pipeline debug | 2–3 | few | run rewriter, gate, features, probe end to end |
| Pilot | ≥ 30 | ≈ 100 | fix prompts, bundle window, matching tolerance, bounds, thresholds |
| Confirmatory Set A | ≥ 75 | 200 | confirmatory validation with the frozen pipeline |
| Replication Set B | another ≥ 75 | 200 | identical frozen pipeline, independent games |

Pilot, Set A and Set B use disjoint games. The bundle counts above are targets from the working
plan; the pilot effect sizes and the between-game variance decide the final n through a power
analysis on the primary contrast (Section 11, D12). Speakers whose branch had any
`fallback_to_original` are excluded from the confirmatory treatment-effect set (D6).

## 4. Generation (per utterance)

1. Low is generated first; its normalized token edit distance to the original becomes the
   target for Neutral.
2. Neutral: five candidate paraphrases are requested in one call (so temperature 0 still yields
   distinct candidates). Every candidate passes the cheap deterministic checks; survivors are
   judged by the parser and the validator; among the passers the candidate whose distance to
   the original is closest to the Low target is chosen. All candidates, distances and verdicts
   are recorded.
3. High (if enabled) is generated like Low.
4. Each condition gets one attempt plus at most two **corrective retries** whose prompt quotes
   the gate's failure reasons and the failed text. After that the utterance is
   `treatment_unavailable`. A judge that returns no verdict (timeout, unparseable output) is
   re-asked once about the same text; if it still has no verdict the attempt is recorded as
   such, nothing is fed back to the generator, and an utterance whose attempts never received
   a real verdict ends as `infrastructure_unavailable`, reported separately from
   `treatment_unavailable`. A generator error (including the Neutral anchor call) likewise
   counts as an attempt but never becomes "feedback". The wrapper fails closed by default
   (raises). A live replay that must keep running opts in explicitly to returning the
   untreated text, and only together with a log file to which every attempt and every
   `fallback_to_original=True` is appended as it happens, because the engine itself records
   only the returned string.
5. Generation QC per condition: first-pass success, retry, treatment-unavailable,
   infrastructure-unavailable and fallback rates.

At replay time the Neutral rewriter obtains its target by generating an ungated Low rewrite
internally; the validation set uses the gated Low. Both are recorded (D4).

## 5. Semantic fidelity: the hard gate

Every candidate must pass, in order: deterministic structural check (predicate slots realized,
no added epistemic hedge, no polarity flip); treatment checks (not a copy of the original, no
meta text); the inverse predicate parser (structured predicate recovered blind, gated slots
match); the dimensional semantic validator (commitment, act force, evidence relation,
referenced statement, speaker perspective, nothing invented, nothing deleted). The proficiency
conditions never license indirectness, so the gate always runs with the DIRECT commitment rule.
A judge that returns no verdict (transport error, unparseable output) never yields a pass.

The parser is shown only the public history **before** the utterance. The game loader stores
the whole history on every source, including the entry that realizes the utterance (its
predicate and text) and later turns; every judge call cuts it at the source's own speech id
(`history_before`), and the source JSONL written by `bundles` is already cut. Without this the
"blind" recovery would have the answer in its input (D16).

For calibration the judges run **independently** on every item (`judge_all`): no
short-circuit after a cheap failure, and the copy-of-original check is reported but not
applied, so identity positive controls are scored and every judge's sensitivity is computed
on the full challenge set. The production gate keeps the short-circuit to save judge calls.

**Judge calibration before the gate is trusted**

*Human benchmark* (`data/judge_benchmark_human.md`, computed from the 48 politeness rewrites
with two blind raters): against the two-rater consensus (3 broken, 41 preserved) the
validator flagged 3/3 broken items and passed 38/41 preserved ones (specificity 0.93); the
structural check flagged 1/3 with no false alarms; the parser flagged 0/3, because all three
losses were in act force or evidence, which slot recovery cannot see. The negative class is
tiny, so the sensitivity intervals are wide; this benchmark anchors specificity on real
rewrites.

*Synthetic challenge set* (`data/challenge_set.jsonl`, 129 items from the 12 untreated
sources): 95 perturbations with a meaning change known by construction (speaker/target swap,
role flip, polarity flip, action-target change, reason deletion, reason substitution,
proposition deletion, certainty shift, directive-force shift) and 46 positive controls
(original vs original; politeness rewrites both raters judged preserved, excluding off-record
ones because the DIRECT rule would count licensed indirectness as failure). Judging it
requires model calls (`run-challenge`); its per-type sensitivity and the positives'
specificity set the minimum judge performance *frozen at pilot*.

## 6. Bundles and measures

**Family A – linguistic range** (`components/metrics/lexical.py`, `syntactic.py`), computed
per bundle passage:

- SUBTLEX-US `Lg10CD` mean/median over content words (NOUN, VERB, ADJ, ADV, PROPN); lookup
  surface → lemma → missing; coverage recorded. Player and statement ids excluded; the
  Werewolf domain list is reported separately and excluded by default.
- MATTR (window frozen at pilot, default 50; undefined below the window), MTLD (undefined when
  the TTR never falls, which cannot happen at bundle length).
- Repetition: 1 − types/tokens over content lemmas; share of the top-5 lemmas.
- Syntax (spaCy `en_core_web_sm`, version pinned at freeze): mean dependency distance, tree
  depth, subordinate clauses per sentence and per 100 words, noun-chunk length and modifiers,
  clause-pattern distinct ratio and entropy (primary syntactic cue), sentence count and length.
- Nuisance: word count, sentence count, edit distance to the original passage.

**Family B – grammar / acceptability guardrail** (`components/metrics/grammar.py`): player ids
→ fixed gender-neutral names and statement ids → "statement N" before the instruments, for
all conditions alike. LanguageTool via its HTTP API; the confirmatory outcome counts only
matches in LanguageTool's GRAMMAR category. Word-choice and casing rules (CONFUSED_WORDS,
COLLOCATIONS, SEMANTICS, CASING) form a separate `usage` bucket, and spelling, punctuation and
style are reported on their own, so a lexical difference between Low and Neutral cannot be
booked as a grammar error. CoLA acceptability probability per
sentence (default `textattack/roberta-base-CoLA`; checkpoint and revision frozen), bundle
mean/median/10th percentile.

**Family C – listener perception** (`components/judges/listener_probe.py`): each matched pair
is shown to three listener model families, in both presentation orders, hidden conditions;
the question is which speaker's English shows the more limited range of vocabulary and sentence
structure, judged on language only. A small blind human pairwise worksheet (two raters) is run
on a subset in the same format (D10). Aggregation: share choosing Low, Wilson and
cluster-bootstrap CIs, by listener and order; Bradley–Terry log-strength separation with
cluster-bootstrap CI (pseudo-count 0.5 so unanimous pairs stay finite).

## 7. Statistics

- All contrasts are paired within bundle. Uncertainty comes from resampling **games**:
  percentile cluster bootstrap (2000 draws) for CIs, cluster sign-flip tests (5000
  permutations) for p-values, Cohen's d_z for standardized effects.
- Benjamini–Hochberg (q = .05) within each metric family per contrast. The family is the
  twelve range metrics (Family A) or the two guardrail metrics (Family B); the nuisance
  diagnostics (word count, sentence count, edit distance) are reported with raw p-values and
  are not part of any family.
- Equivalence (guardrails, Neutral control) by TOST via the 90% cluster-bootstrap CI.
- Set B executes the identical frozen pipeline and is reported side by side with Set A.
- A mixed model `metric ~ condition + (1|game) + (1|bundle)` can be fitted in R as a
  sensitivity analysis; the confirmatory inference is the cluster bootstrap (D8).

## 8. Threshold freeze

`freeze` writes `data/frozen_config.json` with: bundle window, MATTR window, neutral matching
tolerance, equivalence bounds, range-effect thresholds, primary metrics, model handles,
instrument versions, statistical constants, whether High is included (`--include-high`), and
SHA-256 hashes of the rewriter prompt **templates** (the system prompt with the
predicate-definition slot left as a placeholder, so one hash per condition regardless of
predicate; the neutral candidate template includes the frozen candidate count), the probe
prompt, the semantic rubric, the parser prompt, and the **source files** of every module that
defines a measurement or a check (structural check with its hedge and surface lists, rewriter,
probe, the metrics modules, the gate). Every rewrite row records both its template hash and the
exact instance prompt hash. Inference code is not hashed; its constants are in the config.

The config is enforced, not just recorded: `generate`, `features`, `guardrails` and `probe`
accept `--frozen`, recompute the hashes, stop on any mismatch, and pin their parameters
(models, candidate count, retries, generation temperature, the set of conditions, MATTR
window, spaCy model and version, SUBTLEX file hash, LanguageTool server version, CoLA
checkpoint and commit, listener set) from the config. `analyze --frozen` re-verifies the
hashes and checks the data's own provenance (template hash, generator/judge handle,
temperature and condition on every rewrite; the feature, guardrail and probe side-cars)
against the config; each check is a criterion in the report. A criterion that cannot be
evaluated is reported as such and never counts as passed.

The pilot audit counts bundle capacity with the same greedy speech-order prefix rule that
builds the bundles (never exceeding the window maximum), not with speakers' total word
counts, so the window it recommends is one the bundler can actually fill. `suggest-bounds`
prints the Original↔Neutral pilot variation per metric; the bound actually frozen is chosen
explicitly and never below a stated smallest effect of interest (D9).

## 9. Actual-stimuli manipulation check

After the behavioral replays are generated, the same frozen pipeline (`features`,
`guardrails`, `probe`, `analyze --frozen`) runs on the actual Low / Neutral / Original stimuli.
Only treatment units meeting the frozen criteria enter the main behavioral analysis.

## 10. High extension rule

The pilot generates a small set of High rewrites under the same gate. High enters Set A only if
High-vs-Neutral separates in listener perception and on at least two Family A sub-families,
with grammar/acceptability inside the frozen guardrail. Set B then uses the same prompt and
thresholds.

## 11. Decisions and deviations from the working plan

- **D1 CEFR dropped.** The construct is a restricted linguistic-range cue, operationalized by
  measurable range features and blind perception, not by CEFR descriptors or a CEFR
  classifier. The paper outline's CEFR sections and UniversalCEFR validation are to be
  replaced accordingly.
- **D2 No manufactured errors.** The docx plan and the outline's draft A-level prompt
  disagreed ("basic mistakes may occur"); the rewriter forbids errors and grammar is a
  guardrail, matching the docx construct box.
- **D3 Statement ids normalized too.** The docx normalizes only P-ids; `s34` is equally
  non-English and is mapped to "statement 34" for the grammar instruments only.
- **D4 Neutral matching.** Candidates come from one call (temperature 0 would otherwise
  return five identical paraphrases). All candidates are gated before selection; the runtime
  wrapper anchors on an ungated internal Low, the validation set on the gated Low; both
  targets are recorded.
- **D5 Gate policy.** A structural failure, including an added hedge such as "I think", is a
  hard failure that triggers a corrective retry, not a queue for adjudication as in the
  politeness pipeline, because proficiency has no licensed indirectness. Human adjudication
  remains available for the confirmatory sets' `treatment_unavailable` cases.
- **D6 Fallback contaminates the branch.** A speaker-level cue with one untreated utterance is
  no longer the assigned condition. Any `fallback_to_original` excludes that speaker branch
  from the confirmatory treatment-effect set; the plan's "replace with a reserve source unit"
  applies at the bundle level.
- **D7 Undefined measures stay undefined.** MATTR below the window and MTLD without any TTR
  drop return null rather than a silently different statistic.
- **D8 Inference without lme4.** Cluster bootstrap and cluster sign-flip tests replace the
  mixed model as the confirmatory procedure; the mixed model is a sensitivity analysis.
- **D9 Bounds need a floor.** Equivalence bounds derived only from Original↔Neutral variation
  can be arbitrarily tight when an LLM paraphrase is very stable; each bound is set explicitly
  as max(pilot p90 |Δ|, smallest effect of interest) and recorded.
- **D10 Human raters added.** Family C in the docx is model-only. A blind pairwise human
  worksheet (two raters, "cannot tell" allowed) is added on a subset so the manipulation
  check does not rest on LLM perception alone and mirrors the politeness validation.
- **D11 Model-family separation.** Listener families must differ from the generator family,
  and the generator should not be the same family as the behavioral agents whose reactions
  are measured (Qwen/DeepSeek), or self-recognition of style becomes a confound. This is a
  decision still open in the outline.
- **D12 Sample size by power, not fiat.** The 200/200 bundle targets are retained as the
  planning figure; the pilot's d_z and game-level variance on the primary contrast determine
  the final n, and Set B may be scaled to budget as long as it stays an independent full run.
- **D13 Positive controls exclude off-record.** See Section 5.
- **D14 Forced choice for models, "cannot tell" for humans.** Model listeners are
  counterbalanced and forced to choose; human raters may abstain and abstentions are reported.
- **D15 Length is not equalized.** Low may be longer or shorter than Neutral; word count is a
  reported nuisance and a covariate in sensitivity analyses, not a matching criterion.
- **D16 Judges see only prior history.** `sources_from_game_dir` (shared `components/helpers.py`,
  owned by the politeness branch) puts the entire game history on every source, including the
  utterance's own `[sN|Pk] predicate: text` entry. The parser's blind recovery is only blind
  if that entry is removed, so this branch cuts the history at the source's speech id before
  any judge call and before writing sources. The politeness validation's reported parser
  results should be re-checked for the same exposure by its owner; this branch does not modify
  the shared loader.
- **D17 Abstentions are data.** A human "cannot tell" is a valid response with no choice. The
  listener tables report the share over decided responses, the share over all responses
  (abstention counted as not choosing Low), and the abstention rate. Model listeners are
  forced-choice and have no abstentions.
- **D18 Nothing passes by absence.** Every confirmatory criterion implied by the frozen config
  is listed; missing data, a missing p-value (fewer than two games) or a missing bound yields
  "not evaluated", and the overall verdict is CONFIRMED only when every criterion passed.

## 12. What remains to be run

| step | command | needs |
| --- | --- | --- |
| audit untreated games | `cli audit games_fresh/* --out data/pilot/audit.json` | game runs (untreated) |
| bundles + sources | `cli bundles games_fresh/* --sources-out data/pilot/sources.jsonl --out data/pilot/bundles.jsonl` | – |
| judge challenge | `cli run-challenge --items data/challenge_set.jsonl --out data/pilot/challenge_judged.jsonl` → `report-challenge` | judge API |
| generate rewrites | `cli generate --sources data/pilot/sources.jsonl --conditions low,neutral,high --out data/pilot/rewrites.jsonl` | generator + judge API |
| features | `cli features --bundles … --rewrites … --subtlex SUBTLEX-US.txt --out data/pilot/features.jsonl` | spaCy, SUBTLEX file |
| guardrails | `cli guardrails --assembled … --languagetool-url … --cola-model … --rewrites … --politeness-judge deepseek-chat --out data/pilot/guardrails.jsonl` | LanguageTool server, torch/transformers |
| listener probe | `cli probe --assembled … --listeners A,B,C --out data/pilot/listener.jsonl` | three listener APIs |
| human worksheet | `cli build-human-worksheet --assembled … --out ws.md --key key.json` | two raters |
| pilot analysis | `cli analyze --no-frozen --rewrites … --features … --guardrails … --listener … --politeness … --out data/pilot/report.md` | – |
| freeze | `cli suggest-bounds …` then `cli freeze --neutral-tolerance … --neutral-match-min-fraction … --range-threshold <each primary metric> --bound <guardrails + primary metrics> --listeners … --spacy-model-version … --languagetool-version … --cola-model … --cola-commit-hash … --subtlex …` (every threshold is required; nothing is left to be "skipped" later) | pilot decisions |
| Set A / Set B | same commands on disjoint games, `analyze --frozen` | – |
