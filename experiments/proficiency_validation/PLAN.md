# Proficiency validation plan

This protocol evaluates controlled changes in English vocabulary and syntactic
range. Three untreated game runs provide the source corpus and descriptive baseline
reported in [VALIDATION.md](VALIDATION.md).

## Construct and conditions

The manipulation changes vocabulary familiarity, lexical diversity and syntactic
range. Low uses simpler language while preserving grammaticality and comprehensibility.
High expands linguistic range while preserving evidence, certainty, persuasive force
and formality.

| Condition | Role | Intended realization |
| --- | --- | --- |
| Original | Untreated baseline | The original Stage-2 game speech |
| Neutral | Paraphrase control | New wording with the original range and pragmatic force |
| Low | Primary treatment | More familiar and repetitive vocabulary, simpler syntactic choices |
| High | Additional primary treatment when included | Less restricted vocabulary and syntax under the same fidelity contract |

Low versus Neutral is the main comparison. Neutral versus Original checks the
paraphrase control. High versus Neutral has its own full criteria whenever High
is included in the frozen design. Low versus Original is descriptive.

Matching edit distance reduces differences in the amount of surface change.
Fidelity checks and equivalence tests assess other properties of the rewrite.
Word count, sentence count and edit distance are reported as diagnostics; matching
alone provides limited control over differences between conditions.

## Sources and sampling

Sources are untreated public speech from game logs. All public speech-act types
are eligible, including role claims and divine-result claims.

The source corpus contains all 280 utterances from run01, run02 and run03.
The manifests reuse an internal game ID, so directory names identify source games.
Exact repeated wording is reported. Actual hidden roles are not supplied to
rewriters or judges.

An analysis passage contains one speaker's ordered utterances from one game.
The current descriptive window is 150 to 250 words. Whole utterances are added
until the minimum is reached without exceeding the maximum. No passage is created
from an insufficient prefix. Conditions use identical source IDs.

The game is the independent sampling unit, with the 24 passages nested within
three games. Passage windows, per-game caps and eligibility rules are specified
before confirmatory rewriting.

Pilot, Set A and Set B must use disjoint games. The three source runs are for
development and source auditing. Automatic judge diagnostics can be run on the
pilot, but confirmatory evidence must come from fresh games after the protocol is frozen.

## Generation and fidelity

The validation workflow uses GatedRewriter or the generate command to apply
the English fidelity gate to each candidate rewrite.

Low and High are generated and gated first. Neutral candidates are checked under
the same gate. With both treatments included, the target edit distance is the
midpoint between their accepted distances to Original. With Low alone, its accepted
distance is the target. Each treatment's matching quality is evaluated separately.
Offline and runtime Neutral use this same gated-anchor policy. A failed anchor
makes Neutral unavailable and remains visible in the attempt log.

Every candidate must preserve the predicate, target, reasons, stance, references,
speaker perspective, politeness and register. The English fidelity rubric evaluates
these properties as constraints on the vocabulary and syntax manipulation.

The inverse parser sees only prior public history. If the utterance's location
cannot be established, generation stops to prevent future-context leakage.
An absent or malformed judge answer never becomes a pass. Corrective retries use
actual fidelity failures as feedback; infrastructure errors are recorded separately.
Retry budgets and completion limits are frozen after development.

Sentence-boundary changes are valid syntactic edits. Copy detection ignores
whitespace and capitalization while preserving punctuation differences, allowing
clause splitting to count as a syntactic change.

## Automated evidence and interpretation

Validation uses automated linguistic measurements and model judgments. Linguistic
features measure the intended range changes, and grammar instruments assess
correctness. The inverse parser and English model judge assess fidelity. Blind
model listeners test whether the range difference affects model perception.

Synthetic perturbations can diagnose the fidelity checks during development.
Identity controls test preservation, while targeted changes test detection of
altered slots, certainty or reasons. Their diagnostic value depends on the
constructed labels being correct. Challenge-set sensitivity describes detection
on these constructed cases. Accuracy on naturally generated rewrites remains
an external-validity limitation of this diagnostic.

Claims concern measured linguistic range and model perception. Generalization to
human judgments of naturalness, human perception or overall English ability requires
separate evidence. Agreement across listener families remains subject to correlated
model errors.

Optional annotation utilities support supplementary fidelity, naturalness and
perception analyses. These analyses are reported separately from the automated
validation criteria.

## Measures and criteria

| Evidence | Frozen criterion |
| --- | --- |
| Lexical familiarity | Low raises SUBTLEX mean relative to Neutral; High reverses the direction |
| Lexical diversity | Low reduces MATTR relative to Neutral; High reverses the direction |
| Syntax | Low reduces subordinate clauses per sentence; High reverses the direction |
| Grammar and acceptability | Each treatment is equivalent to Neutral under justified raw-unit margins, with acceptable absolute observed quality |
| Paraphrase control | Neutral is equivalent to Original on primary range measures and model perception |
| Automated fidelity | Accepted rewrites pass structural, inverse-parser and English semantic checks under the frozen generation policy |
| Perception | Every declared model listener separates each treatment in the predicted direction |

SUBTLEX mean, MATTR and subordinate clauses per sentence are the default primary
metrics. Alternatives must be chosen before confirmation and cover familiarity,
diversity and syntax. Custom clause-pattern entropy and distinct-ratio measures
remain exploratory. Lexical definitions pin the SUBTLEX column, file hash,
domain-word treatment, MATTR window and spaCy versions. Insufficient coverage
prevents confirmation.

LanguageTool grammar counts use the versioned category and rule mapping, including
the explicit article-agreement override. Usage, spelling, punctuation and style
are recorded separately. CoLA provides model-based acceptability estimates. The
acceptable-label index, checkpoint and sentence segmentation are recorded and
verified. Overlong sentences produce an explicit failure. Interpretation of these
instruments is limited to their operational measures of grammar and acceptability.

At least three distinct listener families are required for confirmation. They
must differ from the generator family. The generator must also differ from
the declared behavioral-agent families. Qwen3.5 is the planned first test family;
the exact checkpoint and its role will be specified before testing. Development
tests can use Qwen alone; confirmation requires the full set of declared families.

Model probes use both presentation orders. Both outcomes are recorded even when
one request fails; directional inference uses complete counterbalanced comparisons.
The effect must hold for each listener and in both orders. Optional human
observations are analyzed separately from model listeners.

## Statistics and research decisions

Pair comparisons within passages and resample games for uncertainty. The frozen
protocol records bootstrap draws, sign-flip repetitions and the random seed.
Sign-flip inference assumes game-level symmetry under the null. These approximate
sampling procedures depend on the number of independent games for reliable uncertainty
estimation.

Range confirmation requires the expected direction, a 95% interval excluding zero,
BH-adjusted significance and the declared minimum paired standardized effect.
The range-test family pools the planned Low and High contrasts across reported
range metrics. Nuisance metrics are excluded. All designated primary metrics must
pass; exploratory findings do not substitute for a missing primary result.

Grammar and Neutral-control equivalence require the entire 90% paired-difference
interval to lie within justified raw-unit bounds. A degenerate bootstrap distribution
does not establish equivalence. Neutral-versus-Original perception requires its
90% interval to lie entirely inside the frozen band around 0.5 in both orders.
An interval merely covering 0.5 is insufficient.

Listener intervals resample games. If observed game data yield a zero-width
proportion interval, a conservative bounded-mean Hoeffding interval is used.
With fewer than two games the interval is unavailable. Wilson response-level
intervals and Bradley–Terry results are descriptive supplements.

Equivalence bounds represent the largest scientifically negligible difference.
The scientific rationale determines each bound before confirmation. Pilot variation
informs prospective power calculations. Absolute-quality cutoffs have separate
justifications.

Choose sample sizes using the full joint decision rule, including equivalence,
both presentation orders, High when included, unavailable rewrites and game clustering.
Archive the power calculation and its assumptions with the study rationale. The
software checks the declared sample minima. Observed quality pass rates and absolute
means serve as sample-level screening rules, with population generalization requiring
additional uncertainty assessment. Set B independently repeats the full protocol.

## Freezing and provenance

Freeze the automated protocol after the pilot. Required decisions
include conditions, primary metrics, equivalence bounds and rationales, effect
thresholds, sample minima, coverage limits, edit-distance tolerances,
measurement versions, model identities, completion budgets and statistical settings.

The freeze also stores complete preselected bundle records, source-file hashes and
disjoint splits. It hashes prompts, the English rubric, measurement code, inference
code and protocol checks. Formal commands pin settings from this artifact and stop
when code or declared models differ. Every empirical threshold is specified in the
protocol. Schema 3 records the automated validation scope. Protocol freezing is
scheduled after the pilot.

Model revisions and serving metadata must be recorded for Qwen3.5 and any later
DeepSeek checkpoint. Archive the actual model artifacts and execution environment
alongside model declarations to document the weights used by each server.

## Behavioral experiments and missing treatments

For a fixed-stimulus experiment, all eligibility decisions can be completed before
behavioral outcomes are collected. Retain construction failures and document the
population to which the resulting stimulus set applies.

For an interactive game, future speech and rewrite failures occur after assignment.
The primary behavioral analysis must retain every assigned branch, including
fallbacks. It estimates the effect of the declared rewriting policy. Excluding
branches because a later rewrite or manipulation check failed would condition
on a treatment-dependent event.

GatedRewriter raises on unavailable treatment by default. Continuing with Original
requires an explicit fallback policy and a persistent attempt log with game, branch
and source identifiers. Report fallback frequency and actual delivered exposure.
Use separate logs for separate branches and include stopped games in the
assigned-sample accounting.

Validate actual gameplay exposure as well as long offline passages. Interactive
trajectories form a separate cohort from the preselected Set A and Set B sources.
Their measurement and outcome analysis follow a separate prespecified protocol.
Linguistic validation establishes the operational manipulation; behavioral causal
effects and broader proficiency claims require their own evidence.
