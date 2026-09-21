# Politeness Validation

## Goal

We rewrite Werewolf game utterances under controlled politeness conditions, so
that a downstream experiment can vary *how* something is said while holding
*what* is said constant. Before that is usable, two things must be established:

1. **Manipulation** — do the rewrites actually realize the requested
   Brown & Levinson strategy?
2. **Meaning preservation** — do they still carry the same game-relevant
   content?

Four strategies, treated as categories rather than points on a scale:

- bald on-record
- positive politeness
- negative politeness
- off-record

Everything below is reproducible from `data/` with `code/validation.py`.

## Manual in-domain validation

### The dataset

Three untreated natural games supplied the source material. A game qualifies
only if no rewriting was applied during play — its manifest records no rewriter
and its natural-language log carries no treatment annotation — so every source
utterance is the model's own unedited speech, paired with the exact structured
predicate that produced it.

From those games we drew three source utterances from each of the four
predicate types that occur naturally in play (`estimate`, `disagree`,
`question`, `recommend_action`), sampled deterministically and balanced across
types. Each source was rewritten under all four politeness conditions, giving
**48 matched rewrites**: the four versions of a source differ only in politeness
strategy.

Candidate sources were screened against every utterance used while developing
the rewriting prompt and the automatic judges. None overlapped — natural game
transcripts share no text with the constructed material used in development — so
this set is genuinely held out.

A fifth predicate, `agree`, is supported by the grammar but did not occur once
in 280 naturally generated speech acts across the three games, while `disagree`
occurred 45 times. We did not synthesize `agree` examples to fill the table;
the validation covers the predicates that actually occur.

### Blind annotation

All 48 rewrites were annotated by hand. The annotator saw the source predicate,
the untreated original, the rewrite, and — where the predicate responds to an
earlier statement — that statement's text, so no one had to guess which speech
act was being manipulated. The requested strategy and every automatic verdict
were withheld, and the items were shuffled so their order carried no signal.

For each item the annotator recorded the dominant strategy, their confidence,
whether game-relevant meaning was preserved, and supporting detail on epistemic
commitment, the force of the act, the evidence relation and the referenced
statement.

### Manipulation result

| strategy | n | recognized as requested |
| --- | ---: | ---: |
| bald on-record | 12 | 12 |
| positive politeness | 12 | 12 |
| negative politeness | 12 | 12 |
| off-record | 12 | 12 |
| **total** | **48** | **48** |

Confidence: **36 high, 12 medium, 0 low**. No item was labelled
`ambiguous_or_mixed` or `failed_manipulation`. The confusion matrix is diagonal.

**Limitation.** There was a single annotator, who is an author of the study.
The annotation was blind to condition, but an author may recognise the design's
structure, and a perfect result should be read with that in mind.

A second, independent annotation has been **prepared but not yet collected**:
`data/independent_annotation_worksheet.md` holds a 20-item subsample of the 48,
balanced five per strategy and five per predicate so that all sixteen strategy x
predicate cells are represented, shuffled under a different seed and blind to
condition and to the first annotator's labels. It happens to include three of
the five items whose meaning the first annotator did not mark `yes`, which is a
consequence of the balanced draw rather than a selection of contested cases.
Until it is returned, the manipulation result rests on one rater.

## Automatic strategy classification

An automatic classifier assigns a strategy to each rewrite without being told
the requested condition.

| requested strategy | manual | automatic |
| --- | ---: | ---: |
| bald on-record | 12/12 | 12/12 |
| positive politeness | 12/12 | 7/12 |
| negative politeness | 12/12 | 10/12 |
| off-record | 12/12 | 5/12 |

Agreement with the manual annotation is **34/48 (70.8%)**, macro-F1 0.703.

| class | precision | recall | F1 | n |
| --- | ---: | ---: | ---: | ---: |
| bald on-record | 0.52 | 1.00 | 0.69 | 12 |
| negative politeness | 0.77 | 0.83 | 0.80 | 12 |
| off-record | 1.00 | 0.42 | 0.59 | 12 |
| positive politeness | 1.00 | 0.58 | 0.74 | 12 |

**The manipulation succeeds; the classifier is what falls short.** Its error
pattern is asymmetric. Positive politeness and off-record have **no
false-positive predictions in this sample** — every item the classifier placed
in either category was placed there by the annotator too, so precision is 1.00 —
but their recall is 0.58 and 0.42.

The dominant error is **over-prediction of bald on-record**: the classifier
assigned it 23 times against the annotator's 12, so 11 of its 14 errors are
items pulled into that category. The remaining three misses go to negative
politeness, which therefore does carry false positives (10 correct out of 13
predicted). The classifier is insensitive to redress and indirectness rather
than wrong about which utterances have them.

There is no item where the requested manipulation failed to register with the
human reader, so the gap is a property of the classifier, not of the rewriter.
**The automatic classifier is therefore used as a diagnostic only.** It is
recorded for every rewrite and is never used to accept or reject one, and manual
annotation is the manipulation evidence.

## Semantic preservation

| strategy | yes | no | unsure |
| --- | ---: | ---: | ---: |
| bald on-record | 12 | 0 | 0 |
| positive politeness | 12 | 0 | 0 |
| negative politeness | 11 | 1 | 0 |
| off-record | 8 | 2 | 2 |
| **total** | **43** | **3** | **2** |

| predicate | yes | no | unsure |
| --- | ---: | ---: | ---: |
| disagree | 12 | 0 | 0 |
| estimate | 10 | 0 | 2 |
| question | 12 | 0 | 0 |
| recommend_action | 9 | 3 | 0 |

`unsure` is reported as its own category throughout and is never folded into
either pass or fail.

The five items that were not straightforwardly preserved fall into two kinds.

**Indirectness dissolving the act.** Four involve off-record rewriting. In two
`recommend_action` cases the recommendation itself disappeared: a direct
instruction to vote a player out became an observation about that player, so the
suspicion survived but the requested action did not. In two `estimate` cases the
supporting evidence survived while the role assessment became implicit rather
than asserted — the annotator marked these `unsure` rather than `no`, since
off-record realization is supposed to leave something to inference, and the
judgement is where the line falls.

**Perspective distortion.** One negative-politeness rewrite of a
`recommend_action` preserved the action and the evidence but shifted who was
speaking: a player announcing what they would do themselves became that same
player being asked to do it. The content survived; the speaker did not.

Both kinds concentrate in `recommend_action`, the one predicate carrying a
directive that indirect phrasing can dissolve.

## Automatic semantic quality control

Stimulus construction runs each rewrite through a four-stage check.

1. **Structural checks.** Deterministic tests driven by the structured
   predicate: every player, role, action and referenced statement the original
   realized must still be recoverable, and polarity must not flip.
2. **Inverse predicate parser.** A model is shown only the rewritten utterance
   and the prior public record — never the source predicate — and must
   reconstruct the predicate from scratch. Preservation then becomes a
   comparison of two structured objects rather than an opinion. Exact recovery
   is strong evidence when it succeeds, but a legitimate indirect realization
   can make reconstruction impossible, so a mismatch is a discrepancy signal,
   not a rejection.
3. **Dimensional semantic validator.** A model judges the properties the
   predicate cannot encode, one boolean per property, asking only the
   dimensions that apply to the predicate at hand: epistemic commitment; the
   force of the act; the evidence/reason relation; whether a referenced
   statement is still the same underlying statement; whether game-relevant
   content was invented; whether any was deleted; and **whether the speaker and
   addressee are preserved** — who is speaking and who is being spoken to may
   not move, though how politely the addressee is approached may, since that is
   the treatment. The validator is never asked for an overall
   "is this equivalent" verdict.
4. **Human adjudication.** A rewrite is accepted automatically only when the
   parser reconstructed the predicate *and* the semantic validator passed it.
   Anything else is queued for a person. Nothing is rejected automatically, and
   a parser mismatch is never silently converted into either a pass or a
   rejection.

Against the manual `meaning` judgement on this set (`unsure` held out):

| check | agreement | accepted but manually not preserved | flagged but manually preserved |
| --- | ---: | ---: | ---: |
| inverse predicate parser | 43/46 (93.5%) | 3 | 0 |
| dimensional semantic validator | 42/46 (91.3%) | 1 | 3 |
| combined policy | 42/46 (91.3%) | 1 | 3 |

**5 of 48 items (10.4%)** were queued for adjudication — three bald on-record,
two off-record. The three flagged-but-preserved items are boundary cases rather
than errors: in each, the annotator's own note identifies the same defect the
validator identified (a deleted justification, an added categorical assertion),
and the two judgements differ only on whether that defect breaks core meaning.
Under the policy those items are reviewed, not discarded.

## External construct validation

Brown & Levinson supplies the theoretical definitions of the four strategies.
The external construct check uses **Zhao & Hawkins (2025)**, which annotates
Brown & Levinson strategy families directly in free production, independently of
this project and in an unrelated domain (feedback on a presentation).

**What is compared, and the unit mismatch.** Their annotation is span-level,
multi-label and at sub-strategy granularity — a single marker such as an
understatement or a counterfactual modal receives its own strategy label. Ours
is one dominant strategy for a whole utterance. The only directly justified
mapping is therefore at the **superordinate** level, where their four families
are our four strategies, and only for utterances where every annotated span
shares one family. Of 1,232 annotated items, 668 satisfy that condition; the
remaining 564 span several families and are excluded rather than relabelled,
since co-occurring markers are ordinary in Brown & Levinson and are not evidence
of a mixed act. Statistics are computed on a held-out split of 135 items, with a
small calibration split reserved and never inspected.

| result | value |
| --- | ---: |
| four-way agreement with the published superordinate label | 55.6% (macro-F1 0.506) |
| coverage into the positive/negative pair | 44/90 (48.9%) |
| conditional discrimination within those 44 | 95.5% (42/44) |

**The four-way figure is not a validator accuracy.** It is dominated by the unit
mismatch: 27 of 45 published negative-politeness items are read here as bald
on-record, because a single softening marker is enough to earn their label while
our judgement asks which strategy the utterance mainly realizes.

**The positive-versus-negative result is conditional, and must be reported as a
pair of numbers.** The held-out set contains 90 items whose published label is
positive or negative politeness — 45 each. The classifier placed only **44 of
those 90** in the positive/negative pair at all; the other 46 it mapped outside
the pair, mostly to bald on-record (34), and the rest to off-record (8) or
`ambiguous_or_mixed` (4). Within the 44 it did place in the pair, it agreed with
the published family on **42, or 95.5%**.

So: when the classifier itself places an utterance in the positive/negative
pair, it distinguishes the two published families well; but it often maps such
utterances outside that pair, so this is **conditional construct evidence rather
than an end-to-end accuracy measure**. Quoting 95.5% as positive-versus-negative
classifier accuracy would misstate it, because it conditions on a selection the
classifier itself makes.

The classifier prompt is also written for this game domain, so applying it to
presentation feedback tests portability along with the scheme. That makes the
four-way figure hard to interpret — it confounds the label scheme, the
annotation unit and the prompt's domain — rather than making it a bound on
anything.

## Final decisions

- **The rewriter is frozen.** All four manipulations were recognized in blind
  manual annotation, 48/48. The generation prompt is fixed for this study and is
  not tuned further.
- **All four conditions are usable.** No condition failed to register with the
  annotator.
- **The automatic strategy classifier is diagnostic only.** It is recorded for
  every rewrite, never used as an acceptance gate, and not tuned further. Manual
  annotation is the manipulation evidence.
- **Semantic quality control is used to build the stimulus bank**, with human
  adjudication on any item the parser and the validator do not both clear. The
  expected review load is about one item in ten.
- **Known limitations**: a single annotator who is an author; the same model
  family generated the rewrites and produced the automatic judgements, so the
  automatic figures are not independent evidence; off-record is the condition
  most at risk of dissolving a directive, and `recommend_action` the predicate
  most exposed to it.
- **Next step**: the listener/replay experiment, in which matched rewrites are
  substituted into identical game states to measure whether politeness alters
  what a listening agent does.
