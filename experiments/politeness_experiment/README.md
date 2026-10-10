# In-game politeness policy experiment

Written before any results were seen.

## Question

In untreated 7-player games, village-side agents already lean toward positive politeness to
players they trust and bald speech to players they distrust, but weakly (p = 0.057 in the
natural study). This experiment asks whether the pattern matters: when speakers are forced to
follow a trust-based politeness rule, do listeners trust them more, follow their vote
suggestions more, and does the village do better?

Background. Brown & Levinson (1987): the threat of an utterance to the listener's face grows
with social distance, relative power and imposition, and a bigger threat calls for more redress
(ordered here as bald, then positive, then negative, then off-record). Brown & Gilman (1989) on
Shakespeare's tragedies found power and imposition behave as predicted but distance read as
liking runs the other way: more liking, more redress; withdrawn liking, less. A thesis
replicating the pattern on four comedies found the redress toward liked hearers was mostly
positive. We read Brown & Gilman through that thesis, so quotations should be checked against
the original paper.

## Setup

Seven players (2 werewolves, seer, guard, 3 villagers), all `deepseek-chat` (DeepSeek's
non-thinking Flash), temperature 0, max 300 tokens, default system prompt. The rewriter and the
two LLM checks use the same model. Twenty new game seeds, `pe01` to `pe20`.

For each seed:

1. Play the `natural` baseline game with checkpoints saved.
2. Replay from the checkpoint taken after night 0 and before any day-1 speech (so before the
   sheriff election). Night 0 is shared by all arms.
3. Play to the end once per arm: `all_bald`, `trust_positive`, `trust_bald`.

On the first three seeds only, also replay `natural` (no rewriter) from the same checkpoint, to
measure how far a temperature-0 replay drifts from the baseline on its own. An optional fourth
arm, `bl_written`, runs on all seeds only if the budget allows.

## Arms

| Arm | Trusted addressee | Distrusted addressee | Middle addressee |
| --- | --- | --- | --- |
| `natural` (baseline game) | no rewrite | no rewrite | no rewrite |
| `all_bald` | bald | bald | bald |
| `trust_positive` | positive | bald | bald |
| `trust_bald` | bald | positive | bald |
| `bl_written` (optional) | positive | negative | bald |

No arm uses off-record.

## Which utterances are rewritten

In every non-natural arm, an utterance is "touched" if all of these hold:

- the speaker is village-side (villager, seer or guard); werewolf speech is always natural;
- the speech act is `question`, `disagree` or `recommend_action` with a single addressee, using
  the same rule as the natural study: a `question` goes to the player asked, a `disagree` to the
  author of the opposed statement, and a `recommend_action` to its actor, only if the actor is
  not the speaker.

Everything else stays natural. Every touched utterance goes through the rewriter in every
non-natural arm, bald ones included, so the rewriting pipeline is the same across arms.

## Trusted and distrusted

The rewriter does not receive reliability scores, so the call site passes the speaker's current
ratings. These are the ratings from the most recent rating round (made at the start of each day,
or before the campaign speech during the sheriff election). The speaker's ratings of all living
other players are ranked, with ties broken by a deterministic shuffle seeded by
(game seed, day, phase, speaker). The top half is trusted and the bottom half distrusted; with an
odd number the single middle player is "middle", and so is any addressee the speaker has not
rated. Middle and unrated addressees get bald in every arm. The score used is logged.

## Checks and fallback

Every rewrite must pass the admission gate: the deterministic structural check, the inverse
parser (it must recover the original predicate from the rewrite alone) and the dimensional
semantic validator, all three agreeing. If the assigned rewrite fails, the bald rewrite is used;
if that also fails, the original natural text is used. Nothing is dropped silently. Every
fallback is logged with its arm, and fallback utterances stay in the analysis under their
assigned arm.

## Outcomes

Per touched utterance we log seed, arm, day, phase, speaker and role, addressee and role,
speech act, the speaker's score for the addressee, rank group, assigned strategy, delivered text
source (rewrite, bald fallback or original), original and final text, and the check results.

**Manipulation check.** After the run the blind politeness judge (`gpt-5.6-luna`, three passes,
majority label) labels every touched utterance. We report, per arm and rank group, how often its
label matches the assigned strategy.

**Primary outcome.** For each touched utterance from speaker S to a village-side addressee A:
A's rating of S at the next rating round, minus A's rating of S in force when S spoke.
Werewolf addressees are excluded (their score is not trust) and reported separately. We average
per seed and arm, then compare arms paired by seed (20 pairs): `trust_positive` vs `all_bald`, `trust_bald`
vs `all_bald`, and `trust_positive` vs `trust_bald`, with a paired permutation test (and Wilcoxon
signed-rank as a check). The outcome is also split by rank group.

**Secondary outcomes** (descriptive; 20 seeds are too few to test them): for `recommend_action`,
whether the addressee then voted as recommended; the share of day executions that hit a
werewolf; the village win rate per arm. We also report how many positive rewrites `trust_positive` and
`trust_bald` each delivered, since the counts can differ despite a half and half split.

## Cost and budget

Token use and estimated cost are logged per game and by category (players, rewriter, checks,
judge). DeepSeek charges double in peak hours (01:00 to 04:00 and 06:00 to 10:00 UTC, Monday to
Friday); estimates apply the rate in force when each call was made, and the share of calls made
in peak hours is reported. The two pilot seeds run first; the measured cost per seed and a
projection for 20 seeds are in `cost.md`. The run continues unattended only if the projection
for the three main arms is under 40 USD.

## Files

| path | contents |
| --- | --- |
| `code/play.py` | plays one game (baseline or one arm) with a usage ledger |
| `code/run.py` | runs seeds in order, resumably, and logs cost |
| `code/analysis.py` | manipulation check and outcomes |
| `data/` | per-game logs, touched utterances and derived tables |
| `results.md` | results (written after the run) |
| `cost.md` | measured cost and projection |

Game directories are available upon request.
