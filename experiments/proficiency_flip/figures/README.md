# What the figures show

Three pilot games (seeds 201–203, 9 day checkpoints). A is the restricted range, B the
paraphrase at the original range, C the expanded range. Every replay is compared with the
identity replay of the same checkpoint; trust is the score the village players alive after
the night give the speaker. With three games the smallest attainable p is 0.25; the
intervals are bootstraps over games.

**change_rates** — Against the identity replay, a rewrite changes 25–31% of the votes, 8–16%
of the skill targets and 16–21% of the reliability scores, in both scopes and at every level.
The hatched bars are the identity replay against the recorded game: 7% of the votes and 11%
of the skill targets.

**scope_contrasts** — A − B and C − B differences in those change rates lie within ±6 points
and every interval but one crosses zero.

**speaker_outcomes** — Trust in a speaker rewritten to A is lower than for B in all three games
(A − B −0.12, interval −0.19 to −0.08); C − B is −0.03. B is above the untouched speech
(+0.13, interval −0.08 to +0.39). The A − B difference comes from village speakers (−0.16);
for wolf speakers it is −0.04. Votes against the speaker move the same way (A − B +0.45) with
an interval that crosses zero.

**self_response** — When everyone else is rewritten, the untreated player picks a different
vote target in 26–32% of checkpoints at every level; A − B is −0.06. A village player's trust
toward the others rises by 0.18 under A (interval 0.09 to 0.23), 0.06 under B and 0.02 under C.

**manipulation_check** — Delivered discussion speech against its source, fallbacks included.
Mean word frequency (SUBTLEX Lg10CD) moves from 2.93 to 3.31 under A and from 2.87 to 2.49
under C, and stays at 2.88 under B; A shortens sentences (23 words per sentence against 32);
utterance length stays at 33–35 words.

The numbers behind each figure are in `figure_stats.json` and `../data/flip_summary.json`.
