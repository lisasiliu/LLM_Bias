# Data export: language flip_self, pe01-pe20

Same design as the politeness export (`experiments/politeness_experiment/data/export/README.md`): each baseline game `peNN` is replayed from each
day-start checkpoint for every living player as the self player. Levels: A = restricted range, B = paraphrase control at the original range,
C = expanded range; the self player's speech is rewritten in the level, or left natural (`control`). One replay per level and two per control.
Rewrites go through the shared gate (player numbers with the I/you exceptions, the action appeal, the reasons check) plus the language
treatment checks; a failed rewrite is regenerated once at temperature 0.7, then the original text is delivered. Level B also needs the three
range labels (vocabulary, repetition, syntax) from the English fidelity judge to be "same".

## Files
- `language_replays.csv`: one row per replay (930). Columns as in the politeness `replays.csv`, with `condition` = control, A, B or C. Reliability size
  measures: `mean_abs_point_change`, `share_scores_moved_2plus`, `self_trust_change_all_assessors`, `self_trust_change_village_assessors`
  (village assessors' scores are trust; a wolf's score is its role-guess confidence).
- `language_rewrites.csv`: one row per rewrite attempted (2,340): `level`, `predicate`, `status` (ok or treatment_unavailable), `original_delivered`,
  `delivered_attempt` (1 or 2), failure reasons of attempts 1 and 2, `original_text`, `delivered_text`.
- `language_seed_level_summary.csv`: per seed and level: rewrites, failed first attempt, rescued by retry, original delivered, replays, cost.
- `language_player_days.csv`: control and A/B/C values of the seven measures per (seed, day, self player), repetitions averaged.
- `../../results.md`: level minus control, and A minus B, C minus B, pooled over player-days with seed-clustered 95% intervals, with and
  without pe01.

## Caveats
- Replays that ran while the judge account was out of credit (pe03, pe05, pe07, pe13, pe15, pe18-pe20) were discarded and rerun. The rerun
  replays used the judge `gpt-5.6-luna` through OpenRouter (`openai/gpt-5.6-luna`); the others used the OpenAI account. Same model name, different route.
- The audit of the shared gate found the English fidelity judge passes far fewer A and C rewrites than B (see the conversation record); no
  human meaning labels exist for these rewrites yet (`../meaning_check_sheet.md` is waiting for labels).
- Repetitions of one (seed, day, player, condition) are not independent seeds.
