# Data export: politeness pe01-pe20 (and language pe01-pe17)

Everything here is derived from the replay outputs in `games/flip_experiments/` (not in git). All rates are shares in [0, 1].
Language files are in `experiments/proficiency_flip/data/export/`.

## Design in one paragraph
Each baseline game `peNN` (7 players, DeepSeek) was replayed from each day-start checkpoint (day 1, 2, 3 where it exists) for every
living player as the "self player". In a replay the self player's eligible speech (nine speech acts) is rewritten into the setting
(`all_bald`, `all_positive`, `trust_positive`, `trust_bald`) or left natural (`control`); the replay then plays discussion, vote, night and a
reliability update. Politeness: 2 repetitions per (seed, day, player, condition). Language: 1 replay per level A/B/C, 2 for control. Model
responses are replayed from the baseline's log until the self player's first eligible utterance, then fresh; so the control's difference from
the baseline is the noise floor. Trust settings: the speaker ranks living players by its own reliability scores (wolves by their own town
consensus guess); top half (middle player included) is trusted.

## Files
- `replays.csv`: one row per replay (1,860). `condition` = control or a setting. `skill_change_rate`, `vote_change_rate`,
  `reliability_change_rate` = share of night-action targets / vote targets / reliability scores that differ from the baseline phase
  (`mean_reliability_delta` = mean signed change). `votes_on_self_*` and `executed_*` compare with the baseline. `treated_utterances`
  / `original_delivered` = rewrites attempted / delivered as the original wording after the gate. `cost_usd_ledger` is the ledger's
  DeepSeek-priced estimate.
- `treated_utterances.csv`: one row per treated utterance (5,109 incl. repetitions). `rank_group` is the group used for the setting
  (trusted/distrusted/unrated/unaddressed); `rank_group_by_reliability` and `rank_group_by_consensus` are logged for every utterance
  (consensus only for wolf speakers). `delivered` = rewrite or original; `delivered_attempt` = 1 or 2 (the second attempt is a
  temperature-0.7 regeneration with the same prompt); `failure_attempt1/2` = gate rules that failed (`lost_player_slot`,
  `evidence_dropped`, `added_epistemic_hedge`, `lost_action_slot`, ...). Texts: `original_text`, `attempt1_text`, `delivered_text`.
- `seed_setting_summary.csv`: per seed and setting: treated, failed first attempt, rescued by retry, original delivered, replays and cost
  of the seed.
- `e3_player_days.csv`: the E3 input, one row per (seed, day, self player) with control and each setting's three rates (repetitions
  averaged). E3 = setting minus control, pooled over player-days, 95% interval from a seed-clustered bootstrap (2000 draws),
  also reported without pe01. Results: `../../results.md`.

## Caveats
- Repetitions of one (seed, day, player, condition) are not independent seeds; use seed-level clustering.
- A treated utterance can appear in several settings and repetitions (the same original text is rewritten per setting).
- Costs are ledger estimates (DeepSeek flash prices, Luna checks priced the same way); the DeepSeek account balance for the second key is
  in CNY (7.2 CNY per USD assumed).
- The bald rewrite drops reasons more often (gate fallback 7.3% for `all_bald`), so the bald settings are slightly less bald than assigned.
