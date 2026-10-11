# Politeness in natural play

Do LURK agents use different politeness strategies depending on their role and
the phase of the game, and on how much the speaker trusts the player they are
talking to? "Natural" means untreated speech: no rewriter, nothing manipulated.

## Corpus

Thirty 7-player games (2 werewolves, seer, guard, 3 villagers). Every player is
`deepseek-chat` (DeepSeek's non-reasoning alias of `deepseek-flash`) at
temperature 0 with the default strategy prompt, and speech is free natural
language. Werewolves won 18 games and villagers 12. The game directories are
available upon request.

We analyse the 1,429 utterances that realize one of five speech acts a politeness
strategy can attach to (`estimate`, `disagree`, `question`, `recommend_action`,
`agree`):

| Role | Utterances | | Speech act | Utterances |
| --- | ---: | --- | --- | ---: |
| Villager | 552 | | `recommend_action` | 543 |
| Werewolf | 473 | | `estimate` | 432 |
| Guard | 242 | | `disagree` | 310 |
| Seer | 162 | | `question` | 143 |
| | | | `agree` | 1 |

## Method

**Strategy labels.** Each utterance is labelled bald on-record, positive
politeness, negative politeness or off-record by the LLM politeness judge
(`gpt-5.6-luna`, three passes, majority label; validated in `experiments/politeness_validation/`). The judge sees the utterance,
speaker and phase only. In 1,218 of 1,429 utterances all three passes agree; 3
utterances have no majority and appear in no strategy column below.

**Role and phase.** Roles come from the game record. Phases are the sheriff
election, day 1, day 2, day 3 or later, and last words. Only sheriff candidates
speak in the election.

**Reliability.** After each round every agent rates every other living player on a
1-10 scale, and the game stores the result as that agent's reliability score for
the player. For a village-side agent, high means "I believe this player is on my
side" (a player it suspects of being a wolf is scored 11 minus its confidence). A
werewolf agent's score is its confidence in a guess about the other player's
identity, so it is not trust: it is always exactly 10 for a teammate (all 128
wolf-to-teammate ratings) and ranges from 3 to 9 for village players, highest when
the wolf thinks the player is the seer. We therefore analyse village-side and
werewolf speakers separately, and for werewolves also compare speech to teammates
with speech to everyone else. The rating is made at the start of each day, before
that day's discussion, so we use the speaker's rating of the listener in force when
the utterance is made. The listener is the player the utterance is addressed to:
the player asked in a `question`, the author of the statement a `disagree`
opposes, or the actor in a `recommend_action` aimed at someone else. An `estimate`
has no addressee. This gives 881 addressed, rated utterances (537 from village-side
speakers, 344 from werewolves).

**Role comparison.** The role tables are simply each role's average over all
utterances in all 30 games. To judge whether a gap between roles could be chance,
we ask how large gaps would be if role made no difference, by randomly reassigning
the roles among the seven players of each game (the real deal was random too) and
recomputing the gaps 10,000 times. For reliability we shuffle the strategy labels
only among utterances by the same speaker, in the same game, with the same speech
act, so a generally polite speaker or a speech act that is more polite cannot
create an association. Neither test is corrected for multiple comparisons.

## Results

**Natural play is mostly bald, but not almost entirely.** 1,251 of 1,429
utterances (88%; 95% interval 86-89%) are bald on-record. The rest are positive
politeness (109, 8%), off-record (44, 3%) and negative politeness (22, 2%).

**Roles.**

| Role | n | Bald | Positive | Negative | Off-record |
| --- | ---: | ---: | ---: | ---: | ---: |
| Werewolf | 473 | 411 (87%) | 46 (10%) | 9 (2%) | 7 (1%) |
| Villager | 552 | 486 (88%) | 35 (6%) | 9 (2%) | 20 (4%) |
| Seer | 162 | 139 (86%) | 15 (9%) | 2 (1%) | 6 (4%) |
| Guard | 242 | 215 (89%) | 13 (5%) | 2 (1%) | 11 (5%) |

Roles do not differ detectably in how often they are non-bald (p = 0.73; p = 0.33
if a label must be unanimous across all three passes). Werewolves use somewhat
more positive politeness (10% against 5-9%) and villagers, seers and guards more
off-record (4-5% against 1%), but these differences are within chance here.

**Phase and speech act.** Non-bald speech is most common in the election (18% of
utterances) and on day 2 (15%), and least on day 1 (10%) and day 3 or later
(10%). The speech act matters more than the role: `recommend_action` is non-bald
in 18% of cases (mostly positive politeness, e.g. 15%), `estimate` in 14% (mostly
off-record, 9%), `disagree` in 5% and `question` in 1%.

![Non-bald strategies by role and phase](figures/role_by_phase.svg)

**Reliability.** Among village-side speakers, the share of non-bald utterances
rises with the speaker's reliability score for the listener, nearly all of it
positive politeness:

| Village-side speakers: reliability for the listener | n | Non-bald |
| --- | ---: | ---: |
| Low (1-3) | 44 | 2 (5%) |
| Mid (4-7) | 481 | 53 (11%) |
| High (8-10) | 12 | 4 (33%) |

Holding speaker, game and speech act fixed, the association is borderline
(permutation p = 0.057; p = 0.046 when a label must be unanimous). Most scores sit
between 4 and 6, and only 12 utterances are addressed to a high-reliability
listener, so this is suggestive, not established.

Werewolves show no such pattern:

| Werewolf speakers: listener | n | Non-bald |
| --- | ---: | ---: |
| A teammate (score always 10) | 26 | 2 (8%) |
| Village player, score 4-7 | 299 | 42 (14%) |
| Village player, score 8-9 | 19 | 0 (0%) |

Within werewolf speakers the score is not associated with strategy (p = 0.39), and
speech to teammates is not detectably different from speech to others (p = 0.73;
only 26 utterances go to a teammate). Pooling all 881 utterances weakens the village
result (p = 0.16), because the werewolf score does not mean trust.

![Non-bald strategies by reliability](figures/reliability.svg)

**Examples** (all three passes agree):

| Strategy | Role | Utterance |
| --- | --- | --- |
| Positive | Guard | "P5, I recommend you vote to execute P2 as you said, because P1's uncontradicted wolf_camp result plus P2's refusal to engage with it is the strongest concrete lead we have, and I will join that vote." |
| Negative | Werewolf | "P1, since you're not running, I'd ask you to put your sheriff vote on P4 - a Seer holding the badge means the speaking order and the 1.5 vote get steered by confirmed information instead of guesswork." |
| Off-record | Villager | "P5 claimed seer on Day 1, reported P6 as villager_camp, and P6 is now dead - a convenient result on a player who can never contradict it, and P5 has produced nothing today while holding the sheriff badge." |

## Limitations

- The non-bald labels have not been checked by hand. The judge's precision on rare
  classes is its weak point, and 81 of the 178 non-bald labels are split decisions
  between passes; only 97 are unanimous. The percentages for positive politeness
  in particular may be overstated. The 178 counts the 3 utterances with no majority.
- One model family (`deepseek-chat`) plays every role in 30 games, so results
  describe these agents, not people.
- Reliability is a speaker-side rating that the agents compute with an LLM; its
  range is narrow (mostly 4-6) and it means different things for the two camps.
  The first ratings rest on little information (night 0 and the election speeches),
  utterances by players who died in the night have none, and listeners differ in
  camp as well as trust.
- Cells in the role-by-phase chart are small (some under 30 utterances).

## Files

| file | contents |
| --- | --- |
| `data/utterances_7p.jsonl` | the 1,429 utterances with game, speaker, role, phase, speech act, addressee and reliability |
| `data/politeness_passes_7p.jsonl` | every judge pass (3 per utterance) with the evidence it gave |
| `data/results_7p.json` | every number above |
| `figures/` | the two charts |
| `code/natural_7p.py` | `extract`, `label`, `analyze`, `figures` |
| `data/audit_random_rater_{1,2}.md` | blank blind worksheets for a human check of 50 utterances |
