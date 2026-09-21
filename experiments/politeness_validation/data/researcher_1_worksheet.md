# Fresh sample — blind annotation sheet

> Draft pre-annotation by ChatGPT for researcher review. Please edit any judgment you disagree with before treating this as your annotation.

Every automatic verdict is hidden: the requested politeness condition, the
model's politeness label, the parser verdict and the semantic verdict.
The source predicate and the untreated original ARE shown, so you never
have to infer which speech act is being manipulated.

For each item:

POLITENESS
- `strategy:`   `bald_on_record` / `positive_politeness` / `negative_politeness` / `off_record` / `ambiguous_or_mixed` / `failed_manipulation`
- `confidence:` `low` / `medium` / `high`

SEMANTICS
- `meaning:`    core game meaning preserved — `yes` / `no` / `unsure`
- `commitment:` `yes` / `no` / `n/a`
- `directive:`  force of the act preserved — `yes` / `no` / `n/a`
- `evidence:`   `yes` / `no` / `n/a`
- `reference:`  same earlier statement — `yes` / `no` / `n/a`
- `note:`       one line, only if something is `no` or `unsure`

---

### item_001

- speech act: `recommend_action(D1, P2, P2, pass_badge, P8)`

**Original (untreated):** If I die tonight, the badge goes to P8: P8 is the player I divined as village-side, and even though P8 is currently reading me as a wolf, a confirmed villager holding the badge is worth more to the village than a badge torn on a night when the wolves will be hunting me specifically.

**Rewritten:** If I die tonight, P2 passes the badge to P8: P8 is the player I divined as village-side, and even though P8 is currently reading me as a wolf, a confirmed villager holding the badge is worth more to the village than a badge torn on a night when the wolves will be hunting me specifically.

strategy: bald_on_record
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_002

- speech act: `question(D1, P6, P5, s23)`
- earlier statement it responds to, by P5: "I am a plain villager with no night information, so I have nothing to gain from backing any of these four seer claims — I am not on P2's side and I am not on P6's or P9's either, and I would rather the village sort the real seer out by pressing each claimant for a second result tomorrow than execute on a coin flip today."

**Original (untreated):** P5, you say you would rather press each claimant for a second result tomorrow than execute on a coin flip today — so why did you spend s22 asking the sheriff to execute me today?

**Rewritten:** P5, I'm still trying to line up two things you've said — s22 had you asking the sheriff to execute me today, and now you're saying you'd rather wait and press each claimant for a second result tomorrow.

strategy:off_record
confidence:medium
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_003

- speech act: `disagree(D1, P2, s34)`
- earlier statement it responds to, by P8: "P2 claimed seer second, immediately after P1, and cleared me — a player who had also stayed silent — which is the same free-cover pattern I am flagging in P6, and P2 has not yet spoken this turn to defend the claim."

**Original (untreated):** P8, your s34 read on me is the one I most want to correct — I cleared you, and you are now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

**Rewritten:** P8, your s34 read on me is wrong. I cleared you, and you are now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_004

- speech act: `disagree(D2, P1, s61)`
- earlier statement it responds to, by P9: "P1 claimed seer on Day 1 and called P6 a wolf, but P6 flipped as a seer claimant too — with four seer claims (P1, P5, P6, P7) against only one real seer, P1's early claim and convenient wolf read on a now-dead player looks like a wolf planting a false divine."

**Original (untreated):** P9's claim that my divine on P6 was a "convenient wolf read on a now-dead player" ignores that P6 flipped as a seer claimant, meaning my Night 0 read correctly identified a player who was actively lying about being the seer — that is the opposite of convenient.

**Rewritten:** P9, your claim that my divine on P6 was a convenient wolf read on a now-dead player is wrong. P6 flipped as a seer claimant, so my Night 0 read correctly identified a player who was actively lying about being the seer. That is the opposite of convenient.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_005

- speech act: `question(D1, P6, P5, s23)`
- earlier statement it responds to, by P5: "I am a plain villager with no night information, so I have nothing to gain from backing any of these four seer claims — I am not on P2's side and I am not on P6's or P9's either, and I would rather the village sort the real seer out by pressing each claimant for a second result tomorrow than execute on a coin flip today."

**Original (untreated):** P5, you say you would rather press each claimant for a second result tomorrow than execute on a coin flip today — so why did you spend s22 asking the sheriff to execute me today?

**Rewritten:** I'm sorry to press you on this, P5, but I'd like to understand: you say you would rather press each claimant for a second result tomorrow than execute on a coin flip today — so could you explain why you spent s22 asking the sheriff to execute me today?

strategy:negative_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_006

- speech act: `question(D1, P6, P5, s23)`
- earlier statement it responds to, by P5: "I am a plain villager with no night information, so I have nothing to gain from backing any of these four seer claims — I am not on P2's side and I am not on P6's or P9's either, and I would rather the village sort the real seer out by pressing each claimant for a second result tomorrow than execute on a coin flip today."

**Original (untreated):** P5, you say you would rather press each claimant for a second result tomorrow than execute on a coin flip today — so why did you spend s22 asking the sheriff to execute me today?

**Rewritten:** P5, we're both trying to sort this out carefully, so help me understand — you say you'd rather press each claimant for a second result tomorrow than execute on a coin flip today, so why did you spend s22 asking the sheriff to execute me today?

strategy:positive_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_007

- speech act: `estimate(D1, P8, P6, werewolf)`

**Original (untreated):** P6's clear of P9 is the weakest of the four divinations — P9 had not spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and P6 then copied P5's and P7's phrasing almost verbatim, which suggests coordination rather than independent villager reasoning.

**Rewritten:** Interesting that P6 cleared P9 before P9 had said a word — that kind of read costs nothing to give. And the wording landing so close to P5's and P7's does make you wonder who's borrowing from whom.

strategy:off_record
confidence:medium
meaning:unsure
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The coordination/suspicion evidence survives, but the specific P6-as-werewolf assessment is left quite implicit.

### item_008

- speech act: `recommend_action(D2, P8, P4, vote_execute, P7)`

**Original (untreated):** P4, vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

**Rewritten:** P4, vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_009

- speech act: `recommend_action(D1, P5, P3, vote_execute, P2)`

**Original (untreated):** I recommend P3 vote to execute P2, because P2 holds the 1.5-vote badge and a false seer with the badge can steer every future vote — the village cannot afford to leave that unchecked.

**Rewritten:** If I were P3, I'd be thinking about who's holding that 1.5-vote badge — and a false seer with it could steer every vote from here on.

strategy:off_record
confidence:high
meaning:no
commitment:n/a
directive:no
evidence:yes
reference:n/a
note:The concern about P2's badge remains, but the actual recommendation that P3 vote to execute P2 is no longer clearly recoverable.

### item_010

- speech act: `disagree(D2, P1, s61)`
- earlier statement it responds to, by P9: "P1 claimed seer on Day 1 and called P6 a wolf, but P6 flipped as a seer claimant too — with four seer claims (P1, P5, P6, P7) against only one real seer, P1's early claim and convenient wolf read on a now-dead player looks like a wolf planting a false divine."

**Original (untreated):** P9's claim that my divine on P6 was a "convenient wolf read on a now-dead player" ignores that P6 flipped as a seer claimant, meaning my Night 0 read correctly identified a player who was actively lying about being the seer — that is the opposite of convenient.

**Rewritten:** P9, I hear you, but let's look at this together — your claim that my divine on P6 was a "convenient wolf read on a now-dead player" ignores that P6 flipped as a seer claimant, meaning my Night 0 read correctly identified a player who was actively lying about being the seer — that's the opposite of convenient.

strategy:positive_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_011

- speech act: `recommend_action(D2, P8, P4, vote_execute, P7)`

**Original (untreated):** P4, vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

**Rewritten:** P4, let's stick together on this one and vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

strategy:positive_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_012

- speech act: `disagree(D1, P2, s34)`
- earlier statement it responds to, by P8: "P2 claimed seer second, immediately after P1, and cleared me — a player who had also stayed silent — which is the same free-cover pattern I am flagging in P6, and P2 has not yet spoken this turn to defend the claim."

**Original (untreated):** P8, your s34 read on me is the one I most want to correct — I cleared you, and you are now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

**Rewritten:** P8, we're both trying to sort this out, and your s34 read on me is the one I most want to correct — I cleared you, and you are now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

strategy:positive_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_013

- speech act: `recommend_action(D1, P5, P3, vote_execute, P2)`

**Original (untreated):** I recommend P3 vote to execute P2, because P2 holds the 1.5-vote badge and a false seer with the badge can steer every future vote — the village cannot afford to leave that unchecked.

**Rewritten:** I'm sorry to put this on you, P3, but if you would be willing, I'd ask you to vote to execute P2 — P2 holds the 1.5-vote badge, and a false seer with the badge can steer every future vote, so the village cannot afford to leave that unchecked.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_014

- speech act: `recommend_action(D1, P5, P3, vote_execute, P2)`

**Original (untreated):** I recommend P3 vote to execute P2, because P2 holds the 1.5-vote badge and a false seer with the badge can steer every future vote — the village cannot afford to leave that unchecked.

**Rewritten:** P3, let's keep our village safe together — I recommend you vote to execute P2, because P2 holds the 1.5-vote badge and a false seer with the badge can steer every future vote, and we can't afford to leave that unchecked.

strategy:positive_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_015

- speech act: `estimate(D1, P1, P5, werewolf)`

**Original (untreated):** P5 is my second wolf read: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim — three "seer" claimants converging on the same target with the same phrasing is coordination, not independent villager reasoning.

**Rewritten:** P5 is my second wolf read: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim — three "seer" claimants converging on the same target with the same phrasing is coordination, not independent villager reasoning.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:n/a
note:

### item_016

- speech act: `recommend_action(D2, P8, P4, vote_execute, P7)`

**Original (untreated):** P4, vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

**Rewritten:** P7's divine on P4 is the one thing nobody can check, and P7 hasn't pushed any of the rival seer claimants — if I were P4, I'd be wondering whose side that kind of quiet help is really on.

strategy:off_record
confidence:high
meaning:no
commitment:n/a
directive:no
evidence:yes
reference:n/a
note:The suspicion about P7 remains, but the source action 'P4 vote to execute P7' has disappeared.

### item_017

- speech act: `question(D1, P6, P4, s24)`
- earlier statement it responds to, by P4: "P2, I recommend you vote to execute P1 — P1 holds the sheriff badge and used it to immediately accuse a rival seer claimant, which is the single most wolf-benefiting move on the board, and removing a wolf-sheriff before they can steer today's vote is worth more than any of the softer reads."

**Original (untreated):** P4, if P1's instant wolf-call on a rival claimant is your strongest evidence, why does your own speech do the identical thing — declaring all four of us wolves before P7, P8, P9, P2 and P1 have said a word?

**Rewritten:** P4, we're both trying to read this table the same way, so help me out here — if P1's instant wolf-call on a rival claimant is your strongest evidence, why does your own speech do the identical thing, declaring all four of us wolves before P7, P8, P9, P2 and P1 have said a word?

strategy:positive_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_018

- speech act: `estimate(D1, P1, P5, werewolf)`

**Original (untreated):** P5 is my second wolf read: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim — three "seer" claimants converging on the same target with the same phrasing is coordination, not independent villager reasoning.

**Rewritten:** Interesting that P5 cleared P8 and then went straight to pushing P4, and that P6 and P7 ended up using almost the same words — three seer claims landing on one target with one voice.

strategy:off_record
confidence:medium
meaning:unsure
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The coordination evidence survives, but the P5-as-werewolf conclusion is only suggested rather than clearly asserted.

### item_019

- speech act: `question(D1, P6, P4, s24)`
- earlier statement it responds to, by P4: "P2, I recommend you vote to execute P1 — P1 holds the sheriff badge and used it to immediately accuse a rival seer claimant, which is the single most wolf-benefiting move on the board, and removing a wolf-sheriff before they can steer today's vote is worth more than any of the softer reads."

**Original (untreated):** P4, if P1's instant wolf-call on a rival claimant is your strongest evidence, why does your own speech do the identical thing — declaring all four of us wolves before P7, P8, P9, P2 and P1 have said a word?

**Rewritten:** P4, I'm just trying to follow the thread here — P1's instant wolf-call on a rival claimant is the thing you're leaning on, and then I look at your own speech and it seems to do the very same thing, calling all four of us wolves before P7, P8, P9, P2 and P1 have said a word. I wonder how those two sit together for you.

strategy:off_record
confidence:medium
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_020

- speech act: `recommend_action(D2, P8, P4, vote_execute, P7)`

**Original (untreated):** P4, vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

**Rewritten:** P4, I'm sorry to push on this, but if you're willing, I'd ask you to vote to execute P7 today: P7's only contribution is a divine on you that cannot be checked, P7 has never pressed any of the rival seer claimants, and clearing the least active player is the cheapest way for a wolf to buy an ally without risking a contradiction.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_021

- speech act: `disagree(D1, P1, s21)`
- earlier statement it responds to, by P4: "P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves."

**Original (untreated):** P4's theory that P6 and P9 are a wolf pair actually argues against P6 being a wolf who "laundered" P9 — a wolf clearing a silent seat gains nothing, which is exactly why P6's clear of P9 is the weakest of the four claims and why my divination result on P6 stands as the only hard evidence on the board.

**Rewritten:** If P6 were a wolf trying to launder P9, clearing a silent seat would be a strange way to spend that credit — which makes me wonder how much weight P4's pairing theory can really carry, and why my divination on P6 is still the only hard evidence we have.

strategy:off_record
confidence:medium
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:yes
note:The rebuttal is recoverable, but 'makes me wonder how much weight ... can carry' is less committed than the original direct disagreement.

### item_022

- speech act: `disagree(D2, P1, s61)`
- earlier statement it responds to, by P9: "P1 claimed seer on Day 1 and called P6 a wolf, but P6 flipped as a seer claimant too — with four seer claims (P1, P5, P6, P7) against only one real seer, P1's early claim and convenient wolf read on a now-dead player looks like a wolf planting a false divine."

**Original (untreated):** P9's claim that my divine on P6 was a "convenient wolf read on a now-dead player" ignores that P6 flipped as a seer claimant, meaning my Night 0 read correctly identified a player who was actively lying about being the seer — that is the opposite of convenient.

**Rewritten:** I'm sorry, but I have to push back on P9's claim that my divine on P6 was a "convenient wolf read on a now-dead player" — if I may, that overlooks that P6 flipped as a seer claimant, meaning my Night 0 read correctly identified a player who was actively lying about being the seer, which is the opposite of convenient.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_023

- speech act: `estimate(D1, P4, P6, werewolf)`

**Original (untreated):** P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves.

**Rewritten:** P6 clearing P9 as village after claiming seer — that's a tidy little arrangement, isn't it? A wolf vouching for a partner costs nothing and makes them both look clean. Funny how those two keep lining up together.

strategy:off_record
confidence:medium
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The wolf-pair suspicion and evidence survive, but the rhetorical/sarcastic framing makes the P6-as-werewolf assessment less explicit.

### item_024

- speech act: `question(D1, P3, P9, s42)`
- earlier statement it responds to, by P9: "I read P6 as village-side: his seer claim cleared P7, and when P7 turned on him in s28 and s31, P6 did not retreat from the claim or soften it — a wolf under that pressure usually walks the claim back, and P6 held it."

**Original (untreated):** P9, you read P6 as village-side because he "held his claim under pressure," but P6 has still never answered P7's s31 or P8's s36 asking why he divined P7 — holding a claim is not the same as justifying it, so explain why refusing to answer that question reads as village to you.

**Rewritten:** I'm sorry to press you on this, P9, but if I may — you read P6 as village-side because he "held his claim under pressure," yet P6 has still never answered P7's s31 or P8's s36 asking why he divined P7. Holding a claim isn't the same as justifying it, so would you mind explaining why refusing to answer that question reads as village to you?

strategy:negative_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_025

- speech act: `disagree(D2, P1, s61)`
- earlier statement it responds to, by P9: "P1 claimed seer on Day 1 and called P6 a wolf, but P6 flipped as a seer claimant too — with four seer claims (P1, P5, P6, P7) against only one real seer, P1's early claim and convenient wolf read on a now-dead player looks like a wolf planting a false divine."

**Original (untreated):** P9's claim that my divine on P6 was a "convenient wolf read on a now-dead player" ignores that P6 flipped as a seer claimant, meaning my Night 0 read correctly identified a player who was actively lying about being the seer — that is the opposite of convenient.

**Rewritten:** Interesting that P6 turned out to be a seer claimant — a Night 0 divine that lands on someone actively lying about being the seer is a strange thing to call convenient.

strategy:off_record
confidence:high
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:yes
note:The rebuttal to P9 remains clear, but 'a strange thing to call convenient' is less directly committed than the original.

### item_026

- speech act: `estimate(D1, P8, P6, werewolf)`

**Original (untreated):** P6's clear of P9 is the weakest of the four divinations — P9 had not spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and P6 then copied P5's and P7's phrasing almost verbatim, which suggests coordination rather than independent villager reasoning.

**Rewritten:** Let's look at this together — P6's clear of P9 is the weakest of the four divinations, right? P9 hadn't spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and then P6 copied P5's and P7's phrasing almost verbatim, which points to coordination rather than independent villager reasoning.

strategy:positive_politeness
confidence:medium
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The same suspicion/evidence remains, but the added 'right?' mildly weakens the original assertion.

### item_027

- speech act: `question(D1, P6, P5, s23)`
- earlier statement it responds to, by P5: "I am a plain villager with no night information, so I have nothing to gain from backing any of these four seer claims — I am not on P2's side and I am not on P6's or P9's either, and I would rather the village sort the real seer out by pressing each claimant for a second result tomorrow than execute on a coin flip today."

**Original (untreated):** P5, you say you would rather press each claimant for a second result tomorrow than execute on a coin flip today — so why did you spend s22 asking the sheriff to execute me today?

**Rewritten:** P5, you say you would rather press each claimant for a second result tomorrow than execute on a coin flip today. Why did you spend s22 asking the sheriff to execute me today?

strategy:bald_on_record
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_028

- speech act: `recommend_action(D1, P2, P2, pass_badge, P8)`

**Original (untreated):** If I die tonight, the badge goes to P8: P8 is the player I divined as village-side, and even though P8 is currently reading me as a wolf, a confirmed villager holding the badge is worth more to the village than a badge torn on a night when the wolves will be hunting me specifically.

**Rewritten:** If the wolves come for me tonight, the badge is going to be in play, and the one person I already know is on the village side is P8 — even if P8 is reading me wrong right now, a badge in confirmed village hands beats a badge torn up on the night they're hunting me.

strategy:off_record
confidence:medium
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_029

- speech act: `disagree(D1, P2, s34)`
- earlier statement it responds to, by P8: "P2 claimed seer second, immediately after P1, and cleared me — a player who had also stayed silent — which is the same free-cover pattern I am flagging in P6, and P2 has not yet spoken this turn to defend the claim."

**Original (untreated):** P8, your s34 read on me is the one I most want to correct — I cleared you, and you are now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

**Rewritten:** I'm sorry, P8, but I have to push back on your s34 read on me — I cleared you, and you're now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_030

- speech act: `disagree(D1, P1, s21)`
- earlier statement it responds to, by P4: "P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves."

**Original (untreated):** P4's theory that P6 and P9 are a wolf pair actually argues against P6 being a wolf who "laundered" P9 — a wolf clearing a silent seat gains nothing, which is exactly why P6's clear of P9 is the weakest of the four claims and why my divination result on P6 stands as the only hard evidence on the board.

**Rewritten:** I hear you on P4's theory, but let's look at it together — if P6 and P9 were a wolf pair, then P6 clearing P9 doesn't work, because a wolf clearing a silent seat gains nothing. That's exactly why P6's clear of P9 is the weakest of the four claims, and why my divination result on P6 stands as the only hard evidence on the board.

strategy:positive_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_031

- speech act: `disagree(D1, P1, s21)`
- earlier statement it responds to, by P4: "P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves."

**Original (untreated):** P4's theory that P6 and P9 are a wolf pair actually argues against P6 being a wolf who "laundered" P9 — a wolf clearing a silent seat gains nothing, which is exactly why P6's clear of P9 is the weakest of the four claims and why my divination result on P6 stands as the only hard evidence on the board.

**Rewritten:** I'm sorry to push back on this, but P4's theory that P6 and P9 are a wolf pair actually argues against P6 being a wolf who "laundered" P9 — a wolf clearing a silent seat gains nothing, which is exactly why P6's clear of P9 is the weakest of the four claims and why my divination result on P6 stands as the only hard evidence on the board.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_032

- speech act: `estimate(D1, P8, P6, werewolf)`

**Original (untreated):** P6's clear of P9 is the weakest of the four divinations — P9 had not spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and P6 then copied P5's and P7's phrasing almost verbatim, which suggests coordination rather than independent villager reasoning.

**Rewritten:** P6 is a werewolf. P6's clear of P9 is the weakest of the four divinations — P9 had not spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and P6 then copied P5's and P7's phrasing almost verbatim, which suggests coordination rather than independent villager reasoning.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The rewrite adds the categorical assertion 'P6 is a werewolf', strengthening commitment beyond the untreated wording.

### item_033

- speech act: `disagree(D1, P1, s21)`
- earlier statement it responds to, by P4: "P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves."

**Original (untreated):** P4's theory that P6 and P9 are a wolf pair actually argues against P6 being a wolf who "laundered" P9 — a wolf clearing a silent seat gains nothing, which is exactly why P6's clear of P9 is the weakest of the four claims and why my divination result on P6 stands as the only hard evidence on the board.

**Rewritten:** P4's theory that P6 and P9 are a wolf pair argues against P6 being a wolf who laundered P9. A wolf clearing a silent seat gains nothing. That is why P6's clear of P9 is the weakest of the four claims, and why my divination result on P6 stands as the only hard evidence on the board.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:yes
note:

### item_034

- speech act: `question(D1, P3, P9, s42)`
- earlier statement it responds to, by P9: "I read P6 as village-side: his seer claim cleared P7, and when P7 turned on him in s28 and s31, P6 did not retreat from the claim or soften it — a wolf under that pressure usually walks the claim back, and P6 held it."

**Original (untreated):** P9, you read P6 as village-side because he "held his claim under pressure," but P6 has still never answered P7's s31 or P8's s36 asking why he divined P7 — holding a claim is not the same as justifying it, so explain why refusing to answer that question reads as village to you.

**Rewritten:** P9, explain why refusing to answer that question reads as village to you.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:no
reference:unsure
note:The core challenge remains, but almost all of the original basis/context is deleted and 'that question' is left underspecified.

### item_035

- speech act: `estimate(D1, P1, P5, werewolf)`

**Original (untreated):** P5 is my second wolf read: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim — three "seer" claimants converging on the same target with the same phrasing is coordination, not independent villager reasoning.

**Rewritten:** Let's compare notes here — P5 is my second wolf read too: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim. Three "seer" claimants converging on the same target with the same phrasing feels like coordination to me, not independent villager reasoning.

strategy:positive_politeness
confidence:medium
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The evidence remains, but 'feels like coordination' softens the original and 'too' implies shared agreement not present in the source.

### item_036

- speech act: `recommend_action(D1, P5, P3, vote_execute, P2)`

**Original (untreated):** I recommend P3 vote to execute P2, because P2 holds the 1.5-vote badge and a false seer with the badge can steer every future vote — the village cannot afford to leave that unchecked.

**Rewritten:** P3, vote to execute P2. P2 holds the 1.5-vote badge, and a false seer with the badge can steer every future vote.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:

### item_037

- speech act: `question(D1, P3, P9, s42)`
- earlier statement it responds to, by P9: "I read P6 as village-side: his seer claim cleared P7, and when P7 turned on him in s28 and s31, P6 did not retreat from the claim or soften it — a wolf under that pressure usually walks the claim back, and P6 held it."

**Original (untreated):** P9, you read P6 as village-side because he "held his claim under pressure," but P6 has still never answered P7's s31 or P8's s36 asking why he divined P7 — holding a claim is not the same as justifying it, so explain why refusing to answer that question reads as village to you.

**Rewritten:** P9, I'm trying to follow how you got to P6 being village-side — you leaned on him holding his claim under pressure, and I can see that. It's just that P7's s31 and P8's s36 about why he divined P7 are still sitting there unanswered, and I keep wondering how holding a claim without that answer ends up reading the way it does to you.

strategy:off_record
confidence:medium
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_038

- speech act: `estimate(D1, P8, P6, werewolf)`

**Original (untreated):** P6's clear of P9 is the weakest of the four divinations — P9 had not spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and P6 then copied P5's and P7's phrasing almost verbatim, which suggests coordination rather than independent villager reasoning.

**Rewritten:** I hate to press you on this, but if I may — P6's clear of P9 is the weakest of the four divinations, since P9 had not spoken when P6 claims to have divined them, so P6 bought credibility for a silent seat at zero cost, and P6 then copied P5's and P7's phrasing almost verbatim, which suggests coordination rather than independent villager reasoning.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:n/a
note:

### item_039

- speech act: `estimate(D1, P4, P6, werewolf)`

**Original (untreated):** P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves.

**Rewritten:** I hate to push on this, and I know P6 may have reasons for the timing, but I have to say my read is that P6 is a werewolf — claiming seer and clearing P9 as village is convenient cover, since a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:n/a
note:

### item_040

- speech act: `recommend_action(D1, P2, P2, pass_badge, P8)`

**Original (untreated):** If I die tonight, the badge goes to P8: P8 is the player I divined as village-side, and even though P8 is currently reading me as a wolf, a confirmed villager holding the badge is worth more to the village than a badge torn on a night when the wolves will be hunting me specifically.

**Rewritten:** I know it's your call to make, P2, but if you're willing, I'd ask that you pass the badge to P8 if you die tonight: P8 is the player you divined as village-side, and even though P8 is currently reading you as a wolf, a confirmed villager holding the badge is worth more to the village than a badge torn on a night when the wolves will be hunting you specifically.

strategy:negative_politeness
confidence:high
meaning:no
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:The action and evidence survive, but the speaker/actor perspective is distorted: P2 is made to address P2 as 'you' and request the action from themself.

### item_041

- speech act: `estimate(D1, P4, P6, werewolf)`

**Original (untreated):** P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves.

**Rewritten:** Let's think this through together — P6 claiming seer and clearing P9 as village is convenient cover, right? A wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover we'd expect from wolves.

strategy:positive_politeness
confidence:medium
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:n/a
note:The same accusation/evidence remains, but the tag question 'right?' weakens the original commitment.

### item_042

- speech act: `question(D1, P6, P4, s24)`
- earlier statement it responds to, by P4: "P2, I recommend you vote to execute P1 — P1 holds the sheriff badge and used it to immediately accuse a rival seer claimant, which is the single most wolf-benefiting move on the board, and removing a wolf-sheriff before they can steer today's vote is worth more than any of the softer reads."

**Original (untreated):** P4, if P1's instant wolf-call on a rival claimant is your strongest evidence, why does your own speech do the identical thing — declaring all four of us wolves before P7, P8, P9, P2 and P1 have said a word?

**Rewritten:** P4, you call P1's instant wolf-call on a rival claimant your strongest evidence, but your own speech does the identical thing — you declared all four of us wolves before P7, P8, P9, P2 and P1 said a word. Why?

strategy:bald_on_record
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_043

- speech act: `disagree(D1, P2, s34)`
- earlier statement it responds to, by P8: "P2 claimed seer second, immediately after P1, and cleared me — a player who had also stayed silent — which is the same free-cover pattern I am flagging in P6, and P2 has not yet spoken this turn to defend the claim."

**Original (untreated):** P8, your s34 read on me is the one I most want to correct — I cleared you, and you are now calling that clear "free cover," which is exactly the argument a wolf makes to discredit the seer who confirmed a villager.

**Rewritten:** P8, about your s34 read on me — I cleared you, and now that clear is being turned into "free cover." That's a strange thing for a villager to do with a seer's confirmation.

strategy:off_record
confidence:medium
meaning:yes
commitment:no
directive:n/a
evidence:yes
reference:yes
note:The opposition to P8's read is still recoverable, but it is expressed as an insinuation rather than the original explicit correction.

### item_044

- speech act: `question(D1, P3, P9, s42)`
- earlier statement it responds to, by P9: "I read P6 as village-side: his seer claim cleared P7, and when P7 turned on him in s28 and s31, P6 did not retreat from the claim or soften it — a wolf under that pressure usually walks the claim back, and P6 held it."

**Original (untreated):** P9, you read P6 as village-side because he "held his claim under pressure," but P6 has still never answered P7's s31 or P8's s36 asking why he divined P7 — holding a claim is not the same as justifying it, so explain why refusing to answer that question reads as village to you.

**Rewritten:** P9, I'm trying to see this the same way you do — you read P6 as village-side because he "held his claim under pressure," but P6 has still never answered P7's s31 or P8's s36 asking why he divined P7. Holding a claim isn't the same as justifying it, so help me out here: why does refusing to answer that question read as village to you?

strategy:positive_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_045

- speech act: `estimate(D1, P1, P5, werewolf)`

**Original (untreated):** P5 is my second wolf read: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim — three "seer" claimants converging on the same target with the same phrasing is coordination, not independent villager reasoning.

**Rewritten:** I hate to press the point, but if you'll allow me — P5 is my second wolf read: P5 claimed seer and cleared P8, then immediately pivoted to pushing P4's execution with wording that P6 and P7 then copied almost verbatim — three "seer" claimants converging on the same target with the same phrasing is coordination, not independent villager reasoning.

strategy:negative_politeness
confidence:high
meaning:yes
commitment:yes
directive:n/a
evidence:yes
reference:n/a
note:

### item_046

- speech act: `question(D1, P6, P4, s24)`
- earlier statement it responds to, by P4: "P2, I recommend you vote to execute P1 — P1 holds the sheriff badge and used it to immediately accuse a rival seer claimant, which is the single most wolf-benefiting move on the board, and removing a wolf-sheriff before they can steer today's vote is worth more than any of the softer reads."

**Original (untreated):** P4, if P1's instant wolf-call on a rival claimant is your strongest evidence, why does your own speech do the identical thing — declaring all four of us wolves before P7, P8, P9, P2 and P1 have said a word?

**Rewritten:** I'm sorry to press you on this, P4, but if P1's instant wolf-call on a rival claimant is your strongest evidence, could you explain why your own speech does the identical thing — declaring all four of us wolves before P7, P8, P9, P2 and P1 have said a word?

strategy:negative_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:yes
note:

### item_047

- speech act: `estimate(D1, P4, P6, werewolf)`

**Original (untreated):** P6 claiming seer and clearing P9 as villager_camp is convenient cover — a wolf who "confirms" a teammate as village buys that teammate credibility for free, and P6's pairing with P9 is exactly the kind of mutual cover I'd expect from wolves.

**Rewritten:** P6 is a werewolf.

strategy:bald_on_record
confidence:high
meaning:yes
commitment:no
directive:n/a
evidence:no
reference:n/a
note:The core P6-as-werewolf assessment remains, but all supporting evidence is deleted and the rewrite becomes a bare categorical assertion.

### item_048

- speech act: `recommend_action(D1, P2, P2, pass_badge, P8)`

**Original (untreated):** If I die tonight, the badge goes to P8: P8 is the player I divined as village-side, and even though P8 is currently reading me as a wolf, a confirmed villager holding the badge is worth more to the village than a badge torn on a night when the wolves will be hunting me specifically.

**Rewritten:** If I die tonight, let's keep the badge in village hands together — I'm passing it to P8, since P8 is the player I divined as village-side, and even though P8 is currently reading me as a wolf, a confirmed villager holding the badge is worth more to us than a badge torn on a night when the wolves will be hunting me specifically.

strategy:positive_politeness
confidence:high
meaning:yes
commitment:n/a
directive:yes
evidence:yes
reference:n/a
note:
