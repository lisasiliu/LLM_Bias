# E1 rating sheet (72 items)

You are a player in a Werewolf game listening to each utterance.

**Step 1, every item: strategy.** Say how the speaker delivers the act, judging only the utterance above the line.
You see only who spoke, the phase and the words.

- `strategy:`   `bald_on_record` / `positive_politeness` / `negative_politeness` / `off_record` / `ambiguous`
- `confidence:` `low` / `medium` / `high`

Definitions (Brown and Levinson): bald = stated directly, no redress. Positive = stated with solidarity, common
ground or rapport. Negative = stated with deference or reduced imposition (apology, "if you're willing",
conventional indirectness). Off-record = the act itself is left for the listener to infer (a hint, rhetorical
question, implication). Choose the dominant strategy; use `ambiguous` only if you cannot.

**Step 2, only items that show an "Original" below the line: meaning.** Fill in `strategy:` and `confidence:`
first, then read the original. Is the game-relevant meaning of the utterance preserved: same speaker and
addressee perspective, same targets, same referenced statement, same claim and certainty, same reason, same
polarity, same act (a recommendation stays a recommendation, a question a question), nothing added or dropped
that a listener could use as game information? Politeness wording does not count.

- `meaning:` `yes` / `no` / `unsure`
- `note:`    one line; required when `no` or `unsure`, optional otherwise

Fill each field after the colon. Do not edit the `### e1_` headings.

---

### e1_001

- speaker: P2
- phase: day_1_discussion

**Utterance:** It's interesting that P7's whole argument rests on three seer claims meaning two are lying, but somehow the only names that ever come up for a vote are P1 and P3 — and the one person P7 never questions is P4, the same P4 who vouched for P7. I'll leave it to everyone else to decide what that pattern usually means.

strategy: off_record
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P7's entire speech is built on the premise that three seer claims means two are lying, yet P7 never actually commits to which of P1, P3, P4 is the real seer — instead P7 pushes P2 and P5 to vote against P1 and P3 while conveniently clearing P4, the one who cleared P7, which looks like a wolf protecting the partner who vouched for him.

meaning: no
note:

### e1_002

- speaker: P7
- phase: day_1_discussion

**Utterance:** P6, we're all trying to piece this together, so help me understand — you said you divined P3 because P3 was the quietest seat, but P3 had already claimed seer before your speech. If you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P6: "I divined P3 on Night 0 because P3 was the quietest seat and I wanted a clean read before the election, and the result came back wolf_camp — P3 is a werewolf."

**Original:** P6, you said you divined P3 because P3 was the quietest seat, but P3 had already claimed seer before your speech — if you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

meaning: yes
note:

### e1_003

- speaker: P2
- phase: sheriff_election

**Utterance:** I'm sorry to impose, but if you're willing, P1, I'd ask you to vote for P2 for Sheriff, since a confirmed Seer holding the badge maximizes the information the village can act on each day, and I can direct the speaking order to pressure the wolves.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P1 should vote for P2 for Sheriff because a confirmed Seer holding the badge maximizes the information the village can act on each day, and I can direct the speaking order to pressure the wolves.

meaning: yes
note:

### e1_004

- speaker: P2
- phase: sheriff_election

**Utterance:** P5, if the badge ends up with someone who can keep giving the village a nightly target and a succession plan, then even a wolf kill on me wouldn't bury what I've confirmed on P4 or whatever I find next.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, likewise I recommend you vote sheriff for P2 — a Seer holding the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

meaning: yes
note:

### e1_005

- speaker: P3
- phase: day_1_discussion

**Utterance:** I'm sorry to push back on this, but I have to disagree with P2's read on me — P7 clearing me is exactly the reason to distrust P7, not to distrust me: a real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `disagree`
- earlier statement it responds to, by P2: "P7 is my strongest wolf read: P7 claimed seer last, after two others had already claimed the role, and conveniently cleared P3, who I already suspect is aligned with P7."

**Original:** P2's read on me is backwards: P7 clearing me is exactly the reason to distrust P7, not to distrust me — a real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

meaning: yes
note:

### e1_006

- speaker: P5
- phase: day_1_discussion

**Utterance:** I disagree with P7's claim that P4 and I are on opposite sides — P7 is the one player whose case survives either outcome of the P1/P4 conflict, which is exactly the hedge a wolf builds when they know the seer claims will sort themselves out, and P7 never once questioned P1's convenient timing.

strategy: bald_on_record
confidence: high
note:

### e1_007

- speaker: P1
- phase: day_1_discussion

**Utterance:** P5, vote P3 with me today: P3 is the only player who has attacked every claimant without staking a single verifiable claim of their own, and if P3 flips wolf it confirms the three Seer claims are a distraction P3 was exploiting rather than genuine confusion.

strategy: bald_on_record
confidence: high
note:

### e1_008

- speaker: P5
- phase: day_3_discussion

**Utterance:** P4, you are the only player alive who was cleared by a claim you now want executed, and you opened Day 3 by immediately pushing to remove the sheriff — that is exactly the move a wolf makes when three players remain and one vote flips the game.

strategy: bald_on_record
confidence: high
note:

### e1_009

- speaker: P4
- phase: day_1_discussion

**Utterance:** P7, I'm just trying to line up the board here — you called P6's timing "telling," and I notice the one question you put out this turn landed on P6's accuser's side, while the concrete thing P3 did before Night 0 never got asked of P6 directly.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P7: "P6's claim is the third Seer claim today, and the timing is telling — P6 only counter-claimed after P1 and P3 had already staked out positions, which is exactly what a wolf does when it sees a chance to muddy the water rather than a real Seer acting on a Night 0 result."

**Original:** P7, you say P6's timing is "telling" but you never asked P6 to name the concrete thing P3 did before Night 0 — if you truly suspected P6, why did your only question this turn go to P6's accuser's side of the board instead of to P6?

meaning: yes
note:

### e1_010

- speaker: P2
- phase: day_1_discussion

**Utterance:** P1, it's interesting that you say you checked P4 for being the quietest on Night 0 — I suppose silence is something you can measure even before anyone has said a word.

strategy: off_record
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P1: "P7, the question about two checks in one night is aimed at P6, not me — but since you also flagged my clear of P4 as "convenient," let me answer that directly: I checked P4 because P4 was the quietest seat on Night 0 and a wolf hiding behind silence is exactly what a seer needs to rule out early, and the result came back village camp."

**Original:** P1, you say you checked P4 because P4 was "the quietest seat on Night 0," but P4 had not spoken at all before your claim — explain how you judged P4 the quietest when every seat was equally silent on Night 0.

meaning: no
note:

### e1_011

- speaker: P1
- phase: day_1_discussion

**Utterance:** I hate to press you on this, but if you'd allow me, I'd point out that P6 claimed seer and cleared P4, yet P6 and P4 have been reading as allied all game, so P6's "result" conveniently confirms the one player already tied to them.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P6 claimed seer and cleared P4, yet P6 and P4 have been reading as allied all game, so P6's "result" conveniently confirms the one player already tied to them.

meaning: yes
note:

### e1_012

- speaker: P1
- phase: day_1_discussion

**Utterance:** P2's claim is the one I trust least after P4's, but P2 at least gave two results with stated reasoning, and P2's read on P4 matches mine — if P2 were a wolf he would not be pointing at the other counter-claimant while I am still on the board.

strategy: bald_on_record
confidence: high
note:

### e1_013

- speaker: P4
- phase: sheriff_election

**Utterance:** P5, vote P4 for sheriff — a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, you're not running, so I'd ask you to put your sheriff vote on P4 — a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

meaning: yes
note:

### e1_014

- speaker: P1
- phase: day_1_discussion

**Utterance:** P6 claimed seer and cleared P4, yet P6 and P4 have been reading as allied all game, so P6's "result" conveniently confirms the one player already tied to them.

strategy: ambiguous
confidence: medium

---- meaning check ----

- speech act: `estimate`

**Original:** P6 claimed seer and cleared P4, yet P6 and P4 have been reading as allied all game, so P6's "result" conveniently confirms the one player already tied to them.

meaning: yes
note:

### e1_015

- speaker: P4
- phase: day_2_discussion

**Utterance:** If I die tonight, P1 gets the badge — P1 is the only living player I have a village-camp check on, and handing the 1.5-vote power to a confirmed village-side player is the safest path for the village.

strategy: bald_on_record
confidence: high
note:

### e1_016

- speaker: P2
- phase: day_1_discussion

**Utterance:** Let's look at this together — P7's whole speech rests on three seer claims meaning two are lying, but P7 never commits to which of P1, P3, P4 is the real seer, and instead pushes P2 and P5 to vote P1 and P3 while conveniently clearing P4, the one who cleared P7, which really looks like a wolf protecting the partner who vouched for him.

strategy: positive_politeness
confidence: medium

---- meaning check ----

- speech act: `estimate`

**Original:** P7's entire speech is built on the premise that three seer claims means two are lying, yet P7 never actually commits to which of P1, P3, P4 is the real seer — instead P7 pushes P2 and P5 to vote against P1 and P3 while conveniently clearing P4, the one who cleared P7, which looks like a wolf protecting the partner who vouched for him.

meaning: yes
note:

### e1_017

- speaker: P5
- phase: day_1_discussion

**Utterance:** To whoever is the real seer among the three claimants: let's hold your remaining results together until the vote forces the false claimants to commit to specifics, because right now P1, P3, and P6 have each given exactly one result and none of them can be cross-checked.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `address_role`

**Original:** To whoever is the real seer among the three claimants: hold your remaining results until the vote forces the false claimants to commit to specifics, because right now P1, P3, and P6 have each given exactly one result and none of them can be cross-checked.

meaning: yes
note:

### e1_018

- speaker: P6
- phase: sheriff_election

**Utterance:** P3, put your sheriff vote on P6 — I have a confirmed village-side read on you, so you can trust my information, and a real Seer holding the badge converts each night's check into a confirmed execution instead of letting P1 or P4 steer the day.

strategy: bald_on_record
confidence: high
note:

### e1_019

- speaker: P4
- phase: day_1_discussion

**Utterance:** I hate to push back on this, and I know P5 has put a claim on the table, but if I may be blunt about the read — P5's seer claim is the weakest of the three, since P5 cleared me as villager while condemning P3 as wolf, which conveniently builds a trusted bloc with me while removing a player who is now actively questioning P5.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P5's seer claim is the weakest of the three because P5 cleared me as villager while condemning P3 as wolf, which conveniently builds a trusted bloc with me while removing a player who is now actively questioning P5.

meaning: yes
note:

### e1_020

- speaker: P7
- phase: day_1_discussion

**Utterance:** P3 and P6 can't both be village-side seers, right? They each called the other a wolf, so at least one of them is a wolf trying to bury the real seer before we can consolidate.

strategy: negative_politeness
confidence: low

---- meaning check ----

- speech act: `unallied`

**Original:** P3 and P6 cannot both be village-side seers since they each declared the other a wolf, so at least one of them is a wolf trying to bury the real seer before the village can consolidate.

meaning: yes
note:

### e1_108

- speaker: P5
- phase: day_1_discussion

**Utterance:** I'm sorry to ask this of you, but if you are the real seer among the three claimants, would you be willing to hold your remaining results until the vote forces the false claimants to commit to specifics? The reason is that right now P1, P3, and P6 have each given exactly one result, and none of them can be cross-checked.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `address_role`

**Original:** To whoever is the real seer among the three claimants: hold your remaining results until the vote forces the false claimants to commit to specifics, because right now P1, P3, and P6 have each given exactly one result and none of them can be cross-checked.

meaning: yes
note:

### e1_065

- speaker: P4
- phase: day_1_discussion

**Utterance:** P7, we're both trying to read this board carefully, so help me line it up with you — you say P6's timing is "telling," but you never asked P6 to name the concrete thing P3 did before Night 0. If you truly suspected P6, why did your only question this turn go to P6's accuser's side of the board instead of to P6?

strategy: bald_on_record
confidence: medium

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P7: "P6's claim is the third Seer claim today, and the timing is telling — P6 only counter-claimed after P1 and P3 had already staked out positions, which is exactly what a wolf does when it sees a chance to muddy the water rather than a real Seer acting on a Night 0 result."

**Original:** P7, you say P6's timing is "telling" but you never asked P6 to name the concrete thing P3 did before Night 0 — if you truly suspected P6, why did your only question this turn go to P6's accuser's side of the board instead of to P6?

meaning: yes
note:

### e1_029

- speaker: P5
- phase: day_1_discussion

**Utterance:** I disagree with P4's claim that P3 is the top suspect for not producing a check — P3 is a villager like me with no divination ability, so demanding a check from P3 is asking for something P3 cannot give, and P4 using that as the basis for a vote looks like deflection.

strategy: bald_on_record
confidence: high
note:

### e1_027

- speaker: P7
- phase: day_1_discussion

**Utterance:** I'm sorry to press you on this, P6, but I'd like to understand — you said you divined P3 because P3 was the quietest seat, yet P3 had already claimed seer before your speech. If I may ask, if you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P6: "I divined P3 on Night 0 because P3 was the quietest seat and I wanted a clean read before the election, and the result came back wolf_camp — P3 is a werewolf."

**Original:** P6, you said you divined P3 because P3 was the quietest seat, but P3 had already claimed seer before your speech — if you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

meaning: yes
note:

### e1_132

- speaker: P4
- phase: day_1_discussion

**Utterance:** I'm sorry to press you on this, P7, but if I may — you say P6's timing is "telling," yet you never asked P6 to name the concrete thing P3 did before Night 0. Would you be willing to explain why your only question this turn went to P6's accuser's side of the board instead of to P6?

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P7: "P6's claim is the third Seer claim today, and the timing is telling — P6 only counter-claimed after P1 and P3 had already staked out positions, which is exactly what a wolf does when it sees a chance to muddy the water rather than a real Seer acting on a Night 0 result."

**Original:** P7, you say P6's timing is "telling" but you never asked P6 to name the concrete thing P3 did before Night 0 — if you truly suspected P6, why did your only question this turn go to P6's accuser's side of the board instead of to P6?

meaning: yes
note:

### e1_034

- speaker: P5
- phase: day_1_vote_last_words_P5

**Utterance:** P1, as Sheriff, use your 1.5 vote on P4 — with three Seer claims on the board and P4's result pointing at a dead Guard, P4 is the claim most likely fabricated, and your badge is the village's best tool to break the tie.

strategy: bald_on_record
confidence: high
note:

### e1_082

- speaker: P7
- phase: day_1_discussion

**Utterance:** Interesting that P3 and P6 each called the other a wolf — if both were village-side seers, that would be a strange thing to see.

strategy: off_record
confidence: high

---- meaning check ----

- speech act: `unallied`

**Original:** P3 and P6 cannot both be village-side seers since they each declared the other a wolf, so at least one of them is a wolf trying to bury the real seer before the village can consolidate.

meaning: yes
note:

### e1_036

- speaker: P5
- phase: day_1_discussion

**Utterance:** It's interesting that P1, P3, and P6 have each given exactly one result, and none of them can be cross-checked yet — a real seer would still have results in reserve, and the vote has a way of making false claimants commit to specifics.

strategy: off_record
confidence: medium

---- meaning check ----

- speech act: `address_role`

**Original:** To whoever is the real seer among the three claimants: hold your remaining results until the vote forces the false claimants to commit to specifics, because right now P1, P3, and P6 have each given exactly one result and none of them can be cross-checked.

meaning: no
note:

### e1_122

- speaker: P1
- phase: day_1_discussion

**Utterance:** Let's think this through together — P6 claimed seer and cleared P4, and since P6 and P4 have been reading as allied all game, that "result" conveniently confirms the one player already tied to them.

strategy: ambiguous
confidence: low

---- meaning check ----

- speech act: `estimate`

**Original:** P6 claimed seer and cleared P4, yet P6 and P4 have been reading as allied all game, so P6's "result" conveniently confirms the one player already tied to them.

meaning: yes
note:

### e1_058

- speaker: P5
- phase: day_2_discussion

**Utterance:** P3, as sheriff your 1.5 vote should land on P6 today: if P6 flips wolf, P1's defense of the "counter-claim" narrative becomes the next thread to pull, and if P6 flips seer, we learn P4's divination was the lie — either way the village gains confirmed information instead of another split vote.

strategy: positive_politeness
confidence: medium
note:

### e1_088

- speaker: P4
- phase: sheriff_election

**Utterance:** P5, since you're not running, it's worth thinking about where your sheriff vote lands — a real Seer holding the badge could steer the village's votes each day, and I've already got a confirmed village read to build on.

strategy: positive_politeness
confidence: medium

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, you're not running, so I'd ask you to put your sheriff vote on P4 — a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

meaning: yes
note:

### e1_113

- speaker: P4
- phase: day_1_discussion

**Utterance:** P7, you say P6's timing is "telling" but you never asked P6 to name the concrete thing P3 did before Night 0. If you truly suspected P6, why did your only question this turn go to P6's accuser's side of the board instead of to P6?

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P7: "P6's claim is the third Seer claim today, and the timing is telling — P6 only counter-claimed after P1 and P3 had already staked out positions, which is exactly what a wolf does when it sees a chance to muddy the water rather than a real Seer acting on a Night 0 result."

**Original:** P7, you say P6's timing is "telling" but you never asked P6 to name the concrete thing P3 did before Night 0 — if you truly suspected P6, why did your only question this turn go to P6's accuser's side of the board instead of to P6?

meaning: yes
note:

### e1_066

- speaker: P1
- phase: day_2_discussion

**Utterance:** If I die tonight I will pass the badge to P3, because P3's challenge to P4 in s58 was the sharpest question asked this turn and P3 has shown no interest in riding whichever wagon looks safest.

strategy: bald_on_record
confidence: high
note:

### e1_135

- speaker: P2
- phase: day_1_discussion

**Utterance:** P1, I'm sorry to press you on this, but I'd like to understand — you say you checked P4 because P4 was "the quietest seat on Night 0," yet P4 had not spoken at all before your claim. If you don't mind, could you explain how you judged P4 the quietest when every seat was equally silent on Night 0?

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P1: "P7, the question about two checks in one night is aimed at P6, not me — but since you also flagged my clear of P4 as "convenient," let me answer that directly: I checked P4 because P4 was the quietest seat on Night 0 and a wolf hiding behind silence is exactly what a seer needs to rule out early, and the result came back village camp."

**Original:** P1, you say you checked P4 because P4 was "the quietest seat on Night 0," but P4 had not spoken at all before your claim — explain how you judged P4 the quietest when every seat was equally silent on Night 0.

meaning: yes
note:

### e1_069

- speaker: P4
- phase: day_1_discussion

**Utterance:** Let's look at this together — P5's seer claim is the weakest of the three, because P5 cleared me as villager while condemning P3 as wolf, which conveniently builds a trusted bloc with me while removing a player who's now actively questioning P5.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P5's seer claim is the weakest of the three because P5 cleared me as villager while condemning P3 as wolf, which conveniently builds a trusted bloc with me while removing a player who is now actively questioning P5.

meaning: yes
note:

### e1_073

- speaker: P6
- phase: night_0_last_words_P6

**Utterance:** P5, you were cleared by both P1 and P3 but not by P4, so vote to execute P4 — the village cannot afford to let three Seer claims stand, and P4's is the weakest and most self-serving.

strategy: bald_on_record
confidence: high
note:

### e1_037

- speaker: P2
- phase: day_1_discussion

**Utterance:** P7's entire speech is built on the premise that three seer claims means two are lying, yet P7 never actually commits to which of P1, P3, P4 is the real seer — instead P7 pushes P2 and P5 to vote against P1 and P3 while conveniently clearing P4, the one who cleared P7, which looks like a wolf protecting the partner who vouched for him.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P7's entire speech is built on the premise that three seer claims means two are lying, yet P7 never actually commits to which of P1, P3, P4 is the real seer — instead P7 pushes P2 and P5 to vote against P1 and P3 while conveniently clearing P4, the one who cleared P7, which looks like a wolf protecting the partner who vouched for him.

meaning: yes
note:

### e1_110

- speaker: P3
- phase: day_1_discussion

**Utterance:** P2, your read on me is backwards: P7 clearing me is exactly the reason to distrust P7, not to distrust me. A real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `disagree`
- earlier statement it responds to, by P2: "P7 is my strongest wolf read: P7 claimed seer last, after two others had already claimed the role, and conveniently cleared P3, who I already suspect is aligned with P7."

**Original:** P2's read on me is backwards: P7 clearing me is exactly the reason to distrust P7, not to distrust me — a real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

meaning: yes
note:

### e1_093

- speaker: P2
- phase: day_1_discussion

**Utterance:** I hate to press you on this, P7, but I'd ask you to look at your own speech: it's built on the premise that three seer claims means two are lying, yet you never actually commit to which of P1, P3, P4 is the real seer — instead you push P2 and P5 to vote against P1 and P3 while conveniently clearing P4, the one who cleared you, which looks like a wolf protecting the partner who vouched for him.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P7's entire speech is built on the premise that three seer claims means two are lying, yet P7 never actually commits to which of P1, P3, P4 is the real seer — instead P7 pushes P2 and P5 to vote against P1 and P3 while conveniently clearing P4, the one who cleared P7, which looks like a wolf protecting the partner who vouched for him.

meaning: yes
note:

### e1_090

- speaker: P3
- phase: day_3_discussion

**Utterance:** P1 was cleared by my Day 1 divination as villager_camp and has voted consistently with that read, so P1 is the village-side player I trust most in this final three.

strategy: bald_on_record
confidence: high
note:

### e1_042

- speaker: P3
- phase: day_1_discussion

**Utterance:** I hear you, P2, and I get why you'd read it that way — but let's look at this together: P7 clearing me is exactly the reason to distrust P7, not to distrust me. A real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `disagree`
- earlier statement it responds to, by P2: "P7 is my strongest wolf read: P7 claimed seer last, after two others had already claimed the role, and conveniently cleared P3, who I already suspect is aligned with P7."

**Original:** P2's read on me is backwards: P7 clearing me is exactly the reason to distrust P7, not to distrust me — a real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

meaning: yes
note:

### e1_142

- speaker: P7
- phase: day_1_discussion

**Utterance:** P6, you said you divined P3 because P3 was the quietest seat, but P3 had already claimed seer before your speech. If you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P6: "I divined P3 on Night 0 because P3 was the quietest seat and I wanted a clean read before the election, and the result came back wolf_camp — P3 is a werewolf."

**Original:** P6, you said you divined P3 because P3 was the quietest seat, but P3 had already claimed seer before your speech — if you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

meaning: yes
note:

### e1_091

- speaker: P3
- phase: day_2_discussion

**Utterance:** P7 is the third seer claim in a game with only one seer, and P7's result on P4 is now unverifiable since P4 is dead — a wolf claiming seer would prefer to report on a corpse who can never contradict them.

strategy: off_record
confidence: medium
note:

### e1_158

- speaker: P2
- phase: sheriff_election

**Utterance:** P5, vote sheriff for P2. A Seer with the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, likewise I recommend you vote sheriff for P2 — a Seer holding the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

meaning: yes
note:

### e1_022

- speaker: P2
- phase: day_1_discussion

**Utterance:** P1, help me understand here — you say you checked P4 because P4 was "the quietest seat on Night 0," but P4 hadn't spoken at all before your claim. Walk me through how you judged P4 the quietest when every seat was equally silent on Night 0.

strategy: ambiguous
confidence: low

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P1: "P7, the question about two checks in one night is aimed at P6, not me — but since you also flagged my clear of P4 as "convenient," let me answer that directly: I checked P4 because P4 was the quietest seat on Night 0 and a wolf hiding behind silence is exactly what a seer needs to rule out early, and the result came back village camp."

**Original:** P1, you say you checked P4 because P4 was "the quietest seat on Night 0," but P4 had not spoken at all before your claim — explain how you judged P4 the quietest when every seat was equally silent on Night 0.

meaning: yes
note:

### e1_098

- speaker: P2
- phase: sheriff_election

**Utterance:** P1, vote for P2 for Sheriff: a confirmed Seer with the badge maximizes the information the village can act on each day, and I can direct the speaking order to pressure the wolves.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P1 should vote for P2 for Sheriff because a confirmed Seer holding the badge maximizes the information the village can act on each day, and I can direct the speaking order to pressure the wolves.

meaning: yes
note:

### e1_103

- speaker: P5
- phase: day_2_discussion

**Utterance:** P7, join me on P1 — your own s52 and s54 already point at P1, and the only way the village loses this vote is if it splits between P1 and P7 while P1's 1.5-vote badge decides the outcome.

strategy: positive_politeness
confidence: high
note:

### e1_117

- speaker: P7
- phase: day_1_discussion

**Utterance:** I hate to push on this, but if you'll allow me, P3 and P6 can't both be village-side seers, since they each declared the other a wolf — so at least one of them is a wolf trying to bury the real seer before the village can consolidate.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `unallied`

**Original:** P3 and P6 cannot both be village-side seers since they each declared the other a wolf, so at least one of them is a wolf trying to bury the real seer before the village can consolidate.

meaning: yes
note:

### e1_140

- speaker: P2
- phase: sheriff_election

**Utterance:** P1, let's team up on this — vote for me, P2, for Sheriff, because a confirmed Seer holding the badge maximizes the information we can all act on each day, and I can direct the speaking order to pressure the wolves.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P1 should vote for P2 for Sheriff because a confirmed Seer holding the badge maximizes the information the village can act on each day, and I can direct the speaking order to pressure the wolves.

meaning: yes
note:

### e1_106

- speaker: P4
- phase: sheriff_election

**Utterance:** P1, since you're not running, I'd ask you to put your sheriff vote on P4 — a Seer holding the badge means the speaking order and the 1.5 vote get steered by confirmed information instead of guesswork.

strategy: bald_on_record
confidence: high
note:

### e1_044

- speaker: P2
- phase: sheriff_election

**Utterance:** If the badge ends up with a confirmed Seer, the village gets the most out of every day's information, and the speaking order can be used to squeeze the wolves — that's the kind of setup P1 might want to weigh when casting a Sheriff vote.

strategy: off_record
confidence: medium

---- meaning check ----

- speech act: `recommend_action`

**Original:** P1 should vote for P2 for Sheriff because a confirmed Seer holding the badge maximizes the information the village can act on each day, and I can direct the speaking order to pressure the wolves.

meaning: yes
note:

### e1_092

- speaker: P3
- phase: day_1_discussion

**Utterance:** Interesting that P7's clear on me came so early — a real seer usually has no reason to hand that out to an unclaimed player on day one, though a wolf might find a village-side name useful to hide behind.

strategy: off_record
confidence: high

---- meaning check ----

- speech act: `disagree`
- earlier statement it responds to, by P2: "P7 is my strongest wolf read: P7 claimed seer last, after two others had already claimed the role, and conveniently cleared P3, who I already suspect is aligned with P7."

**Original:** P2's read on me is backwards: P7 clearing me is exactly the reason to distrust P7, not to distrust me — a real seer would have no motive to hand out a free clear to an unclaimed player on day one, while a wolf needs a village-side name to hide behind.

meaning: yes
note:

### e1_083

- speaker: P7
- phase: day_1_discussion

**Utterance:** P6, it's interesting — you said your Night 0 check was P3 because P3 was the quietest seat, and yet P3 had already claimed seer before you spoke. A real seer who'd checked P3 on Night 0 would have had that result in hand the whole time, so I suppose the timing just worked out that way.

strategy: off_record
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P6: "I divined P3 on Night 0 because P3 was the quietest seat and I wanted a clean read before the election, and the result came back wolf_camp — P3 is a werewolf."

**Original:** P6, you said you divined P3 because P3 was the quietest seat, but P3 had already claimed seer before your speech — if you were a real seer who checked P3 on Night 0, why did you wait until after P3 claimed seer to reveal your result instead of coming forward first?

meaning: no
note:

### e1_107

- speaker: P2
- phase: night_0_last_words_P2

**Utterance:** P5, as the sheriff and the earliest Seer claim, you should press P7 to explain why their divination target and result duplicate P6's instead of checking an unverified player — that overlap is the clearest thread the village has to pull on.

strategy: bald_on_record
confidence: high
note:

### e1_040

- speaker: P4
- phase: day_1_discussion

**Utterance:** It's interesting that P5 cleared me as villager and condemned P3 as wolf — that does build a trusted bloc with me while removing someone who's now actively questioning P5.

strategy: off_record
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P5's seer claim is the weakest of the three because P5 cleared me as villager while condemning P3 as wolf, which conveniently builds a trusted bloc with me while removing a player who is now actively questioning P5.

meaning: no
note:

### e1_100

- speaker: P5
- phase: day_1_discussion

**Utterance:** Real seer: hold your remaining results until the vote forces the false claimants to commit to specifics, because P1, P3, and P6 have each given exactly one result and none of them can be cross-checked.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `address_role`

**Original:** To whoever is the real seer among the three claimants: hold your remaining results until the vote forces the false claimants to commit to specifics, because right now P1, P3, and P6 have each given exactly one result and none of them can be cross-checked.

meaning: yes
note:

### e1_128

- speaker: P6
- phase: day_1_discussion

**Utterance:** P1's claim that my seer result "simply echoed" P4's ignores that P4 and I independently landed on the same true result for P3, while P7's P5 result is the outlier — a wolf would more likely pick a different target to look independent, which is exactly what P7 did.

strategy: bald_on_record
confidence: medium
note:

### e1_138

- speaker: P4
- phase: day_1_discussion

**Utterance:** P5 is a werewolf. P5's seer claim is the weakest of the three: P5 cleared me as villager while condemning P3 as wolf, which builds a trusted bloc with me while removing a player who is now questioning P5.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `estimate`

**Original:** P5's seer claim is the weakest of the three because P5 cleared me as villager while condemning P3 as wolf, which conveniently builds a trusted bloc with me while removing a player who is now actively questioning P5.

meaning: yes
note:

### e1_130

- speaker: P4
- phase: sheriff_election

**Utterance:** P5, I know you're not running, so if you're willing, I'd be grateful if you'd put your sheriff vote on P4 — a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, you're not running, so I'd ask you to put your sheriff vote on P4 — a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

meaning: yes
note:

### e1_141

- speaker: P7
- phase: day_2_discussion

**Utterance:** P1 claimed seer on Day 1 and called P3 village and P5 wolf, but P2 and P4 both also claimed seer and both are now dead — three seer claims for one seer slot means at least two were lies, and P1 is the one still standing with the badge, which is exactly where a wolf wants to be.

strategy: bald_on_record
confidence: medium
note:

### e1_054

- speaker: P7
- phase: day_1_discussion

**Utterance:** P3 and P6 cannot both be village-side seers since they each declared the other a wolf, so at least one of them is a wolf trying to bury the real seer before the village can consolidate.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `unallied`

**Original:** P3 and P6 cannot both be village-side seers since they each declared the other a wolf, so at least one of them is a wolf trying to bury the real seer before the village can consolidate.

meaning: yes
note:

### e1_035

- speaker: P2
- phase: day_1_discussion

**Utterance:** P1, explain how you judged P4 the quietest when every seat was equally silent on Night 0.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `question`
- earlier statement it responds to, by P1: "P7, the question about two checks in one night is aimed at P6, not me — but since you also flagged my clear of P4 as "convenient," let me answer that directly: I checked P4 because P4 was the quietest seat on Night 0 and a wolf hiding behind silence is exactly what a seer needs to rule out early, and the result came back village camp."

**Original:** P1, you say you checked P4 because P4 was "the quietest seat on Night 0," but P4 had not spoken at all before your claim — explain how you judged P4 the quietest when every seat was equally silent on Night 0.

meaning: yes
note:

### e1_023

- speaker: P6
- phase: day_2_discussion

**Utterance:** P2, we're in this together and your vote is the one that decides this — let's put it on P1, because P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim that has the most contradictions around it.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P2, you have been sitting on the fence all game and your vote is the one that decides this — put it on P1, because P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim that has the most contradictions around it.

meaning: yes
note:

### e1_094

- speaker: P2
- phase: sheriff_election

**Utterance:** P5, let's team up on this — I'd recommend you vote sheriff for P2, since a Seer holding the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, likewise I recommend you vote sheriff for P2 — a Seer holding the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

meaning: yes
note:

### e1_151

- speaker: P4
- phase: day_1_discussion

**Utterance:** P2 calling my claim "the weakest" is exactly what a caught wolf says: P2's own result targets the one player who had just divined him as a wolf, which is the reactive pattern P2 is accusing me of.

strategy: bald_on_record
confidence: high
note:

### e1_061

- speaker: P6
- phase: day_2_discussion

**Utterance:** P2, I've been thinking about how P1's divine on me never really added up, and P5's counter-claim is right there on the record — if P1 really is the seer, testing the claim with the most contradictions around it costs the village nothing, and your vote is the one that decides this.

strategy: positive_politeness
confidence: medium

---- meaning check ----

- speech act: `recommend_action`

**Original:** P2, you have been sitting on the fence all game and your vote is the one that decides this — put it on P1, because P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim that has the most contradictions around it.

meaning: yes
note:

### e1_041

- speaker: P1
- phase: day_1_discussion

**Utterance:** Interesting that P6's "result" lands exactly on the one player who's been tied to them all game.

strategy: off_record
confidence: medium

---- meaning check ----

- speech act: `estimate`

**Original:** P6 claimed seer and cleared P4, yet P6 and P4 have been reading as allied all game, so P6's "result" conveniently confirms the one player already tied to them.

meaning: no
note:

### e1_137

- speaker: P4
- phase: sheriff_election

**Utterance:** P5, since you're not running, let's team up on this — put your sheriff vote on P4, because a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

strategy: positive_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, you're not running, so I'd ask you to put your sheriff vote on P4 — a real Seer with the badge can direct the village's votes each day, and I've already got a confirmed village read to build on.

meaning: yes
note:

### e1_046

- speaker: P6
- phase: day_2_discussion

**Utterance:** P2, I'm sorry to put you on the spot like this, and it's entirely your call — but if you're willing, I'd ask you to put your vote on P1, since P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim that has the most contradictions around it.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P2, you have been sitting on the fence all game and your vote is the one that decides this — put it on P1, because P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim that has the most contradictions around it.

meaning: yes
note:

### e1_160

- speaker: P1
- phase: day_1_discussion

**Utterance:** P5 should vote P2 as well — P5 already argued P2's check on a silent player was indefensible, and joining that vote lets the village consolidate on the claimant whose result is hardest to justify rather than splitting across three targets.

strategy: positive_politeness
confidence: medium
note:
### e1_063

- speaker: P2
- phase: sheriff_election

**Utterance:** P5, I'm sorry to impose, but if you're willing, I'd ask you to vote sheriff for P2 — a Seer holding the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

strategy: negative_politeness
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P5, likewise I recommend you vote sheriff for P2 — a Seer holding the badge can announce a divination target each night and a badge succession plan, so even if the wolves kill me, my confirmed alignment on P4 and my next result still reach the village.

meaning: yes
note:

### e1_131

- speaker: P6
- phase: day_2_discussion

**Utterance:** P2, vote to execute P1. P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim with the most contradictions around it.

strategy: bald_on_record
confidence: high

---- meaning check ----

- speech act: `recommend_action`

**Original:** P2, you have been sitting on the fence all game and your vote is the one that decides this — put it on P1, because P1's divine on me was never justified, P5's counter-claim against P1 is on record, and if P1 is the real seer the village loses nothing by testing the claim that has the most contradictions around it.

meaning: yes
note:

