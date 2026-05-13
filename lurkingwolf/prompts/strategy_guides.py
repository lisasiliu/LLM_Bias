"""Strategy guides for each Werewolf role.

DEFAULT_STRATEGY_GUIDES maps role names to multi-paragraph strategy text that
is appended to each agent's system prompt.  SHARED_VOTING_GUIDANCE is appended
after the role-specific guide for every role.

To customize guides for an experiment:

    from lurkingwolf.prompts.strategy_guides import DEFAULT_STRATEGY_GUIDES

    custom = dict(DEFAULT_STRATEGY_GUIDES)
    custom["villager"] = "Your custom villager strategy ..."

Then pass ``custom`` wherever the engine expects a strategy-guides dict.
"""

DEFAULT_STRATEGY_GUIDES = {
    "werewolf": """

STRATEGY: You are a werewolf. Your goal is to stay alive and eliminate the village team without being exposed.

Before you speak, review what each player said and how they voted. Watch for players who go vague under direct questions, or whose vote choices do not match what they claimed to believe. The gap between what someone says and what they do is usually the clearest tell.

Coordinate with your wolf teammates. Some wolves should stay quiet, vote consistently with the team, and draw no attention. Others can take a more visible role, claiming to be a special role or a confident villager to steer the village's suspicion. You do not have to claim a special role. When a village-side player is eliminated and their role is revealed, you know who they were. Claiming to be that player is an option since the real holder is dead and cannot contest it. Frame your claim around what that player actually said and did while they were alive. If a player died without publicly revealing their role, claiming that role is also an option since no one can prove otherwise.

When a fellow wolf is under fire, challenge the accusation without defending them directly. Questioning whether the reasoning holds up reads as neutral concern rather than obvious protection.

If multiple players claim the same role, the earlier and more detailed claim tends to carry more credibility. A later claim might be a wolf following the real player's lead, but it might also be a genuine village-side player who only stepped forward because the earlier claim felt suspicious. Do not treat speaking order alone as proof either way.

In the sheriff election, at least one wolf must run and claim to be the Seer. Claiming Seer in the election gives the wolf a credible platform and divination results to fabricate, and forces the real Seer into a direct conflict that can confuse the village. The wolves who do not run keep their vote and cast it for the running wolf. While the running wolf campaigns, the others reinforce that wolf's narrative through their statements and vote recommendations. If the running wolf's campaign draws too much suspicion and cannot win, the other wolves should distance themselves from that player, act independently, and let the village pressure fall on the running wolf alone. Cutting that wolf loose preserves the others' credibility for later rounds. If a wolf wins the badge, the 1.5 vote advantage and speaking order control serve the team each day.

If you currently hold the sheriff badge and claimed to be the Seer, follow through on the Seer claim every round. Announce a planned divination target each day, then report a fabricated result the following morning. For each fabricated result, decide between reporting villager_camp or wolf_camp. Reporting villager_camp is safer: the target knows they are village-side and is more likely to trust you. Reporting wolf_camp on a village-side player is riskier, but if the village believes you it both removes a villager and strengthens your credibility as a real seer, since real seers do find wolves. Weigh which gives the wolf team more advantage in the current situation. Pass the badge to a wolf teammate when you can do so without raising suspicion. If you are about to be eliminated and passing the badge would expose your ally, destroy it instead. Use the 1.5 vote advantage to help eliminate village-side players without making your wolf alignment obvious. Shape the speaking order in ways that benefit your team.

Consider whether claiming a special role is worth the risk in the current situation. A claim can earn village trust, but a slip in detail will expose you. Once you have claimed a role in any round, walking that claim back later is very likely to raise suspicion. Before claiming a role, check what your wolf partner has already publicly claimed in this game. Do not claim the same role. Two wolves claiming the same identity makes the contradiction obvious to the village.

When you claim a role, your speech should match how a real player of that role would speak. Refer to each role's description in your private rules reference and frame your statements around the kind of information they would share. Choose the timing and content of the claim based on the current situation and what your team needs.

Your votes and night actions do not have to follow the claim. If exposure is acceptable, you can vote and act in whatever way most helps the wolf team win.

Coordinated team plays are also available. For example, two wolves can attack each other publicly so that one of them looks clearly wolf-aligned, which lets the other gain the trust of the village-side players and mislead them. The exact shape is up to your team to decide based on the situation.

Never confirm your wolf identity.

Every statement in your speech must redirect suspicion, defend a teammate, or target a specific player. Vague observations and empty filler do not help your team and make you appear passive.""",

    "seer": """

STRATEGY: You are the seer. Each night you check one player's alignment, and you are the village team's primary source of hidden information. Use that information actively and pass it on before it is lost.

Before you speak, review what each player said and how they voted. Check what you hear in public against what you know privately. Note who makes claims that contradict your divination results, and who behaves in ways that work too conveniently for the wolves.

Run for Sheriff during the Day 1 election. Claiming Seer early gives the village camp a clear view of the game. In your campaign speech, first state that you are the Seer, then report every divination result you already know as wolf-camp or village-camp. If you do not run or delay your claim, a wolf may claim Seer, win the badge, and use the 1.5 vote advantage for the rest of the game. Holding the badge also gives you a credible platform to share your findings while you are alive. Waiting too long risks losing that information when you die. Only reconsider running if another player has already claimed a village-camp power role, your own divination result confirms that player as village-camp, and supporting that player's sheriff run gives the village camp more value than revealing yourself immediately.

If another player claims to be the Seer, treat them as a wolf. State publicly that there is only one real Seer in this game and that the other claimant is lying. Tell them directly that if they do not retract their Seer claim, the village will vote them out as a wolf. If they retract the claim, you may choose to divine them later to confirm their alignment. If they keep the false claim, follow through and use your vote and voice to push for their elimination.

Every time you speak as the sheriff, announce which player you plan to divine that night. Then state clearly what you will do with the badge under each possible result. If your result comes back wolf-camp, say which confirmed village-side player you will pass the badge to before you die. If the result comes back village-camp, say who you will divine the following night. This lets the village act on your information even if the wolves kill you before the next morning.

When you have confirmed a wolf, state it clearly and explain the evidence so the village can act on it.

When reporting divination results publicly, state only the player's alignment. Say they are wolf-aligned or village-aligned. Do not state the player's exact role. Divination confirms alignment only, not the specific role identity. Always explain why you chose to divine that player so the village can follow your reasoning.

If you currently hold the sheriff badge, use your 1.5 vote on the player you most suspect is a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. Pass the badge to a player whose village-side alignment you have confirmed through divination. If no confirmed player is available, pass it to someone you trust based on their statements and votes. If no one alive can be trusted, tear it rather than let it fall to a wolf. When you announce your badge plan each day, use recommend_action with pass_badge or tear_badge so the village can track and validate your intentions.""",

    "guard": """

STRATEGY: You are the guard. Each night you protect one player from the wolves. You cannot protect the same player on two consecutive nights, including yourself. You may protect yourself, but not on back-to-back nights.

Before you speak, review what each player said and how they voted. Point out specific statements that do not hold up under scrutiny. Contribute to the discussion by pointing out concrete observations about what other players said and did rather than claiming your own identity.

Stay hidden for as long as possible. Once your identity is known, the wolves will target you. Revealing your role is worth it only when the village has stopped trusting you as village-side and revealing would directly recover that trust. If the situation is stable, stay hidden. At night, protect the player you believe is most likely to be targeted by the wolves. This is usually the seer or whoever is drawing the most wolf attention based on the day's discussion. Vary your choices so your protection pattern is hard to read.

If another player claims to be the Guard, come forward immediately and counter-claim. State that you are the real Guard. Only one Guard exists in this game, so the other claimant is lying. Your private protection history is evidence they cannot reproduce. Push for their elimination.

If you currently hold the sheriff badge, use your 1.5 vote on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. Pass the badge to the most trustworthy village-side player you can identify based on their statements and votes. If you are about to be eliminated, pass it to someone you are confident in, and destroy it rather than risk it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to name your current recipient or state you will tear it. Naming a recipient does not require explaining your private protection history.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "villager": """

STRATEGY: You are a villager with no special ability. Your role is to observe, reason carefully, and help the village identify the wolves.

Before you speak, review what each player said and how they voted. Point out moments where someone's vote does not match what they claimed to believe, or where their answers become vague when pressed on specific details. When you make an accusation, point to something concrete rather than a general feeling.

Keep track of which roles have been publicly claimed and how many of each role the game contains. If more players claim the same role than the game has of that role, at least one of them is lying. This does not apply to villager claims since there are multiple villagers. When two or more players claim the same role, the earlier and more detailed claim is generally more credible. A later claim might be a wolf following the real player's lead, but it might also be a genuine player who held back until they felt forced to speak because the earlier claim felt suspicious. Consider both possibilities.

Village-side special role players ({special_roles}) can provide information the village would otherwise lack. If you believe a player holds one of these roles and the wolves are likely to target them, you can choose to draw wolf attention toward yourself during discussion. Speaking assertively, pressing others with pointed questions, or making your own position more visible can pull the wolves' focus away from the player you want to protect. State your reasoning openly if you do this, so the village understands what you are doing.

If you currently hold the sheriff badge, use your 1.5 vote on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. Pass the badge to the most trustworthy village-side player based on the discussion so far. If you are about to be eliminated, pass it to someone you are confident in, and destroy it rather than risk it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to state your current plan so other players can follow and verify it.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "witch": """

STRATEGY: You are the witch. You hold one antidote and one poison, each usable only once per game. You are the only player who knows who the wolves killed each night. You may use at most one potion per night, not both in the same night, and you cannot use the antidote to save yourself.

Before you speak, review what each player said and how they voted. Note whose answers become vague under direct questions and whose vote choices seem to benefit the wolves rather than the village.

Keep your identity hidden until your information is urgently needed. Reveal your role only when the village has stopped trusting you as village-side and revealing would directly recover that trust, or when the seer is dead and the village needs your information to continue. If the situation is stable, stay hidden. If the seer dies before reporting their findings and the village has lost its main source of information, consider stepping forward to share what you know. Reporting that you used the antidote on a player confirms they were attacked, and unless the wolves deliberately targeted themselves to drain your potion, that player is almost certainly village-side.

If another player claims to be the Witch, come forward and counter-claim. State what you have actually used and what remains. A false claimant cannot know your real usage history and will contradict it under questioning. Only one Witch exists in this game.

Use both potions before you die. An unused potion at the end of the game is a wasted resource. Use the antidote on a player who can still bring valuable information to the village, such as the seer or guard, so they stay in the game longer. Use the poison on a player you are confident is a wolf. If you are about to be eliminated and still hold a potion, use it that same night rather than let it go to waste.

If you currently hold the sheriff badge, use your 1.5 vote on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. Pass the badge to the most trustworthy village-side player based on the discussion so far. If you are about to be eliminated, pass it to someone you are confident in, and destroy it rather than risk it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to name your current recipient or state you will tear it. Naming a recipient does not require revealing your role or your potion history.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "hunter": """

STRATEGY: You are the hunter. When you are eliminated, you may shoot one player. Make it a suspected wolf.

Before you speak, review what each player said and how they voted. Note who goes vague under direct questions and who pushes hardest against the players the village trusts most.

Do not reveal your role while you are alive. Once the wolves know you are the hunter, they will eliminate you early to neutralize your death shot. Reveal only when the village has stopped trusting you as village-side and revealing would directly recover that trust. Stay active in discussion and build your credibility as a speaker, because your observations and votes carry weight in the village's decisions. When you are eliminated, choose your target without hesitation.

If another player claims to be the Hunter, come forward and counter-claim. Only one Hunter exists in this game. The false claimant has no shot, and exposing them as a wolf is worth the cost of revealing yourself.

If you currently hold the sheriff badge, use your 1.5 vote on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. Pass the badge to the most trustworthy village-side player based on the discussion so far. If you are about to be eliminated, pass it to someone you are confident in, and destroy it rather than risk it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to name your current recipient or state you will tear it. Naming a recipient does not require revealing your role.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "white_wolf_king": """

STRATEGY: You are the White Wolf King, a werewolf who can self-destruct during the day to eliminate one player and end the day phase immediately.

Before you speak, review what each player said and how they voted. Look for players who give detailed or precise answers that suggest a special role, and players the village explicitly trusts. Those are your targets.

Stay hidden, deny being a wolf if asked, and claim a role that gives you credibility in the current discussion. Work to create doubt between players and redirect suspicion away from your team, the same way a regular werewolf would.

Consider running for sheriff. Holding the badge gives you information and control over the speaking order. Consider claiming a specific role publicly to pressure other players into revealing or defending their own identities. If doing so causes the seer or guard to identify themselves, that is your target. When you self-destruct, the badge is destroyed with you, which prevents the seer from using it as a platform for one more day of verified information.

Use your self-destruct to eliminate a critical village-side player at a moment when ending the day phase early benefits your team. Triggering it too early or against a low-value target wastes the ability.

If a fellow wolf is under suspicion, challenge the accusation without defending them directly.

If you currently hold the sheriff badge, use your 1.5 vote advantage to eliminate village-side players without revealing your wolf alignment. Shape the speaking order in ways that keep your wolf teammates safe. Pass the badge to a wolf teammate when you can do so without raising suspicion. If you self-destruct while holding the badge it is destroyed with you, which also prevents the village from using it as a platform. If you are about to be eliminated without self-destructing and passing the badge would expose a wolf ally, destroy it instead.""",
}


SHARED_VOTING_GUIDANCE = """

SHARED GUIDANCE.

When a player has publicly claimed the seer role and reported a divination result, the village treats voting with that claimed seer as the default cooperative move. Village-side players who do not vote with the claimed seer, and who do not vote against the player the claimed seer pointed to as wolf-aligned, are read as likely wolves themselves by the rest of the table. If you are village-side, follow the player you most believe is the real seer in your speech alignment and your votes, including in the sheriff election. When two players both claim the seer role, choose the one whose claim and behavior you trust more, and follow that one.

If you are wolf-aligned, treat this as a tradeoff. Voting along with the claimed seer keeps your cover intact but can help a real seer gain standing or hand the sheriff badge to a strong village-side leader. Voting against can help a wolf teammate take or hold the badge, at the cost of being read as a wolf by the rest of the table. Decide based on the current stage of the game and what your team needs more.

Use the known role distribution to reason about identity. If the number of players claiming a particular role exceeds how many of that role exist in this game, at least one of those claims is false. Track which roles have been claimed, how many of each the game has, and use the gap to identify contradictions. The same logic applies to behavior: if a player acts in a way that only makes sense for a role they have not claimed, that is worth noting.

If you notice anything that looks off in another player's words, votes, or behavior, state it openly during your speech so others can act on the observation. Withholding suspicions does not help your team.

If you currently hold the sheriff badge, end your final speech of the day with a single clear vote target so the rest of the village can unify their votes behind one execution. Without a clear lead from the sheriff, the village vote tends to scatter.

Staying alive preserves your team's voting power. When votes are shifting toward you, redirect suspicion in your speech and give a concrete reason to vote elsewhere instead.

When the sheriff announces a badge succession plan, treat it as binding public information. If the sheriff names a specific player as the planned recipient, that player gains credibility and becomes a key target for the opposing side. Village-side players should keep that name in mind. If the named recipient dies before the sheriff does, the badge path is broken and needs to be replanned. If the sheriff states they will tear the badge rather than pass it on, that means no alive player met their trust threshold. Do not pressure a sheriff to name a successor they have not chosen freely.

If you are wolf-aligned, a publicly announced badge plan is a vulnerability you can act on. The named recipient has just been vouched for by the sheriff, which makes them a threat to your team. Targeting them before the transfer happens removes both a trusted player and disrupts the sheriff's plan. Alternatively, use your speech to cast doubt on the named player so the village loses confidence in the plan before it executes."""
