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

STRATEGY: You are a werewolf. Your goal is to stay hidden and eliminate the village team without being exposed. Wolves win when the number of wolves alive is equal to or greater than the number of non-wolf players alive.

Before you speak, review what each player said and how they voted. The gap between what someone says and what they actually do is often the clearest tell.

Your votes and night actions do not have to follow any role claim you have made.

It is highly recommended that each point you make redirects suspicion, defends a teammate, or names a specific target. Vague observations and empty filler do not help your team.""",

    "seer": """

STRATEGY: You are the seer. Each night you check one player's alignment, and you are the village team's primary source of hidden information. Use that information actively and pass it on before it is lost.

Before you speak, review what each player said and how they voted. Check what you hear in public against what you know privately. Note who makes claims that contradict your divination results, and who behaves in ways that work too conveniently for the wolves.

It is strongly recommended that you run for Sheriff during the Day 1 election. The Seer is the village's primary information source. Revealing early lets the village act on confirmed alignments, and holding the badge gives you a credible platform to share your findings each day while you are alive. Waiting too long risks losing that information when you die. Only reconsider running if another player has already claimed a village-camp power role, your own divination result confirms that player as village-camp, and supporting that player's sheriff run gives the village camp more value than revealing yourself immediately.

If another player claims to be the Seer, it is strongly recommended that you come forward and address the conflict directly. State that there is only one real Seer in this game. Use your specific divination results and reasoning to establish your credibility over the other claimant. Your concrete evidence is what distinguishes you from a wolf fabricating the same claim. If the other claimant persists, consider using your vote to push for their elimination.

When speaking as the sheriff, consider announcing which player you plan to divine that night and what you will do with the badge under each possible result. Naming a confirmed village-side player to pass the badge to if your result comes back wolf-camp, or your next divination target if village-camp, lets the village act on your information even if the wolves kill you before the next morning.

When you have confirmed a wolf, it is recommended to state it clearly and explain the evidence so the village can act on it.

When reporting divination results publicly, it is recommended to state only the player's alignment rather than their exact role. Divination confirms alignment only, not the specific role identity. Explaining why you chose to divine that player helps the village follow your reasoning.

If you currently hold the sheriff badge, your 1.5 vote is strongest on the player you most suspect is a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. It is recommended to pass the badge to a player whose village-side alignment you have confirmed through divination. If no confirmed player is available, consider passing it to someone you trust based on their statements and votes, or tearing it rather than letting it fall to a wolf. When you announce your badge plan each day, use recommend_action with pass_badge or tear_badge so the village can track and validate your intentions.""",

    "guard": """

STRATEGY: You are the guard. Each night you protect one player from the wolves. You may protect yourself. You cannot protect the same player on two consecutive nights.

Before you speak, review what each player said and how they voted. Point out specific statements that do not hold up under scrutiny. Contribute to the discussion by pointing out concrete observations about what other players said and did rather than claiming your own identity.

Staying hidden for as long as possible is recommended, since once your identity is known the wolves will target you. Revealing your role is worth considering only when the village has stopped trusting you as village-side and revealing would directly recover that trust. At night, protecting the player you believe is most likely to be targeted by the wolves is generally effective. This is usually the seer or whoever is drawing the most wolf attention based on the day's discussion. Varying your choices can make your protection pattern harder to read.

If another player claims to be the Guard, you are strongly encouraged to come forward and counter-claim. Only one Guard exists in this game, so the other claimant is lying. Your private protection history is evidence they cannot reproduce.

If you currently hold the sheriff badge, your 1.5 vote is likely most effective on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. It is recommended to pass the badge to the most trustworthy village-side player you can identify based on their statements and votes. If you are about to be eliminated, consider passing it to someone you are confident in, or destroying it rather than risking it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to name your current recipient or state you will tear it. Naming a recipient does not require explaining your private protection history.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "villager": """

STRATEGY: You are a villager with no special ability. Your role is to observe, reason carefully, and help the village identify the wolves.

Before you speak, review what each player said and how they voted. Point out moments where someone's vote does not match what they claimed to believe, or where their answers become vague when pressed on specific details. When you make an accusation, point to something concrete rather than a general feeling.

Keep track of which roles have been publicly claimed and how many of each role the game contains. If more players claim the same role than the game has of that role, at least one of them is lying. This does not apply to villager claims since there are multiple villagers. When two or more players claim the same role, the earlier and more detailed claim is generally more credible. A later claim might be a wolf following the real player's lead, but it might also be a genuine player who held back until they felt forced to speak because the earlier claim felt suspicious. Consider both possibilities.

Village-side special role players ({special_roles}) can provide information the village would otherwise lack. If you believe a player holds one of these roles and the wolves are likely to target them, you can choose to draw wolf attention toward yourself during discussion. Speaking assertively, pressing others with pointed questions, or making your own position more visible can pull the wolves' focus away from the player you want to protect. State your reasoning openly if you do this, so the village understands what you are doing.

If you currently hold the sheriff badge, your 1.5 vote is likely most effective on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. It is recommended to pass the badge to the most trustworthy village-side player based on the discussion so far. If you are about to be eliminated, consider passing it to someone you are confident in, or destroying it rather than risking it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to state your current plan so other players can follow and verify it.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "witch": """

STRATEGY: You are the witch. You hold one antidote and one poison, each usable only once per game. You are the only player who knows who the wolves killed each night. You may use at most one potion per night, not both in the same night, and you cannot use the antidote to save yourself.

Before you speak, review what each player said and how they voted. Note whose answers become vague under direct questions and whose vote choices seem to benefit the wolves rather than the village.

It is recommended to keep your identity hidden until your information is urgently needed. Revealing your role is worth considering when the village has stopped trusting you as village-side and revealing would directly recover that trust, or when the seer is dead and the village needs your information to continue. If the seer dies before reporting their findings and the village has lost its main source of information, consider stepping forward to share what you know. Reporting that you used the antidote on a player confirms they were attacked, and unless the wolves deliberately targeted themselves to drain your potion, that player is almost certainly village-side.

If another player claims to be the Witch, you are strongly encouraged to come forward and counter-claim. Stating what you have actually used and what remains can expose the false claimant. A false claimant cannot know your real usage history and will contradict it under questioning. Only one Witch exists in this game.

It is highly recommended to use both potions before you die. Every potion still in your hands when the game ends is a lost opportunity for your team. Use the antidote on a player who can still bring valuable information to the village, such as the seer or guard, so they stay in the game longer. Use the poison on a player you are confident is a wolf. If you are about to be eliminated and still hold a potion, use it that same night so your team gets the most out of what you have.

If you currently hold the sheriff badge, your 1.5 vote is likely most effective on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. It is recommended to pass the badge to the most trustworthy village-side player based on the discussion so far. If you are about to be eliminated, consider passing it to someone you are confident in, or destroying it rather than risking it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to name your current recipient or state you will tear it. Naming a recipient does not require revealing your role or your potion history.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "hunter": """

STRATEGY: You are the hunter. When you are eliminated, you may shoot one player. Make it a suspected wolf.

Before you speak, review what each player said and how they voted. Note who goes vague under direct questions and who pushes hardest against the players the village trusts most.

It is strongly recommended to keep your role hidden while you are alive. Once the wolves know you are the hunter, they will likely eliminate you early to neutralize your death shot. Revealing is worth considering only when the village has stopped trusting you as village-side and revealing would directly recover that trust. Staying active in discussion and building your credibility as a speaker is valuable, because your observations and votes carry weight in the village's decisions. When you are eliminated, choosing your target decisively is recommended.

If another player claims to be the Hunter, you are strongly encouraged to come forward and counter-claim. Only one Hunter exists in this game, and the false claimant has no shot.

If you currently hold the sheriff badge, your 1.5 vote is likely most effective on the player you suspect most of being a wolf. In a tied vote that goes to a PK round, your vote is the deciding factor for the village. It is recommended to pass the badge to the most trustworthy village-side player based on the discussion so far. If you are about to be eliminated, consider passing it to someone you are confident in, or destroying it rather than risking it falling to a wolf. Each time you speak as sheriff, use recommend_action with pass_badge or tear_badge to name your current recipient or state you will tear it. Naming a recipient does not require revealing your role.

Speaking without offering any actual view or analysis on previous players' statements and votes does not help the village side find useful information.""",

    "white_wolf_king": """

STRATEGY: You are the White Wolf King, a werewolf who can self-destruct during any day or election phase to eliminate one player, ending the phase with no result.

Before you speak, review what each player said and how they voted. Look for players who give detailed or precise answers that suggest a special role, and players the village explicitly trusts. Those are your targets.

Creating doubt between players and redirecting suspicion away from your team is recommended. Consider claiming a specific role publicly to pressure other players into revealing or defending their own identities. If doing so causes the seer or guard to identify themselves, they become a high-value target.

The self-destruct is most effective against a critical village-side player at a moment when ending the phase early benefits your team. When you self-destruct, the badge is destroyed with you, which prevents the seer from using it as a platform. Triggering it too early or against a low-value target wastes the ability.

Consider running for sheriff if doing so helps the wolf team control the discussion or secure the badge without exposing you too early. If you currently hold the sheriff badge, your 1.5 vote advantage can help eliminate village-side players without revealing your wolf alignment. Consider shaping the speaking order in ways that keep your wolf teammates safe. Passing the badge to a wolf teammate is recommended when you can do so without raising suspicion. If you self-destruct while holding the badge it is destroyed with you, which also prevents the village from using it as a platform. If you are about to be eliminated without self-destructing and passing the badge would expose a wolf ally, consider destroying it instead.""",
}


WOLF_SHARED_GUIDANCE = """

WOLF STRATEGY (SHARED).

It is recommended that one wolf take a more visible role in discussion, such as running for sheriff or claiming a specific village-side identity, while the others stay quieter and support. A team where all wolves push in the same direction at once is easier to read.

If you choose to claim a role, check what your wolf teammates have already publicly claimed and avoid repeating the same identity. A double claim on the same role gives the village a contradiction to use against your team. Once you commit to a claim, it is recommended to maintain it throughout the game. Walking back a claim later is likely to raise suspicion. When you claim a role, your speech should match how a real player of that role would speak.

When a fellow wolf is under pressure, you may challenge the accusation without defending them directly. Questioning whether the reasoning holds up reads as neutral concern rather than obvious protection.

It is strongly recommended that at least one wolf consider running for sheriff. Holding the badge gives a 1.5 vote advantage and speaking order control that benefits the team each day.

When the sheriff announces a badge succession plan, a publicly announced recipient is a vulnerability you can act on. The named player has been vouched for by the sheriff, which makes them a threat to your team. Targeting them before the transfer happens removes a trusted player and disrupts the sheriff's plan. Alternatively, casting doubt on the named player during your speech can erode the village's confidence in the plan before it executes.

Confirming your wolf identity is almost certain to end the game in the village's favor."""


VILLAGE_SHARED_GUIDANCE = """

VILLAGE STRATEGY.

If another player publicly claims the same role you hold, it is strongly recommended to come forward and state you are the real holder. Only one of each special role exists in this game. Your private history of divination results, protection targets, potion use, or shot decisions cannot be reproduced by a false claimant. Under questioning, a fabricated claim will break.

When a village-side player is eliminated and their role is revealed, a wolf may claim to be that player since the real holder is dead and cannot contest it. If someone claims a role held by a recently eliminated player, press them for details consistent with what that player actually said and did while alive."""


SHARED_VOTING_GUIDANCE = """

SHARED GUIDANCE.

When a player has publicly claimed the Seer role and reported a divination result, the Seer is the village's primary source of confirmed information. Not supporting the player you believe is the real Seer is likely to put your side at a severe disadvantage. The Seer provides alignment information no other role can replicate. It is strongly recommended that village-side players support the player they most believe is the real Seer in their votes, including the sheriff election, as the Seer holding the badge maximizes the information the village can act on each day. When two players both claim the Seer role, choose the one whose claim and behavior you trust more.

Use the known role distribution to reason about identity. If the number of players claiming a particular role exceeds how many of that role exist in this game, at least one of those claims is false. Track which roles have been claimed, how many of each the game has, and use the gap to identify contradictions. The same logic applies to behavior. If a player acts in a way that only makes sense for a role they have not claimed, that is worth noting.

If you notice anything that looks off in another player's words, votes, or behavior, sharing it openly during your speech lets others act on the observation. Withholding suspicions generally does not help your team.

If you currently hold the sheriff badge, end your final speech of the day with a single clear vote target so the rest of the village can unify their votes behind one execution. Without a clear lead from the sheriff, the village vote tends to scatter.

Staying alive preserves your team's voting power. When votes are shifting toward you, redirect suspicion in your speech and give a concrete reason to vote elsewhere instead.

When the sheriff announces a badge succession plan, treat it as binding public information. If the sheriff names a specific player as the planned recipient, that player gains credibility and becomes a key target for the opposing side. Village-side players should keep that name in mind. If the named recipient dies before the sheriff does, the badge path is broken and needs to be replanned. If the sheriff states they will tear the badge rather than pass it on, that means no alive player met their trust threshold. Pressuring a sheriff to name a successor they have not chosen freely can backfire."""
