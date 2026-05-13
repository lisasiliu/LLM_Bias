"""
prompt_templates.py -- All prompt templates given to LLM agents.

Follows codebase/prompt.py conventions: module-level string constants.

Memory templates (REASONING_TEMPLATE, RELATIONSHIP_ESTIMATE_TEMPLATE) use
positional {} placeholders like codebase.

Action templates use named {player_id}, {game_id}, {day}, {role} placeholders
and are looked up via get_phase_template(phase, role).

System prompts and predicate grammar remain in prompt.json (structured data,
not agent prompts).
"""

from .strategy_guides import DEFAULT_STRATEGY_GUIDES as STRATEGY_HINTS, SHARED_VOTING_GUIDANCE


# recommend_action uses a fixed action vocabulary so the predicate stays
# interpretable across experiments and platform logs.
RECOMMEND_ACTIONS = {
    "vote_sheriff",
    "vote_execute",
    "divine",
    "protect",
    "use_antidote",
    "use_poison",
    "shoot",
    "pass_badge",
    "tear_badge",
    "question",
    "pressure",
    "run_sheriff",
    "withdraw_sheriff",
}

ADDRESS_ROLE_TARGETS = {
    "seer", "guard", "witch", "hunter", "villager", "werewolf", "village",
}

ADDRESS_ROLE_ADVICE = {
    "hide", "reveal", "vote_execute", "hold", "act_now",
}


def normalize_recommend_action(action: str) -> str:
    action_key = (action or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "save": "use_antidote",
        "antidote": "use_antidote",
        "use_antidote": "use_antidote",
        "poison": "use_poison",
        "use_poison": "use_poison",
        "execute": "vote_execute",
        "vote": "vote_execute",
        "eliminate": "vote_execute",
        "elect": "vote_sheriff",
    }
    return aliases.get(action_key, action_key)


# ═══════════════════════════════════════════════════════════════
# Night action templates (adapted from codebase/prompt.py)
# ═══════════════════════════════════════════════════════════════

WEREWOLF_ACTION_TEMPLATE = '''Now it is night {day} round. As {player_id} and a werewolf, you should choose one player to kill. Wolves win when the number of wolves alive is equal to or greater than the number of non-wolf players alive, so every eliminated non-wolf player moves your team closer to victory regardless of their specific role. Killing any non-wolf reduces the village's voting power and tightens the win condition.

Count the alive wolves and alive non-wolves before choosing. If killing one non-wolf right now would make wolves >= non-wolves, that kill ends the game in your favor. In that case choose a non-wolf target. Do not choose yourself or a wolf teammate when a winning kill is available.

God roles have special abilities that make the village more likely to win, so removing one is generally higher value than removing a plain villager when you can identify them. If a player publicly claimed the Seer role or reported a divination result during the previous day's speeches, that player is the highest-priority kill target. A revealed Seer will identify another wolf each night until they are dead. Kill them immediately. If you do not know any god identities yet, pick whichever non-wolf player you judge most worth removing based on the day's discussion and trust patterns. On the very first night before any public discussion has happened, choose any non-wolf player rather than skipping the kill. In games with a Witch, consider that targeting a teammate as bait can be used to draw out the antidote in some situations.

If your choice does not match your teammate's, no one dies, so agree on a target.
Output exactly one predicate:
attacked({game_id}, {day}, TARGET)
OR attacked({game_id}, {day}, none) to kill no one'''

SEER_ACTION_TEMPLATE = '''Now it is night {day} round. As {player_id} and the seer, you should choose one player other than yourself to divine tonight. Divination reveals only whether the target is wolf-camp or village-camp. You will not learn their specific role. Reason about the current situation, then choose one target.
Output exactly one predicate:
divined({game_id}, {day}, {player_id}, TARGET, ?)'''

GUARD_ACTION_TEMPLATE = '''Now it is night {day} round. As {player_id} and the guard, you should choose one player to protect without knowing who the werewolves have chosen. You cannot protect the same player two nights in a row.

On the first night, protect the player you believe is most likely to be the real Seer. If one player claimed to be the Seer during the sheriff election and won the badge, protect that player. If multiple players claimed Seer and only one gave specific divination results or was elected sheriff, protect that one. If no explicit Seer claim was made but a player ran for sheriff and won the badge, protect that player because candidates who step forward are often special roles and will be wolf targets.
Output exactly one predicate:
guarded({game_id}, {day}, {player_id}, TARGET)'''

WITCH_ACTION_TEMPLATE = '''Now it is night {day} round. As {player_id} and the witch, the werewolves' target has been revealed to you. You have two one-use potions. The antidote saves tonight's victim. The poison kills one player of your choice. You may use at most one potion per night, not both. You cannot use the antidote to save yourself.
Output exactly one predicate:
  witch_save(N{day}, {player_id})           to save the victim with the antidote
  witch_poison(N{day}, {player_id}, TARGET)  to poison a player
  fluff(N{day}, {player_id})                 to do nothing'''

HUNTER_SHOOT_TEMPLATE = '''You are {player_id}, the hunter, and you have been eliminated. As your dying action, you may shoot one alive player, or choose not to shoot.
Output exactly one predicate:
  hunter_shot(D{day}, {player_id}, TARGET)  to shoot TARGET
  hunter_shot(D{day}, {player_id}, none)    to not shoot anyone'''

WWK_EXPLODE_TEMPLATE = '''You are {player_id}, the White Wolf King. You may choose to self-destruct right now, killing yourself and one other player of your choice. This will immediately end the current day phase and transition to night. If you do not want to self-destruct, pass.
Output exactly one predicate:
  wwk_explode(D{day}, {player_id}, TARGET)  to self-destruct and kill TARGET
  wwk_pass(D{day}, {player_id})              to pass'''


# ═══════════════════════════════════════════════════════════════
# Day discussion templates (adapted from codebase/prompt.py)
# ═══════════════════════════════════════════════════════════════

_STATEMENT_BASE = '''Now it is day {day} discussion phase and it is your turn to speak. As {player_id} and a {role}, your goal is to win this game. Say whatever is most useful for your team right now.

Before speaking, reason privately. Do not recite any of this out loud.

Check the current role distribution: how many wolves and special roles remain alive based on what is known? Which identities have been confirmed or ruled out? What does the remaining composition suggest about players who have not yet claimed anything?

Go through each player who has already spoken. Did their vote match what they said they believed? Did they go vague when pressed? Did they shift position once they saw how others were leaning? Note any contradiction between what someone claimed and how they actually acted, and compare that against what others said about them.

Once you have done that, decide what matters most right now. You do not need to say everything. Pick the strongest one or two points and explain your reasoning out loud, not just the conclusion. If your analysis points at a specific player, name them and say why.

If another player has named you as a suspect in the speeches above, address that accusation and give concrete reasons why the suspicion does not hold.

If you are voluntarily revealing your own role, state it openly at the start of your speech before adding anything else.
If you are the Sheriff, end your speech with a clear vote target and your badge plan. Name who you will pass the badge to if you die tonight, or state you will tear it if no trusted recipient exists. Use recommend_action with pass_badge or tear_badge for this.
Do not output vague statements or empty encouragement. Every sentence must name a specific player and give a concrete reason.
If another player has already made the same point this turn, express your view in your own words. Do not repeat their exact phrasing.

Use recommend_action with vote_execute for day execution votes, not vote_sheriff.
Use address_role to speak to a whole role group without naming a specific player.
Available: roleclaim, estimate, divined, guarded, question, answer, agree, disagree, allied, unallied, dissociate, recommend_action, address_role, fluff'''

STATEMENT_TEMPLATE = _STATEMENT_BASE + '''
Output 1-5 predicates. Each predicate must be distinct. Do not output the same predicate twice.'''

_RECOMMEND_ACTION_NATURAL_RULE = '''

If you use recommend_action, the natural sentence must explicitly name the Actor, the Action, and the Target. It must also say why this move makes sense in the current game. Do not turn it into a general appeal to everyone. If Actor or Target is yourself, still say the player number.
If you use divined, the natural sentence must explain why you chose to divine that specific player and what you found. Do not only report the result.
If you use address_role, speak directly to that role group. Do not address a specific player by name or number. State what you want them to do and ground the reason in something specific from this round, such as a player's claim or a voting pattern. Do not give generic advice that could apply to any game.'''

STATEMENT_TEMPLATE_NATURAL = _STATEMENT_BASE + '''
Output 1-5 lines, each pairing a predicate with its natural language expression. Each predicate must be distinct. Do not output the same predicate twice.
[s1: predicate(args)] Your natural language sentence for this point.
[s2: predicate(args)] Another sentence.

Use s1, s2, ... as placeholder IDs. For agree/disagree, reference the speech ID you respond to (e.g. agree({game_id}, {day}, {player_id}, s5)).''' + _RECOMMEND_ACTION_NATURAL_RULE


_LAST_SPEECH_BASE = '''You have been eliminated. As {player_id} and a {role}, give your final statement to the remaining players. Reveal your true role if it helps your team. You may also accuse players or share information.

Consider starting with: roleclaim(D{day}, {player_id}, {role})
Available: roleclaim, estimate, divined, guarded, question, answer, agree, disagree, allied, unallied, dissociate, recommend_action, address_role, fluff'''

LAST_SPEECH_TEMPLATE = _LAST_SPEECH_BASE + '''
Output 1-3 predicates.'''

LAST_SPEECH_TEMPLATE_NATURAL = _LAST_SPEECH_BASE + '''
Output 1-3 lines, each pairing a predicate with its natural language expression:
[s1: predicate(args)] Your natural language sentence for this point.''' + _RECOMMEND_ACTION_NATURAL_RULE

_LAST_SPEECH_WOLF_BASE = '''You have been eliminated. As {player_id} and a {role}, you are dead but your words still change the game. Do not reveal your wolf identity. If you claimed a role while alive, restate that claim. If you never publicly claimed a role, pick a plausible village-side role to claim now. Then redirect suspicion toward a specific player and recommend who the village should vote out next.

Available: roleclaim, estimate, dissociate, recommend_action, fluff'''

LAST_SPEECH_WOLF_TEMPLATE = _LAST_SPEECH_WOLF_BASE + '''
Output 1-3 predicates.'''

LAST_SPEECH_WOLF_TEMPLATE_NATURAL = _LAST_SPEECH_WOLF_BASE + '''
Output 1-3 lines, each pairing a predicate with its natural language expression:
[s1: predicate(args)] Your natural language sentence for this point.''' + _RECOMMEND_ACTION_NATURAL_RULE


# ═══════════════════════════════════════════════════════════════
# Voting templates (role-specific, adapted from codebase/prompt.py)
# ═══════════════════════════════════════════════════════════════

WEREWOLF_VOTE_TEMPLATE = '''Now it is day {day} voting phase, you should vote to eliminate one player or do not vote to maximize the Werewolves' benefit. As {player_id} and a werewolf, you should first reason about the current situation, and then choose from the alive players.
Output exactly one predicate:
voted({game_id}, {day}, {player_id}, TARGET)'''

VILLAGER_VOTE_TEMPLATE = '''Now it is day {day} voting phase. As {player_id} and a {role}, voting someone out is the primary way the village camp progresses toward winning. Vote to eliminate the player you believe is most likely to be a werewolf. Reason about the current situation, then choose from the alive players.

Check the speeches from this round. If any player referred to wolf allies or placed themselves in the wolf camp in their own words, vote for that player regardless of anything else said this round.
Output exactly one predicate:
voted({game_id}, {day}, {player_id}, TARGET)'''


# ═══════════════════════════════════════════════════════════════
# Sheriff election templates (adapted from codebase/prompt.py)
# ═══════════════════════════════════════════════════════════════

SHERIFF_NOMINATE_TEMPLATE = '''Now it is day 1 election phase. You can decide whether or not to run for Sheriff based on your identity. The Sheriff is a special role that does not conflict with the players' roles. The elected Sheriff has the authority to choose which player starts the next round of discussions, and as the last speaker can summarize the discussion and give a vote recommendation. The Sheriff's vote counts as 1.5 in the daytime elimination vote.

Running for Sheriff forfeits your vote in this election. This applies whether or not you later withdraw. If you do not run, you keep your vote and may vote for any candidate.

If you choose to run as a non-Seer, running gives you an extra speech turn before the vote. You can withdraw before voting begins if you only wanted that turn. If a White Wolf King is in the game, running and claiming to be the Seer can pressure them into self-destructing. Only do this when removing the White Wolf King is worth more than keeping your real identity hidden. Either way your team gains something.

If you run, output: run_sheriff({game_id}, {day}, {player_id})
If not, output: fluff({game_id}, {day}, {player_id})'''

SHERIFF_NOMINATE_WOLF_TEMPLATE = '''Now it is day 1 election phase. Decide whether to run for Sheriff. Running forfeits your vote in this election even if you later withdraw.

As a werewolf, the wolf team needs at least one wolf running and claiming Seer. A wolf Seer claimant forces the real Seer into a direct conflict, confuses the village, and gives your team a shot at the badge. You should run. Both wolves running is acceptable and can be resolved at the withdrawal window after speeches.

If you run, commit to staying in the race. You can decide at the withdrawal window whether to step back after seeing all the speeches. Do not withdraw before your campaign speech.

If you run, output: run_sheriff({game_id}, {day}, {player_id})
If not, output: fluff({game_id}, {day}, {player_id})'''

_ELECTION_SPEECH_BASE = '''Now it is day 1 election phase, you are running for Sheriff. As {player_id} and a {role}, convince other players to vote for you. Explain why you would be a good Sheriff. Do not reveal your role unless your role strategy says it is useful in this phase. In this election only players who are not running may vote.

Use recommend_action with vote_sheriff for any sheriff election vote recommendation. The Actor in that recommend_action must be a player who is not currently running for Sheriff, since candidates cannot vote.'''

ELECTION_SPEECH_TEMPLATE = _ELECTION_SPEECH_BASE + '''
Output 1-3 predicates.'''

ELECTION_SPEECH_TEMPLATE_NATURAL = _ELECTION_SPEECH_BASE + '''
Output 1-3 lines, each pairing a predicate with its natural language expression:
[s1: predicate(args)] Your natural language sentence for this point.''' + _RECOMMEND_ACTION_NATURAL_RULE

_ELECTION_SPEECH_SEER_BASE = '''Now it is day 1 election phase, you are running for Sheriff as {player_id} and the Seer. In this election only players who are not running may vote. This is the normal phase for a Seer to claim and give the village camp your information.

Your speech should first establish your Seer claim, then for each divination result explain why you chose to divine that player and what the result was, then say what you want the table to do with that information. Use roleclaim for the Seer claim and divined for each result. Use recommend_action with vote_sheriff for any sheriff election vote recommendation; the Actor must be a non-running player. Do not use fluff to represent a Seer claim or divination report. There is only one real Seer in this game. If any other candidate also claims to be the Seer, point this out directly and state that the other claimant is lying. Your specific results with concrete reasoning are what establish your credibility over a wolf fake-claiming your role.'''

ELECTION_SPEECH_TEMPLATE_SEER = _ELECTION_SPEECH_SEER_BASE + '''
Output 1-4 predicates.'''

ELECTION_SPEECH_TEMPLATE_SEER_NATURAL = _ELECTION_SPEECH_SEER_BASE + '''
Output 1-4 lines, each pairing a predicate with its natural language expression:
[s1: predicate(args)] Your natural language sentence for this point.''' + _RECOMMEND_ACTION_NATURAL_RULE

_ELECTION_SPEECH_WOLF_BASE = '''Now it is day 1 election phase, you are running for Sheriff as {player_id} and a {role}. Your goal is to appear as the most trustworthy village-side candidate. In this election only players who are not running may vote. Decide what identity, if any, to claim.

If you claim to be the seer, your speech must include a roleclaim for the seer and at least one divined result. The result can be fabricated. Structure the speech as a real seer would. State your role, give the divination result, explain why you chose to divine that specific player, and say what you want the table to do with it. Do not use fluff to represent a role claim or divination result. The real seer will also report a specific result with reasoning and will challenge any other seer claimant as a wolf. Make your fabricated result concrete, name the player and alignment clearly, and give a plausible reason for why you picked that target. You can also preempt the challenge by pointing out that only one real seer exists and that any other seer claimant in this election is lying.

For the fabricated result, reporting villager_camp is safer: the target knows they are village-side and is more likely to trust and support you. Reporting wolf_camp on a village-side player is riskier but if the table believes you, it both redirects elimination toward a villager and makes your seer claim more credible, since real seers do find wolves. Weigh the current situation and choose whichever gives the wolf team more advantage.
If you claim another village-side role, use roleclaim for that role and include information consistent with what a real player of that role would say at this point.
If you run without a role claim, speak as a player who has read the situation carefully and give a concrete reason why you should hold the badge.

Use recommend_action with vote_sheriff for any vote recommendation. The Actor in that recommend_action must be a player who is not currently running for Sheriff.'''

ELECTION_SPEECH_TEMPLATE_WOLF = _ELECTION_SPEECH_WOLF_BASE + '''
Output 1-4 predicates.'''

ELECTION_SPEECH_TEMPLATE_WOLF_NATURAL = _ELECTION_SPEECH_WOLF_BASE + '''
Output 1-4 lines, each pairing a predicate with its natural language expression:
[s1: predicate(args)] Your natural language sentence for this point.''' + _RECOMMEND_ACTION_NATURAL_RULE

ELECTION_VOTE_TEMPLATE = '''Now it is day 1 election phase. As {player_id} and a {role}, the sheriff holds 1.5 votes and controls speaking order, so placing the badge with the right candidate matters for your team. Vote for the candidate you believe is the seer. The seer can verify players' alignments and is the village's most valuable source of confirmed information. Base that judgment on how each candidate spoke and what they claimed during the campaign. If you have no reason to believe any candidate is the seer, vote for whoever you find most convincing as a village-side player. Not voting hands influence to whichever candidate wins by chance and gives up any control over who holds the badge. The sheriff badge is one of the most valuable resources in the game for your team. Reason about the current situation, then choose from the candidates.
Output exactly one predicate:
voted({game_id}, {day}, {player_id}, TARGET)'''

ELECTION_VOTE_TEMPLATE_WEREWOLF = '''Now it is day 1 election phase. As {player_id} and a werewolf, vote for the candidate that maximizes the wolf team's advantage. A wolf candidate, including yourself, holding the badge controls speaking order and gains a 1.5 vote weight that can be turned against the village. If no wolf candidate is viable, vote in a way that avoids handing the badge to a likely village leader such as a probable seer or witch. Not voting hands the outcome to chance and risks putting the badge with a strong village player with no effort spent to prevent it. Reason about the current situation, then choose from the candidates.
Output exactly one predicate:
voted({game_id}, {day}, {player_id}, TARGET)'''


# ═══════════════════════════════════════════════════════════════
# Sheriff action templates
# ═══════════════════════════════════════════════════════════════

SHERIFF_WITHDRAW_TEMPLATE = '''All sheriff campaign speeches are finished. As {player_id} and a {role}, you may withdraw your candidacy now or stay in the race. Withdrawing removes your name from the ballot and you will not receive any votes. Staying in keeps you eligible to win the badge. Reason about the speeches that have been made, the strength of the other candidates, and what helps your team most.
Output exactly one predicate:
  withdraw_sheriff({game_id}, {day}, {player_id})  to withdraw your candidacy
  fluff({game_id}, {day}, {player_id})              to stay in the race'''

SHERIFF_WITHDRAW_SEER_TEMPLATE = '''All sheriff campaign speeches are finished. You are {player_id} and the Seer, and you have run for Sheriff. Do not withdraw. Backing down after claiming the Seer role undermines your credibility with the village and hands the badge to another candidate who may be a wolf. Staying in keeps your name on the ballot and, if you win, gives you a speaking platform to share divination results each day while you are alive.

Only withdraw if a candidate in these speeches has made a concrete, specific Seer claim that you judge to be a real village-side Seer, and supporting that player benefits the village more than holding the badge yourself. That bar is high.
Output exactly one predicate:
  fluff({game_id}, {day}, {player_id})              to stay in the race
  withdraw_sheriff({game_id}, {day}, {player_id})  to withdraw your candidacy'''


SHERIFF_WITHDRAW_WOLF_TEMPLATE = '''All sheriff campaign speeches are finished. You are {player_id} and a {role}, currently running for Sheriff. Decide whether to withdraw or stay in.

If you claimed to be the Seer in your campaign speech, do not withdraw. A wolf who claimed Seer and then withdrew immediately signals to the village that the claim was false. Stay in and maintain the Seer identity through the vote.

If your wolf partner is also still running and made a more convincing speech than you, you may withdraw to consolidate wolf votes behind them. Only do this if you judge their chance of winning is clearly higher than yours and the badge will go to a wolf. If withdrawing would hand the badge to a village-side candidate, stay in.
Output exactly one predicate:
  fluff({game_id}, {day}, {player_id})              to stay in the race
  withdraw_sheriff({game_id}, {day}, {player_id})  to withdraw your candidacy'''


SHERIFF_ORDER_TEMPLATE = '''Now it is day {day} discussion phase and you are the Sheriff. As {player_id}, a {role} and the Sheriff, choose whether speaking goes clockwise or counterclockwise around the table. The order starts from your neighbor in that direction, continues through all alive players, and you speak last to summarize and give a vote direction.
Output exactly one predicate:
  sheriff_order({game_id}, {day}, {player_id}, left)   to go counterclockwise
  sheriff_order({game_id}, {day}, {player_id}, right)  to go clockwise'''

SHERIFF_SUCCESSOR_TEMPLATE = '''You are dying. Pass the sheriff badge to a successor, or tear it so no one else holds it. As {player_id}, choose based on who you trust for your team.

Pass the badge to a specific alive player if you trust them to use it for your side. Tear it if no alive player can be trusted with it.

Output exactly one predicate:
  sheriff_successor({game_id}, {day}, {player_id}, SUCCESSOR)   to pass the badge to SUCCESSOR
  sheriff_successor({game_id}, {day}, {player_id}, NO_SHERIFF)  to tear the badge'''


# ═══════════════════════════════════════════════════════════════
# Phase -> template mapping
# ═══════════════════════════════════════════════════════════════

def normalize_role_name(role: str) -> str:
    role_key = (role or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "whitewolfking": "white_wolf_king",
        "white_wolf_king": "white_wolf_king",
    }
    return aliases.get(role_key, role_key)


def get_phase_template(phase: str, role: str, speech_mode: str = "predicate") -> str:
    """Get the appropriate prompt template for a phase + role combination.

    Role-specific templates are used for voting (werewolf vs villager/seer/guard)
    and night actions (each role has its own template).
    When speech_mode="natural", speech phases use templates that elicit
    natural language alongside predicates.
    """
    natural = speech_mode == "natural"
    role = normalize_role_name(role)
    if phase == "night_wolf":
        return WEREWOLF_ACTION_TEMPLATE
    if phase == "night_seer":
        return SEER_ACTION_TEMPLATE
    if phase == "night_guard":
        return GUARD_ACTION_TEMPLATE
    if phase == "day_speech":
        return STATEMENT_TEMPLATE_NATURAL if natural else STATEMENT_TEMPLATE
    if phase == "day_vote":
        if role in ("werewolf", "white_wolf_king"):
            return WEREWOLF_VOTE_TEMPLATE
        return VILLAGER_VOTE_TEMPLATE
    if phase == "last_speech":
        if role in ("werewolf", "white_wolf_king"):
            return LAST_SPEECH_WOLF_TEMPLATE_NATURAL if natural else LAST_SPEECH_WOLF_TEMPLATE
        return LAST_SPEECH_TEMPLATE_NATURAL if natural else LAST_SPEECH_TEMPLATE
    if phase == "sheriff_nominate":
        if role in ("werewolf", "white_wolf_king"):
            return SHERIFF_NOMINATE_WOLF_TEMPLATE
        return SHERIFF_NOMINATE_TEMPLATE
    if phase == "sheriff_election_speech":
        if role == "seer":
            return ELECTION_SPEECH_TEMPLATE_SEER_NATURAL if natural else ELECTION_SPEECH_TEMPLATE_SEER
        if role in ("werewolf", "white_wolf_king"):
            return ELECTION_SPEECH_TEMPLATE_WOLF_NATURAL if natural else ELECTION_SPEECH_TEMPLATE_WOLF
        return ELECTION_SPEECH_TEMPLATE_NATURAL if natural else ELECTION_SPEECH_TEMPLATE
    if phase == "sheriff_election_vote":
        if role in ("werewolf", "white_wolf_king"):
            return ELECTION_VOTE_TEMPLATE_WEREWOLF
        return ELECTION_VOTE_TEMPLATE
    if phase == "sheriff_withdraw":
        if role == "seer":
            return SHERIFF_WITHDRAW_SEER_TEMPLATE
        if role in ("werewolf", "white_wolf_king"):
            return SHERIFF_WITHDRAW_WOLF_TEMPLATE
        return SHERIFF_WITHDRAW_TEMPLATE
    if phase == "sheriff_order":
        return SHERIFF_ORDER_TEMPLATE
    if phase == "sheriff_successor":
        return SHERIFF_SUCCESSOR_TEMPLATE
    if phase == "night_witch":
        return WITCH_ACTION_TEMPLATE
    if phase == "hunter_shoot":
        return HUNTER_SHOOT_TEMPLATE
    if phase == "wwk_explode_check":
        return WWK_EXPLODE_TEMPLATE
    return f"Phase: {phase}"


# ═══════════════════════════════════════════════════════════════
# Deductive reasoning template (follows codebase/prompt.py REASONING_TEMPLATE)
#
# Called once per other alive player, like codebase/role.py line 659:
#   REASONING_TEMPLATE.format(time_str, player_id, role, target, target, role_options)
# ═══════════════════════════════════════════════════════════════

REASONING_TEMPLATE = '''
Now it is {}, As {} and a {}, you should reflect on your previous deduction and reconsider the hidden role of {}. You should provide your reasoning, rate your confidence.

You should only respond in JSON format as described below.
Response Format:
{{
    "{}": {{
        "role": select the most likely hidden role of this player from this game's role set: {},
        "reasoning": your reflection and reasoning,
        "confidence": an integer from 1 to 10. Start at 5 (no evidence, pure guess). Increase with concrete evidence (e.g. seer result, contradictory claims, voting patterns). 10 = absolute certainty with hard proof. Lower below 5 if evidence suggests the opposite of your guess.
    }}
}}
Ensure the response can be parsed by Python json.loads.
'''


# ═══════════════════════════════════════════════════════════════
# Relationship estimation template (third-party: allied/unallied)
#
# Called once with all other alive players:
#   RELATIONSHIP_ESTIMATE_TEMPLATE.format(time_str, player_id, role, others_str)
# ═══════════════════════════════════════════════════════════════

RELATIONSHIP_ESTIMATE_TEMPLATE = '''
Now it is {}, As {} and a {}, you should assess the relationships between the other remaining players: {}. Based on their statements, actions, and voting patterns, determine which pairs of OTHER players (not yourself) appear to be allied with each other and which appear to be unallied. Do NOT include yourself in any relationship pair — only assess relationships between other players.

You should only respond in JSON format as described below.
Response Format:
{{
    "player_i": [
        {{
            "target_player": "player_j",
            "relation": "allied" if they seem to be on the same team, or "unallied" if they seem to be working against each other,
            "confidence": an integer from 1 to 10. Start at 5 (no evidence). Only increase with concrete evidence (voting alignment, coordinated accusations, etc.). 10 = absolute certainty,
            "reasoning": "brief explanation of why you think this relationship exists"
        }}
    ]
}}
Both player_i and player_j must be other players, NOT yourself.
Ensure the response can be parsed by Python json.loads.
'''
