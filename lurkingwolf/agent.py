"""
agent.py -- LLM agent for predicate-only Werewolf games.

Each agent wraps one player. Builds prompts from prompt.json templates,
calls the LLM, and parses/validates predicate output.

After each round, the agent performs deductive reasoning following
codebase/role.py conventions:
  - fact, potential_truth, potential_deception
  - reliability (int 5-10), reasoning_history, relationship_estimates
  - action_history, statement_history

Predicate format (no game_id):
  predicate(phase_label, speaker, args...)
  e.g. estimate(D1, P3, P4, villager)
       attacked(N0, P1)
       voted(D1, P3, P6)

Two-stage NL: Stage 1 generates predicates; Stage 2 (optional, temperature=0)
generates one natural language sentence per predicate.
"""

import json
import re
import random
from typing import Dict, List, Optional

from .engine import GameState, SPEECH_PREDICATES, NO_SHERIFF, ROLE_CONFIGS, WOLF_ROLES
from .utils import TokenTracker
from .town_consensus import town_consensus
from .prompts.templates import (
    REASONING_TEMPLATE, RELATIONSHIP_ESTIMATE_TEMPLATE,
    get_phase_template, normalize_recommend_action, normalize_role_name,
    RECOMMEND_ACTIONS, ADDRESS_ROLE_TARGETS, ADDRESS_ROLE_ADVICE,
    STRATEGY_HINTS, SHARED_VOTING_GUIDANCE, VILLAGE_SHARED_GUIDANCE, WOLF_SHARED_GUIDANCE,
)
from .utils import clean_natural_text

from .prompts.rules_guides import RULES_GUIDES as _RULES_GUIDES

def _load_rules_guide(num_players: int) -> str:
    if num_players not in _RULES_GUIDES:
        supported = ", ".join(str(n) for n in sorted(_RULES_GUIDES))
        raise ValueError(
            f"Unsupported player count for rules guide: {num_players}. "
            f"Supported player counts are: {supported}."
        )
    return _RULES_GUIDES[num_players]


# ═══════════════════════════════════════════════════════════════
# Predicate parsing (adapted from generate.py parse_action)
# ═══════════════════════════════════════════════════════════════

def parse_action(action: str) -> Dict[str, object]:
    """Parse a predicate string like 'estimate(D1, P3, P4, villager)'.

    Returns {"name": str, "args": list, "raw": str}.
    Malformed input returns name=action with empty args (never raises).
    """
    if not action or not isinstance(action, str):
        return {"name": "", "args": [], "raw": str(action)}
    action = action.strip()
    match = re.match(r"(\w+)\((.+)\)$", action)
    if not match:
        return {"name": action, "args": [], "raw": action}

    name = match.group(1)
    args_str = match.group(2)
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())

    return {"name": name, "args": args, "raw": action}


def _format_predicate_defs(predicates: dict, **subs) -> str:
    lines = []
    for category in ("speech", "action", "sheriff"):
        for name, entry in predicates.get(category, {}).items():
            sig = entry["sig"]
            desc = entry["desc"]
            try:
                desc = desc.format(**subs)
            except KeyError:
                pass
            lines.append(f"  {name}{sig} - {desc}")
    return "\n".join(lines)


def _close_json(s: str) -> str:
    """Attempt to close truncated JSON by appending missing brackets/braces."""
    s = s.rstrip()

    stack = []
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    if in_string:
        s += '"'

    s = s.rstrip().rstrip(',')

    for opener in reversed(stack):
        s += ']' if opener == '[' else '}'
    return s


def _format_result_defs(predicates: dict) -> str:
    lines = []
    for name, entry in predicates.get("result", {}).items():
        lines.append(f"  {name}{entry['sig']} - {entry['desc']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# LLM Agent
# ═══════════════════════════════════════════════════════════════



class LLMAgent:
    # Wolves keep a snapshot of the engine's town-consensus score (never shown to any agent)
    track_town_consensus = True
    town_consensus = None

    # Phases where natural language speech is allowed (two-stage)
    SPEECH_PHASES = {"day_speech", "sheriff_summary", "last_speech",
                     "sheriff_election_speech"}

    def __init__(self, player_id: str, role: str, game_id: str,
                 model: str, client, prompts: dict,
                 players: List[str],
                 *, temperature: float = 0, max_tokens: int = 300,
                 token_tracker: Optional[TokenTracker] = None,
                 speech_mode: str = "predicate"):
        self.player_id = player_id
        self.role = normalize_role_name(role)
        self.game_id = game_id
        self.model = model
        self.client = client
        self.prompts = prompts
        self.players = players
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.call_log: List[dict] = []
        self.token_tracker = token_tracker
        self.speech_mode = speech_mode
        self.rewriter = None
        self.last_reasoning: str = ""  # captured <think> traces from last act()
        self.last_speech: List[dict] = []  # [{pred, text, thinking}, ...] for NL mode
        self.thinking_traces: List[dict] = []  # accumulated across all act() calls

        # Persistent per-agent state (follows codebase/role.py BaseAgent)
        self.fact: List[str] = []
        self.potential_truth: List[dict] = []
        self.potential_deception: List[dict] = []
        self.reliability: Dict[str, int] = {}  # empty until first reasoning
        self.reasoning_history: List[dict] = []
        self.relationship_estimates: Dict[str, list] = {p: [] for p in players}
        self.action_history: List[str] = []
        self.statement_history: List[dict] = []

    def build_system_prompt(self) -> str:
        template = self.prompts["system"]["game"]
        n = len(self.players)
        roles = ROLE_CONFIGS.get(n, {})
        role_dist = ", ".join(f"{c} {r}{'s' if c > 1 else ''}" for r, c in roles.items())
        base = template.format(
            player_id=self.player_id,
            game_id=self.game_id,
            role=self.role,
            num_players=n,
            role_distribution=role_dist,
        )

        # Prepend the rules guide for this player count so agents understand
        # game-specific rules before seeing the predicate grammar.
        rules_text = _load_rules_guide(n)
        if rules_text:
            base = "GAME RULES:\n" + rules_text + "\n\n" + base

        grammar = "\n\nPREDICATE GRAMMAR:\n"
        _rc = ROLE_CONFIGS.get(len(self.players), {})
        _vtr = "|".join(sorted(_rc.keys()) + ["village"])
        _vr = "|".join(sorted(_rc.keys()))
        grammar += _format_predicate_defs(self.prompts["predicates"], valid_target_roles=_vtr, valid_roles=_vr)
        grammar += "\n\nRESULT PREDICATES (system-generated):\n"
        grammar += _format_result_defs(self.prompts["predicates"])

        strategy = STRATEGY_HINTS.get(self.role, "")
        if strategy and "{special_roles}" in strategy:
            village_special = sorted(r for r in roles if r not in WOLF_ROLES and r != "villager")
            strategy = strategy.format(special_roles=", ".join(village_special) if village_special else "special roles")
        village_guidance = VILLAGE_SHARED_GUIDANCE if self.role not in WOLF_ROLES else ""
        wolf_guidance = WOLF_SHARED_GUIDANCE if self.role in WOLF_ROLES else ""
        base += grammar + strategy + village_guidance + wolf_guidance + SHARED_VOTING_GUIDANCE
        return base

    def _format_beliefs(self) -> str:
        """Format current reliability scores and relationships into readable text."""
        lines = []

        if self.reliability:
            lines.append("  Reliability scores:")
            for pid, score in self.reliability.items():
                if pid != self.player_id:
                    lines.append(f"    {pid}: {score}/10")
        else:
            lines.append("  Reliability scores: (no assessments yet)")

        has_relations = False
        for pid, rels in self.relationship_estimates.items():
            if rels:
                if not has_relations:
                    lines.append("  Relationship estimates:")
                    has_relations = True
                for r in rels:
                    lines.append(
                        f"    {pid} <-> {r['target_player']}: "
                        f"{r['relation']} (confidence {r.get('confidence', 5)})"
                    )
        if not has_relations:
            lines.append("  (no relationship estimates yet)")

        if self.reasoning_history:
            latest = self.reasoning_history[-1]
            lines.append(f"  Latest reasoning ({latest.get('timeline', '?')}):")
            for pid, info in latest.get("reasoning_result", {}).items():
                lines.append(
                    f"    {pid}: likely {info.get('role', '?')} "
                    f"(confidence {info.get('confidence', 5)})"
                )

        return "\n".join(lines)

    def _uses_natural_speech(self, phase: str) -> bool:
        return self.speech_mode != "predicate" and phase in self.SPEECH_PHASES

    def build_user_prompt(self, state: GameState, phase: str,
                          extra_context: str = "") -> str:
        template_lines = self.prompts.get("game_user_template", [])

        phase_idx = state.night if phase.startswith("night_") else state.day

        phase_inst = get_phase_template(
            phase,
            self.role,
            "natural" if self._uses_natural_speech(phase) else "predicate",
        )
        phase_inst = phase_inst.format(
            game_id=self.game_id,
            day=phase_idx,
            player_id=self.player_id,
            role=self.role,
        )

        private = state.private_knowledge.get(self.player_id, [])
        private_str = "\n".join(f"  {p}" for p in private) if private else "  (none)"

        beliefs_str = self._format_beliefs()

        alive_shuffled = [p for p in state.get_alive()
                          if p not in state.pending_night_deaths]
        random.shuffle(alive_shuffled)

        # Current turn speeches (buffer) — only this phase's speeches
        turn_speeches = state.current_turn_buffer
        turn_str = "\n".join(f"  {s}" for s in turn_speeches) if turn_speeches else "  (none yet)"

        roles_config = ROLE_CONFIGS.get(len(self.players), {})
        role_dist_note = ", ".join(
            f"{c} {r}{'s' if c > 1 else ''}" for r, c in roles_config.items()
        ) if roles_config else "(unknown)"

        valid_target_roles = "|".join(sorted(ADDRESS_ROLE_TARGETS))
        valid_roles = "|".join(sorted(roles_config.keys()))

        variables = {
            "predicate_defs": _format_predicate_defs(
                self.prompts["predicates"],
                valid_target_roles=valid_target_roles,
                valid_roles=valid_roles,
            ),
            "result_defs": _format_result_defs(self.prompts["predicates"]),
            "game_id": self.game_id,
            "day": f"Night {state.night}" if phase.startswith("night_") else f"Day {state.day}",
            "alive_players": ", ".join(alive_shuffled),
            "dead_players": ", ".join(state.get_dead()) or "(none)",
            "sheriff": state.sheriff or "(none)",
            "role_distribution_note": role_dist_note,
            "current_turn_speeches": turn_str,
            "private_knowledge": private_str,
            "beliefs": beliefs_str,
            "phase_instruction": phase_inst,
        }

        prompt = "\n".join(template_lines).format_map(variables)

        # Inject historical role claims and divinations from all previous rounds
        # so agents can detect contradictions (e.g., two players both claiming seer).
        claims_on_record = []
        seen_claims: set = set()
        for entry in state.public_history:
            if not isinstance(entry, str):
                continue
            for match in re.finditer(
                r"\[s\d+\|(\w+)\]\s*(claim_role|claim_divine|roleclaim|divined)\([^)]+\)", entry
            ):
                claim = match.group(0).split("] ", 1)[1].strip()
                if claim not in seen_claims:
                    seen_claims.add(claim)
                    claims_on_record.append(claim)
        if claims_on_record:
            prompt += "\n\nCLAIMS ON RECORD (previous rounds):\n" + "\n".join(
                f"  {c}" for c in claims_on_record
            )

        if extra_context:
            prompt += "\n" + extra_context
        return prompt

    def call_llm(self, system_prompt: str, user_prompt: str,
                 **extra_kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **extra_kwargs,
        )
        text = resp.choices[0].message.content.strip()

        token_info = {}
        if self.token_tracker:
            input_text = system_prompt + "\n" + user_prompt
            token_info = self.token_tracker.record(input_text, text)

        self.call_log.append({
            "player": self.player_id,
            "model": self.model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": text,
            **token_info,
        })
        return text

    def _thinking_kwargs(self) -> dict:
        """Return extra kwargs to enable thinking for thinking models on vLLM."""
        from .utils.gen_config import thinking_extra_body
        return thinking_extra_body(self.model)

    # ── Main action method ──

    def act(self, state: GameState, phase: str,
            valid_targets: Optional[List[str]] = None,
            dry_run: bool = False,
            extra_context: str = "") -> List[str]:
        """Have the agent act. Returns list of valid predicates.

        For natural speech_mode in speech phases, also runs Stage 2 (NL generation)
        and populates self.last_speech = [{pred, text}, ...].
        """
        self.last_reasoning = ""
        self.last_speech = []

        saved_tokens = self.max_tokens

        # Boost max_tokens for thinking models so they can reason + answer
        from .utils.gen_config import stage1_max_tokens
        self.max_tokens = stage1_max_tokens(self.model, self.max_tokens)

        think_kw = self._thinking_kwargs()

        sys_prompt = self.build_system_prompt()
        usr_prompt = self.build_user_prompt(state, phase, extra_context=extra_context)

        if dry_run:
            self.call_log.append({
                "player": self.player_id,
                "phase": phase,
                "system_prompt": sys_prompt,
                "user_prompt": usr_prompt,
                "response": "[dry-run]",
            })
            self.max_tokens = saved_tokens
            return self._fallback(phase, state, valid_targets)

        raw = self.call_llm(sys_prompt, usr_prompt, **think_kw)
        self.last_reasoning = self._extract_think_tags(raw)
        paired_speech = self._validated_paired_speech(raw, phase, state, valid_targets)
        predicates = self._parse_and_validate(raw, phase, state, valid_targets)

        # Retry once if nothing valid
        if not predicates:
            retry_hint = "\n\nYour previous output was invalid or empty. "
            if self._uses_natural_speech(phase):
                retry_hint += "Output lines in this format: [s1: predicate(args)] natural sentence."
            else:
                retry_hint += "Output ONLY valid predicates, one per line. No natural language."
            retry_prompt = usr_prompt + retry_hint
            raw = self.call_llm(sys_prompt, retry_prompt, **think_kw)
            paired_speech = self._validated_paired_speech(raw, phase, state, valid_targets)
            predicates = self._parse_and_validate(raw, phase, state, valid_targets)

        if not predicates:
            predicates = self._fallback(phase, state, valid_targets)

        # Save Stage 1 thinking traces
        if self.last_reasoning:
            self.thinking_traces.append({
                "phase": phase,
                "stage": "predicate",
                "thinking": self.last_reasoning,
            })

        # Stage 2: generate natural language for speech phases (if enabled)
        if self._uses_natural_speech(phase) and predicates:
            texts = []
            thinkings = []
            paired_by_pred = {}
            for item in paired_speech:
                text = item.get("text", "")
                if text:
                    paired_by_pred.setdefault(item["pred"], []).append(text)
            for pred in predicates:
                bucket = paired_by_pred.get(pred) or []
                texts.append(bucket.pop(0) if bucket else "")
                thinkings.append("")

            missing = [idx for idx, text in enumerate(texts) if not text]
            if missing:
                generated_texts, generated_thinkings = self._generate_nl_speech(
                    [predicates[idx] for idx in missing],
                    state,
                )
                for idx, text, thinking in zip(missing, generated_texts, generated_thinkings):
                    texts[idx] = text
                    thinkings[idx] = thinking
            if self.rewriter is not None:
                from .rewriter import RewriteContext
                for idx, (pred, text) in enumerate(zip(predicates, texts)):
                    if not text:
                        continue
                    applies = self.rewriter.applies_to_predicates
                    pred_name = parse_action(pred).get("name", "")
                    if applies and pred_name not in applies:
                        continue
                    ctx = RewriteContext(
                        predicate=pred,
                        original_nl=text,
                        phase=phase,
                        speaker_id=self.player_id,
                        speaker_role_claimed=self.role,
                        public_history=list(state.public_history),
                        day=getattr(state, "day", None),
                        alive_players=list(state.get_alive()),
                        speaker_reliability=dict(self.reliability),
                        town_consensus=dict(self.town_consensus) if self.town_consensus else None,
                    )
                    texts[idx] = self.rewriter.rewrite(ctx)

            self.last_speech = [
                {"pred": pred, "text": text, "thinking": thinking}
                for pred, text, thinking in zip(predicates, texts, thinkings)
            ]
            # Save Stage 2 thinking traces
            for entry in self.last_speech:
                if entry.get("thinking"):
                    self.thinking_traces.append({
                        "phase": phase,
                        "stage": "nl",
                        "pred": entry["pred"],
                        "thinking": entry["thinking"],
                    })

        self.max_tokens = saved_tokens
        return predicates

    # ── Two-stage NL generation ──

    # Predicates that reference a speech ID as their last argument
    _REF_PREDICATES = {"agree", "disagree", "answer", "question"}
    def _build_nl_system(self) -> str:
        """Build the Stage 2 NL system prompt."""
        preamble = self.prompts["system"]["generate_preamble"]
        speech_preds = self.prompts["predicates"].get("speech", {})
        pred_lines = []
        for name, entry in speech_preds.items():
            parts = [entry["desc"]]
            if "nl_hint" in entry:
                parts.append(entry["nl_hint"])
            pred_lines.append(f"- {name}{entry['sig']}: {' '.join(parts)}")
        base = f"{preamble}\n\nPredicate meanings (the speaker is always the Player argument):\n" + "\n".join(pred_lines) + "\n"
        ext = self.prompts.get("system_extensions", {})
        out = base
        if "baseline" in ext:
            out += ext["baseline"]
        return out

    def _generate_nl_speech(self, predicates: List[str],
                            state: GameState) -> tuple:
        """Stage 2: generate natural language per predicate.

        Returns (texts, thinkings), a list of NL strings and a parallel list
        of thinking trace strings (empty string if no thinking).
        """
        from .utils.gen_config import deterministic_kwargs

        saved_temp = self.temperature
        saved_max = self.max_tokens

        # Pull deterministic settings from shared config
        det_kw = deterministic_kwargs(self.model)
        self.temperature = det_kw.pop("temperature")
        self.max_tokens = det_kw.pop("max_tokens")
        det_kw.pop("model")  # already set on self
        # det_kw now contains seed, top_p, and possibly extra_body

        texts = []
        thinkings = []
        for pred in predicates:
            # Check if this predicate references a speech ID
            ref_context = ""
            parsed = parse_action(pred)
            if parsed["name"] in self._REF_PREDICATES and parsed["args"]:
                ref_id = parsed["args"][-1].strip()
                ref_entry = state.speech_registry.get(ref_id)
                if ref_entry:
                    # Prefer NL text for natural mode; fall back to predicate
                    ref_text = ref_entry.get("text") or ref_entry.get("predicate", "")
                    ref_speaker = ref_entry.get("speaker", "?")
                    ref_context = (
                        f"\nReferenced speech [{ref_id}|{ref_speaker}]: {ref_text}\n"
                    )

            prompt = (
                f"You are {self.player_id}, a {self.role} in a Werewolf game.\n"
                f"{ref_context}\n"
                f"[{pred}] → Speak this in character as natural game dialogue. "
                f"Refer to players by their player number."
            )
            raw = self.call_llm(self._build_nl_system(), prompt, **det_kw)
            think = self._extract_think_tags(raw)
            text = self._strip_think_tags(raw).strip()
            # Strip any predicate prefix the LLM might include
            text = re.sub(r'^\[.*?\]\s*→?\s*', '', text).strip()

            # Retry once if Stage 2 did not return playable dialogue.
            if self._looks_like_refusal(text):
                retry_prompt = prompt + (
                    "\n\nOutput the in-character game dialogue directly. "
                    "Do not add disclaimers."
                )
                raw = self.call_llm(self._build_nl_system(), retry_prompt, **det_kw)
                think = self._extract_think_tags(raw)
                text = self._strip_think_tags(raw).strip()
                text = re.sub(r'^\[.*?\]\s*→?\s*', '', text).strip()

            thinkings.append(think)
            texts.append(text)

        self.temperature = saved_temp
        self.max_tokens = saved_max
        return texts, thinkings

    def _extract_paired_speech(self, raw: str) -> List[dict]:
        """Extract [{pred, text}, ...] from [s1: predicate(args)] text lines."""
        raw = self._strip_think_tags(raw or "")
        pairs = []
        for line in raw.splitlines():
            match = re.match(r"^\s*\[s\d+\s*:\s*([^\]]+)\]\s*(.*)$", line, re.IGNORECASE)
            if not match:
                continue
            pred = match.group(1).strip()
            text = clean_natural_text(match.group(2).strip())
            if pred:
                pairs.append({"pred": pred, "text": text})
        return pairs

    def _validated_paired_speech(self, raw: str, phase: str, state: GameState,
                                 valid_targets: Optional[List[str]]) -> List[dict]:
        if not self._uses_natural_speech(phase):
            return []
        pairs = []
        for item in self._extract_paired_speech(raw):
            preds = self._parse_and_validate(item["pred"], phase, state, valid_targets)
            if preds:
                pairs.append({"pred": preds[0], "text": item.get("text", "")})
        return pairs

    # ── End-of-round deductive reasoning (follows codebase/role.py) ──

    def update_memory(self, state: GameState, time_str: str,
                      dry_run: bool = False) -> dict:
        """Deductive reasoning + relationship estimation."""
        if dry_run:
            reasoning_result = self._generate_dummy_reasoning(state)
            relationship_result = self._generate_dummy_relationships(state)
        else:
            reasoning_result = self._run_reasoning(state, time_str)
            relationship_result = self._run_relationship_estimation(state, time_str)

        for pid, info in reasoning_result.items():
            confidence = info.get("confidence", 5)
            suspected_role = normalize_role_name(info.get("role", "villager"))
            if suspected_role in WOLF_ROLES and self.role not in WOLF_ROLES:
                self.reliability[pid] = 11 - confidence
            else:
                self.reliability[pid] = confidence

        self.relationship_estimates = relationship_result
        if self.track_town_consensus and self.role in WOLF_ROLES:
            self.town_consensus = town_consensus(state)

        entry = {
            "timeline": time_str,
            "reasoning_result": reasoning_result,
        }
        self.reasoning_history.append(entry)

        return {
            "reasoning": reasoning_result,
            "reliability": dict(self.reliability),
            "relationship_estimates": {
                k: v for k, v in relationship_result.items() if v
            },
        }

    # ── Deductive reasoning ──

    def _run_reasoning(self, state: GameState, time_str: str) -> dict:
        """Call LLM once per other alive player for role deduction."""
        sys_prompt = self.build_system_prompt()
        context_prompt = self.build_user_prompt(state, "day_speech")
        alive = state.get_alive()
        others = [p for p in alive if p != self.player_id]

        saved = self.max_tokens
        self.max_tokens = max(self.max_tokens, 1000)
        think_kw = self._thinking_kwargs()
        reasoning_result = {}
        role_options = sorted({normalize_role_name(role) for role in state.roles.values()})
        role_options_text = json.dumps(role_options)

        for target in others:
            reasoning_prompt = REASONING_TEMPLATE.format(
                time_str, self.player_id, self.role, target, target, role_options_text
            )
            prompt = context_prompt + "\n" + reasoning_prompt
            raw = self.call_llm(sys_prompt, prompt, **think_kw)
            parsed = self._parse_single_reasoning(raw, target, state)
            if parsed:
                reasoning_result[target] = parsed

        self.max_tokens = saved
        return reasoning_result

    def _parse_single_reasoning(self, raw: str, target: str,
                                state: GameState) -> Optional[dict]:
        """Parse reasoning JSON for a single target player."""
        data = self._extract_json_from_response(raw)
        if data is None:
            return None

        valid_roles = {normalize_role_name(role) for role in state.roles.values()}

        info = data.get(target)
        if info is None:
            for key, val in data.items():
                if key in state.players and isinstance(val, dict):
                    info = val
                    break
        if not isinstance(info, dict):
            return None

        role = normalize_role_name(info.get("role", "villager"))
        if role not in valid_roles:
            role = "villager"
        confidence = info.get("confidence", 5)
        if not isinstance(confidence, int) or confidence < 1:
            confidence = 5
        elif confidence > 10:
            confidence = 10

        return {
            "role": role,
            "reasoning": str(info.get("reasoning", "")),
            "confidence": confidence,
        }

    def _generate_dummy_reasoning(self, state: GameState) -> dict:
        """Generate plausible dummy reasoning for dry-run."""
        alive = state.get_alive()
        others_alive = [p for p in alive if p != self.player_id]
        roles_pool = sorted({normalize_role_name(role) for role in state.roles.values()})

        reasoning_result = {}
        for pid in others_alive:
            role_guess = random.choice(roles_pool)
            confidence = random.randint(3, 7)
            reasoning_result[pid] = {
                "role": role_guess,
                "reasoning": f"Based on {pid}'s behavior, guessing {role_guess}.",
                "confidence": confidence,
            }

        return reasoning_result

    # ── Relationship estimation ──

    def _run_relationship_estimation(self, state: GameState,
                                     time_str: str) -> Dict[str, list]:
        """Call LLM to assess relationships between other players."""
        sys_prompt = self.build_system_prompt()
        context_prompt = self.build_user_prompt(state, "day_speech")
        alive = state.get_alive()
        others = [p for p in alive if p != self.player_id]

        if len(others) < 2:
            return {p: [] for p in self.players}

        others_str = ", ".join(others)
        rel_prompt = RELATIONSHIP_ESTIMATE_TEMPLATE.format(
            time_str, self.player_id, self.role, others_str
        )
        prompt = context_prompt + "\n" + rel_prompt

        saved = self.max_tokens
        self.max_tokens = max(self.max_tokens, 1500)
        raw = self.call_llm(sys_prompt, prompt, **self._thinking_kwargs())
        self.max_tokens = saved

        return self._parse_relationship_json(raw, state)

    def _parse_relationship_json(self, raw: str,
                                 state: GameState) -> Dict[str, list]:
        """Parse relationship estimation JSON."""
        result = {p: [] for p in self.players}
        data = self._extract_json_from_response(raw)
        if data is None:
            return result

        valid_relations = {"allied", "unallied"}
        for pid, entries in data.items():
            if pid not in state.players:
                continue
            if pid == self.player_id:
                continue
            if not isinstance(entries, list):
                continue
            cleaned = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                target = entry.get("target_player", "")
                if target not in state.players:
                    continue
                if target == self.player_id:
                    continue
                relation = entry.get("relation", "unallied")
                if relation not in valid_relations:
                    relation = "unallied"
                confidence = entry.get("confidence", 5)
                if not isinstance(confidence, int) or confidence < 1:
                    confidence = 5
                elif confidence > 10:
                    confidence = 10
                rel_entry = {
                    "target_player": target,
                    "relation": relation,
                    "confidence": confidence,
                }
                reasoning = entry.get("reasoning", "")
                if reasoning:
                    rel_entry["reasoning"] = str(reasoning)
                cleaned.append(rel_entry)
            result[pid] = cleaned

        return result

    def _generate_dummy_relationships(self, state: GameState) -> Dict[str, list]:
        """Generate plausible dummy relationship estimates for dry-run."""
        alive = state.get_alive()
        others = [p for p in alive if p != self.player_id]
        result = {p: [] for p in self.players}

        for pid in others:
            pool = [p for p in others if p != pid]
            n = min(len(pool), random.randint(0, 2))
            picks = random.sample(pool, n) if n > 0 else []
            result[pid] = [
                {
                    "target_player": t,
                    "relation": random.choice(["allied", "unallied"]),
                    "confidence": random.randint(5, 8),
                }
                for t in picks
            ]

        return result

    # ── JSON extraction helper ──

    _REFUSAL_PATTERNS = (
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i won't",
        "as an ai",
        "i'm not able to",
        "i am not able to",
        "sorry, but i",
        "i must decline",
    )

    @classmethod
    def _looks_like_refusal(cls, text: str) -> bool:
        """Detect Qwen/RLHF-style refusals in Stage 2 output."""
        if not text:
            return True  # empty output also treated as refusal
        lower = text.strip().lower()
        return any(p in lower for p in cls._REFUSAL_PATTERNS)

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip <think>...</think> blocks (complete and truncated)."""
        # 1. Complete <think>...</think>
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # 2. Truncated <think>... with no closing tag
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        # 3. Model emitted reasoning WITHOUT opening <think> but with closing </think>
        #    (observed with Qwen3.5 when "Thinking Process:" precedes </think>)
        if '</think>' in text:
            text = text.split('</think>')[-1].strip()
        return text

    @staticmethod
    def _extract_think_tags(text: str) -> str:
        """Extract reasoning content from thinking blocks.

        Handles three shapes: <think>...</think>, <think>... (truncated),
        and ...</think> (no opening tag — observed with Qwen3.5).
        """
        blocks = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
        if blocks:
            return "\n".join(b.strip() for b in blocks if b.strip())
        # Truncated: opening tag, no closing
        m = re.search(r'<think>(.*)', text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # Closing only: everything before </think>
        if '</think>' in text:
            return text.split('</think>')[0].strip()
        return ""

    def _extract_json_from_response(self, raw: str) -> Optional[dict]:
        """Extract JSON object from LLM response text."""
        raw = self._strip_think_tags(raw)
        idx = raw.find("{")
        if idx >= 0:
            json_str = raw[idx:]
        else:
            json_str = raw.strip()

        json_str = json_str.replace("{{", "{").replace("}}", "}")

        data = self._try_parse_json(json_str)

        if data is None:
            cleaned = raw.replace("{{", "{").replace("}}", "}")
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                data = self._try_parse_json(match.group())

        return data

    @staticmethod
    def _try_parse_json(s: str) -> Optional[dict]:
        """Try to parse JSON, with repairs for common LLM output issues."""
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            depth = 0
            end = 0
            for i, ch in enumerate(s):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > 0:
                return json.loads(s[:end])
        except (json.JSONDecodeError, TypeError):
            pass

        repaired = _close_json(s)
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            pass

        for cut in [
            r',\s*"[^"]*"?\s*$',
            r',\s*\{[^}]*$',
            r',\s*"[^"]*":\s*"?[^"]*$',
        ]:
            trimmed = re.sub(cut, '', s)
            if trimmed != s:
                repaired = _close_json(trimmed)
                try:
                    return json.loads(repaired)
                except (json.JSONDecodeError, TypeError):
                    continue

        return None

    # ── Memory tracking helpers ──

    def receive_fact(self, fact_str: str) -> None:
        """Add a verified fact to agent memory."""
        if fact_str not in self.fact:
            self.fact.append(fact_str)

    def classify_claim(self, speaker: str, claim: str,
                       text: str = "", sid: str = "",
                       suspected_deception: bool = False) -> None:
        """Classify a claim from another player as potential truth or deception."""
        parsed = parse_action(claim)
        if parsed["name"] == "fluff":
            return

        entry = {"speaker": speaker, "claim": claim}
        if text:
            entry["text"] = text
        if sid:
            entry["speech_id"] = sid
        if suspected_deception:
            self.potential_deception.append(entry)
        else:
            self.potential_truth.append(entry)

    def classify_speech_nl(self, state: GameState, speaker: str,
                           nl_text: str, *, dry_run: bool = False) -> List[dict]:
        """Listener-side parse: convert another player's natural-language
        speech into predicate claims as THIS agent understands them, and
        classify each as truth or deception based on this agent's beliefs.

        Each surviving claim is appended to potential_truth or
        potential_deception. Returns the list of {kind, claim} entries
        added (for caller logging).
        """
        if dry_run or not nl_text or not nl_text.strip():
            return []
        if speaker == self.player_id:
            return []  # never classify your own speech

        sys_prompt = self.build_system_prompt()
        phase_label = (
            f"D{state.day}" if not state.phase.startswith("night") else f"N{state.night}"
        )
        alive = ", ".join(state.get_alive())
        beliefs = self._format_beliefs()
        valid_predicates = ", ".join(sorted(SPEECH_PREDICATES))

        parse_prompt = (
            f"You are {self.player_id}. Another player {speaker} just spoke "
            f"publicly during {phase_label}.\n"
            f"Alive players: {alive}\n"
            f"Your beliefs so far:\n{beliefs}\n\n"
            f"{speaker} said:\n\"{nl_text.strip()}\"\n\n"
            f"Parse {speaker}'s speech into predicate claims as YOU understand "
            f"them. For each concrete claim, classify it as truth (you find it "
            f"credible given your role and beliefs) or deception (you suspect "
            f"misdirection or a lie).\n"
            f"Allowed predicate names: {valid_predicates}.\n"
            f"Output one line per claim, exactly:\n"
            f"  [truth] predicate({phase_label}, {speaker}, ...)\n"
            f"  [deception] predicate({phase_label}, {speaker}, ...)\n"
            f"If {speaker} only hedged or made no concrete claim, output nothing."
        )

        try:
            raw = self.call_llm(sys_prompt, parse_prompt)
        except Exception as e:
            print(f"[agent {self.player_id}] classify_speech_nl failed: {e}")
            return []
        raw = self._strip_think_tags(raw)

        classified: List[dict] = []
        for line in raw.splitlines():
            line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
            if not line:
                continue
            m = re.match(r'^\[(truth|deception)\]\s*(.+)$', line, re.IGNORECASE)
            if not m:
                continue
            kind = m.group(1).lower()
            pred_str = m.group(2).strip()
            paired = re.match(r'^\[s\d+\s*:\s*([^\]]+)\]', pred_str, re.IGNORECASE)
            if paired:
                pred_str = paired.group(1).strip()
            parsed = parse_action(pred_str)
            if not parsed.get("name") or not parsed.get("args"):
                continue
            if parsed["name"] == "fluff":
                continue
            entry = {
                "speaker": speaker,
                "claim": pred_str,
                "text": nl_text.strip(),
                "listener_parsed": True,
            }
            if kind == "deception":
                self.potential_deception.append(entry)
            else:
                self.potential_truth.append(entry)
            classified.append({"kind": kind, "claim": pred_str})

        return classified

    def record_own_action(self, action_str: str) -> None:
        """Record an action this agent took."""
        self.action_history.append(action_str)

    def record_own_statement(self, timeline: str,
                             flat_strings: List[str]) -> None:
        """Record a statement this agent made (as flat predicate strings)."""
        self.statement_history.append({
            "timeline": timeline,
            "speech": list(flat_strings),
        })

    # ── Predicate validation ──

    def _parse_and_validate(self, raw: str, phase: str,
                            state: GameState,
                            valid_targets: Optional[List[str]]) -> List[str]:
        """Parse and validate predicate output.

        New predicate format: predicate(phase_label, speaker_or_target, args...)
        No game_id. Phase label = N{n} or D{d}.
        Speaker index = args[1] for speech predicates.
        """
        raw = self._strip_think_tags(raw)

        valid = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if line.startswith("REASONING_JSON:") or line.startswith("{"):
                continue
            line = re.sub(r"^[\d\.\-\*\s]+", "", line).strip()
            if not line:
                continue
            paired_match = re.match(r"^\s*\[s\d+\s*:\s*([^\]]+)\]", line, re.IGNORECASE)
            if paired_match:
                line = paired_match.group(1).strip()

            parsed = parse_action(line)
            name = parsed["name"]
            args = parsed["args"]

            _PREDICATE_ALIASES = {
                "roleclaim": "claim_role",
                "witch_save": "antidote",
                "witch_poison": "poison",
                "hunter_shot": "shoot",
                "wwk_explode": "explode",
                "wwk_pass": "pass",
            }
            if name in _PREDICATE_ALIASES:
                name = _PREDICATE_ALIASES[name]
                line = self._rebuild_predicate(name, args)
            if name == "divined":
                name = "divine" if phase.startswith("night_") else "claim_divine"
                line = self._rebuild_predicate(name, args)
            if name == "guarded":
                name = "guard" if phase.startswith("night_") else "claim_guard"
                line = self._rebuild_predicate(name, args)

            if not args:
                continue

            # Normalize role arguments (args[-1]) for role-taking predicates
            _ROLE_ALIASES = {
                "innocent": "villager", "town": "villager", "townsfolk": "villager",
                "wolf": "werewolf", "wolves": "werewolf",
                "doctor": "guard", "bodyguard": "guard", "protector": "guard",
                "detective": "seer", "oracle": "seer",
                "healer": "witch", "medic": "witch", "potionmaster": "witch",
                "shooter": "hunter", "gunner": "hunter",
                "wwk": "white_wolf_king", "wolf_king": "white_wolf_king",
            }
            if name in ("estimate", "claim_role") and len(args) >= 3:
                role_arg = args[-1].strip().lower()
                if role_arg not in set(state.roles.values()):
                    role_arg = _ROLE_ALIASES.get(role_arg, "villager")
                args[-1] = role_arg
                line = self._rebuild_predicate(name, args)

            if name == "claim_divine" and len(args) >= 4:
                align_arg = args[-1].strip().lower().replace("-", "_").replace(" ", "_")
                _ALIGNMENT_ALIASES = {
                    "wolf": "wolf_camp",
                    "wolves": "wolf_camp",
                    "werewolf": "wolf_camp",
                    "white_wolf_king": "wolf_camp",
                    "bad": "wolf_camp",
                    "evil": "wolf_camp",
                    "good": "villager_camp",
                    "village": "villager_camp",
                    "villager": "villager_camp",
                    "village_camp": "villager_camp",
                    "town": "villager_camp",
                }
                if align_arg not in {"wolf_camp", "villager_camp"}:
                    align_arg = _ALIGNMENT_ALIASES.get(align_arg, "villager_camp")
                args[-1] = align_arg
                line = self._rebuild_predicate(name, args)

            # Track whether this phase block manually builds the final line
            skip_auto_correct = False

            if phase in ("day_speech", "last_speech", "sheriff_summary",
                         "sheriff_election_speech"):
                if name not in SPEECH_PREDICATES:
                    continue
                # Strict arity check, reject malformed predicates so they don't
                # leak into the game log and get misinterpreted by Stage 2 NL.
                # Expected arity = (Phase, Player, ...rest)
                _EXPECTED_ARITY = {
                    "allied": 4, "unallied": 4,
                    "dissociate": 3,
                    "question": 4,
                    "agree": 3, "disagree": 3,
                    "estimate": 4, "claim_divine": 4,
                    "claim_role": 3, "claim_guard": 3,
                    "claim_antidote": 2, "claim_poison": 3,
                    "answer": 3,
                    "recommend_action": 5,
                    "address_role": 4,
                    "fluff": 2,
                }
                # Auto-repair 4-arg recommend_action: LLM omitted speaker, insert at pos 1
                if name == "recommend_action" and len(args) == 4:
                    args.insert(1, self.player_id)
                    line = self._rebuild_predicate(name, args)
                if name in _EXPECTED_ARITY and len(args) != _EXPECTED_ARITY[name]:
                    continue
                # Speaker at args[1] in new format
                if len(args) >= 2 and args[1] != self.player_id:
                    args[1] = self.player_id
                    line = self._rebuild_predicate(name, args)
                # Self-reference checks: can't target self / reference own speech
                if name in ("estimate", "dissociate") and len(args) >= 3:
                    if args[2] == self.player_id:
                        continue  # don't estimate/dissociate yourself
                if name in ("allied", "unallied") and len(args) >= 4:
                    if args[2] == self.player_id or args[3] == self.player_id:
                        continue  # relationship predicates shouldn't include self
                if name == "question" and len(args) >= 4:
                    if args[2] == self.player_id:
                        continue  # don't question yourself
                if name in ("agree", "disagree", "answer", "question"):
                    ref_id = args[-1].strip()
                    ref_entry = state.speech_registry.get(ref_id)
                    if not ref_entry:
                        continue  # drop references to non-existent speech IDs
                    if ref_entry.get("speaker") == self.player_id:
                        continue  # don't agree/disagree/etc with your own speech
                # answer must reference a question predicate
                if name == "answer" and len(args) >= 3:
                    ref_id = args[2].strip()
                    ref_entry = state.speech_registry.get(ref_id)
                    if not ref_entry or "question" not in ref_entry.get("predicate", ""):
                        continue  # drop answer if it doesn't reference a question
                if name == "recommend_action" and len(args) >= 5:
                    action = normalize_recommend_action(args[3])
                    if action not in RECOMMEND_ACTIONS:
                        continue
                    args[3] = action
                    line = self._rebuild_predicate(name, args)
                    # Actor cannot be the same as target for vote_execute
                    if action == "vote_execute" and args[2] == args[4]:
                        continue
                    # Candidates can't vote in sheriff elections — drop if actor is still running
                    if action == "vote_sheriff":
                        actor = args[2]
                        run_pred = f"run_sheriff(D{state.day}, {actor})"
                        withdraw_pred = f"withdraw_sheriff(D{state.day}, {actor})"
                        has_run = any(run_pred in h for h in state.public_history)
                        has_withdrawn = any(withdraw_pred in h for h in state.public_history)
                        if has_run and not has_withdrawn:
                            continue
                if name == "claim_role" and len(args) >= 3:
                    claimed = args[2].strip().lower()
                    # Wolves must not publicly claim their wolf identity
                    if self.role in WOLF_ROLES and claimed in WOLF_ROLES:
                        continue
                if name == "address_role" and len(args) >= 4:
                    target_role = args[2].strip().lower()
                    advice = args[3].strip().lower().replace("-", "_").replace(" ", "_")
                    if target_role not in ADDRESS_ROLE_TARGETS:
                        continue
                    if advice not in ADDRESS_ROLE_ADVICE:
                        continue
                    args[2] = target_role
                    args[3] = advice
                    line = self._rebuild_predicate(name, args)

            elif phase == "sheriff_nominate":
                if name == "run_sheriff":
                    # run_sheriff(phase, candidate) — candidate at args[1]
                    if len(args) >= 2:
                        args[1] = self.player_id
                        line = self._rebuild_predicate(name, args)
                elif name == "fluff":
                    pass  # declining nomination
                else:
                    continue

            elif phase in ("day_vote", "sheriff_election_vote"):
                if name != "voted":
                    continue
                # Templates emit voted(game_id, day, voter, target) — 4 args.
                # Predicate definition is voted(phase, voter, target) — 3 args.
                # Support both: target is args[3] for 4-arg, args[2] for 3-arg.
                if len(args) < 3:
                    continue
                target_idx = 3 if len(args) >= 4 else 2
                target = args[target_idx]
                if target not in state.alive:
                    if valid_targets:
                        args[target_idx] = random.choice(valid_targets)
                        line = self._rebuild_predicate(name, args)
                    else:
                        continue

            elif phase == "night_wolf":
                if name not in ("attacked", "voted"):
                    continue
                # attacked(phase, target) — target at args[1]
                # voted(phase, voter, target) — target at args[2]
                if name == "attacked":
                    target = args[1] if len(args) >= 2 else None
                    # Handle old format attacked(game_id, phase, target)
                    if target and target not in (valid_targets or []) and len(args) >= 3:
                        target = args[2]
                elif name == "voted":
                    target = args[2] if len(args) >= 3 else None
                    if target is None and len(args) >= 4:
                        target = args[3]
                else:
                    continue
                # Allow "none"/"null" as explicit no-kill
                if target and target.lower() in ("none", "null"):
                    line = f"attacked(N{state.night}, none)"
                    skip_auto_correct = True
                    valid.append(line)
                    continue
                if not target:
                    continue
                if valid_targets and target not in valid_targets:
                    target = random.choice(valid_targets)
                line = f"attacked(N{state.night}, {target})"
                skip_auto_correct = True

            elif phase == "night_seer":
                if name != "divine":
                    continue
                # divine(phase, seer, target) — target at args[2]
                # Find the first arg (after phase label) that is a valid target.
                target = None
                for a in args[1:]:
                    if not valid_targets or a in valid_targets:
                        target = a
                        break
                if target is None:
                    target = random.choice(valid_targets) if valid_targets else None
                if target is None:
                    continue
                # Ensure args[2] is the target
                if len(args) >= 3:
                    args[2] = target
                    # Trim extra args (e.g. old format role args at wrong position)
                    args = args[:3]
                    line = self._rebuild_predicate(name, args)
                else:
                    continue

            elif phase == "night_guard":
                if name != "guard":
                    continue
                # guard(phase, guard, target) — target at args[2]
                # Find the first arg (after phase+guard) that is a valid target.
                target = None
                for a in args[1:]:
                    if not valid_targets or a in valid_targets:
                        target = a
                        break
                if target is None:
                    target = random.choice(valid_targets) if valid_targets else None
                if target is None:
                    continue
                if len(args) >= 3:
                    args[2] = target
                    args = args[:3]
                    line = self._rebuild_predicate(name, args)
                else:
                    continue

            elif phase == "night_witch":
                if name == "antidote":
                    line = f"antidote(N{state.night}, {self.player_id})"
                    skip_auto_correct = True
                elif name == "poison":
                    # poison(phase, witch, target)
                    target = args[2] if len(args) >= 3 else None
                    if not target:
                        for a in args[1:]:
                            if valid_targets and a in valid_targets:
                                target = a
                                break
                    if target and valid_targets and target not in valid_targets:
                        target = random.choice(valid_targets) if valid_targets else None
                    if target:
                        line = f"poison(N{state.night}, {self.player_id}, {target})"
                        skip_auto_correct = True
                    else:
                        continue
                elif name == "fluff":
                    pass  # do nothing - valid
                else:
                    continue

            elif phase == "hunter_shoot":
                if name != "shoot":
                    continue
                # shoot(phase, hunter, target) — target can be "none"
                target = args[2] if len(args) >= 3 else None
                # Allow "none"/"null" as explicit no-shoot
                if target and target.lower() in ("none", "null"):
                    line = f"shoot(D{state.day}, {self.player_id}, none)"
                    skip_auto_correct = True
                    valid.append(line)
                    continue
                if not target:
                    for a in args[1:]:
                        if valid_targets and a in valid_targets:
                            target = a
                            break
                if not target:
                    continue
                if valid_targets and target not in valid_targets:
                    target = random.choice(valid_targets)
                line = f"shoot(D{state.day}, {self.player_id}, {target})"
                skip_auto_correct = True

            elif phase == "wwk_explode_check":
                if name == "explode":
                    # explode(phase, wwk, target)
                    target = args[2] if len(args) >= 3 else None
                    if not target and valid_targets:
                        target = random.choice(valid_targets)
                    if not target:
                        continue
                    if valid_targets and target not in valid_targets:
                        target = random.choice(valid_targets)
                    line = f"explode(D{state.day}, {self.player_id}, {target})"
                    skip_auto_correct = True
                elif name == "pass":
                    line = f"pass(D{state.day}, {self.player_id})"
                    skip_auto_correct = True
                else:
                    continue

            elif phase == "sheriff_order":
                if name != "sheriff_order":
                    continue
                # Validate direction arg: must be 'left' or 'right'
                if len(args) >= 3:
                    direction = args[2].strip().lower()
                    if direction not in ("left", "right"):
                        direction = random.choice(["left", "right"])
                    args[2] = direction
                    line = self._rebuild_predicate(name, args)

            elif phase == "sheriff_successor":
                if name != "sheriff_successor":
                    continue
                # sheriff_successor(phase, sheriff, successor) — successor at args[2]
                if len(args) >= 3:
                    successor = args[2]
                    # Handle old format with 4 args
                    if successor not in state.alive and successor != NO_SHERIFF and len(args) >= 4:
                        successor = args[3]
                    # Treat "none" / "no_sheriff" as badge destruction
                    if successor.lower() in ("none", "no_sheriff"):
                        successor = NO_SHERIFF
                        args[2] = NO_SHERIFF
                        line = self._rebuild_predicate(name, args)
                    if successor == NO_SHERIFF:
                        pass
                    elif successor not in state.alive:
                        if valid_targets:
                            args[2] = random.choice(valid_targets)
                            line = self._rebuild_predicate(name, args)
                        else:
                            continue

            # Auto-correct phase label in args[0] (unless night_wolf already built it)
            if not skip_auto_correct and len(args) >= 1:
                phase_label = f"N{state.night}" if phase.startswith("night_") else f"D{state.day}"
                args[0] = phase_label
                line = self._rebuild_predicate(name, args)

            # Deduplicate: skip if same predicate already in this output
            if line in valid:
                continue

            valid.append(line)

        # Enforce limits
        if phase in ("day_speech", "last_speech", "sheriff_summary",
                      "sheriff_election_speech"):
            valid = valid[:5]
        elif phase in ("day_vote", "night_wolf", "night_seer", "night_guard",
                        "night_witch", "hunter_shoot", "wwk_explode_check",
                        "sheriff_order", "sheriff_successor",
                        "sheriff_nominate", "sheriff_election_vote"):
            valid = valid[:1]

        return valid

    def _rebuild_predicate(self, name: str, args: List[str]) -> str:
        return f"{name}({', '.join(args)})"

    def _fallback(self, phase: str, state: GameState,
                  valid_targets: Optional[List[str]]) -> List[str]:
        """Return a valid fallback predicate for dry-run or parse failure."""
        # Phase label: N{n} for night phases, D{d} for day phases
        if phase.startswith("night_"):
            ph = f"N{state.night}"
        else:
            ph = f"D{state.day}"
        p = self.player_id

        if phase in ("day_speech", "last_speech", "sheriff_summary",
                      "sheriff_election_speech"):
            return [f"fluff({ph}, {p})"]
        if phase == "sheriff_nominate":
            if random.random() < 0.5:
                return [f"run_sheriff({ph}, {p})"]
            return [f"fluff({ph}, {p})"]
        if phase in ("day_vote", "sheriff_election_vote"):
            targets = valid_targets or [t for t in state.get_alive() if t != p]
            target = random.choice(targets) if targets else p
            return [f"voted({ph}, {p}, {target})"]
        if phase == "night_wolf":
            targets = valid_targets or [t for t in state.get_alive()
                                         if state.roles.get(t) not in WOLF_ROLES]
            target = random.choice(targets) if targets else state.get_alive()[0]
            return [f"attacked(N{state.night}, {target})"]
        if phase == "night_seer":
            targets = valid_targets or [t for t in state.get_alive() if t != p]
            target = random.choice(targets) if targets else state.get_alive()[0]
            return [f"divine(N{state.night}, {p}, {target})"]
        if phase == "night_guard":
            targets = valid_targets or list(state.alive)
            target = random.choice(targets)
            return [f"guard(N{state.night}, {p}, {target})"]
        if phase == "night_witch":
            return [f"fluff(N{state.night}, {p})"]
        if phase == "hunter_shoot":
            return [f"shoot({ph}, {p}, none)"]
        if phase == "wwk_explode_check":
            return [f"pass({ph}, {p})"]
        if phase == "sheriff_order":
            direction = random.choice(["left", "right"])
            return [f"sheriff_order({ph}, {p}, {direction})"]
        if phase == "sheriff_successor":
            targets = valid_targets or [t for t in state.get_alive() if t != p]
            target = random.choice(targets) if targets else state.get_alive()[0]
            return [f"sheriff_successor({ph}, {p}, {target})"]
        return [f"fluff({ph}, {p})"]


# ═══════════════════════════════════════════════════════════════
# Extraction helpers
# ═══════════════════════════════════════════════════════════════

def extract_vote_target(predicate: str) -> Optional[str]:
    """Extract target from voted(phase, voter, target) or voted(game_id, day, voter, target)."""
    parsed = parse_action(predicate)
    if parsed["name"] == "voted":
        args = parsed["args"]
        if len(args) >= 4:
            return args[3]
        if len(args) >= 3:
            return args[2]
    return None

def extract_attack_target(predicate: str) -> Optional[str]:
    """Extract target from attacked(phase, target)."""
    parsed = parse_action(predicate)
    if parsed["name"] == "attacked" and len(parsed["args"]) >= 2:
        return parsed["args"][1]
    return None

def extract_night_target(predicate: str) -> Optional[str]:
    """Extract target from divine(phase, seer, target) or guard(phase, guard, target)."""
    parsed = parse_action(predicate)
    if parsed["name"] in ("divine", "guard") and len(parsed["args"]) >= 3:
        return parsed["args"][2]
    return None

def extract_sheriff_direction(predicate: str) -> Optional[str]:
    """Extract direction from sheriff_order(phase, sheriff, left|right)."""
    parsed = parse_action(predicate)
    if parsed["name"] != "sheriff_order" or len(parsed["args"]) < 3:
        return None
    direction = parsed["args"][2].strip().lower()
    if direction in ("left", "right"):
        return direction
    return None

def extract_successor(predicate: str) -> Optional[str]:
    """Extract successor from sheriff_successor(phase, sheriff, successor)."""
    parsed = parse_action(predicate)
    if parsed["name"] == "sheriff_successor" and len(parsed["args"]) >= 3:
        return parsed["args"][2]
    return None

def extract_run_sheriff(predicate: str) -> bool:
    """Returns True if the predicate is a run_sheriff nomination."""
    parsed = parse_action(predicate)
    return parsed["name"] == "run_sheriff"

def extract_withdraw_sheriff(predicate: str) -> bool:
    """Returns True if the predicate is a withdraw_sheriff action."""
    parsed = parse_action(predicate)
    return parsed["name"] == "withdraw_sheriff"

def extract_witch_save(predicate: str) -> bool:
    """Returns True if the predicate is an antidote action."""
    parsed = parse_action(predicate)
    return parsed["name"] == "antidote"

def extract_witch_poison_target(predicate: str) -> Optional[str]:
    """Extract target from poison(phase, witch, target)."""
    parsed = parse_action(predicate)
    if parsed["name"] == "poison" and len(parsed["args"]) >= 3:
        return parsed["args"][2]
    return None

def extract_hunter_target(predicate: str) -> Optional[str]:
    """Extract target from shoot(phase, hunter, target)."""
    parsed = parse_action(predicate)
    if parsed["name"] == "shoot" and len(parsed["args"]) >= 3:
        return parsed["args"][2]
    return None

def extract_wwk_explode(predicate: str) -> Optional[str]:
    """Extract target from explode(phase, wwk, target). Returns None if pass."""
    parsed = parse_action(predicate)
    if parsed["name"] == "explode" and len(parsed["args"]) >= 3:
        return parsed["args"][2]
    return None
