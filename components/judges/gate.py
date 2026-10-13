"""The in-game gate: which rewrites may be delivered.

`structural_check` fails a rewrite that drops a player number, role, speech ID, action word or literal that the original named,
adds a real hedge (might, maybe, perhaps, ...), or flips agree/disagree. The speaker's own number may become I/me/my, and the
addressee's number may become you/your (only when the utterance is addressed to that player).

`AdmissionGate` adds two questions to gpt-5.6-luna, both answered yes or no:
- when the action rule fires: does the rewrite still recommend or describe the same action toward the same target, in any
  wording? A yes clears that failure. The question and the answer are logged.
- the reasons check: does the rewrite still give each reason or piece of evidence the original gave (listed once per original and
  cached)? It fails only if the original gives reasons and the answer is no.

If the checking model cannot be reached or gives no usable reply, the gate raises `GateUnavailable`, so the replay fails and is
rerun instead of passing a rewrite unchecked.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.helpers import CheckResult, SourceRecord

_PLAYER_RE = re.compile(r"^P\d+$", re.IGNORECASE)
_SPEECH_RE = re.compile(r"^s\d+$", re.IGNORECASE)
_PHASE_RE = re.compile(r"^[DN]\d+$", re.IGNORECASE)

ROLE_SURFACE: Dict[str, List[str]] = {
    "werewolf": ["werewolf", "werewolves", "wolf", "wolves"],
    "white_wolf_king": ["white wolf king", "white-wolf king", "wolf king", "wwk"],
    "villager": ["villager", "villagers", "village", "good", "townsfolk"],
    "seer": ["seer", "prophet"],
    "guard": ["guard", "bodyguard", "protector"],
    "witch": ["witch"],
    "hunter": ["hunter"],
    "wolf_camp": ["wolf camp", "wolf side", "wolf team", "werewolf", "werewolves", "wolf", "wolves"],
    "villager_camp": ["villager camp", "village side", "village team", "villager", "villagers", "good"],
}

ACTION_SURFACE: Dict[str, List[str]] = {
    "vote_sheriff": ["vote for sheriff", "sheriff vote", "vote", "badge", "sheriff"],
    "vote_execute": ["vote out", "vote to execute", "vote for", "execute", "exile", "vote", "lynch"],
    "divine": ["divine", "check", "inspect", "verify", "look at"],
    "protect": ["protect", "guard", "shield", "save"],
    "use_antidote": ["antidote", "save", "heal", "revive"],
    "use_poison": ["poison"],
    "shoot": ["shoot", "shot", "gun", "take down"],
    "pass_badge": ["pass the badge", "pass badge", "hand the badge", "give the badge", "badge"],
    "tear_badge": ["tear the badge", "tear badge", "destroy the badge", "rip the badge", "badge"],
    "question": ["question", "ask", "press", "challenge"],
    "pressure": ["pressure", "press", "push"],
    "run_sheriff": ["run for sheriff", "run", "campaign", "sheriff"],
    "withdraw_sheriff": ["withdraw", "drop out", "step down", "pull out"],
    "hide": ["hide", "stay hidden", "keep quiet", "stay quiet"],
    "reveal": ["reveal", "come out", "claim", "show yourself", "speak up"],
    "hold": ["hold", "wait", "hold off", "stand by"],
    "act_now": ["act now", "act", "move now", "do it now"],
}

HEDGE_PATTERNS: List[str] = [
    r"\bmaybe\b", r"\bmight\b", r"\bperhaps\b", r"\bpossibly\b", r"\bprobably\b",
    r"\bpotentially\b", r"\bseems?\b", r"\bseemed\b", r"\bappears?\b",
    r"\bi guess\b", r"\bi suspect\b",
    r"\bi'?m not sure\b", r"\bnot certain\b", r"\bunsure\b", r"\bcould be\b",
    r"\bmay be\b", r"\bkind of\b", r"\bsort of\b", r"\bpretty sure\b",
    r"\bif i'?m right\b", r"\blikely\b",
]

_AGREEMENT_FLIP = [
    r"\bi disagree\b", r"\bi don'?t agree\b", r"\bi do not agree\b",
    r"\bthat'?s wrong\b", r"\bi oppose\b", r"\bi can'?t back\b",
]
_DISAGREEMENT_FLIP = [
    r"\bi agree\b", r"\bi'?m with (you|p\d+)\b", r"\byou'?re right\b",
    r"\bexactly right\b", r"\bi support\b", r"\bi back\b",
]
_CONTRAST = [r"\bbut\b", r"\bhowever\b", r"\bthough\b", r"\byet\b", r"\bstill\b", r"\bexcept\b"]


def classify_arg(arg: str) -> str:
    arg = (arg or "").strip()
    if _PLAYER_RE.match(arg):
        return "player"
    if _SPEECH_RE.match(arg):
        return "speech_id"
    if _PHASE_RE.match(arg):
        return "phase"
    if arg.lower() in ROLE_SURFACE:
        return "role"
    if arg.lower() in ACTION_SURFACE:
        return "action"
    return "literal"


def surface_forms(arg: str, kind: str) -> List[str]:
    """Regex alternatives that count as realizing ``arg`` on the surface."""
    value = (arg or "").strip()
    low = value.lower()
    if kind == "player":
        num = re.sub(r"\D", "", value)
        return [rf"\bp\s*{num}\b", rf"\bplayer\s*#?\s*{num}\b", rf"#\s*{num}\b"]
    if kind == "speech_id":
        num = re.sub(r"\D", "", value)
        return [rf"\bs\s*{num}\b", rf"\bstatement\s*{num}\b"]
    if kind == "role":
        return [rf"\b{re.escape(form)}\b" for form in ROLE_SURFACE.get(low, [low])]
    if kind == "action":
        return [rf"\b{re.escape(form)}\b" for form in ACTION_SURFACE.get(low, [low])]
    return [rf"\b{re.escape(low)}\b"]


_CONTRACTIONS: List[tuple[str, str]] = [
    (r"\bi am\b", "i'm"), (r"\bwe are\b", "we're"), (r"\byou are\b", "you're"),
    (r"\bthey are\b", "they're"), (r"\bthat is\b", "that's"), (r"\bit is\b", "it's"),
    (r"\bthere is\b", "there's"), (r"\blet us\b", "let's"),
    (r"\bdo not\b", "don't"), (r"\bdoes not\b", "doesn't"), (r"\bdid not\b", "didn't"),
    (r"\bis not\b", "isn't"), (r"\bare not\b", "aren't"), (r"\bwas not\b", "wasn't"),
    (r"\bcan not\b", "can't"), (r"\bcannot\b", "can't"), (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"), (r"\bcould not\b", "couldn't"),
    (r"\bhave not\b", "haven't"), (r"\bhas not\b", "hasn't"),
    (r"\bi would\b", "i'd"), (r"\bi will\b", "i'll"), (r"\bi have\b", "i've"),
]


def text_variants(text: str) -> List[str]:
    """The lowered text plus its contracted spelling (deduplicated)."""
    lowered = (text or "").lower()
    contracted = lowered
    for pattern, replacement in _CONTRACTIONS:
        contracted = re.sub(pattern, replacement, contracted)
    return [lowered] if contracted == lowered else [lowered, contracted]


def _matches(text: str, patterns: List[str]) -> List[str]:
    variants = text_variants(text)
    return [p for p in patterns if any(re.search(p, v) for v in variants)]


def _realized(text: str, patterns: List[str]) -> bool:
    return bool(_matches(text, patterns))


FIRST_PERSON = re.compile(r"\b(i|me|my|mine|myself|i'm|i'll|i've|i'd)\b", re.I)
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself|you're|you'll|you've|you'd)\b", re.I)


def player_numbers(text: str) -> set:
    return {int(n) for n in re.findall(r"\bp\s*(\d+)\b", text or "", re.I)} | \
           {int(n) for n in re.findall(r"\bplayer\s*#?\s*(\d+)\b", text or "", re.I)}


def player_reference_failures(original: str, rewrite: str, speaker: str, addressee: Optional[str]) -> List[str]:
    speaker_n = int(speaker[1:]) if re.fullmatch(r"P\d+", speaker or "") else None
    addressee_n = int(addressee[1:]) if addressee and re.fullmatch(r"P\d+", addressee) else None
    kept = player_numbers(rewrite)
    missing = []
    for n in sorted(player_numbers(original)):
        if n in kept:
            continue
        if n == speaker_n and FIRST_PERSON.search(rewrite):
            continue
        if n == addressee_n and SECOND_PERSON.search(rewrite):
            continue
        missing.append(f"P{n}")
    return missing


def structural_check(
    source: SourceRecord,
    rewritten_nl: str,
    *,
    addressee: Optional[str] = None,
) -> CheckResult:
    """Deterministic preservation checks driven by the structured predicate."""
    reasons: List[str] = []
    details: Dict[str, Any] = {
        "slots": [],
        "unverifiable_slots": [],
        "added_hedges": [],
        "identical_to_original": False,
    }

    text = (rewritten_nl or "").strip()
    if not text:
        return CheckResult(ok=False, reasons=["empty_rewrite"], details=details)

    original = source.original_nl or ""
    details["identical_to_original"] = _tokens(text) == _tokens(original)

    for index, arg in enumerate(source.predicate_args):
        kind = classify_arg(arg)
        if kind in ("phase", "player"):                   # player numbers are checked on the text below
            continue
        patterns = surface_forms(arg, kind)
        in_original = _realized(original, patterns)
        in_rewrite = _realized(text, patterns)
        slot = {
            "index": index,
            "arg": arg,
            "kind": kind,
            "in_original": in_original,
            "in_rewrite": in_rewrite,
        }
        details["slots"].append(slot)
        if not in_original:
            details["unverifiable_slots"].append(arg)
            continue
        if in_rewrite:
            continue
        reasons.append(f"lost_{kind}_slot:{arg}")

    for missing in player_reference_failures(original, text, getattr(source, "speaker_id", ""), addressee):
        reasons.append(f"lost_player_slot:{missing}")

    added = [h for h in _matches(text, HEDGE_PATTERNS) if not _matches(original, [h])]
    details["added_hedges"] = added
    if added:
        reasons.append("added_epistemic_hedge:" + ",".join(h.strip("\\b") for h in added))

    if source.predicate_name == "agree" and _matches(text, _AGREEMENT_FLIP):
        reasons.append("polarity_flip_suspected:agreement_became_disagreement")
    if source.predicate_name == "disagree":
        if _matches(text, _DISAGREEMENT_FLIP) and not _matches(text, _CONTRAST):
            reasons.append("polarity_flip_suspected:disagreement_became_agreement")

    return CheckResult(ok=not reasons, reasons=reasons, details=details)


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


ACTION_SYSTEM = 'You compare a rewrite with an action from a Werewolf game. Answer with JSON only: {"answer": "yes"} or {"answer": "no"}.'


def action_and_target(source: SourceRecord, action_arg: str) -> tuple:
    args = list(source.predicate_args)
    if source.predicate_name == "recommend_action" and len(args) >= 5:
        target = args[4]
    elif source.predicate_name == "address_role" and len(args) >= 3:
        target = f"the {args[2]} players"
    else:
        target = "the same target as in the original"
    return action_arg.replace("_", " "), target


class GateUnavailable(RuntimeError):
    """The checking model could not be reached or gave no usable reply. A rewrite is never passed (or failed) on that."""


RETRY_WAITS = (2, 6, 15)


REASONS_SYSTEM = ('You read one utterance from a Werewolf game. Answer with JSON only: '
                  '{"reasons": ["...", "..."]}, using an empty list when it gives none.')
KEPT_SYSTEM = 'You compare a rewrite with the reasons an original gave. Answer with JSON only: {"answer": "yes"} or {"answer": "no"}.'


class AdmissionGate:
    def __init__(self, model, cache_path: Optional[Path] = None):
        """`cache_path` is one file shared by every replay: each original utterance's reason list is
        computed once (keyed by the hash of its text) and reused across settings and repetitions."""
        self.model = model
        self._reasons: Dict[str, Optional[List[str]]] = {}
        self._lock = threading.Lock()
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._reasons.setdefault(row["key"], row["reasons"])

    @staticmethod
    def _key(original: str) -> str:
        return hashlib.sha256(original.encode("utf-8")).hexdigest()[:20]

    def _json(self, system: str, user: str, max_tokens: int = 500) -> dict:
        """One JSON reply from the checking model. Transport errors and unreadable replies are retried; if they persist the
        gate raises GateUnavailable so the replay fails and is rerun, never silently skipping a check."""
        last = ""
        for wait in (0,) + RETRY_WAITS:
            time.sleep(wait)
            try:
                parsed, _ = self.model.complete_json(system=system, user=user, max_tokens=max_tokens, temperature=0.0)
                if isinstance(parsed, dict):
                    return parsed
                last = "unreadable reply"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"[:200]
        raise GateUnavailable(last)

    def original_reasons(self, original: str) -> Optional[List[str]]:
        """The reasons or evidence the original gives for its claim ([] for none, None if unreadable)."""
        key = self._key(original)
        with self._lock:
            if key in self._reasons:
                return self._reasons[key]
        if self.cache_path and self.cache_path.exists():           # another process may have written it
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip() and json.loads(line)["key"] == key:
                    with self._lock:
                        self._reasons[key] = json.loads(line)["reasons"]
                    return self._reasons[key]
        parsed = self._json(REASONS_SYSTEM, f'Utterance: "{original}"\nList the reasons or evidence this utterance '
                            f'gives for its claim, or say none.\nJSON:')
        if not isinstance(parsed.get("reasons"), list):
            raise GateUnavailable("reasons reply had no list")
        reasons = [str(r).strip() for r in parsed["reasons"] if str(r).strip() and str(r).strip().lower() != "none"]
        with self._lock:
            self._reasons[key] = reasons
            if self.cache_path and reasons is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "original": original, "reasons": reasons}, ensure_ascii=False) + "\n")
        return reasons

    def reasons_kept(self, reasons: List[str], rewrite: str) -> Optional[bool]:
        listed = "\n".join(f"- {r}" for r in reasons)
        parsed = self._json(KEPT_SYSTEM, f'Reasons the original gave:\n{listed}\nRewrite: "{rewrite}"\n'
                            f'Does the rewrite still give each of these reasons, in any wording? Answer yes or no.\nJSON:')
        answer = str(parsed.get("answer", "")).strip().lower()
        return True if answer == "yes" else False if answer == "no" else None

    def appeal(self, source: SourceRecord, rewrite: str, action_arg: str) -> Dict[str, Any]:
        action, target = action_and_target(source, action_arg)
        question = (f"Original action: {action} toward {target}. Does the rewrite still recommend or describe "
                    f"the same action toward the same target, in any wording?")
        user = f'Rewrite: "{rewrite}"\n{question}\nJSON:'
        raw = str(self._json(ACTION_SYSTEM, user, 400).get("answer", "")).strip().lower()
        answer = True if raw == "yes" else False if raw == "no" else None
        return {"action": action_arg, "question": question, "rewrite": rewrite, "answer": answer}

    def reasons_check(self, source: SourceRecord, rewrite: str) -> Dict[str, Any]:
        """The original's listed reasons, and whether the rewrite still gives each (None if not asked/readable)."""
        evidence: Dict[str, Any] = {"original_reasons": None, "kept": None}
        if (rewrite or "").strip():
            listed = self.original_reasons(source.original_nl or "")
            evidence["original_reasons"] = listed
            if listed:
                evidence["kept"] = self.reasons_kept(listed, rewrite)
        return evidence

    def check(self, source: SourceRecord, rewrite: str, *, addressee: Optional[str] = None) -> Dict[str, Any]:
        det = structural_check(source, rewrite, addressee=addressee)
        reasons: List[str] = list(det.reasons)
        appeals: List[Dict[str, Any]] = []
        for reason in list(reasons):
            if reason.startswith("lost_action_slot:"):
                appeal = self.appeal(source, rewrite, reason.split(":", 1)[1])
                appeals.append(appeal)
                if appeal["answer"] is True:
                    reasons.remove(reason)
        evidence = self.reasons_check(source, rewrite)
        if evidence["kept"] is False:
            reasons.append("evidence_dropped")
        return {"admitted": not reasons, "structural_reasons": list(det.reasons),
                "gate_failure_reasons": reasons, "action_appeals": appeals, "evidence": evidence,
                "identical_to_original": bool(det.details.get("identical_to_original"))}
