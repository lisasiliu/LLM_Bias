"""The in-game gate: the deterministic structural check, with one model appeal for the action rule.

`structural_check` fails a rewrite that drops a role, speech ID, literal, action or player number the original
named, adds a real hedge (might, maybe, perhaps, ...), or flips agree/disagree. When the action rule fires, one
yes/no question goes to gpt-5.6-luna: does the rewrite still recommend or describe the same action toward the
same target, in any wording? A yes clears that failure. Every other rule has no appeal. The question and the
answer are logged.

Evidence: gpt-5.6-luna lists the reasons or evidence the ORIGINAL gives for its claim (cached per original).
If any are listed, it is asked whether the rewrite still gives each of them, in any wording. The rewrite fails
only if the original has reasons and the answer is no. An unparseable reply never fails a rewrite.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.helpers import SourceRecord
from components.judges.structural_check import structural_check

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

    def _json(self, system: str, user: str) -> Optional[dict]:
        try:
            parsed, _ = self.model.complete_json(system=system, user=user, max_tokens=500, temperature=0.0)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

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
        reasons = None
        if parsed is not None and isinstance(parsed.get("reasons"), list):
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
        answer = str((parsed or {}).get("answer", "")).strip().lower()
        return True if answer == "yes" else False if answer == "no" else None

    def appeal(self, source: SourceRecord, rewrite: str, action_arg: str) -> Dict[str, Any]:
        action, target = action_and_target(source, action_arg)
        question = (f"Original action: {action} toward {target}. Does the rewrite still recommend or describe "
                    f"the same action toward the same target, in any wording?")
        user = f'Rewrite: "{rewrite}"\n{question}\nJSON:'
        answer = None
        try:
            parsed, _ = self.model.complete_json(system=ACTION_SYSTEM, user=user, max_tokens=400, temperature=0.0)
            raw = str((parsed or {}).get("answer", "")).strip().lower()
            answer = True if raw == "yes" else False if raw == "no" else None
        except Exception:
            answer = None
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
