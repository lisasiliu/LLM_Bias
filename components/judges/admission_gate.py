"""The in-game gate: the deterministic structural check, with one model appeal for the action rule.

`structural_check` fails a rewrite that drops a role, speech ID, literal, action or player number the original
named, adds a real hedge (might, maybe, perhaps, ...), or flips agree/disagree. When the action rule fires, one
yes/no question goes to gpt-5.6-luna: does the rewrite still recommend or describe the same action toward the
same target, in any wording? A yes clears that failure. Every other rule has no appeal. The question and the
answer are logged.
"""

from __future__ import annotations

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


class AdmissionGate:
    def __init__(self, model):
        self.model = model

    def _appeal(self, source: SourceRecord, rewrite: str, action_arg: str) -> Dict[str, Any]:
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

    def check(self, source: SourceRecord, rewrite: str, *, addressee: Optional[str] = None) -> Dict[str, Any]:
        det = structural_check(source, rewrite, addressee=addressee)
        reasons: List[str] = list(det.reasons)
        appeals: List[Dict[str, Any]] = []
        for reason in list(reasons):
            if reason.startswith("lost_action_slot:"):
                appeal = self._appeal(source, rewrite, reason.split(":", 1)[1])
                appeals.append(appeal)
                if appeal["answer"] is True:
                    reasons.remove(reason)
        return {"admitted": not reasons, "structural_reasons": list(det.reasons),
                "gate_failure_reasons": reasons, "action_appeals": appeals,
                "identical_to_original": bool(det.details.get("identical_to_original"))}
