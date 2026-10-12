"""Blind pairwise listener probes for perceived linguistic range and perceived English proficiency."""
from __future__ import annotations
import hashlib
from components.judges.judge_model import JudgeModelError, JudgeTransportError
from components.judges.listener_probe import ProbeResult, parse_choice, build_user_prompt, system_prompt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from components.api_usage import UsageLedger
from components.helpers import read_jsonl
from experiments.proficiency_validation.rewrite.runtime import ValidationModel as JudgeModel
from experiments.proficiency_validation.rewrite.runtime import artifact_provenance, record_artifact, verify_artifact
from experiments.proficiency_validation.validation.protocol import cohort
from experiments.proficiency_validation.rewrite.runtime import assert_unique
from experiments.proficiency_validation.validation.sources import matched_complete
from experiments.proficiency_validation.rewrite.runtime import EXPERIMENT_NAME, now_iso


CONSTRUCTS = ("range", "perceived_proficiency")
VERSIONS = {"range": "range_pairwise", "perceived_proficiency": "perceived_proficiency_pairwise"}
THREE_WAY = ('Answer "A" if passage A shows the more limited range, "B" if passage B does, or "same" when you\n'
             'cannot find a meaningful difference between them. Do not force a choice.')
RANGE_PROMPT = (system_prompt()
    .replace("two short passages, A and B,\neach spoken by a different player during a Werewolf (Mafia) game.",
             "two transcripts, A and B. Each transcript contains one player's speech from a Werewolf game.")
    .replace("If they seem equal, still choose the one that is\neven slightly more limited.", THREE_WAY)
    .replace('{"more_limited": "A" | "B",', '{"more_limited": "A" | "B" | "same",'))
assert THREE_WAY in RANGE_PROMPT and '"same"' in RANGE_PROMPT, "three-way range prompt did not apply"
PROFICIENCY_PROMPT = """You are a blind rater in a study of perceived English proficiency.
You will read two transcripts, A and B. Each contains one player's speech from a
Werewolf game. Which speaker appears to have a weaker command of English from
their use of language across the transcript?

Make a holistic judgment of the impression conveyed by the English. Consider how
clearly and flexibly the speaker expresses relationships and qualifications, and
how effectively the wording and constructions fit their intended meaning. Assess
the whole transcript. Correct grammar and simple sentences alone do not determine
the answer. Ignore strategic skill, the truth of claims, persuasiveness, politeness,
player identities and how much the speaker talks. Do not assume anything about
nationality, first language or education. The judgment concerns the impression
from this sample, not an established ability level or a CEFR classification.

Answer "A" or "B" for the speaker with the weaker command of English, or "same" when you
cannot find a meaningful difference; do not force a choice. Return JSON only:
{"more_limited": "A" | "B" | "same", "confidence": "low" | "medium" | "high",
 "reason": "brief explanation about the English"}
"""

SAME_ANSWERS = {"same", "neither", "equal", "no difference", "no meaningful difference", "tie"}


def parse_three_way(value):
    """``"A"`` / ``"B"`` / ``"same"`` from the model's answer; ``None`` when unreadable."""
    choice = parse_choice(value)
    if choice:
        return choice
    text = " ".join(str(value or "").strip().lower().split()).strip('.!"\' ')
    if text in SAME_ANSWERS:
        return "same"
    return None


def prompt(construct):
    if construct not in CONSTRUCTS:
        raise ValueError(f"unknown perception construct: {construct}")
    base = RANGE_PROMPT if construct == "range" else PROFICIENCY_PROMPT
    return base + "\nCheck that every example in your explanation belongs to the chosen text. Presentation position provides no linguistic evidence. Keep the reason within 40 words and return the complete JSON object."


def prompt_sha(construct="range"):
    return hashlib.sha256(prompt(construct).encode()).hexdigest()


class BlindLanguageProbe:
    def __init__(self, model, max_tokens=768, construct="range"):
        self.model, self.max_tokens, self.construct = model, int(max_tokens), construct
        self.system = prompt(construct)
        self.prompt_version = VERSIONS[construct]

    def compare(self, text_a, text_b):
        if not text_a.strip() or not text_b.strip():
            return ProbeResult(ok=False, error="empty_passage")
        try:
            parsed, completion = self.model.complete_json(system=self.system,
                user=build_user_prompt(text_a, text_b), max_tokens=self.max_tokens)
        except (JudgeModelError, JudgeTransportError) as exc:
            return ProbeResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        provenance = {**completion.to_dict(), "construct": self.construct, "prompt_version": self.prompt_version}
        if not isinstance(parsed, dict):
            return ProbeResult(ok=False, error="unparseable_json", provenance=provenance)
        provenance["raw_choice"] = parsed.get("more_limited")
        choice = parse_three_way(parsed.get("more_limited"))
        if choice is None:
            return ProbeResult(ok=False, error="unknown_choice", provenance=provenance)
        return ProbeResult(ok=True, choice=choice, confidence=str(parsed.get("confidence", "")),
                           reason=str(parsed.get("reason", "")), provenance=provenance)


DEFAULT_PAIRS = "low:neutral,high:neutral,low:high,neutral:original"


def parse_pairs(spec: str) -> List[Tuple[str, str]]:
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split(":")
        pairs.append((a.strip(), b.strip()))
    return pairs


def passages(assembled: Sequence[dict]) -> Dict[Tuple[str, str], dict]:
    return {(r["bundle_id"], r["condition"]): r for r in assembled if r.get("complete") and r.get("text")}


def run_probe(assembled: Sequence[dict], listeners: Dict[str, BlindLanguageProbe],
              pairs: Sequence[Tuple[str, str]], *, ledger: Optional[UsageLedger] = None,
              limit: Optional[int] = None, log=print, completed_rows=None, on_row=None,
              concurrency: int = 1) -> List[dict]:
    """Every (pair, passage, listener, order) comparison once. Model calls run on ``concurrency``
    threads; rows are appended to the checkpoint from the main thread as they complete."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    texts = passages(assembled)
    prior = list(completed_rows or [])
    assert_unique(prior, ("bundle_id", "pair", "listener", "order"))
    done = {(r["bundle_id"], r["pair"], r["listener"], r["order"]): r for r in prior}
    tasks, reused = [], set()
    for a, b in pairs:
        ids = matched_complete(assembled, (a, b))
        if limit:
            ids = ids[:limit]
        for bundle_id in ids:
            for handle, probe in listeners.items():
                for order, (left, right) in (("ab", (a, b)), ("ba", (b, a))):
                    key = (bundle_id, f"{a}:{b}", handle, order)
                    if key in done:
                        reused.add(key)
                    else:
                        tasks.append((key, a, b, bundle_id, handle, probe, order, left, right))
    if reused != set(done):
        raise ValueError("checkpoint contains unexpected probe rows")

    def passage_text(bundle_id, condition):
        row = texts[(bundle_id, condition)]
        return "\n\n".join(row["utterances"]) if row.get("utterances") else row["text"]

    def make_row(task, result):
        key, a, b, bundle_id, handle, probe, order, left, right = task
        ra = texts[(bundle_id, a)]
        chosen, signed = None, None
        if result.ok and result.choice == "same":
            signed = 0                       # no meaningful difference: a tie, not a missing answer
        elif result.ok:
            chosen = left if result.choice == "A" else right
            signed = 1 if chosen == a else -1
        if ledger is not None and result.provenance:
            ledger.record(probe.model.spec.provider, result.provenance.get("model_served") or probe.model.spec.model,
                          "listener_probe", result.provenance.get("usage"))
        return {"bundle_id": bundle_id, "game_id": ra["game_id"], "speaker": ra["speaker"],
                "pair": f"{a}:{b}", "treatment": a, "comparator": b,
                "listener": handle, "listener_family": probe.model.family,
                "order": order, "left_condition": left, "right_condition": right,
                "ok": result.ok, "choice": result.choice, "position": result.choice if result.ok else None,
                "chosen_condition": chosen, "tie": bool(result.ok and result.choice == "same"),
                "signed": signed, "chose_treatment": (chosen == a) if chosen else None,
                "confidence": result.confidence, "reason": result.reason,
                "error": result.error, "construct": probe.construct,
                "prompt_version": probe.prompt_version,
                "model_served": result.provenance.get("model_served"),
                "provenance": result.provenance, "max_tokens": probe.max_tokens,
                "created_at": now_iso()}
    produced: Dict[tuple, dict] = {}
    finished = 0
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        futures = {pool.submit(task[5].compare, passage_text(task[3], task[7]), passage_text(task[3], task[8])): task
                   for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            row = make_row(task, future.result())
            produced[task[0]] = row
            if on_row:
                on_row(row)
            finished += 1
            if finished % 20 == 0:
                log(f"  probe {finished}/{len(tasks)}")
    rows: List[dict] = []
    for a, b in pairs:
        ids = matched_complete(assembled, (a, b))
        if limit:
            ids = ids[:limit]
        for bundle_id in ids:
            for handle in listeners:
                for order in ("ab", "ba"):
                    key = (bundle_id, f"{a}:{b}", handle, order)
                    rows.append(done.get(key) or produced[key])
    return rows


# ── commands ──

def add_parser(sub) -> None:
    p = sub.add_parser("probe", help="blind pairwise listener probe with order counterbalancing")
    p.add_argument("--assembled", type=Path, required=True)
    p.add_argument("--listeners", required=True,
                   help="comma-separated JudgeModel handles, e.g. deepseek-chat,openai:gpt-5-mini")
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--construct", choices=CONSTRUCTS, default="range")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=768,
                   help="completion budget per comparison; raise for reasoning listeners")
    p.add_argument("--concurrency", type=int, default=8, help="parallel comparison requests")
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--retry-failed", action="store_true", help="with --resume: ask again for comparisons whose earlier response failed")
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify hashes and pin the listener set")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=main_probe)

def main_probe(args) -> int:
    frozen = None
    if args.frozen:
        from experiments.proficiency_validation.validation.protocol import enforce_frozen
        frozen = enforce_frozen(args.frozen, check_models=True)
        pinned = sorted(frozen["models"]["listeners"])
        given = sorted(h.strip() for h in args.listeners.split(",") if h.strip())
        if given != pinned:
            print(f"--listeners {given} differs from the frozen set {pinned}", file=sys.stderr)
            return 2
        cohort(frozen, args.split)
        if args.construct not in frozen["perception_constructs"]:
            raise ValueError("perception construct is outside the frozen design")
        args.pairs = ",".join([f"{c}:neutral" for c in ("low", "high") if c in frozen["conditions"]]
                              + (["low:high"] if "high" in frozen["conditions"] else []) + ["neutral:original"])
        args.max_tokens = frozen["probe_max_tokens"]
        if args.limit:
            raise ValueError("a frozen probe cannot use --limit")
        _, errors = verify_artifact(args.assembled, frozen, args.split)
        if errors:
            raise ValueError("; ".join(errors))
    assembled = read_jsonl(args.assembled)
    pairs = parse_pairs(args.pairs)
    listeners: Dict[str, BlindLanguageProbe] = {}
    for handle in [h.strip() for h in args.listeners.split(",") if h.strip()]:
        model = JudgeModel.resolve(handle)
        ok, msg = model.preflight()
        if not ok:
            print(f"listener {handle} preflight failed: {msg}", file=sys.stderr)
            return 1
        listeners[handle] = BlindLanguageProbe(model, max_tokens=args.max_tokens, construct=args.construct)
    run_id = args.run_id or f"probe_{now_iso().replace(':', '')}"
    ledger = UsageLedger(EXPERIMENT_NAME, run_id=run_id)
    from experiments.proficiency_validation.rewrite.runtime import Checkpoint
    from experiments.proficiency_validation.validation.protocol import frozen_hashes
    settings = {"artifact": artifact_provenance(frozen, args.split, inputs={"assembled": args.assembled}),
                "hashes": frozen_hashes(), "pairs": pairs, "construct": args.construct, "max_tokens": args.max_tokens,
                "limit": args.limit, "listeners": {h: p.model.identity_record for h, p in listeners.items()}}
    # JSON normalization keeps tuples stable across resumed processes.
    checkpoint = Checkpoint(args.out, json.loads(json.dumps(settings)), resume=args.resume)
    try:
        if getattr(args, "retry_failed", False):
            dropped = checkpoint.reopen(lambda r: r.get("ok"))
            print(f"re-asking {dropped} failed comparisons")
        rows = run_probe(assembled, listeners, pairs, ledger=ledger, limit=args.limit,
                         completed_rows=checkpoint.rows, on_row=checkpoint.append,
                         concurrency=getattr(args, "concurrency", 1))
        checkpoint.finish()
    finally:
        checkpoint.close()
    ledger.write()
    record_artifact(args.out, {
        **artifact_provenance(frozen, args.split, inputs={"assembled": args.assembled}),
        "run_id": run_id, "created_at": now_iso(), "pairs": [f"{a}:{b}" for a, b in pairs],
        "construct": args.construct, "prompt_version": VERSIONS[args.construct], "prompt_sha256": prompt_sha(args.construct),
        "listeners": {h: p.model.provenance() for h, p in listeners.items()},
        "frozen_config": str(args.frozen) if args.frozen else None,
        "usage": ledger.totals()})
    print(f"{len(rows)} listener choices -> {args.out}")
    return 0
