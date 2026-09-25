"""Family C listener-side perception (PLAN 9): model probes and a human worksheet.

``probe`` shows each matched pair (default Low vs Neutral) to each listener
model in both presentation orders and records the choice. The listeners are
perception instruments: separate model families from the generator and the
semantic judges, and never told the conditions.

``build-human-worksheet`` / ``import-human-worksheet`` add a small blind human
pairwise judgement in the same format, mirroring the two-rater design of the
politeness validation; the analysis treats a human rater as one more listener.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from components.api_usage import UsageLedger
from components.helpers import read_jsonl, write_jsonl
from components.judges.judge_model import JudgeModel
from components.judges.listener_probe import (
    PROMPT_VERSION, BlindProficiencyProbe, parse_choice, probe_prompt_sha256)

from .bundles import matched_complete
from .common import EXPERIMENT_NAME, dump_json, now_iso

DEFAULT_PAIRS = "low:neutral,neutral:original"
HUMAN_FIELDS = ("more_limited", "confidence", "note")


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


def run_probe(assembled: Sequence[dict], listeners: Dict[str, BlindProficiencyProbe],
              pairs: Sequence[Tuple[str, str]], *, ledger: Optional[UsageLedger] = None,
              limit: Optional[int] = None, log=print) -> List[dict]:
    texts = passages(assembled)
    rows: List[dict] = []
    for a, b in pairs:
        ids = matched_complete(assembled, (a, b))
        if limit:
            ids = ids[:limit]
        for index, bundle_id in enumerate(ids, 1):
            ra, rb = texts[(bundle_id, a)], texts[(bundle_id, b)]
            for handle, probe in listeners.items():
                for order, (left, right) in (("ab", (a, b)), ("ba", (b, a))):
                    left_text = texts[(bundle_id, left)]["text"]
                    right_text = texts[(bundle_id, right)]["text"]
                    result = probe.compare(left_text, right_text)
                    chosen = None
                    if result.ok:
                        chosen = left if result.choice == "A" else right
                    if ledger is not None and result.provenance:
                        ledger.record(probe.model.spec.provider,
                                      result.provenance.get("model_served") or probe.model.spec.model,
                                      "listener_probe", result.provenance.get("usage"))
                    rows.append({
                        "bundle_id": bundle_id, "game_id": ra["game_id"], "speaker": ra["speaker"],
                        "pair": f"{a}:{b}", "treatment": a, "comparator": b,
                        "listener": handle, "listener_family": probe.model.family,
                        "order": order, "left_condition": left, "right_condition": right,
                        "ok": result.ok, "choice": result.choice, "chosen_condition": chosen,
                        "chose_treatment": (chosen == a) if chosen else None,
                        "cannot_tell": False,
                        "confidence": result.confidence, "reason": result.reason,
                        "error": result.error, "prompt_version": PROMPT_VERSION,
                        "model_served": result.provenance.get("model_served"),
                        "created_at": now_iso(),
                    })
            if index % 10 == 0:
                log(f"  probe {a}:{b} {index}/{len(ids)}")
    return rows


# ── human worksheet ──

def build_human_worksheet(assembled: Sequence[dict], pair: Tuple[str, str], *, n: int,
                          seed: int = 101) -> Tuple[str, Dict[str, dict]]:
    a, b = pair
    texts = passages(assembled)
    ids = matched_complete(assembled, (a, b))
    rng = random.Random(seed)
    rng.shuffle(ids)
    ids = ids[:n] if n else ids
    lines = ["# Language-range annotation worksheet", "",
             "Each item shows two passages spoken by two different players in a Werewolf game.",
             "Judge LANGUAGE only, not content: ignore who is accused, whether the reasoning",
             "is good, and how polite or forceful the speaker is. Both passages are grammatical.", "",
             "- `more_limited:` which speaker's English shows the more limited range of vocabulary",
             "  and sentence structure — `A` / `B` / `cannot_tell`",
             "- `confidence:`   `low` / `medium` / `high`",
             "- `note:`         one line, optional", "",
             "Leave the `### item_...` headings untouched.", "", "---", ""]
    key: Dict[str, dict] = {}
    for index, bundle_id in enumerate(ids, 1):
        item_id = f"item_{index:03d}"
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        key[item_id] = {"bundle_id": bundle_id, "game_id": texts[(bundle_id, a)]["game_id"],
                        "pair": f"{a}:{b}", "treatment": a, "comparator": b,
                        "left_condition": left, "right_condition": right}
        lines += [f"### {item_id}", "",
                  f"**Passage A:** {texts[(bundle_id, left)]['text']}", "",
                  f"**Passage B:** {texts[(bundle_id, right)]['text']}", ""]
        lines += [f"{field}:" for field in HUMAN_FIELDS]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", key


LEAK_RE = re.compile(r"^\s*[-*]?\s*(condition|left_condition|right_condition|treatment|comparator|bundle_id)\s*:",
                     re.IGNORECASE)


def find_leaks(worksheet: str) -> List[str]:
    body = worksheet.split("\n---\n", 1)[-1]
    return sorted({line.strip() for line in body.splitlines() if LEAK_RE.match(line)})


def parse_human_worksheet(text: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        heading = re.match(r"^#+\s*(item_\d+)\s*$", line.strip())
        if heading:
            current = heading.group(1)
            out[current] = {}
            continue
        if current is None:
            continue
        match = re.match(r"^(\w+):\s*`?([^`]*)`?\s*$", line.strip())
        if match and match.group(1).lower() in HUMAN_FIELDS and match.group(2).strip():
            out[current][match.group(1).lower()] = match.group(2).strip()
    return {k: v for k, v in out.items() if v.get("more_limited")}


def import_human_worksheet(text: str, key: Dict[str, dict], annotator: str) -> List[dict]:
    rows = []
    for item_id, answers in sorted(parse_human_worksheet(text).items()):
        k = key.get(item_id)
        if not k:
            continue
        raw = answers["more_limited"].strip()
        letter = parse_choice(raw)                   # accepts "A", "Passage B", "(a)" ...
        choice = letter or raw.upper().replace(" ", "_").replace("-", "_")
        chosen = None
        if letter == "A":
            chosen = k["left_condition"]
        elif letter == "B":
            chosen = k["right_condition"]
        abstained = letter is None and choice in ("CANNOT_TELL", "CANNOTTELL", "TIE", "EQUAL", "SAME")
        # An abstention is a VALID response (ok=True) with no chosen condition;
        # the analysis reports it as such rather than dropping it. Only an
        # unrecognised answer is ok=False.
        rows.append({**k, "item_id": item_id, "listener": f"human:{annotator}",
                     "listener_family": "human", "order": "ab" if k["left_condition"] == k["treatment"] else "ba",
                     "ok": chosen is not None or abstained,
                     "choice": choice if choice in ("A", "B") else ("cannot_tell" if abstained else None),
                     "chosen_condition": chosen,
                     "chose_treatment": (chosen == k["treatment"]) if chosen else None,
                     "cannot_tell": abstained,
                     "confidence": answers.get("confidence", "").lower(),
                     "reason": answers.get("note", ""), "created_at": now_iso()})
    return rows


# ── commands ──

def add_parser(sub) -> None:
    p = sub.add_parser("probe", help="blind pairwise listener probe with order counterbalancing")
    p.add_argument("--assembled", type=Path, required=True)
    p.add_argument("--listeners", required=True,
                   help="comma-separated JudgeModel handles, e.g. deepseek-chat,openai:gpt-5-mini")
    p.add_argument("--pairs", default=DEFAULT_PAIRS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=400,
                   help="completion budget per comparison; raise for reasoning listeners")
    p.add_argument("--run-id", default=None)
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify hashes and pin the listener set")
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=main_probe)

    w = sub.add_parser("build-human-worksheet", help="blind pairwise worksheet for a human rater")
    w.add_argument("--assembled", type=Path, required=True)
    w.add_argument("--pair", default="low:neutral")
    w.add_argument("--n", type=int, default=48)
    w.add_argument("--seed", type=int, default=101)
    w.add_argument("--out", type=Path, required=True)
    w.add_argument("--key", type=Path, required=True)
    w.set_defaults(func=main_worksheet)

    i = sub.add_parser("import-human-worksheet", help="merge a filled worksheet into choice rows")
    i.add_argument("--worksheet", type=Path, required=True)
    i.add_argument("--key", type=Path, required=True)
    i.add_argument("--annotator", default="rater_1")
    i.add_argument("--out", type=Path, required=True)
    i.set_defaults(func=main_import)


def main_probe(args) -> int:
    if args.frozen:
        from .frozen import enforce_frozen
        frozen = enforce_frozen(args.frozen)
        pinned = sorted(frozen["models"]["listeners"])
        given = sorted(h.strip() for h in args.listeners.split(",") if h.strip())
        if given != pinned:
            print(f"--listeners {given} differs from the frozen set {pinned}", file=sys.stderr)
            return 2
    assembled = read_jsonl(args.assembled)
    pairs = parse_pairs(args.pairs)
    listeners: Dict[str, BlindProficiencyProbe] = {}
    for handle in [h.strip() for h in args.listeners.split(",") if h.strip()]:
        model = JudgeModel.resolve(handle)
        ok, msg = model.preflight()
        if not ok:
            print(f"listener {handle} preflight failed: {msg}", file=sys.stderr)
            return 1
        listeners[handle] = BlindProficiencyProbe(model, max_tokens=args.max_tokens)
    run_id = args.run_id or f"probe_{now_iso().replace(':', '')}"
    ledger = UsageLedger(EXPERIMENT_NAME, run_id=run_id)
    rows = run_probe(assembled, listeners, pairs, ledger=ledger, limit=args.limit)
    write_jsonl(args.out, rows)
    ledger.write()
    dump_json(args.out.with_name(args.out.stem + "_provenance.json"), {
        "run_id": run_id, "created_at": now_iso(), "pairs": [f"{a}:{b}" for a, b in pairs],
        "prompt_version": PROMPT_VERSION, "prompt_sha256": probe_prompt_sha256(),
        "listeners": {h: p.model.provenance() for h, p in listeners.items()},
        "frozen_config": str(args.frozen) if args.frozen else None,
        "usage": ledger.totals()})
    print(f"{len(rows)} listener choices -> {args.out}")
    return 0


def main_worksheet(args) -> int:
    assembled = read_jsonl(args.assembled)
    a, b = parse_pairs(args.pair)[0]
    worksheet, key = build_human_worksheet(assembled, (a, b), n=args.n, seed=args.seed)
    leaks = find_leaks(worksheet)
    if leaks:
        print(f"refusing to write a leaking worksheet: {leaks}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(worksheet, encoding="utf-8")
    args.key.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(key)} items -> {args.out} (key: {args.key}; do not show the key to the rater)")
    return 0


def main_import(args) -> int:
    key = json.loads(args.key.read_text(encoding="utf-8"))
    rows = import_human_worksheet(args.worksheet.read_text(encoding="utf-8"), key, args.annotator)
    if not rows:
        print("no completed items found", file=sys.stderr)
        return 2
    write_jsonl(args.out, rows)
    print(f"{len(rows)} human choices -> {args.out}")
    return 0
