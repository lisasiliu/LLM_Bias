"""The blind listeners' answers about the replays' speech, summarized for the report.

`experiments.proficiency_validation.validation probe` writes one row per comparison;
this reads those rows back and reduces them to the numbers the summary prints.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from components.helpers import read_jsonl

CONSTRUCTS = {"range": "range", "perceived_proficiency": "perceived proficiency"}


def listener_summary(validation_dir: Path) -> Dict[str, Dict[str, dict]]:
    """``{construct: {f"{pair} · {listener}": numbers}}`` over every probe file present."""
    out: Dict[str, Dict[str, dict]] = {}
    for path in sorted(validation_dir.glob("*_listener_*.jsonl")):
        construct = next((c for c in CONSTRUCTS if path.name.startswith(c)), None)
        if construct is None:
            continue
        rows = [r for r in read_jsonl(path) if r.get("ok")]
        if not rows:
            continue
        by_pair: Dict[tuple, List[dict]] = defaultdict(list)
        for r in rows:
            by_pair[(r["pair"], r["listener"])].append(r)
        for (pair, listener), rs in sorted(by_pair.items()):
            signed = [r["signed"] for r in rs if r["signed"] is not None]
            per_passage: Dict[str, List[int]] = defaultdict(list)
            for r in rs:
                per_passage[r["bundle_id"]].append(r["signed"])
            both = [v for v in per_passage.values() if len(v) == 2]
            consistent = sum(1 for v in both if v[0] == v[1] and v[0] != 0)
            decided = [r for r in rs if not r["tie"]]
            first_chosen = sum(1 for r in decided if r.get("position") == "A")
            by_game: Dict[str, List[int]] = defaultdict(list)
            for r in rs:
                by_game[r["game_id"]].append(r["signed"])
            game_means = sorted(statistics.mean(v) for v in by_game.values())
            out.setdefault(construct, {})[f"{pair} · {listener}"] = {
                "n": len(rs), "mean_signed": statistics.mean(signed) if signed else None,
                "tie_share": sum(r["tie"] for r in rs) / len(rs),
                "order_consistent": consistent, "passages": len(both),
                "decided": len(decided),
                "first_position_share": (first_chosen / len(decided)) if decided else None,
                "n_games": len(game_means),
                "game_min": game_means[0] if game_means else None,
                "game_max": game_means[-1] if game_means else None}
    return out


def cross_listener_agreement(validation_dir: Path) -> Dict[str, dict]:
    """How often two listeners give the same answer to the same comparison."""
    out: Dict[str, dict] = {}
    for construct in CONSTRUCTS:
        answers: Dict[tuple, Dict[str, int]] = {}
        for path in sorted(validation_dir.glob(f"{construct}_listener_*.jsonl")):
            for r in read_jsonl(path):
                if r.get("ok"):
                    answers.setdefault((r["bundle_id"], r["pair"], r["order"]), {})[r["listener"]] = r["signed"]
        pairs = [v for v in answers.values() if len(v) == 2]
        if pairs:
            same = sum(1 for v in pairs if len(set(v.values())) == 1)
            out[construct] = {"n": len(pairs), "agreement": same / len(pairs),
                              "listeners": sorted({h for v in pairs for h in v})}
    return out


def render(summary: Dict[str, Dict[str, dict]], agreement: Optional[Dict[str, dict]] = None) -> List[str]:
    if not summary:
        return []
    lines = ["", "## What blind listeners hear in the replays' speech", "",
             "Each comparison shows one listener two passages from the same replay and speaker, the "
             "delivered speech and its source, in both presentation orders, and asks which is the more "
             "limited. A positive mean means the treated passage was chosen. `first chosen` counts only the "
             "answers that named a passage: 0.5 there is no position bias.", ""]
    for construct, block in summary.items():
        lines += [f"### {CONSTRUCTS[construct]}", "",
                  "| comparison · listener | answers | mean | ties | answers naming a passage | same answer in both orders | first chosen | per-game range |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for key, s in block.items():
            first = f"{s['first_position_share']:.0%}" if s.get("first_position_share") is not None else "–"
            lines.append(f"| {key} | {s['n']} | {s['mean_signed']:+.3f} | {s['tie_share']:.0%} | "
                         f"{s['decided']} | {s['order_consistent']}/{s['passages']} | {first} | "
                         f"{s['game_min']:+.2f} to {s['game_max']:+.2f} |")
        lines.append("")
    if agreement:
        lines += ["Two listeners from different model families answered every comparison:", "",
                  "| construct | comparisons | identical answer |", "|---|--:|--:|"]
        for construct, a in agreement.items():
            lines.append(f"| {CONSTRUCTS[construct]} | {a['n']} | {a['agreement']:.1%} |")
        lines.append("")
    return lines
