"""Export the replays' discussion speech as passages for the listener validation.

    python -m experiments.language_flip.code.export_passages [--games DIR] [--out JSONL]

One bundle per replay and treated speaker: the ``original`` row holds the source utterances
of that speaker's discussion turns in that replay, the condition row holds what the other
players heard (rewrites, or the original after a gate fallback). The rows are the
``--assembled`` input of ``experiments.language_validation.validation probe``, which
compares ``low:original``, ``neutral:original`` and ``high:original`` within a bundle.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import List

from components.metrics.text_normalization import word_count
from experiments.language_flip.code import analyze as fa
from experiments.language_flip.code.figures import delivered_utterances

EXPORT_PATH = fa.RUN_DIR / "replay_passages.jsonl"


def passages(utterances: List[dict]) -> List[dict]:
    groups = defaultdict(list)
    for u in utterances:
        groups[(u["seed"], u["day"], u["mode"], u["self"], u["speaker"], u["condition"])].append(u)
    rows = []
    for (seed, day, mode, self_p, speaker, condition), items in sorted(groups.items()):
        bundle = f"{seed}:d{day}:{mode}:{self_p.lower()}:{speaker}:{condition}"
        for label, side in ((condition, "delivered"), ("original", "source")):
            texts = [u[side] for u in items if u[side]]
            rows.append({"bundle_id": bundle, "game_id": seed, "day": day, "mode": mode, "self_player": self_p,
                         "speaker": speaker, "condition": label, "replay": items[0]["replay"],
                         "complete": bool(texts), "n_utterances": len(texts),
                         "n_fallbacks": sum(u["fallback"] for u in items) if side == "delivered" else 0,
                         "utterances": texts, "text": " ".join(texts) if texts else None,
                         "word_count": word_count(" ".join(texts)) if texts else None})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=Path, default=fa.GAMES_DIR)
    parser.add_argument("--out", type=Path, default=EXPORT_PATH)
    args = parser.parse_args(argv)
    rows = passages(delivered_utterances(args.games))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    bundles = len({r["bundle_id"] for r in rows})
    print(f"{bundles} bundles, {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
