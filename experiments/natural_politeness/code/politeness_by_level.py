"""Politeness of untreated speech across the five prompt levels (L1-L5).

    python -m experiments.natural_politeness.code.politeness_by_level extract
    python -m experiments.natural_politeness.code.politeness_by_level analyze

``extract`` reads the untreated game runs in ``games_fresh/prompt_levels/<level>/``
(gitignored) and writes every primary-predicate utterance (D009) to
``data/utterances.jsonl``. ``analyze`` reads only the committed files in ``data/``
and prints every number in ``summary.md``. Nothing here calls a model.

``data/labelled_sample.jsonl`` is the 30-per-level random sample and its strategy
labels. The sample was drawn once and is recorded as data rather than redrawn:
its shuffle was seeded through Python's per-process ``hash()``, so it cannot be
regenerated.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from components.helpers import read_jsonl, sources_from_game_dir, write_jsonl
from components.rewriters.politeness_rewriter import PRIMARY_PREDICATES

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
UTTERANCES = DATA_DIR / "utterances.jsonl"
SAMPLE = DATA_DIR / "labelled_sample.jsonl"
GAMES = EXPERIMENT_DIR.parents[1] / "games_fresh" / "prompt_levels"

LEVELS = ("L1", "L2", "L3", "L4", "L5")
STRATEGIES = ("bald_on_record", "positive_politeness", "negative_politeness",
              "off_record")


def extract(games: Path = GAMES) -> int:
    records = []
    for level in LEVELS:
        for game_dir in sorted((games / level).iterdir()):
            roles = json.loads((game_dir / "game_record.json").read_text())[
                "hidden_info"]["role_distribution"]
            for s in sources_from_game_dir(game_dir, set(PRIMARY_PREDICATES)):
                records.append({
                    "id": f"{level}/{s.source_id}",
                    "level": level,
                    "game": s.game_id,
                    "speech_id": s.speech_id,
                    "phase": s.phase,
                    "speaker": s.speaker_id,
                    "role": roles.get(s.speaker_id),
                    "predicate": s.predicate_name,
                    "predicate_text": s.predicate_text,
                    "text": s.original_nl,
                })
    write_jsonl(UTTERANCES, records)
    return len(records)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    centre = (k + z * z / 2) / (n + z * z)
    half = z * math.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return max(0.0, centre - half), min(1.0, centre + half)


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def analyze() -> None:
    utterances = read_jsonl(UTTERANCES)
    sample = read_jsonl(SAMPLE)

    print("## Primary-predicate utterances per level\n")
    rows = []
    for level in LEVELS:
        items = [u for u in utterances if u["level"] == level]
        games = len({u["game"] for u in items})
        mix = Counter(u["predicate"] for u in items)
        rows.append([level, games, len(items)]
                    + [mix[p] for p in PRIMARY_PREDICATES]
                    + [f"{mix['question'] / games:.1f}"])
    print(table(["level", "games", "n", *PRIMARY_PREDICATES, "question/game"], rows))

    print("\n## Strategy of the labelled sample\n")
    rows = []
    for level in (*LEVELS, "all"):
        items = [s for s in sample if level in ("all", s["level"])]
        labels = Counter(s["label"] for s in items)
        lo, hi = wilson(labels["bald_on_record"], len(items))
        rows.append([level, len(items)] + [labels[s] for s in STRATEGIES]
                    + [f"{100 * lo:.1f}-{100 * hi:.1f}%"])
    print(table(["level", "n", *STRATEGIES, "bald 95% CI"], rows))

    print("\n## Non-bald items\n")
    for s in sample:
        if s["label"] != "bald_on_record":
            print(f"- {s['sample_id']} ({s['predicate']}, {s['role']}) "
                  f"{s['label']}: {s['text']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("extract", "analyze"))
    args = parser.parse_args()
    if args.command == "extract":
        print(f"wrote {extract()} utterances to {UTTERANCES}")
    else:
        analyze()


if __name__ == "__main__":
    main()
