"""Bundle construction (PLAN 5): one matched speaker-passage per condition.

A source bundle is the ordered set of one speaker's utterances in one game up
to a fixed word window. Every condition (Original / Neutral / Low / High) is
assembled from the SAME source utterances, so each bundle id names a matched
comparison. A speaker contributes at most one bundle per game; games contribute
at most ``max_per_game`` bundles, chosen with a seeded RNG so the sample spreads
across games rather than exhausting a few.

Utterance-level semantic QC happens before this step (generate.py); a bundle is
``complete`` for a condition only when every one of its utterances has an
accepted rewrite in that condition. Incomplete bundles are excluded from the
matched analysis and counted in the QC report.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence

from components.helpers import SourceRecord
from components.metrics.text_normalization import word_count

from .common import CONDITIONS


def _speech_number(record: SourceRecord) -> int:
    match = re.search(r"s(\d+)", record.speech_id or "")
    return int(match.group(1)) if match else 10 ** 9


def select_prefix(ordered: Sequence[SourceRecord], window_min: int, window_max: int
                  ) -> tuple:
    """``(chosen, words)``: the longest speech-order prefix that reaches ``window_min``
    without exceeding ``window_max``; ``words < window_min`` means no bundle fits."""
    chosen: List[SourceRecord] = []
    words = 0
    for rec in ordered:
        w = word_count(rec.original_nl)
        if words + w > window_max:
            # Never exceed the upper bound: a passage longer than the window
            # would not be length-matched to the others. If the bundle is
            # still below the minimum here, the speaker is insufficient.
            break
        chosen.append(rec)
        words += w
        if words >= window_min:
            break
    return chosen, words


def speech_order(records: Sequence[SourceRecord]) -> List[SourceRecord]:
    return sorted(records, key=lambda r: (_speech_number(r), r.source_id))


def build_bundles(sources: Sequence[SourceRecord], *, window_min: int = 150,
                  window_max: int = 250, max_per_game: Optional[int] = None,
                  seed: int = 101) -> Dict[str, List[dict]]:
    """``{"bundles": [...], "insufficient": [...], "dropped_by_cap": [...]}``."""
    by_game_speaker: Dict[tuple, List[SourceRecord]] = {}
    for src in sources:
        by_game_speaker.setdefault((src.game_id or "", src.speaker_id), []).append(src)

    bundles: List[dict] = []
    insufficient: List[dict] = []
    for (game, speaker), records in sorted(by_game_speaker.items()):
        ordered = speech_order(records)
        chosen, words = select_prefix(ordered, window_min, window_max)
        if words < window_min:
            insufficient.append({"game_id": game, "speaker": speaker, "words": words,
                                 "n_utterances": len(ordered)})
            continue
        bundles.append({
            "bundle_id": f"{game}:{speaker}", "game_id": game, "speaker": speaker,
            "source_ids": [r.source_id for r in chosen], "speech_ids": [r.speech_id for r in chosen],
            "predicates": [r.predicate_name for r in chosen], "phases": sorted({r.phase for r in chosen}),
            "n_utterances": len(chosen), "word_count": words,
            "window": [window_min, window_max],
        })

    dropped: List[dict] = []
    if max_per_game is not None:
        rng = random.Random(seed)
        kept: List[dict] = []
        by_game: Dict[str, List[dict]] = {}
        for b in bundles:
            by_game.setdefault(b["game_id"], []).append(b)
        for game in sorted(by_game):
            candidates = sorted(by_game[game], key=lambda b: b["bundle_id"])
            if len(candidates) > max_per_game:
                selected = rng.sample(candidates, max_per_game)
                dropped += [b for b in candidates if b not in selected]
                candidates = selected
            kept += sorted(candidates, key=lambda b: b["bundle_id"])
        bundles = kept
    return {"bundles": bundles, "insufficient": insufficient, "dropped_by_cap": dropped}


def assemble_condition_texts(bundles: Sequence[dict], rewrites: Sequence[dict],
                             conditions: Sequence[str] = CONDITIONS) -> List[dict]:
    """Per bundle and condition: the concatenated passage, or ``complete=False``.

    ``rewrites`` rows come from generate.py (one per source x condition) and
    carry ``original_text``; the ``original`` condition is assembled from them.
    """
    by_key: Dict[tuple, dict] = {}
    originals: Dict[str, str] = {}
    for row in rewrites:
        by_key[(row["source_id"], row["condition"])] = row
        originals[row["source_id"]] = row.get("original_text", "")
    out: List[dict] = []
    for bundle in bundles:
        for condition in conditions:
            texts: List[str] = []
            missing: List[str] = []
            for sid in bundle["source_ids"]:
                if condition == "original":
                    text = originals.get(sid)
                else:
                    row = by_key.get((sid, condition))
                    text = row.get("rewritten_text") if row and row.get("status") == "ok" else None
                if text:
                    texts.append(text)
                else:
                    missing.append(sid)
            out.append({"bundle_id": bundle["bundle_id"], "game_id": bundle["game_id"],
                        "speaker": bundle["speaker"], "condition": condition,
                        "complete": not missing, "missing_source_ids": missing,
                        "n_utterances": len(texts), "text": " ".join(texts) if not missing else None,
                        "word_count": word_count(" ".join(texts)) if not missing else None})
    return out


def matched_complete(assembled: Sequence[dict], conditions: Sequence[str]) -> List[str]:
    """Bundle ids complete in EVERY listed condition (the matched analysis set)."""
    status: Dict[str, Dict[str, bool]] = {}
    for row in assembled:
        status.setdefault(row["bundle_id"], {})[row["condition"]] = row["complete"]
    return sorted(b for b, s in status.items() if all(s.get(c) for c in conditions))
