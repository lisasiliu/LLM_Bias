"""Untreated source loading and audit, passage bundles (whole game, stratified, prefix) and condition text assembly."""
from __future__ import annotations
import random
import re
import hashlib
from typing import Dict, List, Optional, Sequence
from components.helpers import SourceRecord
from components.metrics.text_normalization import word_count
from experiments.proficiency_validation.rewrite.runtime import CONDITIONS
from collections import Counter
from pathlib import Path
from components.helpers import TreatedSourceError, sources_from_game_dir
from components.stats import mean, median, percentile, sd
from experiments.proficiency_validation.rewrite.runtime import fmt, with_prior_history


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


def select_stratified(ordered, window_min, window_max, seed):
    """Whole utterances sampled across early, middle and late chronological thirds."""
    strata = [[], [], []]
    for i, record in enumerate(ordered):
        strata[min(2, 3 * i // max(1, len(ordered)))].append(record)
    rng = random.Random(seed)
    for group in strata:
        rng.shuffle(group)
    chosen, words = [], 0
    while any(strata):
        for group in strata:
            if group:
                record = group.pop()
                size = word_count(record.original_nl)
                if words + size <= window_max:
                    chosen.append(record)
                    words += size
        if words >= window_min:
            break
    third_of = {r.source_id: min(2, 3 * i // max(1, len(ordered))) for i, r in enumerate(ordered)}
    if {third_of[r.source_id] for r in chosen} != set(third_of.values()):
        return [], words          # every chronological third must be represented
    return speech_order(chosen), words


def build_bundles(sources: Sequence[SourceRecord], *, window_min: int = 150,
                  window_max: int = 250, max_per_game: Optional[int] = None,
                  seed: int = 101, selection: str = "prefix") -> Dict[str, List[dict]]:
    """``{"bundles": [...], "insufficient": [...], "dropped_by_cap": [...]}``."""
    if selection not in ("prefix", "whole_game", "phase_stratified"):
        raise ValueError("unknown passage selection rule")
    if window_min < 1 or (selection != "whole_game" and window_max < window_min):
        raise ValueError("invalid passage word bounds")
    by_game_speaker: Dict[tuple, List[SourceRecord]] = {}
    for src in sources:
        by_game_speaker.setdefault((src.game_id or "", src.speaker_id), []).append(src)

    bundles: List[dict] = []
    insufficient: List[dict] = []
    for (game, speaker), records in sorted(by_game_speaker.items()):
        ordered = speech_order(records)
        if selection == "whole_game":
            chosen, words = ordered, sum(word_count(r.original_nl) for r in ordered)
        elif selection == "phase_stratified":
            unit_seed = int(hashlib.sha256(f"{seed}:{game}:{speaker}".encode()).hexdigest(), 16)
            chosen, words = select_stratified(ordered, window_min, window_max, unit_seed)
        else:
            chosen, words = select_prefix(ordered, window_min, window_max)
        if not chosen or words < window_min:
            insufficient.append({"game_id": game, "speaker": speaker, "words": words,
                                 "n_utterances": len(ordered),
                                 "reason": "below_window_min" if words < window_min else "chronological_third_uncovered"})
            continue
        bundles.append({
            "bundle_id": f"{game}:{speaker}", "game_id": game, "speaker": speaker,
            "source_ids": [r.source_id for r in chosen], "speech_ids": [r.speech_id for r in chosen],
            "predicates": [r.predicate_name for r in chosen], "phases": sorted({r.phase for r in chosen}),
            "n_utterances": len(chosen), "word_count": words,
            "window": [window_min, None if selection == "whole_game" else window_max],
            **({"selection": selection} if selection != "prefix" else {}),
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
    from experiments.proficiency_validation.rewrite.runtime import assert_unique
    assert_unique(bundles, ("bundle_id",))
    assert_unique(rewrites, ("source_id", "condition"))
    originals: Dict[str, str] = {}
    for row in rewrites:
        by_key[(row["source_id"], row["condition"])] = row
        if row["source_id"] in originals and originals[row["source_id"]] != row.get("original_text", ""):
            raise ValueError(f"conflicting original texts for {row['source_id']}")
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
                        "source_ids": list(bundle["source_ids"]), "selection": bundle.get("selection"),
                        "utterances": texts if not missing else [],
                        "n_utterances": len(texts), "text": " ".join(texts) if not missing else None,
                        "word_count": word_count(" ".join(texts)) if not missing else None})
    return out


def matched_complete(assembled: Sequence[dict], conditions: Sequence[str]) -> List[str]:
    """Bundle ids complete in EVERY listed condition (the matched analysis set)."""
    status: Dict[str, Dict[str, bool]] = {}
    for row in assembled:
        status.setdefault(row["bundle_id"], {})[row["condition"]] = row["complete"]
    return sorted(b for b, s in status.items() if all(s.get(c) for c in conditions))


SPEECH_PREDICATES = {
    "claim_role", "estimate", "claim_divine", "claim_guard", "claim_antidote", "claim_poison",
    "question", "answer", "agree", "disagree", "allied", "unallied", "dissociate",
    "recommend_action", "address_role", "fluff",
}


def load_sources(game_dirs: Sequence[Path], *, allow_treated: bool = False,
                 predicates: Optional[set] = None) -> Dict[str, List[SourceRecord]]:
    out: Dict[str, List[SourceRecord]] = {}
    for game_dir in game_dirs:
        game_dir = Path(game_dir)
        if game_dir.name in out:
            raise ValueError(f"duplicate run directory name: {game_dir.name}; game IDs must be unique")
        try:
            records = sources_from_game_dir(game_dir, predicates or SPEECH_PREDICATES,
                                            allow_treated=allow_treated)
        except TreatedSourceError as exc:
            raise SystemExit(f"refusing treated run: {exc}")
        # Cut each source's history at its own turn: the stored history must not
        # contain the utterance's own predicate or anything said after it.
        out[game_dir.name] = [with_prior_history(r) for r in records]
    return out


def audit(sources_by_game: Dict[str, List[SourceRecord]],
          windows: Sequence[int] = (150, 200, 250), window_span: int = 100) -> dict:
    """Bundle capacity is counted with the SAME greedy speech-order prefix rule that
    ``build_bundles`` applies, for the window ``[w, w + window_span]``; a speaker's
    total word count alone would overstate it."""
    per_game = {}
    all_words: List[int] = []
    speaker_totals: List[int] = []
    predicates: Counter = Counter()
    phases: Counter = Counter()
    bundle_capable = {w: 0 for w in windows}
    total_reaching = {w: 0 for w in windows}
    for game, records in sources_by_game.items():
        words = [word_count(r.original_nl) for r in records]
        all_words += words
        by_speaker: Dict[str, int] = {}
        utts_by_speaker: Counter = Counter()
        records_by_speaker: Dict[str, List[SourceRecord]] = {}
        for r, w in zip(records, words):
            by_speaker[r.speaker_id] = by_speaker.get(r.speaker_id, 0) + w
            utts_by_speaker[r.speaker_id] += 1
            records_by_speaker.setdefault(r.speaker_id, []).append(r)
            predicates[r.predicate_name] += 1
            phases[r.phase] += 1
        speaker_totals += list(by_speaker.values())
        capable = {}
        for w in windows:
            fits = 0
            for speaker, recs in records_by_speaker.items():
                _, got = select_prefix(speech_order(recs), w, w + window_span)
                fits += int(got >= w)
            capable[w] = fits
            bundle_capable[w] += fits
            total_reaching[w] += sum(1 for total in by_speaker.values() if total >= w)
        per_game[game] = {
            "n_utterances": len(records), "n_speakers": len(by_speaker),
            "words_per_utterance_mean": mean(words), "words_per_utterance_median": median(words),
            "words_per_speaker": dict(sorted(by_speaker.items())),
            "utterances_per_speaker": dict(sorted(utts_by_speaker.items())),
            "speakers_reaching_window": capable,
        }
    return {
        "n_games": len(sources_by_game), "n_utterances": len(all_words),
        "words_per_utterance": {"mean": mean(all_words), "sd": sd(all_words),
                                "median": median(all_words), "p10": percentile(all_words, 10),
                                "p90": percentile(all_words, 90)},
        "words_per_speaker": {"mean": mean(speaker_totals), "median": median(speaker_totals),
                              "p10": percentile(speaker_totals, 10),
                              "p90": percentile(speaker_totals, 90)},
        "predicates": dict(predicates.most_common()), "phases": dict(phases.most_common()),
        "candidate_bundles_by_window": bundle_capable,
        "speakers_with_total_words_at_least": total_reaching,
        "windows": list(windows), "window_span": window_span,
        "per_game": per_game,
    }


def render_audit(report: dict) -> str:
    wpu, wps = report["words_per_utterance"], report["words_per_speaker"]
    lines = ["# Untreated-game audit", "",
             f"Games: {report['n_games']}; speech utterances: {report['n_utterances']}.", "",
             "| quantity | mean | median | p10 | p90 |", "| --- | ---: | ---: | ---: | ---: |",
             f"| words per utterance | {fmt(wpu['mean'], 1)} | {fmt(wpu['median'], 1)} | "
             f"{fmt(wpu['p10'], 1)} | {fmt(wpu['p90'], 1)} |",
             f"| words per speaker per game | {fmt(wps['mean'], 1)} | {fmt(wps['median'], 1)} | "
             f"{fmt(wps['p10'], 1)} | {fmt(wps['p90'], 1)} |", "",
             f"| bundle window [min, min+{report['window_span']}] | speakers whose speech-order prefix fits (candidate bundles) | speakers with total words ≥ min |",
             "| ---: | ---: | ---: |"]
    for w in report["windows"]:
        lines.append(f"| [{w}, {w + report['window_span']}] | {report['candidate_bundles_by_window'][w]} | "
                     f"{report['speakers_with_total_words_at_least'][w]} |")
    lines += ["", "| predicate | n |", "| --- | ---: |"]
    for name, n in report["predicates"].items():
        lines.append(f"| {name} | {n} |")
    return "\n".join(lines) + "\n"
