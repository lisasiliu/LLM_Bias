"""Untreated-game audit (PLAN 16.8): what the raw corpus can support.

For a set of LURK game directories, refuses treated runs, then reports per
game and overall: speakers, utterances, words per utterance, words per speaker,
predicate composition, and how many speakers could yield a bundle under each
candidate word window. The pilot uses this to fix the bundle window (PLAN 5).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from components.helpers import SourceRecord, TreatedSourceError, sources_from_game_dir
from components.metrics.text_normalization import word_count
from components.stats import mean, median, percentile, sd

from .bundles import select_prefix, speech_order
from .common import fmt, with_prior_history

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
        try:
            records = sources_from_game_dir(game_dir, predicates or SPEECH_PREDICATES,
                                            allow_treated=allow_treated)
        except TreatedSourceError as exc:
            raise SystemExit(f"refusing treated run: {exc}")
        # Cut each source's history at its own turn: the stored history must not
        # contain the utterance's own predicate or anything said after it.
        out[game_dir.name] = [with_prior_history(r) for r in records]
    return out


def roles_from_manifest(game_dir: Path) -> Dict[str, str]:
    manifest = game_dir / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    roles = data.get("roles") or data.get("role_assignment") or {}
    return {str(k): str(v) for k, v in roles.items()} if isinstance(roles, dict) else {}


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
