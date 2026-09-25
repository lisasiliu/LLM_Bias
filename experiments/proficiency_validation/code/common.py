"""Paths, constants and small helpers shared by the proficiency-validation commands."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from components.helpers import SourceRecord, parse_history_entry, read_jsonl, write_jsonl  # noqa: F401

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
FROZEN_CONFIG = DATA_DIR / "frozen_config.json"

EXPERIMENT_NAME = "proficiency_validation"

# ``original`` is the untreated Stage-2 text; the other three are rewriter conditions.
CONDITIONS: tuple = ("original", "neutral", "low", "high")
TREATED_CONDITIONS: tuple = ("neutral", "low", "high")
CORE_CONDITIONS: tuple = ("original", "neutral", "low")

# Planned contrasts as (treatment, comparator, role).
CONTRASTS: tuple = (
    ("low", "neutral", "primary"),
    ("neutral", "original", "control"),
    ("low", "original", "secondary"),
    ("high", "neutral", "primary when included"),
)

# Family A metrics with the direction expected for Low relative to Neutral.
# +1: Low higher, -1: Low lower. ``word_count`` is a nuisance diagnostic.
RANGE_METRICS: Dict[str, int] = {
    "subtlex_mean": +1,
    "mattr": -1,
    "mtld": -1,
    "repetition_rate": +1,
    "top5_share": +1,
    "clause_pattern_entropy": -1,
    "clause_pattern_distinct_ratio": -1,
    "subordinate_clauses_per_sentence": -1,
    "mean_dependency_distance": -1,
    "tree_depth_mean": -1,
    "np_mean_length": -1,
    "mean_sentence_length": -1,
}
PRIMARY_RANGE_METRICS: tuple = ("subtlex_mean", "mattr", "subordinate_clauses_per_sentence")
NUISANCE_METRICS: tuple = ("word_count", "n_sentences", "edit_distance_to_original")

# Family B guardrail metrics (equivalence, not difference).
GUARDRAIL_METRICS: tuple = ("grammar_errors_per_100_words", "cola_mean")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dump_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_frozen(path: Optional[Path] = None) -> Optional[dict]:
    path = Path(path) if path else FROZEN_CONFIG
    return load_json(path) if path.exists() else None


def history_before(source: SourceRecord) -> List[str]:
    """The public history a listener had when this utterance was spoken.

    ``sources_from_game_dir`` stores the WHOLE game history on every source,
    including the entry that realizes the utterance itself (predicate and
    text) and everything said later. A judge that is meant to recover the
    predicate blind must never see that, so every judge call goes through
    this cut: entries strictly before the source's own speech id; failing
    that, before the entry that realizes this exact utterance. Otherwise only
    explicitly prior-only contexts or histories with every ID earlier than
    this speech are accepted. An ambiguous history fails closed.
    """
    entries = list(source.public_history or [])
    sid = (source.speech_id or "").strip()
    if sid:
        for index, entry in enumerate(entries):
            if parse_history_entry(entry).get("sid") == sid:
                return entries[:index]
    text = (source.original_nl or "").strip()
    for index, entry in enumerate(entries):
        item = parse_history_entry(entry)
        if text and (item.get("text") or "").strip() == text:
            return entries[:index]
    if not entries or source.provenance == "runtime_context" or "|prior_history" in source.provenance:
        return entries
    if sid and sid.startswith("s") and sid[1:].isdigit():
        numbers = [int(item["sid"][1:]) for item in map(parse_history_entry, entries)
                   if item.get("sid", "").startswith("s") and item["sid"][1:].isdigit()]
        if len(numbers) == len(entries) and numbers and max(numbers) < int(sid[1:]):
            return entries
    raise ValueError(f"Cannot establish prior-only history for {source.source_id}; regenerate sources from game logs")


def with_prior_history(source: SourceRecord) -> SourceRecord:
    """A copy of ``source`` whose ``public_history`` is cut at its own turn."""
    data = source.to_dict()
    data["public_history"] = history_before(source)
    data["provenance"] = source.provenance + "|prior_history"
    return SourceRecord.from_dict(data)


def group_by(records: Iterable[dict], key: str) -> Dict[Any, List[dict]]:
    out: Dict[Any, List[dict]] = {}
    for r in records:
        out.setdefault(r.get(key), []).append(r)
    return out


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"
