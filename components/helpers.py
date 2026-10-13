"""Shared record types and readers: untreated utterances paired with their exact predicate, read out of a game directory.

Also the predicate registry (`prompt.json`) every rewriter reads its definitions from.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from lurkingwolf.prompts import load_prompts
from lurkingwolf.rewriter import RewriteContext
from lurkingwolf.utils.speech_nl import parse_predicate


@lru_cache(maxsize=1)
def _predicate_registry() -> dict:
    """Flatten prompt.json's predicate groups into {name: entry}, with the role and action placeholders filled in."""
    from collections.abc import Mapping
    from lurkingwolf.engine import VALID_ROLES
    try:  # ADDRESS_ROLE_TARGETS is only used by address_role
        from lurkingwolf.agent import ADDRESS_ROLE_TARGETS
        target_roles = "|".join(sorted(ADDRESS_ROLE_TARGETS))
    except Exception:  # pragma: no cover - the agent import is heavy
        target_roles = ""
    roles = "|".join(sorted(VALID_ROLES))
    registry: dict = {}
    for group, entries in load_prompts().get("predicates", {}).items():
        if not isinstance(entries, Mapping):
            continue
        for name, entry in entries.items():
            filled = dict(entry)
            for key in ("desc", "nl_hint"):
                if isinstance(filled.get(key), str):
                    filled[key] = filled[key].replace("{valid_roles}", roles).replace("{valid_target_roles}", target_roles)
            filled["group"] = group
            registry[name] = filled
    return registry


def predicate_entry(name: str) -> dict:
    """The prompt.json entry for a predicate name ({} if unknown)."""
    return dict(_predicate_registry().get(name, {}))


def predicate_definition(name: str) -> str:
    """Signature, description and nl_hint for a predicate name."""
    entry = _predicate_registry().get(name)
    if not entry:
        return ""
    parts = [f"{name}{entry.get('sig', '')}: {entry.get('desc', '')}".strip()]
    if entry.get("nl_hint"):
        parts.append(entry["nl_hint"])
    return " ".join(p for p in parts if p).strip()


@dataclass
class SourceRecord:
    """One untreated Stage-2 utterance paired with its exact predicate."""

    source_id: str
    predicate_name: str
    predicate_text: str                       # full predicate string, e.g. estimate(D1, P3, P7, werewolf)
    predicate_args: List[str] = field(default_factory=list)
    predicate_definition: str = ""
    nl_hint: str = ""
    original_nl: str = ""
    phase: str = "day_speech"
    speaker_id: str = ""
    speaker_role_claimed: Optional[str] = None
    game_id: Optional[str] = None
    speech_id: Optional[str] = None
    public_history: List[str] = field(default_factory=list)
    referenced_speech: Optional[Dict[str, str]] = None   # {speech_id, speaker, predicate, text}
    provenance: str = "constructed"                       # "constructed" | "game_run:<dir>"
    addressee: Optional[str] = None                       # the one player the utterance is addressed to, if any

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CheckResult:
    """Outcome of one validator. ``ok=None`` means "could not be evaluated"."""

    ok: Optional[bool]
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.ok is True


def write_jsonl(path: Path, records: Iterable[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_ENTRY_RE = re.compile(
    r"^\[(?P<sid>[^|\]]+)\|(?P<speaker>[^\]]+)\]\s+(?P<pred>\w+\([^)]*\))(?::\s*(?P<text>.*))?$"
)


_TEXT_ONLY_RE = re.compile(r"^\[(?P<sid>s\d+)\|(?P<speaker>[^\]]+)\]\s+(?P<text>.+)$", re.S)


class TreatedSourceError(RuntimeError):
    """The game directory contains rewritten (treated) speech (VALIDATION_SPEC Phase A)."""


def treatment_evidence(game_dir: Path) -> List[str]:
    """Reasons to believe this run's natural language was rewritten by a treatment."""
    game_dir = Path(game_dir)
    reasons: List[str] = []

    manifest_path = game_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        rewriter = manifest.get("rewriter")
        if rewriter:
            reasons.append(f"manifest.rewriter={json.dumps(rewriter, sort_keys=True)}")

    nl_path = game_dir / "game_nl.json"
    if nl_path.exists():
        try:
            nl_log = json.loads(nl_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            nl_log = []
        treated = [turn for turn in nl_log if isinstance(turn, dict) and turn.get("treatment")]
        if treated:
            kinds = sorted({json.dumps(t["treatment"].get("kind", "?")) for t in treated})
            reasons.append(f"game_nl.json has {len(treated)} treated turn(s): {','.join(kinds)}")
    return reasons


def sources_from_game_dir(game_dir: Path,
                          predicates: Optional[set] = None,
                          *,
                          allow_treated: bool = False) -> List[SourceRecord]:
    """Pair every untreated Stage-2 utterance in a real LURK run with its predicate."""
    game_dir = Path(game_dir)
    history_path = game_dir / "public_history.json"
    nl_path = game_dir / "game_nl.json"
    if not history_path.exists():
        raise FileNotFoundError(f"{history_path} not found; is this a LURK game directory?")

    evidence = treatment_evidence(game_dir)
    if evidence and not allow_treated:
        raise TreatedSourceError(
            f"{game_dir} looks like a TREATED run, so its speech may not be used as "
            f"untreated source material (VALIDATION_SPEC Phase A): {'; '.join(evidence)}. "
            "Pass --allow-treated-sources only if you know the text is untreated."
        )

    history = json.loads(history_path.read_text(encoding="utf-8"))
    phases: Dict[str, List[str]] = history.get("phases", {})
    flat: List[str] = [entry for entries in phases.values() for entry in entries
                       if isinstance(entry, str)]

    by_sid: Dict[str, dict] = {}
    by_content: Dict[tuple, str] = {}
    for entry in flat:
        if not isinstance(entry, str):
            continue
        match = _ENTRY_RE.match(entry.strip())
        if not match:
            continue
        item = {k: (v or "") for k, v in match.groupdict().items()}
        by_sid[item["sid"]] = item
        by_content[(item["speaker"], item["pred"], item["text"])] = item["sid"]

    nl_log = json.loads(nl_path.read_text(encoding="utf-8")) if nl_path.exists() else []
    game_id = game_dir.name

    sources: List[SourceRecord] = []
    for turn in nl_log:
        phase = turn.get("phase", "day_speech")
        speaker = turn.get("speaker", "")
        for item in turn.get("speech", []):
            pred, text = item.get("pred", ""), item.get("text", "")
            if not pred or not text:
                continue
            parsed = parse_predicate(pred)
            name = str(parsed.get("name", ""))
            if predicates and name not in predicates:
                continue
            args = list(parsed.get("args", []))
            sid = by_content.get((speaker, pred, text))
            ref = None
            ref_arg = args[-1].strip() if args else ""
            if re.fullmatch(r"s\d+", ref_arg or "") and ref_arg in by_sid:
                found = by_sid[ref_arg]
                ref = {"speech_id": found["sid"], "speaker": found["speaker"],
                       "predicate": found["pred"], "text": found["text"]}
            entry = predicate_entry(name)
            sources.append(SourceRecord(
                source_id=f"{game_id}:{sid or len(sources)}",
                predicate_name=name,
                predicate_text=pred,
                predicate_args=args,
                predicate_definition=predicate_definition(name),
                nl_hint=entry.get("nl_hint", ""),
                original_nl=text,
                phase=phase,
                speaker_id=speaker,
                speaker_role_claimed=None,
                game_id=game_id,
                speech_id=sid,
                public_history=flat,
                referenced_speech=ref,
                provenance=("game_run_TREATED:" if evidence else "game_run:") + str(game_dir),
            ))
    return sources


def parse_history_entry(entry: str) -> Dict[str, str]:
    """Split a ``[sN|Pk] predicate(args): text`` public-history line."""
    if not isinstance(entry, str):
        return {}
    match = _ENTRY_RE.match(entry.strip())
    if not match:
        text_only = _TEXT_ONLY_RE.match(entry.strip())          # what players see: no predicate
        return {"sid": text_only["sid"], "speaker": text_only["speaker"], "pred": "",
                "text": text_only["text"]} if text_only else {}
    data = match.groupdict()
    return {k: (v or "") for k, v in data.items()}


def build_context(source: SourceRecord) -> RewriteContext:
    """The exact ``RewriteContext`` ``LLMAgent.act`` would build for this utterance."""
    history = list(source.public_history)
    return RewriteContext(
        predicate=source.predicate_text,
        original_nl=source.original_nl,
        phase=source.phase,
        speaker_id=source.speaker_id,
        speaker_role_claimed=source.speaker_role_claimed,
        public_history=history,
    )
