"""Shared pieces the judges and the experiments both need.

Four small topics, in order: the record types, reading untreated sources out of
a real game directory, the write-back round trip that proves a rewrite can be
inserted without disturbing its predicate, and the accept/queue policy.

A rewrite is auto-accepted only when the parser reconstructed the predicate AND
the semantic validator passed it; anything else is queued for a person. A parser
mismatch is never silently turned into a pass or a rejection.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from components.rewriters.politeness_rewriter import predicate_definition, predicate_entry
from lurkingwolf.engine import GameState
from lurkingwolf.rewriter import RewriteContext
from lurkingwolf.schemas import build_public_history
from lurkingwolf.utils.speech_nl import parse_predicate


# ======================================================================
# Records
# ======================================================================

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

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SourceRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def balanced_sample(
    sources: Iterable[SourceRecord],
    per_predicate: Optional[int] = None,
    seed: int = 101,
) -> List[SourceRecord]:
    """Take up to ``per_predicate`` sources for each predicate, deterministically."""
    by_pred: Dict[str, List[SourceRecord]] = {}
    for src in sources:
        by_pred.setdefault(src.predicate_name, []).append(src)

    rng = random.Random(seed)
    out: List[SourceRecord] = []
    for name in sorted(by_pred):
        bucket = sorted(by_pred[name], key=lambda s: s.source_id)
        if per_predicate is not None and per_predicate < len(bucket):
            bucket = sorted(rng.sample(bucket, per_predicate), key=lambda s: s.source_id)
        out.extend(bucket)
    return out


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


# ======================================================================
# Reading untreated sources from a real game run
# ======================================================================
_ENTRY_RE = re.compile(
    r"^\[(?P<sid>[^|\]]+)\|(?P<speaker>[^\]]+)\]\s+(?P<pred>\w+\([^)]*\))(?::\s*(?P<text>.*))?$"
)


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


# ======================================================================
# Write-back round trip
# ======================================================================
class InvalidRewriteError(RuntimeError):
    """Raised when a non-validated rewrite is offered for write-back (D007)."""


_ENTRY_RE = re.compile(
    r"^\[(?P<sid>[^|\]]+)\|(?P<speaker>[^\]]+)\]\s+(?P<pred>\w+\([^)]*\))(?::\s*(?P<text>.*))?$"
)


def parse_history_entry(entry: str) -> Dict[str, str]:
    """Split a ``[sN|Pk] predicate(args): text`` public-history line."""
    if not isinstance(entry, str):
        return {}
    match = _ENTRY_RE.match(entry.strip())
    if not match:
        return {}
    data = match.groupdict()
    return {k: (v or "") for k, v in data.items()}


def players_in(source: SourceRecord) -> List[str]:
    """Every player id the source mentions, as a sorted P-list (min 7 for engine sanity)."""
    ids = {source.speaker_id} if source.speaker_id else set()
    ids |= set(re.findall(r"\bP\d+\b", source.predicate_text))
    if source.referenced_speech:
        ids |= set(re.findall(r"\bP\d+\b", json.dumps(source.referenced_speech)))
    numbers = sorted(int(p[1:]) for p in ids if p[1:].isdigit())
    top = max(numbers + [7])
    return [f"P{i}" for i in range(1, top + 1)]


def build_state(source: SourceRecord, game_id: Optional[str] = None) -> GameState:
    """A real ``GameState`` big enough to host this source's utterance."""
    players = players_in(source)
    roles = {p: "villager" for p in players}
    if source.speaker_role_claimed in ("werewolf", "seer", "guard", "witch", "hunter"):
        roles[source.speaker_id] = source.speaker_role_claimed
    state = GameState(game_id or source.game_id or "writeback", players, roles)
    state.phase = source.phase
    for entry in source.public_history:
        state.public_history.append(entry)
    return state


def _max_speech_number(state: GameState) -> int:
    numbers = [int(m.group(1))
               for m in (re.match(r"^\[s(\d+)\|", entry) for entry in state.public_history)
               if m]
    return max(numbers) if numbers else 0


def seed_referenced_speech(state: GameState, source: SourceRecord) -> Optional[str]:
    """Make sure the referenced speech id really exists before the focal utterance."""
    state.speech_id_counter = max(state.speech_id_counter, _max_speech_number(state))
    ref = source.referenced_speech
    if not ref:
        return None
    target_sid = ref.get("speech_id") or ""
    if any(entry.startswith(f"[{target_sid}|") for entry in state.public_history):
        return target_sid
    if target_sid.startswith("s") and target_sid[1:].isdigit():
        state.speech_id_counter = int(target_sid[1:]) - 1
    flat = state.record_speech(
        ref.get("speaker", "P1"),
        [ref.get("predicate", "fluff(D1, P1)")],
        speech=[{"pred": ref.get("predicate", "fluff(D1, P1)"), "text": ref.get("text", "")}],
    )
    state.flush_turn_buffer()
    state.speech_id_counter = max(state.speech_id_counter, _max_speech_number(state))
    parsed = parse_history_entry(flat[0])
    return parsed.get("sid")


def build_context(source: SourceRecord, state: Optional[GameState] = None) -> RewriteContext:
    """The exact ``RewriteContext`` ``LLMAgent.act`` would build for this utterance."""
    history = list(state.public_history) if state is not None else list(source.public_history)
    return RewriteContext(
        predicate=source.predicate_text,
        original_nl=source.original_nl,
        phase=source.phase,
        speaker_id=source.speaker_id,
        speaker_role_claimed=source.speaker_role_claimed,
        public_history=history,
    )


@dataclass
class RoundTripResult:
    ok: bool
    source_id: str
    strategy: str
    speech_id: str = ""
    predicate_before: str = ""
    predicate_after: str = ""
    args_before: List[str] = field(default_factory=list)
    args_after: List[str] = field(default_factory=list)
    text_after: str = ""
    entry: str = ""
    out_dir: Optional[str] = None
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        return data


def write_back_record(
    record: Dict[str, Any],
    source: SourceRecord,
    out_dir: Path,
    *,
    require_validated: bool = True,
    treatment_kind: str = "politeness",
) -> RoundTripResult:
    """Write ONE validated rewrite through the real path, reload it, and compare."""
    from lurkingwolf.run_game import _compact_json

    strategy = record.get("strategy", "")
    rewritten = (record.get("rewritten_nl") or "").strip()

    if require_validated and not record.get("validated"):
        raise InvalidRewriteError(
            f"refusing to write back non-validated record "
            f"{record.get('source_id')}/{strategy}: {record.get('failure_reasons')}"
        )
    if not rewritten:
        raise InvalidRewriteError(
            f"refusing to write back empty rewrite for {record.get('source_id')}/{strategy}"
        )
    if record.get("predicate_text") != source.predicate_text:
        raise InvalidRewriteError(
            "record/source predicate mismatch: "
            f"{record.get('predicate_text')!r} != {source.predicate_text!r}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = build_state(source)
    seed_referenced_speech(state, source)

    predicate_before = source.predicate_text
    last_speech = [{"pred": predicate_before, "text": rewritten, "thinking": ""}]
    flat = state.record_speech(source.speaker_id, [predicate_before], speech=last_speech)
    state.flush_turn_buffer()

    phase_key = source.phase or "day_speech"
    per_phase = {phase_key: list(state.public_history)}
    nl_log = [{
        "phase": phase_key,
        "speaker": source.speaker_id,
        "speech": [{"pred": item["pred"], "text": item["text"]} for item in last_speech],
        "treatment": {"kind": treatment_kind, "strategy": strategy,
                      "source_id": record.get("source_id")},
    }]

    (out_dir / "public_history.json").write_text(
        _compact_json(build_public_history(state, per_phase)), encoding="utf-8")
    (out_dir / "game_nl.json").write_text(_compact_json(nl_log), encoding="utf-8")

    reloaded = json.loads((out_dir / "public_history.json").read_text(encoding="utf-8"))
    reloaded_nl = json.loads((out_dir / "game_nl.json").read_text(encoding="utf-8"))

    entries = reloaded.get("phases", {}).get(phase_key, [])
    target_entry = flat[0]
    reasons: List[str] = []
    if target_entry not in entries:
        reasons.append("entry_missing_after_reload")
        parsed: Dict[str, str] = {}
    else:
        parsed = parse_history_entry(target_entry)
        if not parsed:
            reasons.append("entry_unparseable_after_reload")

    predicate_after = parsed.get("pred", "")
    args_before = list(parse_predicate(predicate_before).get("args", []))
    args_after = list(parse_predicate(predicate_after).get("args", []))
    text_after = parsed.get("text", "")

    if predicate_after != predicate_before:
        reasons.append(f"predicate_changed:{predicate_before!r}->{predicate_after!r}")
    if args_after != args_before:
        reasons.append(f"predicate_args_changed:{args_before}->{args_after}")
    if text_after != rewritten:
        reasons.append("rewritten_nl_changed_after_round_trip")
    if text_after == source.original_nl:
        reasons.append("untreated_source_was_written_back")

    nl_entry = reloaded_nl[0]["speech"][0] if reloaded_nl else {}
    if nl_entry.get("pred") != predicate_before:
        reasons.append("nl_log_predicate_changed")
    if nl_entry.get("text") != rewritten:
        reasons.append("nl_log_text_changed")
    if (reloaded_nl[0].get("treatment") or {}).get("strategy") != strategy:
        reasons.append("treatment_metadata_lost")

    return RoundTripResult(
        ok=not reasons,
        source_id=record.get("source_id", ""),
        strategy=strategy,
        speech_id=parsed.get("sid", ""),
        predicate_before=predicate_before,
        predicate_after=predicate_after,
        args_before=args_before,
        args_after=args_after,
        text_after=text_after,
        entry=target_entry,
        out_dir=str(out_dir),
        reasons=reasons,
    )


# ======================================================================
# Adjudication policy
# ======================================================================
PARSER_EXACT = "exact"
PARSER_MISMATCH = "mismatch"
SEMANTIC_PASS = "pass"
SEMANTIC_FAIL = "fail"
SEMANTIC_NO_VERDICT = "no_verdict"


@dataclass
class Adjudication:
    """Explicit state for one rewrite. Never collapses to a bare boolean."""

    parser_status: str
    semantic_status: str
    adjudication_required: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def auto_accepted(self) -> bool:
        return not self.adjudication_required

    def to_dict(self) -> dict:
        return {"parser_status": self.parser_status,
                "semantic_status": self.semantic_status,
                "adjudication_required": self.adjudication_required,
                "auto_accepted": self.auto_accepted,
                "adjudication_reasons": list(self.reasons)}


def decide(*, parser_structural_match: Optional[bool],
           semantic_ok: Optional[bool],
           deterministic_ok: Optional[bool] = None) -> Adjudication:
    """Combine the checks into an explicit, auditable state."""
    parser_status = PARSER_EXACT if parser_structural_match else PARSER_MISMATCH
    if semantic_ok is True:
        semantic_status = SEMANTIC_PASS
    elif semantic_ok is False:
        semantic_status = SEMANTIC_FAIL
    else:
        semantic_status = SEMANTIC_NO_VERDICT

    reasons: List[str] = []
    if parser_status == PARSER_MISMATCH:
        reasons.append("parser_mismatch")
    if semantic_status == SEMANTIC_FAIL:
        reasons.append("semantic_fail")
    if semantic_status == SEMANTIC_NO_VERDICT:
        reasons.append("semantic_no_verdict")
    if deterministic_ok is False:
        reasons.append("deterministic_fail")
    elif deterministic_ok is None:
        reasons.append("deterministic_not_evaluated")

    return Adjudication(parser_status=parser_status,
                        semantic_status=semantic_status,
                        adjudication_required=bool(reasons),
                        reasons=reasons)

