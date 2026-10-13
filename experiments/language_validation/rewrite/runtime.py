"""Shared runtime: constants, model transport (ValidationModel), checkpoints, run manifests, provenance and source helpers."""
from __future__ import annotations
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from components.helpers import SourceRecord, parse_history_entry, read_jsonl
import os
import requests
from components.judges.judge_model import JudgeModel, ModelSpec, Completion, JudgeModelError, JudgeTransportError, THINK_OPEN, resolve_spec
from lurkingwolf.utils.speech_nl import strip_reasoning
import fcntl
import tempfile
from contextlib import contextmanager
from components.rewriters.politeness_rewriter import predicate_definition, predicate_entry
from lurkingwolf.utils.speech_nl import parse_predicate


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data" / "corpus"
# Displayed names of the conditions: the three levels are A (restricted), B (source-level paraphrase)
# and C (expanded); the untreated source stays Original.
CONDITION_LABELS = {"low": "A", "neutral": "B", "high": "C", "original": "Original"}
CONDITION_ORDER = ("original", "low", "neutral", "high")


def ordered_conditions(conditions):
    """Conditions in display order: Original, A, B, C."""
    return sorted(conditions, key=lambda c: (CONDITION_ORDER.index(c) if c in CONDITION_ORDER else len(CONDITION_ORDER), c))


def condition_label(condition):
    return CONDITION_LABELS.get(condition, condition)


def pair_label(pair, sep=" vs "):
    """``"low:neutral"`` or ``"low_vs_neutral"`` -> ``"A vs B"``."""
    parts = pair.split(":") if ":" in pair else pair.split("_vs_")
    return sep.join(condition_label(p) for p in parts)
RUNS_DIR = EXPERIMENT_DIR / "data" / "runs"
FROZEN_CONFIG = EXPERIMENT_DIR / "data" / "protocol" / "frozen_config.json"

EXPERIMENT_NAME = "language_validation"

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
    "clause_pattern_distinct_ratio": -1,
    "subordinate_clauses_per_sentence": -1,
    "subordinate_clause_rate": -1,
    "mean_dependency_distance": -1,
    "tree_depth_mean": -1,
    "np_mean_length": -1,
    "mean_sentence_length": -1,
}
PRIMARY_RANGE_METRICS: tuple = ("subtlex_mean", "mattr", "subordinate_clause_rate")
# clause_pattern_entropy is descriptive: splitting long sentences into several short ones raises the
# number of distinct clause patterns per passage, so it carries no range expectation.
NUISANCE_METRICS: tuple = ("word_count", "n_sentences", "edit_distance_to_original", "clause_pattern_entropy")

# Family B guardrail metrics (equivalence, not difference).
GUARDRAIL_METRICS: tuple = ("grammar_errors_per_100_words", "cola_mean")
# Which direction is "worse" for a treatment: +1 higher values are worse (errors), -1 lower values are worse (acceptability).
GUARDRAIL_HARM_DIRECTION: dict = {"grammar_errors_per_100_words": +1, "cola_mean": -1}


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_commit() -> Optional[str]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True,
                                        stderr=subprocess.DEVNULL).strip()
        return head + ("+dirty" if dirty else "")
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
    data["provenance"] = source.provenance if "|prior_history" in source.provenance else source.provenance + "|prior_history"
    return SourceRecord.from_dict(data)


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


class ValidationTransportError(JudgeTransportError):
    """Structured local-server failure suitable for an attempt log."""

    def __init__(self, details):
        self.details = details
        super().__init__(json.dumps(details, ensure_ascii=False, sort_keys=True))


def registry():
    path = os.getenv("PROFICIENCY_MODEL_CONFIG")
    return json.loads(Path(path).read_text(encoding="utf-8")).get("models", {}) if path else {}


def model_identity(handle):
    entry = registry().get(handle)
    if entry is None:
        spec = resolve_spec(handle)
        return {"handle": handle, "model": spec.model, "family": spec.family,
                "revision": None, "tokenizer_revision": None, "serving_version": None,
                "precision": None, "chat_template": None, "extra_body": {}}
    return {"handle": handle, **{key: entry.get(key) for key in (
        "model", "family", "revision", "tokenizer_revision", "serving_version",
        "precision", "chat_template")}, "extra_body": entry.get("extra_body", {})}


def completion_url(base_url):
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1") or base.endswith("/v1beta/openai"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


class ValidationModel(JudgeModel):
    def __init__(self, spec, *, identity=None, extra_body=None, ledger=None, purpose="other", **kwargs):
        super().__init__(spec, **kwargs)
        self.identity_record = identity or model_identity(spec.handle)
        self.extra_body = dict(extra_body or {})
        forbidden = {"model", "messages", "temperature", "max_tokens", "stream"} & self.extra_body.keys()
        if forbidden:
            raise ValueError(f"extra_body must not override {sorted(forbidden)}")
        self.ledger, self.purpose = ledger, purpose
        self.last_error = None

    @classmethod
    def resolve(cls, handle, **kwargs):
        entry = registry().get(handle)
        if entry is None:
            spec = resolve_spec(handle)
        else:
            base = os.getenv(entry.get("base_url_env", "")) or entry.get("base_url")
            if not base or not entry.get("model") or not entry.get("family"):
                raise ValueError(f"{handle}: model, family and a configured server address are required")
            spec = ModelSpec(handle=handle, provider="local", model=entry["model"],
                             base_url=base, family=entry["family"],
                             api_key_env=tuple([entry["api_key_env"]]) if entry.get("api_key_env") else ())
        return cls(spec, identity=model_identity(handle),
                   extra_body=(entry or {}).get("extra_body"), **kwargs)

    def complete(self, *, system, user, max_tokens=None, temperature=None):
        self.last_error = None
        payload = {**self.extra_body, "model": self.spec.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "temperature": self.temperature if temperature is None else temperature,
                   "max_tokens": self.max_tokens if max_tokens is None else max_tokens}
        if self.spec.provider == "openai":
            # Current OpenAI models take max_completion_tokens and only the default temperature.
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            if payload["temperature"] != 1:
                payload.pop("temperature")
        headers = self._headers()
        request_metadata = {"model": self.spec.model, "purpose": self.purpose,
                            "system_chars": len(system), "user_chars": len(user),
                            "prompt_sha256": hashlib.sha256(json.dumps([system, user]).encode()).hexdigest(),
                            "max_output_tokens": payload.get("max_tokens", payload.get("max_completion_tokens"))}
        for attempt in range(self.retries + 1):
            response = None
            try:
                response = requests.post(completion_url(self.spec.base_url), headers=headers,
                                         json=payload, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
            except (requests.RequestException, ValueError) as exc:
                status = getattr(response, "status_code", None)
                if isinstance(exc, ValueError):          # HTTP 2xx with a non-JSON body
                    retryable = True
                elif status is not None:
                    retryable = status in (408, 429, 500, 502, 503, 504)
                else:
                    retryable = isinstance(exc, (requests.Timeout, requests.ConnectionError))
                body = str(getattr(response, "text", ""))[:2000] if response is not None else ""
                self.last_error = {**request_metadata, "http_status": status,
                                   "response_excerpt": body, "error_type": type(exc).__name__,
                                   "retryable": retryable, "attempt": attempt}
                if attempt == self.retries or not retryable:
                    self.transport_retries += attempt
                    raise ValidationTransportError(self.last_error) from exc
                self._sleep(self._backoff * 2 ** attempt)
                continue
            break
        self.calls += 1
        self.transport_retries += attempt
        self.last_error = None
        self.model_served = str(data.get("model") or "")
        if self.model_served != self.spec.model and not self.model_served.startswith(self.spec.model + "-"):
            # OpenAI answers an alias with its dated snapshot name.
            raise JudgeModelError(f"requested {self.spec.model!r}, server returned {self.model_served!r}")
        usage = dict(data.get("usage") or {})
        if self.ledger is not None:
            self.ledger.record(self.spec.provider, self.model_served, self.purpose, usage)
        try:
            choice = data["choices"][0]
            raw = choice["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeModelError("malformed completion") from exc
        if choice.get("finish_reason") == "length":
            raise JudgeModelError("completion exceeded the frozen token budget")
        text = strip_reasoning(raw).strip()
        if not text or THINK_OPEN.search(text):
            raise JudgeModelError("empty or incomplete completion")
        return Completion(text, self.spec.model, self.model_served, usage, attempt)

    def preflight(self):
        # Check the served model without requiring a reasoning model to answer in eight tokens.
        url = completion_url(self.spec.base_url).removesuffix("chat/completions") + "models"
        try:
            response = requests.get(url, headers=self._headers(), timeout=min(self.timeout, 30))
            response.raise_for_status()
            ids = {m["id"] for m in response.json()["data"]}
            if self.spec.model not in ids:
                return False, f"server does not expose {self.spec.model!r}"
            return True, "model is served; revision metadata is operator-declared"
        except (requests.RequestException, ValueError, KeyError, JudgeModelError) as exc:
            return False, str(exc)

    def provenance(self):
        return {**super().provenance(), "identity": self.identity_record,
                "revision_verification": "operator-declared; archive the vLLM launch command and environment"}


def manifest_path(artifact):
    """One manifest for a run's rewrite and validation artifacts."""
    path = Path(artifact).resolve()
    for parent in path.parents:
        if parent.name in {"rewrite", "validation"}:
            return parent.parent / "manifest.json"
        if parent == EXPERIMENT_DIR:
            break
    return path.parent / "manifest.json"


def _manifest(path):
    if not path.exists():
        return {"format": "proficiency_artifacts", "artifacts": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != "proficiency_artifacts":
        raise ValueError(f"unrecognized proficiency manifest: {path}")
    return data


def read_metadata(artifact, section="provenance"):
    path = Path(artifact).resolve()
    manifest = manifest_path(path)
    if manifest.exists():
        key = path.relative_to(manifest.parent).as_posix()
        record = _manifest(manifest).get("artifacts", {}).get(key, {}).get(section)
        if record is not None:
            return record
    # Archived inputs can still be read; new outputs never create sidecar files.
    suffix = {"provenance": path.stem + "_provenance.json",
              "checkpoint": path.name + ".checkpoint.json",
              "summary": path.stem + "_summary.json", "qc": path.stem + "_qc.json"}.get(section)
    legacy = path.with_name(suffix) if suffix else None
    return json.loads(legacy.read_text()) if legacy and legacy.exists() else {}


@contextmanager
def artifact_lock(path, *, wait=True):
    """Lock the shared directory or data inode without a separate lock file."""
    path = Path(path)
    descriptor = os.open(path, os.O_RDONLY if path.is_dir() else os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise ValueError("another job owns this output; use a different output path") from exc
        yield
    finally:
        os.close(descriptor)


def update_metadata(artifact, section, value):
    path = Path(artifact).resolve()
    manifest = manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # The directory inode stays stable across atomic manifest replacement.
    with artifact_lock(manifest.parent):
        data = _manifest(manifest)
        key = path.relative_to(manifest.parent).as_posix()
        if section is None:
            data.setdefault("artifacts", {}).pop(key, None)
        else:
            data.setdefault("artifacts", {}).setdefault(key, {})[section] = value
        _write_manifest(manifest, data)


def _write_manifest(manifest, data):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=manifest.parent,
                                     prefix=".manifest_", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, manifest)
    finally:
        temporary.unlink(missing_ok=True)


def comparable_settings(settings):
    """Compare model-output dependencies while allowing offline report changes."""
    report_modules = {f"experiments.language_validation.validation.{name}"
                      for name in ("analysis", "report", "export", "power")}
    def strip(value):
        if isinstance(value, dict):
            return {k: ({name: sha for name, sha in v.items() if name not in report_modules}
                        if k == "module_source_sha256" else strip(v)) for k, v in value.items()}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(json.loads(json.dumps(settings, sort_keys=True, default=str, allow_nan=False)))


def configure_run(manifest, models, settings):
    """Record model roles and batch settings before workers start; reject changed runs."""
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with artifact_lock(manifest.parent):
        data = _manifest(manifest)
        configuration = json.loads(json.dumps(settings, default=str, allow_nan=False))
        if data.get("artifacts") or data.get("configuration"):
            if (comparable_settings(data.get("configuration")) != comparable_settings(configuration)
                    or data.get("models") != models):
                raise ValueError("run configuration changed or is unavailable; select a new RUN")
        data.update(models=models, configuration=configuration)
        _write_manifest(manifest, data)


class Checkpoint:
    def __init__(self, output, settings, *, resume=False):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.partial = self.output.with_name(self.output.name + ".partial")
        self.manifest = manifest_path(self.output)
        existed = self.output.exists() or self.partial.exists()
        while True:
            lock_target = self.output if self.output.exists() else self.partial
            self.lock = artifact_lock(lock_target, wait=False)
            self.lock.__enter__()
            if lock_target == self.partial and self.output.exists():
                self.close()
                if self.partial.exists() and self.partial.stat().st_size == 0:
                    self.partial.unlink()
                existed = True
                continue
            break
        try:
            # JSON round-trip once, so tuples/Paths compare equal to what the manifest stores.
            settings = json.loads(json.dumps(settings, sort_keys=True, default=str, allow_nan=False))
            self.settings = {"settings_sha256": digest(settings), "settings": settings}
            record = read_metadata(self.output, "checkpoint")
            exists = existed or bool(record)
            if exists and not resume:
                raise ValueError("output/checkpoint already exists; use --resume with unchanged settings")
            if exists and comparable_settings((record or {}).get("settings")) == comparable_settings(settings):
                self.settings = record          # the run keeps the record it was started with
            elif exists and (self.output.exists() or (self.partial.exists() and self.partial.stat().st_size > 0)):
                raise ValueError("checkpoint inputs, code or model settings differ; refusing to resume")
            else:
                # New run, or a resume whose earlier attempt wrote nothing: nothing is lost by starting over.
                update_metadata(self.output, "checkpoint", self.settings)
            self.finished = self.output.exists()
            path = self.output if self.finished else self.partial
            # A malformed partial row raises.
            self.rows = read_jsonl(path) if path.exists() else []
        except Exception:
            self.close()
            raise

    def append(self, row):
        if self.finished:
            raise ValueError("completed output is missing expected rows; refusing to extend it")
        with self.partial.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.rows.append(row)

    def finish(self):
        if not self.finished:
            if not self.partial.exists():
                self.partial.touch()
            os.replace(self.partial, self.output)
            self.finished = True

    def reopen(self, keep):
        """Drop the rows that fail ``keep`` and accept rows again, so those items can be re-asked."""
        kept = [r for r in self.rows if keep(r)]
        if len(kept) == len(self.rows) and not self.finished:
            return 0
        with self.partial.open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        if self.output.exists():
            self.output.unlink()
        self.finished = False
        dropped = len(self.rows) - len(kept)
        self.rows = kept
        return dropped

    def close(self):
        if self.lock is not None:
            self.lock.__exit__(None, None, None)
            self.lock = None


def digest(value):
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def assert_unique(rows, keys):
    seen = set()
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if any(v is None for v in key) or key in seen:
            raise ValueError(f"missing or duplicate {keys}: {key}")
        seen.add(key)


def artifact_provenance(frozen, split, *, inputs=None):
    return {"frozen_sha256": digest(frozen) if frozen else None, "split": split,
            "input_sha256": {k: sha256_file(Path(p)) for k, p in (inputs or {}).items() if p}}


def record_artifact(path, provenance):
    path = Path(path)
    update_metadata(path, "provenance", {**provenance, "output_sha256": sha256_file(path)})


def verify_artifact(path, frozen, split):
    path = Path(path)
    prov = read_metadata(path)
    if not prov:
        return None, [f"missing provenance: {path.name}"]
    errors = []
    for key, expected in (("output_sha256", sha256_file(path)), ("frozen_sha256", digest(frozen)), ("split", split)):
        if prov.get(key) != expected:
            errors.append(f"{path.name}: {key} mismatch")
    return prov, errors


def source_record_from_dict(d: dict) -> SourceRecord:
    """A ``SourceRecord`` from a provenance/JSONL source dict."""
    predicate_text = d.get("predicate_text") or d.get("predicate") or ""
    parsed = parse_predicate(predicate_text)
    name = d.get("predicate_name") or str(parsed.get("name", ""))
    args = [str(a) for a in parsed.get("args", [])]
    speaker = d.get("speaker") or d.get("speaker_id") or (args[1] if len(args) > 1 else "")
    ref = d.get("referenced_speech")
    if ref is None and d.get("referenced_text"):
        ref = {"speech_id": d.get("referenced_speech_id"), "speaker": d.get("referenced_speaker"),
               "predicate": "", "text": d.get("referenced_text")}
    return SourceRecord(
        source_id=d.get("source_id", ""), predicate_name=name, predicate_text=predicate_text,
        predicate_args=args, predicate_definition=predicate_definition(name),
        nl_hint=predicate_entry(name).get("nl_hint", ""),
        original_nl=d.get("original_nl") or d.get("original_text") or "",
        # ``phase`` is the engine phase NAME (day_speech); the D1/N1 token in the
        # predicate is the Phase SLOT and is read separately by the parser.
        phase=d.get("phase") or "day_speech", speaker_id=speaker,
        game_id=d.get("game_id"), speech_id=d.get("speech_id"),
        public_history=list(d.get("public_history") or []), referenced_speech=ref,
        provenance=d.get("provenance") or d.get("source_provenance") or "unknown")
