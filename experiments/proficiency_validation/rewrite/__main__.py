"""Proficiency rewrite __main__."""
from __future__ import annotations
from experiments.proficiency_validation.rewrite.runtime import read_metadata, update_metadata
import sys
import hashlib
from pathlib import Path
from typing import List, Optional, Sequence
from components.api_usage import UsageLedger
from components.helpers import SourceRecord, build_context, read_jsonl
from experiments.proficiency_validation.rewrite.runtime import ValidationModel as JudgeModel
from components.metrics.edit_distance import normalized_token_edit_distance
from components.rewriters.language_rewriter import PROMPT_VERSION, CONDITIONS as REWRITE_CONDITIONS, LanguageProficiencyRewriter
from experiments.proficiency_validation.rewrite.runtime import EXPERIMENT_NAME, git_commit, now_iso, sha256_text, with_prior_history
from experiments.proficiency_validation.rewrite.runtime import artifact_provenance, record_artifact, assert_unique
from experiments.proficiency_validation.validation.protocol import cohort, sha256_file
from experiments.proficiency_validation.rewrite.runtime import source_record_from_dict
from experiments.proficiency_validation.rewrite.gate import SemanticGate, generate_with_gate, generation_qc
import json
from experiments.proficiency_validation.rewrite.runtime import digest


class JudgeModelClient:
    """Adapter: the rewriter's OpenAI-style ``chat.completions.create`` over a
    ``JudgeModel`` so that usage and the served model id are recorded."""

    class _Msg:
        def __init__(self, content): self.content = content

    class _Choice:
        def __init__(self, content): self.message = JudgeModelClient._Msg(content)

    class _Resp:
        def __init__(self, content): self.choices = [JudgeModelClient._Choice(content)]

    def __init__(self, model: JudgeModel, ledger: Optional[UsageLedger] = None,
                 purpose: str = "rewrite"):
        self.model = model
        self.ledger = ledger
        self.purpose = purpose
        self.chat = self
        self.completions = self

    def create(self, *, model: str, messages: list, temperature: float = 0.0,
               max_tokens: int = 400, **kwargs):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        completion = self.model.complete(system=system, user=user, max_tokens=max_tokens,
                                         temperature=temperature)
        if self.ledger is not None:
            self.ledger.record(self.model.spec.provider, completion.model_served or self.model.spec.model,
                               self.purpose, completion.usage)
        return JudgeModelClient._Resp(completion.text)


def load_source_records(path: Path) -> List[SourceRecord]:
    """Sources from a JSONL of source dicts, or from provenance.json's ``sources``."""
    path = Path(path)
    if path.suffix == ".json":
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("sources", data) if isinstance(data, dict) else data
    else:
        rows = read_jsonl(path)
    return [with_prior_history(source_record_from_dict(r)) for r in rows]


def generate(sources: Sequence[SourceRecord], *, conditions: Sequence[str],
             generator: JudgeModel, gate: SemanticGate, n_neutral_candidates: int = 5,
             max_corrective_retries: int = 2, max_tokens: int = 400, temperature: float = 0.0,
             ledger: Optional[UsageLedger] = None, log=print,
             completed_rows=None, on_row=None, settings_sha256=None) -> List[dict]:
    if "neutral" in conditions and not any(c in conditions for c in ("low", "high")):
        raise ValueError("Neutral needs its treatment anchor in the same run; include low or high")
    client = JudgeModelClient(generator, ledger, purpose="rewrite")
    rows: List[dict] = []
    commit = git_commit()
    prior = list(completed_rows or [])
    assert_unique(prior, ("source_id", "condition"))
    done = {(r["source_id"], r["condition"]): r for r in prior}
    allowed = {(s.source_id, c) for s in sources for c in conditions}
    if not set(done) <= allowed:
        raise ValueError("checkpoint contains unexpected source/condition rows")
    for index, src in enumerate(sources, 1):
        ctx = build_context(src)
        anchors = {}
        ordered = [c for c in ("low", "high", "neutral") if c in conditions]
        for condition in ordered:
            previous = done.get((src.source_id, condition))
            if previous is not None:
                rows.append(previous)
                if condition in ("low", "high"):
                    anchors[condition] = outcome_from_row(previous)
                continue
            rewriter = LanguageProficiencyRewriter(
                llm_client=client, condition=condition, model=generator.spec.model,
                temperature=temperature, max_tokens=max_tokens,
                n_neutral_candidates=n_neutral_candidates,
                neutral_match_to="low" if "low" in conditions else "high",
                neutral_anchor_conditions=["low" if "low" in conditions else "high"])
            outcome = generate_with_gate(rewriter, gate, ctx, src,
                                         max_corrective_retries=max_corrective_retries,
                                         anchor_outcomes={c: anchors[c] for c in rewriter.neutral_anchor_conditions if c in anchors}
                                         if condition == "neutral" else None)
            text = outcome.text
            distance = normalized_token_edit_distance(text, src.original_nl) if text else None
            if condition in ("low", "high"):
                anchors[condition] = outcome.to_dict()
            row = {
                "item_id": f"{src.source_id}::{condition}", "source_id": src.source_id,
                "game_id": src.game_id, "speech_id": src.speech_id, "predicate": src.predicate_text,
                "predicate_name": src.predicate_name, "phase": src.phase, "speaker": src.speaker_id,
                "referenced_speech": src.referenced_speech, "source_provenance": src.provenance,
                "original_text": src.original_nl, "condition": condition,
                "rewritten_text": text, "edit_distance_to_original": distance,
                **{k: v for k, v in outcome.to_dict().items() if k not in ("condition", "text")},
                "generator_handle": generator.spec.handle,
                "generator_model": generator.spec.model, "generator_model_served": generator.model_served,
                "generator_provider": generator.spec.provider, "generator_temperature": temperature,
                "generator_provenance": generator.provenance(),
                "prompt_version": PROMPT_VERSION,
                # template hash: identical for every predicate, the value a frozen config pins
                "prompt_template_sha256": rewriter.template_sha256(candidates=(condition == "neutral")),
                # instance hash: the exact system prompt sent, definition included; None when
                # the context itself was unusable (that failure is already in the attempts)
                "prompt_sha256": _safe_instance_hash(rewriter, ctx, condition),
                "gate_provenance": gate.provenance(), "git_commit": commit, "created_at": now_iso(),
                "generation_settings_sha256": settings_sha256,
            }
            rows.append(row)
            if on_row:
                on_row(row)
            log(f"[{index}/{len(sources)}] {src.source_id} {condition:8s} {outcome.status:22s} "
                f"attempts={len(outcome.attempts)} dist={distance if distance is None else round(distance, 3)}")
    return rows


OUTCOME_KEYS = ("condition", "status", "failure_kind", "first_pass_ok", "n_attempts", "n_retries",
                "fallback_to_original", "target_edit_distance", "anchor_outcomes", "attempts")


def outcome_from_row(row: dict) -> dict:
    """``GenerationOutcome.to_dict()`` recovered from a checkpointed rewrite row, so a
    resumed run records the same anchor structure as an uninterrupted one."""
    out = {k: row.get(k) for k in OUTCOME_KEYS}
    out["text"] = row.get("rewritten_text")
    return out


def _safe_instance_hash(rewriter: LanguageProficiencyRewriter, ctx, condition: str) -> Optional[str]:
    try:
        return rewriter.prompt_sha256(ctx, candidates=(condition == "neutral"))
    except (ValueError, RuntimeError):
        return None


def dry_run(sources: Sequence[SourceRecord], conditions: Sequence[str], n_neutral_candidates: int) -> str:
    """Print the exact prompts and their hashes without calling any model."""
    out: List[str] = []
    for src in sources[:1]:
        ctx = build_context(src)
        for condition in conditions:
            rewriter = LanguageProficiencyRewriter(llm_client=None, condition=condition,
                                                   n_neutral_candidates=n_neutral_candidates)
            system, user = rewriter.build_prompts(ctx)
            out += [f"===== {condition.upper()} system prompt (sha256 {sha256_text(system)}) =====",
                    system, "", f"----- {condition.upper()} user prompt -----", user, ""]
            if condition == "neutral":
                system_c, user_c = rewriter.build_candidate_prompts(ctx)
                out += [f"===== NEUTRAL candidate prompt (sha256 {sha256_text(system_c)}) =====",
                        system_c, "", "----- candidate user prompt -----", user_c, ""]
    return "\n".join(out)


def add_generate_parser(sub) -> None:
    p = sub.add_parser("generate", help="generate Low/Neutral/(High) rewrites through the semantic gate")
    p.add_argument("--sources", type=Path, required=True,
                   help="sources JSONL, or a provenance.json with a sources list")
    p.add_argument("--conditions", default="low,neutral",
                   help="comma-separated subset of low,neutral,high")
    p.add_argument("--generator", default="generator", help="handle in PROFICIENCY_MODEL_CONFIG")
    p.add_argument("--judge", default="judge", help="handle in PROFICIENCY_MODEL_CONFIG")
    p.add_argument("--judge-max-tokens", type=int, default=3072)
    p.add_argument("--neutral-edit-tolerance", type=float, default=.10,
                   help="maximum absolute token-edit-distance gap per accepted Neutral; pilot default, justify before freezing")
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--n-neutral-candidates", type=int, default=5)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1,
                   help="partition sources by index without writing duplicate source files")
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true", help="resume completed rows, including recorded failures")
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify code hashes and pin models, candidates, retries")
    p.add_argument("--out", type=Path, required=True, help="rewrites JSONL")
    p.add_argument("--dry-run", action="store_true", help="print prompts and hashes; no model calls")
    p.set_defaults(func=generate_command)


def generate_command(args) -> int:
    frozen = None
    if args.frozen:
        from experiments.proficiency_validation.validation.protocol import enforce_frozen, pin
        frozen = enforce_frozen(args.frozen, check_models=True)
        pin(args, frozen, {"n_neutral_candidates": ("n_neutral_candidates",),
                           "max_retries": ("max_corrective_retries",),
                           "generator": ("models", "generator"), "judge": ("models", "judge"),
                           "temperature": ("generation_temperature",),
                           "max_tokens": ("generation_max_tokens",), "judge_max_tokens": ("judge_max_tokens",),
                           "neutral_edit_tolerance": ("neutral_edit_distance_tolerance",)})
        args.conditions = ",".join(c for c in frozen["conditions"] if c != "original")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if frozen:
        allowed = set(frozen.get("conditions") or [])
        extra = [c for c in conditions if c not in allowed]
        if extra:
            print(f"conditions {extra} are not in the frozen design {sorted(allowed)}; "
                  "use the declared treatment for this study", file=sys.stderr)
            return 2
    bad = [c for c in conditions if c not in REWRITE_CONDITIONS]
    if bad:
        print(f"unknown conditions {bad}; choose from {REWRITE_CONDITIONS}", file=sys.stderr)
        return 2
    if "neutral" in conditions and not any(c in conditions for c in ("low", "high")):
        print("neutral needs its anchor in the same run: include low or high in --conditions", file=sys.stderr)
        return 2
    sources = load_source_records(args.sources)
    if not 0 <= args.neutral_edit_tolerance <= 1:
        raise ValueError("neutral edit tolerance must be between zero and one")
    assert_unique([s.to_dict() for s in sources], ("source_id",))
    if frozen:
        if sha256_file(args.sources) != frozen["sources_sha256"].get(args.split):
            raise ValueError("source text artifact differs from the frozen file")
        expected = {sid for b in cohort(frozen, args.split) for sid in b["source_ids"]}
        game_of = {sid: b["game_id"] for b in cohort(frozen, args.split) for sid in b["source_ids"]}
        available = {s.source_id for s in sources}
        if not expected <= available or args.limit:
            raise ValueError("frozen generation needs every selected source and does not permit --limit")
        sources = [s for s in sources if s.source_id in expected]
        if any(s.game_id != game_of[s.source_id] for s in sources):
            raise ValueError("source game IDs differ from the frozen cohort")
    if args.limit:
        sources = sources[:args.limit]
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid generation shard index/count")
    if frozen and args.shard_count != 1:
        raise ValueError("frozen generation uses the complete selected cohort")
    sources = sources[args.shard_index::args.shard_count]
    if not sources:
        raise ValueError("no sources in this generation shard")
    if args.dry_run:
        print(dry_run(sources, conditions, args.n_neutral_candidates))
        return 0
    generator = JudgeModel.resolve(args.generator, temperature=args.temperature,
                                   max_tokens=args.max_tokens * max(2, args.n_neutral_candidates))
    ok, msg = generator.preflight()
    if not ok:
        print(f"generator preflight failed: {msg}", file=sys.stderr)
        return 1
    run_id = args.run_id or f"generate_{now_iso().replace(':', '')}"
    ledger = UsageLedger(EXPERIMENT_NAME, run_id=run_id)
    gate = SemanticGate.from_handle(args.judge, ledger=ledger, max_tokens=args.judge_max_tokens)
    gate.neutral_edit_tolerance = args.neutral_edit_tolerance
    ok, msg = gate.judge_model.preflight()
    if not ok:
        raise RuntimeError(f"judge preflight failed: {msg}")
    from experiments.proficiency_validation.rewrite.runtime import Checkpoint
    from experiments.proficiency_validation.validation.protocol import frozen_hashes
    hashes = frozen_hashes(args.n_neutral_candidates)
    policy_modules = ("gate", "__main__", "fidelity", "runtime")
    settings = {"artifact": artifact_provenance(frozen, args.split, inputs={"sources": args.sources}),
                "hashes": {k: v for k, v in hashes.items() if k != "module_source_sha256"},
                "generation_policy_sha256": {k: v for k, v in hashes["module_source_sha256"].items()
                    if k.rsplit(".", 1)[-1] in policy_modules or k.startswith("components.judges.")
                    or k in ("components.helpers", "components.rewriters.language_rewriter")},
                "conditions": conditions,
                "generator": generator.identity_record, "judge": gate.judge_model.identity_record,
                "temperature": args.temperature, "max_tokens": args.max_tokens, "judge_max_tokens": args.judge_max_tokens,
                "n_neutral_candidates": args.n_neutral_candidates, "retries": args.max_retries,
                "neutral_edit_tolerance": args.neutral_edit_tolerance,
                "source_ids": [s.source_id for s in sources]}
    checkpoint = Checkpoint(args.out, settings, resume=args.resume)
    try:
        rows = generate(sources, conditions=conditions, generator=generator, gate=gate,
                        n_neutral_candidates=args.n_neutral_candidates,
                        max_corrective_retries=args.max_retries, max_tokens=args.max_tokens,
                        temperature=args.temperature, ledger=ledger,
                        completed_rows=checkpoint.rows, on_row=checkpoint.append,
                        settings_sha256=checkpoint.settings["settings_sha256"])
        checkpoint.finish()
    finally:
        checkpoint.close()
    record_artifact(args.out, {**artifact_provenance(frozen, args.split, inputs={"sources": args.sources}),
                              **checkpoint.settings})
    qc = generation_qc(rows)
    # Parser and validator usage was booked per call by the gate's LedgerJudgeModels,
    # so the totals below are complete.
    ledger.write()
    update_metadata(args.out, "qc",
              {"run_id": run_id, "n_sources": len(sources), "conditions": conditions,
               "generation_temperature": args.temperature,
               "generator": generator.provenance(), "gate": gate.provenance(),
               "frozen_config": str(args.frozen) if args.frozen else None,
               "generation_qc": qc, "usage": ledger.totals(), "created_at": now_iso()})
    print(f"{len(rows)} rows -> {args.out}")
    for condition, q in qc.items():
        print(f"  {condition:8s} first-pass {q['first_pass_success']}/{q['n']}  "
              f"retry {q['needed_retry']}  unavailable {q['treatment_unavailable']}")
    return 0


def check_generation_file(path):
    path = Path(path)
    side = read_metadata(path)
    errors = [] if side.get("output_sha256") == sha256_file(path) else [f"{path.name}: missing or mismatched output hash"]
    return errors + check_generation_rows(read_jsonl(path), read_metadata(path, "checkpoint") or side, path.name)


def check_generation_rows(rows, record, name):
    path = Path(name)
    errors = []
    settings = record.get("settings")
    if not isinstance(settings, dict):
        errors.append(f"{path.name}: generation settings unavailable")
        return errors
    if record.get("settings_sha256") != digest(settings):
        errors.append(f"{path.name}: settings digest mismatch")
    try:
        assert_unique(rows, ("source_id", "condition"))
    except ValueError:
        errors.append(f"{path.name}: duplicate source/condition outcomes")
    expected = {(s, c) for s in settings.get("source_ids", []) for c in settings.get("conditions", [])}
    if not expected or {(r["source_id"], r["condition"]) for r in rows} != expected:
        errors.append(f"{path.name}: incomplete or unexpected source/condition outcomes")
    hashes = settings.get("hashes", {})
    bad = []
    for row in rows:
        gate = row.get("gate_provenance") or {}
        condition = row["condition"]
        template = hashes.get("rewriter_prompt_sha256", {}).get("neutral_candidates" if condition == "neutral" else condition)
        if (row.get("prompt_version") != hashes.get("rewriter_prompt_version")
                or row.get("prompt_template_sha256") != template
                or gate.get("semantic_rubric_version") != hashes.get("semantic_rubric_version")
                or gate.get("semantic_rubric_sha256") != hashes.get("semantic_rubric_sha256")):
            bad.append(row["source_id"])
        if row.get("generation_settings_sha256") is not None and row["generation_settings_sha256"] != record.get("settings_sha256"):
            bad.append(row["source_id"])
        for key in ("validator_max_tokens", "parser_max_tokens"):
            if gate.get(key) is not None and gate[key] != settings.get("judge_max_tokens"):
                bad.append(row["source_id"])
        # Legacy runs lack per-row setting digests. New runs bind identities and
        # matching policy as well as versions to the immutable checkpoint.
        if row.get("generation_settings_sha256"):
            if ((row.get("generator_provenance") or {}).get("identity") != settings.get("generator")
                    or row.get("generator_temperature") != settings.get("temperature")
                    or gate.get("neutral_edit_tolerance") != settings.get("neutral_edit_tolerance")):
                bad.append(row["source_id"])
    if bad:
        errors.append(f"{path.name}: {len(set(bad))} sources disagree with recorded prompt/rubric/settings")
    return errors


def audit_generation(path):
    path = Path(path)
    shards = sorted((path.parent / "shards").glob("rewrites_[0-9][0-9].jsonl"))
    side = read_metadata(path)
    if side.get("compacted_shards"):
        raw = path.read_bytes()
        errors, policies, end = [], [], 0
        if side.get("output_sha256") != sha256_file(path):
            errors.append("merged rewrite output hash mismatch")
        for shard in side["compacted_shards"]:
            start, stop = shard["byte_range"]
            if start != end or stop <= start or stop > len(raw):
                errors.append("invalid compacted shard byte ranges")
                break
            chunk = raw[start:stop]
            if hashlib.sha256(chunk).hexdigest() != shard["output_sha256"]:
                errors.append(f"compacted shard hash mismatch: {shard['name']}")
            rows = [json.loads(line) for line in chunk.splitlines() if line.strip()]
            errors.extend(check_generation_rows(rows, shard["checkpoint"], shard["name"]))
            settings = shard["checkpoint"].get("settings", {})
            policies.append({k: v for k, v in settings.items() if k not in ("artifact", "source_ids")})
            end = stop
        if end != len(raw):
            errors.append("compacted shards do not cover the merged output")
        if policies and any(policy != policies[0] for policy in policies):
            errors.append("compacted shards use different generation policies")
        if policies and side.get("generation_policy_sha256") != digest(policies[0]):
            errors.append("compacted generation policy digest mismatch")
        try:
            assert_unique(read_jsonl(path), ("source_id", "condition"))
        except ValueError:
            errors.append("merged rewrite rows contain duplicate source/condition outcomes")
        return errors
    if side.get("shard_paths"):
        shards = [path.parent / p for p in side["shard_paths"]]
    if not shards:
        return check_generation_file(path)
    errors = [e for shard in shards for e in check_generation_file(shard)]
    rows = [r for shard in shards for r in read_jsonl(shard)]
    merged = read_jsonl(path)
    try:
        assert_unique(rows, ("source_id", "condition"))
    except ValueError:
        errors.append("generation shards contain duplicate source/condition outcomes")
    key = lambda r: (r["source_id"], r["condition"])
    if sorted(rows, key=key) != sorted(merged, key=key):
        errors.append("merged rewrite rows differ from the recorded shards")
    if side.get("output_sha256") != sha256_file(path):
        errors.append("merged rewrite output hash mismatch")
    return errors


def add_record_parsers(sub):
    p = sub.add_parser("verify-run", help="check archived generation settings without changing records")
    p.add_argument("--rewrites", type=Path, required=True)
    p.set_defaults(func=lambda args: verify(args.rewrites))
    p = sub.add_parser("merge-rewrites", help="merge verified generation shards with complete source coverage")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--shards", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=merge)


def verify(path):
    errors = audit_generation(path)
    print("\n".join(errors) if errors else "generation records match their archived settings")
    return int(bool(errors))


def merge(args):
    if args.out.exists():
        raise ValueError("merged output already exists; use a new run directory")
    rows, policies, conditions = [], [], None
    for path in args.shards:
        errors = check_generation_file(path)
        if errors:
            raise ValueError("; ".join(errors))
        record = read_metadata(path, "checkpoint")
        settings = record["settings"]
        if settings.get("artifact", {}).get("frozen_sha256"):
            raise ValueError("merge-rewrites supports pilot shards; frozen generation uses its complete cohort checkpoint")
        policies.append({k: v for k, v in settings.items() if k not in ("artifact", "source_ids")})
        conditions = settings["conditions"]
        rows.extend(read_jsonl(path))
    if not policies or any(p != policies[0] for p in policies):
        raise ValueError("generation shards use different policies")
    assert_unique(rows, ("source_id", "condition"))
    source_rows = read_jsonl(args.sources)
    assert_unique(source_rows, ("source_id",))
    sources = {s["source_id"]: s for s in source_rows}
    if {(r["source_id"], r["condition"]) for r in rows} != {(s, c) for s in sources for c in conditions}:
        raise ValueError("merged generation lacks complete source/condition outcomes")
    for r in rows:
        if r["original_text"] != sources[r["source_id"]]["original_nl"]:
            raise ValueError("shard text differs from source artifact")
    # Preserve exact shard bytes so their recorded hashes remain independently verifiable.
    chunks = [p.read_bytes() for p in args.shards]
    compacted, offset = [], 0
    for path, chunk in zip(args.shards, chunks):
        compacted.append({"name": path.name, "byte_range": [offset, offset + len(chunk)],
            "output_sha256": hashlib.sha256(chunk).hexdigest(),
            "checkpoint": read_metadata(path, "checkpoint"),
            "provenance": read_metadata(path), "qc": read_metadata(path, "qc")})
        offset += len(chunk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(b"".join(chunks))
    record_artifact(args.out, {"frozen_sha256": None, "split": "pilot",
        "input_sha256": {"sources": sha256_file(args.sources), **{f"shard_{i:02}": sha256_file(p) for i, p in enumerate(args.shards)}},
        "compacted_shards": compacted,
        "generation_policy_sha256": digest(policies[0])})
    errors = audit_generation(args.out)
    if errors:
        raise ValueError("; ".join(errors))
    for path, chunk in zip(args.shards, chunks):
        if path.read_bytes() != chunk:
            raise ValueError("shard changed during compaction")
        path.unlink()
        update_metadata(path, None, None)
        # Legacy sidecars are redundant only after successful merged verification.
        for suffix in (path.name + ".checkpoint.json", path.stem + "_provenance.json", path.stem + "_qc.json"):
            path.with_name(suffix).unlink(missing_ok=True)
    for directory in {p.parent for p in args.shards}:
        if directory != args.out.parent and not any(directory.iterdir()):
            directory.rmdir()
    print(f"{len(rows)} verified rows -> {args.out}")
    return 0


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Generate and verify proficiency rewrites.")
    sub = parser.add_subparsers(dest="command", required=True)
    add_generate_parser(sub)
    add_record_parsers(sub)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
