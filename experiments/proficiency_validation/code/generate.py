"""Generate gated Low, Neutral and optional High rewrites from source utterances.

Generate and gate Low and High first, then match Neutral to their edit-distance
midpoint. Every attempt, candidate, distance and gate verdict is kept.

Outputs one JSONL row per source x condition plus ``generation_qc.json``. The
generator and the judges are recorded with the model actually served, and API
usage is appended to the experiment ledger.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

from components.api_usage import UsageLedger
from components.helpers import SourceRecord, build_context, read_jsonl
from .models import ValidationModel as JudgeModel
from components.metrics.edit_distance import normalized_token_edit_distance
from components.rewriters.language_rewriter import (
    PROMPT_VERSION, CONDITIONS as REWRITE_CONDITIONS, LanguageProficiencyRewriter)

from .common import EXPERIMENT_NAME, dump_json, git_commit, now_iso, sha256_text, with_prior_history
from .protocol import artifact_provenance, artifact_sidecar, assert_unique, cohort, sha256_file
from .judge_benchmark import source_record_from_dict
from .semantic_gate import SemanticGate, generate_with_gate, generation_qc


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
        rows = data.get("manual_validation", {}).get("sources") or data.get("sources") or data
    else:
        rows = read_jsonl(path)
    return [with_prior_history(source_record_from_dict(r)) for r in rows]


def generate(sources: Sequence[SourceRecord], *, conditions: Sequence[str],
             generator: JudgeModel, gate: SemanticGate, n_neutral_candidates: int = 5,
             max_corrective_retries: int = 2, max_tokens: int = 400, temperature: float = 0.0,
             ledger: Optional[UsageLedger] = None, log=print,
             completed_rows=None, on_row=None) -> List[dict]:
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
                neutral_match_to="low" if condition == "neutral" else None,
                neutral_anchor_conditions=[c for c in ("low", "high") if c in conditions])
            outcome = generate_with_gate(rewriter, gate, ctx, src,
                                         max_corrective_retries=max_corrective_retries,
                                         anchor_outcomes=anchors if condition == "neutral" else None)
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
            }
            rows.append(row)
            if on_row:
                on_row(row)
            log(f"[{index}/{len(sources)}] {src.source_id} {condition:8s} {outcome.status:22s} "
                f"attempts={len(outcome.attempts)} dist={distance if distance is None else round(distance, 3)}")
    return rows


OUTCOME_KEYS = ("condition", "status", "first_pass_ok", "n_attempts", "n_retries",
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


def add_parser(sub) -> None:
    p = sub.add_parser("generate", help="generate Low/Neutral/(High) rewrites through the semantic gate")
    p.add_argument("--sources", type=Path, required=True,
                   help="sources JSONL, or a provenance.json with a sources list")
    p.add_argument("--conditions", default="low,neutral,high",
                   help="comma-separated subset of low,neutral,high")
    p.add_argument("--generator", default="generator", help="handle in PROFICIENCY_MODEL_CONFIG")
    p.add_argument("--judge", default="judge", help="handle in PROFICIENCY_MODEL_CONFIG")
    p.add_argument("--judge-max-tokens", type=int, default=800)
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--n-neutral-candidates", type=int, default=5)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--resume", action="store_true", help="resume completed rows, including recorded failures")
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify code hashes and pin models, candidates, retries")
    p.add_argument("--out", type=Path, required=True, help="rewrites JSONL")
    p.add_argument("--dry-run", action="store_true", help="print prompts and hashes; no model calls")
    p.set_defaults(func=main)


def main(args) -> int:
    frozen = None
    if args.frozen:
        from .frozen import enforce_frozen, pin
        frozen = enforce_frozen(args.frozen, check_models=True)
        pin(args, frozen, {"n_neutral_candidates": ("n_neutral_candidates",),
                           "max_retries": ("max_corrective_retries",),
                           "generator": ("models", "generator"), "judge": ("models", "judge"),
                           "temperature": ("generation_temperature",),
                           "max_tokens": ("generation_max_tokens",), "judge_max_tokens": ("judge_max_tokens",)})
        args.conditions = ",".join(c for c in frozen["conditions"] if c != "original")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    if frozen:
        allowed = set(frozen.get("conditions") or [])
        extra = [c for c in conditions if c not in allowed]
        if extra:
            print(f"conditions {extra} are not in the frozen design {sorted(allowed)}; "
                  "re-freeze with --include-high if High passed the pilot gate", file=sys.stderr)
            return 2
    bad = [c for c in conditions if c not in REWRITE_CONDITIONS]
    if bad:
        print(f"unknown conditions {bad}; choose from {REWRITE_CONDITIONS}", file=sys.stderr)
        return 2
    sources = load_source_records(args.sources)
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
    ok, msg = gate.judge_model.preflight()
    if not ok:
        raise RuntimeError(f"judge preflight failed: {msg}")
    from .checkpoint import Checkpoint
    from .frozen import frozen_hashes
    settings = {"artifact": artifact_provenance(frozen, args.split, inputs={"sources": args.sources}),
                # Prompt/rubric hashes only: an unrelated code edit must not orphan a paid run.
                # (--frozen has already refused drifted code before any resume.)
                "hashes": {k: v for k, v in frozen_hashes(args.n_neutral_candidates).items()
                           if k != "module_source_sha256"},
                "conditions": conditions,
                "generator": generator.identity_record, "judge": gate.judge_model.identity_record,
                "temperature": args.temperature, "max_tokens": args.max_tokens, "judge_max_tokens": args.judge_max_tokens,
                "n_neutral_candidates": args.n_neutral_candidates, "retries": args.max_retries,
                "source_ids": [s.source_id for s in sources]}
    checkpoint = Checkpoint(args.out, settings, resume=args.resume)
    try:
        rows = generate(sources, conditions=conditions, generator=generator, gate=gate,
                        n_neutral_candidates=args.n_neutral_candidates,
                        max_corrective_retries=args.max_retries, max_tokens=args.max_tokens,
                        temperature=args.temperature, ledger=ledger,
                        completed_rows=checkpoint.rows, on_row=checkpoint.append)
        checkpoint.finish()
    finally:
        checkpoint.close()
    artifact_sidecar(args.out, artifact_provenance(frozen, args.split, inputs={"sources": args.sources}))
    qc = generation_qc(rows)
    # Parser and validator usage was booked per call by the gate's LedgerJudgeModels,
    # so the totals below are complete.
    ledger.write()
    dump_json(args.out.with_name(args.out.stem + "_qc.json"),
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
