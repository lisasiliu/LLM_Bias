"""Grammar and acceptability instruments for proficiency rewrites.

Player and statement IDs are normalized identically in every condition.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

from components.helpers import read_jsonl, write_jsonl
from .protocol import artifact_provenance, artifact_sidecar, verify_artifact, cohort

from .common import now_iso


LT_METRIC_KEYS = ("grammar_errors_per_100_words", "usage_errors_per_100_words",
                  "all_errors_per_100_words", "n_words", "lt_grammar", "lt_usage", "lt_spelling",
                  "lt_punctuation", "lt_style", "lt_other")
COLA_METRIC_KEYS = ("cola_mean", "cola_median", "cola_p10", "cola_min")


def run_grammar(assembled: Sequence[dict], *, languagetool=None, cola=None, nlp=None,
                log=print) -> List[dict]:
    from components.metrics.syntactic import sentences as split
    rows: List[dict] = []
    complete = list(assembled)
    for index, r in enumerate(complete, 1):
        row = {k: r[k] for k in ("bundle_id", "game_id", "speaker", "condition")}
        row["complete"] = bool(r.get("complete") and r.get("text"))
        if not row["complete"]:
            rows.append(row)
            continue
        # Each instrument's full summary is nested under its own key; only the
        # analysis metrics are lifted to the top level, so the two cannot collide.
        if languagetool is not None:
            try:
                lt = languagetool.summarize(r["text"])
                row["languagetool"] = lt
                row.update({k: lt.get(k) for k in LT_METRIC_KEYS})
            except Exception as exc:  # instrument failure is recorded, not hidden
                row["languagetool_error"] = f"{type(exc).__name__}: {exc}"
        if cola is not None:
            try:
                summary = cola.summarize(split(r["text"], nlp))
                row["cola"] = summary
                row.update({k: summary.get(k) for k in COLA_METRIC_KEYS})
            except Exception as exc:
                row["cola_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if index % 20 == 0:
            log(f"  guardrails {index}/{len(complete)}")
    return rows


def add_parser(sub) -> None:
    p = sub.add_parser("guardrails", help="LanguageTool and CoLA guardrails")
    p.add_argument("--assembled", type=Path, help="assembled passages JSONL (from `features`)")
    p.add_argument("--out", type=Path, required=True, help="grammar guardrail JSONL")
    p.add_argument("--languagetool-url", default=None, help="e.g. http://localhost:8081")
    p.add_argument("--languagetool-language", default="en-US")
    p.add_argument("--cola-model", default=None, help="e.g. textattack/roberta-base-CoLA")
    p.add_argument("--cola-revision", default=None)
    p.add_argument("--cola-acceptable-index", type=int, default=1)
    p.add_argument("--cola-max-length", type=int, default=512)
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--run-id", default=None)
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify code hashes and pin the CoLA checkpoint and judge")
    p.set_defaults(func=main)


def main(args) -> int:
    frozen = None
    if args.frozen:
        from .frozen import enforce_frozen, pin
        frozen = enforce_frozen(args.frozen)
        pin(args, frozen, {"cola_model": ("instruments", "cola_model"),
                           "spacy_model": ("instruments", "spacy_model"),
                           "cola_acceptable_index": ("instruments", "cola_acceptable_index"),
                           "cola_max_length": ("instruments", "cola_max_length"),
                           "languagetool_language": ("instruments", "languagetool_language")})
        args.cola_revision = frozen["instruments"].get("cola_revision")
        cohort(frozen, args.split)
        if not args.assembled or not args.languagetool_url:
            raise ValueError("frozen guardrails require assembled passages and both grammar instruments")
    run_id = args.run_id or f"guardrails_{now_iso().replace(':', '')}"
    provenance = {"run_id": run_id, "created_at": now_iso(),
                  "frozen_config": str(args.frozen) if args.frozen else None}
    provenance.update(artifact_provenance(frozen, args.split, inputs={"assembled": args.assembled}))
    wrote_anything = False

    if args.assembled and (args.languagetool_url or args.cola_model):
        from components.metrics.grammar import ColaScorer, LanguageToolClient
        from .features import load_nlp_or_none
        assembled = read_jsonl(args.assembled)
        if frozen:
            _, errors = verify_artifact(args.assembled, frozen, args.split)
            if errors:
                raise ValueError("; ".join(errors))
        lt = LanguageToolClient(args.languagetool_url, args.languagetool_language) if args.languagetool_url else None
        cola = (ColaScorer(args.cola_model, revision=args.cola_revision,
                           acceptable_label_index=args.cola_acceptable_index, max_length=args.cola_max_length)
                if args.cola_model else None)
        if args.frozen:
            # The recorded instrument versions are enforced, not just written down.
            instruments = frozen.get("instruments") or {}
            if lt is not None and instruments.get("languagetool_version"):
                lt.check("This is a warm-up sentence.")          # learns the server version
                if lt.version != instruments["languagetool_version"]:
                    print(f"LanguageTool server is {lt.version!r}, frozen "
                          f"{instruments['languagetool_version']!r}", file=sys.stderr)
                    return 2
            if cola is not None and instruments.get("cola_commit_hash"):
                if cola.commit_hash != instruments["cola_commit_hash"]:
                    print(f"CoLA checkpoint commit {cola.commit_hash!r} differs from the frozen "
                          f"{instruments['cola_commit_hash']!r}", file=sys.stderr)
                    return 2
        nlp = load_nlp_or_none(args.spacy_model, quiet=True)
        from components.metrics.syntactic import nlp_provenance
        provenance["spacy"] = nlp_provenance(nlp) if nlp is not None else None
        if frozen and (nlp is None or provenance["spacy"]["model_version"] != instruments["spacy_model_version"]
                       or provenance["spacy"]["spacy_version"] != instruments["spacy_version"]):
            raise ValueError("frozen grammar sentence segmentation requires the frozen spaCy version")
        rows = run_grammar(assembled, languagetool=lt, cola=cola, nlp=nlp)
        write_jsonl(args.out, rows)
        provenance["languagetool"] = lt.provenance() if lt else None
        provenance["cola"] = cola.provenance() if cola else None
        print(f"{len(rows)} guardrail rows -> {args.out}")
        wrote_anything = True

    if not wrote_anything:
        print("pass --assembled and --languagetool-url or --cola-model", file=sys.stderr)
        return 2
    if args.out.exists():
        artifact_sidecar(args.out, provenance)
    return 0
