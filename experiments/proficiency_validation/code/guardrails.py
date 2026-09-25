"""Family B grammar/acceptability guardrails and the politeness-drift diagnostic
(PLAN 8, 11).

Grammar and acceptability run on the assembled passages after player-id
normalization (P3 -> Jordan, s12 -> statement 12), identically in every
condition. Politeness drift runs the existing blind politeness classifier on
each utterance's original and each of its rewrites and records whether the label
was preserved; it is diagnostic (the classifier agreed with humans on 34/48 in
the politeness validation), never a gate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from components.api_usage import UsageLedger
from components.helpers import read_jsonl, write_jsonl
from components.judges.judge_model import JudgeModel

from .common import EXPERIMENT_NAME, dump_json, now_iso


LT_METRIC_KEYS = ("grammar_errors_per_100_words", "usage_errors_per_100_words",
                  "all_errors_per_100_words", "n_words", "lt_grammar", "lt_usage", "lt_spelling",
                  "lt_punctuation", "lt_style", "lt_other")
COLA_METRIC_KEYS = ("cola_mean", "cola_median", "cola_p10", "cola_min")


def run_grammar(assembled: Sequence[dict], *, languagetool=None, cola=None, nlp=None,
                log=print) -> List[dict]:
    from components.metrics.syntactic import sentences as split
    rows: List[dict] = []
    complete = [r for r in assembled if r.get("complete") and r.get("text")]
    for index, r in enumerate(complete, 1):
        row = {k: r[k] for k in ("bundle_id", "game_id", "speaker", "condition")}
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


def run_politeness_drift(rewrites: Sequence[dict], classifier, *, ledger: Optional[UsageLedger] = None,
                         log=print) -> List[dict]:
    """Label each original once and each accepted rewrite; preserved = same label."""
    rows: List[dict] = []
    original_labels: Dict[str, dict] = {}
    for index, r in enumerate(rewrites, 1):
        sid = r["source_id"]
        if sid not in original_labels:
            res = classifier.classify(r["original_text"], phase=r.get("phase", ""),
                                      speaker_id=r.get("speaker", ""))
            original_labels[sid] = res.to_dict()
            if ledger is not None and res.provenance:
                ledger.record(classifier.model.spec.provider,
                              res.provenance.get("model_served") or classifier.model.spec.model,
                              "politeness_classifier", res.provenance.get("usage"))
            rows.append({"source_id": sid, "game_id": r.get("game_id"), "speaker": r.get("speaker"),
                         "predicate_name": r.get("predicate_name"), "condition": "original",
                         "label": res.label, "confidence": res.confidence, "ok": res.ok,
                         "original_label": res.label, "preserved": True if res.ok else None})
        if r.get("status") != "ok" or not r.get("rewritten_text"):
            continue
        res = classifier.classify(r["rewritten_text"], phase=r.get("phase", ""),
                                  speaker_id=r.get("speaker", ""))
        if ledger is not None and res.provenance:
            ledger.record(classifier.model.spec.provider,
                          res.provenance.get("model_served") or classifier.model.spec.model,
                          "politeness_classifier", res.provenance.get("usage"))
        original = original_labels[sid]
        rows.append({"source_id": sid, "game_id": r.get("game_id"), "speaker": r.get("speaker"),
                     "predicate_name": r.get("predicate_name"), "condition": r["condition"],
                     "label": res.label, "confidence": res.confidence, "ok": res.ok,
                     "original_label": original.get("label"),
                     "preserved": (res.label == original.get("label"))
                     if res.ok and original.get("ok") else None,
                     "prompt_version": classifier.prompt_version})
        if index % 20 == 0:
            log(f"  politeness drift {index}/{len(rewrites)}")
    return rows


def add_parser(sub) -> None:
    p = sub.add_parser("guardrails", help="LanguageTool + CoLA guardrails and politeness drift")
    p.add_argument("--assembled", type=Path, help="assembled passages JSONL (from `features`)")
    p.add_argument("--rewrites", type=Path, help="rewrites JSONL (for politeness drift)")
    p.add_argument("--out", type=Path, required=True, help="grammar guardrail JSONL")
    p.add_argument("--politeness-out", type=Path, default=None,
                   help="politeness drift JSONL (default: <out dir>/politeness_drift.jsonl)")
    p.add_argument("--languagetool-url", default=None, help="e.g. http://localhost:8081")
    p.add_argument("--languagetool-language", default="en-US")
    p.add_argument("--cola-model", default=None, help="e.g. textattack/roberta-base-CoLA")
    p.add_argument("--cola-revision", default=None)
    p.add_argument("--cola-acceptable-index", type=int, default=1)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--politeness-judge", default=None, help="JudgeModel handle, e.g. deepseek-chat")
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
                           "cola_revision": ("instruments", "cola_revision"),
                           "spacy_model": ("instruments", "spacy_model")})
        if args.politeness_judge and args.politeness_judge != frozen["models"].get("judge"):
            print(f"--politeness-judge {args.politeness_judge!r} differs from the frozen judge "
                  f"{frozen['models'].get('judge')!r}", file=sys.stderr)
            return 2
    run_id = args.run_id or f"guardrails_{now_iso().replace(':', '')}"
    ledger = UsageLedger(EXPERIMENT_NAME, run_id=run_id)
    provenance = {"run_id": run_id, "created_at": now_iso(),
                  "frozen_config": str(args.frozen) if args.frozen else None}
    wrote_anything = False

    if args.assembled and (args.languagetool_url or args.cola_model):
        from components.metrics.grammar import ColaScorer, LanguageToolClient
        from .features import load_nlp_or_none
        assembled = read_jsonl(args.assembled)
        lt = LanguageToolClient(args.languagetool_url, args.languagetool_language) if args.languagetool_url else None
        cola = (ColaScorer(args.cola_model, revision=args.cola_revision,
                           acceptable_label_index=args.cola_acceptable_index)
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
        rows = run_grammar(assembled, languagetool=lt, cola=cola, nlp=nlp)
        write_jsonl(args.out, rows)
        provenance["languagetool"] = lt.provenance() if lt else None
        provenance["cola"] = cola.provenance() if cola else None
        print(f"{len(rows)} guardrail rows -> {args.out}")
        wrote_anything = True

    if args.rewrites and args.politeness_judge:
        from components.judges.blind_politeness_classifier import BlindPolitenessClassifier
        model = JudgeModel.resolve(args.politeness_judge)
        ok, msg = model.preflight()
        if not ok:
            print(f"politeness judge preflight failed: {msg}", file=sys.stderr)
            return 1
        classifier = BlindPolitenessClassifier(model)
        rows = run_politeness_drift(read_jsonl(args.rewrites), classifier, ledger=ledger)
        out = args.politeness_out or args.out.with_name("politeness_drift.jsonl")
        write_jsonl(out, rows)
        provenance["politeness_classifier"] = {"model": model.provenance(),
                                               "prompt_version": classifier.prompt_version}
        ledger.write()
        print(f"{len(rows)} politeness-drift rows -> {out}")
        wrote_anything = True

    if not wrote_anything:
        print("nothing to do: pass --assembled with --languagetool-url/--cola-model, "
              "and/or --rewrites with --politeness-judge", file=sys.stderr)
        return 2
    dump_json(args.out.with_name(args.out.stem + "_provenance.json"), provenance)
    return 0
