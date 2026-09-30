"""Proficiency validation measurements."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from components.helpers import read_jsonl, write_jsonl
from components.metrics.edit_distance import normalized_token_edit_distance
from components.metrics.lexical import SubtlexNorms, lexical_profile
from components.metrics.text_normalization import split_sentences, tokenize, word_count
from experiments.proficiency_validation.validation.sources import assemble_condition_texts
from experiments.proficiency_validation.rewrite.runtime import CONDITIONS, now_iso
from experiments.proficiency_validation.rewrite.runtime import artifact_provenance, record_artifact, verify_artifact
from experiments.proficiency_validation.validation.protocol import cohort, sha256_file


SYNTAX_FIELDS = ("n_sentences", "mean_sentence_length", "sd_sentence_length",
                 "mean_dependency_distance", "tree_depth_mean", "tree_depth_max",
                 "subordinate_clauses_per_sentence", "subordinate_clause_rate",
                 "np_mean_length", "np_mean_modifiers", "clause_pattern_distinct_ratio",
                 "clause_pattern_entropy")


def load_nlp_or_none(model: str, quiet: bool = False):
    try:
        from components.metrics.syntactic import load_nlp
        return load_nlp(model)
    except ImportError as exc:
        if not quiet:
            print(f"WARNING: syntactic features disabled ({exc})", file=sys.stderr)
        return None


def text_features(text: str, *, nlp=None, norms: Optional[SubtlexNorms] = None,
                  mattr_window: int = 50, include_domain: bool = False) -> dict:
    """All Family A measures for one passage."""
    if nlp is not None:
        from components.metrics.syntactic import analyse
        syntax = analyse(text, nlp)
        tokens = syntax.pop("tokens")
        syntax.pop("clause_patterns", None)
        syntax["syntax_available"] = True
    else:
        tokens = tokenize(text)
        syntax = {field: None for field in SYNTAX_FIELDS}
        syntax.update({"n_words": len(tokens), "n_sentences": len(split_sentences(text)),
                       "syntax_available": False})
    lexical = lexical_profile(tokens, norms, mattr_window=mattr_window,
                              include_domain=include_domain)
    return {"word_count": word_count(text), **syntax, **lexical.to_dict()}


def compute_features(assembled: Sequence[dict], *, nlp=None, norms=None,
                     mattr_window: int = 50, include_domain: bool = False) -> List[dict]:
    originals: Dict[str, str] = {r["bundle_id"]: r["text"] for r in assembled
                                 if r["condition"] == "original" and r.get("complete")}
    rows: List[dict] = []
    for r in assembled:
        base = {k: r[k] for k in ("bundle_id", "game_id", "speaker", "condition", "complete",
                                  "n_utterances")}
        if not r.get("complete") or not r.get("text"):
            rows.append({**base, "missing_source_ids": r.get("missing_source_ids", [])})
            continue
        feats = text_features(r["text"], nlp=nlp, norms=norms, mattr_window=mattr_window,
                              include_domain=include_domain)
        original = originals.get(r["bundle_id"])
        feats["edit_distance_to_original"] = (
            normalized_token_edit_distance(r["text"], original) if original else None)
        rows.append({**base, **feats})
    return rows


def add_features_parser(sub) -> None:
    p = sub.add_parser("features", help="assemble bundle passages and compute Family A features")
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--rewrites", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="features JSONL (bundle x condition)")
    p.add_argument("--assembled-out", type=Path, default=None,
                   help="also write the assembled passages (default: <out>_assembled.jsonl)")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--subtlex", type=Path, default=None, help="SUBTLEX-US delimited file")
    p.add_argument("--subtlex-measure", default="Lg10CD")
    p.add_argument("--mattr-window", type=int, default=50)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--no-spacy", action="store_true")
    p.add_argument("--include-domain-terms", action="store_true",
                   help="count Werewolf domain terms in lexical measures (default: separate)")
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify code hashes and pin the MATTR window and spaCy model")
    p.set_defaults(func=features_command)


def features_command(args) -> int:
    frozen = None
    if args.frozen:
        from experiments.proficiency_validation.validation.protocol import enforce_frozen, pin
        frozen = enforce_frozen(args.frozen)
        pin(args, frozen, {"mattr_window": ("mattr_window",),
                           "spacy_model": ("instruments", "spacy_model"),
                           "subtlex_measure": ("instruments", "subtlex_measure"),
                           "include_domain_terms": ("instruments", "include_domain_terms")})
        args.conditions = ",".join(frozen["conditions"])
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    # Instruments first, and every frozen check BEFORE anything is written, so a
    # mismatch cannot leave analysis-ready files on disk without provenance.
    nlp = None if args.no_spacy else load_nlp_or_none(args.spacy_model)
    norms = SubtlexNorms(args.subtlex, args.subtlex_measure) if args.subtlex else None
    if norms is None:
        print("WARNING: no --subtlex file; lexical-frequency fields will be null", file=sys.stderr)
    spacy_prov = None
    if nlp is not None:
        from components.metrics.syntactic import nlp_provenance
        spacy_prov = nlp_provenance(nlp)
    if frozen:
        instruments = frozen.get("instruments") or {}
        if nlp is None:
            print("a frozen run needs spaCy: the syntactic and content-word measures are "
                  "confirmatory metrics and cannot be computed without it", file=sys.stderr)
            return 2
        want = instruments.get("spacy_model_version")
        have = (spacy_prov or {}).get("model_version")
        if have != want or (spacy_prov or {}).get("spacy_version") != instruments["spacy_version"]:
            print(f"spaCy model version {have} differs from the frozen {want}", file=sys.stderr)
            return 2
        want_sha = instruments.get("subtlex_sha256")
        if want_sha and (norms is None or norms.sha256 != want_sha):
            print("SUBTLEX file is missing or differs from the frozen norms "
                  f"(frozen sha256 {want_sha[:12]}…)", file=sys.stderr)
            return 2

    bundles = read_jsonl(args.bundles)
    rewrites = read_jsonl(args.rewrites)
    if frozen:
        if sorted(bundles, key=lambda r: r["bundle_id"]) != sorted(cohort(frozen, args.split), key=lambda r: r["bundle_id"]):
            raise ValueError("bundle selection differs from the frozen split")
        _, errors = verify_artifact(args.rewrites, frozen, args.split)
        if errors:
            raise ValueError("; ".join(errors))
    assembled = assemble_condition_texts(bundles, rewrites, conditions)
    assembled_out = args.assembled_out or args.out.with_name(args.out.stem + "_assembled.jsonl")
    rows = compute_features(assembled, nlp=nlp, norms=norms, mattr_window=args.mattr_window,
                            include_domain=args.include_domain_terms)
    provenance = {"created_at": now_iso(), "mattr_window": args.mattr_window,
                  "include_domain_terms": args.include_domain_terms,
                  "subtlex": norms.provenance() if norms else None,
                  "spacy": spacy_prov, "frozen_config": str(args.frozen) if args.frozen else None}
    provenance.update(artifact_provenance(frozen, args.split, inputs={"bundles": args.bundles, "rewrites": args.rewrites}))
    write_jsonl(assembled_out, assembled)
    write_jsonl(args.out, rows)
    provenance["assembled_sha256"] = sha256_file(assembled_out)
    record_artifact(assembled_out, provenance)
    record_artifact(args.out, provenance)
    complete = sum(1 for r in rows if r.get("complete"))
    print(f"{len(rows)} bundle x condition rows ({complete} complete) -> {args.out}")
    print(f"assembled passages -> {assembled_out}")
    return 0


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


def add_guardrails_parser(sub) -> None:
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
    p.set_defaults(func=guardrails_command)


def guardrails_command(args) -> int:
    frozen = None
    if args.frozen:
        from experiments.proficiency_validation.validation.protocol import enforce_frozen, pin
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
        record_artifact(args.out, provenance)
    return 0
