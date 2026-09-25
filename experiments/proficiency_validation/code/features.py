"""Family A features for every bundle x condition (PLAN 7).

Assembles the matched passages from ``bundles.jsonl`` + ``rewrites.jsonl``,
then computes lexical (SUBTLEX ``Lg10CD``, MATTR, MTLD, repetition) and
syntactic (spaCy dependency) measures on each complete passage, plus the
nuisance diagnostics (word count, sentence count, edit distance to the
original passage). Runs offline; spaCy and the SUBTLEX file are optional and
their absence is recorded in every row rather than silently degrading a number.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from components.helpers import read_jsonl, write_jsonl
from components.metrics.edit_distance import normalized_token_edit_distance
from components.metrics.lexical import SubtlexNorms, lexical_profile
from components.metrics.text_normalization import split_sentences, tokenize, word_count

from .bundles import assemble_condition_texts
from .common import CONDITIONS, dump_json, now_iso

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


def add_parser(sub) -> None:
    p = sub.add_parser("features", help="assemble bundle passages and compute Family A features")
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--rewrites", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="features JSONL (bundle x condition)")
    p.add_argument("--assembled-out", type=Path, default=None,
                   help="also write the assembled passages (default: <out>_assembled.jsonl)")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--subtlex", type=Path, default=None, help="SUBTLEX-US delimited file")
    p.add_argument("--subtlex-measure", default="Lg10CD")
    p.add_argument("--mattr-window", type=int, default=50)
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--no-spacy", action="store_true")
    p.add_argument("--include-domain-terms", action="store_true",
                   help="count Werewolf domain terms in lexical measures (default: separate)")
    p.add_argument("--frozen", type=Path, default=None,
                   help="frozen_config.json: verify code hashes and pin the MATTR window and spaCy model")
    p.set_defaults(func=main)


def main(args) -> int:
    frozen = None
    if args.frozen:
        from .frozen import enforce_frozen, pin
        frozen = enforce_frozen(args.frozen)
        pin(args, frozen, {"mattr_window": ("mattr_window",),
                           "spacy_model": ("instruments", "spacy_model")})
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
        if want and have != want:
            print(f"spaCy model version {have} differs from the frozen {want}", file=sys.stderr)
            return 2
        want_sha = instruments.get("subtlex_sha256")
        if want_sha and (norms is None or norms.sha256 != want_sha):
            print("SUBTLEX file is missing or differs from the frozen norms "
                  f"(frozen sha256 {want_sha[:12]}…)", file=sys.stderr)
            return 2

    bundles = read_jsonl(args.bundles)
    rewrites = read_jsonl(args.rewrites)
    assembled = assemble_condition_texts(bundles, rewrites, conditions)
    assembled_out = args.assembled_out or args.out.with_name(args.out.stem + "_assembled.jsonl")
    rows = compute_features(assembled, nlp=nlp, norms=norms, mattr_window=args.mattr_window,
                            include_domain=args.include_domain_terms)
    provenance = {"created_at": now_iso(), "mattr_window": args.mattr_window,
                  "include_domain_terms": args.include_domain_terms,
                  "subtlex": norms.provenance() if norms else None,
                  "spacy": spacy_prov, "frozen_config": str(args.frozen) if args.frozen else None}
    write_jsonl(assembled_out, assembled)
    write_jsonl(args.out, rows)
    dump_json(args.out.with_name(args.out.stem + "_provenance.json"), provenance)
    complete = sum(1 for r in rows if r.get("complete"))
    print(f"{len(rows)} bundle x condition rows ({complete} complete) -> {args.out}")
    print(f"assembled passages -> {assembled_out}")
    return 0
