"""Untreated proficiency source preparation and reproducible descriptive report."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from components.helpers import read_jsonl, write_jsonl, SourceRecord, treatment_evidence
from components.metrics.text_normalization import word_count
from components.stats import mean, median
from .audit import load_sources, audit
from .bundles import build_bundles
from .common import DATA_DIR, EXPERIMENT_DIR, dump_json, sha256_file, now_iso, fmt
from .features import compute_features, load_nlp_or_none
from .protocol import assert_unique

ARTIFACTS = ("sources.jsonl", "bundles.jsonl", "original_features.jsonl")
BASELINE_METRICS = (
    ("word_count", "Words per bundle"), ("mattr", "MATTR"), ("mtld", "MTLD"),
    ("repetition_rate", "Content-lemma repetition"),
    ("subordinate_clauses_per_sentence", "Subordinate clauses per sentence"),
    ("mean_dependency_distance", "Mean dependency distance"), ("subtlex_mean", "SUBTLEX mean"),
)


def assemble_originals(sources, bundles):
    by_id = {s.source_id: s for s in sources}
    rows = []
    for b in bundles:
        selected = [by_id[sid] for sid in b["source_ids"]]
        if any(s.game_id != b["game_id"] or s.speaker_id != b["speaker"] for s in selected):
            raise ValueError("bundle sources do not belong to its game and speaker")
        text = " ".join(s.original_nl for s in selected)
        rows.append({"bundle_id": b["bundle_id"], "game_id": b["game_id"], "speaker": b["speaker"],
                     "condition": "original", "complete": True, "n_utterances": len(selected),
                     "text": text, "word_count": word_count(text)})
    return rows


def check_sources(sources):
    if not sources:
        raise ValueError("no untreated source utterances were found")
    assert_unique([s.to_dict() for s in sources], ("source_id",))
    bad = [s.source_id for s in sources if not s.speech_id or not s.original_nl
           or len(s.predicate_args) < 2 or s.speaker_id != s.predicate_args[1]
           or "|prior_history" not in s.provenance or not s.provenance.startswith("game_run:")]
    if bad:
        raise ValueError(f"source identity/history checks failed: {bad}")


def prepare(args):
    paths = [Path(p) for p in args.game_dirs]
    if not paths:
        raise ValueError("supply at least one untreated game directory")
    grouped = load_sources(paths)
    sources = [r for rows in grouped.values() for r in rows]
    check_sources(sources)
    result = build_bundles(sources, window_min=args.window[0], window_max=args.window[1], max_per_game=None)
    nlp = load_nlp_or_none(args.spacy_model)
    from components.metrics.syntactic import nlp_provenance
    from components.metrics.lexical import SubtlexNorms
    norms = SubtlexNorms(args.subtlex, "Lg10CD") if args.subtlex else None
    features = compute_features(assemble_originals(sources, result["bundles"]), nlp=nlp, norms=norms,
                                mattr_window=50, include_domain=False)
    out = args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sources.jsonl", [r.to_dict() for r in sources])
    write_jsonl(out / "bundles.jsonl", result["bundles"])
    write_jsonl(out / "original_features.jsonl", features)
    runs = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        runs.append({"run_id": path.name, "path": str(path), "manifest_game_id": manifest.get("game_id"),
                     "source_model": manifest.get("model"), "seed": manifest.get("random_seed"),
                     "players": manifest.get("num_players"), "outcome": manifest.get("game_state_status"),
                     "rewriter": manifest.get("rewriter"), "treatment_evidence": treatment_evidence(path),
                     "raw_sha256": {name: sha256_file(path / name) for name in
                                    ("manifest.json", "game_nl.json", "public_history.json")}})
    provenance = {
        "schema_version": 1, "created_at": now_iso(), "source_runs": runs,
        "source_type": "untreated Stage-2 game speech", "planned_test_model_family": "Qwen3.5",
        "possible_additional_test_model_family": "DeepSeek",
        "sampling": {"rule": "all public speech records; one prefix bundle per game/speaker",
                     "window_words": list(args.window), "max_per_game": None,
                     "use": "descriptive baseline and pilot preparation"},
        "history": "strictly before the source speech; private roles and future turns are excluded",
        "instruments": {"spacy": nlp_provenance(nlp) if nlp else None, "mattr_window": 50,
                        "include_domain_terms": False, "subtlex": norms.provenance() if norms else None},
        "artifacts": {name: sha256_file(out / name) for name in ARTIFACTS},
        "validation_scope": "automated linguistic measures and model perception",
        "validation_evidence": {"proficiency_rewrites": False,
                                "grammar_guardrails": False, "listener_perception": False},
    }
    dump_json(out / "provenance.json", provenance)
    report = summarize(out)
    dump_json(out / "report.json", report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(report), encoding="utf-8")
    print(f"{len(sources)} original utterances, {len(result['bundles'])} bundles -> {out}")
    print(f"report -> {args.report}")
    return 0


def summarize(data_dir):
    data_dir = Path(data_dir)
    prov = json.loads((data_dir / "provenance.json").read_text(encoding="utf-8"))
    for name in ARTIFACTS:
        if prov["artifacts"].get(name) != sha256_file(data_dir / name):
            raise ValueError(f"{name} no longer matches its provenance; rebuild the prepared data")
    sources = [SourceRecord.from_dict(r) for r in read_jsonl(data_dir / "sources.jsonl")]
    check_sources(sources)
    bundles = read_jsonl(data_dir / "bundles.jsonl")
    features = read_jsonl(data_dir / "original_features.jsonl")
    assert_unique(bundles, ("bundle_id",))
    assert_unique(features, ("bundle_id", "condition"))
    window = prov["sampling"]["window_words"]
    expected = build_bundles(sources, window_min=window[0], window_max=window[1], max_per_game=None)
    if bundles != expected["bundles"]:
        raise ValueError("bundles do not follow the archived source selection rule")
    if {(r["bundle_id"], r["condition"]) for r in features} != {(b["bundle_id"], "original") for b in bundles}:
        raise ValueError("original feature cohort mismatch")
    grouped, texts = defaultdict(list), defaultdict(list)
    for source in sources:
        grouped[source.game_id].append(source)
        texts[source.original_nl].append(source.source_id)
    corpus = audit(grouped)
    duplicate_groups = [ids for ids in texts.values() if len(ids) > 1]
    cross_game = [ids for ids in duplicate_groups if len({x.split(":")[0] for x in ids}) > 1]
    baseline = {}
    for key, label in BASELINE_METRICS:
        values = [r[key] for r in features if r.get(key) is not None]
        baseline[key] = {"label": label, "n_bundles": len(values), "mean": mean(values), "median": median(values)}
    return {"provenance": prov, "corpus": corpus, "n_bundles": len(bundles),
            "bundles_by_game": dict(Counter(b["game_id"] for b in bundles)),
            "n_insufficient_speakers": len(expected["insufficient"]),
            "selected_source_count": len({sid for b in bundles for sid in b["source_ids"]}),
            "unique_original_texts": len(texts), "duplicate_text_groups": duplicate_groups,
            "cross_game_duplicate_text_groups": cross_game, "baseline": baseline,
            "validation_status": "NOT YET VALIDATED"}


def render(report):
    c, p = report["corpus"], report["provenance"]
    lines = ["# Proficiency Validation", "",
        "This study evaluates controlled changes in vocabulary and syntactic range while preserving "
        "meaning, grammaticality and pragmatic force. This report describes the untreated source corpus "
        "and its baseline language measures.", "",
        "The statistics are computed from `data/` using "
        "`python -m experiments.proficiency_validation.code.cli report`.", "", "## Setup", "",
        f"- {c['n_games']} untreated games with {c['n_utterances']} public speech utterances",
        f"- {len(c['predicates'])} observed speech-act types",
        f"- {report['n_bundles']} original passages assembled from {report['selected_source_count']} utterances",
        "- descriptive analysis of Original speech", "",
        "| Run | Recorded model | Seed | Players | Utterances | Bundles |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for run in p["source_runs"]:
        rid = run["run_id"]
        lines.append(f"| {rid} | {run['source_model']} | {run['seed']} | {run['players']} | "
                     f"{c['per_game'][rid]['n_utterances']} | {report['bundles_by_game'].get(rid, 0)} |")
    lines += ["", "The source manifests and speech records identify all runs as untreated. "
              "Directory names identify games because the manifests reuse the same internal game ID. "
              "Source hashes are recorded in data/provenance.json.", "",
              "Initial validation testing is planned with Qwen3.5. DeepSeek is a candidate "
              "for subsequent tests.", "",
              "## Source coverage", "", "| Speech act | Utterances |", "| --- | ---: |"]
    lines += [f"| {name} | {n} |" for name, n in c["predicates"].items()]
    lines += ["", "Role claims and divine-result claims are included. Only public information before "
              "each utterance is retained for rewriting and judgement.", "",
              f"There are {report['unique_original_texts']} distinct original texts. "
              f"{len(report['cross_game_duplicate_text_groups'])} exact-text groups occur in more than one game. "
              "The corpus retains repeated wording, with games serving as the independent sampling units.", "",
              "## Passage construction", "",
              f"The descriptive window is {p['sampling']['window_words'][0]} to "
              f"{p['sampling']['window_words'][1]} words. Whole utterances are appended in speech order "
              "until the minimum is reached, without crossing the maximum. There is one passage per speaker "
              "per game and no per-game cap in this source audit.", "",
              f"{report['n_insufficient_speakers']} speaker records cannot fill this window. "
              "This baseline eligibility rule is applied before any proficiency rewrite.", "",
              "| Candidate window | Eligible passages |", "| --- | ---: |"]
    for minimum in c["windows"]:
        count = c["candidate_bundles_by_window"].get(minimum, c["candidate_bundles_by_window"].get(str(minimum)))
        lines.append(f"| {minimum} to {minimum + c['window_span']} words | {count} |")
    lines += ["", "## Original language measures", "",
              "| Measure | Valid passages | Mean | Median |", "| --- | ---: | ---: | ---: |"]
    for row in report["baseline"].values():
        lines.append(f"| {row['label']} | {row['n_bundles']} | {fmt(row['mean'])} | {fmt(row['median'])} |")
    spacy = p["instruments"]["spacy"]
    syntax_note = (f"Syntax uses {spacy['lang']}_{spacy['model']} {spacy['model_version']} "
                   f"with spaCy {spacy['spacy_version']}.") if spacy else "Syntax was not measured."
    lines += ["", syntax_note, "MATTR uses a 50-token window. Player and statement IDs are excluded "
              "from lexical diversity; domain words are excluded from content-word measures. "
              "Unavailable SUBTLEX measurements are reported as missing. "
              "These statistics describe the Original condition.", "",
              "## Validation evidence", "",
              "| Required evidence | Current status |", "| --- | --- |",
              "| Untreated originals with traceable source records | Available |",
              "| Low, Neutral and High proficiency rewrites | Not collected |",
              "| English structural, inverse-parser and model fidelity checks on rewrites | Not run |",
              "| LanguageTool and CoLA guardrails on the rewrites | Not run |",
              "| Blind model perception comparisons | Not run |",
              "| Frozen confirmatory protocol and independent replication | Not available |", "",
              "## Interpretation", "",
              "The corpus provides a descriptive baseline for the proficiency pilot. Validation of the "
              "Low and High manipulations is pending rewrite generation, automated measurements "
              "and independent model checks.", "",
              "The validation scope is measured linguistic range and model perception. Generalization "
              "to human perception or overall English ability requires separate evidence.", "",
              "The three source games support development and source auditing. Confirmation and replication "
              "will use fresh, disjoint game cohorts under a frozen protocol with justified equivalence "
              "margins and sample size.", ""]
    return "\n".join(lines)


def report_command(args):
    report = summarize(args.data_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(report), encoding="utf-8")
        print(f"report -> {args.out}")
    else:
        print(render(report))
    if args.json:
        dump_json(args.json, report)
    return 0


def add_parser(sub):
    p = sub.add_parser("prepare-data", help="prepare untreated proficiency sources and descriptive report")
    p.add_argument("game_dirs", nargs="+")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--window", type=int, nargs=2, default=[150, 250])
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--subtlex", type=Path)
    p.add_argument("--report", type=Path, default=EXPERIMENT_DIR / "VALIDATION.md")
    p.set_defaults(func=prepare)
    p = sub.add_parser("report", help="recompute the source report from curated proficiency data")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    p.set_defaults(func=report_command)
