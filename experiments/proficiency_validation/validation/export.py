"""Publication-facing export of one pilot run: ``data/provenance.json`` (models, settings, hashes,
counts) and ``data/automatic_validation.jsonl`` (one row per passage and condition with every
automatic measurement). Raw and intermediate artifacts stay under ``data/runs/<run>/``."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from components.helpers import read_jsonl, write_jsonl
from experiments.proficiency_validation.rewrite.runtime import DATA_DIR, assert_unique, condition_label, dump_json, git_commit, now_iso, read_metadata, sha256_file
from experiments.proficiency_validation.validation.analysis import complete_listener_rows, signed_value
from experiments.proficiency_validation.rewrite.fidelity import audit_accepted_alignments

FILES = {"features": "linguistic_features.jsonl", "assembled": "linguistic_features_assembled.jsonl",
         "guardrails": "grammar_guardrails.jsonl", "range_listener": "range_listener.jsonl",
         "proficiency_listener": "perceived_proficiency_listener.jsonl", "cefr": "cefr_auxiliary.jsonl"}
MEASURE_KEYS = ("word_count", "n_sentences", "mean_sentence_length", "subtlex_mean", "mattr", "mtld", "repetition_rate",
                "top5_share", "subordinate_clause_rate", "subordinate_clauses_per_sentence", "mean_dependency_distance",
                "tree_depth_mean", "np_mean_length", "clause_pattern_entropy", "clause_pattern_distinct_ratio",
                "edit_distance_to_original")


def _rows(measured: Path, cohort: str, *, alignment_issues=()) -> List[dict]:
    if not (measured / FILES["features"]).exists():
        return []
    def indexed(key, fields):
        path = measured / FILES[key]
        rows = read_jsonl(path) if path.exists() else []
        assert_unique(rows, fields)
        return {tuple(r[k] for k in fields): r for r in rows}

    features = indexed("features", ("bundle_id", "condition"))
    assembled = indexed("assembled", ("bundle_id", "condition"))
    guard = indexed("guardrails", ("bundle_id", "condition"))
    cefr = defaultdict(dict)
    for (bundle, condition, rater), c in indexed("cefr", ("bundle_id", "condition", "rater")).items():
        cefr[(bundle, condition)][rater] = {"band": c.get("level") if c.get("ok") and c.get("status") == "rated" else None,
                                          "status": c.get("status"), "family": c.get("rater_family")}
    listeners: Dict[tuple, dict] = defaultdict(lambda: defaultdict(dict))
    for key in ("range_listener", "proficiency_listener"):
        paths = sorted(p for p in measured.glob(FILES[key].replace(".jsonl", "*.jsonl")) if not p.name.endswith(".partial"))
        if paths:
            rows, _, _ = complete_listener_rows([r for p in paths for r in read_jsonl(p)])
            for r in rows:
                rating = listeners[(r["bundle_id"], r["pair"])][r.get("construct", "range")].setdefault(
                    r["listener"], {"family": r.get("listener_family"), "by_order": {}})
                rating["by_order"][r["order"]] = signed_value(r)
    for constructs in listeners.values():
        for ratings in constructs.values():
            for rating in ratings.values():
                rating["signed_preference"] = sum(rating["by_order"].values()) / 2
    out = []
    for (bundle, condition), f in sorted(features.items()):
        g = guard.get((bundle, condition), {})
        row = {"cohort": cohort, "bundle_id": bundle, "game_id": f.get("game_id"), "speaker": f.get("speaker"),
               "condition": condition, "condition_label": condition_label(condition), "complete": f.get("complete"), "n_utterances": f.get("n_utterances"),
               **{k: f.get(k) for k in MEASURE_KEYS},
               "grammar_errors_per_100_words": g.get("grammar_errors_per_100_words"), "cola_mean": g.get("cola_mean"),
               "cefr_auxiliary_ratings": cefr.get((bundle, condition), {}), "listener_ratings": {},
               "alignment_evidence_issues": [r for r in alignment_issues if r["condition"] == condition
                   and r["source_id"] in assembled.get((bundle, condition), {}).get("source_ids", [])]}
        for (b, pair), constructs in listeners.items():
            if b == bundle and pair.split(":")[0] == condition:
                for construct, ratings in constructs.items():
                    row["listener_ratings"].setdefault(pair, {})[construct] = ratings
                    row[f"listener_signed_{construct}_{pair.replace(':', '_vs_')}"] = (
                        sum(r["signed_preference"] for r in ratings.values()) / len(ratings))
        out.append(row)
    return out


def export(pilot_dir: Path, data_dir: Path) -> dict:
    pilot_dir, data_dir = Path(pilot_dir), Path(data_dir)
    manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    rewrites = read_jsonl(pilot_dir / "rewrite" / "rewrites.jsonl")
    alignment_audit = audit_accepted_alignments(rewrites)
    rows = _rows(pilot_dir / "validation", "primary_phase_stratified", alignment_issues=alignment_audit["issues"])
    rows += _rows(pilot_dir / "validation" / "whole_game", "robustness_whole_game", alignment_issues=alignment_audit["issues"])
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "automatic_validation.jsonl", rows)
    report_path = pilot_dir / "validation" / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    pilot = report.get("pilot", {})
    calibration_path = pilot_dir / "validation" / "calibration" / "report.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8")).get("overall") if calibration_path.exists() else None
    instruments = {}
    for name, key in (("features", "features"), ("guardrails", "guardrails")):
        meta = read_metadata(pilot_dir / "validation" / FILES[key]) or {}
        instruments[name] = {k: v for k, v in meta.items() if k in ("spacy", "subtlex", "languagetool", "cola", "instruments", "mattr_window")}
    provenance = {
        "schema_version": 2, "created_at": now_iso(), "run_directory": pilot_dir.name,
        "git_commit": git_commit(), "manifest_format": manifest.get("format"),
        "models": manifest.get("models"), "configuration": manifest.get("configuration"),
        "prompt_versions": dict(Counter(r.get("prompt_version") for r in rewrites)),
        "fidelity_rubric_versions": dict(Counter((r.get("gate_provenance") or {}).get("semantic_rubric_version") for r in rewrites)),
        "gate_policy": {k: (rewrites[0].get("gate_provenance") or {}).get(k) for k in
                        ("parser_abstention", "neutral_exposure_floor", "parser_context_words", "second_opinion")} if rewrites else {},
        "conditions": sorted({r["condition"] for r in rewrites}),
        "acceptance": {c: sum(1 for r in rewrites if r["condition"] == c and r["status"] == "ok") for c in sorted({r["condition"] for r in rewrites})},
        "n_sources": len({r["source_id"] for r in rewrites}),
        "accepted_alignment_audit": alignment_audit,
        "passage_rules": {"primary": "phase_stratified", "robustness": "whole_game"},
        "passage_coverage": pilot.get("coverage") and {k: {kk: vv for kk, vv in v.items() if kk != "bundles" and not kk.endswith("_ids")}
                                                       for k, v in pilot["coverage"].items()},
        "judge_calibration_overall": calibration, "instruments": instruments,
        "input_sha256": {"rewrites": sha256_file(pilot_dir / "rewrite" / "rewrites.jsonl"),
                         "manifest": sha256_file(pilot_dir / "manifest.json"),
                         "automatic_validation": sha256_file(data_dir / "automatic_validation.jsonl")},
        "n_automatic_validation_rows": len(rows),
    }
    dump_json(data_dir / "provenance.json", provenance)
    return provenance


def add_parser(sub) -> None:
    p = sub.add_parser("export", help="write data/provenance.json and data/automatic_validation.jsonl for a pilot run")
    p.add_argument("--pilot-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR.parent)
    p.set_defaults(func=main)


def main(args) -> int:
    provenance = export(args.pilot_dir, args.data_dir)
    print(f"{provenance['n_automatic_validation_rows']} rows -> {args.data_dir / 'automatic_validation.jsonl'}; provenance -> {args.data_dir / 'provenance.json'}")
    return 0
