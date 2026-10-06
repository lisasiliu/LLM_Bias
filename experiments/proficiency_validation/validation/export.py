"""Publication-facing export of one pilot run: ``data/provenance.json`` (models, settings, hashes,
counts) and ``data/automatic_validation.jsonl`` (one row per passage and condition with every
automatic measurement). Raw and intermediate artifacts stay under ``data/runs/<run>/``."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from components.helpers import read_jsonl, write_jsonl
from experiments.proficiency_validation.rewrite.runtime import DATA_DIR, dump_json, git_commit, now_iso, read_metadata, sha256_file

FILES = {"features": "linguistic_features.jsonl", "assembled": "linguistic_features_assembled.jsonl",
         "guardrails": "grammar_guardrails.jsonl", "range_listener": "range_listener.jsonl",
         "proficiency_listener": "perceived_proficiency_listener.jsonl", "cefr": "cefr_auxiliary.jsonl"}
MEASURE_KEYS = ("word_count", "n_sentences", "mean_sentence_length", "subtlex_mean", "mattr", "mtld", "repetition_rate",
                "top5_share", "subordinate_clause_rate", "subordinate_clauses_per_sentence", "mean_dependency_distance",
                "tree_depth_mean", "np_mean_length", "clause_pattern_entropy", "clause_pattern_distinct_ratio",
                "edit_distance_to_original")


def _rows(measured: Path, cohort: str) -> List[dict]:
    if not (measured / FILES["features"]).exists():
        return []
    features = {(r["bundle_id"], r["condition"]): r for r in read_jsonl(measured / FILES["features"])}
    guard = {(r["bundle_id"], r["condition"]): r for r in read_jsonl(measured / FILES["guardrails"])} if (measured / FILES["guardrails"]).exists() else {}
    cefr = {(r["bundle_id"], r["condition"]): r for r in read_jsonl(measured / FILES["cefr"])} if (measured / FILES["cefr"]).exists() else {}
    listeners: Dict[tuple, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for key in ("range_listener", "proficiency_listener"):
        if (measured / FILES[key]).exists():
            from experiments.proficiency_validation.validation.analysis import signed_value
            for r in read_jsonl(measured / FILES[key]):
                s = signed_value(r)
                if s is not None:
                    listeners[(r["bundle_id"], r["pair"])][r.get("construct", "range")].append(s)
    out = []
    for (bundle, condition), f in sorted(features.items()):
        g, c = guard.get((bundle, condition), {}), cefr.get((bundle, condition), {})
        row = {"cohort": cohort, "bundle_id": bundle, "game_id": f.get("game_id"), "speaker": f.get("speaker"),
               "condition": condition, "complete": f.get("complete"), "n_utterances": f.get("n_utterances"),
               **{k: f.get(k) for k in MEASURE_KEYS},
               "grammar_errors_per_100_words": g.get("grammar_errors_per_100_words"), "cola_mean": g.get("cola_mean"),
               "cefr_auxiliary_band": c.get("level") if c.get("status") == "rated" else None}
        for (b, pair), constructs in listeners.items():
            if b == bundle and pair.split(":")[0] == condition:
                for construct, values in constructs.items():
                    row[f"listener_signed_{construct}_{pair.replace(':', '_vs_')}"] = sum(values) / len(values)
        out.append(row)
    return out


def export(pilot_dir: Path, data_dir: Path) -> dict:
    pilot_dir, data_dir = Path(pilot_dir), Path(data_dir)
    manifest = json.loads((pilot_dir / "manifest.json").read_text(encoding="utf-8"))
    rewrites = read_jsonl(pilot_dir / "rewrite" / "rewrites.jsonl")
    rows = _rows(pilot_dir / "validation", "primary_phase_stratified") + _rows(pilot_dir / "validation" / "whole_game", "robustness_whole_game")
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
        "schema_version": 1, "created_at": now_iso(), "run_directory": pilot_dir.name,
        "git_commit": git_commit(), "manifest_format": manifest.get("format"),
        "models": manifest.get("models"), "configuration": manifest.get("configuration"),
        "prompt_versions": dict(Counter(r.get("prompt_version") for r in rewrites)),
        "fidelity_rubric_versions": dict(Counter((r.get("gate_provenance") or {}).get("semantic_rubric_version") for r in rewrites)),
        "gate_policy": {k: (rewrites[0].get("gate_provenance") or {}).get(k) for k in
                        ("parser_abstention", "neutral_exposure_floor", "parser_context_words", "second_opinion")} if rewrites else {},
        "conditions": sorted({r["condition"] for r in rewrites}),
        "acceptance": {c: sum(1 for r in rewrites if r["condition"] == c and r["status"] == "ok") for c in sorted({r["condition"] for r in rewrites})},
        "n_sources": len({r["source_id"] for r in rewrites}),
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
