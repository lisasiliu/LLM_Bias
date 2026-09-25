"""Record explicit automated-validation decisions and preselected source splits."""
from __future__ import annotations
import json
from pathlib import Path

from components.helpers import read_jsonl
from components.stats import mean, percentile, sd
from .common import FROZEN_CONFIG, RANGE_METRICS, GUARDRAIL_METRICS, dump_json, now_iso, git_commit, sha256_file
from .frozen import frozen_hashes, enforce_frozen
from .models import model_identity
from .protocol import validate_design


def suggest_bounds(features, guardrails=None):
    from .analysis import paired_clusters
    out = {}
    for rows, metrics in ((features, RANGE_METRICS), (guardrails or [], GUARDRAIL_METRICS)):
        for metric in metrics:
            clusters, n = paired_clusters(rows, metric, "neutral", "original")
            values = [v for group in clusters.values() for v in group]
            if values:
                out[metric] = {"n": n, "mean_diff": mean(values), "sd_diff": sd(values),
                               "p90_abs_diff": percentile([abs(v) for v in values], 90)}
    return out


def render_bounds(table):
    return (json.dumps(table, indent=2) + "\n"
            "Pilot differences describe observed variability for prospective power calculations.\n"
            "Equivalence bounds represent the largest scientifically negligible difference in raw units.\n"
            "Each bound requires a scientific justification recorded before confirmation.")


def add_parser(sub):
    p = sub.add_parser("suggest-bounds")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--guardrails", type=Path)
    p.set_defaults(func=main_suggest)
    p = sub.add_parser("freeze", help="freeze the automated English validation protocol")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--pilot-bundles", type=Path, required=True)
    p.add_argument("--set-a-bundles", type=Path, required=True)
    p.add_argument("--set-b-bundles", type=Path, required=True)
    p.add_argument("--pilot-sources", type=Path, required=True)
    p.add_argument("--set-a-sources", type=Path, required=True)
    p.add_argument("--set-b-sources", type=Path, required=True)
    p.add_argument("--subtlex", type=Path, required=True)
    p.add_argument("--out", type=Path, default=FROZEN_CONFIG)
    p.set_defaults(func=main_freeze)
    p = sub.add_parser("verify-frozen")
    p.add_argument("--frozen", type=Path, default=FROZEN_CONFIG)
    p.set_defaults(func=main_verify)


def main_suggest(args):
    print(render_bounds(suggest_bounds(read_jsonl(args.features),
                                       read_jsonl(args.guardrails) if args.guardrails else None)))
    return 0


def main_freeze(args):
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing freeze; use a new protocol filename.")
    config = json.loads(args.decisions.read_text(encoding="utf-8"))
    config.update(schema_version=3, validation_scope="automated", frozen_at=now_iso(), git_commit=git_commit())
    config["splits"] = {key: read_jsonl(path) for key, path in (
        ("pilot", args.pilot_bundles), ("set_a", args.set_a_bundles), ("set_b", args.set_b_bundles))}
    config["sources_sha256"] = {key: sha256_file(path) for key, path in (
        ("pilot", args.pilot_sources), ("set_a", args.set_a_sources), ("set_b", args.set_b_sources))}
    config.setdefault("instruments", {})["subtlex_sha256"] = sha256_file(args.subtlex)
    config["instruments"]["subtlex_path"] = str(args.subtlex)
    models = config.setdefault("models", {})
    handles = [models.get("generator"), models.get("judge")] + models.get("listeners", [])
    if any(not h for h in handles):
        raise ValueError("generator, judge and listeners must be explicitly selected")
    models["identities"] = {h: model_identity(h) for h in handles}
    errors = validate_design(config)
    if errors:
        raise ValueError("Cannot freeze:\n" + "\n".join(errors))
    config["hashes"] = frozen_hashes(config["n_neutral_candidates"])
    dump_json(args.out, config)
    print(f"frozen protocol -> {args.out}")
    return 0


def main_verify(args):
    enforce_frozen(args.frozen, check_models=True)
    print("design, code and model declarations match the frozen protocol")
    return 0
