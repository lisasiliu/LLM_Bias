"""Record explicit pilot decisions, independent calibration and preselected splits."""
from __future__ import annotations
import json
from pathlib import Path

from components.helpers import read_jsonl
from components.stats import mean, percentile, sd
from .common import FROZEN_CONFIG, RANGE_METRICS, GUARDRAIL_METRICS, dump_json, now_iso, git_commit, sha256_file
from .frozen import HASHED_MODULES, frozen_hashes, verify_hashes, enforce_frozen
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
            "These are descriptive pilot differences, NOT automatic equivalence bounds.\n"
            "Choose the largest scientifically negligible difference in raw units and justify it.\n"
            "Use pilot variability for prospective power; do not loosen a margin to obtain equivalence.")


def add_parser(sub):
    p = sub.add_parser("suggest-bounds")
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--guardrails", type=Path)
    p.set_defaults(func=main_suggest)
    p = sub.add_parser("freeze", help="validate and freeze explicit decisions and independent English calibration")
    p.add_argument("--decisions", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--pilot-bundles", type=Path, required=True)
    p.add_argument("--set-a-bundles", type=Path, required=True)
    p.add_argument("--set-b-bundles", type=Path, required=True)
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
    config.update(schema_version=2, frozen_at=now_iso(), git_commit=git_commit())
    config["splits"] = {key: read_jsonl(path) for key, path in (
        ("pilot", args.pilot_bundles), ("set_a", args.set_a_bundles), ("set_b", args.set_b_bundles))}
    config["calibration"] = json.loads(args.calibration.read_text(encoding="utf-8"))
    config["calibration_sha256"] = sha256_file(args.calibration)
    config.setdefault("instruments", {})["subtlex_sha256"] = sha256_file(args.subtlex)
    config["instruments"]["subtlex_path"] = str(args.subtlex)
    models = config.setdefault("models", {})
    handles = [models.get("generator"), models.get("judge")] + models.get("listeners", [])
    if any(not h for h in handles):
        raise ValueError("generator, judge and listeners must be explicitly selected")
    models["identities"] = {h: model_identity(h) for h in handles}
    errors = validate_design(config)
    calibration_games = set(config["calibration"].get("game_ids", []))
    pilot_games = {b["game_id"] for b in config["splits"]["pilot"]}
    if not calibration_games or not calibration_games <= pilot_games:
        errors.append("calibration games must belong to the pilot split, never Set A/B")
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
