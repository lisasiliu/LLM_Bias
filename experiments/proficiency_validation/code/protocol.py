"""Validation and data provenance for the confirmatory English protocol."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from .common import GUARDRAIL_METRICS, RANGE_METRICS, sha256_text, sha256_file, dump_json


def digest(value):
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def number(value, lo=None, hi=None, integer=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (lo is None or value >= lo) and (hi is None or value <= hi)
            and (not integer or isinstance(value, int)))


def validate_design(f):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    require(f.get("schema_version") == 3, "schema_version must be 3")
    require(f.get("validation_scope") == "automated", "validation_scope must be automated")
    conditions = f.get("conditions", [])
    require(conditions in (["original", "neutral", "low"], ["original", "neutral", "low", "high"]), "invalid conditions")
    primary = f.get("primary_range_metrics", [])
    require(bool(primary) and len(primary) == len(set(primary)) and set(primary) <= set(RANGE_METRICS), "invalid primary_range_metrics")
    families = ({"subtlex_mean"}, {"mattr", "mtld", "repetition_rate", "top5_share"},
                {"subordinate_clauses_per_sentence", "mean_dependency_distance", "tree_depth_mean", "np_mean_length", "mean_sentence_length"})
    require(all(set(primary) & x for x in families), "primary metrics must cover lexical familiarity, diversity and syntax")
    require(not (set(primary) & {"clause_pattern_entropy", "clause_pattern_distinct_ratio"}), "custom clause-pattern metrics are exploratory")
    for m in set(primary) | set(GUARDRAIL_METRICS):
        require(number(f.get("equivalence_bounds", {}).get(m), 1e-12), f"positive equivalence bound required: {m}")
        require(len(str(f.get("bound_rationale", {}).get(m, "")).strip()) >= 20, f"substantive bound rationale required: {m}")
    for c in (c for c in ("low", "high") if c in conditions):
        for m in primary:
            require(number(f.get("range_effect_thresholds", {}).get(f"{c}_vs_neutral", {}).get(m), 0), f"effect threshold required: {c}/{m}")
    limits = {
        "min_games": (2, None, True), "min_bundles": (2, None, True),
        "min_valid_fraction": (0.01, 1, False), "min_subtlex_coverage": (0.01, 1, False),
        "listener_equivalence_margin": (1e-6, .49, False),
        "neutral_edit_distance_tolerance": (0, 1, False), "neutral_match_min_fraction": (.01, 1, False),
        "mattr_window": (2, None, True), "max_bundles_per_game": (1, None, True),
        "n_neutral_candidates": (1, None, True), "max_corrective_retries": (0, None, True),
        "generation_temperature": (0, 2, False), "generation_max_tokens": (1, None, True),
        "judge_max_tokens": (1, None, True), "probe_max_tokens": (1, None, True)}
    for key, (lo, hi, integer) in limits.items():
        require(number(f.get(key), lo, hi, integer), f"valid {key} required")
    window = f.get("bundle_window_words", [])
    require(len(window) == 2 and all(number(x, 1, integer=True) for x in window) and window[0] <= window[1], "invalid bundle window")
    for key in ("power_rationale", "listener_margin_rationale", "absolute_quality_rationale"):
        require(len(str(f.get(key, "")).strip()) >= 20, f"{key} required")
    absolute = f.get("absolute_quality", {})
    require(number(absolute.get("grammar_errors_max"), 0), "absolute grammar_errors_max required")
    require(number(absolute.get("cola_mean_min"), 0, 1), "absolute cola_mean_min required")
    for key, lo in (("n_boot", 1000), ("n_perm", 1000), ("seed", 0)):
        require(number(f.get("statistics", {}).get(key), lo, integer=True), f"statistics.{key} required")
    inst = f.get("instruments", {})
    for key in ("spacy_model", "spacy_model_version", "spacy_version", "languagetool_version", "languagetool_language",
                "cola_model", "cola_commit_hash", "subtlex_sha256", "subtlex_measure", "cola_label_rationale"):
        require(bool(inst.get(key)), f"instruments.{key} required")
    require(inst.get("cola_acceptable_index") in (0, 1), "cola_acceptable_index must be 0 or 1")
    require(isinstance(inst.get("include_domain_terms"), bool), "include_domain_terms must be a boolean")
    require(number(inst.get("cola_max_length"), 8, integer=True), "cola_max_length required")
    if f.get("human_validation") is not None:
        # Optional annotation settings never become automated acceptance criteria.
        human = f["human_validation"]
        for key, lo in (("min_raters", 2), ("min_items_per_condition", 1), ("min_games_per_condition", 2)):
            require(number(human.get(key), lo, integer=True), f"human_validation.{key} required")
        require(number(human.get("min_pass_rate"), .01, 1), "human_validation.min_pass_rate required")
        for key, lo in (("min_perception_items", 1), ("min_perception_games", 2)):
            require(number(human.get(key), lo, integer=True), f"human_validation.{key} required")
        require(number(human.get("max_abstention_fraction"), 0, .99), "human_validation.max_abstention_fraction required")
    models = f.get("models", {})
    listeners = models.get("listeners", [])
    require(len(listeners) >= 3 and len(listeners) == len(set(listeners)), "three distinct listener handles required")
    identities = models.get("identities", {})
    for handle in [models.get("generator"), models.get("judge")] + listeners:
        identity = identities.get(handle, {})
        for key in ("model", "family", "revision", "tokenizer_revision", "serving_version", "precision", "chat_template"):
            v = identity.get(key)
            require(isinstance(v, str) and bool(v.strip()) and v.lower() not in ("local", "main", "latest", "todo") and not v.startswith("<"), f"model metadata {handle}.{key} required")
        for key in ("revision", "tokenizer_revision"):
            require(bool(re.fullmatch(r"(?:sha256:)?(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})", str(identity.get(key, "")))),
                    f"{handle}.{key} must be a commit hash or artifact SHA-256")
    lf = [identities.get(h, {}).get("family") for h in listeners]
    gen_family = identities.get(models.get("generator"), {}).get("family")
    require(len(lf) == len(set(lf)) and gen_family not in lf, "listener families must differ from one another and generator")
    behavior = models.get("behavioral_families", [])
    require(bool(behavior) and gen_family not in behavior, "declare behavioral families; generator must be separate")
    seen_games = set()
    for split in ("pilot", "set_a", "set_b"):
        require(bool(re.fullmatch(r"[a-f0-9]{64}", str(f.get("sources_sha256", {}).get(split, "")))),
                f"source artifact hash required for {split}")
        bundles = f.get("splits", {}).get(split, [])
        require(bool(bundles), f"preselected bundles required for {split}")
        ids, games = [b.get("bundle_id") for b in bundles], {b.get("game_id") for b in bundles}
        require(None not in games and "" not in games and None not in ids and len(ids) == len(set(ids)), f"invalid IDs in {split}")
        require(not (seen_games & games), f"games overlap across splits at {split}")
        seen_games |= games
        if split != "pilot":
            require(len(games) >= (f.get("min_games") or 2) and len(bundles) >= (f.get("min_bundles") or 2), f"{split} is below frozen minimum n")
        for b in bundles:
            require(bool(b.get("source_ids")) and b.get("window") == window, f"invalid sources/window in {split}")
        cap = f.get("max_bundles_per_game")
        if isinstance(cap, int):
            require(all(sum(b.get("game_id") == g for b in bundles) <= cap for g in games), f"bundle cap exceeded in {split}")
    return errors


def cohort(frozen, split):
    if split not in ("set_a", "set_b"):
        raise ValueError("--split set_a or set_b is required for a frozen run")
    return frozen["splits"][split]


def assert_unique(rows, keys):
    seen = set()
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if any(v is None for v in key) or key in seen:
            raise ValueError(f"missing or duplicate {keys}: {key}")
        seen.add(key)


def artifact_provenance(frozen, split, *, inputs=None):
    return {"frozen_sha256": digest(frozen) if frozen else None, "split": split,
            "input_sha256": {k: sha256_file(Path(p)) for k, p in (inputs or {}).items() if p}}


def artifact_sidecar(path, provenance):
    path = Path(path)
    dump_json(path.with_name(path.stem + "_provenance.json"), {**provenance, "output_sha256": sha256_file(path)})


def verify_artifact(path, frozen, split):
    path = Path(path)
    side = path.with_name(path.stem + "_provenance.json")
    if not side.exists():
        return None, [f"missing provenance: {path.name}"]
    prov = json.loads(side.read_text(encoding="utf-8"))
    errors = []
    for key, expected in (("output_sha256", sha256_file(path)), ("frozen_sha256", digest(frozen)), ("split", split)):
        if prov.get(key) != expected:
            errors.append(f"{path.name}: {key} mismatch")
    return prov, errors
