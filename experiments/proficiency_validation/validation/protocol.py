"""Proficiency validation protocol."""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from experiments.proficiency_validation.rewrite.runtime import GUARDRAIL_METRICS, RANGE_METRICS, sha256_text, sha256_file, dump_json
import hashlib
import importlib
import inspect
import sys
from typing import Dict, List, Optional
from experiments.proficiency_validation.rewrite.fidelity import SEMANTIC_RUBRIC_VERSION, semantic_prompt_sha
from components.judges.inverse_predicate_parser import build_system_prompt as parser_system_prompt
from components.rewriters.language_rewriter import PROMPT_VERSION, LanguageProficiencyRewriter
from lurkingwolf.rewriter import RewriteContext
from experiments.proficiency_validation.rewrite.runtime import FROZEN_CONFIG, load_json
from components.helpers import read_jsonl
from components.stats import mean, percentile, sd
from experiments.proficiency_validation.rewrite.runtime import now_iso, git_commit
from experiments.proficiency_validation.rewrite.runtime import model_identity


def number(value, lo=None, hi=None, integer=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (lo is None or value >= lo) and (hi is None or value <= hi)
            and (not integer or isinstance(value, int)))


def validate_design(f):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(message)
    require(f.get("schema_version") == 6, "schema_version must be 6")
    require(f.get("validation_scope") == "automated", "validation_scope must be automated")
    require(f.get("estimand") == "equal_game_weight", "estimand must be equal_game_weight")
    constructs = f.get("perception_constructs", [])
    require(isinstance(constructs, list) and len(constructs) == len(set(constructs))
            and "range" in constructs and set(constructs) <= {"range", "perceived_proficiency"},
            "perception_constructs must declare range and any perceived_proficiency claim")
    conditions = f.get("conditions", [])
    require(isinstance(conditions, list) and len(conditions) == 3
            and set(conditions) in ({"original", "neutral", "low"}, {"original", "neutral", "high"}),
            "one treatment and its Neutral control are required per frozen study")
    require(f.get("neutral_anchor_condition") in ("low", "high") and f.get("neutral_anchor_condition") in conditions,
            "neutral_anchor_condition must name the declared treatment")
    primary = f.get("primary_range_metrics", [])
    require(bool(primary) and len(primary) == len(set(primary)) and set(primary) <= set(RANGE_METRICS), "invalid primary_range_metrics")
    families = ({"subtlex_mean"}, {"mattr", "mtld", "repetition_rate", "top5_share"},
                {"subordinate_clause_rate", "mean_dependency_distance", "tree_depth_mean", "np_mean_length"})
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
        "listener_min_separation": (1e-6, .49, False),
        "min_treatment_availability": (.01, 1, False), "max_differential_attrition": (0, 1, False),
        "neutral_edit_distance_tolerance": (0, 1, False), "neutral_match_min_fraction": (.01, 1, False),
        "mattr_window": (2, None, True), "max_bundles_per_game": (1, None, True),
        "n_neutral_candidates": (1, None, True), "max_corrective_retries": (0, None, True),
        "generation_temperature": (0, 2, False), "generation_max_tokens": (1, None, True),
        "judge_max_tokens": (1, None, True), "probe_max_tokens": (1, None, True)}
    for key, (lo, hi, integer) in limits.items():
        require(number(f.get(key), lo, hi, integer), f"valid {key} required")
    window = f.get("bundle_window_words", [])
    selection = f.get("bundle_selection")
    require(selection in ("whole_game", "phase_stratified"), "bundle_selection must be whole_game or phase_stratified")
    require(len(window) == 2 and number(window[0], 1, integer=True)
            and ((selection == "whole_game" and window[1] is None)
                 or (selection == "phase_stratified" and number(window[1], window[0], integer=True))), "invalid bundle window")
    for key in ("power_rationale", "listener_margin_rationale", "absolute_quality_rationale",
                "listener_effect_rationale", "availability_rationale"):
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
    models = f.get("models", {})
    listeners = models.get("listeners", [])
    require(len(listeners) >= 3 and len(listeners) == len(set(listeners)), "three distinct listener handles required")
    identities = models.get("identities", {})
    cefr = f.get("cefr", {})
    require(isinstance(cefr, dict) and isinstance(cefr.get("enabled"), bool), "cefr.enabled must be declared")
    cefr_raters = cefr.get("raters", []) if cefr.get("enabled") else []
    if cefr.get("enabled"):
        require(selection == "whole_game", "auxiliary CEFR requires whole-game speaker samples")
        require(bool(cefr_raters) and len(cefr_raters) == len(set(cefr_raters)), "CEFR rater handles required")
        require(number(cefr.get("max_tokens"), 1, integer=True), "CEFR token budget required")
        require(bool(re.fullmatch(r"[a-f0-9]{64}", str(cefr.get("calibration_sha256", "")))), "CEFR calibration artifact hash required")
        require(cefr.get("examples_sha256") is None or bool(re.fullmatch(r"[a-f0-9]{64}", str(cefr.get("examples_sha256")))), "invalid CEFR examples hash")
        for key in ("min_calibration_accuracy", "min_calibration_valid_fraction"):
            require(number(cefr.get(key), .01, 1), f"CEFR {key} required")
        require(number(cefr.get("min_calibration_per_level"), 1, integer=True), "CEFR minimum calibration coverage per A/B/C band required")
        require(len(str(cefr.get("calibration_rationale", "")).strip()) >= 20, "CEFR calibration rationale required")
    for handle in [models.get("generator"), models.get("judge")] + listeners + cefr_raters:
        identity = identities.get(handle, {})
        for key in ("model", "family", "revision", "tokenizer_revision", "serving_version", "precision", "chat_template"):
            v = identity.get(key)
            require(isinstance(v, str) and bool(v.strip()) and v.lower() not in ("local", "main", "latest", "todo") and not v.startswith("<"), f"model metadata {handle}.{key} required")
        for key in ("revision", "tokenizer_revision"):
            require(bool(re.fullmatch(r"(?:sha256:)?(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})", str(identity.get(key, "")))),
                    f"{handle}.{key} must be a commit hash or artifact SHA-256")
    lf = [identities.get(h, {}).get("family") for h in listeners]
    gen_family = identities.get(models.get("generator"), {}).get("family")
    judge_family = identities.get(models.get("judge"), {}).get("family")
    require(gen_family != judge_family, "generator and semantic judge families must differ")
    require(all(identities.get(h, {}).get("family") != gen_family for h in cefr_raters), "CEFR rater families must differ from the generator")
    require(len(lf) == len(set(lf)) and gen_family not in lf, "listener families must differ from one another and generator")
    require(judge_family not in lf, "listener families must differ from the fidelity judge")
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
            require(b.get("selection") == selection, f"passage selection differs in {split}")
        cap = f.get("max_bundles_per_game")
        if isinstance(cap, int):
            require(all(sum(b.get("game_id") == g for b in bundles) <= cap for g in games), f"bundle cap exceeded in {split}")
    return errors


def cohort(frozen, split):
    if split not in ("set_a", "set_b"):
        raise ValueError("--split set_a or set_b is required for a frozen run")
    return frozen["splits"][split]


# A fixed context so the rewriter prompt hashes do not depend on any dataset.
CANONICAL_CONTEXT = RewriteContext(
    predicate="estimate(D1, P3, P7, werewolf)",
    original_nl=("P7 is a werewolf. He pushed the vote onto P2 without ever giving a read of his own."),
    phase="day_speech", speaker_id="P3", speaker_role_claimed="villager",
    public_history=["[s1|P1] claim_role(D1, P1, seer)"])

# Modules whose source defines a measurement or a check. Imported by dotted
# name so a package ``__init__`` that re-exports a function of the same name
# (``components.judges.structural_check``) cannot shadow the module.
HASHED_MODULES = ('components.judges.structural_check', 'components.judges.inverse_predicate_parser', 'components.judges.listener_probe', 'components.helpers', 'experiments.proficiency_validation.rewrite.runtime', 'experiments.proficiency_validation.validation.sources', 'experiments.proficiency_validation.validation.measurements', 'components.rewriters.language_rewriter', 'components.metrics.text_normalization', 'components.metrics.edit_distance', 'components.metrics.lexical', 'components.metrics.syntactic', 'components.metrics.grammar', 'experiments.proficiency_validation.rewrite.gate', 'experiments.proficiency_validation.rewrite.fidelity', 'experiments.proficiency_validation.validation.cefr', 'experiments.proficiency_validation.rewrite.__main__', 'experiments.proficiency_validation.validation.calibration', 'experiments.proficiency_validation.validation.perception', 'experiments.proficiency_validation.validation.analysis', 'experiments.proficiency_validation.validation.protocol', 'components.stats', 'experiments.proficiency_validation.validation.__main__')


def module_file_sha256(dotted: str) -> str:
    module = importlib.import_module(dotted)
    path = inspect.getsourcefile(module)
    if not path:
        raise RuntimeError(f"no source file for module {dotted}")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes(n_neutral_candidates: int = 5) -> dict:
    """Prompt TEMPLATE hashes (definition slot fixed, so one value per condition
    regardless of predicate), plus the probe, rubric, parser and module hashes."""
    from .perception import CONSTRUCTS, VERSIONS, prompt_sha
    rw = LanguageProficiencyRewriter(llm_client=None, condition="low")
    prompts = {condition: rw.template_sha256(condition) for condition in ("low", "neutral", "high")}
    prompts["neutral_candidates"] = rw.template_sha256("neutral", candidates=True, n=n_neutral_candidates)
    prompts["canonical_low_instance"] = rw.prompt_sha256(CANONICAL_CONTEXT, "low")
    from experiments.proficiency_validation.validation.cefr import VERSION as CEFR_VERSION, prompt_sha as cefr_prompt_sha
    return {
        "rewriter_prompt_version": PROMPT_VERSION,
        "rewriter_prompt_sha256": prompts,
        "n_neutral_candidates": n_neutral_candidates,
        "listener_probe_version": VERSIONS["range"],
        "listener_probe_sha256": prompt_sha("range"),
        "perception_prompt_sha256": {c: prompt_sha(c) for c in CONSTRUCTS},
        "perception_prompt_versions": VERSIONS,
        "semantic_rubric_version": SEMANTIC_RUBRIC_VERSION,
        "semantic_rubric_sha256": semantic_prompt_sha(),
        "cefr_rubric_version": CEFR_VERSION,
        "cefr_rubric_sha256": cefr_prompt_sha(),
        "parser_prompt_sha256": sha256_text(parser_system_prompt()),
        "module_source_sha256": {name: module_file_sha256(name) for name in HASHED_MODULES},
    }


def verify_hashes(frozen: dict) -> List[str]:
    """Names of every frozen artifact whose current hash differs. Empty = clean."""
    recorded = frozen.get("hashes", {}) or {}
    problems: List[str] = []

    def check(key: str, expected, current) -> None:
        if isinstance(current, dict):
            for sub, value in current.items():
                if (expected or {}).get(sub) != value:
                    problems.append(f"{key}.{sub}")
            for sub in (expected or {}):
                if sub not in current:
                    problems.append(f"{key}.{sub} (recorded but no longer computed)")
        elif expected != current:
            problems.append(key)

    current = frozen_hashes(int(frozen.get("n_neutral_candidates") or recorded.get("n_neutral_candidates") or 5))
    for key, value in current.items():
        check(key, recorded.get(key), value)
    return problems


def expected_template_hash(frozen: dict, condition: str) -> Optional[str]:
    """The template hash a rewrite row in ``condition`` must carry."""
    prompts = (frozen.get("hashes") or {}).get("rewriter_prompt_sha256") or {}
    return prompts.get("neutral_candidates" if condition == "neutral" else condition)


def enforce_frozen(path: Optional[Path] = None, *, check_models: bool = False) -> dict:
    """Load the frozen config and stop unless the code still matches it."""
    path = Path(path) if path else FROZEN_CONFIG
    if not path.exists():
        raise SystemExit(f"no frozen config at {path}")
    frozen = load_json(path)

    problems = validate_design(frozen) + verify_hashes(frozen)
    if check_models:
        from experiments.proficiency_validation.rewrite.runtime import model_identity
        for handle, expected in frozen.get("models", {}).get("identities", {}).items():
            if model_identity(handle) != expected:
                problems.append(f"model identity/settings changed: {handle}")
    if problems:
        raise SystemExit(f"code no longer matches {path}: {', '.join(problems)}. "
                         "Restore the frozen protocol. Protocol changes require a new version and fresh confirmatory data.")
    return frozen


def pin(args, frozen: dict, mapping: Dict[str, tuple], *, log=print) -> None:
    """Set ``args.<attr>`` from the frozen config for each ``attr -> key path``.

    Values already equal are left alone; differing values are overwritten and
    reported, so a command run under ``--frozen`` cannot silently deviate.
    """
    for attr, keys in mapping.items():
        node = frozen
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            raise ValueError(f"missing frozen parameter: {'.'.join(keys)}")
        if getattr(args, attr, None) != node:
            log(f"  pinned {attr} = {node!r} (from frozen config)", file=sys.stderr)
            setattr(args, attr, node)


def suggest_bounds(features, guardrails=None):
    from experiments.proficiency_validation.validation.analysis import paired_clusters
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
    config.update(schema_version=6, validation_scope="automated", frozen_at=now_iso(), git_commit=git_commit())
    config["splits"] = {key: read_jsonl(path) for key, path in (
        ("pilot", args.pilot_bundles), ("set_a", args.set_a_bundles), ("set_b", args.set_b_bundles))}
    config["sources_sha256"] = {key: sha256_file(path) for key, path in (
        ("pilot", args.pilot_sources), ("set_a", args.set_a_sources), ("set_b", args.set_b_sources))}
    config.setdefault("instruments", {})["subtlex_sha256"] = sha256_file(args.subtlex)
    config["instruments"]["subtlex_path"] = str(args.subtlex)
    models = config.setdefault("models", {})
    handles = [models.get("generator"), models.get("judge")] + models.get("listeners", [])
    if config.get("cefr", {}).get("enabled"):
        handles += config["cefr"].get("raters", [])
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
