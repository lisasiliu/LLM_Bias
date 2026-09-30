"""Proficiency validation analysis."""
from __future__ import annotations
from itertools import product
import random
from components.stats import mean
from experiments.proficiency_validation.rewrite.runtime import GUARDRAIL_METRICS
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
import sys
from components.helpers import read_jsonl
from components.metrics.edit_distance import normalized_token_edit_distance
from components.stats import benjamini_hochberg, bradley_terry_bootstrap, cluster_bootstrap, flatten, median, paired_cohens_dz, percentile, sd, tost_equivalence, wilson_interval, proportion_bootstrap
from experiments.proficiency_validation.rewrite.runtime import assert_unique
from experiments.proficiency_validation.rewrite.runtime import CONTRASTS, NUISANCE_METRICS, RANGE_METRICS, dump_json, fmt, load_frozen, now_iso
from experiments.proficiency_validation.rewrite.gate import generation_qc, failure_cause


def cluster_sign_flip_test(clusters, *, n_perm=5000, seed=101):
    """Enumerate all signs for up to 12 games; sample signs for larger cohorts."""
    effects = [mean([x for x in values if x is not None]) for values in clusters.values()]
    effects = [x for x in effects if x is not None]
    observed = mean(effects)
    n = len(effects)
    if n < 2:
        return {"p_value": None, "observed_mean": observed, "n_perm": 0,
                "method": "insufficient_games", "minimum_two_sided_p": None}
    threshold = abs(sum(effects))
    tolerance = 1e-12 * max(1., sum(abs(x) for x in effects))
    if n <= 12:
        totals = (sum(s * x for s, x in zip(signs, effects))
                  for signs in product((-1, 1), repeat=n))
        count = sum(abs(total) >= threshold - tolerance for total in totals)
        p, draws, method = count / 2**n, 2**n, "exact"
    else:
        if n_perm < 1:
            raise ValueError("n_perm must be positive")
        rng = random.Random(seed)
        count = sum(abs(sum(x if rng.random() < .5 else -x for x in effects))
                    >= threshold - tolerance for _ in range(n_perm))
        p, draws, method = (count + 1) / (n_perm + 1), n_perm, "monte_carlo"
    return {"p_value": p, "observed_mean": observed, "n_perm": draws,
            "method": method, "minimum_two_sided_p": 2.**(1 - n)}


def criteria(report, frozen):
    if not frozen:
        return []
    out = []
    def add(name, passed, detail=None, note=""):
        out.append({"criterion": name, "passed": passed, "detail": detail, "note": note})
    def enough(row):
        games, bundles = frozen.get("min_games"), frozen.get("min_bundles")
        if games is None or bundles is None or row is None:
            return False
        expected = report.get("expected_bundles", bundles)
        return (row.get("n_games", 0) >= games and row.get("n_bundles", 0) >= bundles
                and row.get("n_bundles", 0) / max(1, expected) >= frozen.get("min_valid_fraction", 1))
    primary = frozen.get("primary_range_metrics", [])
    treatments = [c for c in ("low", "high") if c in frozen.get("conditions", ["low"])]
    listeners = list(frozen.get("models", {}).get("listeners", [])) + ["all_models"]
    for treatment in treatments:
        qc = report.get("generation_qc", {})
        availability = [(qc.get(c) or {}).get("final_success_rate") for c in (treatment, "neutral")]
        minimum = frozen.get("min_treatment_availability")
        max_gap = frozen.get("max_differential_attrition")
        available = None
        if all(v is not None for v in availability) and minimum is not None and max_gap is not None:
            available = min(availability) >= minimum and abs(availability[0] - availability[1]) <= max_gap
        add(f"{treatment.title()} and Neutral treatment availability", available,
            {"rates": availability, "minimum": minimum, "max_differential_attrition": max_gap})
        contrast = f"{treatment}_vs_neutral"
        thresholds = frozen.get("range_effect_thresholds", {}).get(contrast, {})
        for metric in primary:
            m = report.get("range", {}).get(contrast, {}).get("metrics", {}).get(metric)
            threshold = thresholds.get(metric)
            name = f"{treatment.title()} vs Neutral: {metric}, direction/FDR/effect/sample size"
            if not m or threshold is None or m.get("reject_fdr_05") is None or m.get("cohens_dz") is None:
                add(name, None, m, "metric, threshold or inferential result missing")
            else:
                add(name, bool(enough(m) and m.get("direction_confirmed") and m["reject_fdr_05"]
                               and abs(m["cohens_dz"]) >= threshold), m)
        for metric in GUARDRAIL_METRICS:
            m = report.get("guardrails", {}).get(contrast, {}).get("metrics", {}).get(metric)
            eq = (m or {}).get("equivalence", {}).get("equivalent")
            add(f"{treatment.title()} vs Neutral: {metric} equivalent",
                None if eq is None else bool(enough(m) and eq), m)
        matching = report.get("neutral_matching_by_condition", {}).get(treatment)
        if treatment == "low" and matching is None:
            matching = report.get("neutral_matching")
        fraction = (matching or {}).get("fraction_within_tolerance")
        minimum = frozen.get("neutral_match_min_fraction")
        add(f"Neutral matching to {treatment}", None if fraction is None or minimum is None else fraction >= minimum, matching)
        block = report.get("listener", {}).get(f"{treatment}:neutral", {}).get("listeners", {})
        for listener in listeners:
            row = block.get(listener)
            ci = (row or {}).get("cluster_bootstrap_ci", [None, None])
            separation = frozen.get("listener_min_separation")
            direction = (lambda x: x[0] is not None and x[0] > .5 + separation) if treatment == "low" else (
                         lambda x: x[1] is not None and x[1] < .5 - separation)
            passed = None
            if row and separation is not None and all(x is not None for x in ci):
                passed = enough(row) and direction(ci)
                for order in ("ab", "ba"):
                    o = row.get("by_order", {}).get(order)
                    passed = passed and enough(o) and direction((o or {}).get("ci95", [None, None]))
            add(f"Listener {listener}: {treatment} vs Neutral, both orders", passed, row)
        for listener in sorted(set(block) - set(listeners)):
            add(f"Unexpected listener {listener}", False)
    for metric in primary:
        m = report.get("range", {}).get("neutral_vs_original", {}).get("metrics", {}).get(metric)
        eq = (m or {}).get("equivalence", {}).get("equivalent")
        add(f"Neutral vs Original: {metric} equivalent", None if eq is None else bool(enough(m) and eq), m)
    margin = frozen.get("listener_equivalence_margin")
    block = report.get("listener", {}).get("neutral:original", {}).get("listeners", {})
    for listener in listeners:
        row = block.get(listener)
        ci = (row or {}).get("equivalence_ci90", [None, None])
        passed = None
        if row and margin is not None and all(x is not None for x in ci):
            within = lambda interval: all(x is not None for x in interval) and interval[0] > .5-margin and interval[1] < .5+margin
            passed = enough(row) and within(ci)
            for order in ("ab", "ba"):
                o = row.get("by_order", {}).get(order)
                passed = passed and enough(o) and within((o or {}).get("ci90", [None, None]))
        add(f"Listener {listener}: Neutral vs Original equivalence, both orders", passed, row)
    for c in frozen.get("conditions", []):
        row = report.get("absolute_quality", {}).get(c)
        limits = frozen.get("absolute_quality", {})
        passed = None
        if row and row.get("grammar_mean") is not None and row.get("cola_mean") is not None and limits:
            passed = bool(enough(row) and row["grammar_mean"] <= limits["grammar_errors_max"]
                          and row["cola_mean"] >= limits["cola_mean_min"])
        add(f"Observed absolute grammar/acceptability quality: {c}", passed, row)
    for row in report.get("integrity", []):
        add("Integrity: " + row["check"], row["passed"], row.get("detail"), row.get("note", ""))
    if not report.get("integrity"):
        add("Frozen data provenance and cohort checks", None)
    if "perceived_proficiency" in frozen.get("perception_constructs", []):
        # Apply the same prespecified listener decision rule to a separate
        # construct. Range judgments cannot fill missing proficiency judgments.
        perception_report = {**report, "listener": report.get("perceived_proficiency", {})}
        range_only = {**frozen, "perception_constructs": ["range"]}
        for item in criteria(perception_report, range_only):
            if item["criterion"].startswith(("Listener ", "Unexpected listener ")):
                out.append({**item, "criterion": "Perceived proficiency: " + item["criterion"]})
    return out


def expected_direction(metric: str, treatment: str, comparator: str) -> Optional[int]:
    """+1/-1 expected sign of (treatment - comparator); 0 = expected equivalence."""
    sign = RANGE_METRICS.get(metric)
    if sign is None:
        return None
    if (treatment, comparator) == ("neutral", "original"):
        return 0
    if treatment == "low":
        return sign
    if treatment == "high":
        return -sign
    return None


def paired_clusters(rows: Sequence[dict], metric: str, treatment: str, comparator: str
                    ) -> Tuple[Dict[str, List[float]], int]:
    assert_unique(rows, ("bundle_id", "condition"))
    by_bc = {(r["bundle_id"], r["condition"]): r for r in rows if r.get("complete", True)}
    ids = sorted({b for (b, c) in by_bc if c == treatment} & {b for (b, c) in by_bc if c == comparator})
    clusters: Dict[str, List[float]] = {}
    n = 0
    for b in ids:
        t, c = by_bc[(b, treatment)].get(metric), by_bc[(b, comparator)].get(metric)
        if t is None or c is None:
            continue
        clusters.setdefault(str(by_bc[(b, treatment)].get("game_id")), []).append(t - c)
        n += 1
    return clusters, n


def contrast_analysis(rows: Sequence[dict], metrics: Sequence[str], *, frozen: Optional[dict],
                      n_boot: int, seed: int, family: str,
                      fdr_metrics: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """Paired contrasts per metric. BH runs over ``fdr_metrics`` only (default: every
    metric that is not a nuisance diagnostic), so word count or sentence count
    cannot change the corrected p-values of the confirmatory range metrics."""
    bounds = (frozen or {}).get("equivalence_bounds", {}) or {}
    fdr_set = set(fdr_metrics) if fdr_metrics is not None else {m for m in metrics if m not in NUISANCE_METRICS}
    out: Dict[str, dict] = {}
    for treatment, comparator, role in CONTRASTS:
        per_metric: Dict[str, dict] = {}
        pvals: Dict[str, Optional[float]] = {}
        for metric in metrics:
            clusters, n = paired_clusters(rows, metric, treatment, comparator)
            if n == 0:
                continue
            diffs = flatten(clusters)
            game_effects = {g: [mean(values)] for g, values in clusters.items()}
            boot = cluster_bootstrap(game_effects, n_boot=n_boot, seed=seed)
            test = cluster_sign_flip_test(game_effects, seed=seed,
                    n_perm=(frozen or {}).get("statistics", {}).get("n_perm", 5000))
            equivalence_from_same_draws = dict(boot)          # reuse, then drop the raw draws
            boot.pop("draws", None)
            expected = expected_direction(metric, treatment, comparator)
            estimate = boot["estimate"]
            excludes_zero = (boot["ci_low"] is not None and
                             (boot["ci_low"] > 0 or boot["ci_high"] < 0))
            entry = {"n_bundles": n, "n_games": boot["n_clusters"], "estimate": estimate,
                     "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                     "sd_diff": sd([v[0] for v in game_effects.values()]),
                     "cohens_dz": paired_cohens_dz([v[0] for v in game_effects.values()]),
                     "bundle_cohens_dz_descriptive": paired_cohens_dz(diffs),
                     "estimand": "equal_game_weight",
                     "p_sign_flip": test["p_value"], "expected_sign": expected,
                     "sign_flip_method": test["method"], "sign_flip_patterns": test["n_perm"],
                     "minimum_two_sided_p": test["minimum_two_sided_p"],
                     "direction_confirmed": None}
            if expected in (1, -1):
                entry["direction_confirmed"] = bool(estimate is not None and excludes_zero
                                                    and (estimate > 0) == (expected > 0))
            bound = bounds.get(metric)
            if expected == 0 or metric in GUARDRAIL_METRICS:
                entry["equivalence"] = (tost_equivalence(game_effects, bound, bootstrap=equivalence_from_same_draws)
                                        if bound is not None else {"bound": None, "equivalent": None})
            per_metric[metric] = entry
            if metric in fdr_set:
                pvals[metric] = test["p_value"]
            else:
                entry["q_bh"] = None
                entry["reject_fdr_05"] = None
                entry["fdr_family"] = ("nuisance (not corrected)" if metric in NUISANCE_METRICS
                                       else "secondary (outside the corrected family)")
        if not per_metric:
            continue
        for metric, adj in benjamini_hochberg(pvals).items():
            per_metric[metric]["q_bh"] = adj["q"]
            per_metric[metric]["reject_fdr_05"] = adj["reject"]
            per_metric[metric]["fdr_family"] = f"{family} (m={len(pvals)})"
        out[f"{treatment}_vs_{comparator}"] = {
            "treatment": treatment, "comparator": comparator, "role": role, "family": family,
            "metrics": per_metric}
    if frozen:
        # Each frozen study contains one treatment and its matched Neutral control.
        pvals = {(contrast, m): values["p_sign_flip"] for contrast, block in out.items()
                 if contrast in ("low_vs_neutral", "high_vs_neutral")
                 for m, values in block["metrics"].items() if m in fdr_set}
        for (contrast, m), adj in benjamini_hochberg(pvals).items():
            out[contrast]["metrics"][m].update(q_bh=adj["q"], reject_fdr_05=adj["reject"],
                                             fdr_family=f"{family}: declared treatment contrast")
    return out


def neutral_matching(rewrites: Sequence[dict], *, tolerance: Optional[float], treatment="low") -> dict:
    by_source: Dict[str, Dict[str, float]] = {}
    for r in rewrites:
        if r.get("status") == "ok" and r.get("edit_distance_to_original") is not None:
            by_source.setdefault(r["source_id"], {})[r["condition"]] = r["edit_distance_to_original"]
    low = [d[treatment] for d in by_source.values() if treatment in d]
    neutral = [d["neutral"] for d in by_source.values() if "neutral" in d]
    gaps = [abs(d[treatment] - d["neutral"]) for d in by_source.values() if treatment in d and "neutral" in d]
    within = (sum(1 for g in gaps if g <= tolerance + 1e-12) / len(gaps)) if gaps and tolerance is not None else None
    paired = [d for d in by_source.values() if treatment in d and "neutral" in d]
    attempted = {r["source_id"] for r in rewrites if r["condition"] == "neutral" and r.get("attempts")}
    excluded = [d[treatment] for sid, d in by_source.items()
                if treatment in d and "neutral" not in d and sid in attempted]
    return {"n_sources": len(by_source), "treatment_condition": treatment,
            "treatment": {"mean": mean(low), "sd": sd(low), "median": median(low), "n": len(low)},
            "neutral": {"mean": mean(neutral), "sd": sd(neutral), "median": median(neutral), "n": len(neutral)},
            "abs_gap": {"mean": mean(gaps), "median": median(gaps), "p90": percentile(gaps, 90), "n": len(gaps)},
            "tolerance": tolerance, "fraction_within_tolerance": within,
            "paired_means": {"n": len(paired), "treatment": mean([d[treatment] for d in paired]),
                             "neutral": mean([d["neutral"] for d in paired])},
            "failed_neutral_attempts": {"n": len(excluded), "mean_treatment_distance": mean(excluded)},
            "search": neutral_search_diagnostics(rewrites)}


def neutral_search_diagnostics(rewrites: Sequence[dict]) -> dict:
    """Audit recorded search behavior without changing any archived verdict."""
    counts = Counter()
    for row in rewrites:
        if row.get("condition") != "neutral" or not row.get("attempts"):
            continue
        counts["attempted_sources"] += 1
        original = row.get("original_text", "")
        target = row.get("target_edit_distance")
        tolerance = (row.get("gate_provenance") or {}).get("neutral_edit_tolerance")
        seen, passing_distances = set(), []
        for attempt in row["attempts"]:
            counts["attempts"] += 1
            selection = attempt.get("neutral_selection") or {}
            for candidate in selection.get("candidate_gates", []):
                text = candidate.get("text")
                if not isinstance(text, str):
                    continue
                counts["candidate_occurrences"] += 1
                counts["repeated_candidate_occurrences"] += int(text in seen)
                counts["reused_verdicts"] += int(candidate.get("reused_verdict", False))
                seen.add(text)
                if candidate.get("ok") is True:
                    passing_distances.append(normalized_token_edit_distance(text, original))
        counts["unique_candidates"] += len(seen)
        if row.get("status") == "ok" or target is None or tolerance is None:
            continue
        if not passing_distances:
            counts["failed_without_fidelity_passer"] += 1
            continue
        best = min(passing_distances, key=lambda d: abs(d - target))
        if abs(best - target) <= tolerance + 1e-12:
            counts["failed_with_eligible_recorded_candidate"] += 1
        else:
            counts["failed_with_only_out_of_band_passers"] += 1
            counts["best_passer_below_band" if best < target else "best_passer_above_band"] += 1
        final = row["attempts"][-1].get("text")
        if final and abs(normalized_token_edit_distance(final, original) - target) > abs(best - target) + 1e-12:
            counts["failed_final_selection_worse_than_best_passer"] += 1
    return {key: counts[key] for key in (
        "attempted_sources", "attempts", "candidate_occurrences", "unique_candidates",
        "repeated_candidate_occurrences", "reused_verdicts", "failed_without_fidelity_passer",
        "failed_with_eligible_recorded_candidate", "failed_with_only_out_of_band_passers",
        "best_passer_below_band", "best_passer_above_band", "failed_final_selection_worse_than_best_passer")}


def neutral_search_lines(nm: dict) -> List[str]:
    """Shared descriptive audit for the source report and analysis report."""
    excluded = nm.get("failed_neutral_attempts", {})
    search = nm.get("search", {})
    if not search:
        return []
    return ["", f"For {excluded.get('n', 0)} accepted treatment anchors with failed Neutral attempts, "
            f"mean treatment distance is {fmt(excluded.get('mean_treatment_distance'))}. "
            "This excluded cohort is reported alongside successful pairs to assess selection by edit intensity.", "",
            f"Across {search['attempts']} Neutral attempts, {search['repeated_candidate_occurrences']} of "
            f"{search['candidate_occurrences']} recorded candidate occurrences repeat an earlier candidate "
            f"for the same source. {search['reused_verdicts']} verdicts are explicitly marked as reused.", "",
            f"Among failed sources, {search['failed_with_only_out_of_band_passers']} have fidelity passers "
            f"but none within the distance band. The closest passer is below the band in "
            f"{search['best_passer_below_band']} and above it in {search['best_passer_above_band']}. "
            f"{search['failed_with_eligible_recorded_candidate']} have a recorded passer within the band. "
            f"The final selection is farther from the target than an earlier fidelity passer in "
            f"{search['failed_final_selection_worse_than_best_passer']} failed sources.", ""]


def listener_analysis(choices: Sequence[dict], *, n_boot: int, seed: int) -> Dict[str, dict]:
    """Treatment-choice proportions from complete counterbalanced model comparisons."""
    constructs = {r.get("construct", "range") for r in choices}
    if len(constructs) > 1:
        raise ValueError("analyze perception constructs separately")
    assert_unique(choices, ("bundle_id", "pair", "listener", "order"))
    out: Dict[str, dict] = {}
    for pair in sorted({r["pair"] for r in choices}):
        pair_rows = [r for r in choices if r["pair"] == pair]
        valid = [r for r in pair_rows if r.get("ok")
                 and r.get("chosen_condition") in (r["treatment"], r["comparator"])]
        valid_orders = {}
        for r in valid:
            valid_orders.setdefault((r["bundle_id"], r["listener"]), set()).add(r["order"])
        rows = [r for r in valid
                if valid_orders[(r["bundle_id"], r["listener"])] == {"ab", "ba"}]
        errors = len(pair_rows) - len(valid)
        incomplete = len(valid) - len(rows)   # valid, but the counterbalanced partner is missing
        if not rows:
            continue
        treatment, comparator = rows[0]["treatment"], rows[0]["comparator"]
        listeners = sorted({r["listener"] for r in rows})

        per_listener: Dict[str, dict] = {}
        for listener in listeners + ["all_models"]:
            subset = rows if listener == "all_models" else [r for r in rows if r["listener"] == listener]
            by_bundle: Dict[str, List[int]] = {}
            game_of: Dict[str, str] = {}
            for r in subset:
                by_bundle.setdefault(r["bundle_id"], []).append(1 if r["chose_treatment"] else 0)
                game_of[r["bundle_id"]] = str(r.get("game_id"))
            clusters: Dict[str, List[float]] = {}
            for b, votes in by_bundle.items():
                clusters.setdefault(game_of[b], []).append(sum(votes) / len(votes))
            clusters = {g: [mean(values)] for g, values in clusters.items()}
            boot = proportion_bootstrap(clusters, n_boot=n_boot, seed=seed)
            eq_boot = proportion_bootstrap(clusters, n_boot=n_boot, seed=seed, alpha=.10)
            boot.pop("draws", None)
            k = sum(1 for r in subset if r["chose_treatment"])
            p, w_lo, w_hi = wilson_interval(k, len(subset))
            by_order = {}
            for order in ("ab", "ba"):
                o = [r for r in subset if r["order"] == order]
                if o:
                    by_order_bundle = {}
                    for r in o:
                        by_order_bundle.setdefault((str(r.get("game_id")), r["bundle_id"]), []).append(int(r["chose_treatment"]))
                    order_clusters = {}
                    for (g, b), vals in by_order_bundle.items():
                        order_clusters.setdefault(g, []).append(mean(vals))
                    order_clusters = {g: [mean(values)] for g, values in order_clusters.items()}
                    order_ci = proportion_bootstrap(order_clusters, n_boot=n_boot, seed=seed)
                    order_eq = proportion_bootstrap(order_clusters, n_boot=n_boot, seed=seed, alpha=.10)
                    by_order[order] = {"n": len(o),
                                       "share_treatment": order_ci["estimate"], "n_games": order_ci["n_clusters"],
                                       "n_bundles": len(by_order_bundle),
                                       "ci95": [order_ci["ci_low"], order_ci["ci_high"]],
                                       "ci90": [order_eq["ci_low"], order_eq["ci_high"]]}
            comps_by_cluster: Dict[str, List[Tuple[str, str]]] = {}
            for r in subset:
                loser = comparator if r["chosen_condition"] == treatment else treatment
                comps_by_cluster.setdefault(str(r.get("game_id")), []).append((r["chosen_condition"], loser))
            bt = bradley_terry_bootstrap(comps_by_cluster, treatment, comparator, n_boot=n_boot, seed=seed)
            ci_low, ci_high = boot["ci_low"], boot["ci_high"]
            paired_orders = {}
            for r in subset:
                paired_orders.setdefault((r["bundle_id"], r["listener"]), {})[r["order"]] = r
            consistent = sum(p["ab"]["chosen_condition"] == p["ba"]["chosen_condition"]
                             for p in paired_orders.values())
            def position(r):
                # Old rows can reconstruct position from the counterbalancing map.
                return "A" if bool(r["chose_treatment"]) == (r["order"] == "ab") else "B"
            always_a = sum(all(position(r) == "A" for r in p.values()) for p in paired_orders.values())
            always_b = sum(all(position(r) == "B" for r in p.values()) for p in paired_orders.values())
            per_listener[listener] = {
                "n_responses": len(subset),
                "estimand": "equal_game_weight",
                "n_bundles": len(by_bundle), "n_games": boot["n_clusters"],
                "share_treatment_more_limited": boot["estimate"],
                "share_treatment_raw_votes": p,
                "wilson_ci": [w_lo, w_hi],
                "cluster_bootstrap_ci": [ci_low, ci_high],
                "equivalence_ci90": [eq_boot["ci_low"], eq_boot["ci_high"]],
                "interval_method": boot["interval_method"],
                "ci_lower_above_half": (ci_low is not None and ci_low > 0.5),
                "ci_covers_half": (ci_low is not None and ci_high is not None and ci_low <= 0.5 <= ci_high),
                "by_order": by_order, "bradley_terry": bt,
                "order_consistency": {"n_comparisons": len(paired_orders), "same_condition": consistent,
                    "reversed_condition": len(paired_orders) - consistent,
                    "fraction_same": consistent / len(paired_orders), "always_a": always_a, "always_b": always_b},
                "position_b_raw_share": sum(position(r) == "B" for r in subset) / len(subset),
                "all_pairs_choose_same_position": always_a == len(paired_orders) or always_b == len(paired_orders),
                "confidence": {c: sum(1 for r in subset if r.get("confidence") == c)
                               for c in ("high", "medium", "low")},
            }
        out[pair] = {"construct": next(iter(constructs), "range"),
                     "treatment": treatment, "comparator": comparator, "n_invalid_responses": errors,
                     "n_dropped_incomplete_counterbalance": incomplete, "listeners": per_listener}
    return out


def passage_retention(features):
    """Describe Original-defined eligible passages and complete treatment pairs."""
    originals = {r["bundle_id"]: r for r in features if r["condition"] == "original" and r.get("complete", True)}
    complete = {(r["bundle_id"], r["condition"]) for r in features if r.get("complete", True)}
    out = {}
    for treatment in ("low", "high"):
        if not any(r["condition"] == treatment for r in features):
            continue
        retained = {b for b in originals if (b, treatment) in complete and (b, "neutral") in complete}
        groups = {}
        for name, ids in (("retained", retained), ("excluded", set(originals) - retained)):
            groups[name] = {"n": len(ids),
                "mean_original_words": mean([originals[b].get("word_count") for b in ids]),
                "mean_original_utterances": mean([originals[b].get("n_utterances") for b in ids]),
                "games": dict(sorted(Counter(originals[b]["game_id"] for b in ids).items()))}
        out[treatment] = {"eligible": len(originals), "matched": len(retained),
            "retained_fraction": len(retained) / len(originals) if originals else None,
            "matched_bundle_ids": sorted(retained), **groups}
    return out


def _criterion(name: str, passed: Optional[bool], detail=None, note: str = "") -> dict:
    """``passed`` is True, False, or None = could not be evaluated (never a pass)."""
    return {"criterion": name, "passed": passed, "detail": detail, "note": note}


def frozen_criteria(report: dict, frozen: Optional[dict]) -> List[dict]:

    return criteria(report, frozen)


def verdict(criteria: List[dict]) -> dict:
    passed = sum(1 for c in criteria if c["passed"] is True)
    failed = sum(1 for c in criteria if c["passed"] is False)
    unevaluated = sum(1 for c in criteria if c["passed"] is None)
    return {"n_criteria": len(criteria), "passed": passed, "failed": failed,
            "not_evaluated": unevaluated,
            "scope": "automated linguistic measures and model perception",
            "confirmed": bool(criteria) and failed == 0 and unevaluated == 0}


def integrity_checks(frozen: dict, *, rewrites=None, features_path=None,
                     listener_paths=None, guardrails_path=None, rewrites_path=None, split=None) -> List[dict]:
    from experiments.proficiency_validation.validation.protocol import verify_hashes, expected_template_hash
    from experiments.proficiency_validation.validation.protocol import validate_design, cohort, sha256_file
    from experiments.proficiency_validation.rewrite.runtime import verify_artifact, assert_unique
    checks = []
    def check(name, passed, detail=None):
        checks.append({"check": name, "passed": passed, "detail": detail})
    problems = validate_design(frozen) + verify_hashes(frozen)
    check("complete frozen design and unchanged code", not problems, problems)
    if split not in ("set_a", "set_b"):
        check("explicit confirmatory split", False)
        return checks
    expected = {b["bundle_id"]: b["game_id"] for b in cohort(frozen, split)}
    conditions = frozen["conditions"]
    sources = {sid for b in cohort(frozen, split) for sid in b["source_ids"]}
    expected_rewrites = {(s, c) for s in sources for c in conditions if c != "original"}
    if rewrites is None or rewrites_path is None:
        check("rewrite data and provenance supplied", None)
    else:
        assert_unique(rewrites, ("source_id", "condition"))
        check("all preselected source/condition outcomes recorded",
              {(r["source_id"], r["condition"]) for r in rewrites} == expected_rewrites)
        prov, errors = verify_artifact(rewrites_path, frozen, split)
        check("rewrites artifact provenance", not errors, errors)
        check("rewrites use the frozen source text artifact", bool(prov) and
              prov.get("input_sha256", {}).get("sources") == frozen.get("sources_sha256", {}).get(split))
        models = frozen["models"]
        bad = []
        for r in rewrites:
            gate = r.get("gate_provenance") or {}
            valid = (r.get("prompt_template_sha256") == expected_template_hash(frozen, r["condition"])
                     and r.get("generator_temperature") == frozen["generation_temperature"]
                     and (r.get("generator_provenance") or {}).get("identity") == models["identities"][models["generator"]]
                     and (gate.get("judge_model") or {}).get("identity") == models["identities"][models["judge"]]
                     and gate.get("semantic_rubric_sha256") == frozen["hashes"]["semantic_rubric_sha256"])
            if not valid:
                bad.append(r["source_id"])
        check("every rewrite uses frozen generator, English gate and prompts", not bad, bad)
        by_source = {(r["source_id"], r["condition"]): r for r in rewrites}
        anchor_condition = frozen["neutral_anchor_condition"]
        bad_anchors = []
        for r in rewrites:
            if r["condition"] != "neutral" or r.get("status") != "ok":
                continue
            anchor = r.get("anchor_outcomes", {}).get(anchor_condition, {})
            actual = by_source.get((r["source_id"], anchor_condition), {})
            if (set(r.get("anchor_outcomes", {})) != {anchor_condition} or actual.get("status") != "ok"
                    or anchor.get("text") != actual.get("rewritten_text")
                    or actual.get("edit_distance_to_original") is None
                    or r.get("target_edit_distance") is None
                    or abs(r["target_edit_distance"] - actual["edit_distance_to_original"]) > 1e-12):
                bad_anchors.append(r["source_id"])
        check("Neutral targets the exact accepted treatment rewrite", not bad_anchors, bad_anchors)
    feature_prov, feature_rows = None, []
    for kind, path in (("features", features_path), ("guardrails", guardrails_path)):
        if path is None:
            check(f"{kind} artifact supplied", None)
            continue
        rows = read_jsonl(path)
        assert_unique(rows, ("bundle_id", "condition"))
        check(f"{kind} uses the entire preselected cohort with recorded missing values",
              {(r["bundle_id"], r["condition"]) for r in rows} == {(b, c) for b in expected for c in conditions}
              and all(expected.get(r["bundle_id"]) == r.get("game_id") for r in rows))
        prov, errors = verify_artifact(path, frozen, split)
        check(f"{kind} artifact provenance", not errors, errors)
        if prov is None:
            continue
        inst = frozen["instruments"]
        if kind == "features":
            feature_prov = prov
            feature_rows = rows
            spacy = prov.get("spacy") or {}
            subtlex = prov.get("subtlex") or {}
            check("feature instruments and lexical definition match freeze",
                  prov.get("mattr_window") == frozen["mattr_window"]
                  and prov.get("include_domain_terms") == inst["include_domain_terms"]
                  and subtlex.get("sha256") == inst["subtlex_sha256"]
                  and subtlex.get("measure") == inst["subtlex_measure"]
                  and spacy.get("model_version") == inst["spacy_model_version"]
                  and spacy.get("spacy_version") == inst["spacy_version"]
                  and f"{spacy.get('lang')}_{spacy.get('model')}" == inst["spacy_model"])
            check("features were computed from the analyzed rewrites",
                  bool(rewrites_path) and prov.get("input_sha256", {}).get("rewrites") == sha256_file(rewrites_path))
            complete = [r for r in rows if r.get("complete")]
            check("complete feature rows have syntax and adequate SUBTLEX coverage",
                  bool(complete) and all(r.get("syntax_available") and r.get("lemmas_available")
                      and r.get("subtlex_coverage") is not None
                      and r["subtlex_coverage"] >= frozen["min_subtlex_coverage"] for r in complete))
        else:
            lt, cola = prov.get("languagetool") or {}, prov.get("cola") or {}
            check("grammar instruments and acceptable label match freeze",
                  lt.get("version") == inst["languagetool_version"]
                  and lt.get("language") == inst["languagetool_language"]
                  and cola.get("model") == inst["cola_model"]
                  and cola.get("commit_hash") == inst["cola_commit_hash"]
                  and cola.get("acceptable_label_index") == inst["cola_acceptable_index"]
                  and cola.get("max_length") == inst["cola_max_length"]
                  and prov.get("spacy") == (feature_prov or {}).get("spacy"))
            check("guardrails measured the same assembled passages",
                  bool(feature_prov) and prov.get("input_sha256", {}).get("assembled") == feature_prov.get("assembled_sha256"))
    for path in listener_paths or []:
        rows = read_jsonl(path)
        prov, errors = verify_artifact(path, frozen, split)
        check(f"{Path(path).name}: probe provenance", not errors, errors)
        if prov is None:
            continue
        used = {r.get("listener") for r in rows}
        construct = prov.get("construct", "range")
        check(f"{Path(path).name}: listener identities and prompts match freeze",
              used <= set(frozen["models"]["listeners"])
              and construct in frozen["perception_constructs"]
              and all(r.get("construct", "range") == construct for r in rows)
              and prov.get("prompt_sha256") == frozen["hashes"]["perception_prompt_sha256"].get(construct)
              and all(r.get("prompt_version") == frozen["hashes"]["perception_prompt_versions"].get(construct) for r in rows)
              and all((prov.get("listeners", {}).get(h) or {}).get("identity") == frozen["models"]["identities"][h] for h in used))
        check(f"{Path(path).name}: same passages/cohort", bool(feature_prov)
              and prov.get("input_sha256", {}).get("assembled") == feature_prov.get("assembled_sha256")
              and all(expected.get(r["bundle_id"]) == r.get("game_id") for r in rows))
    model_rows = [r for p in listener_paths or [] for r in read_jsonl(p)]
    complete = {(r["bundle_id"], r["condition"]) for r in feature_rows if r.get("complete")}
    required = {(b, f"{a}:{c}", h, o, construct) for b in expected
                for a, c in [(t, "neutral") for t in ("low", "high") if t in conditions] + [("neutral", "original")]
                if (b, a) in complete and (b, c) in complete
                for h in frozen["models"]["listeners"] for o in ("ab", "ba")
                for construct in frozen["perception_constructs"]}
    actual = {(r["bundle_id"], r["pair"], r["listener"], r["order"], r.get("construct", "range")) for r in model_rows}
    check("every eligible model probe has both presentation outcomes recorded",
          bool(required) and actual == required and len(actual) == len(model_rows))
    return checks


# ── rendering ──

def _contrast_table(block: dict, metrics: Sequence[str]) -> List[str]:
    lines = [f"### {block['treatment']} − {block['comparator']}  ({block['role']}; "
             f"n bundles = {next(iter(block['metrics'].values()))['n_bundles']}, "
             f"games = {next(iter(block['metrics'].values()))['n_games']})", "",
             "| metric | expected | Δ game mean | 95% CI (game bootstrap) | game d_z | p (sign-flip) | q (BH) | CI direction only / equivalence |",
             "| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |"]
    for metric in metrics:
        m = block["metrics"].get(metric)
        if not m:
            continue
        exp = {1: "↑", -1: "↓", 0: "="}.get(m["expected_sign"], "")
        conf = m.get("direction_confirmed")
        if m.get("expected_sign") == 0 and m.get("equivalence"):
            conf = m["equivalence"].get("equivalent")
        lines.append(f"| {metric} | {exp} | {fmt(m['estimate'])} | [{fmt(m['ci_low'])}, {fmt(m['ci_high'])}] | "
                     f"{fmt(m['cohens_dz'], 2)} | {fmt(m['p_sign_flip'])} | {fmt(m.get('q_bh'))} | "
                     f"{fmt(conf)} |")
    first = next(iter(block["metrics"].values()))
    if first.get("sign_flip_method") == "exact":
        lines += ["", f"The sign-flip test enumerates all {first['sign_flip_patterns']} sign patterns. "
                  f"The smallest attainable two-sided p-value is {fmt(first['minimum_two_sided_p'], 4)}. "
                  "A directional bootstrap interval alone does not establish significance."]
    return lines + [""]


def render(report: dict) -> str:
    lines = ["# Proficiency validation report", "",
             f"Generated {report['created_at']}. Frozen config: "
             f"{'yes' if report.get('frozen_config_used') else 'no (pilot / exploratory)'}.", ""]
    lines += ["The primary perception summary is the probability that a listener selects the treatment "
              "as more limited. Speakers receive equal weight within each game and games receive equal weight. "
              "This probability measures model perception of the displayed texts. It is not a CEFR level.", ""]
    if not report.get("frozen_config_used"):
        lines += ["**NOT CONFIRMED.** These results are exploratory. Directional intervals do not establish "
                  "validation without the planned effect size, multiplicity and quality criteria.", ""]
    if report.get("recorded_policy"):
        lines += [f"Recorded rewrite prompts: {report['recorded_policy']['rewrite_versions']}. "
                  f"Recorded fidelity rubrics: {report['recorded_policy']['fidelity_versions']}. "
                  "Acceptance describes those recorded checks and is not an estimate of semantic accuracy.", ""]
        if report["recorded_policy"].get("requires_revalidation"):
            lines += ["These model outputs have not been evaluated under the current generation and fidelity policy.", ""]
    if report.get("generation_record_errors"):
        lines += ["## Generation record inconsistencies", "",
                  "Archived settings disagree with the outputs. These records require the actual run metadata before reproducibility can be established.", ""]
        lines += [f"- {error}" for error in report["generation_record_errors"]] + [""]
    qc = report.get("generation_qc")
    if qc:
        lines += ["## Generation QC", "",
                  "| condition | n | accepted | anchor blocked | first-pass | retried | treatment unavailable | infrastructure unavailable | fallback | mean attempts |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for c, q in qc.items():
            lines.append(f"| {c} | {q['n']} | {q['final_success']} | {q['anchor_blocked']} | {q['first_pass_success']} | {q['needed_retry']} | "
                         f"{q['treatment_unavailable']} | {q.get('infrastructure_unavailable', 0)} | "
                         f"{q['fallback_to_original']} | {fmt(q['mean_attempts'], 2)} |")
        lines.append("")
    if report.get("failure_causes"):
        lines += ["| Condition | Recorded outcome cause | Sources |", "| --- | --- | ---: |"]
        for c, counts in report["failure_causes"].items():
            lines += [f"| {c} | {cause} | {n} |" for cause, n in counts.items() if cause != "accepted"]
        lines += ["", "Evaluation-output failures are separate from transport failures. Legacy infrastructure status labels are retained in archived data.", ""]
    blocks = report.get("neutral_matching_by_condition") or (
        {"low": report["neutral_matching"]} if report.get("neutral_matching") else {})
    blocks = {c: b for c, b in blocks.items() if b and (b.get("treatment") or {}).get("n")}
    if blocks:
        lines += ["## Neutral matching (normalized token edit distance to the original)", ""]
        for cond, nm in blocks.items():
            t, n, g = nm["treatment"], nm["neutral"], nm["abs_gap"]
            lines += ["| condition | n | mean | SD | median |", "| --- | ---: | ---: | ---: | ---: |",
                      f"| {cond} | {t['n']} | {fmt(t['mean'])} | {fmt(t['sd'])} | {fmt(t['median'])} |",
                      f"| neutral | {n['n']} | {fmt(n['mean'])} | {fmt(n['sd'])} | {fmt(n['median'])} |",
                      "", f"|{cond} − neutral| gap: mean {fmt(g['mean'])}, median {fmt(g['median'])}, "
                      f"p90 {fmt(g['p90'])}"
                      + (f"; within tolerance {nm['tolerance']}: {fmt(nm['fraction_within_tolerance'], 2)}"
                         if nm.get('tolerance') is not None else ""), ""]
            paired = nm.get("paired_means", {})
            lines += [f"On {paired.get('n', 0)} successful matched sources, mean distances are "
                      f"{fmt(paired.get('treatment'))} for {cond} and {fmt(paired.get('neutral'))} for Neutral. "
                      "An anchor relationship alone does not establish matched editing intensity.", ""]
            lines += neutral_search_lines(nm)
    if report.get("passage_retention"):
        lines += ["## Complete-passage retention", "",
                  "| Treatment | Eligible | Matched | Retained fraction | Retained mean original words | Excluded mean original words |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for c, row in report["passage_retention"].items():
            lines.append(f"| {c} | {row['eligible']} | {row['matched']} | {fmt(row['retained_fraction'])} | "
                         f"{fmt(row['retained']['mean_original_words'])} | {fmt(row['excluded']['mean_original_words'])} |")
        lines += ["", "Eligible passages are defined by Original. A matched passage needs every selected utterance "
                  "in both treatment and Neutral. Results describe the retained cohort.", ""]
    if report.get("measurement_status"):
        lines += ["## Measurement coverage", "", "| Measure | Status |", "| --- | --- |"]
        lines += [f"| {name} | {status} |" for name, status in report["measurement_status"].items()]
        lines.append("")
    if report.get("range"):
        lines += ["## Family A: linguistic range", ""]
        for block in report["range"].values():
            lines += _contrast_table(block, list(RANGE_METRICS) + list(NUISANCE_METRICS))
    if report.get("guardrails"):
        lines += ["## Family B: grammar / acceptability guardrail (equivalence)", ""]
        for block in report["guardrails"].values():
            lines += _contrast_table(block, GUARDRAIL_METRICS)
    for key, title in (("listener", "Family C: perceived linguistic range"),
                       ("perceived_proficiency", "Perceived English proficiency")):
        if not report.get(key):
            continue
        lines += [f"## {title}", ""]
        for pair, block in report[key].items():
            lines += [f"### {block['treatment']} vs {block['comparator']}: share choosing "
                      f"**{block['treatment']}** as more limited"
                      + (f"  ({block['n_invalid_responses']} invalid responses excluded)"
                         if block.get("n_invalid_responses") else ""), "",
                      "| listener | responses | bundles | game-weighted share | raw-vote share | "
                      "Wilson 95% on raw votes, descriptive | game interval 95% | order ab | order ba | same choice across orders | position B share, descriptive |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for listener, row in block["listeners"].items():
                ab = row["by_order"].get("ab", {}).get("share_treatment")
                ba = row["by_order"].get("ba", {}).get("share_treatment")
                lines.append(f"| {listener} | {row['n_responses']} | {row['n_bundles']} | "
                             f"{fmt(row['share_treatment_more_limited'])} | "
                             f"{fmt(row['share_treatment_raw_votes'])} | "
                             f"[{fmt(row['wilson_ci'][0])}, {fmt(row['wilson_ci'][1])}] | "
                             f"[{fmt(row['cluster_bootstrap_ci'][0])}, {fmt(row['cluster_bootstrap_ci'][1])}] | "
                             f"{fmt(ab)} | {fmt(ba)} | {row['order_consistency']['same_condition']}/"
                             f"{row['order_consistency']['n_comparisons']} | {fmt(row['position_b_raw_share'])} |")
            lines.append("")
            if pair == "neutral:original":
                lines += ["A pooled choice share near 0.5 does not establish equivalence. Both orders must satisfy "
                          "the prespecified equivalence rule; position reversals remain visible above.", ""]
            for name, row in block["listeners"].items():
                if name != "all_models" and row.get("all_pairs_choose_same_position"):
                    lines += [f"{name} selected the same presentation position in both orders for every valid pair. "
                              "This result cannot establish a condition effect or control equivalence.", ""]
    if report.get("cefr"):
        from experiments.proficiency_validation.validation.cefr import render_cefr
        lines += render_cefr(report["cefr"])
        lines += ["Calibration status and artifact checks are recorded in the JSON report. These auxiliary "
                  "ratings do not determine acceptance or the primary validation verdict.", ""]
    if report.get("criteria"):
        v = report.get("verdict") or verdict(report["criteria"])
        lines += ["## Automated validation criteria", "",
                  f"**{'CONFIRMED' if v['confirmed'] else 'NOT CONFIRMED'}**. "
                  f"{v['passed']} passed, {v['failed']} failed, {v['not_evaluated']} not evaluated "
                  "under the frozen protocol. Confirmation requires all criteria to be evaluated and satisfied.", "",
                  "| criterion | result | note |", "| --- | :-: | --- |"]
        for c in report["criteria"]:
            status = {True: "pass", False: "FAIL", None: "not evaluated"}[c["passed"]]
            lines.append(f"| {c['criterion']} | {status} | {c.get('note', '')} |")
        lines.append("")
        lines += ["The validation scope is measured linguistic range and model perception. "
                  "Generalization to human perception or overall English ability requires separate evidence.", ""]
    return "\n".join(lines)


def add_parser(sub) -> None:
    p = sub.add_parser("analyze", help="planned contrasts, guardrails, listener perception, report")
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--guardrails", type=Path, default=None)
    p.add_argument("--listener", type=Path, action="append", default=None,
                   help="range or perceived-proficiency choices JSONL; constructs are analyzed separately (repeatable)")
    p.add_argument("--rewrites", type=Path, default=None)
    p.add_argument("--cefr", type=Path, default=None, help="auxiliary CEFR A/B/C sample ratings JSONL")
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--frozen", type=Path, default=None, help="frozen_config.json (default: data/protocol/)")
    p.add_argument("--no-frozen", action="store_true", help="ignore any frozen config (pilot)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out", type=Path, default=None, help="Markdown report")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=main)


def build_analysis(args) -> dict:
    frozen = None if args.no_frozen else load_frozen(args.frozen)
    if args.frozen and frozen is None:
        raise ValueError(f"no frozen config at {args.frozen}")
    if frozen:
        from experiments.proficiency_validation.validation.protocol import enforce_frozen
        from experiments.proficiency_validation.validation.protocol import cohort
        if args.split not in ("set_a", "set_b"):
            raise ValueError("a frozen analysis needs --split set_a or set_b; for pilot diagnostics pass --no-frozen")
        frozen = enforce_frozen(args.frozen)
        selected = cohort(frozen, args.split)
        args.n_boot, args.seed = frozen["statistics"]["n_boot"], frozen["statistics"]["seed"]
    report: dict = {"created_at": now_iso(), "frozen_config_used": bool(frozen),
                    "frozen_config_path": str(args.frozen) if frozen else None,
                    "n_boot": args.n_boot, "seed": args.seed}
    from experiments.proficiency_validation.rewrite.runtime import sha256_file
    from components import stats as stats_module
    report["analysis_code_sha256"] = {name: sha256_file(path) for name, path in (
        ("analysis", Path(__file__)),
        ("stats", Path(stats_module.__file__)),
        ("cefr", Path(__file__).with_name("cefr.py")), ("rewrite", Path(__file__).parent.parent / "rewrite" / "__main__.py"))}
    report["input_sha256"] = {name: sha256_file(path) for name, path in (
        ("features", args.features), ("guardrails", args.guardrails), ("rewrites", args.rewrites)) if path}
    report["input_sha256"]["listeners"] = {str(path): sha256_file(path) for path in args.listener or []}
    report["measurement_status"] = {"SUBTLEX": "not measured", "Grammar and acceptability": "not measured",
        "Perceived linguistic range": "not measured", "Perceived English proficiency": "not measured",
        "Auxiliary CEFR-referenced A/B/C ratings": "not measured"}
    if frozen:
        report["expected_bundles"] = len(selected)
    rewrites = None
    if args.rewrites:
        rewrites = read_jsonl(args.rewrites)
        from experiments.proficiency_validation.rewrite.fidelity import SEMANTIC_RUBRIC_VERSION
        from components.rewriters.language_rewriter import PROMPT_VERSION
        rw_versions = dict(Counter(r.get("prompt_version", "unknown") for r in rewrites))
        fidelity_versions = dict(Counter((r.get("gate_provenance") or {}).get("semantic_rubric_version", "unknown") for r in rewrites))
        report["recorded_policy"] = {"rewrite_versions": rw_versions, "fidelity_versions": fidelity_versions,
            "requires_revalidation": set(rw_versions) != {PROMPT_VERSION} or set(fidelity_versions) != {SEMANTIC_RUBRIC_VERSION}}
        report["generation_qc"] = generation_qc(rewrites)
        report["failure_causes"] = {c: dict(Counter(failure_cause(r) for r in rewrites if r["condition"] == c))
                                    for c in sorted({r["condition"] for r in rewrites})}
        from experiments.proficiency_validation.rewrite.__main__ import audit_generation
        report["generation_record_errors"] = audit_generation(args.rewrites)
        report["neutral_matching"] = neutral_matching(
            rewrites, tolerance=(frozen or {}).get("neutral_edit_distance_tolerance"))
        report["neutral_matching_by_condition"] = {c: neutral_matching(rewrites, treatment=c,
            tolerance=(frozen or {}).get("neutral_edit_distance_tolerance")) for c in ("low", "high")}
    if args.features:
        features = read_jsonl(args.features)
        report["passage_retention"] = passage_retention(features)
        n_subtlex = sum(r.get("subtlex_mean") is not None for r in features)
        report["measurement_status"]["SUBTLEX"] = f"{n_subtlex}/{len(features)} rows measured"
        report["range"] = contrast_analysis(features, list(RANGE_METRICS) + list(NUISANCE_METRICS),
                                            frozen=frozen, n_boot=args.n_boot, seed=args.seed,
                                            family="linguistic_range",
                                            fdr_metrics=frozen["primary_range_metrics"] if frozen else None)
    if args.guardrails:
        guard = read_jsonl(args.guardrails)
        n_grammar = sum(r.get("grammar_errors_per_100_words") is not None and r.get("cola_mean") is not None for r in guard)
        report["measurement_status"]["Grammar and acceptability"] = f"{n_grammar}/{len(guard)} rows measured by both instruments"
        report["guardrails"] = contrast_analysis(guard, list(GUARDRAIL_METRICS), frozen=frozen,
                                                 n_boot=args.n_boot, seed=args.seed, family="guardrail")
        report["absolute_quality"] = {}
        for c in (frozen or {}).get("conditions", ("original", "neutral", "low", "high")):
            valid = [r for r in guard if r["condition"] == c and r.get("grammar_errors_per_100_words") is not None
                     and r.get("cola_mean") is not None]
            games = {r["game_id"] for r in valid}
            report["absolute_quality"][c] = {"n_games": len({r["game_id"] for r in valid}),
                "n_bundles": len(valid), "estimand": "equal_game_weight",
                "grammar_mean": mean([mean([r["grammar_errors_per_100_words"] for r in valid if r["game_id"] == g]) for g in sorted(games)]),
                "cola_mean": mean([mean([r["cola_mean"] for r in valid if r["game_id"] == g]) for g in sorted(games)])}
    if args.listener:
        choices = [r for path in args.listener for r in read_jsonl(path)]
        from experiments.proficiency_validation.validation.perception import CONSTRUCTS
        if any(r.get("construct", "range") not in CONSTRUCTS for r in choices):
            raise ValueError("unknown listener construct")
        for construct, key, label in (("range", "listener", "Perceived linguistic range"),
                ("perceived_proficiency", "perceived_proficiency", "Perceived English proficiency")):
            subset = [r for r in choices if r.get("construct", "range") == construct]
            report[key] = listener_analysis(subset, n_boot=args.n_boot, seed=args.seed)
            if subset:
                report["measurement_status"][label] = f"{len(subset)} recorded responses; incomplete order pairs excluded"
    if frozen:
        report["integrity"] = integrity_checks(frozen, rewrites=rewrites, features_path=args.features,
                                               listener_paths=args.listener, guardrails_path=args.guardrails,
                                               rewrites_path=args.rewrites, split=args.split)
        if args.rewrites:
            report["integrity"].append({"check": "generation settings match archived output rows",
                "passed": not report["generation_record_errors"], "detail": report["generation_record_errors"]})
    if getattr(args, "cefr", None):
        from experiments.proficiency_validation.validation.cefr import ordinal_analysis, verify_ratings
        rows = read_jsonl(args.cefr)
        provenance, errors = verify_ratings(args.cefr, rewrites_path=args.rewrites, frozen=frozen, split=args.split)
        if errors:
            raise ValueError("; ".join(errors))
        report["cefr"] = ordinal_analysis(rows, n_boot=args.n_boot, seed=args.seed)
        report["cefr"]["calibration"] = provenance.get("calibration")
        report["cefr"]["calibration_status"] = provenance["calibration_status"]
        report["input_sha256"]["cefr"] = sha256_file(args.cefr)
        report["measurement_status"]["Auxiliary CEFR-referenced A/B/C ratings"] = (
            f"{sum(r.get('ok', False) for r in rows)}/{len(rows)} rated; calibration {provenance['calibration_status']}")
    if frozen and frozen.get("cefr", {}).get("enabled"):
        report["integrity"].append({"check": "declared auxiliary CEFR outcomes and provenance recorded",
            "passed": bool(getattr(args, "cefr", None)), "note": "Rating direction is not a selection or confirmation criterion."})
    report["criteria"] = frozen_criteria(report, frozen)
    if report["criteria"]:
        report["verdict"] = verdict(report["criteria"])
    return report


def main(args) -> int:
    try:
        report = build_analysis(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    text = render(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json:
        dump_json(args.json, report)
        print(f"wrote {args.json}")
    return 1 if report["frozen_config_used"] and not report.get("verdict", {}).get("confirmed") else 0
