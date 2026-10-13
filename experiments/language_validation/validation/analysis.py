"""Planned contrasts, Neutral matching diagnostics, listener analysis, integrity checks and the analysis report."""
from __future__ import annotations
from itertools import product
import random
from components.stats import mean
from experiments.language_validation.rewrite.runtime import GUARDRAIL_METRICS
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple
import sys
from components.helpers import read_jsonl
from components.stats import benjamini_hochberg, bradley_terry_bootstrap, cluster_bootstrap, flatten, median, paired_cohens_dz, percentile, sd, tost_equivalence, wilson_interval, proportion_bootstrap
from experiments.language_validation.rewrite.runtime import assert_unique
from experiments.language_validation.rewrite.runtime import CONTRASTS, NUISANCE_METRICS, RANGE_METRICS, PRIMARY_RANGE_METRICS, GUARDRAIL_HARM_DIRECTION, condition_label, dump_json, fmt, load_frozen, now_iso, pair_label
from experiments.language_validation.rewrite.gate import generation_qc, failure_cause


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
        add(f"{condition_label(treatment)} and B treatment availability", available,
            {"rates": availability, "minimum": minimum, "max_differential_attrition": max_gap})
        contrast = f"{treatment}_vs_neutral"
        thresholds = frozen.get("range_effect_thresholds", {}).get(contrast, {})
        for metric in primary:
            m = report.get("range", {}).get(contrast, {}).get("metrics", {}).get(metric)
            threshold = thresholds.get(metric)
            name = f"{condition_label(treatment)} vs B: {metric}, direction/FDR/effect/sample size"
            if not m or threshold is None or m.get("reject_fdr_05") is None or m.get("cohens_dz") is None:
                add(name, None, m, "metric, threshold or inferential result missing")
            else:
                add(name, bool(enough(m) and m.get("direction_confirmed") and m["reject_fdr_05"]
                               and abs(m["cohens_dz"]) >= threshold), m)
        for metric in GUARDRAIL_METRICS:
            m = report.get("guardrails", {}).get(contrast, {}).get("metrics", {}).get(metric)
            ni = (m or {}).get("noninferiority", {}).get("noninferior")
            add(f"{condition_label(treatment)} vs B: {metric} not meaningfully worse (non-inferiority)",
                None if ni is None else bool(enough(m) and ni), m)
        block = report.get("listener", {}).get(f"{treatment}:neutral", {}).get("listeners", {})
        min_effect = frozen.get("listener_min_effect")
        sign = 1 if treatment == "low" else -1

        def meets(ci, estimate):
            if min_effect is None or estimate is None or ci is None or any(x is None for x in ci):
                return None
            return (ci[0] > 0 and estimate >= min_effect) if sign == 1 else (ci[1] < 0 and estimate <= -min_effect)
        for listener in listeners:
            row = block.get(listener) or {}
            main = meets(row.get("signed_ci95"), row.get("signed_preference")) if row else None
            passed = None
            if main is not None:
                passed = enough(row) and main
                for order in ("ab", "ba"):
                    o = (row.get("by_order") or {}).get(order) or {}
                    passed = passed and bool(meets(o.get("ci95"), o.get("signed"))) and enough(o)
            add(f"Listener {listener}: {condition_label(treatment)} vs B, both orders", passed, row or None)
        for listener in sorted(set(block) - set(listeners)):
            add(f"Unexpected listener {listener}", False)
        _position_bias_criterion(add, enough, frozen, block, f"{treatment}:neutral")
    for metric in primary:
        m = report.get("range", {}).get("neutral_vs_original", {}).get("metrics", {}).get(metric)
        eq = (m or {}).get("equivalence", {}).get("equivalent")
        add(f"B vs Original: {metric} equivalent", None if eq is None else bool(enough(m) and eq), m)
    for metric in GUARDRAIL_METRICS:
        m = report.get("guardrails", {}).get("neutral_vs_original", {}).get("metrics", {}).get(metric)
        eq = (m or {}).get("equivalence", {}).get("equivalent")
        add(f"B vs Original: {metric} equivalent", None if eq is None else bool(enough(m) and eq), m)
    margin = frozen.get("listener_equivalence_margin")
    block = report.get("listener", {}).get("neutral:original", {}).get("listeners", {})
    for listener in listeners:
        row = block.get(listener) or {}
        ci = row.get("signed_ci90") or [None, None]
        passed = None
        if row and margin is not None and all(x is not None for x in ci):
            within = lambda interval: (interval is not None and all(x is not None for x in interval)
                                       and interval[0] > -margin and interval[1] < margin)
            passed = enough(row) and within(ci)
            for order in ("ab", "ba"):
                o = (row.get("by_order") or {}).get(order) or {}
                passed = passed and enough(o) and within(o.get("ci90"))
        add(f"Listener {listener}: B vs Original equivalence, both orders", passed, row or None)
    _position_bias_criterion(add, enough, frozen, block, "neutral:original")
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
        # Evaluate perceived proficiency with the same listener criteria.
        perception_report = {**report, "listener": report.get("perceived_proficiency", {})}
        range_only = {**frozen, "perception_constructs": ["range"]}
        for item in criteria(perception_report, range_only):
            if item["criterion"].startswith(("Listener ", "Unexpected listener ")):
                out.append({**item, "criterion": "Perceived proficiency: " + item["criterion"]})
    return out


def _position_bias_criterion(add, enough, frozen, block, pair):
    """Cross-family position score (A = +1, B = -1, same = 0) must sit within a frozen margin of zero."""
    margin = frozen.get("position_bias_margin")
    row = block.get("all_models") or {}
    bias = row.get("position_bias") or {}
    ci = bias.get("ci90") or [None, None]
    passed = None
    if row and margin is not None and all(x is not None for x in ci):
        passed = enough(row) and ci[0] > -margin and ci[1] < margin
    add(f"Listener position bias within margin: {pair}", passed, bias or None)


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
    """Paired game contrasts with BH correction over the specified metric family."""
    bounds = (frozen or {}).get("equivalence_bounds", {}) or {}
    ni_margins = (frozen or {}).get("noninferiority_margins", {}) or {}
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
            if metric in GUARDRAIL_METRICS and comparator == "neutral":
                # Use the harmful tail of the game-bootstrap distribution.
                draws = equivalence_from_same_draws.get("draws") or []
                harm = GUARDRAIL_HARM_DIRECTION[metric]
                margin = ni_margins.get(metric)
                estimable = boot["n_clusters"] >= 2 and draws and min(draws) < max(draws)
                bound_value = (percentile(draws, 95) if harm > 0 else percentile(draws, 5)) if estimable else None
                verdict_ni = None
                if bound_value is not None and margin is not None:
                    verdict_ni = bool(bound_value < margin) if harm > 0 else bool(bound_value > -margin)
                entry["noninferiority"] = {"margin": margin, "harm_direction": "higher is worse" if harm > 0 else "lower is worse",
                                           "interval_method": "game_bootstrap" if estimable else "insufficient_variation",
                                           "one_sided_95_bound": bound_value, "noninferior": verdict_ni}
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


def splitting_sensitivity(features: Sequence[dict], *, primary: Sequence[str], n_boot: int, seed: int) -> dict:
    """Primary treatment-versus-Neutral effects restricted to passages with modest sentence splitting:
    a sentence-count change no larger than the median absolute change. Shows whether the lexical and
    syntactic separation survives when Low did not split sentences much."""
    by_bc = {(r["bundle_id"], r["condition"]): r for r in features if r.get("complete", True)}
    out: Dict[str, dict] = {}
    for treatment in ("low", "high"):
        ids = sorted({b for (b, c) in by_bc if c == treatment} & {b for (b, c) in by_bc if c == "neutral"})
        deltas = {}
        for b in ids:
            t, c = by_bc[(b, treatment)].get("n_sentences"), by_bc[(b, "neutral")].get("n_sentences")
            if t is not None and c is not None:
                deltas[b] = t - c
        if not deltas:
            continue
        threshold = median([abs(d) for d in deltas.values()])
        modest = {b for b, d in deltas.items() if abs(d) <= threshold}
        metrics: Dict[str, dict] = {}
        for metric in primary:
            clusters: Dict[str, List[float]] = {}
            for b in sorted(modest):
                t, c = by_bc[(b, treatment)].get(metric), by_bc[(b, "neutral")].get(metric)
                if t is not None and c is not None:
                    clusters.setdefault(str(by_bc[(b, treatment)].get("game_id")), []).append(t - c)
            if not clusters:
                continue
            boot = cluster_bootstrap({g: [mean(v)] for g, v in clusters.items()}, n_boot=n_boot, seed=seed)
            expected = expected_direction(metric, treatment, "neutral")
            excludes = boot["ci_low"] is not None and (boot["ci_low"] > 0 or boot["ci_high"] < 0)
            metrics[metric] = {"estimate": boot["estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                               "n_games": boot["n_clusters"], "n_bundles": sum(len(v) for v in clusters.values()),
                               "expected_sign": expected,
                               "direction_confirmed": (bool(excludes and (boot["estimate"] > 0) == (expected > 0))
                                                       if expected in (1, -1) and boot["estimate"] is not None else None)}
        out[f"{treatment}_vs_neutral"] = {"threshold_abs_delta_sentences": threshold,
                                          "n_passages_all": len(deltas), "n_passages_modest": len(modest),
                                          "mean_delta_sentences_all": mean(list(deltas.values())),
                                          "mean_delta_sentences_modest": mean([deltas[b] for b in modest]) if modest else None,
                                          "metrics": metrics}
    return out


def sensitivity_lines(sensitivity: dict) -> List[str]:
    lines: List[str] = []
    for contrast, block in (sensitivity or {}).items():
        lines += [f"### Sentence-splitting sensitivity: {pair_label(contrast, ' − ')}", "",
                  f"Passages whose sentence count changed by at most {fmt(block['threshold_abs_delta_sentences'], 0)} "
                  f"(the median absolute change): {block['n_passages_modest']} of {block['n_passages_all']} passages, mean change "
                  f"{fmt(block['mean_delta_sentences_modest'], 2)} sentences against {fmt(block['mean_delta_sentences_all'], 2)} overall. "
                  "Sentence count and length are diagnostics of how A rewrites.", "",
                  "| metric | expected | Δ game mean, modest splitting | 95% CI (game bootstrap) | games | passages | CI direction |",
                  "| --- | :-: | ---: | ---: | ---: | ---: | :-: |"]
        for metric, m in block["metrics"].items():
            exp = {1: "↑", -1: "↓", 0: "="}.get(m["expected_sign"], "")
            lines.append(f"| {metric} | {exp} | {fmt(m['estimate'])} | [{fmt(m['ci_low'])}, {fmt(m['ci_high'])}] | "
                         f"{m['n_games']} | {m['n_bundles']} | {fmt(m['direction_confirmed'])} |")
        lines.append("")
    return lines


def _distribution(values: Sequence[float]) -> dict:
    return {"n": len(values), "mean": mean(values), "sd": sd(values), "median": median(values),
            "p10": percentile(values, 10), "p90": percentile(values, 90)}


def rewrite_exposure(rewrites: Sequence[dict]) -> dict:
    """How much rewriting each accepted condition carries: normalized token edit distance to the
    original, per condition, with the Low/High-versus-Neutral gap and distribution overlap on
    sources where both were accepted. Diagnostic only: Neutral is generated independently of Low,
    and the Neutral control is judged by its equivalence to Original."""
    by_source: Dict[str, Dict[str, float]] = {}
    for r in rewrites:
        if r.get("status") == "ok" and r.get("edit_distance_to_original") is not None:
            by_source.setdefault(r["source_id"], {})[r["condition"]] = r["edit_distance_to_original"]
    conditions = sorted({c for d in by_source.values() for c in d})
    out = {"n_sources": len(by_source),
           "conditions": {c: _distribution([d[c] for d in by_source.values() if c in d]) for c in conditions},
           "pairs": {}}
    bins = 20
    def histogram(values):
        counts = [0] * bins
        for v in values:
            counts[min(bins - 1, int(max(0., v) * bins))] += 1
        return [c / len(values) for c in counts]
    for treatment in ("low", "high"):
        paired = [d for d in by_source.values() if treatment in d and "neutral" in d]
        if not paired:
            continue
        t_values, n_values = [d[treatment] for d in paired], [d["neutral"] for d in paired]
        gaps = [d[treatment] - d["neutral"] for d in paired]
        out["pairs"][f"{treatment}_vs_neutral"] = {
            "n": len(paired), "treatment_mean": mean(t_values), "neutral_mean": mean(n_values),
            "gap": {"mean": mean(gaps), "median": median(gaps), "p10": percentile(gaps, 10), "p90": percentile(gaps, 90)},
            "distribution_overlap": sum(min(a, b) for a, b in zip(histogram(t_values), histogram(n_values))),
            "neutral_at_least_half_of_treatment": sum(1 for d in paired if d["neutral"] >= .5 * d[treatment]) / len(paired)}
    out["search"] = neutral_search_diagnostics(rewrites)
    return out


def neutral_search_diagnostics(rewrites: Sequence[dict]) -> dict:
    """Summarize the recorded Neutral candidate search."""
    counts = Counter()
    for row in rewrites:
        if row.get("condition") != "neutral" or not row.get("attempts"):
            continue
        counts["attempted_sources"] += 1
        seen, any_passer = set(), False
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
                counts["passing_candidates"] += int(candidate.get("ok") is True)
                any_passer = any_passer or candidate.get("ok") is True
                seen.add(text)
        counts["unique_candidates"] += len(seen)
        if row.get("status") == "ok":
            counts["accepted_first_round" if len(row["attempts"]) == 1 else "accepted_after_retry"] += 1
        elif not any_passer:
            counts["failed_without_fidelity_passer"] += 1
    return {key: counts[key] for key in (
        "attempted_sources", "attempts", "candidate_occurrences", "unique_candidates", "passing_candidates",
        "repeated_candidate_occurrences", "reused_verdicts", "accepted_first_round", "accepted_after_retry",
        "failed_without_fidelity_passer")}


def exposure_lines(exposure: dict) -> List[str]:
    """Shared rendering of the rewrite-exposure diagnostics for the source and analysis reports."""
    if not exposure or not exposure.get("conditions"):
        return []
    lines = ["", "| Condition | Accepted rewrites | Mean edit distance | SD | Median | p10 | p90 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c, d in exposure["conditions"].items():
        lines.append(f"| {condition_label(c)} | {d['n']} | {fmt(d['mean'])} | {fmt(d['sd'])} | {fmt(d['median'])} | {fmt(d['p10'])} | {fmt(d['p90'])} |")
    for pair, d in exposure.get("pairs", {}).items():
        treatment = condition_label(pair.split("_vs_")[0])
        lines += ["", f"On {d['n']} sources with both {treatment} and B accepted, {treatment} changes a mean "
                  f"{fmt(d['treatment_mean'])} of the tokens and B {fmt(d['neutral_mean'])} (median gap "
                  f"{fmt(d['gap']['median'])}; distribution overlap {fmt(d['distribution_overlap'], 2)}; B reaches at "
                  f"least half of {treatment}'s distance in {fmt(d['neutral_at_least_half_of_treatment'], 2)} of sources)."]
    search = exposure.get("search") or {}
    if search.get("attempts"):
        lines += ["", f"Across {search['attempts']} B rounds, {search['passing_candidates']} of "
                  f"{search['candidate_occurrences']} judged candidates passed the gate; {search['accepted_first_round']} sources "
                  f"were accepted in the first round and {search['accepted_after_retry']} after a retry. "
                  f"{search['failed_without_fidelity_passer']} sources never produced a passing candidate. "
                  f"{search['repeated_candidate_occurrences']} candidate occurrences repeat an earlier candidate; "
                  f"{search['reused_verdicts']} verdicts are marked as reused."]
    lines += ["", "Edit distance is normalized token edit distance to the original and describes rewrite exposure. "
              "B is generated independently of A; its neutrality is tested by the B-versus-Original comparisons.", ""]
    return lines


def signed_value(r: dict) -> Optional[int]:
    """+1 treatment judged more limited, -1 comparator judged more limited, 0 no meaningful
    difference; ``None`` when the response is unusable."""
    if not r.get("ok"):
        return None
    if type(r.get("signed")) is int and r["signed"] in (1, 0, -1):
        return r["signed"]
    if r.get("choice") == "same" or r.get("tie"):
        return 0
    chosen = r.get("chosen_condition")
    if chosen == r.get("treatment"):
        return 1
    if chosen == r.get("comparator"):
        return -1
    return None


def position_of(r: dict) -> Optional[str]:
    """Which presentation slot the listener picked: ``"A"``, ``"B"`` or ``"same"``."""
    if r.get("position") in ("A", "B", "same"):
        return r["position"]
    if r.get("choice") in ("A", "B", "same"):
        return r["choice"]
    s = signed_value(r)
    if s is None:
        return None
    if s == 0:
        return "same"
    return "A" if (s == 1) == (r.get("order") == "ab") else "B"


def _game_means(per_bundle: Dict[str, float], game_of: Dict[str, str]) -> Dict[str, List[float]]:
    clusters: Dict[str, List[float]] = {}
    for bundle, value in per_bundle.items():
        clusters.setdefault(game_of[bundle], []).append(value)
    return {g: [mean(values)] for g, values in clusters.items()}


def signed_bootstrap(clusters, *, n_boot, seed, alpha=.05):
    """Bootstrap signed scores on [-1, 1], with a bounded-mean interval for constant draws."""
    scaled = {g: [(v + 1) / 2 for v in values] for g, values in clusters.items()}
    result = proportion_bootstrap(scaled, n_boot=n_boot, seed=seed, alpha=alpha)
    for key in ("estimate", "ci_low", "ci_high"):
        if result[key] is not None:
            result[key] = 2 * result[key] - 1
    result["draws"] = [2 * v - 1 for v in result.get("draws", [])]
    return result


def complete_listener_rows(choices):
    """Keep passage/listener comparisons with valid responses in both orders."""
    assert_unique(choices, ("bundle_id", "pair", "listener", "order"))
    valid = [r for r in choices if signed_value(r) is not None]
    orders = {}
    for r in valid:
        orders.setdefault((r["bundle_id"], r["pair"], r["listener"]), set()).add(r["order"])
    rows = [r for r in valid if orders[(r["bundle_id"], r["pair"], r["listener"])] == {"ab", "ba"}]
    return rows, len(choices) - len(valid), len(valid) - len(rows)


def listener_analysis(choices: Sequence[dict], *, n_boot: int, seed: int) -> Dict[str, dict]:
    """Signed preference from complete counterbalanced three-way comparisons.

    For every passage and listener the two presentation orders are averaged, then listeners
    (within ``all_models``), then passages within a game; the game-level value is the datum.
    +1 means the treatment was judged more limited, 0 no meaningful difference, -1 the comparator.
    The share of treatment choices among non-tie responses and a Bradley-Terry fit are kept as
    secondary descriptions. A position score (+1 when slot A is chosen, -1 for B, 0 for ties)
    diagnoses position bias."""
    constructs = {r.get("construct", "range") for r in choices}
    if len(constructs) > 1:
        raise ValueError("analyze perception constructs separately")
    out: Dict[str, dict] = {}
    for pair in sorted({r["pair"] for r in choices}):
        pair_rows = [r for r in choices if r["pair"] == pair]
        rows, errors, incomplete = complete_listener_rows(pair_rows)
        if not rows:
            continue
        treatment, comparator = rows[0]["treatment"], rows[0]["comparator"]
        listeners = sorted({r["listener"] for r in rows})
        per_listener: Dict[str, dict] = {}
        for listener in listeners + ["all_models"]:
            subset = rows if listener == "all_models" else [r for r in rows if r["listener"] == listener]
            game_of = {r["bundle_id"]: str(r.get("game_id")) for r in subset}

            def bundle_means(values_of) -> Dict[str, float]:
                by_bl: Dict[tuple, List[float]] = {}
                for r in subset:
                    by_bl.setdefault((r["bundle_id"], r["listener"]), []).append(values_of(r))
                per_bundle: Dict[str, List[float]] = {}
                for (b, _), vals in by_bl.items():
                    per_bundle.setdefault(b, []).append(mean(vals))
                return {b: mean(vals) for b, vals in per_bundle.items()}

            signed_bundle = bundle_means(signed_value)
            signed_clusters = _game_means(signed_bundle, game_of)
            s95 = signed_bootstrap(signed_clusters, n_boot=n_boot, seed=seed); s95.pop("draws", None)
            s90 = signed_bootstrap(signed_clusters, n_boot=n_boot, seed=seed, alpha=.10); s90.pop("draws", None)
            position_bundle = bundle_means(lambda r: {"A": 1, "B": -1, "same": 0}[position_of(r)])
            p90 = signed_bootstrap(_game_means(position_bundle, game_of), n_boot=n_boot, seed=seed, alpha=.10)
            p90.pop("draws", None)
            ties = sum(1 for r in subset if signed_value(r) == 0)
            non_tie = [r for r in subset if signed_value(r) != 0]
            share = raw = interval_method = None
            share_ci, wilson = [None, None], [None, None]
            if non_tie:
                by_b: Dict[str, List[int]] = {}
                for r in non_tie:
                    by_b.setdefault(r["bundle_id"], []).append(1 if signed_value(r) == 1 else 0)
                boot = proportion_bootstrap(_game_means({b: mean(v) for b, v in by_b.items()}, game_of),
                                            n_boot=n_boot, seed=seed)
                boot.pop("draws", None)
                share, share_ci, interval_method = boot["estimate"], [boot["ci_low"], boot["ci_high"]], boot["interval_method"]
                k = sum(1 for r in non_tie if signed_value(r) == 1)
                raw, w_lo, w_hi = wilson_interval(k, len(non_tie))
                wilson = [w_lo, w_hi]
            by_order = {}
            for order in ("ab", "ba"):
                o = [r for r in subset if r["order"] == order]
                if not o:
                    continue
                ob: Dict[str, List[float]] = {}
                for r in o:
                    ob.setdefault(r["bundle_id"], []).append(signed_value(r))
                oc = _game_means({b: mean(v) for b, v in ob.items()}, game_of)
                o95 = signed_bootstrap(oc, n_boot=n_boot, seed=seed)
                o90 = signed_bootstrap(oc, n_boot=n_boot, seed=seed, alpha=.10)
                by_order[order] = {"n": len(o), "n_games": o95["n_clusters"], "n_bundles": len(ob),
                                   "signed": o95["estimate"], "ci95": [o95["ci_low"], o95["ci_high"]],
                                   "ci90": [o90["ci_low"], o90["ci_high"]],
                                   "interval_method": o95["interval_method"],
                                   "tie_rate": sum(1 for r in o if signed_value(r) == 0) / len(o)}
            comps_by_cluster: Dict[str, List[Tuple[str, str]]] = {}
            for r in non_tie:
                winner, loser = (treatment, comparator) if signed_value(r) == 1 else (comparator, treatment)
                comps_by_cluster.setdefault(str(r.get("game_id")), []).append((winner, loser))
            bt = (bradley_terry_bootstrap(comps_by_cluster, treatment, comparator, n_boot=n_boot, seed=seed)
                  if comps_by_cluster else None)
            paired_orders: Dict[tuple, Dict[str, dict]] = {}
            for r in subset:
                paired_orders.setdefault((r["bundle_id"], r["listener"]), {})[r["order"]] = r

            def label(r):
                s = signed_value(r)
                return "same" if s == 0 else (treatment if s == 1 else comparator)
            consistent = sum(label(p["ab"]) == label(p["ba"]) for p in paired_orders.values())
            always_a = sum(all(position_of(r) == "A" for r in p.values()) for p in paired_orders.values())
            always_b = sum(all(position_of(r) == "B" for r in p.values()) for p in paired_orders.values())
            per_listener[listener] = {
                "n_responses": len(subset), "estimand": "equal_game_weight",
                "n_bundles": len(signed_bundle), "n_games": s95["n_clusters"],
                "signed_preference": s95["estimate"], "sd_games": sd([v[0] for v in signed_clusters.values()]),
                "signed_ci95": [s95["ci_low"], s95["ci_high"]], "signed_ci90": [s90["ci_low"], s90["ci_high"]],
                "signed_interval_method": s95["interval_method"],
                "signed_ci_excludes_zero": (s95["ci_low"] is not None and (s95["ci_low"] > 0 or s95["ci_high"] < 0)),
                "n_ties": ties, "tie_rate": ties / len(subset),
                "share_treatment_more_limited": share, "share_treatment_raw_votes": raw, "wilson_ci": wilson,
                "cluster_bootstrap_ci": share_ci, "interval_method": interval_method,
                "ci_lower_above_half": (share_ci[0] is not None and share_ci[0] > 0.5),
                "ci_covers_half": (share_ci[0] is not None and share_ci[1] is not None and share_ci[0] <= 0.5 <= share_ci[1]),
                "by_order": by_order, "bradley_terry": bt,
                "position_bias": {"score": p90["estimate"], "ci90": [p90["ci_low"], p90["ci_high"]],
                                  "interval_method": p90["interval_method"],
                                  "n_responses": len(subset),
                                  "share_a": sum(position_of(r) == "A" for r in subset) / len(subset),
                                  "share_b": sum(position_of(r) == "B" for r in subset) / len(subset),
                                  "share_same": ties / len(subset)},
                "order_consistency": {"n_comparisons": len(paired_orders), "same_condition": consistent,
                                      "reversed_condition": len(paired_orders) - consistent,
                                      "fraction_same": consistent / len(paired_orders),
                                      "always_a": always_a, "always_b": always_b},
                "position_b_raw_share": sum(position_of(r) == "B" for r in subset) / len(subset),
                "all_pairs_choose_same_position": always_a == len(paired_orders) or always_b == len(paired_orders),
                "confidence": {c: sum(1 for r in subset if r.get("confidence") == c) for c in ("high", "medium", "low")},
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
    from experiments.language_validation.validation.protocol import verify_hashes, expected_template_hash
    from experiments.language_validation.validation.protocol import validate_design, cohort, sha256_file
    from experiments.language_validation.rewrite.runtime import verify_artifact, assert_unique
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
    lines = [f"### {condition_label(block['treatment'])} − {condition_label(block['comparator'])}  ({block['role']}; "
             f"n bundles = {next(iter(block['metrics'].values()))['n_bundles']}, "
             f"games = {next(iter(block['metrics'].values()))['n_games']})", "",
             "| metric | expected | Δ game mean | 95% CI (game bootstrap) | game d_z | p (sign-flip) | q (BH) | CI direction only / equivalence |",
             "| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |"]
    for metric in metrics:
        m = block["metrics"].get(metric)
        if not m:
            continue
        label = metric + (" (primary)" if block.get("family") == "linguistic_range" and metric in PRIMARY_RANGE_METRICS else "")
        exp = {1: "↑", -1: "↓", 0: "="}.get(m["expected_sign"], "")
        conf = m.get("direction_confirmed")
        if block["treatment"] == "neutral" and block["comparator"] == "original" and m.get("equivalence"):
            exp = "="
            conf = m["equivalence"].get("equivalent")
        if m.get("noninferiority"):
            ni = m["noninferiority"]
            conf = (f"non-inferior: {fmt(ni['noninferior'])} (one-sided bound {fmt(ni['one_sided_95_bound'])}"
                    + (f", margin {fmt(ni['margin'])})" if ni.get("margin") is not None else ", no frozen margin)"))
        lines.append(f"| {label} | {exp} | {fmt(m['estimate'])} | [{fmt(m['ci_low'])}, {fmt(m['ci_high'])}] | "
                     f"{fmt(m['cohens_dz'], 2)} | {fmt(m['p_sign_flip'])} | {fmt(m.get('q_bh'))} | "
                     f"{conf if isinstance(conf, str) else fmt(conf)} |")
    first = next(iter(block["metrics"].values()))
    if first.get("sign_flip_method") == "exact":
        lines += ["", f"The sign-flip test enumerates all {first['sign_flip_patterns']} sign patterns. "
                  f"The smallest attainable two-sided p-value is {fmt(first['minimum_two_sided_p'], 4)}."]
    return lines + [""]


def render(report: dict) -> str:
    lines = ["# Proficiency validation report", "",
             f"Generated {report['created_at']}. Frozen config: "
             f"{'yes' if report.get('frozen_config_used') else 'no (pilot / exploratory)'}.", ""]
    lines += ["The primary perception summary is the signed preference: +1 when the listener judges the treatment "
              "more limited, 0 for no meaningful difference, -1 when it judges the comparator more limited. Presentation "
              "orders, listeners and passages within a game are averaged; games receive equal weight. "
              "Intervals resample games, with a Hoeffding bounded-mean interval for constant listener scores.", ""]
    if not report.get("frozen_config_used"):
        lines += ["Pilot analysis: the frozen effect-size, multiplicity and quality criteria apply to the confirmatory sets.", ""]
    if report.get("recorded_policy"):
        lines += [f"Recorded rewrite prompts: {report['recorded_policy']['rewrite_versions']}. "
                  f"Recorded fidelity rubrics: {report['recorded_policy']['fidelity_versions']}.", ""]
        if report["recorded_policy"].get("requires_revalidation"):
            lines += ["These model outputs have not been evaluated under the current generation and fidelity policy.", ""]
    if report.get("generation_record_errors"):
        lines += ["## Generation record inconsistencies", "",
                  "Archived settings disagree with the outputs. These records require the actual run metadata before reproducibility can be established.", ""]
        lines += [f"- {error}" for error in report["generation_record_errors"]] + [""]
    audit = report.get("accepted_alignment_audit", {})
    if audit.get("issues"):
        lines += ["## Alignment evidence recheck", "",
                  f"The current check flags {len(audit['issues'])} of {audit['checked']} accepted records. "
                  "These records need a repeated fidelity evaluation. Passage tables use the recorded acceptance decisions.", ""]
        lines += [f"- {r['source_id']} / {r['condition']}: {', '.join(r['errors'])}" for r in audit["issues"]] + [""]
    qc = report.get("generation_qc")
    if qc:
        lines += ["## Generation QC", "",
                  "| condition | n | accepted | first-pass | retried | treatment unavailable | infrastructure unavailable | fallback | mean attempts |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for c, q in qc.items():
            lines.append(f"| {condition_label(c)} | {q['n']} | {q['final_success']} | {q['first_pass_success']} | {q['needed_retry']} | "
                         f"{q['treatment_unavailable']} | {q.get('infrastructure_unavailable', 0)} | "
                         f"{q['fallback_to_original']} | {fmt(q['mean_attempts'], 2)} |")
        lines.append("")
    if report.get("failure_causes"):
        lines += ["| Condition | Recorded outcome cause | Sources |", "| --- | --- | ---: |"]
        for c, counts in report["failure_causes"].items():
            lines += [f"| {condition_label(c)} | {cause} | {n} |" for cause, n in counts.items() if cause != "accepted"]
        lines += ["", "Evaluation-output failures are separate from transport failures.", ""]
    if report.get("rewrite_exposure"):
        lines += ["## Rewrite exposure (normalized token edit distance to the original)"]
        lines += exposure_lines(report["rewrite_exposure"])
    if report.get("passage_retention"):
        lines += ["## Complete-passage retention", "",
                  "| Treatment | Eligible | Matched | Retained fraction | Retained mean original words | Excluded mean original words |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for c, row in report["passage_retention"].items():
            lines.append(f"| {condition_label(c)} | {row['eligible']} | {row['matched']} | {fmt(row['retained_fraction'])} | "
                         f"{fmt(row['retained']['mean_original_words'])} | {fmt(row['excluded']['mean_original_words'])} |")
        lines += ["", "Eligible passages are defined by Original. A matched passage needs every selected utterance "
                  "in both treatment and B. Results describe the retained cohort.", ""]
    if report.get("measurement_status"):
        lines += ["## Measurement coverage", "", "| Measure | Status |", "| --- | --- |"]
        lines += [f"| {name} | {status} |" for name, status in report["measurement_status"].items() if status != "not measured"]
        lines.append("")
    if report.get("range"):
        lines += ["## Linguistic range", ""]
        for block in report["range"].values():
            lines += _contrast_table(block, list(RANGE_METRICS) + list(NUISANCE_METRICS))
        lines += sensitivity_lines(report.get("splitting_sensitivity"))
    if report.get("guardrails"):
        lines += ["## Grammar and acceptability guardrails", ""]
        for block in report["guardrails"].values():
            lines += _contrast_table(block, GUARDRAIL_METRICS)
    for key, title in (("listener", "Perceived linguistic range"),
                       ("perceived_proficiency", "Perceived English proficiency")):
        if not report.get(key):
            continue
        lines += [f"## {title}", ""]
        for pair, block in report[key].items():
            lines += [f"### {condition_label(block['treatment'])} vs {condition_label(block['comparator'])}: share choosing "
                      f"**{condition_label(block['treatment'])}** as more limited"
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
                lines += ["The equivalence rule applies to the 90% interval of the signed preference in both orders.", ""]
            for name, row in block["listeners"].items():
                if name != "all_models" and row.get("all_pairs_choose_same_position"):
                    lines += [f"{name} selected the same presentation position in both orders for every valid pair.", ""]
    if report.get("cefr"):
        from experiments.language_validation.validation.cefr import render_cefr
        lines += render_cefr(report["cefr"])
        lines += ["Calibration status and artifact checks are recorded in the JSON report.", ""]
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
        from experiments.language_validation.validation.protocol import enforce_frozen
        from experiments.language_validation.validation.protocol import cohort
        if args.split not in ("set_a", "set_b"):
            raise ValueError("a frozen analysis needs --split set_a or set_b; for pilot diagnostics pass --no-frozen")
        frozen = enforce_frozen(args.frozen)
        selected = cohort(frozen, args.split)
        args.n_boot, args.seed = frozen["statistics"]["n_boot"], frozen["statistics"]["seed"]
    report: dict = {"created_at": now_iso(), "frozen_config_used": bool(frozen),
                    "frozen_config_path": str(args.frozen) if frozen else None,
                    "n_boot": args.n_boot, "seed": args.seed}
    from experiments.language_validation.rewrite.runtime import sha256_file
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
        "Auxiliary CEFR band ratings": "not measured"}
    if frozen:
        report["expected_bundles"] = len(selected)
    rewrites = None
    if args.rewrites:
        rewrites = read_jsonl(args.rewrites)
        from experiments.language_validation.rewrite.fidelity import SEMANTIC_RUBRIC_VERSION, audit_accepted_alignments
        from components.rewriters.language_rewriter import PROMPT_VERSION
        rw_versions = dict(Counter(r.get("prompt_version", "unknown") for r in rewrites))
        fidelity_versions = dict(Counter((r.get("gate_provenance") or {}).get("semantic_rubric_version", "unknown") for r in rewrites))
        report["recorded_policy"] = {"rewrite_versions": rw_versions, "fidelity_versions": fidelity_versions,
            "requires_revalidation": set(rw_versions) != {PROMPT_VERSION} or set(fidelity_versions) != {SEMANTIC_RUBRIC_VERSION}}
        report["generation_qc"] = generation_qc(rewrites)
        report["failure_causes"] = {c: dict(Counter(failure_cause(r) for r in rewrites if r["condition"] == c))
                                    for c in sorted({r["condition"] for r in rewrites})}
        from experiments.language_validation.rewrite.__main__ import audit_generation
        report["generation_record_errors"] = audit_generation(args.rewrites)
        report["accepted_alignment_audit"] = audit_accepted_alignments(rewrites)
        report["rewrite_exposure"] = rewrite_exposure(rewrites)
    if args.features:
        features = read_jsonl(args.features)
        report["passage_retention"] = passage_retention(features)
        n_subtlex = sum(r.get("subtlex_mean") is not None for r in features)
        report["measurement_status"]["SUBTLEX"] = f"{n_subtlex}/{len(features)} rows measured"
        primary = frozen["primary_range_metrics"] if frozen else list(PRIMARY_RANGE_METRICS)
        # BH runs over the small frozen primary family only; the other range measures are secondary diagnostics.
        report["range"] = contrast_analysis(features, list(RANGE_METRICS) + list(NUISANCE_METRICS),
                                            frozen=frozen, n_boot=args.n_boot, seed=args.seed,
                                            family="linguistic_range", fdr_metrics=primary)
        report["splitting_sensitivity"] = splitting_sensitivity(features, primary=primary, n_boot=args.n_boot, seed=args.seed)
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
        from experiments.language_validation.validation.perception import CONSTRUCTS
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
            issues = report["accepted_alignment_audit"]["issues"]
            report["integrity"].append({"check": "accepted rewrites have grounded alignment evidence",
                "passed": not issues, "detail": issues})
    if getattr(args, "cefr", None):
        from experiments.language_validation.validation.cefr import ordinal_analysis, verify_ratings
        rows = read_jsonl(args.cefr)
        provenance, errors = verify_ratings(args.cefr, rewrites_path=args.rewrites, frozen=frozen, split=args.split)
        if errors:
            raise ValueError("; ".join(errors))
        report["cefr"] = ordinal_analysis(rows, n_boot=args.n_boot, seed=args.seed)
        report["cefr"]["calibration"] = provenance.get("calibration")
        report["cefr"]["calibration_status"] = provenance["calibration_status"]
        report["input_sha256"]["cefr"] = sha256_file(args.cefr)
        report["measurement_status"]["Auxiliary CEFR band ratings"] = (
            f"{sum(r.get('ok', False) for r in rows)}/{len(rows)} rated"
            + (f"; calibration {provenance['calibration_status']}" if provenance.get("calibration_status") not in (None, "not_measured") else ""))
    if frozen and frozen.get("cefr", {}).get("enabled"):
        report["integrity"].append({"check": "declared auxiliary CEFR outcomes and provenance recorded",
            "passed": bool(getattr(args, "cefr", None)), "note": "auxiliary rating, recorded alongside the primary criteria"})
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
