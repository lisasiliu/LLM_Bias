"""Explicit, fail-closed criteria for every condition in the frozen design."""
from __future__ import annotations
from .common import GUARDRAIL_METRICS


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
            direction = (lambda x: x[0] is not None and x[0] > .5) if treatment == "low" else (
                         lambda x: x[1] is not None and x[1] < .5)
            passed = None
            if row and all(x is not None for x in ci):
                passed = enough(row) and direction(ci)
                for order in ("ab", "ba"):
                    o = row.get("by_order", {}).get(order)
                    passed = passed and enough(o) and direction((o or {}).get("ci95", [None, None]))
            add(f"Listener {listener}: {treatment} vs Neutral, both orders", passed, row)
        unexpected = set(block) - set(listeners) - {"all_humans"}
        for listener in sorted(unexpected):
            if not listener.startswith("human:"):
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
    return out
