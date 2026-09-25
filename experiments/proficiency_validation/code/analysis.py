"""Primary analysis (PLAN 12) and the report (PLAN 17).

Everything is a matched comparison within a source bundle, and every interval
or p-value treats the GAME as the independent unit (cluster bootstrap and
cluster sign-flip tests). Within a metric family the planned contrasts are
corrected with Benjamini-Hochberg. Guardrails and the Neutral control are
equivalence questions and use TOST with the frozen bounds.

Inputs are the JSONL files the other commands write; nothing here calls a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys

from components.helpers import read_jsonl
from components.stats import (
    benjamini_hochberg, bradley_terry_bootstrap, cluster_bootstrap, cluster_sign_flip_test,
    cohen_kappa, flatten, mean, median, paired_cohens_dz, percentile, sd, tost_equivalence,
    wilson_interval)

from .common import (
    CONTRASTS, GUARDRAIL_METRICS, NUISANCE_METRICS, PRIMARY_RANGE_METRICS, RANGE_METRICS,
    dump_json, fmt, load_frozen, now_iso)
from .semantic_gate import generation_qc


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
            boot = cluster_bootstrap(clusters, n_boot=n_boot, seed=seed)
            test = cluster_sign_flip_test(clusters, seed=seed)
            equivalence_from_same_draws = dict(boot)          # reuse, then drop the raw draws
            boot.pop("draws", None)
            expected = expected_direction(metric, treatment, comparator)
            estimate = boot["estimate"]
            excludes_zero = (boot["ci_low"] is not None and
                             (boot["ci_low"] > 0 or boot["ci_high"] < 0))
            entry = {"n_bundles": n, "n_games": boot["n_clusters"], "estimate": estimate,
                     "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                     "sd_diff": sd(diffs), "cohens_dz": paired_cohens_dz(diffs),
                     "p_sign_flip": test["p_value"], "expected_sign": expected,
                     "direction_confirmed": None}
            if expected in (1, -1):
                entry["direction_confirmed"] = bool(estimate is not None and excludes_zero
                                                    and (estimate > 0) == (expected > 0))
            bound = bounds.get(metric)
            if expected == 0 or metric in GUARDRAIL_METRICS:
                entry["equivalence"] = (tost_equivalence(clusters, bound, bootstrap=equivalence_from_same_draws)
                                        if bound is not None else {"bound": None, "equivalent": None})
            per_metric[metric] = entry
            if metric in fdr_set:
                pvals[metric] = test["p_value"]
            else:
                entry["q_bh"] = None
                entry["reject_fdr_05"] = None
                entry["fdr_family"] = "nuisance (not corrected)"
        if not per_metric:
            continue
        for metric, adj in benjamini_hochberg(pvals).items():
            per_metric[metric]["q_bh"] = adj["q"]
            per_metric[metric]["reject_fdr_05"] = adj["reject"]
            per_metric[metric]["fdr_family"] = f"{family} (m={len(pvals)})"
        out[f"{treatment}_vs_{comparator}"] = {
            "treatment": treatment, "comparator": comparator, "role": role, "family": family,
            "metrics": per_metric}
    return out


def neutral_matching(rewrites: Sequence[dict], *, tolerance: Optional[float]) -> dict:
    by_source: Dict[str, Dict[str, float]] = {}
    for r in rewrites:
        if r.get("status") == "ok" and r.get("edit_distance_to_original") is not None:
            by_source.setdefault(r["source_id"], {})[r["condition"]] = r["edit_distance_to_original"]
    low = [d["low"] for d in by_source.values() if "low" in d]
    neutral = [d["neutral"] for d in by_source.values() if "neutral" in d]
    gaps = [abs(d["low"] - d["neutral"]) for d in by_source.values() if "low" in d and "neutral" in d]
    within = (sum(1 for g in gaps if g <= tolerance) / len(gaps)) if gaps and tolerance is not None else None
    return {"n_sources": len(by_source),
            "low": {"mean": mean(low), "sd": sd(low), "median": median(low), "n": len(low)},
            "neutral": {"mean": mean(neutral), "sd": sd(neutral), "median": median(neutral), "n": len(neutral)},
            "abs_gap": {"mean": mean(gaps), "median": median(gaps), "p90": percentile(gaps, 90), "n": len(gaps)},
            "tolerance": tolerance, "fraction_within_tolerance": within}


def listener_analysis(choices: Sequence[dict], *, n_boot: int, seed: int) -> Dict[str, dict]:
    """Share choosing the treatment as more limited, per listener and pooled.

    A row is a valid response when ``ok`` is true. A response may be an
    ABSTENTION (human ``cannot_tell``): it has no ``chosen_condition``. The
    share and its CIs are computed over decided responses, and every table
    also reports the abstention rate and the conservative share over ALL valid
    responses (an abstention counted as not choosing the treatment), so a
    rater who abstains on most items cannot produce a 100% share.
    """
    out: Dict[str, dict] = {}
    for pair in sorted({r["pair"] for r in choices}):
        rows = [r for r in choices if r["pair"] == pair and r.get("ok")]
        errors = sum(1 for r in choices if r["pair"] == pair and not r.get("ok"))
        if not rows:
            continue
        treatment, comparator = rows[0]["treatment"], rows[0]["comparator"]
        listeners = sorted({r["listener"] for r in rows})

        def is_human(r: dict) -> bool:
            return r.get("listener_family") == "human" or str(r["listener"]).startswith("human:")

        # Pooled rows never mix model listeners with human raters: the frozen
        # criterion is about the model instruments, and the humans are the
        # independent check on them.
        pooled = {"all_models": [r for r in rows if not is_human(r)]}
        if any(is_human(r) for r in rows):
            pooled["all_humans"] = [r for r in rows if is_human(r)]
        per_listener: Dict[str, dict] = {}
        for listener in listeners + [k for k, v in pooled.items() if v]:
            subset = pooled[listener] if listener in pooled else [r for r in rows if r["listener"] == listener]
            decided = [r for r in subset if r.get("chosen_condition")]
            n_abstain = len(subset) - len(decided)
            by_bundle: Dict[str, List[int]] = {}
            game_of: Dict[str, str] = {}
            for r in decided:
                by_bundle.setdefault(r["bundle_id"], []).append(1 if r["chose_treatment"] else 0)
                game_of[r["bundle_id"]] = str(r.get("game_id"))
            clusters: Dict[str, List[float]] = {}
            for b, votes in by_bundle.items():
                clusters.setdefault(game_of[b], []).append(sum(votes) / len(votes))
            boot = cluster_bootstrap(clusters, n_boot=n_boot, seed=seed)
            boot.pop("draws", None)
            k = sum(1 for r in decided if r["chose_treatment"])
            p, w_lo, w_hi = wilson_interval(k, len(decided))
            by_order = {}
            for order in ("ab", "ba"):
                o = [r for r in decided if r["order"] == order]
                if o:
                    by_order[order] = {"n": len(o),
                                       "share_treatment": sum(1 for r in o if r["chose_treatment"]) / len(o)}
            comps_by_cluster: Dict[str, List[Tuple[str, str]]] = {}
            for r in decided:
                loser = comparator if r["chosen_condition"] == treatment else treatment
                comps_by_cluster.setdefault(str(r.get("game_id")), []).append((r["chosen_condition"], loser))
            bt = bradley_terry_bootstrap(comps_by_cluster, treatment, comparator, n_boot=n_boot, seed=seed)
            ci_low, ci_high = boot["ci_low"], boot["ci_high"]
            per_listener[listener] = {
                "n_responses": len(subset), "n_decided": len(decided), "n_abstained": n_abstain,
                "abstention_rate": n_abstain / len(subset) if subset else None,
                "n_bundles": len(by_bundle), "n_games": boot["n_clusters"],
                "share_treatment_more_limited": p,
                "share_treatment_all_responses": k / len(subset) if subset else None,
                "wilson_ci": [w_lo, w_hi],
                "cluster_bootstrap_ci": [ci_low, ci_high],
                "ci_lower_above_half": (ci_low is not None and ci_low > 0.5),
                "ci_covers_half": (ci_low is not None and ci_high is not None and ci_low <= 0.5 <= ci_high),
                "by_order": by_order, "bradley_terry": bt,
                "confidence": {c: sum(1 for r in decided if r.get("confidence") == c)
                               for c in ("high", "medium", "low")},
            }
        # Agreement between human raters (both decided on the same bundle+order).
        humans = sorted({r["listener"] for r in rows if is_human(r)})
        agreement = {}
        for i, a in enumerate(humans):
            for b in humans[i + 1:]:
                by_key = {}
                for r in rows:
                    if r["listener"] in (a, b) and r.get("chosen_condition"):
                        by_key.setdefault((r["bundle_id"], r.get("order")), {})[r["listener"]] = r["chosen_condition"]
                pairs = [(v[a], v[b]) for v in by_key.values() if a in v and b in v]
                agreement[f"{a} vs {b}"] = {"n": len(pairs),
                                            "agreed": sum(1 for x, y in pairs if x == y),
                                            "kappa": cohen_kappa(pairs)}
        out[pair] = {"treatment": treatment, "comparator": comparator, "n_invalid_responses": errors,
                     "listeners": per_listener, "human_agreement": agreement}
    return out


def politeness_analysis(drift: Sequence[dict], *, n_boot: int, seed: int) -> dict:
    conditions = sorted({r["condition"] for r in drift if r["condition"] != "original"})
    per_condition = {}
    for c in conditions:
        rows = [r for r in drift if r["condition"] == c and r.get("preserved") is not None]
        k = sum(1 for r in rows if r["preserved"])
        p, lo, hi = wilson_interval(k, len(rows))
        labels: Dict[str, int] = {}
        for r in rows:
            labels[r["label"]] = labels.get(r["label"], 0) + 1
        per_condition[c] = {"n": len(rows), "preserved": k, "preservation_rate": p,
                            "wilson_ci": [lo, hi], "label_distribution": labels}
    contrast = None
    by_source: Dict[str, Dict[str, int]] = {}
    game_of: Dict[str, str] = {}
    for r in drift:
        if r["condition"] in ("low", "neutral") and r.get("preserved") is not None:
            by_source.setdefault(r["source_id"], {})[r["condition"]] = int(r["preserved"])
            game_of[r["source_id"]] = str(r.get("game_id"))
    clusters: Dict[str, List[float]] = {}
    for sid, d in by_source.items():
        if "low" in d and "neutral" in d:
            clusters.setdefault(game_of[sid], []).append(d["low"] - d["neutral"])
    if clusters:
        boot = cluster_bootstrap(clusters, n_boot=n_boot, seed=seed)
        boot.pop("draws", None)
        contrast = {"n_sources": sum(len(v) for v in clusters.values()), "estimate": boot["estimate"],
                    "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                    "p_sign_flip": cluster_sign_flip_test(clusters, seed=seed)["p_value"]}
    return {"per_condition": per_condition, "low_minus_neutral_preservation": contrast}


def _criterion(name: str, passed: Optional[bool], detail=None, note: str = "") -> dict:
    """``passed`` is True, False, or None = could not be evaluated (never a pass)."""
    return {"criterion": name, "passed": passed, "detail": detail, "note": note}


def frozen_criteria(report: dict, frozen: Optional[dict]) -> List[dict]:
    """Confirmatory criteria (PLAN 9.1, 12, 13) evaluated against the frozen config.

    Every criterion the config implies is listed, whether or not the data to
    evaluate it was supplied: a missing listener file or a metric with no
    p-value yields ``passed=None`` ("not evaluated"), which the verdict counts
    as not passed. Nothing passes by absence.
    """
    if not frozen:
        return []
    crit: List[dict] = []
    thresholds = frozen.get("range_effect_thresholds", {}) or {}
    primary = frozen.get("primary_range_metrics", list(PRIMARY_RANGE_METRICS))
    low_neu = (report.get("range") or {}).get("low_vs_neutral", {}).get("metrics", {})
    for metric in primary:
        m = low_neu.get(metric)
        threshold = thresholds.get(metric)
        name = f"Low vs Neutral: {metric} in expected direction, FDR q<.05, |dz|>={threshold}"
        if m is None:
            crit.append(_criterion(name, None, note="no feature data for this metric"))
            continue
        dz = m.get("cohens_dz")
        if threshold is None:
            crit.append(_criterion(name, None, m, note="no |d_z| threshold frozen for this metric"))
            continue
        if m.get("reject_fdr_05") is None or dz is None or m.get("direction_confirmed") is None:
            crit.append(_criterion(name, None, m, note="p-value, d_z or CI unavailable (too few games?)"))
            continue
        crit.append(_criterion(name, bool(m["direction_confirmed"] and m["reject_fdr_05"] is True
                                          and abs(dz) >= threshold), m))

    expected_listeners = list((frozen.get("models") or {}).get("listeners") or []) + ["all_models"]
    lst = (report.get("listener") or {}).get("low:neutral", {}).get("listeners", {})
    for listener in expected_listeners:
        row = lst.get(listener)
        name = (f"Listener {listener}: share choosing Low as more limited, "
                "cluster-bootstrap 95% CI lower bound > 0.50")
        if row is None:
            crit.append(_criterion(name, None, note="no listener data"))
        elif row["cluster_bootstrap_ci"][0] is None:
            crit.append(_criterion(name, None, row, note="CI unavailable (too few games?)"))
        else:
            crit.append(_criterion(name, bool(row["ci_lower_above_half"]),
                                   {"share": row["share_treatment_more_limited"],
                                    "share_all_responses": row["share_treatment_all_responses"],
                                    "abstention_rate": row["abstention_rate"],
                                    "ci": row["cluster_bootstrap_ci"]}))
    for listener, row in lst.items():
        if (listener not in expected_listeners and not listener.startswith("human:")
                and listener != "all_humans"):
            crit.append(_criterion(f"Listener {listener} is not in the frozen listener set", False, row))
    control = (report.get("listener") or {}).get("neutral:original", {}).get("listeners", {})
    for listener in expected_listeners:
        row = control.get(listener)
        name = (f"Listener {listener}: Neutral vs Original control, cluster-bootstrap 95% CI "
                "covers 0.50 (paraphrasing alone is not perceived as more limited)")
        if row is None:
            crit.append(_criterion(name, None, note="no neutral:original probe data"))
        elif row["cluster_bootstrap_ci"][0] is None:
            crit.append(_criterion(name, None, row, note="CI unavailable (too few games?)"))
        else:
            crit.append(_criterion(name, bool(row["ci_covers_half"]),
                                   {"share": row["share_treatment_more_limited"], "ci": row["cluster_bootstrap_ci"]}))

    bounds = frozen.get("equivalence_bounds", {}) or {}
    guard = (report.get("guardrails") or {}).get("low_vs_neutral", {}).get("metrics", {})
    for metric in GUARDRAIL_METRICS:
        m = guard.get(metric)
        eq = (m or {}).get("equivalence") or {}
        name = f"Low vs Neutral: {metric} equivalent within ±{bounds.get(metric)} (TOST)"
        if m is None:
            crit.append(_criterion(name, None, note="no guardrail data"))
        elif bounds.get(metric) is None:
            crit.append(_criterion(name, None, m, note="no equivalence bound frozen"))
        else:
            crit.append(_criterion(name, eq.get("equivalent"), m,
                                   note="" if eq.get("equivalent") is not None else "CI unavailable"))
    neu_orig = (report.get("range") or {}).get("neutral_vs_original", {}).get("metrics", {})
    for metric in primary:
        m = neu_orig.get(metric)
        eq = (m or {}).get("equivalence") or {}
        name = f"Neutral vs Original: {metric} equivalent within ±{bounds.get(metric)} (TOST)"
        if m is None:
            crit.append(_criterion(name, None, note="no feature data for this metric"))
        elif bounds.get(metric) is None:
            crit.append(_criterion(name, None, m, note="no equivalence bound frozen"))
        else:
            crit.append(_criterion(name, eq.get("equivalent"), m,
                                   note="" if eq.get("equivalent") is not None else "CI unavailable"))

    tol = report.get("neutral_matching") or {}
    min_frac = frozen.get("neutral_match_min_fraction")
    name = (f"Neutral matching: >= {min_frac:.0%} of sources within tolerance "
            f"{frozen.get('neutral_edit_distance_tolerance')}" if min_frac is not None
            else "Neutral matching: minimum fraction within tolerance")
    frac = tol.get("fraction_within_tolerance")
    if min_frac is None:
        crit.append(_criterion(name, None, tol, note="no minimum fraction frozen"))
    else:
        crit.append(_criterion(name, None if frac is None else frac >= min_frac, tol,
                               note="" if frac is not None else "no rewrites data"))
    for row in report.get("integrity") or []:
        crit.append(_criterion(f"Integrity: {row['check']}", row["passed"], row.get("detail"),
                               note=row.get("note", "")))
    return crit


def verdict(criteria: List[dict]) -> dict:
    passed = sum(1 for c in criteria if c["passed"] is True)
    failed = sum(1 for c in criteria if c["passed"] is False)
    unevaluated = sum(1 for c in criteria if c["passed"] is None)
    return {"n_criteria": len(criteria), "passed": passed, "failed": failed,
            "not_evaluated": unevaluated,
            "confirmed": bool(criteria) and failed == 0 and unevaluated == 0}


def integrity_checks(frozen: dict, *, rewrites: Optional[Sequence[dict]] = None,
                     features_path: Optional[Path] = None, listener_paths: Optional[Sequence[Path]] = None,
                     guardrails_path: Optional[Path] = None) -> List[dict]:
    """Does the data on disk come from the frozen pipeline?

    Compares the recorded prompt hashes, model handles and instrument versions
    in the data and its provenance side-cars with the frozen config, and
    recomputes the code hashes. Anything that cannot be checked is reported as
    not evaluated rather than assumed fine.
    """
    from .frozen import verify_hashes
    checks: List[dict] = []
    problems = verify_hashes(frozen)
    checks.append({"check": "current code matches the frozen hashes", "passed": not problems,
                   "detail": problems})
    hashes = frozen.get("hashes", {}) or {}
    models = frozen.get("models", {}) or {}
    if rewrites is not None:
        from .frozen import expected_template_hash
        bad = sorted({(r["condition"], r.get("prompt_template_sha256")) for r in rewrites
                      if expected_template_hash(frozen, r["condition"])
                      and r.get("prompt_template_sha256") != expected_template_hash(frozen, r["condition"])},
                     key=str)
        checks.append({"check": "every rewrite was generated under the frozen prompt template", "passed": not bad,
                       "detail": bad})
        conditions = sorted({r["condition"] for r in rewrites}, key=str)
        allowed = set(frozen.get("conditions") or [])
        checks.append({"check": f"rewrite conditions within the frozen design {sorted(allowed)}",
                       "passed": (set(conditions) <= allowed) if allowed else None, "detail": conditions})
        temps = sorted({r.get("generator_temperature") for r in rewrites}, key=str)
        frozen_temp = frozen.get("generation_temperature")
        checks.append({"check": f"rewrites generated at the frozen temperature {frozen_temp}",
                       "passed": (temps == [frozen_temp]) if frozen_temp is not None else None, "detail": temps})
        gens = sorted({r.get("generator_handle") or r.get("generator_model") for r in rewrites}, key=str)
        checks.append({"check": f"rewrites generated by the frozen generator {models.get('generator')!r}",
                       "passed": gens == [models.get("generator")] if gens else None, "detail": gens})
        judges = sorted({((r.get("gate_provenance") or {}).get("judge_handle")) for r in rewrites}, key=str)
        # A row without a judge handle is an un-provenanced rewrite: that is a FAIL, not a skip.
        checks.append({"check": f"rewrites gated by the frozen judge {models.get('judge')!r}",
                       "passed": judges == [models.get("judge")] if judges else None, "detail": judges})
    if features_path is not None:
        rows = read_jsonl(features_path)
        complete = [r for r in rows if r.get("complete")]
        lacking = [r["bundle_id"] for r in complete
                   if not r.get("syntax_available") or not r.get("lemmas_available")]
        checks.append({"check": "every feature row has spaCy syntax and lemma-based measures",
                       "passed": (not lacking) if complete else None,
                       "detail": {"rows_without": len(lacking), "of": len(complete)}})
        side = Path(features_path).with_name(Path(features_path).stem + "_provenance.json")
        if side.exists():
            prov = json.loads(side.read_text(encoding="utf-8"))
            inst = frozen.get("instruments", {}) or {}
            checks.append({"check": "features computed with the frozen MATTR window",
                           "passed": prov.get("mattr_window") == frozen.get("mattr_window"),
                           "detail": {"features": prov.get("mattr_window"), "frozen": frozen.get("mattr_window")}})
            spacy_prov = prov.get("spacy") or {}
            model_name = f"{spacy_prov.get('lang', '')}_{spacy_prov.get('model', '')}" if spacy_prov else None
            checks.append({"check": "features computed with the frozen spaCy model/version",
                           "passed": (bool(spacy_prov) and model_name == inst.get("spacy_model")
                                      and (inst.get("spacy_model_version") in (None, spacy_prov.get("model_version")))),
                           "detail": {"features": spacy_prov, "frozen": {"model": inst.get("spacy_model"),
                                                                          "version": inst.get("spacy_model_version")}}})
        else:
            checks.append({"check": "features provenance side-car present", "passed": None,
                           "note": f"{side.name} not found"})
    if guardrails_path is not None:
        side = Path(guardrails_path).with_name(Path(guardrails_path).stem + "_provenance.json")
        inst = frozen.get("instruments", {}) or {}
        if side.exists():
            prov = json.loads(side.read_text(encoding="utf-8"))
            lt = prov.get("languagetool") or {}
            if inst.get("languagetool_version"):
                checks.append({"check": "guardrails computed with the frozen LanguageTool version",
                               "passed": lt.get("version") == inst["languagetool_version"],
                               "detail": {"used": lt.get("version"), "frozen": inst["languagetool_version"]}})
            cola = prov.get("cola") or {}
            if inst.get("cola_model"):
                checks.append({"check": "guardrails computed with the frozen CoLA checkpoint",
                               "passed": (cola.get("model") == inst["cola_model"]
                                          and (not inst.get("cola_commit_hash")
                                               or cola.get("commit_hash") == inst["cola_commit_hash"])),
                               "detail": {"used": {"model": cola.get("model"), "commit": cola.get("commit_hash")},
                                          "frozen": {"model": inst["cola_model"], "commit": inst.get("cola_commit_hash")}}})
        else:
            checks.append({"check": "guardrails provenance side-car present", "passed": None,
                           "note": f"{side.name} not found"})
    for path in listener_paths or []:
        rows = read_jsonl(path)
        if rows and all(r.get("listener_family") == "human" or str(r.get("listener", "")).startswith("human:")
                        for r in rows):
            continue        # human worksheets carry no model provenance; nothing to check
        side = Path(path).with_name(Path(path).stem + "_provenance.json")
        if not side.exists():
            checks.append({"check": f"listener provenance for {Path(path).name}", "passed": None,
                           "note": "no provenance side-car for a model listener file"})
            continue
        prov = json.loads(side.read_text(encoding="utf-8"))
        checks.append({"check": f"{Path(path).name}: probe prompt is the frozen one",
                       "passed": prov.get("prompt_sha256") == hashes.get("listener_probe_sha256"),
                       "detail": prov.get("prompt_sha256")})
        used = sorted((prov.get("listeners") or {}).keys())
        checks.append({"check": f"{Path(path).name}: listeners are the frozen set",
                       "passed": used == sorted(models.get("listeners") or []), "detail": used})
    return checks


# ── rendering ──

def _contrast_table(block: dict, metrics: Sequence[str]) -> List[str]:
    lines = [f"### {block['treatment']} − {block['comparator']}  ({block['role']}; "
             f"n bundles = {next(iter(block['metrics'].values()))['n_bundles']}, "
             f"games = {next(iter(block['metrics'].values()))['n_games']})", "",
             "| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |",
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
    return lines + [""]


def render(report: dict) -> str:
    lines = ["# Proficiency-cue validation report", "",
             f"Generated {report['created_at']}. Frozen config: "
             f"{'yes' if report.get('frozen_config_used') else 'no (pilot / exploratory)'}.", ""]
    qc = report.get("generation_qc")
    if qc:
        lines += ["## Generation QC", "",
                  "| condition | n | first-pass | retried | treatment unavailable | infrastructure unavailable | fallback | mean attempts |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for c, q in qc.items():
            lines.append(f"| {c} | {q['n']} | {q['first_pass_success']} | {q['needed_retry']} | "
                         f"{q['treatment_unavailable']} | {q.get('infrastructure_unavailable', 0)} | "
                         f"{q['fallback_to_original']} | {fmt(q['mean_attempts'], 2)} |")
        lines.append("")
    nm = report.get("neutral_matching")
    if nm and nm.get("n_sources"):
        lines += ["## Neutral matching (normalized token edit distance to the original)", "",
                  "| condition | n | mean | SD | median |", "| --- | ---: | ---: | ---: | ---: |",
                  f"| low | {nm['low']['n']} | {fmt(nm['low']['mean'])} | {fmt(nm['low']['sd'])} | {fmt(nm['low']['median'])} |",
                  f"| neutral | {nm['neutral']['n']} | {fmt(nm['neutral']['mean'])} | {fmt(nm['neutral']['sd'])} | {fmt(nm['neutral']['median'])} |",
                  "", f"|low − neutral| gap: mean {fmt(nm['abs_gap']['mean'])}, median "
                  f"{fmt(nm['abs_gap']['median'])}, p90 {fmt(nm['abs_gap']['p90'])}"
                  + (f"; within tolerance {nm['tolerance']}: {fmt(nm['fraction_within_tolerance'], 2)}"
                     if nm.get('tolerance') is not None else ""), ""]
    if report.get("range"):
        lines += ["## Family A: linguistic range", ""]
        for block in report["range"].values():
            lines += _contrast_table(block, list(RANGE_METRICS) + list(NUISANCE_METRICS))
    if report.get("guardrails"):
        lines += ["## Family B: grammar / acceptability guardrail (equivalence)", ""]
        for block in report["guardrails"].values():
            lines += _contrast_table(block, GUARDRAIL_METRICS)
    if report.get("listener"):
        lines += ["## Family C: listener-side perception", ""]
        for pair, block in report["listener"].items():
            lines += [f"### {block['treatment']} vs {block['comparator']}: share choosing "
                      f"**{block['treatment']}** as more limited"
                      + (f"  ({block['n_invalid_responses']} invalid responses excluded)"
                         if block.get("n_invalid_responses") else ""), "",
                      "| listener | responses | abstained | bundles | share (decided) | share (all) | "
                      "Wilson 95% | cluster-bootstrap 95% | order ab | order ba | BT log-strength Δ [95%] |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for listener, row in block["listeners"].items():
                bt = row["bradley_terry"]
                ab = row["by_order"].get("ab", {}).get("share_treatment")
                ba = row["by_order"].get("ba", {}).get("share_treatment")
                lines.append(f"| {listener} | {row['n_responses']} | {row['n_abstained']} | {row['n_bundles']} | "
                             f"{fmt(row['share_treatment_more_limited'])} | "
                             f"{fmt(row['share_treatment_all_responses'])} | "
                             f"[{fmt(row['wilson_ci'][0])}, {fmt(row['wilson_ci'][1])}] | "
                             f"[{fmt(row['cluster_bootstrap_ci'][0])}, {fmt(row['cluster_bootstrap_ci'][1])}] | "
                             f"{fmt(ab)} | {fmt(ba)} | {fmt(bt['estimate'], 2)} "
                             f"[{fmt(bt['ci_low'], 2)}, {fmt(bt['ci_high'], 2)}] |")
            for name, agr in (block.get("human_agreement") or {}).items():
                lines.append(f"Human agreement {name}: {agr['agreed']}/{agr['n']}, "
                             f"Cohen's κ = {fmt(agr['kappa'])}")
            lines.append("")
    pol = report.get("politeness")
    if pol and pol.get("per_condition"):
        lines += ["## Pragmatic guardrail: politeness-label preservation (diagnostic)", "",
                  "| condition | n | preserved | rate | Wilson 95% |", "| --- | ---: | ---: | ---: | ---: |"]
        for c, row in pol["per_condition"].items():
            lines.append(f"| {c} | {row['n']} | {row['preserved']} | {fmt(row['preservation_rate'])} | "
                         f"[{fmt(row['wilson_ci'][0])}, {fmt(row['wilson_ci'][1])}] |")
        d = pol.get("low_minus_neutral_preservation")
        if d:
            lines += ["", f"Low − Neutral preservation: Δ = {fmt(d['estimate'])} "
                      f"[{fmt(d['ci_low'])}, {fmt(d['ci_high'])}], p = {fmt(d['p_sign_flip'])}"]
        lines.append("")
    if report.get("criteria"):
        v = report.get("verdict") or verdict(report["criteria"])
        lines += ["## Confirmatory criteria (frozen config)", "",
                  f"**Verdict: {'CONFIRMED' if v['confirmed'] else 'NOT CONFIRMED'}** — "
                  f"{v['passed']} passed, {v['failed']} failed, {v['not_evaluated']} not evaluated "
                  f"(a criterion that could not be evaluated never counts as passed).", "",
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
    p.add_argument("--politeness", type=Path, default=None)
    p.add_argument("--listener", type=Path, action="append", default=None,
                   help="listener choices JSONL (repeatable; human rows can be added)")
    p.add_argument("--rewrites", type=Path, default=None)
    p.add_argument("--frozen", type=Path, default=None, help="frozen_config.json (default: data/)")
    p.add_argument("--no-frozen", action="store_true", help="ignore any frozen config (pilot)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out", type=Path, default=None, help="Markdown report")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=main)


def main(args) -> int:
    frozen = None if args.no_frozen else load_frozen(args.frozen)
    if args.frozen and frozen is None:
        print(f"no frozen config at {args.frozen}", file=sys.stderr)
        return 2
    report: dict = {"created_at": now_iso(), "frozen_config_used": bool(frozen),
                    "frozen_config_path": str(args.frozen) if frozen else None,
                    "n_boot": args.n_boot, "seed": args.seed}
    rewrites = None
    if args.rewrites:
        rewrites = read_jsonl(args.rewrites)
        report["generation_qc"] = generation_qc(rewrites)
        report["neutral_matching"] = neutral_matching(
            rewrites, tolerance=(frozen or {}).get("neutral_edit_distance_tolerance"))
    if args.features:
        features = read_jsonl(args.features)
        report["range"] = contrast_analysis(features, list(RANGE_METRICS) + list(NUISANCE_METRICS),
                                            frozen=frozen, n_boot=args.n_boot, seed=args.seed,
                                            family="linguistic_range")
    if args.guardrails:
        guard = read_jsonl(args.guardrails)
        report["guardrails"] = contrast_analysis(guard, list(GUARDRAIL_METRICS), frozen=frozen,
                                                 n_boot=args.n_boot, seed=args.seed, family="guardrail")
    if args.listener:
        choices = [r for path in args.listener for r in read_jsonl(path)]
        report["listener"] = listener_analysis(choices, n_boot=args.n_boot, seed=args.seed)
    if args.politeness:
        report["politeness"] = politeness_analysis(read_jsonl(args.politeness), n_boot=args.n_boot,
                                                   seed=args.seed)
    if frozen:
        report["integrity"] = integrity_checks(frozen, rewrites=rewrites, features_path=args.features,
                                               listener_paths=args.listener, guardrails_path=args.guardrails)
    report["criteria"] = frozen_criteria(report, frozen)
    if report["criteria"]:
        report["verdict"] = verdict(report["criteria"])
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
    return 0
