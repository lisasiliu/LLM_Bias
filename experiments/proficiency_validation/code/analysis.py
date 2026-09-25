"""Paired statistical analysis and reproducible validation reporting.

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
    wilson_interval, proportion_bootstrap)
from .protocol import assert_unique

from .common import (
    CONTRASTS, GUARDRAIL_METRICS, NUISANCE_METRICS, RANGE_METRICS,
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
            boot = cluster_bootstrap(clusters, n_boot=n_boot, seed=seed)
            test = cluster_sign_flip_test(clusters, seed=seed,
                    n_perm=(frozen or {}).get("statistics", {}).get("n_perm", 5000))
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
    if frozen:
        # Low and High are planned together, so their range tests share one BH family.
        pvals = {(contrast, m): values["p_sign_flip"] for contrast, block in out.items()
                 if contrast in ("low_vs_neutral", "high_vs_neutral")
                 for m, values in block["metrics"].items() if m in fdr_set}
        for (contrast, m), adj in benjamini_hochberg(pvals).items():
            out[contrast]["metrics"][m].update(q_bh=adj["q"], reject_fdr_05=adj["reject"],
                                             fdr_family=f"{family}: planned Low/High contrasts")
    return out


def neutral_matching(rewrites: Sequence[dict], *, tolerance: Optional[float], treatment="low") -> dict:
    by_source: Dict[str, Dict[str, float]] = {}
    for r in rewrites:
        if r.get("status") == "ok" and r.get("edit_distance_to_original") is not None:
            by_source.setdefault(r["source_id"], {})[r["condition"]] = r["edit_distance_to_original"]
    low = [d[treatment] for d in by_source.values() if treatment in d]
    neutral = [d["neutral"] for d in by_source.values() if "neutral" in d]
    gaps = [abs(d[treatment] - d["neutral"]) for d in by_source.values() if treatment in d and "neutral" in d]
    within = (sum(1 for g in gaps if g <= tolerance) / len(gaps)) if gaps and tolerance is not None else None
    return {"n_sources": len(by_source), "treatment_condition": treatment,
            "treatment": {"mean": mean(low), "sd": sd(low), "median": median(low), "n": len(low)},
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
    assert_unique(choices, ("bundle_id", "pair", "listener", "order"))
    out: Dict[str, dict] = {}
    for pair in sorted({r["pair"] for r in choices}):
        def is_human(r: dict) -> bool:
            return r.get("listener_family") == "human" or str(r["listener"]).startswith("human:")
        valid_orders = {}
        for r in choices:
            if r["pair"] == pair and r.get("ok") and not is_human(r):
                valid_orders.setdefault((r["bundle_id"], r["listener"]), set()).add(r["order"])
        # Supplementary human rows retain unresolved answers in their denominators.
        # Model comparisons require valid responses in both presentation orders.
        rows = [r for r in choices if r["pair"] == pair and (is_human(r) or
                (r.get("ok") and valid_orders.get((r["bundle_id"], r["listener"])) == {"ab", "ba"}))]
        errors = sum(1 for r in choices if r["pair"] == pair and not r.get("ok") and not is_human(r))
        if not rows:
            continue
        treatment, comparator = rows[0]["treatment"], rows[0]["comparator"]
        listeners = sorted({r["listener"] for r in rows})

        # Pool model comparisons and supplementary human observations separately.
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
            boot = proportion_bootstrap(clusters, n_boot=n_boot, seed=seed)
            eq_boot = proportion_bootstrap(clusters, n_boot=n_boot, seed=seed, alpha=.10)
            boot.pop("draws", None)
            k = sum(1 for r in decided if r["chose_treatment"])
            p, w_lo, w_hi = wilson_interval(k, len(decided))
            by_order = {}
            for order in ("ab", "ba"):
                o = [r for r in decided if r["order"] == order]
                if o:
                    by_order_bundle = {}
                    for r in o:
                        by_order_bundle.setdefault((str(r.get("game_id")), r["bundle_id"]), []).append(int(r["chose_treatment"]))
                    order_clusters = {}
                    for (g, b), vals in by_order_bundle.items():
                        order_clusters.setdefault(g, []).append(mean(vals))
                    order_ci = proportion_bootstrap(order_clusters, n_boot=n_boot, seed=seed)
                    order_eq = proportion_bootstrap(order_clusters, n_boot=n_boot, seed=seed, alpha=.10)
                    by_order[order] = {"n": len(o),
                                       "share_treatment": order_ci["estimate"], "n_games": order_ci["n_clusters"],
                                       "n_bundles": len(by_order_bundle),
                                       "ci95": [order_ci["ci_low"], order_ci["ci_high"]],
                                       "ci90": [order_eq["ci_low"], order_eq["ci_high"]]}
            comps_by_cluster: Dict[str, List[Tuple[str, str]]] = {}
            for r in decided:
                loser = comparator if r["chosen_condition"] == treatment else treatment
                comps_by_cluster.setdefault(str(r.get("game_id")), []).append((r["chosen_condition"], loser))
            bt = bradley_terry_bootstrap(comps_by_cluster, treatment, comparator, n_boot=n_boot, seed=seed)
            ci_low, ci_high = boot["ci_low"], boot["ci_high"]
            conservative_by_bundle = {}
            for r in subset:
                score = int(r["chose_treatment"]) if r.get("chosen_condition") else int(treatment == "high")
                conservative_by_bundle.setdefault((str(r.get("game_id")), r["bundle_id"]), []).append(score)
            conservative_clusters = {}
            for (g, b), scores in conservative_by_bundle.items():
                conservative_clusters.setdefault(g, []).append(mean(scores))
            conservative = proportion_bootstrap(conservative_clusters, n_boot=n_boot, seed=seed)
            per_listener[listener] = {
                "n_responses": len(subset), "n_decided": len(decided), "n_abstained": n_abstain,
                "n_cannot_tell": sum(1 for r in subset if r.get("cannot_tell")),
                "n_blank_or_unrecognised": sum(1 for r in subset if not r.get("ok")),
                "abstention_rate": n_abstain / len(subset) if subset else None,
                "n_bundles": len(by_bundle), "n_games": boot["n_clusters"],
                "min_raters_per_item": min((len({r["listener"] for r in subset if r["bundle_id"] == b})
                                            for b in {r["bundle_id"] for r in subset}), default=0),
                "share_treatment_more_limited": boot["estimate"],
                "share_treatment_raw_votes": p,
                "share_treatment_all_responses": k / len(subset) if subset else None,
                "wilson_ci": [w_lo, w_hi],
                "cluster_bootstrap_ci": [ci_low, ci_high],
                "equivalence_ci90": [eq_boot["ci_low"], eq_boot["ci_high"]],
                "interval_method": boot["interval_method"],
                "conservative_ci95": [conservative["ci_low"], conservative["ci_high"]],
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


def _criterion(name: str, passed: Optional[bool], detail=None, note: str = "") -> dict:
    """``passed`` is True, False, or None = could not be evaluated (never a pass)."""
    return {"criterion": name, "passed": passed, "detail": detail, "note": note}


def frozen_criteria(report: dict, frozen: Optional[dict]) -> List[dict]:
    from .confirmatory import criteria
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
    from .frozen import verify_hashes, expected_template_hash
    from .protocol import validate_design, verify_artifact, cohort, assert_unique, sha256_file
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
        if rows and all(str(r.get("listener", "")).startswith("human:") for r in rows):
            check(f"{Path(path).name}: human perception passages match analyzed stimuli",
                  bool(feature_prov) and all(r.get("assembled_sha256") == feature_prov.get("assembled_sha256")
                      and expected.get(r["bundle_id"]) == r.get("game_id") for r in rows))
            continue
        prov, errors = verify_artifact(path, frozen, split)
        check(f"{Path(path).name}: probe provenance", not errors, errors)
        if prov is None:
            continue
        used = {r.get("listener") for r in rows}
        check(f"{Path(path).name}: listener identities and prompts match freeze",
              used <= set(frozen["models"]["listeners"])
              and prov.get("prompt_sha256") == frozen["hashes"]["listener_probe_sha256"]
              and all((prov.get("listeners", {}).get(h) or {}).get("identity") == frozen["models"]["identities"][h] for h in used))
        check(f"{Path(path).name}: same passages/cohort", bool(feature_prov)
              and prov.get("input_sha256", {}).get("assembled") == feature_prov.get("assembled_sha256")
              and all(expected.get(r["bundle_id"]) == r.get("game_id") for r in rows))
    model_rows = [r for p in listener_paths or [] for r in read_jsonl(p)
                  if not str(r.get("listener", "")).startswith("human:")]
    complete = {(r["bundle_id"], r["condition"]) for r in feature_rows if r.get("complete")}
    required = {(b, f"{a}:{c}", h, o) for b in expected
                for a, c in [(t, "neutral") for t in ("low", "high") if t in conditions] + [("neutral", "original")]
                if (b, a) in complete and (b, c) in complete
                for h in frozen["models"]["listeners"] for o in ("ab", "ba")}
    actual = {(r["bundle_id"], r["pair"], r["listener"], r["order"]) for r in model_rows}
    check("every eligible model probe has both presentation outcomes recorded",
          bool(required) and actual == required and len(actual) == len(model_rows))
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
    lines = ["# Proficiency validation report", "",
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
    if report.get("human_quality_criteria"):
        lines += ["## Supplementary human observations", "",
                  "Supplementary observations are reported separately from the automated validation criteria.", "",
                  "| Check | Result |", "| --- | --- |"]
        for row in report["human_quality_criteria"]:
            lines.append(f"| {row['criterion']} | {row['passed']} |")
        lines.append("")
    return "\n".join(lines)


def add_parser(sub) -> None:
    p = sub.add_parser("analyze", help="planned contrasts, guardrails, listener perception, report")
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--guardrails", type=Path, default=None)
    p.add_argument("--listener", type=Path, action="append", default=None,
                   help="listener choices JSONL (repeatable; human rows can be added)")
    p.add_argument("--rewrites", type=Path, default=None)
    p.add_argument("--split", choices=["pilot", "set_a", "set_b"], default="pilot")
    p.add_argument("--human-qc-key", type=Path)
    p.add_argument("--human-qc", type=Path, nargs="+")
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
    if frozen:
        from .frozen import enforce_frozen
        from .protocol import cohort
        if args.split not in ("set_a", "set_b"):
            print("a frozen analysis needs --split set_a or set_b; for pilot diagnostics pass --no-frozen",
                  file=sys.stderr)
            return 2
        frozen = enforce_frozen(args.frozen)
        selected = cohort(frozen, args.split)
        args.n_boot, args.seed = frozen["statistics"]["n_boot"], frozen["statistics"]["seed"]
    report: dict = {"created_at": now_iso(), "frozen_config_used": bool(frozen),
                    "frozen_config_path": str(args.frozen) if frozen else None,
                    "n_boot": args.n_boot, "seed": args.seed}
    if frozen:
        report["expected_bundles"] = len(selected)
    rewrites = None
    if args.rewrites:
        rewrites = read_jsonl(args.rewrites)
        report["generation_qc"] = generation_qc(rewrites)
        report["neutral_matching"] = neutral_matching(
            rewrites, tolerance=(frozen or {}).get("neutral_edit_distance_tolerance"))
        report["neutral_matching_by_condition"] = {c: neutral_matching(rewrites, treatment=c,
            tolerance=(frozen or {}).get("neutral_edit_distance_tolerance")) for c in ("low", "high")}
    if args.features:
        features = read_jsonl(args.features)
        report["range"] = contrast_analysis(features, list(RANGE_METRICS) + list(NUISANCE_METRICS),
                                            frozen=frozen, n_boot=args.n_boot, seed=args.seed,
                                            family="linguistic_range")
    if args.guardrails:
        guard = read_jsonl(args.guardrails)
        report["guardrails"] = contrast_analysis(guard, list(GUARDRAIL_METRICS), frozen=frozen,
                                                 n_boot=args.n_boot, seed=args.seed, family="guardrail")
        report["absolute_quality"] = {}
        for c in (frozen or {}).get("conditions", ("original", "neutral", "low", "high")):
            valid = [r for r in guard if r["condition"] == c and r.get("grammar_errors_per_100_words") is not None
                     and r.get("cola_mean") is not None]
            report["absolute_quality"][c] = {"n_games": len({r["game_id"] for r in valid}),
                "n_bundles": len(valid), "grammar_mean": mean([r["grammar_errors_per_100_words"] for r in valid]),
                "cola_mean": mean([r["cola_mean"] for r in valid])}
    if args.listener:
        choices = [r for path in args.listener for r in read_jsonl(path)]
        report["listener"] = listener_analysis(choices, n_boot=args.n_boot, seed=args.seed)
    if frozen:
        report["integrity"] = integrity_checks(frozen, rewrites=rewrites, features_path=args.features,
                                               listener_paths=args.listener, guardrails_path=args.guardrails,
                                               rewrites_path=args.rewrites, split=args.split)
        if args.human_qc_key and args.human_qc and rewrites:
            from .human_validation import quality_criteria
            key = json.loads(args.human_qc_key.read_text())
            annotations = [r for p in args.human_qc for r in read_jsonl(p)]
            report["human_quality_criteria"], report["human_quality"] = quality_criteria(key, annotations, frozen, rewrites)
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
    return 1 if frozen and not report.get("verdict", {}).get("confirmed") else 0
