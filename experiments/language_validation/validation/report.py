"""Source audit report and pilot report rendering (VALIDATION.md and its JSON)."""
from __future__ import annotations
from experiments.language_validation.rewrite.runtime import read_metadata, update_metadata
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import re
from components.helpers import SourceRecord, read_jsonl
from components.judges.inverse_predicate_parser import PARSEABLE_PREDICATES
from components.metrics.text_normalization import split_sentences, word_count
from components.stats import mean, median
from experiments.language_validation.validation.analysis import listener_analysis, passage_retention, rewrite_exposure, exposure_lines
from experiments.language_validation.validation.sources import build_bundles, assemble_condition_texts, matched_complete
from experiments.language_validation.rewrite.runtime import condition_label, fmt, ordered_conditions, pair_label, sha256_file
from experiments.language_validation.rewrite.runtime import assert_unique
from experiments.language_validation.rewrite.gate import generation_qc, failure_cause
from components.helpers import write_jsonl, treatment_evidence
from experiments.language_validation.validation.sources import load_sources, audit
from experiments.language_validation.rewrite.runtime import DATA_DIR, EXPERIMENT_DIR, dump_json, now_iso
from experiments.language_validation.validation.measurements import compute_features, load_nlp_or_none, verify_guardrail_materials


def listener_files(measured, stem):
    """Every listener output for one construct: the run's own file plus files added for other
    listener models (``<stem>_<model>.jsonl``), in a fixed order."""
    measured = Path(measured)
    return sorted(p for p in measured.glob(f"{stem}*.jsonl") if not p.name.endswith(".partial"))


def read_listener_rows(measured, stem):
    return [r for path in listener_files(measured, stem) for r in read_jsonl(path)]


def measured_sampling(data_dir, pilot_dir, rewrites, errors):
    """Resolve the actual listener input through content hashes, then reconstruct it."""
    feature = read_metadata(pilot_dir / "validation/linguistic_features.jsonl")
    listener = read_metadata(pilot_dir / "validation/range_listener.jsonl")
    expected_bundles = feature.get("input_sha256", {}).get("bundles")
    expected_assembled = listener.get("input_sha256", {}).get("assembled")
    bundles_path = next((p for p in (pilot_dir / "validation/bundles.jsonl", data_dir / "bundles.jsonl")
                         if p.exists() and sha256_file(p) == expected_bundles), None)
    assembled_path = pilot_dir / "validation/linguistic_features_assembled.jsonl"
    if (not bundles_path or not assembled_path.exists() or not expected_assembled
            or sha256_file(assembled_path) != expected_assembled
            or feature.get("assembled_sha256") != expected_assembled):
        errors.append("listener passage selection could not be verified through the artifact hashes")
        return {"selection": "unverified", "verified": False}
    bundles = read_jsonl(bundles_path)
    assembled = read_jsonl(assembled_path)
    conditions = sorted({r["condition"] for r in assembled})
    reconstructed = assemble_condition_texts(bundles, rewrites, conditions)
    def core(rows):
        assert_unique(rows, ("bundle_id", "condition"))
        return {(r["bundle_id"], r["condition"]): (r.get("text"), r["complete"], r["game_id"], r["speaker"])
                for r in rows}
    if core(reconstructed) != core(assembled):
        errors.append("listener passages do not reconstruct from the recorded rewrites and source selection")
        return {"selection": "unverified", "verified": False}
    return {"selection": sorted({b.get("selection", "prefix") for b in bundles}), "verified": True,
            "windows": sorted({str(b.get("window")) for b in bundles}),
            "bundles_sha256": expected_bundles, "assembled_sha256": expected_assembled}


def summarize_pilot(data_dir, pilot_dir):
    from components import stats as stats_module
    data_dir, pilot_dir = Path(data_dir), Path(pilot_dir)
    for measured in (pilot_dir / "validation", pilot_dir / "validation/whole_game"):
        guardrails = measured / "grammar_guardrails.jsonl"
        if guardrails.exists():
            verify_guardrail_materials(guardrails, measured / "linguistic_features_assembled.jsonl")
    sources = [SourceRecord.from_dict(r) for r in read_jsonl(data_dir / "sources.jsonl")]
    source_map = {s.source_id: s for s in sources}
    rows = read_jsonl(pilot_dir / "rewrite/rewrites.jsonl")
    assert_unique(rows, ("source_id", "condition"))
    conditions = sorted({r["condition"] for r in rows})
    expected = {(s, c) for s in source_map for c in conditions}
    if {(r["source_id"], r["condition"]) for r in rows} != expected:
        raise ValueError("pilot must record every source/condition outcome, including failures")
    for r in rows:
        s = source_map[r["source_id"]]
        if (r["original_text"] != s.original_nl or r["game_id"] != s.game_id
                or r["speaker"] != s.speaker_id or r["predicate"] != s.predicate_text):
            raise ValueError(f"pilot source mismatch: {r['source_id']}")
    by_key = {(r["source_id"], r["condition"]): r for r in rows}
    accepted = {c: {r["source_id"] for r in rows if r["condition"] == c and r["status"] == "ok"}
                for c in conditions}
    paired = accepted.get("low", set()) & accepted.get("neutral", set())
    by_group = {}
    for key in ("predicate_name", "phase", "game_id"):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        by_group[key] = {k: generation_qc(v) for k, v in sorted(groups.items())}
    reasons, http400 = {}, {}
    for c in conditions:
        counts, errors = Counter(), 0
        for r in rows:
            if r["condition"] != c:
                continue
            for a in r.get("attempts", []):
                reasons_a = (a.get("gate") or {}).get("reasons", [])
                counts.update(reasons_a)
                if re.search(r'\b400(?: Client Error| Bad Request)|"http_status": 400\b', json.dumps(a)):
                    errors += 1
        reasons[c], http400[c] = dict(counts.most_common()), errors
    coverage = {}
    for selection, minimum, maximum in (("prefix", 150, 250), ("whole_game", 150, 250),
                                       ("phase_stratified", 150, 250)):
        bundles = build_bundles(sources, selection=selection, window_min=minimum, window_max=maximum)
        assembled = assemble_condition_texts(bundles["bundles"], rows, ["original"] + conditions)
        matched = matched_complete(assembled, ["low", "neutral"])
        matched_high = matched_complete(assembled, ["high", "neutral"]) if "high" in conditions else []
        coverage[selection] = {"eligible": len(bundles["bundles"]), "insufficient": len(bundles["insufficient"]),
            "matched_low_neutral": len(matched), "matched_bundle_ids": matched,
            "matched_high_neutral": len(matched_high), "matched_high_bundle_ids": matched_high,
            "matched_games": len({r["game_id"] for r in assembled if r["bundle_id"] in matched}),
            "complete_by_condition": dict(Counter(r["condition"] for r in assembled if r["complete"])),
            "bundles": bundles["bundles"]}
    whole = build_bundles(sources, selection="whole_game", window_min=1)["bundles"]
    lengths = [b["word_count"] for b in whole]
    sentence_diagnostics = {}
    for c in ("original", "low", "neutral"):
        texts = [source_map[s].original_nl if c == "original" else by_key[(s, c)]["rewritten_text"]
                 for s in sorted(paired)]
        sentence_diagnostics[c] = {"n": len(texts), "mean_sentences": mean([len(split_sentences(t)) for t in texts]),
                                    "mean_words": mean([word_count(t) for t in texts])}
    retained = {}
    for name, ids in (("retained_low_neutral", paired), ("excluded_low_neutral", set(source_map) - paired)):
        ss = [source_map[s] for s in sorted(ids)]
        retained[name] = {"n": len(ss), "mean_original_words": mean([word_count(s.original_nl) for s in ss]),
                          "predicates": dict(Counter(s.predicate_name for s in ss)),
                          "games": dict(Counter(s.game_id for s in ss))}
    models = json.loads((pilot_dir / "manifest.json").read_text())["models"]
    inputs = {"sources": sha256_file(data_dir / "sources.jsonl")}
    provenance_errors = []
    from experiments.language_validation.rewrite.__main__ import audit_generation
    generation_errors = audit_generation(pilot_dir / "rewrite/rewrites.jsonl")
    provenance_errors.extend(generation_errors)
    measured_dir = pilot_dir / "validation"
    listener_paths = listener_files(measured_dir, "range_listener") + listener_files(measured_dir, "perceived_proficiency_listener")
    optional = [n for n in ("cefr_auxiliary.jsonl", "grammar_guardrails.jsonl") if (measured_dir / n).exists()]
    checked = [pilot_dir / "rewrite/rewrites.jsonl", measured_dir / "linguistic_features.jsonl", pilot_dir / "manifest.json"] \
        + listener_paths + [measured_dir / n for n in optional]
    for path in checked:
        name = path.name
        inputs[name] = sha256_file(path)
        prov = read_metadata(path) if name != "manifest.json" else {}
        if prov:
            if prov.get("output_sha256") != inputs[name]:
                provenance_errors.append(f"{name} output hash differs from manifest")
        elif name != "manifest.json":
            provenance_errors.append(f"{name} has no artifact provenance in the manifest")
    features = read_jsonl(measured_dir / "linguistic_features.jsonl")
    listeners = read_listener_rows(measured_dir, "range_listener")
    proficiency = read_listener_rows(measured_dir, "perceived_proficiency_listener")
    for path in listener_paths:
        recorded = read_metadata(path)
        if recorded and recorded.get("input_sha256", {}).get("assembled") != sha256_file(measured_dir / "linguistic_features_assembled.jsonl"):
            provenance_errors.append(f"{path.name}: assembled passage hash mismatch")
    cefr = None
    if "cefr_auxiliary.jsonl" in optional:
        from experiments.language_validation.validation.cefr import ordinal_analysis, verify_ratings, rating_errors
        cefr_prov, errors = verify_ratings(pilot_dir / "validation/cefr_auxiliary.jsonl", rewrites_path=pilot_dir / "rewrite/rewrites.jsonl")
        provenance_errors.extend(errors)
        if not errors:
            ratings = read_jsonl(pilot_dir / "validation/cefr_auxiliary.jsonl")
            cefr = ordinal_analysis(ratings)
            cefr["calibration"] = cefr_prov.get("calibration")
            cefr["calibration_status"] = cefr_prov["calibration_status"]
            passages = {(r["bundle_id"], r["condition"]): r for r in
                        read_jsonl(pilot_dir / "validation/linguistic_features_assembled.jsonl")}
            invalid = {}
            for row in ratings:
                if row.get("status") == "invalid_response":
                    passage = passages[(row["bundle_id"], row["condition"])]
                    invalid[f"{row['rater']}::{row['sample_id']}"] = rating_errors(
                        row.get("raw_response"), "\n\n".join(passage["utterances"]))
            cefr["invalid_response_diagnostics"] = invalid
    from experiments.language_validation.rewrite.fidelity import SEMANTIC_RUBRIC_VERSION, audit_accepted_alignments
    from components.rewriters.language_rewriter import PROMPT_VERSION
    rubric_versions = dict(Counter((r.get("gate_provenance") or {}).get("semantic_rubric_version", "unknown") for r in rows))
    rewrite_versions = dict(Counter(r.get("prompt_version", "unknown") for r in rows))
    sampling = measured_sampling(data_dir, pilot_dir, rows, provenance_errors)
    secondary = {}
    for c in conditions:
        verdicts = [r["secondary_judge"] for r in rows if r["condition"] == c and r["status"] == "ok" and r.get("secondary_judge")]
        if verdicts:
            secondary[c] = {"handle": verdicts[0].get("handle"), "family": verdicts[0].get("family"), "n": len(verdicts),
                            "agree": sum(v.get("ok") is True for v in verdicts),
                            "disagree": sum(v.get("ok") is False for v in verdicts),
                            "no_verdict": sum(v.get("ok") is None for v in verdicts)}
    calibration_path = pilot_dir / "validation" / "calibration" / "report.json"
    judge_calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    return {"input_sha256": inputs, "secondary_judge": secondary, "judge_calibration": judge_calibration, "directory": pilot_dir.name, "n_sources": len(sources), "n_rows": len(rows),
            "analysis_code_sha256": {name: sha256_file(Path(__file__).with_name(name + ".py"))
                                      for name in ("report", "analysis", "sources", "cefr")},
            "statistics_code_sha256": sha256_file(Path(stats_module.__file__)),
            "n_games": len({s.game_id for s in sources}), "generation_qc": generation_qc(rows),
            "by_group": by_group, "attempt_reasons": reasons, "attempts_containing_http400": http400,
            "coverage": coverage, "speaker_game_units": len(whole), "median_original_words": median(lengths),
            "matched_utterances": len(paired), "retained_composition": retained,
            "sentence_diagnostics": sentence_diagnostics,
            "measured_sampling": sampling, "passage_retention": passage_retention(features),
            "recorded_rubric_versions": rubric_versions,
            "requires_revalidation": set(rubric_versions) != {SEMANTIC_RUBRIC_VERSION} or set(rewrite_versions) != {PROMPT_VERSION},
            "parser_eligible_sources": sum(s.predicate_name in PARSEABLE_PREDICATES for s in sources),
            "models": models, "generation_commits": dict(Counter(r.get("git_commit") for r in rows)),
            "prompt_versions": dict(Counter(r.get("prompt_version") for r in rows)),
            "provenance_errors": provenance_errors,
            "generation_record_errors": generation_errors,
            "accepted_alignment_audit": audit_accepted_alignments(rows),
            "failure_causes": {c: dict(Counter(failure_cause(r) for r in rows if r["condition"] == c)) for c in conditions},
            "rewrite_exposure": rewrite_exposure(rows),
            "cefr": cefr,
            "perceived_proficiency": listener_analysis(proficiency, n_boot=2000, seed=101),
            "subtlex_measured": any(r.get("subtlex_mean") is not None for r in features),
            "guardrails_file_present": (pilot_dir / "validation/grammar_guardrails.jsonl").exists(),
            "listener": listener_analysis([r for r in listeners if r.get("construct", "range") == "range"], n_boot=2000, seed=101)}


def render_pilot(p):
    has_high = "high" in p["generation_qc"]
    lines = ["## Qwen pilot", "", "The pilot contains "
             f"{p['n_rows']} condition outcomes for {p['n_sources']} source utterances from {p['n_games']} games. "
             "Acceptance rates are the gate's recorded decisions.", "",
             "| Condition | Accepted | Total | Acceptance | Own failed attempts | Infrastructure unavailable | Accepted with parser abstaining |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c, q in p["generation_qc"].items():
        lines.append(f"| {condition_label(c)} | {q['final_success']} | {q['n']} | {fmt(q['final_success_rate'])} | "
                     f"{q['own_failed_attempts']} | {q['infrastructure_unavailable']} | "
                     f"{q.get('accepted_with_parser_abstention', 0)} |")
    lines += ["", "Parser abstaining: the inverse parser recovered no predicate from the rewrite although the "
              "original parsed, so the fidelity judge decided alone, as it does for predicates outside the parser "
              "grammar. Subtracting that column gives the acceptance count under the stricter no-verdict policy."]
    opinions = {c: q.get("second_opinion") or {} for c, q in p["generation_qc"].items()}
    if any(opinions.values()):
        lines += ["", "| Condition | Second opinion agrees | Conflicts (no verdict) | Unavailable (no verdict) |",
                  "| --- | ---: | ---: | ---: |"]
        lines += [f"| {condition_label(c)} | {o.get('agree', 0)} | {o.get('conflict', 0)} | {o.get('unavailable', 0)} |" for c, o in opinions.items()]
        lines += ["", "A second fidelity judge from another model family re-judged every parser-abstained candidate "
                  "the primary judge passed; only agreement counts as a pass."]
    lines += ["", f"Recorded rewrite prompt versions: {p['prompt_versions']}. "
              f"Recorded English fidelity versions: {p['recorded_rubric_versions']}."]
    if p["requires_revalidation"]:
        lines += ["These outputs have not been evaluated under the current generation and fidelity policy."]
    audit = p.get("accepted_alignment_audit", {})
    if audit.get("issues"):
        lines += ["", "### Alignment evidence recheck", "",
                  f"The current word-boundary check flags {len(audit['issues'])} of {audit['checked']} accepted records. "
                  "These records need a repeated fidelity evaluation. Passage tables use the recorded acceptance decisions.", "",
                  "| Source | Condition | Evidence errors |", "| --- | --- | --- |"]
        lines += [f"| {r['source_id']} | {condition_label(r['condition'])} | {', '.join(r['errors'])} |" for r in audit["issues"]]
    lines += ["", "| Condition | Recorded outcome cause | Sources |", "| --- | --- | ---: |"]
    for c, counts in p.get("failure_causes", {}).items():
        lines += [f"| {condition_label(c)} | {cause} | {n} |" for cause, n in counts.items() if cause != "accepted"]
    lines += ["", "Outcome causes distinguish candidate rejection, incomplete evaluation and transport failure.",
              "", "### Rewrite exposure", "",
              "Every condition is generated independently from the original. "
              "The table describes how much rewriting each accepted condition carries."]
    lines += exposure_lines(p.get("rewrite_exposure", {}))
    lines += ["Failure reasons below count attempt records, so one source can contribute more than once. "
              "Construction failures remain in the denominator.", "",
              "### Passage coverage", "",
              f"There are {p['speaker_game_units']} speaker-game transcripts. Their median original length is "
              f"{fmt(p['median_original_words'], 0)} words. Eligibility uses only original text. "
              "A matched transcript requires an accepted rewrite for every selected source in both conditions.", "",
              "| Selection | Eligible | Complete A | Complete B | Matched A–B | Matched games |"
              + (" Complete C | Matched C–B |" if has_high else ""),
              "| --- | ---: | ---: | ---: | ---: | ---: |" + (" ---: | ---: |" if has_high else "")]
    for name, q in p["coverage"].items():
        lines.append(f"| {name} | {q['eligible']} | {q['complete_by_condition'].get('low', 0)} | "
                     f"{q['complete_by_condition'].get('neutral', 0)} | {q['matched_low_neutral']} | {q['matched_games']} |"
                     + (f" {q['complete_by_condition'].get('high', 0)} | {q.get('matched_high_neutral', 0)} |" if has_high else ""))
    lines += ["", "| Measured passage cohort | Passages | Mean original words | Mean original utterances |",
              "| --- | ---: | ---: | ---: |"]
    for treatment, retained in p["passage_retention"].items():
        for group in ("retained", "excluded"):
            row = retained[group]
            lines.append(f"| {condition_label(treatment)} {group} | {row['n']} | {fmt(row['mean_original_words'])} | {fmt(row['mean_original_utterances'])} |")
    lines += ["", "Whole-game eligibility requires 150 original words. The prefix and chronological-third "
              "passages use a 150 to 250 word window.", "",
              "### Sentence structure", "",
              f"The {p['matched_utterances']} accepted A–B utterance pairs have the following means. "
              "Sentence counts use the deterministic punctuation-based splitter for this diagnostic.", "",
              "| Condition | Mean sentences per utterance | Mean words per utterance |",
              "| --- | ---: | ---: |"]
    for c, q in p["sentence_diagnostics"].items():
        lines.append(f"| {condition_label(c)} | {fmt(q['mean_sentences'])} | {fmt(q['mean_words'])} |")
    lines += ["", "Large sentence-count differences can mechanically reduce per-sentence complexity. "
              "Subordinate clauses per 100 words provide a complementary syntax measure.", "",
              "### Availability by source type", ""]
    for group, title in (("predicate_name", "Speech act"), ("phase", "Phase"), ("game_id", "Game")):
        conditions = list(p["generation_qc"])
        lines += [f"| {title} | " + " | ".join(condition_label(c) + " accepted/total" for c in conditions) + " |",
                  "| --- | " + " | ".join("---:" for _ in conditions) + " |"]
        for name, qc in p["by_group"][group].items():
            values = [f"{qc[c]['final_success']}/{qc[c]['n']}" if c in qc else "n/a" for c in conditions]
            lines.append(f"| {name} | " + " | ".join(values) + " |")
        lines.append("")
    lines += ["| Source cohort | Utterances | Mean original words |", "| --- | ---: | ---: |"]
    for name, q in p["retained_composition"].items():
        label = name.replace("_low_neutral", " sources, A–B")
        lines.append(f"| {label} | {q['n']} | {fmt(q['mean_original_words'])} |")
    lines += ["", "The JSON report retains the predicate and game composition of retained and excluded sources.", "",
              "### Gate diagnostics", "", "| Condition | Failure reason | Attempt records |", "| --- | --- | ---: |"]
    for c, reasons in p["attempt_reasons"].items():
        for reason, count in list(reasons.items())[:5]:
            label = "parser HTTP 400" if "400" in reason else reason.replace("|", "\\|")
            lines.append(f"| {condition_label(c)} | {label[:130]} | {count} |")
    lines += ["", f"The inverse parser supports {p['parser_eligible_sources']}/{p['n_sources']} source predicates. "
              "Other predicates use the structural and English semantic checks.", "",
              "| Condition | Attempt records containing HTTP 400 |", "| --- | ---: |"]
    lines += [f"| {condition_label(c)} | {n} |" for c, n in p["attempts_containing_http400"].items()]
    if any(p["attempts_containing_http400"].values()):
        lines += ["", "HTTP 400 counts refer to attempt records, not individual requests."]
    if p.get("secondary_judge"):
        lines += ["", "### Second-judge agreement on accepted rewrites", "",
                  "| Condition | Judge | Accepted rewrites re-judged | Agrees | Disagrees | No verdict |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for c, q in p["secondary_judge"].items():
            lines.append(f"| {condition_label(c)} | {q['handle']} ({q['family']}) | {q['n']} | {q['agree']} | {q['disagree']} | {q['no_verdict']} |")
    if p.get("judge_calibration"):
        cal = p["judge_calibration"]
        lines += ["", "### Judge calibration on constructed rewrites", "",
                  f"{cal['n_items']} constructed items: perturbations of real utterances that break one thing "
                  "(target, role, polarity, reason, certainty, directive force) and preserved paraphrases. "
                  "Sensitivity is the share of broken items flagged; specificity the share of preserved items passed; "
                  "both are among items with a verdict.", "",
                  "| Judge | Items | With verdict | Sensitivity | Specificity | Broken flagged, all items | Preserved passed, all items |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for judge, c in cal["overall"].items():
            lines.append(f"| {judge} | {c['n']} | {c['n'] - c['no_verdict']} | {fmt(c['sensitivity'])} | {fmt(c['specificity'])} | "
                         f"{fmt(c.get('broken_detection_all'))} | {fmt(c.get('preserved_acceptance_all'))} |")
    lines += ["", "### Model perception", "",
              f"The recorded listener input uses selection {p['measured_sampling']['selection']}. "
              f"Passage reconstruction and input hashes verified: {p['measured_sampling']['verified']}.", "",
              "Listeners answer A, B or \"same\" (no meaningful difference). The signed preference is +1 when the "
              "treatment is judged more limited, -1 when the comparator is, 0 for a tie; orders, then listeners, then "
              "passages within a game are averaged, and games carry equal weight.", "",
              "| Pair | Listener | Games | Passages | Signed preference (+ = treatment more limited) | 95% interval | Tie rate | P(treatment more limited \\| not tie) |",
              "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    perception_blocks = {f"{label} / {pair_label(pair)}": block for label, key in (("range", "listener"), ("proficiency", "perceived_proficiency"))
                         for pair, block in p.get(key, {}).items()}
    for pair, block in perception_blocks.items():
        for name, q in block["listeners"].items():
            if name == "all_models":
                continue
            lo, hi = q["signed_ci95"]
            lines.append(f"| {pair} | {name} | {q['n_games']} | {q['n_bundles']} | {fmt(q['signed_preference'])} | "
                         f"[{fmt(lo)}, {fmt(hi)}] | {fmt(q['tie_rate'], 2)} | {fmt(q['share_treatment_more_limited'])} |")
    lines += ["", "| Pair | Listener | Same answer in both orders | Always position A | Always position B | Position score (A = +1, B = -1) | 90% interval |",
              "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for pair, block in perception_blocks.items():
        for name, row in block["listeners"].items():
            if name != "all_models":
                q, b = row["order_consistency"], row["position_bias"]
                lines.append(f"| {pair} | {name} | {q['same_condition']}/{q['n_comparisons']} | {q['always_a']} | {q['always_b']} | "
                             f"{fmt(b['score'])} | [{fmt(b['ci90'][0])}, {fmt(b['ci90'][1])}] |")
    lines += ["", "The frozen equivalence margin applies to the 90% interval in both orders. "
              "The position score measures preference for slot A over slot B.", ""]
    if p.get("cefr"):
        from experiments.language_validation.validation.cefr import render_cefr
        lines += render_cefr(p["cefr"])
        invalid = p["cefr"].get("invalid_response_diagnostics", {})
        if invalid:
            counts = Counter(code.split(":")[0] for codes in invalid.values() for code in set(
                error.split(":")[0] for error in codes))
            lines += ["### Invalid CEFR responses", "",
                      "Counts refer to responses containing each error type. The JSON report retains sample IDs and evidence-level diagnostics.", "",
                      "| Error type | Responses |", "| --- | ---: |"]
            lines += [f"| {code} | {count} |" for code, count in sorted(counts.items())]
            lines += ["", "A quote with omitted or altered words counts as ungrounded; such a response keeps a missing rating.", ""]
    lines += ["", "Listener intervals resample games and retain both presentation orders for each passage and model. "
              "Constant game scores use a Hoeffding bounded-mean interval on the signed scale.", "",
              "### Measurement and provenance", "", "| Role | Model | Family | Revision |", "| --- | --- | --- | --- |"]
    for role, m in p["models"].items():
        lines.append(f"| {role} | {m['model']} | {m['family']} | {m.get('revision')} |")
    family_roles = defaultdict(list)
    for role, model in p["models"].items():
        family_roles[model.get("family", "unknown")].append(role)
    shared = {family: roles for family, roles in family_roles.items() if len(roles) > 1}
    if shared:
        lines += ["", f"Model families shared across roles: {shared}."]
    lines += ["",
              f"SUBTLEX measurements present: {p['subtlex_measured']}. "
              f"Grammar guardrail file present: {p['guardrails_file_present']}.", "",
              "| Generation commit | Rows |", "| --- | ---: |"]
    lines += [f"| {commit} | {n} |" for commit, n in sorted(p["generation_commits"].items())]
    lines += ["", "Input file hashes, analysis-code hashes and prompt-version counts are included in the JSON report. "
              "Generation commit IDs describe the archived run; reanalysis uses the current analysis code.", ""]
    lines += [f"- {error}" for error in p["provenance_errors"]]
    return "\n".join(lines)


ARTIFACTS = ("sources.jsonl", "bundles.jsonl", "original_features.jsonl")
BASELINE_METRICS = (
    ("word_count", "Words per bundle"), ("mattr", "MATTR"), ("mtld", "MTLD"),
    ("repetition_rate", "Content-lemma repetition"),
    ("subordinate_clauses_per_sentence", "Subordinate clauses per sentence"),
    ("mean_dependency_distance", "Mean dependency distance"), ("subtlex_mean", "SUBTLEX mean"),
)


def assemble_originals(sources, bundles):
    by_id = {s.source_id: s for s in sources}
    rows = []
    for b in bundles:
        selected = [by_id[sid] for sid in b["source_ids"]]
        if any(s.game_id != b["game_id"] or s.speaker_id != b["speaker"] for s in selected):
            raise ValueError("bundle sources do not belong to its game and speaker")
        text = " ".join(s.original_nl for s in selected)
        rows.append({"bundle_id": b["bundle_id"], "game_id": b["game_id"], "speaker": b["speaker"],
                     "condition": "original", "complete": True, "n_utterances": len(selected),
                     "text": text, "word_count": word_count(text)})
    return rows


def check_sources(sources):
    if not sources:
        raise ValueError("no untreated source utterances were found")
    assert_unique([s.to_dict() for s in sources], ("source_id",))
    bad = [s.source_id for s in sources if not s.speech_id or not s.original_nl
           or len(s.predicate_args) < 2 or s.speaker_id != s.predicate_args[1]
           or "|prior_history" not in s.provenance or not s.provenance.startswith("game_run:")]
    if bad:
        raise ValueError(f"source identity/history checks failed: {bad}")


def prepare(args):
    paths = [Path(p) for p in args.game_dirs]
    if not paths:
        raise ValueError("supply at least one untreated game directory")
    grouped = load_sources(paths)
    sources = [r for rows in grouped.values() for r in rows]
    check_sources(sources)
    result = build_bundles(sources, window_min=args.window[0], window_max=args.window[1], max_per_game=None,
                           selection=args.selection)
    nlp = load_nlp_or_none(args.spacy_model)
    from components.metrics.syntactic import nlp_provenance
    from components.metrics.lexical import SubtlexNorms
    norms = SubtlexNorms(args.subtlex, "Lg10CD") if args.subtlex else None
    features = compute_features(assemble_originals(sources, result["bundles"]), nlp=nlp, norms=norms,
                                mattr_window=50, include_domain=False)
    out = args.data_dir
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "sources.jsonl", [r.to_dict() for r in sources])
    write_jsonl(out / "bundles.jsonl", result["bundles"])
    write_jsonl(out / "original_features.jsonl", features)
    runs = []
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        runs.append({"run_id": path.name, "path": str(path), "manifest_game_id": manifest.get("game_id"),
                     "source_model": manifest.get("model"), "seed": manifest.get("random_seed"),
                     "players": manifest.get("num_players"), "outcome": manifest.get("game_state_status"),
                     "rewriter": manifest.get("rewriter"), "treatment_evidence": treatment_evidence(path),
                     "raw_sha256": {name: sha256_file(path / name) for name in
                                    ("manifest.json", "game_nl.json", "public_history.json")}})
    provenance = {
        "schema_version": 1, "created_at": now_iso(), "source_runs": runs,
        "source_type": "untreated Stage-2 game speech", "planned_test_model_family": "Qwen3.5",
        "possible_additional_test_model_family": "DeepSeek",
        "sampling": {"rule": ("all public speech records; one passage per game/speaker drawn from the early, middle "
                              "and late thirds of the speaker's timeline with a fixed seed" if args.selection == "phase_stratified"
                              else f"all public speech records; one {args.selection} bundle per game/speaker"),
                     "selection": args.selection, "window_words": list(args.window), "max_per_game": None,
                     "use": "descriptive baseline and pilot preparation"},
        "history": "strictly before the source speech; private roles and future turns are excluded",
        "instruments": {"spacy": nlp_provenance(nlp) if nlp else None, "mattr_window": 50,
                        "include_domain_terms": False, "subtlex": norms.provenance() if norms else None},
        "artifacts": {name: sha256_file(out / name) for name in ARTIFACTS},
        "validation_scope": "automated linguistic measures and model perception",
        "validation_evidence": {"proficiency_rewrites": False,
                                "grammar_guardrails": False, "listener_perception": False},
    }
    update_metadata(out / "sources.jsonl", "provenance", provenance)
    report = summarize(out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render(report), encoding="utf-8")
    print(f"{len(sources)} original utterances, {len(result['bundles'])} bundles -> {out}")
    print(f"report -> {args.report}")
    return 0


def summarize(data_dir, pilot_dir=None, annotations_path=None):
    if annotations_path is not None and (pilot_dir is None or not Path(annotations_path).is_file()):
        raise ValueError("an annotation report requires a pilot run and an existing annotations file")
    data_dir = Path(data_dir)
    prov = read_metadata(data_dir / "sources.jsonl")
    for name in ARTIFACTS:
        if prov["artifacts"].get(name) != sha256_file(data_dir / name):
            raise ValueError(f"{name} no longer matches its provenance; rebuild the prepared data")
    sources = [SourceRecord.from_dict(r) for r in read_jsonl(data_dir / "sources.jsonl")]
    check_sources(sources)
    bundles = read_jsonl(data_dir / "bundles.jsonl")
    features = read_jsonl(data_dir / "original_features.jsonl")
    assert_unique(bundles, ("bundle_id",))
    assert_unique(features, ("bundle_id", "condition"))
    window = prov["sampling"]["window_words"]
    selection = prov["sampling"].get("selection", "prefix")   # archived audits predate the key
    expected = build_bundles(sources, window_min=window[0], window_max=window[1], max_per_game=None,
                             selection=selection)
    if bundles != expected["bundles"]:
        raise ValueError("bundles do not follow the archived source selection rule")
    if {(r["bundle_id"], r["condition"]) for r in features} != {(b["bundle_id"], "original") for b in bundles}:
        raise ValueError("original feature cohort mismatch")
    grouped, texts = defaultdict(list), defaultdict(list)
    for source in sources:
        grouped[source.game_id].append(source)
        texts[source.original_nl].append(source.source_id)
    corpus = audit(grouped)
    duplicate_groups = [ids for ids in texts.values() if len(ids) > 1]
    cross_game = [ids for ids in duplicate_groups if len({x.split(":")[0] for x in ids}) > 1]
    baseline = {}
    for key, label in BASELINE_METRICS:
        values = [r[key] for r in features if r.get(key) is not None]
        baseline[key] = {"label": label, "n_bundles": len(values), "mean": mean(values), "median": median(values)}
    result = {"provenance": prov, "corpus": corpus, "n_bundles": len(bundles),
            "bundles_by_game": dict(Counter(b["game_id"] for b in bundles)),
            "n_insufficient_speakers": len(expected["insufficient"]),
            "selected_source_count": len({sid for b in bundles for sid in b["source_ids"]}),
            "unique_original_texts": len(texts), "duplicate_text_groups": duplicate_groups,
            "cross_game_duplicate_text_groups": cross_game, "baseline": baseline,
            "validation_status": "pilot"}
    if pilot_dir:
        result["pilot"] = summarize_pilot(data_dir, pilot_dir)
        from experiments.language_validation.validation.analysis import build_analysis
        pilot_dir = Path(pilot_dir)
        measured = pilot_dir / "validation"
        optional = lambda name: measured / name if (measured / name).exists() else None
        result["analysis"] = build_analysis(SimpleNamespace(
            no_frozen=True, frozen=None, split="pilot", n_boot=2000, seed=101,
            rewrites=pilot_dir / "rewrite/rewrites.jsonl", features=measured / "linguistic_features.jsonl",
            guardrails=optional("grammar_guardrails.jsonl"), cefr=optional("cefr_auxiliary.jsonl"),
            listener=listener_files(measured, "range_listener") + listener_files(measured, "perceived_proficiency_listener")))
        robustness = measured / "whole_game"
        if (robustness / "linguistic_features.jsonl").exists():
            # Whole-game transcripts form the secondary cohort.
            opt = lambda name: robustness / name if (robustness / name).exists() else None
            result["analysis_whole_game"] = build_analysis(SimpleNamespace(
                no_frozen=True, frozen=None, split="pilot", n_boot=2000, seed=101,
                rewrites=pilot_dir / "rewrite/rewrites.jsonl", features=robustness / "linguistic_features.jsonl",
                guardrails=opt("grammar_guardrails.jsonl"), cefr=opt("cefr_auxiliary.jsonl"),
                listener=listener_files(robustness, "range_listener") + listener_files(robustness, "perceived_proficiency_listener")))
            result["whole_game"] = secondary_cohort(robustness, pilot_dir / "rewrite/rewrites.jsonl")
        manual = Path(annotations_path) if annotations_path is not None else data_dir.parent / "manual_validation.jsonl"
        if manual.exists():
            from experiments.language_validation.validation.annotation import verified_summary
            result["human_listeners"] = verified_summary(read_jsonl(manual),
                measured / "linguistic_features_assembled.jsonl", listener_files(measured, "range_listener"))
            result["human_listeners"]["annotations_sha256"] = sha256_file(manual)
    return result


def secondary_cohort(measured, rewrites_path):
    """Perception and CEFR summaries for a secondary passage cohort, in the same form as the primary ones."""
    from experiments.language_validation.validation.cefr import ordinal_analysis, verify_ratings
    out = {"complete_by_condition": dict(Counter(r["condition"] for r in read_jsonl(measured / "linguistic_features_assembled.jsonl") if r.get("complete")))}
    for key, stem, construct in (("listener", "range_listener", "range"), ("perceived_proficiency", "perceived_proficiency_listener", None)):
        rows = read_listener_rows(measured, stem)
        if construct:
            rows = [r for r in rows if r.get("construct", "range") == construct]
        out[key] = listener_analysis(rows, n_boot=2000, seed=101)
    if (measured / "cefr_auxiliary.jsonl").exists():
        prov, errors = verify_ratings(measured / "cefr_auxiliary.jsonl", rewrites_path=rewrites_path)
        out["cefr"] = None if errors else {**ordinal_analysis(read_jsonl(measured / "cefr_auxiliary.jsonl")),
                                           "calibration": prov.get("calibration"), "calibration_status": prov["calibration_status"]}
        out["cefr_errors"] = errors
    return out


def _selection_rule(sampling):
    w = sampling["window_words"]
    if sampling.get("selection", "prefix") == "whole_game":
        return (f"Passages are whole speaker-game transcripts with at least {w[0]} original words; "
                "there is no upper truncation.")
    if sampling.get("selection") == "phase_stratified":
        return (f"Passages are {w[0]} to {w[1]} words of whole utterances by one speaker in one game, drawn with a "
                "fixed seed from the early, middle and late thirds of that speaker's timeline and kept in speech order; "
                "the utterance IDs are fixed before any rewrite exists and are identical across conditions.")
    return (f"The descriptive window is {w[0]} to {w[1]} words. Whole utterances are appended in speech order "
            "until the minimum is reached, without crossing the maximum.")


def render(report):
    c, p = report["corpus"], report["provenance"]
    lines = ["# Proficiency Validation", "",
        "This study evaluates controlled changes in vocabulary and syntactic range while preserving "
        "meaning, grammaticality and pragmatic force. This report describes the untreated source corpus "
        "and its baseline language measures" + (" alongside the Qwen pilot results." if report.get("pilot") else "."), "",
        "The statistics are computed from `data/corpus/` using "
        "`python -m experiments.language_validation.validation report`.", "", "## Setup", "",
        f"- {c['n_games']} untreated games with {c['n_utterances']} public speech utterances",
        f"- {len(c['predicates'])} observed speech-act types",
        f"- {report['n_bundles']} original passages assembled from {report['selected_source_count']} utterances",
        "- descriptive analysis of Original speech", "",
        "| Run | Recorded model | Seed | Players | Utterances | Bundles |",
        "| --- | --- | --- | ---: | ---: | ---: |"]
    for run in p["source_runs"]:
        rid = run["run_id"]
        lines.append(f"| {rid} | {run['source_model']} | {run['seed']} | {run['players']} | "
                     f"{c['per_game'][rid]['n_utterances']} | {report['bundles_by_game'].get(rid, 0)} |")
    lines += ["", "The source manifests and speech records identify all runs as untreated. "
              "Directory names identify games because the manifests reuse the same internal game ID. "
              "Source hashes are recorded in data/corpus/manifest.json.", "",
              "## Source coverage", "", "| Speech act | Utterances |", "| --- | ---: |"]
    lines += [f"| {name} | {n} |" for name, n in c["predicates"].items()]
    lines += ["", "Role claims and divine-result claims are included. Only public information before "
              "each utterance is retained for rewriting and judgement.", "",
              f"There are {report['unique_original_texts']} distinct original texts. "
              f"{len(report['cross_game_duplicate_text_groups'])} exact-text groups occur in more than one game. "
              "The corpus retains repeated wording, with games serving as the independent sampling units.", "",
              "## Passage construction", "",
              _selection_rule(p["sampling"]) + " There is one passage per speaker "
              "per game and no per-game cap in this source audit.", "",
              f"{report['n_insufficient_speakers']} speaker records cannot fill this window. "
              "This baseline eligibility rule is applied before any proficiency rewrite.", "",
              "| Candidate window | Eligible passages |", "| --- | ---: |"]
    for minimum in c["windows"]:
        count = c["candidate_bundles_by_window"].get(minimum, c["candidate_bundles_by_window"].get(str(minimum)))
        lines.append(f"| {minimum} to {minimum + c['window_span']} words | {count} |")
    lines += ["", "## Original language measures", "",
              "| Measure | Valid passages | Mean | Median |", "| --- | ---: | ---: | ---: |"]
    for row in report["baseline"].values():
        lines.append(f"| {row['label']} | {row['n_bundles']} | {fmt(row['mean'])} | {fmt(row['median'])} |")
    spacy = p["instruments"]["spacy"]
    syntax_note = (f"Syntax uses {spacy['lang']}_{spacy['model']} {spacy['model_version']} "
                   f"with spaCy {spacy['spacy_version']}. ") if spacy else ""
    lines += ["", syntax_note + "MATTR uses a 50-token window. Player and statement IDs are excluded "
              "from lexical diversity; domain words are excluded from content-word measures. "
              "These statistics describe the Original condition.", "",
              "## Validation evidence", "",
              "| Evidence | Source |", "| --- | --- |",
              "| Untreated originals with traceable source records | `data/corpus/` |"]
    pilot = report.get("pilot", {})
    evidence = [("Proficiency rewrites", ", ".join(condition_label(c) for c in ordered_conditions(pilot["generation_qc"])) + " pilot outputs" if pilot else None),
                ("English structural, inverse-parser and model fidelity checks on rewrites", "every attempt recorded" if pilot else None),
                ("LanguageTool and CoLA guardrails on the rewrites", "measured on every complete passage" if pilot.get("guardrails_file_present") else None),
                ("Blind linguistic-range comparisons", "both presentation orders recorded" if pilot.get("listener") else None),
                ("Blind perceived-proficiency comparisons", "both presentation orders recorded" if pilot.get("perceived_proficiency") else None),
                ("Auxiliary CEFR band ratings", "rated per passage" if pilot.get("cefr") else None),
                ("Human blind linguistic-range comparisons", f"two researchers, {report['human_listeners']['n_pairs']} pairs" if report.get("human_listeners") else None)]
    lines += [f"| {name} | {status} |" for name, status in evidence if status]
    lines += ["", "## Interpretation", "",
              "The three source games are the development cohort. Confirmation and replication use fresh, disjoint "
              "game cohorts under the frozen protocol, with the equivalence margins, non-inferiority margins and "
              "sample size fixed beforehand.", ""]
    if report.get("pilot"):
        lines.append(render_pilot(report["pilot"]))
    if report.get("analysis"):
        from experiments.language_validation.validation.analysis import _contrast_table
        lines += ["## Measured passage contrasts", "",
                  "These comparisons use complete paired passages with equal game weight. Estimates describe the retained cohort.", ""]
        for family in ("range", "guardrails"):
            for block in report["analysis"].get(family, {}).values():
                lines += _contrast_table(block, list(block["metrics"]))
        from experiments.language_validation.validation.analysis import sensitivity_lines
        lines += sensitivity_lines(report["analysis"].get("splitting_sensitivity"))
    if report.get("human_listeners"):
        from experiments.language_validation.validation.annotation import render_summary
        lines += render_summary(report["human_listeners"])
    if report.get("analysis_whole_game"):
        from experiments.language_validation.validation.analysis import _contrast_table
        s = report.get("whole_game", {})
        lines += ["## Whole-game transcripts (robustness cohort)", "",
                  "Each passage is every public utterance one speaker made in one game, in order (at least 150 original words). "
                  "A passage is complete in a condition only when every one of its utterances was accepted, so one missing "
                  "rewrite removes the whole transcript and the cohort is smaller and less stable than the primary one. It "
                  "checks whether the cue persists over a whole game.", "",
                  "| Condition | Complete passages |", "| --- | ---: |"]
        lines += [f"| {condition_label(c)} | {n} |" for c, n in sorted(s.get("complete_by_condition", {}).items())]
        lines.append("")
        for family in ("range", "guardrails"):
            for block in report["analysis_whole_game"].get(family, {}).values():
                lines += _contrast_table(block, list(block["metrics"]))
        perception = {f"{label} / {pair_label(pair)}": block for label, key in (("range", "listener"), ("proficiency", "perceived_proficiency"))
                      for pair, block in (s.get(key) or {}).items()}
        if perception:
            lines += ["### Model perception (robustness cohort)", "",
                      "| Pair | Listener | Games | Passages | Signed preference (+ = treatment more limited) | 95% interval | Same condition in both orders |",
                      "| --- | --- | ---: | ---: | ---: | --- | ---: |"]
            for pair, block in perception.items():
                for name, q in block["listeners"].items():
                    if name == "all_models":
                        continue
                    lo, hi = q["signed_ci95"]; oc = q["order_consistency"]
                    lines.append(f"| {pair} | {name} | {q['n_games']} | {q['n_bundles']} | {fmt(q['signed_preference'])} | "
                                 f"[{fmt(lo)}, {fmt(hi)}] | {oc['same_condition']}/{oc['n_comparisons']} |")
            lines.append("")
        if s.get("cefr"):
            from experiments.language_validation.validation.cefr import render_cefr
            lines += render_cefr(s["cefr"])
    return "\n".join(lines)


def report_command(args):
    report = summarize(args.data_dir, args.pilot_dir, getattr(args, "annotations", None))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(report), encoding="utf-8")
        print(f"report -> {args.out}")
    else:
        print(render(report))
    if args.json:
        dump_json(args.json, report)
    return 0


def add_parser(sub):
    p = sub.add_parser("prepare-data", help="prepare untreated proficiency sources and descriptive report")
    p.add_argument("game_dirs", nargs="+")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--window", type=int, nargs=2, default=[150, 250])
    p.add_argument("--selection", choices=["phase_stratified", "whole_game", "prefix"], default="phase_stratified",
                   help="passage rule for the descriptive baseline cohort (the primary rule is phase_stratified)")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    p.add_argument("--subtlex", type=Path)
    p.add_argument("--report", type=Path, default=EXPERIMENT_DIR / "VALIDATION.md")
    p.set_defaults(func=prepare)
    p = sub.add_parser("report", help="recompute the source report from curated proficiency data")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--pilot-dir", type=Path, help="archived rewrite, feature and listener outputs to audit")
    p.add_argument("--annotations", type=Path, help="human labels for this run; defaults to data/manual_validation.jsonl")
    p.set_defaults(func=report_command)
