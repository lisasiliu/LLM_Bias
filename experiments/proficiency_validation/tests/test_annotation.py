"""Human annotation: pair construction, worksheets, import, reliability, consensus and the model comparison."""
import json
from types import SimpleNamespace

import pytest

from experiments.proficiency_validation.validation import annotation as ann
from experiments.proficiency_validation.rewrite.runtime import condition_label, pair_label
from components.helpers import read_jsonl, write_jsonl
from experiments.proficiency_validation.rewrite.runtime import record_artifact, sha256_file


def _assembled():
    rows = []
    for game, speakers in (("g1", ("P1", "P2")), ("g2", ("P3", "P4"))):
        for speaker in speakers:
            for condition in ("original", "low", "neutral", "high"):
                complete = not (condition == "high" and speaker == "P4")
                rows.append({"bundle_id": f"{game}:{speaker}", "game_id": game, "speaker": speaker, "condition": condition,
                             "complete": complete, "text": f"{condition} text of {speaker}" if complete else "",
                             "utterances": [f"{condition} first utterance of {speaker}.", f"{condition} second utterance of {speaker}."] if complete else []})
    return rows


def test_labels():
    assert [condition_label(c) for c in ("low", "neutral", "high", "original")] == ["A", "B", "C", "Original"]
    assert pair_label("low:neutral") == "A vs B" and pair_label("neutral_vs_original", " − ") == "B − Original"


def test_pairs_cover_every_complete_passage_and_fix_sides_by_seed():
    pairs = ann.build_pairs(_assembled(), seed=7)
    counts = {}
    for p in pairs:
        counts[p["comparison_type"]] = counts.get(p["comparison_type"], 0) + 1
    assert counts == {"low:neutral": 4, "neutral:high": 3, "low:high": 3, "neutral:original": 4}
    again = ann.build_pairs(_assembled(), seed=7)
    assert [(p["pair_id"], p["passage_a_condition"]) for p in pairs] == [(p["pair_id"], p["passage_a_condition"]) for p in again]
    for p in pairs:
        assert {p["passage_a_condition"], p["passage_b_condition"]} == {p["first_condition"], p["second_condition"]}
        assert p["presentation_order"] == ("ab" if p["passage_a_condition"] == p["first_condition"] else "ba")
        assert p["passage_a_text"].startswith(p["passage_a_condition"]) and "\n\n" in p["passage_a_text"]
    capped = ann.build_pairs(_assembled(), seed=7, per_comparison=2)
    low_neutral = [p for p in capped if p["comparison_type"] == "low:neutral"]
    assert len(low_neutral) == 2 and {p["game_id"] for p in low_neutral} == {"g1", "g2"}


def test_worksheets_share_pairs_and_sides_but_not_order():
    pairs = ann.build_pairs(_assembled())
    one, two = ann.worksheet_items(pairs, "researcher_1"), ann.worksheet_items(pairs, "researcher_2")
    assert {i["pair_id"] for i in one} == {i["pair_id"] for i in two}
    assert [i["pair_id"] for i in one] != [i["pair_id"] for i in two]
    sides = {i["pair_id"]: i["passage_a_condition"] for i in one}
    assert all(sides[i["pair_id"]] == i["passage_a_condition"] for i in two)
    text = ann.render_worksheet(one, "researcher_1")
    for word in ("low", "neutral", "high", "original"):
        assert f"**{word}" not in text and f"{word}:" not in text.split("### item_001")[0]
    assert text.count("\nmore_limited:\n") == len(pairs)
    key = ann.worksheet_key(one)
    assert set(key["item_001"]) == {"pair_id", "bundle_id", "game_id", "speaker_id", "comparison_type", "first_condition",
                                    "second_condition", "passage_a_condition", "passage_b_condition", "presentation_order",
                                    "passage_a_sha256", "passage_b_sha256"}


def _fill(text, choose):
    out, item = [], None
    for line in text.splitlines():
        if line.startswith("### item_"):
            item = line[4:].strip()
        if line.strip() == "more_limited:":
            line = f"more_limited: {choose(item)}"
        out.append(line)
    return "\n".join(out)


def test_import_round_trip():
    pairs = ann.build_pairs(_assembled())
    items = ann.worksheet_items(pairs, "researcher_1")
    key = ann.key_record(items, "researcher_1", "x")
    filled = _fill(ann.render_worksheet(items, "researcher_1"), lambda item: "A" if item.endswith(("1", "3", "5")) else "same")
    rows = ann.import_annotations(filled, key, "researcher_1")
    assert len(rows) == len(pairs) and {r["rater_type"] for r in rows} == {"human"}
    by_item = {r["item_id"]: r for r in rows}
    for item in items:
        r = by_item[item["item_id"]]
        expected = item["passage_a_condition"] if r["raw_label"] == "A" else "same"
        assert r["condition_level_label"] == expected and r["pair_id"] == item["pair_id"]
    with pytest.raises(ValueError):
        ann.import_annotations(filled.replace("more_limited: A", "more_limited:", 1), key, "researcher_1")
    with pytest.raises(ValueError):
        ann.normalize_label("maybe")


def _human_rows(labels_by_rater):
    pairs = ann.build_pairs(_assembled())
    rows = []
    for rater, label_of in labels_by_rater.items():
        items = ann.worksheet_items(pairs, rater)
        key = ann.key_record(items, rater, "x")
        answers = {i["item_id"]: label_of(i) for i in items}
        text = _fill(ann.render_worksheet(items, rater), answers.__getitem__)
        rows = ann.merge_rater(rows, ann.import_annotations(text, key, rater), rater)
    return pairs, rows


def _first_side(item):
    return "A" if item["passage_a_condition"] == item["first_condition"] else "B"


def test_reliability_consensus_and_model_agreement():
    pairs, rows = _human_rows({"researcher_1": lambda i: _first_side(i) if i["comparison_type"] != "low:high" else "same",
                               "researcher_2": lambda i: _first_side(i) if i["comparison_type"] != "neutral:original" else "same"})
    rel = ann.reliability(rows)
    assert rel["n_pairs"] == 14 and rel["percent_agreement"] == pytest.approx(7 / 14)
    assert -1 < rel["cohen_kappa"] < 0.5 and rel["by_comparison"]["low:neutral"]["percent_agreement"] == 1.0
    assert rel["confusion"]["same"]["first"] == 3 and rel["confusion"]["first"]["same"] == 4
    assert ann.cohen_kappa([("first", "first"), ("same", "same")]) == 1.0
    assert ann.cohen_kappa([("first", "same"), ("same", "first")]) == pytest.approx(-1.0)
    assert ann.cohen_kappa([("a", "a"), ("a", "b"), ("b", "b"), ("b", "b")]) == pytest.approx(0.5)
    with_consensus, unresolved = ann.apply_consensus(rows)
    assert len(unresolved) == 7 and {p.split("|")[1] for p in unresolved} == {"low:high", "neutral:original"}
    with_consensus, unresolved = ann.apply_consensus(rows, {p: "same" for p in unresolved})
    assert not unresolved and all(r["consensus_label"] for r in with_consensus)
    with pytest.raises(ValueError):
        ann.apply_consensus(rows, {p: "high" for p in ann.apply_consensus(rows)[1]})

    def listener_row(pair, listener, order, choice):
        first, second = pair["first_condition"], pair["second_condition"]
        left, right = (first, second) if order == "ab" else (second, first)
        chosen = None if choice == "same" else (left if choice == "A" else right)
        treatment = "high" if {first, second} == {"neutral", "high"} else first
        comparator = ({first, second} - {treatment}).pop()
        return {"bundle_id": pair["bundle_id"], "game_id": pair["game_id"], "speaker": pair["speaker_id"],
                "pair": f"{treatment}:{comparator}", "treatment": treatment, "comparator": comparator, "listener": listener,
                "listener_family": "test", "order": "ab" if left == treatment else "ba", "left_condition": left, "right_condition": right, "ok": True,
                "choice": choice, "chosen_condition": chosen, "tie": choice == "same", "construct": "range"}
    listener_rows = []
    for pair in pairs:
        for order in ("ab", "ba"):
            first_side = "A" if (order == "ab") else "B"
            listener_rows.append(listener_row(pair, "model_x", order, first_side))        # always picks the first condition
            listener_rows.append(listener_row(pair, "model_y", order, "A"))               # always picks slot A
    models = ann.model_rows(listener_rows, pairs)
    assert len(models) == 4 * len(pairs)
    neutral_high = [m for m in models if m["comparison_type"] == "neutral:high" and m["rater_id"] == "model_x"]
    assert neutral_high and all(m["condition_level_label"] == "neutral" for m in neutral_high)
    s = ann.summary(with_consensus, listener_rows)
    assert s["listeners"] == ["model_x", "model_y"] and s["n_consensus"] == 14
    assert s["comparisons"]["low:high"]["raters"]["model_x"]["signed_preference"] == 1.0
    assert s["comparisons"]["low:high"]["raters"]["consensus"]["same"] == 1.0
    assert s["comparisons"]["low:neutral"]["raters"]["consensus"]["first_more_limited"] == 1.0
    assert s["order_consistency"]["model_x"]["share"] == 1.0 and s["order_consistency"]["model_y"]["share"] == 0.0
    agree_x = s["agreement_with_consensus"]["model_x"]
    assert agree_x["n_judgments"] == 28 and agree_x["percent_agreement"] == pytest.approx(14 / 28)
    text = "\n".join(ann.render_summary(s))
    assert "## Human listeners" in text and "A vs B" in text and "B vs Original" in text and "model_y" in text


def test_write_and_consensus_sheet(tmp_path):
    assembled = tmp_path / "assembled.jsonl"
    assembled.write_text("\n".join(json.dumps(r) for r in _assembled()) + "\n")
    result = ann.write_worksheets(assembled, tmp_path / "out", seed=3)
    assert result["n_pairs"] == 14 and len(result["files"]) == 4
    key = json.loads((tmp_path / "out/researcher_1_key.json").read_text())
    assert key["rater"] == "researcher_1" and len(key["items"]) == 14
    text = (tmp_path / "out/researcher_1_worksheet.md").read_text()
    answers = _fill(text, lambda item: "B")
    rows = ann.import_annotations(answers, key, "researcher_1")
    key2 = json.loads((tmp_path / "out/researcher_2_key.json").read_text())
    text2 = (tmp_path / "out/researcher_2_worksheet.md").read_text()
    rows = ann.merge_rater(rows, ann.import_annotations(_fill(text2, lambda item: "A"), key2, "researcher_2"), "researcher_2")
    pairs_by_id = ann.load_pairs_from_keys(tmp_path / "out", assembled)
    sheet = ann.render_consensus_worksheet(rows, pairs_by_id)
    assert sheet.count("\nconsensus:\n") == 14
    decisions = ann.parse_consensus(sheet.replace("consensus:\n", "consensus: same\n"), pairs_by_id, rows)
    resolved, unresolved = ann.apply_consensus(rows, decisions)
    assert not unresolved and {r["consensus_label"] for r in resolved} == {"same"}


def _material_fixture(tmp_path):
    assembled = tmp_path / "run/validation/linguistic_features_assembled.jsonl"
    write_jsonl(assembled, _assembled())
    folder = tmp_path / "forms"
    ann.write_worksheets(assembled, folder)
    rows = []
    for rater, choice in zip(ann.RATERS, ("A", "B")):
        key = json.loads((folder / f"{rater}_key.json").read_text())
        text = _fill((folder / f"{rater}_worksheet.md").read_text(), lambda item: choice)
        rows = ann.merge_rater(rows, ann.import_annotations(text, key, rater), rater)
    return assembled, folder, rows


def test_import_binds_rater_text_and_item_order():
    pairs = ann.build_pairs(_assembled())
    one, two = [ann.worksheet_items(pairs, r) for r in ann.RATERS]
    key1, key2 = [ann.key_record(items, r, "x") for items, r in zip((one, two), ann.RATERS)]
    filled = _fill(ann.render_worksheet(one, "researcher_1"), lambda item: "A")
    with pytest.raises(ValueError, match="rater"):
        ann.import_annotations(filled, key2, "researcher_1")
    with pytest.raises(ValueError, match="text and item order"):
        ann.import_annotations(filled.replace("researcher_1", "researcher_2"), key2, "researcher_2")
    with pytest.raises(ValueError, match="text and item order"):
        ann.import_annotations(filled.replace("first utterance", "changed utterance", 1), key1, "researcher_1")
    with pytest.raises(ValueError, match="text and item order"):
        ann.import_annotations(filled.replace("item_001", "item_999", 1), key1, "researcher_1")


def test_duplicate_items_and_answers_fail():
    with pytest.raises(ValueError, match="duplicate worksheet item"):
        ann.parse_worksheet("### item_001\nmore_limited: A\n### item_001\nmore_limited: B")
    with pytest.raises(ValueError, match="duplicate more_limited"):
        ann.parse_worksheet("### item_001\nmore_limited: A\nmore_limited: B")


def test_build_preserves_filled_forms_and_rejects_changed_materials(tmp_path):
    assembled, folder, _ = _material_fixture(tmp_path)
    form = folder / "researcher_1_worksheet.md"
    filled = _fill(form.read_text(), lambda item: "same")
    form.write_text(filled)
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    ann.write_worksheets(assembled, folder)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
    with pytest.raises(ValueError, match="existing worksheet materials differ"):
        ann.write_worksheets(assembled, folder, seed=2)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before


def test_reimport_invalidates_only_changed_consensus_and_retains_unchanged_adjudication():
    _, rows = _human_rows({"researcher_1": lambda i: "A", "researcher_2": lambda i: "B"})
    rows, _ = ann.apply_consensus(rows, {r["pair_id"]: "same" for r in rows})
    incoming = [{**r, "consensus_label": None, "consensus_basis_sha256": None}
                for r in rows if r["rater_id"] == "researcher_1"]
    unchanged = ann.merge_rater(rows, incoming, "researcher_1")
    assert not ann.apply_consensus(unchanged)[1]
    incoming[0]["note"] = "Reconsidered the sentence structure."
    changed = ann.merge_rater(rows, incoming, "researcher_1")
    changed, unresolved = ann.apply_consensus(changed)
    assert unresolved == [incoming[0]["pair_id"]]
    assert all(r["consensus_label"] is None for r in changed if r["pair_id"] in unresolved)
    assert all(r["consensus_label"] == "same" for r in changed if r["pair_id"] not in unresolved)


def test_new_disagreement_clears_previous_automatic_consensus():
    _, old = _human_rows({"researcher_1": lambda i: "same", "researcher_2": lambda i: "same"})
    incoming = [{**r, "raw_label": "A", "condition_level_label": r["passage_a_condition"],
                 "consensus_label": None, "consensus_basis_sha256": None}
                for r in old if r["rater_id"] == "researcher_1"]
    merged = ann.merge_rater(old, incoming, "researcher_1")
    _, unresolved = ann.apply_consensus(merged)
    assert len(unresolved) == 14
    assert all(r["consensus_label"] is None for r in merged)


def test_consensus_requires_two_distinct_complete_raters():
    pairs, rows = _human_rows({"researcher_1": lambda i: "same"})
    with pytest.raises(ValueError, match="both researchers"):
        ann.apply_consensus(rows, {r["pair_id"]: "same" for r in rows})
    with pytest.raises(ValueError, match="both researchers"):
        ann.render_consensus_worksheet(rows, {p["pair_id"]: p for p in pairs})
    with pytest.raises(ValueError):
        ann.apply_consensus(rows + [dict(rows[0])])
    s = ann.summary(rows)
    assert s["n_consensus"] == 0 and s["reliability"]["n_pairs"] == 0
    text = "\n".join(ann.render_summary(s))
    assert "1 of 2 researchers" in text and "Two researchers labelled" not in text
    assert "Model summaries" not in text


def test_consensus_form_is_blind_and_bound_to_current_independent_answers(tmp_path):
    assembled, folder, rows = _material_fixture(tmp_path)
    pairs = ann.load_pairs_from_keys(folder, assembled)
    sheet = ann.render_consensus_worksheet(rows, pairs)
    headings = [line for line in sheet.splitlines() if line.startswith("###")]
    assert len(headings) == 14
    assert not any(label in heading for heading in headings for label in ("low", "neutral", "high", "original", "g1", "P1"))
    filled = sheet.replace("consensus:\n", "consensus: same\n")
    assert len(ann.parse_consensus(filled, pairs, rows)) == 14
    changed = [dict(r) for r in rows]
    changed[0]["note"] = "Updated explanation."
    with pytest.raises(ValueError, match="current materials or independent labels"):
        ann.parse_consensus(filled, pairs, changed)
    with pytest.raises(ValueError, match="unanswered"):
        ann.parse_consensus(sheet, pairs, rows)


def test_material_and_listener_provenance_checks(tmp_path):
    assembled, folder, rows = _material_fixture(tmp_path)
    ann.verify_materials(rows, assembled)
    other = tmp_path / "other.jsonl"
    altered = _assembled()
    altered[0]["utterances"][0] = "A different original."
    write_jsonl(other, altered)
    with pytest.raises(ValueError, match="materials"):
        ann.verified_summary(rows, other)
    with pytest.raises(ValueError, match="materials differ"):
        ann.load_pairs_from_keys(folder, other)
    listener = assembled.parent / "range_listener.jsonl"
    write_jsonl(listener, [])
    record_artifact(listener, {"input_sha256": {"assembled": sha256_file(other)}})
    with pytest.raises(ValueError, match="listener outputs"):
        ann.verified_summary(rows, assembled, [listener])
    record_artifact(listener, {"input_sha256": {"assembled": sha256_file(assembled)}})
    assert ann.verified_summary(rows, assembled, [listener])["n_pairs"] == 14
    listener.write_text("\n")
    with pytest.raises(ValueError, match="listener outputs"):
        ann.verified_summary(rows, assembled, [listener])


def test_constant_category_kappa_is_undefined_and_renders():
    assert ann.cohen_kappa([("same", "same")] * 10) is None
    _, rows = _human_rows({r: lambda i: "same" for r in ann.RATERS})
    s = ann.summary(rows)
    assert s["reliability"]["percent_agreement"] == 1.
    assert s["reliability"]["cohen_kappa"] is None
    assert "Cohen's κ = n/a" in "\n".join(ann.render_summary(s))


def test_direction_weights_games_equally():
    rows = [{"pair_id": f"g1:P{i}|low:neutral", "game_id": "g1", "condition_level_label": "low"} for i in range(3)]
    rows += [{"pair_id": "g2:P1|low:neutral", "game_id": "g2", "condition_level_label": "neutral"}]
    result = ann._direction(rows, "low")
    assert result["signed_preference"] == 0.
    assert result["first_more_limited"] == result["second_more_limited"] == .5
    assert (result["n_games"], result["n_pairs"], result["n_judgments"]) == (2, 4, 4)


def test_model_rows_require_complete_unique_orders():
    pair = ann.build_pairs(_assembled())[0]
    base = {"bundle_id": pair["bundle_id"], "game_id": pair["game_id"], "speaker": pair["speaker_id"],
            "pair": "low:neutral", "treatment": "low", "comparator": "neutral", "listener": "model",
            "ok": True, "choice": "A", "chosen_condition": "low", "tie": False, "order": "ab",
            "left_condition": "low", "right_condition": "neutral"}
    assert ann.model_rows([base], [pair]) == []
    reverse = {**base, "order": "ba", "choice": "B", "left_condition": "neutral", "right_condition": "low"}
    assert len(ann.model_rows([base, reverse], [pair])) == 2
    assert ann.model_rows([base, {**reverse, "ok": False}], [pair]) == []
    with pytest.raises(ValueError):
        ann.model_rows([base, dict(base)], [pair])


def test_cli_consensus_preserves_filled_sheet_and_rejects_stale_answers(tmp_path):
    assembled, folder, rows = _material_fixture(tmp_path)
    path, sheet = folder / "manual.jsonl", folder / "consensus.md"
    write_jsonl(path, rows)
    args = SimpleNamespace(annotations=path, assembled=assembled, data_dir=folder, out=sheet, worksheet=sheet)
    ann.cmd_consensus_sheet(args)
    filled = sheet.read_text().replace("consensus:\n", "consensus: same\n")
    sheet.write_text(filled)
    ann.cmd_consensus_sheet(args)
    assert sheet.read_text() == filled
    ann.cmd_import_consensus(args)
    assert all(r["consensus_label"] == "same" for r in read_jsonl(path))
    rows[0]["note"] = "Revised explanation."
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="different labels or materials"):
        ann.cmd_consensus_sheet(args)
    with pytest.raises(ValueError, match="current materials or independent labels"):
        ann.cmd_import_consensus(args)


def test_pilot_report_checks_explicit_annotations_and_reports_partial_completion(tmp_path):
    from experiments.proficiency_validation.validation import report
    from experiments.proficiency_validation.rewrite.runtime import DATA_DIR, RUNS_DIR
    data = DATA_DIR.parent
    key = json.loads((data / "researcher_1_key.json").read_text())
    text = _fill((data / "researcher_1_worksheet.md").read_text(), lambda item: "same")
    rows = ann.import_annotations(text, key, "researcher_1")
    path = tmp_path / "manual.jsonl"
    write_jsonl(path, rows)
    result = report.summarize(DATA_DIR, RUNS_DIR / "pilot_qwen35_27b", annotations_path=path)
    assert result["human_listeners"]["raters"] == ["researcher_1"]
    assert result["human_listeners"]["n_consensus"] == 0
    assert result["human_listeners"]["annotations_sha256"] == sha256_file(path)
    assert "1 of 2 researchers" in report.render(result)
    for pair, model_pair, sign in (("low:neutral", "low:neutral", 1), ("neutral:high", "high:neutral", -1),
                                   ("low:high", "low:high", 1), ("neutral:original", "neutral:original", 1)):
        for listener in result["human_listeners"]["listeners"]:
            actual = result["human_listeners"]["comparisons"][pair]["raters"][listener]["signed_preference"]
            expected = result["analysis"]["listener"][model_pair]["listeners"][listener]["signed_preference"]
            assert actual == pytest.approx(sign * expected, abs=1e-12)
    write_jsonl(path, [{**r, "assembled_sha256": "another-run"} for r in rows])
    with pytest.raises(ValueError, match="current run's materials"):
        report.summarize(DATA_DIR, RUNS_DIR / "pilot_qwen35_27b", annotations_path=path)
