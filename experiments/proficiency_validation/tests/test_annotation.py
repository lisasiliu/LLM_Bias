"""Human annotation: pair construction, worksheets, import, reliability, consensus and the model comparison."""
import json

import pytest

from experiments.proficiency_validation.validation import annotation as ann
from experiments.proficiency_validation.rewrite.runtime import condition_label, pair_label


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
                                    "second_condition", "passage_a_condition", "passage_b_condition", "presentation_order"}


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
    key = {"worksheet_version": ann.WORKSHEET_VERSION, "assembled_sha256": "x", "items": ann.worksheet_key(items)}
    filled = _fill(ann.render_worksheet(items, "researcher_1"), lambda item: "A" if item.endswith(("1", "3", "5")) else "same")
    rows = ann.import_annotations(ann.parse_worksheet(filled), key, "researcher_1")
    assert len(rows) == len(pairs) and {r["rater_type"] for r in rows} == {"human"}
    by_item = {r["item_id"]: r for r in rows}
    for item in items:
        r = by_item[item["item_id"]]
        expected = item["passage_a_condition"] if r["raw_label"] == "A" else "same"
        assert r["condition_level_label"] == expected and r["pair_id"] == item["pair_id"]
    with pytest.raises(ValueError):
        ann.import_annotations({k: v for k, v in ann.parse_worksheet(filled).items() if k != "item_001"}, key, "researcher_1")
    with pytest.raises(ValueError):
        ann.normalize_label("maybe")


def _human_rows(labels_by_rater):
    pairs = ann.build_pairs(_assembled())
    rows = []
    for rater, label_of in labels_by_rater.items():
        items = ann.worksheet_items(pairs, rater)
        key = {"worksheet_version": ann.WORKSHEET_VERSION, "assembled_sha256": "x", "items": ann.worksheet_key(items)}
        answers = {i["item_id"]: {"more_limited": label_of(i), "note": ""} for i in items}
        rows = ann.merge_rater(rows, ann.import_annotations(answers, key, rater), rater)
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
                "listener_family": "test", "order": order, "left_condition": left, "right_condition": right, "ok": True,
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
    answers = ann.parse_worksheet(_fill(text, lambda item: "B"))
    rows = ann.import_annotations(answers, key, "researcher_1")
    key2 = json.loads((tmp_path / "out/researcher_2_key.json").read_text())
    text2 = (tmp_path / "out/researcher_2_worksheet.md").read_text()
    rows = ann.merge_rater(rows, ann.import_annotations(ann.parse_worksheet(_fill(text2, lambda item: "A")), key2, "researcher_2"), "researcher_2")
    pairs_by_id = ann.load_pairs_from_keys(tmp_path / "out", assembled)
    sheet = ann.render_consensus_worksheet(rows, pairs_by_id)
    assert sheet.count("\nconsensus:\n") == 14
    decisions = ann.parse_consensus(sheet.replace("consensus:\n", "consensus: same\n"), pairs_by_id)
    resolved, unresolved = ann.apply_consensus(rows, decisions)
    assert not unresolved and {r["consensus_label"] for r in resolved} == {"same"}
