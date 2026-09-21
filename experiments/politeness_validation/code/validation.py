"""Politeness validation: build annotation worksheets, import them, analyse them.

One file, three commands:

    python -m experiments.politeness_validation.code.validation build-worksheet \\
        --items <items.jsonl> --out <worksheet.md> --key <key.json>

    python -m experiments.politeness_validation.code.validation import-annotations \\
        --worksheet <filled.md> --key <key.json> --out <validation.jsonl>

    python -m experiments.politeness_validation.code.validation analyze

Plus a reproducibility check:

    python -m experiments.politeness_validation.code.validation verify-provenance

Nothing here calls a model. The worksheet builder and the importer are generic
over any number of items; the analysis reads only the canonical JSONL files in
``data/``.

The design keeps three reference standards apart, because they disagree and the
disagreements are the result:

* the **requested strategy** — what the rewriter was told to produce;
* the **manual annotation** — what a blinded reader perceives, and the primary
  evidence for both manipulation success and meaning preservation;
* the **automatic judges** — diagnostics, reported but never treated as ground
  truth.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from components.helpers import read_jsonl, write_jsonl

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"
MANUAL = DATA_DIR / "manual_validation.jsonl"
EXTERNAL = DATA_DIR / "external_validation.jsonl"
PROVENANCE = DATA_DIR / "provenance.json"

STRATEGIES = ("bald_on_record", "positive_politeness", "negative_politeness",
              "off_record")
STRATEGY_LABELS = STRATEGIES + ("ambiguous_or_mixed", "failed_manipulation")

# Fields an annotator fills in. The first four are always asked; the semantic
# detail fields are optional, so a future round can use the compact form.
CORE_FIELDS = ("strategy", "confidence", "meaning", "note")
DETAIL_FIELDS = ("commitment", "directive", "evidence", "reference")


# ═══════════════════════════════════════════════════════════════
# Worksheet
# ═══════════════════════════════════════════════════════════════

# Everything an annotator must not see. Kept as an explicit list so that adding
# a new automatic field cannot silently leak it into a worksheet.
HIDDEN_FIELDS = (
    "requested_strategy", "automatic_politeness_strategy",
    "automatic_politeness_confidence", "inverse_parser_predicate",
    "inverse_parser_status", "inverse_parser_structural_match",
    "inverse_parser_exact_match", "semantic_status", "semantic_ok",
    "semantic_failed_dimensions", "semantic_dimensions", "semantic_notes",
    "structural_check_ok", "structural_check_reasons",
    "adjudication_required", "adjudication_reasons",
)


# Values that would give an annotator the answer. Checked as whole words, and
# only in the item section -- the instructions necessarily list the label
# vocabulary, and the source predicate is shown on purpose.
LEAK_TOKENS = STRATEGY_LABELS + ("exact", "mismatch", "pass", "fail")


def find_leaks(worksheet: str) -> List[str]:
    """Hidden verdicts that appear in the worksheet's item section.

    Only standalone `field: value` lines count. Searching for bare substrings
    produces false alarms -- "exact" occurs in ordinary prose, and a correctly
    recovered predicate is textually identical to the source predicate, which is
    deliberately shown.
    """
    body = worksheet.split("\n---\n", 1)[-1]
    found: List[str] = []
    for line in body.splitlines():
        match = re.match(r"^\s*[-*]?\s*(\w+)\s*:\s*`?([\w/ ]+)`?\s*$", line)
        if not match:
            continue
        field, value = match.group(1).lower(), match.group(2).strip().lower()
        # An annotation field left blank for the annotator is not a leak.
        if field in CORE_FIELDS + DETAIL_FIELDS and not value:
            continue
        if field in HIDDEN_FIELDS or value in LEAK_TOKENS:
            found.append(f"{field}: {value}")
    return sorted(set(found))


def build_manual_annotation_worksheet(
    items: Sequence[dict], *, seed: int = 101, detailed: bool = True,
) -> Tuple[str, Dict[str, dict]]:
    """``(worksheet_markdown, hidden_key)`` for a blind annotation round.

    The annotator sees the source predicate, the untreated original, the rewrite
    and, where the predicate refers to one, the earlier statement being
    responded to — so they never have to infer which speech act is being
    manipulated. They do not see the requested strategy or any automatic verdict.

    Items are shuffled so their order carries no information about condition.
    """
    fields = CORE_FIELDS[:3] + (DETAIL_FIELDS if detailed else ()) + ("note",)
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)

    lines = [
        "# Politeness annotation worksheet",
        "",
        "Each entry is one rewritten utterance. The condition it was written",
        "under, and every automatic verdict, are deliberately not shown.",
        "",
        "Fill in each field. Leave the `### item_...` headings untouched.",
        "",
        "- `strategy:`   `bald_on_record` / `positive_politeness` / "
        "`negative_politeness` / `off_record` / `ambiguous_or_mixed` / "
        "`failed_manipulation`",
        "- `confidence:` `low` / `medium` / `high`",
        "- `meaning:`    is the game-relevant meaning preserved — "
        "`yes` / `no` / `unsure`",
    ]
    if detailed:
        lines += [
            "- `commitment:` does the speaker still commit as strongly — "
            "`yes` / `no` / `n/a`",
            "- `directive:`  is the force of the act preserved — "
            "`yes` / `no` / `n/a`",
            "- `evidence:`   same reason, same relation — `yes` / `no` / `n/a`",
            "- `reference:`  same earlier statement — `yes` / `no` / `n/a`",
        ]
    lines += [
        "- `note:`       one line, only where something is `no` or `unsure`",
        "",
        "---",
        "",
    ]

    key: Dict[str, dict] = {}
    for index, item in enumerate(shuffled, 1):
        item_id = f"item_{index:03d}"
        key[item_id] = dict(item)
        lines += [f"### {item_id}", "",
                  f"- speech act: `{item.get('predicate', '')}`"]
        if item.get("referenced_text"):
            lines.append(f"- earlier statement it responds to, by "
                         f"{item.get('referenced_speaker', '?')}: "
                         f"\"{item['referenced_text']}\"")
        lines += ["",
                  f"**Original (untreated):** {item.get('original_text', '')}", "",
                  f"**Rewritten:** {item.get('rewritten_text', '')}", ""]
        lines += [f"{field}:" for field in fields]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n", key


def parse_worksheet(text: str) -> Dict[str, Dict[str, str]]:
    """``{item_id: {field: value}}`` from a filled-in worksheet."""
    answered = CORE_FIELDS[:3] + DETAIL_FIELDS
    out: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        heading = re.match(r"^#+\s*(item_\d+)\s*$", line.strip())
        if heading:
            current = heading.group(1)
            out[current] = {}
            continue
        if current is None:
            continue
        match = re.match(r"^(\w+):\s*`?([^`]*)`?\s*$", line.strip())
        if not match:
            continue
        field, value = match.group(1).lower(), match.group(2).strip()
        if field in answered and value:
            out[current][field] = value.lower()
        elif field == "note" and value:
            out[current]["note"] = value
    return {k: v for k, v in out.items() if v.get("strategy")}


def import_annotations(worksheet_text: str, key: Dict[str, dict]) -> List[dict]:
    """Merge a completed worksheet back into the hidden experimental record."""
    answers = parse_worksheet(worksheet_text)
    merged: List[dict] = []
    for item_id, filled in sorted(answers.items()):
        record = dict(key.get(item_id, {}))
        record["item_id"] = item_id
        for field, value in filled.items():
            record[f"human_{field}"] = value
        merged.append(record)
    return merged


# ═══════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════

def rate(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.1%})" if den else f"{num}/0 (n/a)"


def confusion(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {}
    for gold, predicted in pairs:
        row = table.setdefault(gold, {})
        row[predicted] = row.get(predicted, 0) + 1
    return table


def per_class(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, float]]:
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1,
                      "support": sum(1 for g, _ in pairs if g == label)}
    return out


def macro_f1(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    """Averaged over classes that occur in the GOLD labels.

    A class the classifier invented but the reference never contains carries no
    information about accuracy, so it is not allowed to drag the average.
    """
    scores = [s["f1"] for s in per_class(pairs).values() if s["support"]]
    return sum(scores) / len(scores) if scores else None


def _matrix(table: Dict[str, Dict[str, int]], corner: str,
            columns: Optional[Sequence[str]] = None) -> List[str]:
    cols = list(columns) if columns else sorted({c for r in table.values() for c in r})
    lines = [f"| {corner} | " + " | ".join(cols) + " |",
             "| --- |" + " ---: |" * len(cols)]
    for gold in sorted(table):
        lines.append(f"| {gold} | " +
                     " | ".join(str(table[gold].get(c, 0)) for c in cols) + " |")
    return lines


# ═══════════════════════════════════════════════════════════════
# Analysis — manual validation
# ═══════════════════════════════════════════════════════════════

def analyse_manual(records: Sequence[dict]) -> Dict[str, Any]:
    requested_vs_human = [(r["requested_strategy"], r["human_strategy"])
                          for r in records]
    human_vs_auto = [(r["human_strategy"], r["automatic_politeness_strategy"])
                     for r in records if r.get("automatic_politeness_strategy")]

    confidence: Dict[str, int] = {}
    for record in records:
        level = record.get("human_confidence") or "(blank)"
        confidence[level] = confidence.get(level, 0) + 1

    def meaning_counts(subset: Sequence[dict]) -> Dict[str, int]:
        counts = {"yes": 0, "no": 0, "unsure": 0}
        for record in subset:
            value = record.get("human_meaning")
            if value in counts:
                counts[value] += 1
        return counts

    strategies = [s for s in STRATEGIES
                  if any(r["requested_strategy"] == s for r in records)]
    predicates = sorted({r["predicate_name"] for r in records})

    # Automatic semantic QC against the manual `meaning` judgement. `unsure` is
    # never assigned to a side; it is carried as its own bucket.
    decided = [r for r in records if r.get("human_meaning") in ("yes", "no")]
    unsure = [r for r in records if r.get("human_meaning") == "unsure"]

    def against_meaning(accepted) -> Dict[str, Any]:
        return {
            "n_decided": len(decided),
            "agreement": sum(1 for r in decided
                             if accepted(r) == (r["human_meaning"] == "yes")),
            "accepted_but_human_no": [r["item_id"] for r in decided
                                      if accepted(r) and r["human_meaning"] == "no"],
            "flagged_but_human_yes": [r["item_id"] for r in decided
                                      if not accepted(r) and r["human_meaning"] == "yes"],
            "unsure_accepted": [r["item_id"] for r in unsure if accepted(r)],
            "unsure_flagged": [r["item_id"] for r in unsure if not accepted(r)],
        }

    queued = [r for r in records if r.get("adjudication_required")]
    return {
        "n": len(records),
        "strategies": strategies,
        "predicates": predicates,
        "manipulation": {
            "overall": sum(1 for a, b in requested_vs_human if a == b),
            "by_strategy": {s: {
                "n": sum(1 for a, _ in requested_vs_human if a == s),
                "recognized": sum(1 for a, b in requested_vs_human if a == s and a == b),
            } for s in strategies},
            "confusion": confusion(requested_vs_human),
            "off_scheme": {
                label: sum(1 for _, b in requested_vs_human if b == label)
                for label in ("ambiguous_or_mixed", "failed_manipulation")},
            "confidence": confidence,
        },
        "automatic_strategy": {
            "n": len(human_vs_auto),
            "agreement_with_human": sum(1 for a, b in human_vs_auto if a == b),
            "macro_f1": macro_f1(human_vs_auto),
            "per_class": per_class(human_vs_auto),
            "confusion": confusion(human_vs_auto),
            # False positives per PREDICTED label: how often the classifier
            # assigns a strategy the annotator did not. This is what tells the
            # error pattern apart from mere low recall.
            "false_positives_by_label": {
                label: {
                    "predicted": sum(1 for _, p in human_vs_auto if p == label),
                    "correct": sum(1 for h, p in human_vs_auto if p == label and h == label),
                    "false_positive": sum(1 for h, p in human_vs_auto
                                          if p == label and h != label),
                } for label in sorted({p for _, p in human_vs_auto})},
            "misses_by_destination": {
                f"{h} -> {p}": sum(1 for a, b in human_vs_auto if a == h and b == p)
                for h, p in sorted({(h, p) for h, p in human_vs_auto if h != p})},
            "by_requested": {s: {
                "n": sum(1 for r in records if r["requested_strategy"] == s),
                "human": sum(1 for r in records if r["requested_strategy"] == s
                             and r["human_strategy"] == s),
                "automatic": sum(1 for r in records if r["requested_strategy"] == s
                                 and r.get("automatic_politeness_strategy") == s),
            } for s in strategies},
        },
        "semantic": {
            "overall": meaning_counts(records),
            "by_strategy": {s: meaning_counts([r for r in records
                                               if r["requested_strategy"] == s])
                            for s in strategies},
            "by_predicate": {p: meaning_counts([r for r in records
                                                if r["predicate_name"] == p])
                             for p in predicates},
            "failures": [
                {"item_id": r["item_id"], "requested": r["requested_strategy"],
                 "predicate": r["predicate_name"], "meaning": r["human_meaning"],
                 "note": r.get("human_note", ""),
                 "broken": [f for f in DETAIL_FIELDS if r.get(f"human_{f}") == "no"]}
                for r in records if r.get("human_meaning") in ("no", "unsure")],
        },
        "automatic_semantic": {
            "inverse_parser": against_meaning(
                lambda r: bool(r.get("inverse_parser_structural_match"))),
            "semantic_validator": against_meaning(
                lambda r: r.get("semantic_ok") is True),
            "combined": against_meaning(lambda r: not r.get("adjudication_required")),
            "n_unsure": len(unsure),
        },
        "adjudication": {
            "queued": len(queued),
            "by_strategy": {s: sum(1 for r in queued if r["requested_strategy"] == s)
                            for s in strategies},
            "items": [{"item_id": r["item_id"], "requested": r["requested_strategy"],
                       "reasons": r.get("adjudication_reasons", []),
                       "human_meaning": r.get("human_meaning")} for r in queued],
        },
    }


def analyse_external(records: Sequence[dict]) -> Dict[str, Any]:
    eligible = [r for r in records
                if r.get("eligible_for_primary_metric") and r.get("automatic_usable")]
    pairs = [(r["mapped_strategy"], r["automatic_strategy"]) for r in eligible]
    # The positive/negative contrast, restricted to items where both the
    # published label and our prediction are one of the two, so this measures
    # the DISTINCTION rather than detection.
    PAIR = ("positive_politeness", "negative_politeness")
    gold_pair = [(g, p) for g, p in pairs if g in PAIR]
    # Only items the classifier ALSO placed in the pair can speak to the
    # positive/negative distinction; the rest it mapped out of the pair entirely.
    pn = [(g, p) for g, p in gold_pair if p in PAIR]
    return {
        "dataset": records[0]["dataset"] if records else None,
        "n_records": len(records),
        "n_eligible": len(eligible),
        "accuracy": sum(1 for g, p in pairs if g == p) / len(pairs) if pairs else None,
        "macro_f1": macro_f1(pairs),
        "per_class": per_class(pairs),
        "confusion": confusion(pairs),
        "positive_vs_negative": {
            "n_gold_pair": len(gold_pair),
            "n_predicted_in_pair": len(pn),
            "coverage": len(pn) / len(gold_pair) if gold_pair else None,
            "conditional_agreement": (sum(1 for g, p in pn if g == p) / len(pn)
                                      if pn else None),
            "conditional_correct": sum(1 for g, p in pn if g == p),
            "mapped_out_of_pair": {
                label: sum(1 for _, p in gold_pair if p == label)
                for label in sorted({p for _, p in gold_pair if p not in PAIR})},
        },
    }


# ═══════════════════════════════════════════════════════════════
# Markdown output
# ═══════════════════════════════════════════════════════════════

PRETTY = {"bald_on_record": "bald on-record",
          "positive_politeness": "positive politeness",
          "negative_politeness": "negative politeness",
          "off_record": "off-record"}


def render(manual: Dict[str, Any], external: Dict[str, Any]) -> str:
    m, a, s = manual["manipulation"], manual["automatic_strategy"], manual["semantic"]
    out: List[str] = ["## Manipulation success (manual annotation)", "",
                      "| strategy | n | recognized as requested |",
                      "| --- | ---: | ---: |"]
    for strategy in manual["strategies"]:
        row = m["by_strategy"][strategy]
        out.append(f"| {PRETTY.get(strategy, strategy)} | {row['n']} | {row['recognized']} |")
    out.append(f"| **total** | **{manual['n']}** | **{m['overall']}** |")

    out += ["", "Annotator confidence: " +
            ", ".join(f"{n} {level}" for level, n in sorted(m["confidence"].items(),
                                                            key=lambda kv: -kv[1])) + ".",
            f"Labelled `ambiguous_or_mixed`: {m['off_scheme']['ambiguous_or_mixed']}; "
            f"`failed_manipulation`: {m['off_scheme']['failed_manipulation']}.",
            "", "## Automatic strategy classification", "",
            "| requested strategy | manual | automatic |", "| --- | ---: | ---: |"]
    for strategy in manual["strategies"]:
        row = a["by_requested"][strategy]
        out.append(f"| {PRETTY.get(strategy, strategy)} | {row['human']}/{row['n']} | "
                   f"{row['automatic']}/{row['n']} |")
    out += ["",
            f"Agreement with the manual annotation: "
            f"**{rate(a['agreement_with_human'], a['n'])}**, macro-F1 "
            f"{a['macro_f1']:.3f}.", "",
            "| class | precision | recall | F1 | n |", "| --- | ---: | ---: | ---: | ---: |"]
    for label, stats in sorted(a["per_class"].items()):
        if stats["support"]:
            out.append(f"| {PRETTY.get(label, label)} | {stats['precision']:.2f} | "
                       f"{stats['recall']:.2f} | {stats['f1']:.2f} | "
                       f"{int(stats['support'])} |")
    out += ["", "Confusion, manual label against automatic label:", ""]
    out += _matrix(a["confusion"], "manual \\ automatic")

    out += ["", "## Semantic preservation (manual annotation)", "",
            "| strategy | yes | no | unsure |", "| --- | ---: | ---: | ---: |"]
    for strategy in manual["strategies"]:
        c = s["by_strategy"][strategy]
        out.append(f"| {PRETTY.get(strategy, strategy)} | {c['yes']} | {c['no']} | "
                   f"{c['unsure']} |")
    total = s["overall"]
    out.append(f"| **total** | **{total['yes']}** | **{total['no']}** | "
               f"**{total['unsure']}** |")
    out += ["", "| predicate | yes | no | unsure |", "| --- | ---: | ---: | ---: |"]
    for predicate in manual["predicates"]:
        c = s["by_predicate"][predicate]
        out.append(f"| {predicate} | {c['yes']} | {c['no']} | {c['unsure']} |")

    auto = manual["automatic_semantic"]
    out += ["", "## Automatic semantic quality control", "",
            "Agreement with the manual `meaning` judgement, over the items the",
            "annotator decided (`unsure` held out of every rate):", "",
            "| check | agreement | accepted, manually marked not preserved | "
            "flagged, manually marked preserved |",
            "| --- | ---: | ---: | ---: |"]
    for name, label in (("inverse_parser", "inverse predicate parser"),
                        ("semantic_validator", "dimensional semantic validator"),
                        ("combined", "combined policy")):
        block = auto[name]
        out.append(f"| {label} | {rate(block['agreement'], block['n_decided'])} | "
                   f"{len(block['accepted_but_human_no'])} | "
                   f"{len(block['flagged_but_human_yes'])} |")
    adj = manual["adjudication"]
    out += ["", f"Items sent to human adjudication: "
            f"**{rate(adj['queued'], manual['n'])}** — "
            + ", ".join(f"{PRETTY.get(k, k)} {v}"
                        for k, v in adj["by_strategy"].items() if v) + ".",
            "", "## External construct validation", ""]
    if external:
        pn = external["positive_vs_negative"]
        out += [f"Dataset `{external['dataset']}`, {external['n_eligible']} "
                f"eligible items of {external['n_records']} in the evaluated split.",
                "",
                f"- four-way agreement with the published superordinate label: "
                f"**{external['accuracy']:.1%}** (macro-F1 {external['macro_f1']:.3f})",
                f"- coverage into the positive/negative pair: "
                f"**{pn['n_predicted_in_pair']}/{pn['n_gold_pair']}** "
                f"({pn['coverage']:.1%}) of gold positive/negative items are also "
                f"predicted positive or negative",
                f"- conditional discrimination within those "
                f"{pn['n_predicted_in_pair']}: "
                f"**{pn['conditional_agreement']:.1%}** "
                f"({pn['conditional_correct']}/{pn['n_predicted_in_pair']})", "",
                "| class | precision | recall | F1 | n |",
                "| --- | ---: | ---: | ---: | ---: |"]
        for label, stats in sorted(external["per_class"].items()):
            if stats["support"]:
                out.append(f"| {PRETTY.get(label, label)} | {stats['precision']:.2f} | "
                           f"{stats['recall']:.2f} | {stats['f1']:.2f} | "
                           f"{int(stats['support'])} |")
        out += ["", "Confusion, published label against automatic label:", ""]
        out += _matrix(external["confusion"], "published \\ automatic")
    return "\n".join(out) + "\n"


# ═══════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════

def cmd_build_worksheet(args) -> int:
    items = read_jsonl(Path(args.items))
    worksheet, key = build_manual_annotation_worksheet(
        items, seed=args.seed, detailed=not args.compact)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(worksheet, encoding="utf-8")
    Path(args.key).write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    leaked = find_leaks(worksheet)
    if leaked:
        print(f"WARNING: the worksheet leaks {sorted(leaked)}", file=sys.stderr)
        return 1
    print(f"{len(items)} items -> {args.out} (key: {args.key})")
    return 0


def cmd_import_annotations(args) -> int:
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    merged = import_annotations(Path(args.worksheet).read_text(encoding="utf-8"), key)
    if not merged:
        print(f"No completed items found in {args.worksheet}.", file=sys.stderr)
        return 2
    write_jsonl(Path(args.out), merged)
    print(f"{len(merged)} annotated items -> {args.out}")
    return 0


def cmd_analyze(args) -> int:
    manual = analyse_manual(read_jsonl(Path(args.manual)))
    external = analyse_external(read_jsonl(Path(args.external))) \
        if Path(args.external).exists() else {}
    tables = render(manual, external)
    if args.out:
        Path(args.out).write_text(tables, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(tables)
    if args.json:
        Path(args.json).write_text(
            json.dumps({"manual": manual, "external": external},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


def cmd_verify_provenance(args) -> int:
    """Recompute the frozen prompt hashes and compare them to provenance.json.

    This is the reproducibility guarantee: if a prompt or the semantic rubric is
    edited, the recorded hashes stop matching and this command says so.
    """
    from components.judges.blind_politeness_classifier import PROMPT_VERSION, system_prompt
    from components.judges.dimensional_semantic_validator import (
        SEMANTIC_RUBRIC_VERSION, semantic_prompt_sha)
    from components.rewriters.politeness_rewriter import (
        POLITENESS_STRATEGIES, PolitenessRewriter)
    from lurkingwolf.rewriter import RewriteContext
    import hashlib

    context = RewriteContext(
        predicate="estimate(D1, P3, P7, werewolf)",
        original_nl=("P7 is a werewolf. He pushed the vote onto P2 without ever "
                     "giving a read of his own."),
        phase="day_speech", speaker_id="P3", speaker_role_claimed="villager",
        public_history=["[s1|P1] claim_role(D1, P1, seer)"])

    recorded = json.loads(Path(args.provenance).read_text(encoding="utf-8"))["frozen_artifacts"]
    problems: List[str] = []
    for strategy in sorted(POLITENESS_STRATEGIES):
        prompt = PolitenessRewriter(llm_client=None, strategy=strategy).build_prompts(context)[0]
        current = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected = recorded["generation_prompt_sha256"].get(strategy)
        status = "ok" if current == expected else "CHANGED"
        if status != "ok":
            problems.append(f"generation prompt {strategy}")
        print(f"  generation prompt {strategy:21s} {status}")

    current = hashlib.sha256(system_prompt(PROMPT_VERSION).encode("utf-8")).hexdigest()
    ok = current == recorded["politeness_classifier_prompt_sha256"]
    problems += [] if ok else ["politeness classifier prompt"]
    print(f"  politeness classifier prompt         {'ok' if ok else 'CHANGED'}")

    rubric = recorded["semantic_rubric"]
    current = semantic_prompt_sha()
    if current == rubric["current_sha256"] and SEMANTIC_RUBRIC_VERSION == rubric["current_version"]:
        print(f"  semantic rubric ({SEMANTIC_RUBRIC_VERSION})  ok")
    else:
        problems.append("semantic rubric")
        print(f"  semantic rubric                      CHANGED "
              f"(recorded {rubric['current_version']})")

    if problems:
        print(f"\n{len(problems)} artifact(s) no longer match provenance.json: "
              f"{', '.join(problems)}", file=sys.stderr)
        return 1
    print("\nall frozen artifacts match provenance.json")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("build-worksheet",
                        help="blind manual-annotation worksheet from validation items")
    p1.add_argument("--items", type=Path, default=MANUAL)
    p1.add_argument("--out", type=Path, required=True)
    p1.add_argument("--key", type=Path, required=True)
    p1.add_argument("--seed", type=int, default=101)
    p1.add_argument("--compact", action="store_true",
                    help="ask only strategy/confidence/meaning/note")

    p2 = sub.add_parser("import-annotations",
                        help="merge a completed worksheet with the hidden record")
    p2.add_argument("--worksheet", type=Path, required=True)
    p2.add_argument("--key", type=Path, required=True)
    p2.add_argument("--out", type=Path, required=True)

    p3 = sub.add_parser("analyze", help="validation tables from the canonical data")
    p3.add_argument("--manual", type=Path, default=MANUAL)
    p3.add_argument("--external", type=Path, default=EXTERNAL)
    p3.add_argument("--out", type=Path, default=None)
    p3.add_argument("--json", type=Path, default=None)

    p4 = sub.add_parser("verify-provenance",
                        help="recompute frozen prompt hashes and compare")
    p4.add_argument("--provenance", type=Path, default=PROVENANCE)

    args = parser.parse_args(argv)
    return {"build-worksheet": cmd_build_worksheet,
            "import-annotations": cmd_import_annotations,
            "analyze": cmd_analyze,
            "verify-provenance": cmd_verify_provenance}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
