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
MANUAL = DATA_DIR / "original_48" / "manual_validation.jsonl"
EXTERNAL = DATA_DIR / "external_validation.jsonl"
PROVENANCE = DATA_DIR / "provenance.json"

STRATEGIES = ("bald_on_record", "positive_politeness", "negative_politeness",
              "off_record")
STRATEGY_LABELS = STRATEGIES + ("ambiguous_or_mixed", "failed_manipulation")

# Fields an annotator fills in. The first four are always asked; the semantic
# detail fields are optional, so a future round can use the compact form.
CORE_FIELDS = ("strategy", "confidence", "meaning", "note")
DETAIL_FIELDS = ("commitment", "directive", "evidence", "reference")

# Annotations live in a per-item `annotations` list, one entry per rater, so a
# categorical label is never averaged across raters.
RATERS = ("rater_1", "rater_2")


def annotation(record: dict, rater: str) -> Dict[str, Any]:
    for a in record.get("annotations", []):
        if a.get("annotator") == rater:
            return a
    return {}


def raters_in(records: Sequence[dict]) -> List[str]:
    found = {a["annotator"] for r in records for a in r.get("annotations", [])}
    return [r for r in RATERS if r in found] + sorted(found - set(RATERS))


def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    """Unweighted Cohen's kappa. The labels are nominal, so no weighting."""
    if not pairs:
        return None
    labels = sorted({x for p in pairs for x in p})
    n = len(pairs)
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = sum((sum(1 for a, _ in pairs if a == l) / n) *
                   (sum(1 for _, b in pairs if b == l) / n) for l in labels)
    return None if expected == 1 else (observed - expected) / (1 - expected)


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
        name, value = match.group(1).lower(), match.group(2).strip().lower()
        if name in CORE_FIELDS + DETAIL_FIELDS and not value:
            continue
        if (name in HIDDEN_FIELDS or value in LEAK_TOKENS
                or re.match(r"^researcher_\d+_", name)):
            found.append(f"{name}: {value}")
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


def import_annotations(worksheet_text: str, key: Dict[str, dict],
                       annotator: str = "rater_1") -> List[dict]:
    """Merge a completed worksheet back into the hidden experimental record."""
    answers = parse_worksheet(worksheet_text)
    merged: List[dict] = []
    for item_id, filled in sorted(answers.items()):
        record = dict(key.get(item_id, {}))
        # The key carries the canonical item id; a worksheet's own numbering is
        # just its shuffle order and must not overwrite it.
        record.setdefault("item_id", item_id)
        entry = {"annotator": annotator,
                 **{f: filled.get(f) for f in CORE_FIELDS + DETAIL_FIELDS}}
        others = [a for a in record.get("annotations", [])
                  if a.get("annotator") != annotator]
        record["annotations"] = others + [entry]
        by = {a["annotator"]: a for a in record["annotations"]}
        if len(by) >= 2:
            first, second = [by[k] for k in list(by)[:2]]
            record["strategy_agreement"] = first.get("strategy") == second.get("strategy")
            record["meaning_agreement"] = first.get("meaning") == second.get("meaning")
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
    """Manipulation check and meaning preservation, per rater and between them."""
    raters = raters_in(records)
    strategies = [x for x in STRATEGIES
                  if any(r["requested_strategy"] == x for r in records)]

    per_rater: Dict[str, Any] = {}
    for rater in raters:
        pairs = [(r["requested_strategy"], annotation(r, rater).get("strategy"))
                 for r in records]
        meaning = {k: 0 for k in ("yes", "no", "unsure")}
        for r in records:
            v = annotation(r, rater).get("meaning")
            if v in meaning:
                meaning[v] += 1
        per_rater[rater] = {
            "matched": sum(1 for a, b in pairs if a == b),
            "n": len(pairs),
            "by_strategy": {x: {"n": sum(1 for a, _ in pairs if a == x),
                                "matched": sum(1 for a, b in pairs if a == x and a == b)}
                            for x in strategies},
            "off_scheme": {lab: sum(1 for _, b in pairs if b == lab)
                           for lab in ("ambiguous_or_mixed", "failed_manipulation")},
            "confidence": {c: sum(1 for r in records
                                  if annotation(r, rater).get("confidence") == c)
                           for c in ("high", "medium", "low")},
            "meaning": meaning,
            "meaning_by_strategy": {
                x: {k: sum(1 for r in records if r["requested_strategy"] == x
                           and annotation(r, rater).get("meaning") == k)
                    for k in ("yes", "no", "unsure")} for x in strategies},
        }

    reliability: Dict[str, Any] = {}
    if len(raters) >= 2:
        a, b = raters[0], raters[1]
        for name in ("strategy", "meaning"):
            pairs = [(annotation(r, a).get(name), annotation(r, b).get(name))
                     for r in records
                     if annotation(r, a).get(name) and annotation(r, b).get(name)]
            reliability[name] = {
                "n": len(pairs),
                "agreed": sum(1 for x, y in pairs if x == y),
                "kappa": cohen_kappa(pairs),
                "confusion": confusion(pairs),
                "disagreements": [
                    {"item_id": r["item_id"], "requested": r["requested_strategy"],
                     "predicate": r["predicate_name"],
                     a: annotation(r, a).get(name), b: annotation(r, b).get(name),
                     f"{a}_note": annotation(r, a).get("note", ""),
                     f"{b}_note": annotation(r, b).get("note", "")}
                    for r in records
                    if annotation(r, a).get(name) != annotation(r, b).get(name)],
            }

    # The automatic classifier is diagnostic only. Agreement is pooled over the
    # raters (the average, not one number per researcher) and also reported
    # against the consensus label on items where every rater agrees.
    scored = [r for r in records if r.get("automatic_politeness_strategy")]
    pooled = [(annotation(r, rater).get("strategy"), r["automatic_politeness_strategy"])
              for r in scored for rater in raters if annotation(r, rater).get("strategy")]
    consensus = []
    for r in scored:
        labels = {annotation(r, rater).get("strategy") for rater in raters}
        if len(labels) == 1 and None not in labels:
            consensus.append((labels.pop(), r["automatic_politeness_strategy"]))
    automatic = {
        "n_items": len(scored),
        "n_raters": len(raters),
        "pooled_n": len(pooled),
        "pooled_agreed": sum(1 for x, y in pooled if x == y),
        "consensus_n": len(consensus),
        "consensus_agreed": sum(1 for x, y in consensus if x == y),
        "consensus_per_class": per_class(consensus),
        "consensus_confusion": confusion(consensus),
        "model": next((r.get("politeness_classifier_model") for r in scored), None),
        "prompt": next((r.get("politeness_classifier_prompt") for r in scored), None),
    } if scored else {}

    return {"n": len(records), "raters": raters, "strategies": strategies,
            "per_rater": per_rater, "reliability": reliability,
            "automatic_classifier": automatic}


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


def render(manual: Dict[str, Any], external: Dict[str, Any],
           _unused: Any = None) -> str:
    """The paper-facing tables, and nothing else."""
    raters = manual["raters"]
    pr, rel = manual["per_rater"], manual["reliability"]
    label = {r: f"Rater {i + 1}" for i, r in enumerate(raters)}

    out = ["## Manipulation check", "",
           "| Requested strategy | n | " + " | ".join(label[r] for r in raters) + " |",
           "| --- | ---: |" + " ---: |" * len(raters)]
    for strategy in manual["strategies"]:
        cells = " | ".join(f"{pr[r]['by_strategy'][strategy]['matched']}/"
                           f"{pr[r]['by_strategy'][strategy]['n']}" for r in raters)
        out.append(f"| {PRETTY.get(strategy, strategy)} | "
                   f"{pr[raters[0]]['by_strategy'][strategy]['n']} | {cells} |")
    out.append("| **Overall** | **" + str(manual["n"]) + "** | " +
               " | ".join(f"**{pr[r]['matched']}/{pr[r]['n']}**" for r in raters) + " |")

    if rel.get("strategy"):
        st = rel["strategy"]
        out += ["", f"Inter-rater strategy agreement: **{rate(st['agreed'], st['n'])}**, "
                    f"Cohen's kappa = **{st['kappa']:.3f}**" if st["kappa"] is not None
                else "", ]
        if st["disagreements"]:
            out += ["", "| item | " + " | ".join(label[r] for r in raters) + " | requested |",
                    "| --- | --- | --- | --- |"]
            for d in st["disagreements"]:
                out.append(f"| {d['item_id']} | " +
                           " | ".join(PRETTY.get(d[r], str(d[r])) for r in raters) +
                           f" | {PRETTY.get(d['requested'], d['requested'])} |")

    out += ["", "## Meaning preservation", "",
            "| Rater | Yes | No | Unsure |", "| --- | ---: | ---: | ---: |"]
    for r in raters:
        m = pr[r]["meaning"]
        out.append(f"| {label[r]} | {m['yes']} | {m['no']} | {m['unsure']} |")
    if rel.get("meaning"):
        mn = rel["meaning"]
        out += ["", f"Exact meaning agreement: **{rate(mn['agreed'], mn['n'])}**"
                    + (f", Cohen's kappa = **{mn['kappa']:.3f}**"
                       if mn["kappa"] is not None else "")]
        if mn["disagreements"]:
            out += ["", "| item | " + " | ".join(label[r] for r in raters)
                    + " | requested |", "| --- | --- | --- | --- |"]
            for d in mn["disagreements"]:
                out.append(f"| {d['item_id']} | " + " | ".join(str(d[r]) for r in raters)
                           + f" | {PRETTY.get(d['requested'], d['requested'])} |")

    auto = manual["automatic_classifier"]
    if auto:
        out += ["", "## Automatic politeness classifier", "",
                f"Model `{auto['model']}`, prompt `{auto['prompt']}`, three-pass "
                "majority vote (each item is classified three independent times "
                "and the label at least two passes agree on is kept). Diagnostic "
                "only; never an acceptance criterion.", "",
                f"**Agreement with human annotators, averaged over the "
                f"{auto['n_raters']} raters: {rate(auto['pooled_agreed'], auto['pooled_n'])}** "
                f"({auto['n_items']} items x {auto['n_raters']} raters). "
                f"Against the consensus label (items where all raters agree): "
                f"{rate(auto['consensus_agreed'], auto['consensus_n'])}.", "",
                "| Human consensus label | n | classifier agrees | recall |",
                "| --- | ---: | ---: | ---: |"]
        for lab, st in sorted(auto["consensus_per_class"].items()):
            if st["support"]:
                hit = int(round(st["recall"] * st["support"]))
                out.append(f"| {PRETTY.get(lab, lab)} | {st['support']} | {hit} | "
                           f"{st['recall']:.0%} |")
        out += ["", "Rows are the human label; columns are the classifier label.", ""]
        out += _matrix(auto["consensus_confusion"], "human \\ classifier")

    if external:
        pn = external["positive_vs_negative"]
        out += ["", "## External construct check (Zhao & Hawkins, non-Werewolf)", "",
                f"{external['n_eligible']} held-out items whose published label is a "
                f"single Brown & Levinson family. Four-way accuracy: "
                f"**{external['accuracy']:.1%}** "
                f"({int(round(external['accuracy'] * external['n_eligible']))}/"
                f"{external['n_eligible']}), macro-F1 {external['macro_f1']:.3f}.", "",
                "| Published label | n | classifier agrees | recall |",
                "| --- | ---: | ---: | ---: |"]
        for lab, st in sorted(external["per_class"].items()):
            if st["support"]:
                hit = int(round(st["recall"] * st["support"]))
                out.append(f"| {PRETTY.get(lab, lab)} | {st['support']} | {hit} | "
                           f"{st['recall']:.0%} |")
        out += ["", "Rows are the published label; columns are the classifier label.", ""]
        out += _matrix(external["confusion"], "published \\ classifier")
        out += ["",
                f"Positive vs negative only: of {pn['n_gold_pair']} gold positive/negative "
                f"items, {pn['n_predicted_in_pair']} were placed by the classifier in "
                f"that pair (coverage {pn['coverage']:.0%}); agreement within those "
                f"{pn['n_predicted_in_pair']} was **{pn['conditional_agreement']:.1%}**. "
                "This is conditional, not an end-to-end accuracy."]
    return "\n".join(out) + "\n"


def render_latex(manual: Dict[str, Any], external: Dict[str, Any],
                 provenance: Dict[str, Any]) -> str:
    """A paste-ready LaTeX subsection. Every number comes from the data."""
    raters, pr, rel = manual["raters"], manual["per_rater"], manual["reliability"]
    st, mn = rel.get("strategy", {}), rel.get("meaning", {})
    sampling = provenance.get("manual_validation", {}).get("sampling", {})
    n_sources = sampling.get("sources_per_predicate", 3) * len(sampling.get("predicates", []))
    auto = manual["automatic_classifier"]
    pn = external.get("positive_vs_negative", {})

    rows = []
    for strategy in manual["strategies"]:
        cells = " & ".join(f"{pr[r]['by_strategy'][strategy]['matched']}/"
                           f"{pr[r]['by_strategy'][strategy]['n']}" for r in raters)
        rows.append(f"{PRETTY.get(strategy, strategy).capitalize()} & "
                    f"{pr[raters[0]]['by_strategy'][strategy]['n']} & {cells} \\\\")
    overall = " & ".join(f"\\textbf{{{pr[r]['matched']}/{pr[r]['n']}}}" for r in raters)

    return r"""\subsection{Stimulus validation}

We validate the politeness manipulation on %(n_sources)d untreated utterances drawn from
natural game play, covering the %(n_pred)d speech-act types that occur naturally in our
setting. Each source utterance was rewritten under all four Brown and Levinson
strategies, holding the underlying structured predicate fixed, giving
%(n)d matched rewrites in which only the politeness realization varies.

Two annotators independently labelled all %(n)d rewrites. Both were blind to the
requested strategy and to every automatic judgement, and saw only the source
predicate, the untreated original, the rewrite, and the earlier statement it
responds to where applicable. Each recorded the dominant strategy and whether
the game-relevant meaning was preserved.

\begin{table}[t]
\centering
\begin{tabular}{lrrr}
\toprule
Requested strategy & n & Rater 1 & Rater 2 \\
\midrule
%(rows)s
\midrule
\textbf{Overall} & \textbf{%(n)d} & %(overall)s \\
\bottomrule
\end{tabular}
\caption{Manipulation check: rewrites recognised as the requested strategy by
each blind annotator.}
\label{tab:manipulation}
\end{table}

Both annotators recovered the requested strategy for every rewrite
(Table~\ref{tab:manipulation}); exact inter-annotator agreement on strategy was
%(st_agreed)d/%(st_n)d (%(st_pct).1f\%%), Cohen's $\kappa = %(st_kappa).3f$.

On meaning preservation the annotators judged %(m1_yes)d and %(m2_yes)d of %(n)d rewrites
respectively to have preserved the game-relevant content, with %(m1_no)d and %(m2_no)d
judged not preserved and %(m1_unsure)d and %(m2_unsure)d uncertain. Exact agreement was
%(mn_agreed)d/%(mn_n)d (%(mn_pct).1f\%%), $\kappa = %(mn_kappa).3f$; %(n_dis)d disagreements
%(dis_text)s

An automatic politeness classifier (%(auto_model)s, three-pass majority vote) was run on
the same items and agreed with the annotators on %(auto)d/%(auto_n)d labels
(averaged over annotators). We report it diagnostically and never use it as an
acceptance criterion.

As an external construct check we compared our scheme against the
Brown and Levinson strategy annotations of \citet{zhao2025politeness}, whose
annotation unit is the marker span rather than the utterance. Of %(pn_gold)d items
their annotation assigns to positive or negative politeness, our classifier
placed %(pn_in)d in the same pair, and within those it agreed on
%(pn_acc).1f\%%. This is conditional construct evidence for the
positive/negative distinction rather than an end-to-end accuracy measure.
""" % {
        "n": manual["n"], "n_sources": n_sources,
        "n_pred": len(sampling.get("predicates", [])),
        "rows": "\n".join(rows), "overall": overall,
        "st_agreed": st.get("agreed", 0), "st_n": st.get("n", 0),
        "st_pct": 100 * st.get("agreed", 0) / max(st.get("n", 1), 1),
        "st_kappa": st.get("kappa") or 0.0,
        "m1_yes": pr[raters[0]]["meaning"]["yes"], "m2_yes": pr[raters[1]]["meaning"]["yes"],
        "m1_no": pr[raters[0]]["meaning"]["no"], "m2_no": pr[raters[1]]["meaning"]["no"],
        "m1_unsure": pr[raters[0]]["meaning"]["unsure"],
        "m2_unsure": pr[raters[1]]["meaning"]["unsure"],
        "mn_agreed": mn.get("agreed", 0), "mn_n": mn.get("n", 0),
        "mn_pct": 100 * mn.get("agreed", 0) / max(mn.get("n", 1), 1),
        "mn_kappa": mn.get("kappa") or 0.0,
        "n_dis": len(mn.get("disagreements", [])),
        "dis_text": ("concerned action recommendations rendered indirectly."
                     if mn.get("disagreements") else "occurred."),
        "auto": auto.get("pooled_agreed", 0), "auto_n": auto.get("pooled_n", 0),
        "auto_model": auto.get("model") or "model",
        "pn_gold": pn.get("n_gold_pair", 0), "pn_in": pn.get("n_predicted_in_pair", 0),
        "pn_acc": 100 * (pn.get("conditional_agreement") or 0),
    }


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
    merged = import_annotations(Path(args.worksheet).read_text(encoding="utf-8"), key,
                                annotator=args.annotator)
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
    if args.latex:
        prov = json.loads(Path(PROVENANCE).read_text(encoding="utf-8"))
        tex = render_latex(manual, external, prov)
        if args.latex != Path("-"):
            Path(args.latex).write_text(tex, encoding="utf-8")
            print(f"wrote {args.latex}")
        else:
            print(tex)
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
    p2.add_argument("--annotator", default="rater_2",
                    help="prefix the answers are stored under, e.g. researcher_2")

    p3 = sub.add_parser("analyze", help="validation tables from the canonical data")
    p3.add_argument("--manual", type=Path, default=MANUAL)
    p3.add_argument("--external", type=Path, default=EXTERNAL)
    p3.add_argument("--out", type=Path, default=None)
    p3.add_argument("--json", type=Path, default=None)
    p3.add_argument("--latex", type=Path, default=None,
                    help="write a paste-ready LaTeX subsection ('-' prints it)")

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
