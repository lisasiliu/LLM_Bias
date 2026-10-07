"""Human listener annotation: blind pairwise worksheets for two researchers, import of their labels,
inter-rater reliability, consensus, and the comparison with the model listeners on the same pairs."""
from __future__ import annotations
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from components.helpers import read_jsonl, write_jsonl
from components.stats import mean
from experiments.proficiency_validation.rewrite.runtime import (
    DATA_DIR, condition_label, dump_json, now_iso, sha256_file, update_metadata)
from experiments.proficiency_validation.validation.sources import matched_complete

# (first, second): the signed score is +1 when the first condition is judged more limited.
COMPARISONS: Tuple[Tuple[str, str], ...] = (("low", "neutral"), ("neutral", "high"), ("low", "high"), ("neutral", "original"))
EXPECTED_SIGN = {("low", "neutral"): 1, ("neutral", "high"): 1, ("low", "high"): 1, ("neutral", "original"): 0}
LABELS = ("A", "B", "same")
RATERS = ("researcher_1", "researcher_2")
WORKSHEET_VERSION = "range_annotation"
DEFINITION = ("A more limited range of English means more common or repetitive vocabulary, less lexical "
              "variety, and simpler or narrower sentence structures with less syntactic embedding.")
QUESTION = "Which passage appears to use a more limited range of English?"


def comparison_key(first: str, second: str) -> str:
    return f"{first}:{second}"


def comparison_title(first: str, second: str) -> str:
    return f"{condition_label(first)} vs {condition_label(second)}"


# ── pairs and worksheets ──

def build_pairs(assembled: Sequence[dict], *, seed: int = 101, per_comparison: Optional[int] = None,
                comparisons: Sequence[Tuple[str, str]] = COMPARISONS) -> List[dict]:
    """One blind pair per complete passage and comparison. The side each condition takes is drawn
    once per pair with a seeded generator; a per-comparison cap takes passages game by game."""
    texts = {(r["bundle_id"], r["condition"]): r for r in assembled if r.get("complete") and r.get("text")}
    pairs = []
    for first, second in comparisons:
        ids = matched_complete(assembled, (first, second))
        if per_comparison is not None:
            by_game: Dict[str, List[str]] = defaultdict(list)
            for b in ids:
                by_game[texts[(b, first)]["game_id"]].append(b)
            picked: List[str] = []
            while len(picked) < per_comparison and any(by_game.values()):
                for game in sorted(by_game):
                    if by_game[game] and len(picked) < per_comparison:
                        picked.append(by_game[game].pop(0))
            ids = picked
        for bundle_id in ids:
            pair_id = f"{bundle_id}|{comparison_key(first, second)}"
            a, b = (first, second) if random.Random(f"{seed}:{pair_id}").random() < .5 else (second, first)
            row = texts[(bundle_id, first)]
            pairs.append({"pair_id": pair_id, "bundle_id": bundle_id, "game_id": row["game_id"], "speaker_id": row["speaker"],
                          "comparison_type": comparison_key(first, second), "first_condition": first, "second_condition": second,
                          "passage_a_condition": a, "passage_b_condition": b,
                          "presentation_order": "ab" if a == first else "ba",
                          "passage_a_text": passage_text(texts[(bundle_id, a)]), "passage_b_text": passage_text(texts[(bundle_id, b)])})
    return pairs


def passage_text(row: dict) -> str:
    return "\n\n".join(row["utterances"]) if row.get("utterances") else row["text"]


def worksheet_items(pairs: Sequence[dict], rater: str, *, seed: int = 101) -> List[dict]:
    """The rater's own item order; the item number is that order and nothing else."""
    order = list(pairs)
    random.Random(f"{seed}:{rater}").shuffle(order)
    return [{"item_id": f"item_{i:03d}", **p} for i, p in enumerate(order, 1)]


def render_worksheet(items: Sequence[dict], rater: str) -> str:
    lines = [f"# Linguistic range: blind annotation sheet ({rater})", "",
             "Each item shows two passages, A and B. Each passage is one player's public speech from one "
             "Werewolf game, in speech order. The two passages of an item come from the same player and game.", "",
             f"**{QUESTION}**", "", DEFINITION, "",
             "Judge only the language. Ignore strategic skill, the truth of claims, persuasiveness, politeness, "
             "player identities and how much the speaker talks.", "",
             "For each item fill in:", "",
             "- `more_limited:` `A` / `B` / `same` (no meaningful difference)",
             "- `note:` one line, optional", "",
             "Work through the items in order and finish before discussing any item with the other researcher.", ""]
    for item in items:
        lines += ["---", "", f"### {item['item_id']}", "", "**Passage A**", ""]
        lines += [f"> {l}" if l else ">" for l in item["passage_a_text"].splitlines()]
        lines += ["", "**Passage B**", ""]
        lines += [f"> {l}" if l else ">" for l in item["passage_b_text"].splitlines()]
        lines += ["", "more_limited:", "note:", ""]
    return "\n".join(lines)



def worksheet_key(items: Sequence[dict]) -> Dict[str, dict]:
    hidden = ("pair_id", "bundle_id", "game_id", "speaker_id", "comparison_type", "first_condition", "second_condition",
              "passage_a_condition", "passage_b_condition", "presentation_order")
    return {item["item_id"]: {k: item[k] for k in hidden} for item in items}


def write_worksheets(assembled_path: Path, out_dir: Path, *, seed: int = 101, per_comparison: Optional[int] = None,
                     raters: Sequence[str] = RATERS) -> dict:
    assembled = read_jsonl(assembled_path)
    pairs = build_pairs(assembled, seed=seed, per_comparison=per_comparison)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rater in raters:
        items = worksheet_items(pairs, rater, seed=seed)
        md, key_path = out_dir / f"{rater}_worksheet.md", out_dir / f"{rater}_key.json"
        md.write_text(render_worksheet(items, rater), encoding="utf-8")
        dump_json(key_path, {"worksheet_version": WORKSHEET_VERSION, "created_at": now_iso(), "seed": seed,
                             "assembled_sha256": sha256_file(assembled_path), "rater": rater, "items": worksheet_key(items)})
        written += [md, key_path]
    return {"n_pairs": len(pairs), "by_comparison": dict(Counter(p["comparison_type"] for p in pairs)),
            "by_game": dict(Counter(p["game_id"] for p in pairs)), "files": [str(p) for p in written]}


# ── import ──

def parse_worksheet(text: str) -> Dict[str, Dict[str, str]]:
    """``{item_id: {"more_limited": label, "note": text}}`` from a filled-in markdown sheet."""
    out: Dict[str, Dict[str, str]] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^#+\s*(item_\d+)\s*$", line.strip())
        if heading:
            current = heading.group(1)
            out[current] = {}
            continue
        if current is None or line.startswith(">"):
            continue
        match = re.match(r"^(more_limited|note):\s*`?([^`]*)`?\s*$", line.strip())
        if match and match.group(2).strip():
            field, value = match.group(1), match.group(2).strip()
            out[current][field] = normalize_label(value) if field == "more_limited" else value
    return {k: v for k, v in out.items() if v.get("more_limited")}



def normalize_label(value: str) -> str:
    v = value.strip().strip('"\'`').lower()
    if v in ("a", "passage a"):
        return "A"
    if v in ("b", "passage b"):
        return "B"
    if v in ("same", "no difference", "no meaningful difference", "neither", "tie", "equal"):
        return "same"
    raise ValueError(f"unreadable label {value!r}; use A, B or same")


def condition_level(raw_label: str, passage_a_condition: str, passage_b_condition: str) -> str:
    return {"A": passage_a_condition, "B": passage_b_condition, "same": "same"}[raw_label]


def import_annotations(answers: Dict[str, Dict[str, str]], key: dict, rater: str) -> List[dict]:
    """One row per judgment in the shared annotation schema."""
    items = key["items"]
    missing = sorted(set(items) - set(answers))
    if missing:
        raise ValueError(f"{len(missing)} items without a label: {', '.join(missing[:5])}")
    unknown = sorted(set(answers) - set(items))
    if unknown:
        raise ValueError(f"items outside the worksheet: {', '.join(unknown[:5])}")
    rows = []
    for item_id, filled in sorted(answers.items()):
        k = items[item_id]
        rows.append({"pair_id": k["pair_id"], "game_id": k["game_id"], "speaker_id": k["speaker_id"],
                     "comparison_type": k["comparison_type"], "passage_a_condition": k["passage_a_condition"],
                     "passage_b_condition": k["passage_b_condition"], "presentation_order": k["presentation_order"],
                     "rater_type": "human", "rater_id": rater, "item_id": item_id, "raw_label": filled["more_limited"],
                     "condition_level_label": condition_level(filled["more_limited"], k["passage_a_condition"], k["passage_b_condition"]),
                     "note": filled.get("note", ""), "consensus_label": None, "worksheet_version": key.get("worksheet_version"),
                     "assembled_sha256": key.get("assembled_sha256"), "imported_at": now_iso()})
    return rows


def merge_rater(existing: Sequence[dict], rows: Sequence[dict], rater: str) -> List[dict]:
    """Replace one rater's rows; the other raters' raw labels stay as imported."""
    others = [r for r in existing if r["rater_id"] != rater]
    hashes = {r.get("assembled_sha256") for r in others} | {r.get("assembled_sha256") for r in rows}
    if len(hashes) > 1:
        raise ValueError("annotations come from worksheets built on different passages")
    return others + list(rows)


# ── reliability and consensus ──

def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    n = len(pairs)
    if not n:
        return None
    categories = {a for a, _ in pairs} | {b for _, b in pairs}
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in categories)
    return 1.0 if pe >= 1 else (po - pe) / (1 - pe)


def relative_label(row: dict, first: str) -> str:
    """``first`` / ``second`` / ``same``: the condition-level label relative to the comparison."""
    label = row["condition_level_label"]
    return "same" if label == "same" else ("first" if label == first else "second")


def signed(row: dict, first: str) -> int:
    return {"first": 1, "second": -1, "same": 0}[relative_label(row, first)]


def human_rows(rows: Iterable[dict]) -> List[dict]:
    return [r for r in rows if r["rater_type"] == "human"]


def reliability(rows: Sequence[dict]) -> dict:
    """Percent agreement and Cohen's kappa between the two human raters on comparison-relative labels."""
    raters = sorted({r["rater_id"] for r in human_rows(rows)})
    if len(raters) != 2:
        return {"raters": raters, "n_pairs": 0}
    by_pair: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in human_rows(rows):
        by_pair[r["pair_id"]][r["rater_id"]] = r
    shared = {p: v for p, v in by_pair.items() if len(v) == 2}

    def relative_pairs(subset):
        return [(relative_label(v[raters[0]], v[raters[0]]["comparison_type"].split(":")[0]),
                 relative_label(v[raters[1]], v[raters[1]]["comparison_type"].split(":")[0])) for v in subset.values()]
    overall = relative_pairs(shared)
    out = {"raters": raters, "n_pairs": len(shared), "percent_agreement": mean([float(a == b) for a, b in overall]),
           "cohen_kappa": cohen_kappa(overall), "confusion": _confusion(overall), "by_comparison": {}}
    for first, second in COMPARISONS:
        subset = {p: v for p, v in shared.items() if v[raters[0]]["comparison_type"] == comparison_key(first, second)}
        if subset:
            rel = relative_pairs(subset)
            out["by_comparison"][comparison_key(first, second)] = {
                "n_pairs": len(rel), "percent_agreement": mean([float(a == b) for a, b in rel]), "cohen_kappa": cohen_kappa(rel)}
    return out


def _confusion(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
    table: Dict[str, Dict[str, int]] = {a: {b: 0 for b in ("first", "second", "same")} for a in ("first", "second", "same")}
    for a, b in pairs:
        table[a][b] += 1
    return table


def apply_consensus(rows: Sequence[dict], decisions: Optional[Dict[str, str]] = None) -> Tuple[List[dict], List[str]]:
    """Agreed pairs take the shared label as consensus; disagreements take the adjudicated condition-level
    label from ``decisions`` (pair_id -> condition or "same"). Returns the rows and the unresolved pair ids."""
    decisions = decisions or {}
    by_pair: Dict[str, List[dict]] = defaultdict(list)
    for r in human_rows(rows):
        by_pair[r["pair_id"]].append(r)
    unresolved = []
    out = [dict(r) for r in rows]
    for pair_id, group in by_pair.items():
        labels = {r["condition_level_label"] for r in group}
        if len(group) == 2 and len(labels) == 1:
            consensus = labels.pop()
        elif pair_id in decisions:
            consensus = decisions[pair_id]
            allowed = {group[0]["passage_a_condition"], group[0]["passage_b_condition"], "same"}
            if consensus not in allowed:
                raise ValueError(f"{pair_id}: consensus {consensus!r} is not one of {sorted(allowed)}")
        else:
            unresolved.append(pair_id)
            continue
        for r in out:
            if r["rater_type"] == "human" and r["pair_id"] == pair_id:
                r["consensus_label"] = consensus
    return out, sorted(unresolved)


def render_consensus_worksheet(rows: Sequence[dict], pairs_by_id: Dict[str, dict]) -> str:
    """Disagreements only, with both raw labels shown, for the joint adjudication."""
    by_pair: Dict[str, List[dict]] = defaultdict(list)
    for r in human_rows(rows):
        by_pair[r["pair_id"]].append(r)
    lines = ["# Linguistic range: consensus sheet", "",
             "Items where the two independent labels differ. Agree on one label per item and fill in "
             "`consensus:` with `A`, `B` or `same`, using the same passage sides as shown here.", ""]
    n = 0
    for pair_id in sorted(by_pair):
        group = by_pair[pair_id]
        if len(group) == 2 and len({r["condition_level_label"] for r in group}) == 1:
            continue
        pair = pairs_by_id[pair_id]
        n += 1
        lines += ["---", "", f"### {pair_id}", ""]
        for r in sorted(group, key=lambda r: r["rater_id"]):
            lines.append(f"- {r['rater_id']}: `{r['raw_label']}`" + (f" ({r['note']})" if r.get("note") else ""))
        lines += ["", "**Passage A**", ""] + [f"> {l}" if l else ">" for l in pair["passage_a_text"].splitlines()]
        lines += ["", "**Passage B**", ""] + [f"> {l}" if l else ">" for l in pair["passage_b_text"].splitlines()]
        lines += ["", "consensus:", ""]
    lines.insert(3, f"{n} items.")
    return "\n".join(lines)


def parse_consensus(text: str, pairs_by_id: Dict[str, dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^###\s+(\S+)\s*$", line.strip())
        if heading:
            current = heading.group(1)
            continue
        match = re.match(r"^consensus:\s*`?([^`]*)`?\s*$", line.strip())
        if current and match and match.group(1).strip():
            pair = pairs_by_id[current]
            out[current] = condition_level(normalize_label(match.group(1)), pair["passage_a_condition"], pair["passage_b_condition"])
    return out


# ── model listeners on the same pairs ──

def model_rows(listener_rows: Iterable[dict], pairs: Sequence[dict]) -> List[dict]:
    """Model listener judgments on the annotated passages, in the annotation schema (one row per order)."""
    wanted = {(p["bundle_id"], frozenset((p["first_condition"], p["second_condition"]))): p for p in pairs}
    out = []
    for r in listener_rows:
        if r.get("construct", "range") != "range" or not r.get("ok"):
            continue
        pair = wanted.get((r["bundle_id"], frozenset((r["treatment"], r["comparator"]))))
        if pair is None:
            continue
        a, b = r["left_condition"], r["right_condition"]
        label = "same" if r.get("tie") else r["chosen_condition"]
        out.append({"pair_id": pair["pair_id"], "game_id": r["game_id"], "speaker_id": r["speaker"],
                    "comparison_type": pair["comparison_type"], "passage_a_condition": a, "passage_b_condition": b,
                    "presentation_order": "ab" if a == pair["first_condition"] else "ba",
                    "rater_type": "model", "rater_id": r["listener"], "rater_family": r.get("listener_family"),
                    "raw_label": r["choice"], "condition_level_label": label, "consensus_label": None})
    return out


# ── summary ──

def summary(rows: Sequence[dict], listener_rows: Sequence[dict] = (), pairs: Sequence[dict] = ()) -> dict:
    humans = human_rows(rows)
    pairs = list(pairs) or _pairs_from_rows(humans)
    models = model_rows(listener_rows, pairs)
    consensus = {r["pair_id"]: r["consensus_label"] for r in humans if r.get("consensus_label")}
    raters = sorted({r["rater_id"] for r in humans})
    listeners = sorted({r["rater_id"] for r in models})
    out = {"n_pairs": len(pairs), "raters": raters, "listeners": listeners, "reliability": reliability(rows),
           "n_consensus": len(consensus), "comparisons": {}, "agreement_with_consensus": {}, "order_consistency": {},
           "by_game": {}}
    for first, second in COMPARISONS:
        key = comparison_key(first, second)
        block = {"title": comparison_title(first, second), "first": first, "second": second,
                 "expected_sign": EXPECTED_SIGN[(first, second)], "n_pairs": sum(p["comparison_type"] == key for p in pairs), "raters": {}}
        for rater in raters:
            block["raters"][rater] = _direction([r for r in humans if r["comparison_type"] == key and r["rater_id"] == rater], first)
        cons = [{"condition_level_label": label} for pid, label in consensus.items() if pid.endswith("|" + key)]
        if cons:
            block["raters"]["consensus"] = _direction(cons, first)
        for listener in listeners:
            block["raters"][listener] = _direction([r for r in models if r["comparison_type"] == key and r["rater_id"] == listener], first)
        out["comparisons"][key] = block
    for listener in listeners:
        judged = [r for r in models if r["rater_id"] == listener]
        matched = [(relative_label({"condition_level_label": consensus[r["pair_id"]]}, r["comparison_type"].split(":")[0]),
                    relative_label(r, r["comparison_type"].split(":")[0])) for r in judged if r["pair_id"] in consensus]
        out["agreement_with_consensus"][listener] = {"n_judgments": len(matched),
            "percent_agreement": mean([float(a == b) for a, b in matched]), "cohen_kappa": cohen_kappa(matched)}
        by_pair: Dict[str, set] = defaultdict(set)
        for r in judged:
            by_pair[r["pair_id"]].add(r["condition_level_label"])
        both = {p: v for p, v in by_pair.items() if len([r for r in judged if r["pair_id"] == p]) == 2}
        out["order_consistency"][listener] = {"n_pairs": len(both),
            "same_label_both_orders": sum(len(v) == 1 for v in both.values()),
            "share": mean([float(len(v) == 1) for v in both.values()])}
    if len(raters) == 2:
        human_pairs: Dict[str, Dict[str, dict]] = defaultdict(dict)
        for r in humans:
            human_pairs[r["pair_id"]][r["rater_id"]] = r
        shared = [(relative_label(v[raters[0]], v[raters[0]]["comparison_type"].split(":")[0]),
                   relative_label(v[raters[1]], v[raters[1]]["comparison_type"].split(":")[0])) for v in human_pairs.values() if len(v) == 2]
        out["agreement_with_consensus"]["human_pair"] = {"n_judgments": len(shared),
            "percent_agreement": mean([float(a == b) for a, b in shared]), "cohen_kappa": cohen_kappa(shared)}
    for game in sorted({p["game_id"] for p in pairs}):
        out["by_game"][game] = {}
        for first, second in COMPARISONS:
            key = comparison_key(first, second)
            subset = [r for r in humans if r["game_id"] == game and r["comparison_type"] == key]
            if subset:
                out["by_game"][game][key] = {"n_pairs": len({r["pair_id"] for r in subset}), **_direction(subset, first)}
    return out


def _pairs_from_rows(rows: Sequence[dict]) -> List[dict]:
    seen: Dict[str, dict] = {}
    for r in rows:
        first, second = r["comparison_type"].split(":")
        seen.setdefault(r["pair_id"], {"pair_id": r["pair_id"], "bundle_id": r["pair_id"].split("|")[0], "game_id": r["game_id"],
                                      "speaker_id": r["speaker_id"], "comparison_type": r["comparison_type"],
                                      "first_condition": first, "second_condition": second})
    return list(seen.values())


def _direction(rows: Sequence[dict], first: str) -> dict:
    labels = [relative_label(r, first) for r in rows]
    n = len(labels)
    return {"n_judgments": n, "first_more_limited": labels.count("first") / n if n else None,
            "second_more_limited": labels.count("second") / n if n else None,
            "same": labels.count("same") / n if n else None,
            "signed_preference": mean([{"first": 1, "second": -1, "same": 0}[l] for l in labels])}


def render_summary(s: dict) -> List[str]:
    """Markdown section: reliability, direction per comparison for every rater, agreement with consensus."""
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    num = lambda v: "n/a" if v is None else f"{v:+.2f}"
    rel = s["reliability"]
    lines = ["## Human listeners", "",
             f"Two researchers labelled {s['n_pairs']} blind passage pairs ({QUESTION.lower()[:-1]}: A, B or same). "
             "Consensus labels resolve the disagreements after the independent labels were recorded. "
             "Model listeners answered the same pairs in both presentation orders.", ""]
    if rel.get("n_pairs"):
        lines += [f"Human agreement on {rel['n_pairs']} shared pairs: {pct(rel['percent_agreement'])}, "
                  f"Cohen's κ = {rel['cohen_kappa']:.2f}.", "",
                  "| Comparison | Pairs | Agreement | κ |", "| --- | ---: | ---: | ---: |"]
        for key, b in rel["by_comparison"].items():
            first, second = key.split(":")
            kappa = "n/a" if b["cohen_kappa"] is None else f"{b['cohen_kappa']:.2f}"
            lines.append(f"| {comparison_title(first, second)} | {b['n_pairs']} | {pct(b['percent_agreement'])} | {kappa} |")
        lines.append("")
    lines += ["| Comparison | Rater | Judgments | First more limited | Second more limited | Same | Signed preference (+ = first more limited) |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for block in s["comparisons"].values():
        for rater, d in block["raters"].items():
            lines.append(f"| {block['title']} | {rater} | {d['n_judgments']} | {pct(d['first_more_limited'])} | "
                         f"{pct(d['second_more_limited'])} | {pct(d['same'])} | {num(d['signed_preference'])} |")
    lines.append("")
    if s["agreement_with_consensus"]:
        lines += ["| Rater | Judgments compared with consensus | Agreement | κ |", "| --- | ---: | ---: | ---: |"]
        for name, a in s["agreement_with_consensus"].items():
            kappa = "n/a" if a["cohen_kappa"] is None else f"{a['cohen_kappa']:.2f}"
            label = "second human rater (vs first)" if name == "human_pair" else name
            lines.append(f"| {label} | {a['n_judgments']} | {pct(a['percent_agreement'])} | {kappa} |")
        lines.append("")
    if s["order_consistency"]:
        lines += ["| Model listener | Pairs | Same label in both orders |", "| --- | ---: | ---: |"]
        for name, o in s["order_consistency"].items():
            lines.append(f"| {name} | {o['n_pairs']} | {o['same_label_both_orders']} ({pct(o['share'])}) |")
        lines.append("")
    if s["by_game"]:
        lines += ["| Game | Comparison | Pairs | Human signed preference | Same |", "| --- | --- | ---: | ---: | ---: |"]
        for game, comps in s["by_game"].items():
            for key, d in comps.items():
                first, second = key.split(":")
                lines.append(f"| {game} | {comparison_title(first, second)} | {d['n_pairs']} | {num(d['signed_preference'])} | {pct(d['same'])} |")
        lines.append("")
    return lines


# ── files ──

MANUAL_FILE = DATA_DIR.parent / "manual_validation.jsonl"


def load_pairs_from_keys(data_dir: Path, assembled_path: Optional[Path] = None) -> Dict[str, dict]:
    """Pairs as the worksheets presented them, with texts when the assembled passages are given."""
    pairs: Dict[str, dict] = {}
    for key_path in sorted(data_dir.glob("*_key.json")):
        key = json.loads(key_path.read_text(encoding="utf-8"))
        for item in key["items"].values():
            pairs.setdefault(item["pair_id"], dict(item))
    if assembled_path is not None:
        texts = {(r["bundle_id"], r["condition"]): r for r in read_jsonl(assembled_path)}
        for p in pairs.values():
            p["passage_a_text"] = passage_text(texts[(p["bundle_id"], p["passage_a_condition"])])
            p["passage_b_text"] = passage_text(texts[(p["bundle_id"], p["passage_b_condition"])])
    return pairs


def read_answers(path: Path) -> Dict[str, Dict[str, str]]:
    return parse_worksheet(path.read_text(encoding="utf-8"))


# ── commands ──

def add_parser(sub) -> None:
    p = sub.add_parser("build-worksheets", help="blind pairwise worksheets and hidden keys for two researchers")
    p.add_argument("--assembled", type=Path, required=True, help="the run's linguistic_features_assembled.jsonl")
    p.add_argument("--out-dir", type=Path, default=DATA_DIR.parent)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--per-comparison", type=int, default=None, help="cap pairs per comparison, taken game by game")
    p.add_argument("--raters", default=",".join(RATERS))
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("import-annotations", help="import one researcher's filled worksheet into manual_validation.jsonl")
    p.add_argument("--worksheet", type=Path, required=True, help="the filled worksheet")
    p.add_argument("--key", type=Path, required=True)
    p.add_argument("--rater", required=True)
    p.add_argument("--out", type=Path, default=MANUAL_FILE)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("consensus-worksheet", help="list the disagreements for joint adjudication")
    p.add_argument("--annotations", type=Path, default=MANUAL_FILE)
    p.add_argument("--assembled", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR.parent)
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_consensus_sheet)

    p = sub.add_parser("import-consensus", help="record the adjudicated labels")
    p.add_argument("--worksheet", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=MANUAL_FILE)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR.parent)
    p.set_defaults(func=cmd_import_consensus)

    p = sub.add_parser("annotation-summary", help="reliability, human direction per comparison and agreement with the model listeners")
    p.add_argument("--annotations", type=Path, default=MANUAL_FILE)
    p.add_argument("--listener", type=Path, nargs="*", default=[], help="range listener files for the same run")
    p.add_argument("--out", type=Path, default=None, help="markdown output; a .json twin is written next to it")
    p.set_defaults(func=cmd_summary)


def cmd_build(args) -> int:
    result = write_worksheets(args.assembled, args.out_dir, seed=args.seed, per_comparison=args.per_comparison,
                              raters=[r.strip() for r in args.raters.split(",") if r.strip()])
    print(f"{result['n_pairs']} pairs {result['by_comparison']} across games {result['by_game']}")
    for f in result["files"]:
        print(f"  {f}")
    return 0


def cmd_import(args) -> int:
    key = json.loads(args.key.read_text(encoding="utf-8"))
    rows = import_annotations(read_answers(args.worksheet), key, args.rater)
    existing = read_jsonl(args.out) if args.out.exists() else []
    merged = merge_rater(existing, rows, args.rater)
    merged, unresolved = apply_consensus(merged, {r["pair_id"]: r["consensus_label"] for r in merged if r.get("consensus_label")})
    write_jsonl(args.out, sorted(merged, key=lambda r: (r["pair_id"], r["rater_id"])))
    update_metadata(args.out, "summary", {"created_at": now_iso(), "raters": sorted({r["rater_id"] for r in merged}),
                                          "n_rows": len(merged), "unresolved_pairs": len(unresolved)})
    print(f"{len(rows)} judgments by {args.rater} -> {args.out}; {len(unresolved)} pairs await consensus")
    return 0


def cmd_consensus_sheet(args) -> int:
    rows = read_jsonl(args.annotations)
    pairs = load_pairs_from_keys(args.data_dir, args.assembled)
    text = render_consensus_worksheet(rows, pairs)
    out = args.out or args.data_dir / "consensus_worksheet.md"
    out.write_text(text, encoding="utf-8")
    print(f"consensus sheet -> {out}")
    return 0


def cmd_import_consensus(args) -> int:
    rows = read_jsonl(args.annotations)
    pairs = load_pairs_from_keys(args.data_dir)
    decisions = parse_consensus(args.worksheet.read_text(encoding="utf-8"), pairs)
    rows, unresolved = apply_consensus(rows, decisions)
    write_jsonl(args.annotations, rows)
    update_metadata(args.annotations, "consensus", {"created_at": now_iso(), "adjudicated_pairs": len(decisions),
                                                    "unresolved_pairs": unresolved})
    print(f"{len(decisions)} adjudicated pairs recorded; {len(unresolved)} unresolved")
    return 0


def cmd_summary(args) -> int:
    rows = read_jsonl(args.annotations)
    listener_rows = [r for path in args.listener for r in read_jsonl(path)]
    s = summary(rows, listener_rows)
    text = "\n".join(render_summary(s))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        dump_json(args.out.with_suffix(".json"), s)
        print(f"summary -> {args.out}")
    else:
        print(text)
    return 0
