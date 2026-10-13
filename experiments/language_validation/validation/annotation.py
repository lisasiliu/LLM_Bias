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
from experiments.language_validation.rewrite.runtime import (
    DATA_DIR, assert_unique, condition_label, digest, dump_json, now_iso, read_metadata,
    sha256_file, sha256_text, update_metadata)
from experiments.language_validation.validation.analysis import complete_listener_rows
from experiments.language_validation.validation.sources import matched_complete

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
    assert_unique(assembled, ("bundle_id", "condition"))
    if per_comparison is not None and per_comparison < 1:
        raise ValueError("per_comparison must be positive")
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
    return {item["item_id"]: {**{k: item[k] for k in hidden},
                            **{f"passage_{side}_sha256": sha256_text(item[f"passage_{side}_text"]) for side in ("a", "b")}}
            for item in items}


def worksheet_hash(text: str, fields=("more_limited", "note")) -> str:
    """Hash the sheet's fixed content, with answer fields left blank."""
    pattern = re.compile(r"^(" + "|".join(fields) + r"):\s*.*$")
    lines = []
    for line in text.splitlines():
        match = pattern.match(line.strip())
        lines.append(match.group(1) + ":" if match else line.rstrip())
    return sha256_text("\n".join(lines).strip())


def key_record(items, rater, assembled_sha256, *, seed=101, raters=RATERS):
    return {"worksheet_version": WORKSHEET_VERSION, "seed": seed, "assembled_sha256": assembled_sha256,
            "rater": rater, "expected_raters": list(raters), "items": worksheet_key(items),
            "worksheet_sha256": worksheet_hash(render_worksheet(items, rater))}


def write_worksheets(assembled_path: Path, out_dir: Path, *, seed: int = 101, per_comparison: Optional[int] = None,
                     raters: Sequence[str] = RATERS) -> dict:
    assembled = read_jsonl(assembled_path)
    pairs = build_pairs(assembled, seed=seed, per_comparison=per_comparison)
    if not pairs or len(raters) != 2 or len(set(raters)) != 2:
        raise ValueError("worksheets require passages and two distinct raters")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", r) for r in raters):
        raise ValueError("rater names must contain letters, numbers, underscores or hyphens")
    planned = []
    for rater in raters:
        items = worksheet_items(pairs, rater, seed=seed)
        md, key_path = out_dir / f"{rater}_worksheet.md", out_dir / f"{rater}_key.json"
        text = render_worksheet(items, rater)
        key = key_record(items, rater, sha256_file(assembled_path), seed=seed, raters=raters)
        if md.exists() or key_path.exists():
            if not (md.exists() and key_path.exists()):
                raise ValueError("worksheet/key file is missing; choose a new output directory")
            recorded = json.loads(key_path.read_text(encoding="utf-8"))
            if (any(recorded.get(k) != v for k, v in key.items())
                    or worksheet_hash(md.read_text(encoding="utf-8")) != key["worksheet_sha256"]):
                raise ValueError("existing worksheet materials differ; choose a new output directory")
        else:
            planned.append((md, key_path, text, key))
    out_dir.mkdir(parents=True, exist_ok=True)
    for md, key_path, text, key in planned:
        md.write_text(text, encoding="utf-8")
        dump_json(key_path, {**key, "created_at": now_iso()})
    written = [out_dir / f"{r}_{suffix}" for r in raters for suffix in ("worksheet.md", "key.json")]
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
            if current in out:
                raise ValueError(f"duplicate worksheet item: {current}")
            out[current] = {}
            continue
        if current is None or line.startswith(">"):
            continue
        match = re.match(r"^(more_limited|note):\s*(.*?)\s*$", line.strip())
        if match and match.group(2).strip():
            field, value = match.group(1), match.group(2).strip()
            if field in out[current]:
                raise ValueError(f"duplicate {field} for {current}")
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


def import_annotations(text: str, key: dict, rater: str) -> List[dict]:
    """One row per judgment in the shared annotation schema."""
    if key.get("rater") != rater:
        raise ValueError("rater does not match the worksheet key")
    if key.get("worksheet_version") != WORKSHEET_VERSION or not key.get("assembled_sha256"):
        raise ValueError("worksheet key has incomplete material provenance")
    if worksheet_hash(text) != key.get("worksheet_sha256"):
        raise ValueError("worksheet does not match the key's text and item order")
    answers = parse_worksheet(text)
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
                     "note": filled.get("note", ""), "consensus_label": None, "consensus_basis_sha256": None,
                     "worksheet_version": key["worksheet_version"], "expected_raters": key["expected_raters"],
                     "worksheet_sha256": key["worksheet_sha256"],
                     "passage_a_sha256": k["passage_a_sha256"], "passage_b_sha256": k["passage_b_sha256"],
                     "assembled_sha256": key.get("assembled_sha256"), "imported_at": now_iso()})
    human_groups(rows)
    return rows


def merge_rater(existing: Sequence[dict], rows: Sequence[dict], rater: str) -> List[dict]:
    """Replace one rater's rows; the other raters' raw labels stay as imported."""
    human_groups(existing)
    others = [r for r in existing if r["rater_id"] != rater]
    if not rows or any(r["rater_id"] != rater for r in rows):
        raise ValueError("import must contain one rater's complete labels")
    previous = {r["pair_id"]: r for r in existing if r["rater_id"] == rater}
    incoming = [dict(r) for r in rows]
    for r in incoming:
        old = previous.get(r["pair_id"])
        if old and label_basis([old]) == label_basis([r]):
            for k in ("consensus_label", "consensus_basis_sha256"):
                r[k] = old.get(k)
    merged = others + incoming
    human_groups(merged)
    return apply_consensus(merged)[0]


# ── reliability and consensus ──

def cohen_kappa(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    n = len(pairs)
    if not n:
        return None
    categories = {a for a, _ in pairs} | {b for _, b in pairs}
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in categories)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def relative_label(row: dict, first: str) -> str:
    """``first`` / ``second`` / ``same``: the condition-level label relative to the comparison."""
    label = row["condition_level_label"]
    return "same" if label == "same" else ("first" if label == first else "second")


def signed(row: dict, first: str) -> int:
    return {"first": 1, "second": -1, "same": 0}[relative_label(row, first)]


def human_rows(rows: Iterable[dict]) -> List[dict]:
    return [r for r in rows if r["rater_type"] == "human"]


def human_groups(rows, *, complete=False):
    """Validate identities, labels and shared materials, then group human judgments."""
    humans = human_rows(rows)
    assert_unique(humans, ("pair_id", "rater_id"))
    groups, cohorts, expected_sets, by_rater = defaultdict(list), set(), set(), defaultdict(set)
    for r in humans:
        expected = tuple(sorted(r.get("expected_raters", [])))
        if len(expected) != 2 or len(set(expected)) != 2 or r["rater_id"] not in expected:
            raise ValueError("annotations require two declared, distinct raters")
        expected_sets.add(expected)
        cohorts.add(r.get("assembled_sha256"))
        if not all(r.get(k) for k in ("worksheet_sha256", "passage_a_sha256", "passage_b_sha256")):
            raise ValueError("annotation material hashes are missing")
        first, second = r["comparison_type"].split(":")
        if ((first, second) not in COMPARISONS or r["raw_label"] not in LABELS
                or {r["passage_a_condition"], r["passage_b_condition"]} != {first, second}
                or r["condition_level_label"] != condition_level(r["raw_label"], r["passage_a_condition"], r["passage_b_condition"])
                or r["presentation_order"] != ("ab" if r["passage_a_condition"] == first else "ba")):
            raise ValueError("annotation label or condition mapping is inconsistent")
        if r.get("consensus_label") not in (None, first, second, "same"):
            raise ValueError("consensus label is outside the comparison")
        groups[r["pair_id"]].append(r)
        by_rater[r["rater_id"]].add(r["pair_id"])
    if len(cohorts) > 1 or None in cohorts or len(expected_sets) > 1:
        raise ValueError("annotations come from different passages or rater assignments")
    if len({frozenset(p) for p in by_rater.values()}) > 1:
        raise ValueError("raters labelled different passage pairs")
    for group in groups.values():
        shared = ("game_id", "speaker_id", "comparison_type", "passage_a_condition", "passage_b_condition",
                  "passage_a_sha256", "passage_b_sha256", "presentation_order")
        if any(any(r[k] != group[0][k] for k in shared) for r in group):
            raise ValueError("raters have different materials or sides for a pair")
        if complete and len(group) != 2:
            raise ValueError("consensus requires complete labels from both researchers")
    if complete and not groups:
        raise ValueError("consensus requires labels from both researchers")
    return dict(groups)


def label_basis(group):
    fields = ("pair_id", "rater_id", "raw_label", "note", "assembled_sha256", "worksheet_sha256",
              "passage_a_condition", "passage_b_condition", "passage_a_sha256", "passage_b_sha256")
    return digest([{k: r.get(k) for k in fields} for r in sorted(group, key=lambda r: r["rater_id"])])


def reliability(rows: Sequence[dict]) -> dict:
    """Percent agreement and Cohen's kappa between the two human raters on comparison-relative labels."""
    human_groups(rows)
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
    by_pair = human_groups(rows, complete=bool(decisions))
    if set(decisions) - set(by_pair):
        raise ValueError("consensus contains unknown pairs")
    unresolved = []
    out = [dict(r) for r in rows]
    for pair_id, group in by_pair.items():
        labels = {r["condition_level_label"] for r in group}
        basis = label_basis(group)
        consensus = None
        if len(group) == 2 and len(labels) == 1:
            consensus = labels.pop()
            if pair_id in decisions and decisions[pair_id] != consensus:
                raise ValueError("adjudication conflicts with an independently agreed label")
        elif pair_id in decisions:
            consensus = decisions[pair_id]
            allowed = {group[0]["passage_a_condition"], group[0]["passage_b_condition"], "same"}
            if consensus not in allowed:
                raise ValueError(f"{pair_id}: consensus {consensus!r} is not one of {sorted(allowed)}")
        elif len(group) == 2 and all(r.get("consensus_basis_sha256") == basis for r in group):
            recorded = {r.get("consensus_label") for r in group}
            if len(recorded) == 1:
                consensus = recorded.pop()
        if consensus is None:
            unresolved.append(pair_id)
        for r in out:
            if r["rater_type"] == "human" and r["pair_id"] == pair_id:
                r["consensus_label"] = consensus
                r["consensus_basis_sha256"] = basis if consensus is not None else None
    return out, sorted(unresolved)


def consensus_items(rows, pairs_by_id):
    groups = human_groups(rows, complete=True)
    out = {}
    for pair_id, group in groups.items():
        if len({r["condition_level_label"] for r in group}) == 1:
            continue
        pair = pairs_by_id[pair_id]
        for side in ("a", "b"):
            if sha256_text(pair[f"passage_{side}_text"]) != group[0][f"passage_{side}_sha256"]:
                raise ValueError("consensus passages differ from the annotated materials")
        item_id = "item_" + digest([pair_id, label_basis(group)])[:20]
        out[item_id] = (pair, group)
    return dict(sorted(out.items()))


def render_consensus_worksheet(rows: Sequence[dict], pairs_by_id: Dict[str, dict]) -> str:
    """Disagreements only, with both raw labels shown, for the joint adjudication."""
    items = consensus_items(rows, pairs_by_id)
    lines = ["# Linguistic range: consensus sheet", "",
             "Items where the two independent labels differ. Agree on one label per item and fill in "
             "`consensus:` with `A`, `B` or `same`, using the same passage sides as shown here.", ""]
    for item_id, (pair, group) in items.items():
        lines += ["---", "", f"### {item_id}", ""]
        for r in sorted(group, key=lambda r: r["rater_id"]):
            lines.append(f"- {r['rater_id']}: `{r['raw_label']}`" + (f" ({r['note']})" if r.get("note") else ""))
        lines += ["", "**Passage A**", ""] + [f"> {l}" if l else ">" for l in pair["passage_a_text"].splitlines()]
        lines += ["", "**Passage B**", ""] + [f"> {l}" if l else ">" for l in pair["passage_b_text"].splitlines()]
        lines += ["", "consensus:", ""]
    lines.insert(3, f"{len(items)} items.")
    return "\n".join(lines)


def parse_consensus(text: str, pairs_by_id: Dict[str, dict], rows: Sequence[dict]) -> Dict[str, str]:
    expected = render_consensus_worksheet(rows, pairs_by_id)
    if worksheet_hash(text, ("consensus",)) != worksheet_hash(expected, ("consensus",)):
        raise ValueError("consensus worksheet differs from the current materials or independent labels")
    items = consensus_items(rows, pairs_by_id)
    out: Dict[str, str] = {}
    current = None
    for line in text.splitlines():
        heading = re.match(r"^###\s+(\S+)\s*$", line.strip())
        if heading:
            current = heading.group(1)
            continue
        match = re.match(r"^consensus:\s*`?([^`]*)`?\s*$", line.strip())
        if current and match and match.group(1).strip():
            pair, _ = items[current]
            out[pair["pair_id"]] = condition_level(normalize_label(match.group(1)), pair["passage_a_condition"], pair["passage_b_condition"])
    if len(out) != len(items):
        raise ValueError("consensus worksheet has unanswered items")
    return out


# ── model listeners on the same pairs ──

def model_rows(listener_rows: Iterable[dict], pairs: Sequence[dict]) -> List[dict]:
    """Model listener judgments on the annotated passages, in the annotation schema (one row per order)."""
    wanted = {(p["bundle_id"], frozenset((p["first_condition"], p["second_condition"]))): p for p in pairs}
    candidates = [r for r in listener_rows if r.get("construct", "range") == "range"]
    complete, _, _ = complete_listener_rows(candidates)
    out = []
    for r in complete:
        pair = wanted.get((r["bundle_id"], frozenset((r["treatment"], r["comparator"]))))
        if pair is None:
            continue
        a, b = r["left_condition"], r["right_condition"]
        expected_sides = ((r["treatment"], r["comparator"]) if r["order"] == "ab" else (r["comparator"], r["treatment"]))
        if (a, b) != expected_sides or r["game_id"] != pair["game_id"] or r["speaker"] != pair["speaker_id"]:
            raise ValueError("model listener passage identity or order is inconsistent")
        label = condition_level(r["choice"], a, b)
        if label != ("same" if r.get("tie") else r.get("chosen_condition")):
            raise ValueError("model listener label is inconsistent")
        out.append({"pair_id": pair["pair_id"], "game_id": r["game_id"], "speaker_id": r["speaker"],
                    "comparison_type": pair["comparison_type"], "passage_a_condition": a, "passage_b_condition": b,
                    "presentation_order": "ab" if a == pair["first_condition"] else "ba",
                    "rater_type": "model", "rater_id": r["listener"], "rater_family": r.get("listener_family"),
                    "raw_label": r["choice"], "condition_level_label": label, "consensus_label": None})
    assert_unique(out, ("pair_id", "rater_id", "presentation_order"))
    return out


# ── summary ──

def summary(rows: Sequence[dict], listener_rows: Sequence[dict] = (), pairs: Sequence[dict] = ()) -> dict:
    rows, unresolved = apply_consensus(rows)
    humans = human_rows(rows)
    pairs = list(pairs) or _pairs_from_rows(humans)
    models = model_rows(listener_rows, pairs)
    consensus = {r["pair_id"]: r["consensus_label"] for r in humans if r.get("consensus_label")}
    raters = sorted({r["rater_id"] for r in humans})
    listeners = sorted({r["rater_id"] for r in models})
    out = {"n_pairs": len(pairs), "raters": raters, "listeners": listeners, "reliability": reliability(rows),
           "n_consensus": len(consensus), "comparisons": {}, "agreement_with_consensus": {}, "order_consistency": {},
           "by_game": {}, "unresolved_pairs": unresolved,
           "labels_per_rater": dict(Counter(r["rater_id"] for r in humans)),
           "expected_raters": humans[0]["expected_raters"] if humans else list(RATERS)}
    for first, second in COMPARISONS:
        key = comparison_key(first, second)
        block = {"title": comparison_title(first, second), "first": first, "second": second,
                 "expected_sign": EXPECTED_SIGN[(first, second)], "n_pairs": sum(p["comparison_type"] == key for p in pairs), "raters": {}}
        for rater in raters:
            block["raters"][rater] = _direction([r for r in humans if r["comparison_type"] == key and r["rater_id"] == rater], first)
        representatives = {r["pair_id"]: r for r in humans if r["comparison_type"] == key}
        cons = [{**representatives[pid], "condition_level_label": label}
                for pid, label in consensus.items() if pid in representatives]
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
    games = defaultdict(lambda: defaultdict(list))
    for r in rows:
        games[r["game_id"]][r["pair_id"]].append(relative_label(r, first))
    shares = {label: mean([mean([mean([float(v == label) for v in values]) for values in pairs.values()])
                           for pairs in games.values()]) for label in ("first", "second", "same")}
    return {"n_judgments": len(rows), "n_pairs": sum(len(p) for p in games.values()), "n_games": len(games),
            "estimand": "equal_game_weight", "first_more_limited": shares["first"],
            "second_more_limited": shares["second"], "same": shares["same"],
            "signed_preference": shares["first"] - shares["second"] if games else None}


def render_summary(s: dict) -> List[str]:
    """Markdown section: reliability, direction per comparison for every rater, agreement with consensus."""
    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    num = lambda v: "n/a" if v is None else f"{v:+.2f}"
    kappa_text = lambda v: "n/a" if v is None else f"{v:.2f}"
    rel = s["reliability"]
    lines = ["## Human listeners", "",
             f"Labels from {len(s['raters'])} of {len(s['expected_raters'])} researchers cover {s['n_pairs']} passage pairs. "
             f"{rel.get('n_pairs', 0)} pairs have both independent labels; {s['n_consensus']} have a consensus label.", "",
             "Directional shares and signed preferences average responses within passages, then passages within games, "
             "with equal weight for each game. Agreement and κ use the raw paired judgments.", ""]
    if s["listeners"]:
        lines += ["Model summaries include passage/model pairs with valid answers in both presentation orders.", ""]
    if rel.get("n_pairs"):
        lines += [f"Human agreement on {rel['n_pairs']} shared pairs: {pct(rel['percent_agreement'])}, "
                  f"Cohen's κ = {kappa_text(rel['cohen_kappa'])}.", "",
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
            lines.append(f"| {name} | {a['n_judgments']} | {pct(a['percent_agreement'])} | {kappa} |")
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
    keys = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(data_dir.glob("*_key.json"))]
    if len(keys) != 2 or len({k["rater"] for k in keys}) != 2:
        raise ValueError("material directory must contain one key for each of two researchers")
    if len({k.get("assembled_sha256") for k in keys}) != 1:
        raise ValueError("worksheet keys refer to different passages")
    if assembled_path is None or keys[0].get("assembled_sha256") != sha256_file(assembled_path):
        raise ValueError("worksheet materials differ from the assembled passages")
    pair_sets = []
    for key in keys:
        if (set(key.get("expected_raters", [])) != {k["rater"] for k in keys}
                or not key.get("worksheet_sha256")):
            raise ValueError("worksheet key has incomplete rater or material provenance")
        assert_unique(list(key["items"].values()), ("pair_id",))
        pair_sets.append(set(i["pair_id"] for i in key["items"].values()))
        for item_id, item in key["items"].items():
            p = pairs.setdefault(item["pair_id"], {**item, "rater_keys": {}})
            if any(p[k] != v for k, v in item.items()):
                raise ValueError("worksheet keys assign different texts or sides to a pair")
            p["rater_keys"][key["rater"]] = {"item_id": item_id, "worksheet_sha256": key["worksheet_sha256"]}
    if pair_sets[0] != pair_sets[1]:
        raise ValueError("worksheet keys contain different pair sets")
    assembled = read_jsonl(assembled_path)
    assert_unique(assembled, ("bundle_id", "condition"))
    texts = {(r["bundle_id"], r["condition"]): r for r in assembled}
    for p in pairs.values():
        for side in ("a", "b"):
            p[f"passage_{side}_text"] = passage_text(texts[(p["bundle_id"], p[f"passage_{side}_condition"])])
            if sha256_text(p[f"passage_{side}_text"]) != p[f"passage_{side}_sha256"]:
                raise ValueError("worksheet key passage hash does not match assembled text")
    return pairs


def verify_materials(rows, assembled_path, pairs_by_id=None):
    """Match imported identities and text hashes to the supplied run."""
    groups = human_groups(rows)
    assembled_sha = sha256_file(assembled_path)
    available = {p["pair_id"]: p for p in build_pairs(read_jsonl(assembled_path))}
    for pair_id, group in groups.items():
        if pair_id not in available:
            raise ValueError("annotated pair is absent from the complete passage cohort")
        p = available[pair_id]
        texts = {p["passage_a_condition"]: p["passage_a_text"], p["passage_b_condition"]: p["passage_b_text"]}
        for r in group:
            if (r["assembled_sha256"] != assembled_sha or r["game_id"] != p["game_id"]
                    or r["speaker_id"] != p["speaker_id"] or r["comparison_type"] != p["comparison_type"]):
                raise ValueError("annotations differ from the current run's materials")
            if any(r[f"passage_{side}_sha256"] != sha256_text(texts[r[f"passage_{side}_condition"]]) for side in ("a", "b")):
                raise ValueError("annotated passage text differs from the current run")
            if pairs_by_id is not None:
                key = pairs_by_id[pair_id]["rater_keys"][r["rater_id"]]
                if any(r[k] != v for k, v in key.items()):
                    raise ValueError("annotations do not match the current worksheet keys")


def verified_summary(rows, assembled_path, listener_paths=()):
    verify_materials(rows, assembled_path)
    listeners = []
    for path in listener_paths:
        metadata = read_metadata(path) or {}
        if (metadata.get("input_sha256", {}).get("assembled") != sha256_file(assembled_path)
                or metadata.get("output_sha256") != sha256_file(path)):
            raise ValueError("model listener outputs do not match the annotated run")
        listeners.extend(read_jsonl(path))
    result = summary(rows, listeners)
    result["assembled_sha256"] = sha256_file(assembled_path)
    result["listener_sha256"] = {str(path): sha256_file(path) for path in listener_paths}
    return result


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
    p.add_argument("--assembled", type=Path, required=True)
    p.add_argument("--annotations", type=Path, default=MANUAL_FILE)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR.parent)
    p.set_defaults(func=cmd_import_consensus)

    p = sub.add_parser("annotation-summary", help="reliability, human direction per comparison and agreement with the model listeners")
    p.add_argument("--annotations", type=Path, default=MANUAL_FILE)
    p.add_argument("--assembled", type=Path, required=True)
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
    rows = import_annotations(args.worksheet.read_text(encoding="utf-8"), key, args.rater)
    existing = read_jsonl(args.out) if args.out.exists() else []
    merged = merge_rater(existing, rows, args.rater)
    merged, unresolved = apply_consensus(merged)
    write_jsonl(args.out, sorted(merged, key=lambda r: (r["pair_id"], r["rater_id"])))
    update_metadata(args.out, "summary", {"created_at": now_iso(), "raters": sorted({r["rater_id"] for r in merged}),
                                          "n_rows": len(merged), "unresolved_pairs": len(unresolved)})
    print(f"{len(rows)} judgments by {args.rater} -> {args.out}; {len(unresolved)} pairs await consensus")
    return 0


def cmd_consensus_sheet(args) -> int:
    rows = read_jsonl(args.annotations)
    pairs = load_pairs_from_keys(args.data_dir, args.assembled)
    verify_materials(rows, args.assembled, pairs)
    text = render_consensus_worksheet(rows, pairs)
    out = args.out or args.data_dir / "consensus_worksheet.md"
    if out.exists():
        if worksheet_hash(out.read_text(encoding="utf-8"), ("consensus",)) != worksheet_hash(text, ("consensus",)):
            raise ValueError("existing consensus sheet uses different labels or materials; choose a new --out path")
    else:
        out.write_text(text, encoding="utf-8")
    print(f"consensus sheet -> {out}")
    return 0


def cmd_import_consensus(args) -> int:
    rows = read_jsonl(args.annotations)
    pairs = load_pairs_from_keys(args.data_dir, args.assembled)
    verify_materials(rows, args.assembled, pairs)
    decisions = parse_consensus(args.worksheet.read_text(encoding="utf-8"), pairs, rows)
    rows, unresolved = apply_consensus(rows, decisions)
    write_jsonl(args.annotations, rows)
    update_metadata(args.annotations, "consensus", {"created_at": now_iso(), "adjudicated_pairs": len(decisions),
                                                    "unresolved_pairs": unresolved})
    print(f"{len(decisions)} adjudicated pairs recorded; {len(unresolved)} unresolved")
    return 0


def cmd_summary(args) -> int:
    rows = read_jsonl(args.annotations)
    s = verified_summary(rows, args.assembled, args.listener)
    s["annotations_sha256"] = sha256_file(args.annotations)
    text = "\n".join(render_summary(s))
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        dump_json(args.out.with_suffix(".json"), s)
        print(f"summary -> {args.out}")
    else:
        print(text)
    return 0
