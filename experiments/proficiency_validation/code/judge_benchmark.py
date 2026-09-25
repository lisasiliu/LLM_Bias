"""Calibrating the semantic judges before they gate anything (PLAN 4.1, 4.2).

Two benchmarks, both scored the same way (a judge FLAGS a rewrite as broken or
PASSES it; the reference says whether meaning was in fact broken):

  human      the 48 politeness-validation rewrites with two blind human
             ``meaning`` judgements. Real errors, but very few of them, so the
             sensitivity estimates carry wide intervals and are reported with them.
  challenge  synthetic perturbations of untreated originals where the meaning
             change is known by construction (speaker/target swap, role flip,
             polarity flip, action-target change, reason deletion / substitution,
             proposition deletion, certainty shift, directive-force shift), plus
             positive controls (original vs original; human-preserved rewrites).

Sensitivity = P(flag | broken), specificity = P(pass | preserved). Reported per
judge (structural, parser, validator) and for the combined pipeline (any flag).
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from components.helpers import SourceRecord
from components.judges.judge_model import JudgeModel
from components.metrics.text_normalization import PLAYER_ID_RE as PLAYER_RE
from components.rewriters.politeness_rewriter import predicate_definition, predicate_entry
from components.stats import wilson_interval
from lurkingwolf.utils.speech_nl import parse_predicate

from .common import fmt, now_iso

# ── scoring ──

JUDGES = ("structural", "parser", "validator", "pipeline")


def confusion(rows: Sequence[Tuple[bool, Optional[bool]]]) -> dict:
    """``rows`` = (gold_broken, judge_flagged); ``None`` flags are no-verdicts."""
    tp = sum(1 for g, f in rows if g and f is True)
    fn = sum(1 for g, f in rows if g and f is False)
    fp = sum(1 for g, f in rows if not g and f is True)
    tn = sum(1 for g, f in rows if not g and f is False)
    nv = sum(1 for _, f in rows if f is None)
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    return {
        "n": len(rows), "n_broken": tp + fn, "n_preserved": tn + fp, "no_verdict": nv,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "sensitivity": sens, "sensitivity_ci95": wilson_interval(tp, tp + fn)[1:],
        "specificity": spec, "specificity_ci95": wilson_interval(tn, tn + fp)[1:],
        "precision_of_flag": tp / (tp + fp) if tp + fp else None,
        "npv_of_pass": tn / (tn + fn) if tn + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def _flag(value: Optional[bool]) -> Optional[bool]:
    """A judge 'ok' value -> flagged (True), passed (False) or no verdict (None)."""
    if value is True:
        return False
    if value is False:
        return True
    return None


def judge_flags(record: dict, *, use_recheck: bool = True) -> Dict[str, Optional[bool]]:
    """Per-judge flags from a politeness manual_validation record."""
    semantic_ok = record.get("semantic_ok")
    if use_recheck and isinstance(record.get("semantic_recheck"), dict):
        semantic_ok = record["semantic_recheck"].get("semantic_ok", semantic_ok)
    flags = {
        "structural": _flag(record.get("structural_check_ok")),
        "parser": _flag(record.get("inverse_parser_structural_match")),
        "validator": _flag(semantic_ok),
    }
    known = [v for v in flags.values() if v is not None]
    flags["pipeline"] = (True if any(known) else (False if len(known) == 3 else None))
    return flags


def human_benchmark(records: Sequence[dict], *, use_recheck: bool = True) -> dict:
    """Judges vs each rater's ``meaning`` label, and vs the two-rater consensus."""
    raters = sorted({a["annotator"] for r in records for a in r.get("annotations", [])})
    references: Dict[str, List[Tuple[dict, bool]]] = {}
    unsure: Dict[str, int] = {}
    for rater in raters:
        rows, skipped = [], 0
        for r in records:
            label = next((a.get("meaning") for a in r.get("annotations", [])
                          if a.get("annotator") == rater), None)
            if label == "no":
                rows.append((r, True))
            elif label == "yes":
                rows.append((r, False))
            else:
                skipped += 1
        references[rater] = rows
        unsure[rater] = skipped
    consensus, disagreements = [], 0
    for r in records:
        labels = {a.get("annotator"): a.get("meaning") for a in r.get("annotations", [])}
        values = set(labels.values())
        if values == {"no"}:
            consensus.append((r, True))
        elif values == {"yes"}:
            consensus.append((r, False))
        else:
            disagreements += 1
    references["consensus"] = consensus

    result: Dict[str, Any] = {"n_items": len(records), "raters": raters,
                              "unsure_or_missing": unsure,
                              "consensus_excluded": disagreements,
                              "semantic_recheck_used": use_recheck, "by_reference": {}}
    for name, rows in references.items():
        per_judge = {}
        for judge in JUDGES:
            per_judge[judge] = confusion([(gold, judge_flags(r, use_recheck=use_recheck)[judge])
                                          for r, gold in rows])
        result["by_reference"][name] = per_judge
    # Where do the human 'no' judgements fall by predicate and strategy?
    broken = [r for r, gold in references["consensus"] if gold]
    result["consensus_broken_items"] = [
        {"item_id": r["item_id"], "predicate": r["predicate_name"],
         "requested_strategy": r["requested_strategy"],
         "flags": judge_flags(r, use_recheck=use_recheck)} for r in broken]
    return result


def render_human_benchmark(bench: dict) -> str:
    lines = ["# Semantic-judge benchmark on the 48 human-annotated politeness rewrites", "",
             f"Items: {bench['n_items']}. Reference = a rater's `meaning` label "
             "(`no` = broken, `yes` = preserved; `unsure` excluded). Judges flag a rewrite "
             "when their check fails; `pipeline` flags when any of the three does.", ""]
    for name, per_judge in bench["by_reference"].items():
        any_row = next(iter(per_judge.values()))
        lines += [f"## Reference: {name}  (broken n={any_row['n_broken']}, "
                  f"preserved n={any_row['n_preserved']})", "",
                  "| judge | sensitivity [95% CI] | specificity [95% CI] | FPR | FNR | precision of flag | no verdict |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for judge, c in per_judge.items():
            s_lo, s_hi = c["sensitivity_ci95"]
            p_lo, p_hi = c["specificity_ci95"]
            lines.append(
                f"| {judge} | {fmt(c['sensitivity'])} [{fmt(s_lo, 2)}, {fmt(s_hi, 2)}] | "
                f"{fmt(c['specificity'])} [{fmt(p_lo, 2)}, {fmt(p_hi, 2)}] | "
                f"{fmt(c['false_positive_rate'])} | {fmt(c['false_negative_rate'])} | "
                f"{fmt(c['precision_of_flag'])} | {c['no_verdict']} |")
        lines.append("")
    lines += ["## Items both raters judged broken", "",
              "| item | predicate | requested strategy | structural | parser | validator |",
              "| --- | --- | --- | --- | --- | --- |"]
    for item in bench["consensus_broken_items"]:
        f = item["flags"]
        lines.append(f"| {item['item_id']} | {item['predicate']} | {item['requested_strategy']} | "
                     f"{fmt(f['structural'])} | {fmt(f['parser'])} | {fmt(f['validator'])} |")
    lines += ["", "**Reading.** The negative class is tiny (a handful of `no` labels), so the "
              "sensitivity intervals are wide; the synthetic challenge set supplies the "
              "sensitivity estimate the gate is frozen on, and this benchmark anchors "
              "specificity on real rewrites.", ""]
    return "\n".join(lines)


# ── synthetic challenge set ──

ROLE_FLIPS = [
    (r"\bwerewolf\b", "villager"), (r"\bwerewolves\b", "villagers"),
    (r"\ba wolf\b", "a villager"), (r"\bwolves\b", "villagers"), (r"\bwolf\b", "villager"),
    (r"\bvillager\b", "wolf"), (r"\bvillagers\b", "wolves"),
    (r"\bvillage[- ]side\b", "wolf-side"), (r"\bwolf[- ]side\b", "village-side"),
    (r"\bvillager_camp\b", "wolf_camp"), (r"\bwolf_camp\b", "villager_camp"),
    (r"\bseer\b", "witch"),
]
POLARITY_RULES = [
    (r"\bis not\b", "is"), (r"\bisn't\b", "is"), (r"\bare not\b", "are"), (r"\baren't\b", "are"),
    (r"\bcannot\b", "can"), (r"\bcan't\b", "can"), (r"\bdid not\b", "did"), (r"\bdidn't\b", "did"),
    (r"\bnever\b", "always"), (r"\bno longer\b", "still"),
    (r"\bis a\b", "is not a"), (r"\bis my\b", "is not my"), (r"\bis the\b", "is not the"),
    (r"\bare\b", "are not"), (r"\bis\b", "is not"), (r"\bhas\b", "has not"), (r"\bhave\b", "have not"),
]
CERTAINTY_RULES = [
    (r"\bmight be\b", "is"), (r"\bcould be\b", "is"), (r"\bmay be\b", "is"),
    (r"\bprobably\b", "definitely"), (r"\bI think\b", "I know"), (r"\bI suspect\b", "I know"),
    (r"\bis definitely\b", "might be"), (r"\bis clearly\b", "might be"),
    (r"\bis a\b", "might be a"), (r"\bis my\b", "might be my"), (r"\bis the\b", "might be the"),
    (r"\bis\b", "might be"),
]
# Ordered: the rules that target the act's own verb come first, so a challenge
# in a question is weakened at "why did you" rather than at an incidental verb.
DIRECTIVE_RULES = [
    (r"\bwhy did you\b", "I wonder if you"), (r"\bexplain why\b", "feel free to say why"),
    (r"\bI recommend\b", "I wonder whether"), (r"\bvote to execute\b", "keep an eye on"),
    (r"\bvote out\b", "keep an eye on"), (r"\bvote for\b", "think about"),
    (r"\bpasses the badge to\b", "could consider the badge for"),
    (r"\bthe badge goes to\b", "the badge could maybe go to"),
    (r"\bshould\b", "could perhaps"), (r"\bmust\b", "might"), (r"\bexecute\b", "watch"),
]
REASON_SPLIT = re.compile(r"(?:\s+because\s+|\s+since\s+|:\s+|\s+—\s+|\s+-\s+|,\s+which\s+|,\s+so\s+)", re.IGNORECASE)
REASON_BANK = [
    "because they voted with the confirmed wolf on the first day",
    "because they went completely silent right after the seer claim",
    "because they changed their vote at the last second without a reason",
    "because their reads have matched the dead wolves every single time",
]
CHALLENGE_TYPES = ("speaker_target_swap", "role_flip", "polarity_flip", "action_target_change",
                   "reason_deletion", "reason_substitution", "proposition_deletion",
                   "certainty_shift", "directive_force_shift")


def _first_rule(text: str, rules) -> Optional[Tuple[str, str]]:
    for pattern, replacement in rules:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE), pattern
    return None


def _segments(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?;])\s+|\s+—\s+|\s+-\s+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def perturb(source: SourceRecord, kind: str, rng: random.Random) -> Optional[dict]:
    """One perturbed text, or ``None`` when the rule does not apply to this text."""
    text = source.original_nl
    name = source.predicate_name
    args = list(source.predicate_args)
    if kind == "speaker_target_swap":
        ids = sorted({m.group(0) for m in PLAYER_RE.finditer(text)})
        if len(ids) < 2:
            return None
        a, b = ids[0], ids[1]
        swapped = re.sub(rf"\b({a}|{b})\b", lambda m: b if m.group(1) == a else a, text)
        return {"text": swapped, "rule": f"swap {a}<->{b}"}
    if kind == "role_flip":
        out = _first_rule(text, ROLE_FLIPS)
        return {"text": out[0], "rule": out[1]} if out else None
    if kind == "polarity_flip":
        out = _first_rule(text, POLARITY_RULES)
        return {"text": out[0], "rule": out[1]} if out else None
    if kind == "action_target_change":
        if name != "recommend_action" or len(args) < 5:
            return None
        target = args[4].strip()
        if not PLAYER_RE.fullmatch(target) or target not in text:
            return None
        mentioned = {m.group(0) for m in PLAYER_RE.finditer(text)} | {source.speaker_id}
        candidates = [f"P{i}" for i in range(1, 13) if f"P{i}" not in mentioned]
        if not candidates:
            return None
        new = rng.choice(candidates)
        return {"text": re.sub(rf"\b{target}\b", new, text), "rule": f"target {target}->{new}"}
    if kind == "reason_deletion":
        parts = REASON_SPLIT.split(text, maxsplit=1)
        if len(parts) < 2 or len(parts[0].split()) < 4:
            return None
        head = parts[0].rstrip(" ,;:—-")
        return {"text": head + ("." if not head.endswith((".", "?", "!")) else ""),
                "rule": "drop clause after first reason connective"}
    if kind == "reason_substitution":
        parts = REASON_SPLIT.split(text, maxsplit=1)
        if len(parts) < 2 or len(parts[0].split()) < 4:
            return None
        head = parts[0].rstrip(" ,;:—-")
        return {"text": f"{head} {rng.choice(REASON_BANK)}.", "rule": "replace reason clause"}
    if kind == "proposition_deletion":
        segs = _segments(text)
        if len(segs) < 2:
            return None
        drop = rng.randrange(1, len(segs))
        kept = [s for i, s in enumerate(segs) if i != drop]
        return {"text": " ".join(kept), "rule": f"drop segment {drop} of {len(segs)}"}
    if kind == "certainty_shift":
        out = _first_rule(text, CERTAINTY_RULES)
        return {"text": out[0], "rule": out[1]} if out else None
    if kind == "directive_force_shift":
        if name not in ("recommend_action", "question"):
            return None
        out = _first_rule(text, DIRECTIVE_RULES)
        return {"text": out[0], "rule": out[1]} if out else None
    raise ValueError(f"unknown challenge type {kind!r}")


def source_record_from_dict(d: dict) -> SourceRecord:
    """A ``SourceRecord`` from a provenance/JSONL source dict."""
    predicate_text = d.get("predicate_text") or d.get("predicate") or ""
    parsed = parse_predicate(predicate_text)
    name = d.get("predicate_name") or str(parsed.get("name", ""))
    args = [str(a) for a in parsed.get("args", [])]
    speaker = d.get("speaker") or d.get("speaker_id") or (args[1] if len(args) > 1 else "")
    ref = d.get("referenced_speech")
    if ref is None and d.get("referenced_text"):
        ref = {"speech_id": d.get("referenced_speech_id"), "speaker": d.get("referenced_speaker"),
               "predicate": "", "text": d.get("referenced_text")}
    return SourceRecord(
        source_id=d.get("source_id", ""), predicate_name=name, predicate_text=predicate_text,
        predicate_args=args, predicate_definition=predicate_definition(name),
        nl_hint=predicate_entry(name).get("nl_hint", ""),
        original_nl=d.get("original_nl") or d.get("original_text") or "",
        # ``phase`` is the engine phase NAME (day_speech); the D1/N1 token in the
        # predicate is the Phase SLOT and is read separately by the parser.
        phase=d.get("phase") or "day_speech", speaker_id=speaker,
        game_id=d.get("game_id"), speech_id=d.get("speech_id"),
        public_history=list(d.get("public_history") or []), referenced_speech=ref,
        provenance=d.get("provenance") or d.get("source_provenance") or "unknown")


def build_challenge_set(sources: Sequence[SourceRecord], *, seed: int = 101,
                        positives: Optional[Sequence[dict]] = None) -> List[dict]:
    """Perturbed negatives for every applicable rule, plus positive controls."""
    rng = random.Random(seed)
    items: List[dict] = []
    for src in sources:
        base = {"source_id": src.source_id, "game_id": src.game_id, "speech_id": src.speech_id,
                "predicate": src.predicate_text, "predicate_name": src.predicate_name,
                "phase": src.phase, "speaker": src.speaker_id, "original_text": src.original_nl,
                "referenced_speech": src.referenced_speech,
                "public_history": list(src.public_history)}
        items.append({**base, "item_id": f"pos_identity_{len(items):03d}",
                      "challenge_type": "positive_identity", "expected": "preserved",
                      "candidate_text": src.original_nl, "rule": "original vs original"})
        for kind in CHALLENGE_TYPES:
            out = perturb(src, kind, rng)
            if out is None or out["text"].strip() == src.original_nl.strip():
                continue
            items.append({**base, "item_id": f"neg_{kind}_{len(items):03d}",
                          "challenge_type": kind, "expected": "broken",
                          "candidate_text": out["text"], "rule": out["rule"]})
    for pos in positives or []:
        items.append({**pos, "item_id": f"pos_human_{len(items):03d}",
                      "challenge_type": "positive_human_preserved", "expected": "preserved"})
    return items


def human_preserved_positives(manual_records: Sequence[dict]) -> List[dict]:
    """Politeness rewrites both raters judged meaning-preserving, excluding
    off-record ones (the proficiency gate runs with the DIRECT commitment rule,
    under which licensed off-record indirectness would count as a failure)."""
    out = []
    for r in manual_records:
        labels = {a.get("meaning") for a in r.get("annotations", [])}
        if labels != {"yes"} or r.get("requested_strategy") == "off_record":
            continue
        out.append({"source_id": r["source_id"], "game_id": r.get("game_id"),
                    "speech_id": r.get("speech_id"), "predicate": r["predicate"],
                    "predicate_name": r["predicate_name"], "phase": r.get("phase"),
                    "speaker": r.get("speaker"), "original_text": r["original_text"],
                    "referenced_speech": ({"speech_id": r.get("referenced_speech_id"),
                                           "speaker": r.get("referenced_speaker"), "predicate": "",
                                           "text": r.get("referenced_text")}
                                          if r.get("referenced_text") else None),
                    "public_history": [], "candidate_text": r["rewritten_text"],
                    "rule": f"human-preserved {r['requested_strategy']} rewrite ({r['item_id']})"})
    return out


def run_challenge(items: Sequence[dict], gate) -> List[dict]:
    """Every judge on every item, independently (calls the judge models).

    Uses ``gate.judge_all`` rather than ``gate.evaluate``: the production gate
    short-circuits after a cheap failure and rejects a copy of the original,
    which would leave the parser and validator unscored on exactly the items a
    benchmark needs and would fail the identity positive controls by design.
    """
    out = []
    for item in items:
        src = source_record_from_dict({**item, "original_nl": item["original_text"],
                                       "predicate_text": item["predicate"]})
        verdicts = gate.judge_all(src, item["candidate_text"])
        parser = verdicts.get("parser")
        semantic = verdicts.get("semantic")
        flags = {
            "structural": _flag(verdicts["structural"]["ok"]),
            "parser": (_flag(parser["structural_match"]) if parser and parser["ok"] else None),
            "validator": _flag(semantic["ok"]) if semantic else None,
        }
        applicable = [j for j in ("structural", "parser", "validator")
                      if (j != "parser" or parser is not None) and (j != "validator" or semantic is not None)]
        known = [flags[j] for j in applicable if flags[j] is not None]
        flags["pipeline"] = (True if any(known)
                             else (False if len(known) == len(applicable) else None))
        out.append({**item, "judges": verdicts, "flags": flags, "judged_at": now_iso()})
    return out


def report_challenge(items: Sequence[dict]) -> dict:
    types = sorted({i["challenge_type"] for i in items})
    per_type: Dict[str, dict] = {}
    for kind in types:
        rows = [i for i in items if i["challenge_type"] == kind and "flags" in i]
        per_type[kind] = {
            "n": len(rows), "expected": rows[0]["expected"] if rows else None,
            **{judge: confusion([(i["expected"] == "broken", i["flags"].get(judge)) for i in rows])
               for judge in JUDGES}}
    overall = {judge: confusion([(i["expected"] == "broken", i["flags"].get(judge))
                                 for i in items if "flags" in i]) for judge in JUDGES}
    return {"n_items": len(items), "per_type": per_type, "overall": overall}


def render_challenge(report: dict) -> str:
    lines = ["# Synthetic semantic challenge set", "",
             f"Items judged: {report['n_items']}.", "",
             "| challenge type | expected | n | structural flags | parser flags | validator flags | pipeline flags |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind, row in report["per_type"].items():
        def hits(judge):
            c = row[judge]
            return f"{c['tp'] + c['fp']}/{c['n'] - c['no_verdict']}"
        lines.append(f"| {kind} | {row['expected']} | {row['n']} | {hits('structural')} | "
                     f"{hits('parser')} | {hits('validator')} | {hits('pipeline')} |")
    lines += ["", "| judge | sensitivity (broken flagged) | specificity (preserved passed) |",
              "| --- | ---: | ---: |"]
    for judge, c in report["overall"].items():
        lines.append(f"| {judge} | {fmt(c['sensitivity'])} "
                     f"[{fmt(c['sensitivity_ci95'][0], 2)}, {fmt(c['sensitivity_ci95'][1], 2)}] | "
                     f"{fmt(c['specificity'])} "
                     f"[{fmt(c['specificity_ci95'][0], 2)}, {fmt(c['specificity_ci95'][1], 2)}] |")
    return "\n".join(lines) + "\n"
