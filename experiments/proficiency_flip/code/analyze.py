"""Paired analysis of the proficiency flip replays.

    python -m experiments.proficiency_flip.code.analyze [--games DIR] [--out JSON] [--md FILE]

Reads every ``games/run_*/nl_vs_proficiency/`` tree written by ``lurkingwolf.eval``:
the baseline game, the identity (``nl``) replays and the ``flip_self`` / ``flip_others``
replays for Low and Neutral. For a treated speaker S the outcomes are the votes S
received, whether S was executed, whether the wolves or the seer targeted S that night,
and the mean reliability other players assign to S. Contrasts are paired on the same
checkpoint (seed, day) and speaker against the identity replay. Games (seeds) are the
independent units. Differences are averaged over days within each speaker, speakers
within each game, and games. Intervals resample games; p-values use sign-flip tests.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from components.helpers import read_jsonl
from components.stats import cluster_bootstrap, mean, paired_cohens_dz
from experiments.proficiency_validation.rewrite.runtime import fmt, now_iso
from experiments.proficiency_validation.validation.analysis import cluster_sign_flip_test

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GAMES_DIR = EXPERIMENT_DIR / "games"
COMPARISON = "nl_vs_proficiency"
SUMMARY_JSON = EXPERIMENT_DIR / "data" / "flip_summary.json"
SUMMARY_MD = EXPERIMENT_DIR / "summary.md"
CONDITIONS = ("low", "neutral", "high")
SPEAKER_OUTCOMES = ("votes_received", "executed", "wolf_target", "seer_target", "trust_in_speaker")
OWN_OUTCOMES = ("changed_vote", "trust_toward_others", "changed_role_target")
CONTRASTS = (("low", "identity"), ("neutral", "identity"), ("high", "identity"), ("low", "neutral"), ("high", "neutral"))
# Prespecified co-primary behavioural family (Low vs Neutral), Holm-corrected; the rest is secondary/exploratory.
CO_PRIMARY = ("trust_in_speaker", "votes_received")
N_BOOT, SEED = 2000, 101
GameDifferences = Dict[str, Dict[str, List[float]]]


def parse_run_id(run_id: str) -> Optional[dict]:
    """``<mode>[__<condition>__<param values...>][__pN]__dD`` as the runner names replays."""
    tokens = run_id.split("__")
    if len(tokens) < 2 or not re.fullmatch(r"d\d+", tokens[-1]):
        return None
    mode, day = tokens[0], int(tokens[-1][1:])
    if mode not in ("nl", "flip_self", "flip_others"):
        return None
    self_player = tokens[-2].upper() if len(tokens) > 2 and re.fullmatch(r"p\d+", tokens[-2]) else None
    condition = tokens[1] if len(tokens) > 2 and tokens[1] in CONDITIONS else None
    return {"mode": mode, "condition": condition, "self": self_player, "day": day}


def speaker_outcomes(result: dict, speaker: str) -> dict:
    """What happened to ``speaker`` in one day phase."""
    trust = [scores[speaker] for assessor, scores in (result.get("reliability") or {}).items()
             if assessor != speaker and isinstance(scores, dict) and scores.get(speaker) is not None]
    return {"votes_received": (result.get("votes_received") or {}).get(speaker, 0),
            "executed": int(result.get("executed") == speaker),
            "wolf_target": int(result.get("wolf_target") == speaker),
            "seer_target": int(result.get("seer_target") == speaker),
            "trust_in_speaker": mean(trust)}


def own_behaviour(result: dict, speaker: str, reference: dict, role: Optional[str]) -> dict:
    """How ``speaker`` acted in ``result`` relative to the same speaker in ``reference``."""
    own = (result.get("reliability") or {}).get(speaker) or {}
    outward = [v for other, v in own.items() if other != speaker and v is not None]
    if role == "werewolf":
        changed = int(result.get("wolf_target") != reference.get("wolf_target"))
    elif role == "seer":
        changed = int(result.get("seer_target") != reference.get("seer_target"))
    else:
        changed = None
    return {"changed_vote": int((result.get("votes") or {}).get(speaker) != (reference.get("votes") or {}).get(speaker)),
            "trust_toward_others": mean(outward), "changed_role_target": changed}


def _logged_text(row: dict) -> Tuple[str, str]:
    """(exposure category, text the listeners heard) for one rewrite-log row."""
    if row.get("fallback_to_original") is True:
        return "fallback", (row.get("original_text") or row.get("original_nl") or "").strip()
    if row.get("status") == "ok" and row.get("rewritten_text"):
        return "rewritten", row["rewritten_text"].strip()
    return "unavailable", ""


def _discussion_counter(rows: List[dict]) -> Counter:
    return Counter((row.get("speaker"), _logged_text(row)[1]) for row in rows
                   if row.get("phase", "day_speech") == "day_speech")


def exposure_summary(rows: List[dict], result: Optional[dict], meta: dict) -> dict:
    """Match the rewrite log to the treated utterances in the replay result.

    The replay result records the discussion speeches; the rewriter also logs the
    executed player's last words (``phase == "last_speech"``), which are counted
    as exposure but have no recorded speech to match against. A replay that the
    runner restarted after an interrupted attempt carries that attempt's rows ahead
    of the completed attempt's rows (the gate appended to the same log before the
    fix in ``GatedRewriter``); the completed attempt is the shortest suffix whose
    discussion rows match the recorded speeches.
    """
    expected = Counter((speech["speaker"], item["text"].strip())
                       for speech in (result or {}).get("speeches", [])
                       if ((speech["speaker"] == meta["self"]) == (meta["mode"] == "flip_self"))
                       for item in speech.get("nl", []) if item.get("text", "").strip())
    kept, interrupted = rows, 0
    if result is not None:
        for k in range(len(rows) + 1):
            if _discussion_counter(rows[k:]) == expected:
                kept, interrupted = rows[k:], k
                break
    counts = Counter(rewritten=0, fallback=0, unavailable=0, last_words=0)
    for row in kept:
        counts[_logged_text(row)[0]] += 1
        if row.get("phase", "day_speech") != "day_speech":
            counts["last_words"] += 1
    return {**counts, "recorded": len(rows), "interrupted_rows": interrupted,
            "expected": sum(expected.values()),
            "verified": bool(expected) and _discussion_counter(kept) == expected
            and all(row.get("condition") == meta["condition"] for row in kept)}


def load_tree(games_dir: Path) -> List[dict]:
    """One record per (seed, day): baseline, identity replay and flip replays, plus bookkeeping."""
    from lurkingwolf.eval.replay import extract_baseline_phase
    records = []
    for root in sorted(games_dir.glob(f"run_*/{COMPARISON}")):
        seed = root.parent.name.rsplit("_", 1)[-1]
        baseline_dir = root / "baseline"
        if not (baseline_dir / "game_record.json").exists():
            continue
        summary_path = root.parent / "experiment_summary.json"
        expected = {}
        if summary_path.exists():
            for r in json.loads(summary_path.read_text()).get("results", []):
                expected[r["run_id"]] = r.get("rc")
        for day_dir in sorted((root / "replays").glob("day_*")):
            day = int(day_dir.name.split("_")[1])
            try:
                baseline = extract_baseline_phase(baseline_dir, day).to_dict()
            except Exception as exc:                       # a day the baseline never reached
                baseline = {"error": f"{type(exc).__name__}: {exc}"}
            replays, gate, exposure = {}, Counter(), {}
            for run_dir in sorted(day_dir.iterdir()):
                meta = parse_run_id(run_dir.name)
                if meta is None or meta["day"] != day:
                    continue
                path = run_dir / "phase_result.json"
                result = json.loads(path.read_text()) if path.exists() and path.stat().st_size else None
                key = (meta["mode"], meta["condition"], meta["self"])
                replays[key] = result
                log = run_dir / "rewrites.jsonl"
                if log.exists():
                    rows = read_jsonl(log)
                    exposure[key] = exposure_summary(rows, result, meta)
                    for row in rows[exposure[key]["interrupted_rows"]:]:
                        gate[(meta["mode"], meta["condition"], row.get("status"))] += 1
            records.append({"seed": seed, "day": day, "baseline": baseline, "replays": replays,
                            "expected": {k: v for k, v in expected.items() if k.endswith(f"__d{day}")},
                            "gate": gate, "exposure": exposure, "root": str(root)})
    return records


def clean_exposure(rec: dict, key) -> bool:
    """A complete matching log with every treated utterance successfully rewritten."""
    exposure = rec.get("exposure", {}).get(key, {})
    return (exposure.get("verified") is True and exposure.get("rewritten", 0) > 0
            and exposure.get("fallback") == 0 and exposure.get("unavailable") == 0)


def paired_differences(records: List[dict], *, per_protocol: bool = False) -> dict:
    """Per-(seed, day, speaker) differences, grouped for the game-weighted analysis. With
    ``per_protocol`` only replays with complete treatment exposure (no fallback) enter."""
    games = lambda: defaultdict(lambda: defaultdict(list))
    speaker_diffs: Dict[Tuple[str, str], Dict[str, GameDifferences]] = defaultdict(lambda: defaultdict(games))
    role_diffs: Dict[Tuple[str, str], Dict[str, Dict[str, GameDifferences]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(games)))
    own_diffs: Dict[str, Dict[str, GameDifferences]] = defaultdict(lambda: defaultdict(games))
    noise: Dict[str, GameDifferences] = defaultdict(games)
    n_pairs = Counter()
    for rec in records:
        identity = rec["replays"].get(("nl", None, None))
        baseline = rec["baseline"]
        if not identity or "error" in baseline:
            continue
        seed, roles = rec["seed"], baseline.get("role_dist", {})
        alive = set(identity.get("alive_at_start") or roles)
        for speaker in sorted(alive):
            # identity-versus-baseline: the model's own run-to-run movement for this speaker
            for k, v in speaker_outcomes(identity, speaker).items():
                b = speaker_outcomes(baseline, speaker)[k]
                if v is not None and b is not None:
                    noise[k][seed][speaker].append(v - b)
            flips = {c: rec["replays"].get(("flip_self", c, speaker)) for c in CONDITIONS
                     if not per_protocol or clean_exposure(rec, ("flip_self", c, speaker))}
            outcomes = {"identity": speaker_outcomes(identity, speaker)}
            outcomes.update({c: speaker_outcomes(r, speaker) for c, r in flips.items() if r})
            for treatment, control in CONTRASTS:
                if treatment in outcomes and control in outcomes:
                    n_pairs[("flip_self", treatment, control)] += 1
                    for k in SPEAKER_OUTCOMES:
                        a, b = outcomes[treatment][k], outcomes[control][k]
                        if a is not None and b is not None:
                            speaker_diffs[(treatment, control)][k][seed][speaker].append(a - b)
                            side = "werewolf" if roles.get(speaker) == "werewolf" else "village"
                            role_diffs[(treatment, control)][side][k][seed][speaker].append(a - b)
            others = {c: rec["replays"].get(("flip_others", c, speaker)) for c in CONDITIONS
                      if not per_protocol or clean_exposure(rec, ("flip_others", c, speaker))}
            own = {"identity": own_behaviour(identity, speaker, identity, roles.get(speaker))}
            own.update({c: own_behaviour(r, speaker, identity, roles.get(speaker)) for c, r in others.items() if r})
            for treatment, control in CONTRASTS:
                if treatment in own and control in own:
                    n_pairs[("flip_others", treatment, control)] += 1
                    for k in OWN_OUTCOMES:
                        a, b = own[treatment][k], own[control][k]
                        if a is not None and b is not None:
                            own_diffs[f"{treatment}_vs_{control}"][k][seed][speaker].append(a - b)
    return {"speaker": speaker_diffs, "by_role": role_diffs, "own": own_diffs, "noise": noise, "n_pairs": n_pairs}


def game_weighted(diffs: GameDifferences) -> dict:
    """Average days within speakers, speakers within games, then games."""
    speaker_means = {g: [mean(days) for days in speakers.values() if days]
                     for g, speakers in diffs.items()}
    game_means = {g: mean(values) for g, values in speaker_means.items() if values}
    if not game_means:
        return {"n_pairs": 0, "n_speakers": 0, "n_games": 0, "estimate": None}
    clusters = {g: [m] for g, m in game_means.items()}
    boot = cluster_bootstrap(clusters, n_boot=N_BOOT, seed=SEED)
    boot.pop("draws", None)
    test = cluster_sign_flip_test(clusters, seed=SEED)
    pooled = [v for speakers in diffs.values() for days in speakers.values() for v in days]
    return {"n_pairs": len(pooled), "n_speakers": sum(len(v) for v in speaker_means.values()),
            "n_games": len(game_means),
            "estimate": boot["estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "cohens_dz": paired_cohens_dz(list(game_means.values())),
            "p_sign_flip": test["p_value"], "minimum_two_sided_p": test.get("minimum_two_sided_p"),
            "pooled_mean": mean(pooled)}


def availability(records: List[dict]) -> dict:
    out: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        for (mode, cond, _), result in rec["replays"].items():
            label = mode if cond is None else f"{mode}:{cond}"
            out[label]["present"] += int(result is not None)
            out[label]["missing_result"] += int(result is None)
        for run_id, rc in rec["expected"].items():
            meta = parse_run_id(run_id)
            if meta:
                label = meta["mode"] if meta["condition"] is None else f"{meta['mode']}:{meta['condition']}"
                out[label]["expected"] += 1
                out[label]["failed_rc"] += int(rc not in (0, None))
    gate: Dict[str, Counter] = defaultdict(Counter)
    exposure: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        for (mode, cond, status), n in rec["gate"].items():
            gate[f"{mode}:{cond}"][status or "unknown"] += n
        for key in rec["replays"]:
            mode, cond, _ = key
            if mode not in ("flip_self", "flip_others"):
                continue
            e = rec.get("exposure", {}).get(key, {})
            for field in ("rewritten", "fallback", "unavailable", "last_words", "interrupted_rows"):
                exposure[f"{mode}:{cond}"][field] += e.get(field, 0)
            exposure[f"{mode}:{cond}"]["replays"] += 1
            exposure[f"{mode}:{cond}"]["unverified_replays"] += int(e.get("verified") is not True)
            exposure[f"{mode}:{cond}"]["clean_replays"] += int(clean_exposure(rec, key))
    for label, e in exposure.items():
        total = e["rewritten"] + e["fallback"] + e["unavailable"]
        e["exposure_rate"] = e["rewritten"] / total if total else None
    return {"replays": {k: dict(v) for k, v in sorted(out.items())}, "gate_outcomes": {k: dict(v) for k, v in sorted(gate.items())},
            "exposure": {k: dict(v) for k, v in sorted(exposure.items())}}


def holm(pvalues: Dict[str, Optional[float]]) -> Dict[str, dict]:
    """Holm step-down adjustment over one prespecified family."""
    items = sorted(((p, k) for k, p in pvalues.items() if p is not None))
    out, running = {}, 0.0
    m = len(items)
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = {"p": p, "p_holm": running, "reject_05": running <= .05}
    for k, p in pvalues.items():
        out.setdefault(k, {"p": p, "p_holm": None, "reject_05": None})
    return out


def summarize(records: List[dict]) -> dict:
    diffs = paired_differences(records)
    effects = {f"{t}_vs_{c}": {k: game_weighted(diffs["speaker"][(t, c)][k]) for k in SPEAKER_OUTCOMES} for t, c in CONTRASTS}
    co_primary = {k: effects["low_vs_neutral"][k] for k in CO_PRIMARY}
    for k, adj in holm({k: e.get("p_sign_flip") for k, e in co_primary.items()}).items():
        co_primary[k] = {**co_primary[k], **adj}
    clean = paired_differences(records, per_protocol=True)
    per_protocol = {f"{t}_vs_{c}": {k: game_weighted(clean["speaker"][(t, c)][k]) for k in SPEAKER_OUTCOMES} for t, c in CONTRASTS}
    by_role = {f"{t}_vs_{c}": {side: {k: game_weighted(diffs["by_role"][(t, c)][side][k]) for k in SPEAKER_OUTCOMES}
                               for side in ("werewolf", "village")} for t, c in CONTRASTS}
    own = {name: {k: game_weighted(block[k]) for k in OWN_OUTCOMES} for name, block in diffs["own"].items()}
    noise = {k: game_weighted(diffs["noise"][k]) for k in SPEAKER_OUTCOMES}
    seeds = sorted({r["seed"] for r in records})
    return {"created_at": now_iso(), "games": seeds, "n_checkpoints": len(records),
            "days_per_game": dict(Counter(r["seed"] for r in records)),
            "n_pairs": {f"{m}:{t}_vs_{c}": n for (m, t, c), n in diffs["n_pairs"].items()},
            "availability": availability(records), "flip_self": effects, "co_primary_low_vs_neutral": co_primary,
            "flip_self_per_protocol": per_protocol, "flip_self_by_role": by_role,
            "flip_others": own, "identity_vs_baseline": noise, "estimand": "equal_game_weight",
            "analysis_policy": "assignment-based analysis of completed replays; per-protocol analysis requires complete logs matching every treated utterance, with all rewrites successful",
            "aggregation": "days within speaker, speakers within game, then games; equal weights at each step",
            "n_boot": N_BOOT, "seed": SEED}


def _row(label: str, e: dict) -> str:
    if not e.get("n_pairs"):
        return f"| {label} | 0 | 0 | – | – | – | – |"
    return (f"| {label} | {e['n_pairs']} | {e['n_games']} | {fmt(e['estimate'], 3)} "
            f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | {fmt(e.get('cohens_dz'), 2)} | "
            f"{fmt(e.get('p_sign_flip'))} | {fmt(e.get('pooled_mean'), 3)} |")


def render(s: dict) -> str:
    head = "| outcome | pairs | games | game-weighted Δ [95% CI] | d_z | p (exact sign-flip) | pooled Δ |"
    sep = "|---|--:|--:|--:|--:|--:|--:|"
    lines = ["# Proficiency flip replays", "",
             f"{len(s['games'])} baseline games ({', '.join(s['games'])}), "
             f"{s['n_checkpoints']} day checkpoints. Every replay repeats one day from its checkpoint; in `flip_self` "
             "one speaker's utterances are rewritten to the assigned level for that day, in `flip_others` everyone "
             "else's are. Differences are paired on the same checkpoint and speaker against the identity replay "
             f"(same checkpoint, no rewriter). Games are the independent units; with {len(s['games'])} games the "
             f"smallest attainable two-sided p is {fmt(2.0 ** (1 - len(s['games'])), 3) if s['games'] else '–'}.", "",
             "## Replay availability", "",
             "| replay | expected | with result | failed (non-zero exit) |", "|---|--:|--:|--:|"]
    for label, c in s["availability"]["replays"].items():
        lines.append(f"| {label} | {c.get('expected', 0)} | {c.get('present', 0)} | {c.get('failed_rc', 0)} |")
    if s["availability"]["gate_outcomes"]:
        lines += ["", "Gate outcomes recorded in the replays' `rewrites.jsonl` (one row per rewritten utterance):", "",
                  "| replay | " + " | ".join(sorted({k for v in s['availability']['gate_outcomes'].values() for k in v})) + " |",
                  "|---|" + "--:|" * len({k for v in s['availability']['gate_outcomes'].values() for k in v})]
        statuses = sorted({k for v in s["availability"]["gate_outcomes"].values() for k in v})
        for label, c in s["availability"]["gate_outcomes"].items():
            lines.append(f"| {label} | " + " | ".join(str(c.get(st, 0)) for st in statuses) + " |")
    if s["availability"].get("exposure"):
        lines += ["", "Treatment exposure (a rewrite that fails the gate after its retries falls back to the original "
                  "utterance; the branch keeps its assignment and the fallback is logged):", "",
                  "| replay | rewritten utterances | fallbacks to original | logged rewrite share | last-words rows | rows of interrupted attempts | replays | unverified logs | complete exposure |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for label, e in s["availability"]["exposure"].items():
            lines.append(f"| {label} | {e['rewritten']} | {e['fallback']} | {fmt(e['exposure_rate'], 3)} | {e.get('last_words', 0)} | {e.get('interrupted_rows', 0)} | {e['replays']} | {e['unverified_replays']} | {e['clean_replays']} |")
    lines += ["", f"Analysis policy: {s['analysis_policy']}.", "", f"Aggregation: {s['aggregation']}.", "",
              "## Co-primary outcomes (Low minus Neutral, `flip_self`, Holm-corrected)", "",
              "| outcome | pairs | games | game-weighted Δ [95% CI] | d_z | p (exact sign-flip) | p (Holm) | reject at 0.05 |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for k, e in s["co_primary_low_vs_neutral"].items():
        lines.append(f"| {k} | {e.get('n_pairs', 0)} | {e.get('n_games', 0)} | {fmt(e.get('estimate'), 3)} "
                     f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | {fmt(e.get('cohens_dz'), 2)} | "
                     f"{fmt(e.get('p_sign_flip'))} | {fmt(e.get('p_holm'))} | {fmt(e.get('reject_05'))} |")
    lines += ["", "## Speaker-directed treatment (`flip_self`): what happens to the treated speaker", ""]
    for name, block in s["flip_self"].items():
        lines += [f"### {name.replace('_vs_', ' minus ')}", "", head, sep]
        lines += [_row(k, block[k]) for k in SPEAKER_OUTCOMES]
        lines.append("")
    lines += ["### Per-protocol sensitivity (replays with complete treatment exposure only)", ""]
    for name, block in s["flip_self_per_protocol"].items():
        lines += [f"#### {name.replace('_vs_', ' minus ')}", "", head, sep]
        lines += [_row(k, block[k]) for k in SPEAKER_OUTCOMES]
        lines.append("")
    lines += ["`votes_received` counts first-round votes against the speaker; `executed`, `wolf_target` and "
              "`seer_target` are 0/1; `trust_in_speaker` is the mean reliability score other players assign the speaker "
              "after the day. The identity-minus-baseline rows below are the noise floor of the model itself.", "",
              "### identity minus baseline (noise floor)", "", head, sep]
    lines += [_row(k, s["identity_vs_baseline"][k]) for k in SPEAKER_OUTCOMES]
    lines += ["", "### By the treated speaker's side (exploratory)", ""]
    for name, sides in s["flip_self_by_role"].items():
        for side, block in sides.items():
            lines += [f"#### {name.replace('_vs_', ' minus ')}, speaker is {side}", "", head, sep]
            lines += [_row(k, block[k]) for k in SPEAKER_OUTCOMES]
            lines.append("")
    lines += ["## Surrounding treatment (`flip_others`): how the untreated speaker behaves", ""]
    for name, block in s["flip_others"].items():
        lines += [f"### {name.replace('_vs_', ' minus ')}", "", head, sep]
        lines += [_row(k, block[k]) for k in OWN_OUTCOMES]
        lines.append("")
    lines += ["`changed_vote` is 1 when the speaker voted for a different target than in the control replay; "
              "`trust_toward_others` is the speaker's mean reliability score for the others; `changed_role_target` "
              "applies to wolves and the seer only.", "",
              "The per-protocol table uses replays whose logs match all treated utterances and record a successful rewrite for each.", "",
              "## Files", "", "| file | contents |", "|---|---|",
              "| `data/flip_summary.json` | every number above, with intervals and tests |",
              "| `games/` (ignored) | baselines, replays, `subprocess.log` and `rewrites.jsonl` per replay |",
              "| `code/analyze.py` | rebuilds this summary from `games/`; no model calls |", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", type=Path, default=GAMES_DIR)
    parser.add_argument("--out", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--md", type=Path, default=SUMMARY_MD)
    args = parser.parse_args(argv)
    records = load_tree(args.games)
    if not records:
        raise SystemExit(f"no {COMPARISON} replays under {args.games}")
    summary = summarize(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    args.md.write_text(render(summary), encoding="utf-8")
    print(f"{summary['n_checkpoints']} checkpoints from {len(summary['games'])} games -> {args.out}, {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
