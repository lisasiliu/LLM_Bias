"""Paired analysis of the politeness flip replays.

    python -m experiments.politeness_flip.code.analyze [--games DIR] [--out JSON] [--md FILE]

Reads every ``games/run_*/nl_vs_politeness/`` tree written by ``lurkingwolf.eval``: the
baseline game, the two identity (``nl``) replays and the ``flip_self`` / ``flip_others``
replays for the four Brown & Levinson strategies. Outcomes, pairing, aggregation and the
exposure checks are the proficiency flip's (``experiments.proficiency_flip.code.analyze``);
the design differs in its conditions.

Every strategy rewrites the utterance, so a contrast between two strategies isolates
politeness. A strategy against the identity replay also carries the effect of being
rewritten at all, which the second identity replay measures on its own.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from experiments.proficiency_validation.rewrite.runtime import fmt, now_iso
from experiments.proficiency_flip.code import analyze as flip
from experiments.proficiency_flip.code.analyze import (
    OWN_OUTCOMES, SPEAKER_OUTCOMES, availability, game_weighted, holm, paired_differences)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GAMES_DIR = EXPERIMENT_DIR / "games"
COMPARISON = "nl_vs_politeness"
RUN = "qwen35_27b"                      # one directory per run, as in experiments/proficiency_validation
RUN_DIR = EXPERIMENT_DIR / "data" / "runs" / RUN
SUMMARY_JSON = RUN_DIR / "politeness_summary.json"
SUMMARY_MD = EXPERIMENT_DIR / "summary.md"
# Brown & Levinson's strategies, from no redress to the most indirect.
CONDITIONS = ("bald_on_record", "positive_politeness", "negative_politeness", "off_record")
REFERENCE = "bald_on_record"          # the unredressed strategy every other one is read against
# The two outcomes Holm corrects, mirroring the proficiency flip. Which family RQ2 treats as
# confirmatory is the politeness experiment's own decision; everything else here is descriptive.
CO_PRIMARY = ("trust_in_speaker", "votes_received")
# Each strategy against the untreated replay, then against the unredressed strategy.
CONTRASTS = tuple((c, "identity") for c in CONDITIONS) + tuple(
    (c, REFERENCE) for c in CONDITIONS if c != REFERENCE)


def load_tree(games_dir: Path = GAMES_DIR) -> List[dict]:
    return flip.load_tree(games_dir, comparison=COMPARISON, conditions=CONDITIONS)


def summarize(records: List[dict]) -> dict:
    diffs = paired_differences(records, conditions=CONDITIONS, contrasts=CONTRASTS)
    clean = paired_differences(records, per_protocol=True, conditions=CONDITIONS, contrasts=CONTRASTS)
    effects = {f"{t}_vs_{c}": {k: game_weighted(diffs["speaker"][(t, c)][k]) for k in SPEAKER_OUTCOMES}
               for t, c in CONTRASTS}
    per_protocol = {f"{t}_vs_{c}": {k: game_weighted(clean["speaker"][(t, c)][k]) for k in SPEAKER_OUTCOMES}
                    for t, c in CONTRASTS}
    by_role = {f"{t}_vs_{c}": {side: {k: game_weighted(diffs["by_role"][(t, c)][side][k]) for k in SPEAKER_OUTCOMES}
                               for side in ("werewolf", "village")} for t, c in CONTRASTS}
    own = {name: {k: game_weighted(block[k]) for k in OWN_OUTCOMES} for name, block in diffs["own"].items()}
    noise = {k: game_weighted(diffs["noise"][k]) for k in SPEAKER_OUTCOMES if k != "trust_in_speaker"}
    repeat_noise = {k: game_weighted(diffs["repeat_noise"][k]) for k in SPEAKER_OUTCOMES + ("changed_vote",)}
    # Holm over the three strategies read against the unredressed one, within each co-primary outcome.
    strategy_family = {f"{t}_vs_{c}": effects[f"{t}_vs_{c}"] for t, c in CONTRASTS if c == REFERENCE}
    adjusted = {k: holm({name: block[k].get("p_sign_flip") for name, block in strategy_family.items()})
                for k in CO_PRIMARY}
    seeds = sorted({r["seed"] for r in records})
    return {"created_at": now_iso(), "games": seeds, "n_checkpoints": len(records),
            "days_per_game": dict(Counter(r["seed"] for r in records)),
            "conditions": list(CONDITIONS), "reference_condition": REFERENCE,
            "co_primary": list(CO_PRIMARY),
            "n_pairs": {f"{m}:{t}_vs_{c}": n for (m, t, c), n in diffs["n_pairs"].items()},
            "availability": availability(records, conditions=CONDITIONS), "flip_self": effects,
            "flip_self_per_protocol": per_protocol, "flip_self_by_role": by_role,
            "flip_others": own, "identity_vs_baseline": noise, "identity_vs_repeat": repeat_noise,
            "holm_vs_reference": adjusted, "estimand": "equal_game_weight",
            "analysis_policy": "assignment-based analysis of completed replays; per-protocol analysis requires "
                               "complete logs matching every treated utterance, with all rewrites successful",
            "aggregation": "days within speaker, speakers within game, then games; equal weights at each step",
            "n_boot": flip.N_BOOT, "seed": flip.SEED}


def render(s: dict) -> str:
    head = "| outcome | pairs | games | game-weighted Δ [95% CI] | d_z | p (exact sign-flip) | pooled Δ |"
    sep = "|---|--:|--:|--:|--:|--:|--:|"
    name = lambda key: key.replace("_vs_", " minus ").replace("_", " ")
    lines = ["# Politeness flip replays", "",
             f"{len(s['games'])} baseline games ({', '.join(s['games'])}), {s['n_checkpoints']} day checkpoints. "
             "Every replay repeats one day from its checkpoint; in `flip_self` one speaker's utterances are "
             "rewritten with the assigned Brown & Levinson strategy for that day, in `flip_others` everyone "
             "else's are. Differences are paired on the same checkpoint and speaker. Games are the independent "
             f"units; with {len(s['games'])} games the smallest attainable two-sided p is "
             f"{fmt(2.0 ** (1 - len(s['games'])), 3) if s['games'] else '–'}.", "",
             f"Strategies: {', '.join(s['conditions'])}. Every strategy rewrites the utterance, so a contrast "
             f"between two of them isolates politeness; `{s['reference_condition']}` carries no redress and is "
             "the reference. A strategy against the identity replay also carries the effect of being rewritten "
             "at all, which the second identity replay measures on its own.", "",
             "## Replay availability", "",
             "| replay | expected | with result | failed (non-zero exit) |", "|---|--:|--:|--:|"]
    for label, c in s["availability"]["replays"].items():
        lines.append(f"| {label} | {c.get('expected', 0)} | {c.get('present', 0)} | {c.get('failed_rc', 0)} |")
    if s["availability"]["gate_outcomes"]:
        statuses = sorted({k for v in s["availability"]["gate_outcomes"].values() for k in v})
        lines += ["", "Rewriter outcomes recorded in the replays' `rewrites.jsonl` (one row per utterance):", "",
                  "| replay | " + " | ".join(statuses) + " |", "|---|" + "--:|" * len(statuses)]
        for label, c in s["availability"]["gate_outcomes"].items():
            lines.append(f"| {label} | " + " | ".join(str(c.get(st, 0)) for st in statuses) + " |")
    if s["availability"].get("exposure"):
        lines += ["", "Treatment exposure. The politeness rewriter fails closed, so a replay either delivers "
                  "its rewrite or fails and is re-run; there is no fallback to untreated text.", "",
                  "| replay | rewritten utterances | fallbacks to original | logged rewrite share | last-words rows "
                  "| rows of interrupted attempts | replays | unverified logs | complete exposure |",
                  "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
        for label, e in s["availability"]["exposure"].items():
            lines.append(f"| {label} | {e['rewritten']} | {e['fallback']} | {fmt(e['exposure_rate'], 3)} | "
                         f"{e.get('last_words', 0)} | {e.get('interrupted_rows', 0)} | {e['replays']} | "
                         f"{e['unverified_replays']} | {e['clean_replays']} |")
    lines += ["", f"Analysis policy: {s['analysis_policy']}.", "", f"Aggregation: {s['aggregation']}.", "",
              f"## Strategies against `{s['reference_condition']}` (Holm-corrected per outcome)", "",
              "| contrast | outcome | pairs | games | game-weighted Δ [95% CI] | p | p (Holm) | reject at 0.05 |",
              "|---|---|--:|--:|--:|--:|--:|--:|"]
    for outcome, family in s["holm_vs_reference"].items():
        for key, adj in family.items():
            e = s["flip_self"][key]
            lines.append(f"| {name(key)} | {outcome} | {e[outcome].get('n_pairs', 0)} | {e[outcome].get('n_games', 0)} | "
                         f"{fmt(e[outcome].get('estimate'), 3)} [{fmt(e[outcome].get('ci_low'), 3)}, "
                         f"{fmt(e[outcome].get('ci_high'), 3)}] | {fmt(adj.get('p'))} | {fmt(adj.get('p_holm'))} | "
                         f"{fmt(adj.get('reject_05'))} |")
    lines += ["", "## Speaker-directed treatment (`flip_self`): what happens to the treated speaker", ""]
    for key, block in s["flip_self"].items():
        lines += [f"### {name(key)}", "", head, sep] + [flip._row(k, block[k]) for k in SPEAKER_OUTCOMES] + [""]
    lines += ["### Per-protocol sensitivity (replays with complete treatment exposure only)", ""]
    for key, block in s["flip_self_per_protocol"].items():
        lines += [f"#### {name(key)}", "", head, sep] + [flip._row(k, block[k]) for k in SPEAKER_OUTCOMES] + [""]
    lines += ["`votes_received` counts first-round votes against the speaker; `executed`, `wolf_target` and "
              "`seer_target` are 0/1; `trust_in_speaker` is the mean reliability score the village players alive "
              "after the night assign the speaker in their post-day memory update.", "",
              "### identity minus recorded game", "", head, sep]
    lines += [flip._row(k, s["identity_vs_baseline"][k]) for k in s["identity_vs_baseline"]]
    if any(e.get("n_pairs") for e in s.get("identity_vs_repeat", {}).values()):
        lines += ["", "### identity minus identity repeat (run-to-run variation)", "", head, sep]
        lines += [flip._row(k, s["identity_vs_repeat"][k]) for k in s["identity_vs_repeat"]]
    lines += ["", "### By the treated speaker's side (exploratory)", ""]
    for key, sides in s["flip_self_by_role"].items():
        for side, block in sides.items():
            lines += [f"#### {name(key)}, speaker is {side}", "", head, sep]
            lines += [flip._row(k, block[k]) for k in SPEAKER_OUTCOMES] + [""]
    lines += ["## Surrounding treatment (`flip_others`): how the untreated speaker behaves", ""]
    for key, block in s["flip_others"].items():
        lines += [f"### {name(key)}", "", head, sep] + [flip._row(k, block[k]) for k in OWN_OUTCOMES] + [""]
    lines += ["`changed_vote` is 1 when the speaker voted for a different target than in the control replay; "
              "`trust_toward_others` is the speaker's mean reliability score for the others; `changed_role_target` "
              "applies to wolves and the seer only.", "",
              "## Files", "", "| file | contents |", "|---|---|",
              f"| `{SUMMARY_JSON.relative_to(EXPERIMENT_DIR)}` | every number above, with intervals and tests |",
              "| `games/` | baselines, replays, `subprocess.log` and `rewrites.jsonl` per replay |",
              "| `code/analyze.py` | rebuilds this summary from `games/`; no model calls |", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=Path, default=GAMES_DIR)
    parser.add_argument("--out", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--md", type=Path, default=SUMMARY_MD)
    args = parser.parse_args(argv)
    records = load_tree(args.games)
    summary = summarize(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    args.md.write_text(render(summary))
    print(f"{len(records)} checkpoints from {len(summary['games'])} games -> {args.out}, {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
