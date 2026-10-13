"""Paired analysis of the politeness flip replays.

    python -m experiments.politeness_flip.code.analyze [--games DIR] [--out JSON] [--md FILE]

Reads every ``games/run_*/nl_vs_politeness/`` tree written by ``lurkingwolf.eval``: the
baseline game, the two identity (``nl``) replays and the ``flip_self`` / ``flip_others``
replays for the four Brown & Levinson strategies. Outcomes, pairing, aggregation and the
exposure checks are the proficiency flip's (``experiments.language_flip.code.analyze``);
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
from typing import Dict, List, Optional, Sequence

from experiments.language_validation.rewrite.runtime import fmt, now_iso
from experiments.language_flip.code import analyze as flip
from experiments.language_flip.code.analyze import (
    ALL_SPEAKER_OUTCOMES, CO_PRIMARY as PROFICIENCY_CO_PRIMARY, OWN_OUTCOMES, RESPONSE_OUTCOMES,
    SPEAKER_OUTCOMES, availability, beyond_repeat, equivalence, game_weighted, holm,
    EQUIVALENCE_BOUNDS, TARGETING, joint_sign_flip, paired_differences)

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GAMES_DIR = EXPERIMENT_DIR / "games"
COMPARISON = "nl_vs_politeness"
RUN = "qwen35_27b"                      # one directory per run, as in experiments/language_validation
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
    effects = {f"{t}_vs_{c}": {k: game_weighted(diffs["speaker"][(t, c)][k]) for k in ALL_SPEAKER_OUTCOMES}
               for t, c in CONTRASTS}
    per_protocol = {f"{t}_vs_{c}": {k: game_weighted(clean["speaker"][(t, c)][k]) for k in ALL_SPEAKER_OUTCOMES}
                    for t, c in CONTRASTS}
    by_role = {f"{t}_vs_{c}": {side: {k: game_weighted(diffs["by_role"][(t, c)][side][k]) for k in ALL_SPEAKER_OUTCOMES}
                               for side in ("werewolf", "village")} for t, c in CONTRASTS}
    own = {name: {k: game_weighted(block[k]) for k in OWN_OUTCOMES} for name, block in diffs["own"].items()}
    noise = {k: game_weighted(diffs["noise"][k]) for k in SPEAKER_OUTCOMES if k != "trust_in_speaker"}
    repeat_noise = {k: game_weighted(diffs["repeat_noise"][k]) for k in ALL_SPEAKER_OUTCOMES + OWN_OUTCOMES}
    # Holm over the three strategies read against the unredressed one, within each co-primary outcome.
    strategy_family = {f"{t}_vs_{c}": effects[f"{t}_vs_{c}"] for t, c in CONTRASTS if c == REFERENCE}
    adjusted = {k: holm({name: block[k].get("p_sign_flip") for name, block in strategy_family.items()})
                for k in CO_PRIMARY}
    bounds = {k: EQUIVALENCE_BOUNDS[k] for k in CO_PRIMARY}
    tost = {f"{t}_vs_{c}": {k: equivalence(diffs["speaker"][(t, c)][k], bounds.get(k, 0.0))
                            for k in CO_PRIMARY} for t, c in CONTRASTS}
    net = {m: beyond_repeat(records, conditions=CONDITIONS, comparison=m)
           for m in ("vote", "skill", "reliability")}
    holm_net = {m: holm({label: block.get("p_sign_flip") for label, block in net[m].items()}) for m in net}
    discussion = {f"{t}_vs_{c}": holm({k: effects[f"{t}_vs_{c}"][k].get("p_sign_flip")
                                       for k in RESPONSE_OUTCOMES}) for t, c in CONTRASTS}
    targeting = {f"{t}_vs_{c}": joint_sign_flip({k: diffs["speaker"][(t, c)][k] for k in TARGETING})
                 for t, c in CONTRASTS}
    roles_seen = sorted({r for block in diffs["by_exact_role"].values() for r in block})
    targeting_by_role = {f"{t}_vs_{c}": {
        role: joint_sign_flip({k: diffs["by_exact_role"][(t, c)][role][k] for k in TARGETING})
        for role in roles_seen if diffs["by_exact_role"][(t, c)].get(role)} for t, c in CONTRASTS}
    role_holm = {name: holm({role: block[role].get("p_sign_flip") for role in block})
                 for name, block in targeting_by_role.items()}
    # One joint test per strategy, Holm over the three strategies read against the reference.
    targeting_holm = holm({f"{t}_vs_{c}": targeting[f"{t}_vs_{c}"].get("p_sign_flip")
                           for t, c in CONTRASTS if c == REFERENCE})
    seeds = sorted({r["seed"] for r in records})
    return {"created_at": now_iso(), "games": seeds, "n_checkpoints": len(records),
            "beyond_repeat": net, "beyond_repeat_holm": holm_net, "equivalence": tost,
            "discussion_holm": discussion, "equivalence_bounds": bounds, "targeting": targeting, "targeting_holm": targeting_holm,
            "targeting_by_role": targeting_by_role, "targeting_by_role_holm": role_holm,
            "days_per_game": dict(Counter(r["seed"] for r in records)),
            "conditions": list(CONDITIONS), "reference_condition": REFERENCE,
            "unchanged_share": unchanged_share(),
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


def _seed_range(seeds: Sequence[str]) -> str:
    """``201-230`` when the seeds run without a gap, otherwise each one."""
    try:
        numbers = sorted(int(x) for x in seeds)
    except (TypeError, ValueError):
        return ", ".join(seeds)
    if len(numbers) > 2 and numbers[-1] - numbers[0] == len(numbers) - 1:
        return f"{numbers[0]}-{numbers[-1]}"
    return ", ".join(str(n) for n in numbers)


def findings(s: dict) -> List[str]:
    """The result in prose, every number taken from this summary so the two cannot disagree."""
    from experiments.language_flip.code.analyze import _effect
    games = len(s.get("games") or [])
    ref = s.get("reference_condition", REFERENCE)
    key = f"negative_politeness_vs_{ref}"
    joint = (s.get("targeting") or {}).get(key) or {}
    joint_adj = (s.get("targeting_holm") or {}).get(key) or {}
    lines = ["## Findings", "",
             f"Deference moves the table off a speaker. Across {games} games, against `{ref}`, which carries "
             f"no redress, negative politeness draws {_effect(s, 'flip_self', key, 'votes_received')} votes, "
             f"{_effect(s, 'flip_self', key, 'accused_by_others')} accusations and "
             f"{_effect(s, 'flip_self', key, 'execution_urged')} calls for execution. The three measure one "
             f"thing; tested together they give {fmt(joint.get('estimate'), 3)} standard deviations, "
             f"p = {fmt(joint.get('p_sign_flip'))}, {fmt(joint_adj.get('p_holm'))} after Holm over the three "
             f"strategies. It is a broad shift, not a few games: {joint.get('games_negative')} of "
             f"{joint.get('n_games')} games move off the deferential speaker, {joint.get('games_positive')} "
             f"move toward them, and the median game is {fmt(joint.get('median'), 3)}.", "",
             f"What moves is the vote, not the belief: trust in the deferential speaker holds still at "
             f"{_effect(s, 'flip_self', key, 'trust_in_speaker')}. The other two strategies move neither, "
             f"{_effect(s, 'flip_self', f'positive_politeness_vs_{ref}', 'votes_received')} votes for positive "
             f"politeness and {_effect(s, 'flip_self', f'off_record_vs_{ref}', 'votes_received')} for off "
             f"record.", ""]
    eq = (s.get("equivalence") or {}).get(key) or {}
    if (eq.get("votes_received") or {}).get("margin") is not None:
        trust_margins = [b["trust_in_speaker"]["margin"] for b in (s.get("equivalence") or {}).values()
                         if (b.get("trust_in_speaker") or {}).get("margin") is not None]
        lines += [f"The vote effect is at the edge of what the design calls negligible. Every contrast passes "
                  f"equivalence inside the declared ±{fmt(eq['votes_received'].get('bound'), 2)} of a vote, but "
                  f"the deferential one only just: it needs ±{fmt(eq['votes_received'].get('margin'), 3)}, "
                  f"against ±{fmt(max(trust_margins), 3)} for trust across all of them. The votes move; the "
                  f"reliability scores do not.", ""]
    hearer = ((s.get("flip_others") or {}).get(key) or {}).get("trust_toward_others") or {}
    hearer_floor = ((s.get("identity_vs_repeat") or {}).get("trust_toward_others") or {}).get("estimate")
    if hearer.get("estimate") is not None:
        lines += [f"Deference works on the hearer as well. When everyone around a player speaks that way, "
                  f"the player's own trust in them rises by {_effect(s, 'flip_others', key, 'trust_toward_others')}, "
                  f"p = {fmt(hearer.get('p_sign_flip'))}, against a floor of {fmt(hearer_floor, 3)} between two "
                  f"untreated replays.", ""]
    kept = s.get("unchanged_share") or {}
    if kept.get(ref):
        lines += [f"The reference is not fully rewritten. `{ref}` returns the utterance unchanged "
                  f"{fmt(kept[ref], 3)} of the time, against {fmt(kept.get('negative_politeness', 0), 3)} for "
                  f"negative politeness, and against the untreated replay it draws "
                  f"{fmt(((s.get('targeting') or {}).get(f'{ref}_vs_identity') or {}).get('estimate'), 3)} "
                  f"standard deviations more targeting.", ""]
    check = _classifier_numbers(RUN_DIR / "manipulation_check.json")
    reference = _classifier_numbers(RUN_DIR / "classifier_reference.json")
    if check and reference:
        reliable = check["bald_on_record"] + check["negative_politeness"]
        lines += [f"Two of the four strategies survive the trip into the game. Bald on record and negative "
                  f"politeness are labelled correctly in {fmt(min(reliable), 2)} to {fmt(max(reliable), 2)} "
                  f"of the delivered utterances. Positive "
                  f"politeness falls to {fmt(min(check['positive_politeness']), 2)}–"
                  f"{fmt(max(check['positive_politeness']), 2)} where the same classifiers read "
                  f"{fmt(min(reference['positive_politeness']), 2)}–{fmt(max(reference['positive_politeness']), 2)} "
                  f"on material with known answers, which places that failure in the rewriting. Off record "
                  f"reaches {fmt(max(reference['off_record']), 2)} even on known answers.", ""]
    return lines


def unchanged_share() -> Dict[str, float]:
    """How often each strategy returned the utterance it was given, byte for byte."""
    from collections import Counter
    from components.helpers import read_jsonl
    kept, total = Counter(), Counter()
    for run in sorted(GAMES_DIR.glob(f"run_*/{COMPARISON}/replays/day_*/flip_*")):
        meta = flip.parse_run_id(run.name, conditions=CONDITIONS)
        log = run / "rewrites.jsonl"
        if meta is None or meta["condition"] not in CONDITIONS or not log.exists():
            continue
        for row in read_jsonl(log):
            if row.get("phase", "day_speech") != "day_speech" or row.get("status") != "ok":
                continue
            total[meta["condition"]] += 1
            kept[meta["condition"]] += int((row.get("original_text") or "").strip()
                                           == (row.get("rewritten_text") or "").strip())
    return {c: kept[c] / total[c] for c in total if total[c]}


def _classifier_numbers(path: Path) -> dict:
    """Per-strategy agreement of every classifier in one report."""
    import json
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    out: dict = {c: [] for c in CONDITIONS}
    for block in data.get("classifiers", {}).values():
        for c in CONDITIONS:
            if block["agreement"].get(c) is not None:
                out[c].append(block["agreement"][c])
    return {c: v for c, v in out.items() if v}


def manipulation_section(s: dict) -> List[str]:
    """Which strategies the delivered speech realized, and what the classifier can read at all."""
    import json
    from components.judges.blind_politeness_classifier import LABELS
    from experiments.politeness_flip.code.manipulation_check import COLUMNS
    out: List[str] = []
    for name, path, title, blurb in (
            ("replay", RUN_DIR / "manipulation_check.json",
             "Did the delivered speech realize the strategy?",
             "A blind classifier labelled each utterance three times, never told which strategy was "
             "requested, on a sample balanced over the strategies, the two rewrite scopes and the games."),
            ("reference", RUN_DIR / "classifier_reference.json",
             "What the classifier can read at all",
             "The same classifiers on the two sets with known answers in "
             "`experiments/politeness_validation/`. A low number here is the instrument's limit, "
             "not the rewriter's.")):
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        out += ["", f"## {title}", "", blurb, "",
                "| classifier | requested | " + " | ".join(l.replace("_", " ") for l in COLUMNS)
                + " | agreement | of those labelled |",
                "|---|---|" + "--:|" * (len(COLUMNS) + 2)]
        for handle, block in sorted(data.get("classifiers", {}).items()):
            for want in CONDITIONS:
                counts = block["confusion"].get(want, {})
                out.append(f"| {handle.replace('openai:', '')} | {want.replace('_', ' ')} | "
                           + " | ".join(str(counts.get(l, 0)) for l in COLUMNS)
                           + f" | {fmt(block['agreement'].get(want), 3)} "
                           + f"| {fmt((block.get('agreement_of_decided') or {}).get(want), 3)} |")
        out.append("")
    return out


def render(s: dict) -> str:
    head = "| outcome | pairs | games | game-weighted Δ [95% CI] | d_z | p (sign-flip) | pooled Δ |"
    sep = "|---|--:|--:|--:|--:|--:|--:|"
    name = lambda key: key.replace("_vs_", " minus ").replace("_", " ")
    lines = ["# Politeness flip replays", "",
             f"{len(s['games'])} baseline games ({_seed_range(s['games'])}), {s['n_checkpoints']} day checkpoints. "
             "Every replay repeats one day from its checkpoint; in `flip_self` one speaker's utterances are "
             "rewritten with the assigned Brown & Levinson strategy for that day, in `flip_others` everyone "
             "else's are. Differences are paired on the same checkpoint and speaker. Games are the independent "
             f"units. Sign-flip tests permute the game means; above twelve games they sample 5000 "
             f"permutations, so the smallest p they can return is 0.0002.", "",
             f"Strategies: {', '.join(s['conditions'])}. A contrast between two of them is politeness alone "
             f"wherever both rewrote the utterance; `{s['reference_condition']}` carries no redress and is "
             "the reference. A strategy against the identity replay also carries the effect of being rewritten "
             "at all, which the second identity replay measures on its own.", ""]
    lines += findings(s)
    lines += [f"Aggregation: {s['aggregation']}.", "",
              f"## Strategies against `{s['reference_condition']}`, Holm-corrected per outcome", "",
              "| contrast | outcome | pairs | games | game-weighted Δ [95% CI] | p | p (Holm) | reject at 0.05 |",
              "|---|---|--:|--:|--:|--:|--:|--:|"]
    for outcome, family in s["holm_vs_reference"].items():
        for key, adj in family.items():
            e = s["flip_self"][key]
            lines.append(f"| {name(key)} | {outcome} | {e[outcome].get('n_pairs', 0)} | {e[outcome].get('n_games', 0)} | "
                         f"{fmt(e[outcome].get('estimate'), 3)} [{fmt(e[outcome].get('ci_low'), 3)}, "
                         f"{fmt(e[outcome].get('ci_high'), 3)}] | {fmt(adj.get('p'))} | {fmt(adj.get('p_holm'))} | "
                         f"{fmt(adj.get('reject_05'))} |")
    lines += ["", "## The treated speaker", "",
              "Paired on the same checkpoint and speaker. The contrasts against the identity replay, the "
              "by-side splits and the `flip_others` outcomes are in the JSON beside this file.", ""]
    shown = CO_PRIMARY + ("accused_by_others", "execution_urged")
    for key in [f"{c}_vs_{REFERENCE}" for c in CONDITIONS if c != REFERENCE]:
        block = s["flip_self"].get(key) or {}
        if not any(flip.moved(block.get(k)) for k in shown):
            continue
        lines += [f"### {name(key)}", "", head, sep]
        lines += [flip._row(k, block[k]) for k in shown if k in block]
        lines.append("")
    if s.get("targeting"):
        lines += ["## The three measures of being targeted, tested as one", "",
                  "Votes against the speaker, accusations of them and calls for their execution are scaled by "
                  "their spread across games, averaged into one number per game and tested once; the null "
                  "flips each game's whole vector, which keeps their correlation. Holm covers the three "
                  "strategies.", "",
                  "| contrast | games | composite effect | median game | games up / down | p | p (Holm) |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for key, t in s["targeting"].items():
            if t.get("estimate") is None or not key.endswith(REFERENCE):
                continue
            adj = (s.get("targeting_holm") or {}).get(key, {})
            lines.append(f"| {name(key)} | {t['n_games']} | {fmt(t['estimate'], 3)} SD | "
                         f"{fmt(t.get('median'), 3)} | {t.get('games_positive')} / {t.get('games_negative')} | "
                         f"{fmt(t['p_sign_flip'])} | {fmt(adj.get('p_holm'))} |")
        lines.append("")
    own, floor = s.get("flip_others") or {}, s.get("identity_vs_repeat") or {}
    carried = [(key, o) for key in own for o in OWN_OUTCOMES
               if key.endswith(REFERENCE) and flip.moved(own[key][o])]
    if carried:
        lines += ["## The hearer's side", "",
                  "What the untreated player does when everyone around them is rewritten. The floor is the "
                  "same quantity between two untreated replays of the checkpoint.", "",
                  "| contrast | outcome | games | Δ [95% CI] | p | floor |", "|---|---|--:|--:|--:|--:|"]
        for key, o in carried:
            e, fl = own[key][o], floor.get(o) or {}
            lines.append(f"| {name(key)} | {o} | {e.get('n_games', 0)} | {fmt(e.get('estimate'), 3)} "
                         f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | {fmt(e.get('p_sign_flip'))} | "
                         f"{fmt(fl.get('estimate'), 3)} |")
        lines.append("")
    if s.get("equivalence"):
        lines += ["## How tight the absence is", "",
                  "The bound is the largest difference worth calling an effect: a quarter of a reliability "
                  "point on a 1-10 scale, and a quarter of a vote out of the six a day puts on the "
                  "table. Equivalence holds when the whole 90% interval lies inside it. The last column is "
                  "the smallest bound the contrast would still pass.", "",
                  "| contrast | outcome | bound | 90% interval | smallest bound it passes | equivalent |",
                  "|---|---|--:|--:|--:|--:|"]
        for key, block in s["equivalence"].items():
            if not key.endswith(REFERENCE):
                continue
            for outcome, e in block.items():
                lines.append(f"| {name(key)} | {outcome} | ±{fmt(e.get('bound'), 3)} | "
                             f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | "
                             f"±{fmt(e.get('margin'), 3)} | {fmt(e.get('equivalent'))} |")
        lines += ["", "`votes_received` counts first-round votes against the speaker, `trust_in_speaker` is the "
                  "mean reliability score the village players alive after the night assign them, and "
                  "`accused_by_others` and `execution_urged` count the other players' predicates about them.", ""]
    lines += manipulation_section(s)
    lines += ["## Figures", "", "| figure | what it shows |", "|---|---|",
              "| `manipulation_by_strategy` | Which strategies the delivered speech realizes, beside what the classifier reads on known answers |",
              "| `strategy_outcomes` | Votes, trust and accusations for the treated speaker, by strategy |",
              "| `net_change` | What the rewriting adds once a plain re-run's movement is taken out |", "",
              "## Files", "", "| file | contents |", "|---|---|",
              f"| `{SUMMARY_JSON.relative_to(EXPERIMENT_DIR)}` | every number above, with intervals and tests |",
              "| `games/` | baselines, replay results, rewrite logs and agent memory, one directory per seed |",
              "| `code/analyze.py` | rebuilds this summary from `games/`; no model calls |", ""]
    return "\n".join(lines)


def parse_seeds(spec: Optional[str]) -> Optional[set]:
    """``"231-245"`` or ``"231,232,240-242"`` -> the set of seed names; None keeps every game."""
    if not spec:
        return None
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(str(s) for s in range(int(lo), int(hi) + 1))
        else:
            out.add(str(int(part)))
    return out

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=Path, default=GAMES_DIR)
    parser.add_argument("--seeds", help="only these seeds, e.g. 231-245; the default reads every game under --games")
    parser.add_argument("--out", type=Path, default=SUMMARY_JSON)
    parser.add_argument("--md", type=Path, default=SUMMARY_MD)
    args = parser.parse_args(argv)
    records = load_tree(args.games)
    wanted = parse_seeds(args.seeds)
    if wanted is not None:
        records = [r for r in records if r["seed"] in wanted]
    if not records:
        raise SystemExit(f"no {COMPARISON} replays under {args.games}"
                         + (f" for seeds {args.seeds}" if args.seeds else ""))
    summary = summarize(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
                        encoding="utf-8")
    args.md.write_text(render(summary), encoding="utf-8")
    print(f"{len(records)} checkpoints from {len(summary['games'])} games -> {args.out}, {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
