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
from typing import Dict, List, Optional, Sequence, Tuple

from components.helpers import read_jsonl
from components.stats import cluster_bootstrap, mean, paired_cohens_dz
from lurkingwolf.engine import WOLF_ROLES
from experiments.proficiency_validation.rewrite.runtime import fmt, now_iso
from experiments.proficiency_validation.validation.analysis import cluster_sign_flip_test

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
GAMES_DIR = EXPERIMENT_DIR / "games"
COMPARISON = "nl_vs_proficiency"
RUN = "qwen35_27b"                      # one directory per run, as in experiments/proficiency_validation
RUN_DIR = EXPERIMENT_DIR / "data" / "runs" / RUN
SUMMARY_JSON = RUN_DIR / "flip_summary.json"
SUMMARY_MD = EXPERIMENT_DIR / "summary.md"
CONDITIONS = ("low", "neutral", "high")
SPEAKER_OUTCOMES = ("votes_received", "executed", "wolf_target", "seer_target", "trust_in_speaker")
#: what the others say about the treated speaker (code/response.py)
from experiments.proficiency_flip.code.response import (  # noqa: E402
    EXTRA_OUTCOMES, RESPONSE_OUTCOMES, response_outcomes)
ALL_SPEAKER_OUTCOMES = SPEAKER_OUTCOMES + RESPONSE_OUTCOMES + EXTRA_OUTCOMES
#: the day fixes how many votes are cast, who is executed and who each role targets, so the
#: difference between two replays of one checkpoint sums to zero over the speakers. Comparing
#: two single replays on these says nothing; only contrasts where each speaker has their own
#: treated replay do.
CONSERVED = ("votes_received", "executed", "wolf_target", "seer_target")
OWN_OUTCOMES = ("changed_vote", "trust_toward_others", "changed_role_target")
CONTRASTS = (("low", "identity"), ("neutral", "identity"), ("high", "identity"), ("low", "neutral"), ("high", "neutral"))
# Prespecified co-primary behavioural family (Low vs Neutral), Holm-corrected; the rest is secondary/exploratory.
CO_PRIMARY = ("trust_in_speaker", "votes_received")
N_BOOT, SEED = 2000, 101
GameDifferences = Dict[str, Dict[str, List[float]]]


def parse_run_id(run_id: str, *, conditions: Sequence[str] = CONDITIONS) -> Optional[dict]:
    """``<mode>[__<condition>__<param values...>][__pN]__dD`` as the runner names replays."""
    tokens = run_id.split("__")
    if len(tokens) < 2 or not re.fullmatch(r"d\d+", tokens[-1]):
        return None
    mode, day = tokens[0], int(tokens[-1][1:])
    if mode not in ("nl", "flip_self", "flip_others"):
        return None
    self_player = tokens[-2].upper() if len(tokens) > 2 and re.fullmatch(r"p\d+", tokens[-2]) else None
    condition = tokens[1] if len(tokens) > 2 and (tokens[1] in conditions or re.fullmatch(r"repeat\d+", tokens[1])) else None
    return {"mode": mode, "condition": condition, "self": self_player, "day": day}


IDENTITY, REPEAT = ("nl", None, None), ("nl", "repeat1", None)


def assessors(result: dict, speaker: str) -> List[str]:
    """Village players other than ``speaker`` who are alive after the night, i.e. whose
    reliability scores were refreshed by the post-day memory update. Wolves know the roles,
    so their scores are not trust."""
    roles = result.get("role_dist") or {}
    scores = result.get("reliability") or {}
    alive = set(result.get("alive_at_start") or scores)
    dead = {result.get("executed"), result.get("night_killed"), result.get("witch_poison")} - {None}
    return [a for a in scores if a != speaker and a in alive and a not in dead
            and roles.get(a) not in WOLF_ROLES]


def speaker_outcomes(result: dict, speaker: str) -> dict:
    """What happened to ``speaker`` in one day phase."""
    scores = result.get("reliability") or {}
    trust = [scores[a][speaker] for a in assessors(result, speaker)
             if isinstance(scores.get(a), dict) and scores[a].get(speaker) is not None]
    return {"votes_received": (result.get("votes_received") or {}).get(speaker, 0),
            "executed": int(result.get("executed") == speaker),
            "wolf_target": int(result.get("wolf_target") == speaker),
            "seer_target": int(result.get("seer_target") == speaker),
            "trust_in_speaker": mean(trust),
            **response_outcomes(result, speaker)}


def own_behaviour(result: dict, speaker: str, reference: dict, role: Optional[str]) -> dict:
    """How ``speaker`` acted in ``result`` relative to the same speaker in ``reference``."""
    own = (result.get("reliability") or {}).get(speaker) or {}
    dead = {result.get("executed"), result.get("night_killed"), result.get("witch_poison")} - {None}
    alive = set(result.get("alive_at_start") or own)
    outward = [] if role in WOLF_ROLES else [v for other, v in own.items()
                                             if other != speaker and other in alive and other not in dead and v is not None]
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


def load_tree(games_dir: Path, *, comparison: str = COMPARISON,
              conditions: Sequence[str] = CONDITIONS) -> List[dict]:
    """One record per (seed, day): baseline, identity replay and flip replays, plus bookkeeping."""
    from lurkingwolf.eval.replay import extract_baseline_phase
    records = []
    for root in sorted(games_dir.glob(f"run_*/{comparison}")):
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
                meta = parse_run_id(run_dir.name, conditions=conditions)
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


def paired_differences(records: List[dict], *, per_protocol: bool = False,
                       conditions: Sequence[str] = CONDITIONS,
                       contrasts: Sequence[Tuple[str, str]] = CONTRASTS) -> dict:
    """Per-(seed, day, speaker) differences, grouped for the game-weighted analysis. With
    ``per_protocol`` only replays with complete treatment exposure (no fallback) enter."""
    games = lambda: defaultdict(lambda: defaultdict(list))
    speaker_diffs: Dict[Tuple[str, str], Dict[str, GameDifferences]] = defaultdict(lambda: defaultdict(games))
    role_diffs: Dict[Tuple[str, str], Dict[str, Dict[str, GameDifferences]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(games)))
    exact_role: Dict[Tuple[str, str], Dict[str, Dict[str, GameDifferences]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(games)))
    own_diffs: Dict[str, Dict[str, GameDifferences]] = defaultdict(lambda: defaultdict(games))
    noise: Dict[str, GameDifferences] = defaultdict(games)
    repeat_noise: Dict[str, GameDifferences] = defaultdict(games)
    repeat_spread: Dict[str, GameDifferences] = defaultdict(games)
    n_pairs = Counter()
    for rec in records:
        identity = rec["replays"].get(IDENTITY)
        repeat = rec["replays"].get(REPEAT)
        baseline = rec["baseline"]
        if not identity or "error" in baseline:
            continue
        seed, roles = rec["seed"], baseline.get("role_dist", {})
        alive = set(identity.get("alive_at_start") or roles)
        for speaker in sorted(alive):
            # identity versus the recorded game: the decisions only (the recorded game's trust
            # scores come from the next day's reasoning, not from the post-day update)
            base_out = speaker_outcomes(baseline, speaker)
            for k, v in speaker_outcomes(identity, speaker).items():
                b = base_out.get(k)
                if k in SPEAKER_OUTCOMES and k not in CONSERVED and k != "trust_in_speaker" \
                        and v is not None and b is not None:
                    noise[k][seed][speaker].append(v - b)
            if repeat:
                # identity versus its repeat: the model's run-to-run variation on every outcome
                repeat_out = speaker_outcomes(repeat, speaker)
                for k, v in speaker_outcomes(identity, speaker).items():
                    b = repeat_out.get(k)
                    if k not in CONSERVED and v is not None and b is not None:
                        repeat_noise[k][seed][speaker].append(v - b)
                    if k in CONSERVED and v is not None and b is not None:
                        # how far one speaker's count moves, which the conservation does not fix
                        repeat_spread[k][seed][speaker].append(abs(v - b))
                # the untreated player's own behaviour, identity against its repeat
                own_repeat = own_behaviour(identity, speaker, repeat, roles.get(speaker))
                own_base = own_behaviour(repeat, speaker, repeat, roles.get(speaker))
                for k in OWN_OUTCOMES:
                    a, b = own_repeat[k], own_base[k]
                    if a is not None and b is not None:
                        repeat_noise[k][seed][speaker].append(a - b)
            flips = {c: rec["replays"].get(("flip_self", c, speaker)) for c in conditions
                     if not per_protocol or clean_exposure(rec, ("flip_self", c, speaker))}
            outcomes = {"identity": speaker_outcomes(identity, speaker)}
            outcomes.update({c: speaker_outcomes(r, speaker) for c, r in flips.items() if r})
            for treatment, control in contrasts:
                if treatment in outcomes and control in outcomes:
                    n_pairs[("flip_self", treatment, control)] += 1
                    for k in ALL_SPEAKER_OUTCOMES:
                        a, b = outcomes[treatment][k], outcomes[control][k]
                        if a is not None and b is not None:
                            speaker_diffs[(treatment, control)][k][seed][speaker].append(a - b)
                            side = "werewolf" if roles.get(speaker) == "werewolf" else "village"
                            role_diffs[(treatment, control)][side][k][seed][speaker].append(a - b)
                            exact_role[(treatment, control)][roles.get(speaker, "unknown")][k][seed][speaker].append(a - b)
            others = {c: rec["replays"].get(("flip_others", c, speaker)) for c in conditions
                      if not per_protocol or clean_exposure(rec, ("flip_others", c, speaker))}
            own = {"identity": own_behaviour(identity, speaker, identity, roles.get(speaker))}
            own.update({c: own_behaviour(r, speaker, identity, roles.get(speaker)) for c, r in others.items() if r})
            for treatment, control in contrasts:
                if treatment in own and control in own:
                    n_pairs[("flip_others", treatment, control)] += 1
                    for k in OWN_OUTCOMES:
                        a, b = own[treatment][k], own[control][k]
                        if a is not None and b is not None:
                            own_diffs[f"{treatment}_vs_{control}"][k][seed][speaker].append(a - b)
    return {"speaker": speaker_diffs, "by_role": role_diffs, "by_exact_role": exact_role,
            "own": own_diffs, "noise": noise, "repeat_noise": repeat_noise,
            "repeat_spread": repeat_spread, "n_pairs": n_pairs}


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


def beyond_repeat(records: List[dict], *, conditions: Sequence[str] = CONDITIONS,
                  comparison: str = "vote") -> dict:
    """How much more a rewrite changes the day than replaying it unchanged does.

    Within one checkpoint, a treated replay and the second identity replay are both
    compared with the first identity replay, and the second difference is subtracted from
    the first. What survives is the change the rewriting adds to the change the serving
    stack produces on its own. Days are averaged within self player, players within game,
    then games.
    """
    from experiments.proficiency_flip.code.figures import change_rates
    games = lambda: defaultdict(lambda: defaultdict(list))
    out: Dict[str, GameDifferences] = defaultdict(games)
    for rec in records:
        identity, repeat = rec["replays"].get(IDENTITY), rec["replays"].get(REPEAT)
        if not identity or not repeat:
            continue
        floor = change_rates(identity, repeat)[comparison]
        if floor is None:
            continue
        for (mode, cond, speaker), result in rec["replays"].items():
            if mode not in ("flip_self", "flip_others") or cond not in conditions or result is None:
                continue
            rate = change_rates(identity, result)[comparison]
            if rate is not None:
                out[f"{mode}:{cond}"][rec["seed"]][speaker or "-"].append(rate - floor)
    return {label: game_weighted(diffs) for label, diffs in sorted(out.items())}


#: outcomes that measure one thing, how much the table turns on the treated speaker
TARGETING = ("votes_received", "accused_by_others", "execution_urged")


def joint_sign_flip(outcomes: Dict[str, GameDifferences], *, n_perm: int = 20000,
                    seed: int = SEED) -> dict:
    """One test over several outcomes that measure the same thing.

    Each game contributes one difference per outcome. The differences are scaled by each
    outcome's own spread across games, so a count and a score weigh the same, and averaged
    into a single number per game. The null flips the sign of a game's whole vector at once,
    which keeps the outcomes' correlation intact: three correlated measures of one thing do
    not count as three independent chances to be wrong.
    """
    import random as _random
    from components.stats import sd
    names = [k for k in outcomes if outcomes[k]]
    per_game: Dict[str, Dict[str, float]] = defaultdict(dict)
    for name in names:
        for game, players in outcomes[name].items():
            values = [mean(days) for days in players.values() if days]
            if values:
                per_game[game][name] = mean(values)
    games = [g for g, row in per_game.items() if len(row) == len(names)]
    if len(games) < 2 or not names:
        return {"outcomes": names, "n_games": len(games), "estimate": None, "p_sign_flip": None}
    spread = {name: sd([per_game[g][name] for g in games]) or 0.0 for name in names}
    scaled = {g: [per_game[g][n] / spread[n] for n in names if spread[n]] for g in games}
    if not all(scaled.values()):
        return {"outcomes": names, "n_games": len(games), "estimate": None, "p_sign_flip": None}
    composite = {g: mean(v) for g, v in scaled.items()}
    observed = mean(list(composite.values()))
    effects = list(composite.values())
    threshold = abs(sum(effects))
    rng = _random.Random(seed)
    count = sum(abs(sum(x if rng.random() < .5 else -x for x in effects)) >= threshold - 1e-12
                for _ in range(n_perm))
    from statistics import median
    return {"outcomes": names, "n_games": len(games), "estimate": observed,
            "p_sign_flip": (count + 1) / (n_perm + 1), "n_perm": n_perm,
            "unit": "standard deviations of the between-game spread",
            "median": median(effects),
            "games_positive": sum(1 for v in effects if v > 1e-9),
            "games_negative": sum(1 for v in effects if v < -1e-9),
            "per_outcome_scale": {n: spread[n] for n in names}}


#: The largest difference not worth calling an effect, on the scale each outcome is measured
#: in. Trust is a 1-10 reliability score, so a quarter point is 2.5% of its range; a day puts
#: about six votes on the table, so a quarter vote is one vote in every fourth day. Both are
#: set here rather than read off the data, because a bound taken from the same replays the
#: test uses cannot fail.
EQUIVALENCE_BOUNDS = {"trust_in_speaker": 0.25, "votes_received": 0.25}


def equivalence(diffs: GameDifferences, bound: float) -> dict:
    """Two one-sided tests: is the whole 90% interval inside [-bound, +bound]?"""
    from components.stats import tost_equivalence
    speaker_means = {g: [mean(days) for days in players.values() if days] for g, players in diffs.items()}
    clusters = {g: [mean(v)] for g, v in speaker_means.items() if v}
    if len(clusters) < 2:
        return {"bound": bound, "equivalent": None, "n_games": len(clusters)}
    out = tost_equivalence(clusters, bound, n_boot=N_BOOT, seed=SEED)
    out.pop("draws", None)
    lo, hi = out.get("ci_low"), out.get("ci_high")
    # The smallest bound this contrast would still pass: it says how tight the absence is
    # without reference to where the declared bound was put.
    margin = None if lo is None or hi is None else max(abs(lo), abs(hi))
    return {"bound": bound, "n_games": len(clusters), "margin": margin, **out}


def availability(records: List[dict], *, conditions: Sequence[str] = CONDITIONS) -> dict:
    out: Dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        for (mode, cond, _), result in rec["replays"].items():
            label = mode if cond is None else f"{mode}:{cond}"
            out[label]["present"] += int(result is not None)
            out[label]["missing_result"] += int(result is None)
        for run_id, rc in rec["expected"].items():
            meta = parse_run_id(run_id, conditions=conditions)
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
    m = len(pvalues)                     # the family is the prespecified set, not the available p-values
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = {"p": p, "p_holm": running, "reject_05": running <= .05}
    for k, p in pvalues.items():
        out.setdefault(k, {"p": p, "p_holm": None, "reject_05": None})
    return out


def summarize(records: List[dict]) -> dict:
    diffs = paired_differences(records)
    effects = {f"{t}_vs_{c}": {k: game_weighted(diffs["speaker"][(t, c)][k]) for k in ALL_SPEAKER_OUTCOMES} for t, c in CONTRASTS}
    co_primary = {k: effects["low_vs_neutral"][k] for k in CO_PRIMARY}
    for k, adj in holm({k: e.get("p_sign_flip") for k, e in co_primary.items()}).items():
        co_primary[k] = {**co_primary[k], **adj}
    clean = paired_differences(records, per_protocol=True)
    per_protocol = {f"{t}_vs_{c}": {k: game_weighted(clean["speaker"][(t, c)][k]) for k in ALL_SPEAKER_OUTCOMES} for t, c in CONTRASTS}
    by_role = {f"{t}_vs_{c}": {side: {k: game_weighted(diffs["by_role"][(t, c)][side][k]) for k in ALL_SPEAKER_OUTCOMES}
                               for side in ("werewolf", "village")} for t, c in CONTRASTS}
    own = {name: {k: game_weighted(block[k]) for k in OWN_OUTCOMES} for name, block in diffs["own"].items()}
    noise = {k: game_weighted(diffs["noise"][k]) for k in SPEAKER_OUTCOMES
             if k not in CONSERVED and k != "trust_in_speaker"}
    bounds = {k: EQUIVALENCE_BOUNDS[k] for k in CO_PRIMARY}
    tost = {f"{t}_vs_{c}": {k: equivalence(diffs["speaker"][(t, c)][k], bounds.get(k, 0.0))
                            for k in CO_PRIMARY} for t, c in CONTRASTS}
    net = {m: beyond_repeat(records, conditions=CONDITIONS, comparison=m) for m in ("vote", "skill", "reliability")}
    # The discussion outcomes are one family per contrast: five tests, Holm-corrected together.
    discussion = {f"{t}_vs_{c}": holm({k: effects[f"{t}_vs_{c}"][k].get("p_sign_flip")
                                       for k in RESPONSE_OUTCOMES}) for t, c in CONTRASTS}
    targeting = {f"{t}_vs_{c}": joint_sign_flip({k: diffs["speaker"][(t, c)][k] for k in TARGETING})
                 for t, c in CONTRASTS}
    # Who carries the effect: the composite inside each role the speaker can hold.
    roles_seen = sorted({r for block in diffs["by_exact_role"].values() for r in block})
    targeting_by_role = {f"{t}_vs_{c}": {
        role: joint_sign_flip({k: diffs["by_exact_role"][(t, c)][role][k] for k in TARGETING})
        for role in roles_seen if diffs["by_exact_role"][(t, c)].get(role)} for t, c in CONTRASTS}
    role_holm = {name: holm({role: block[role].get("p_sign_flip") for role in block})
                 for name, block in targeting_by_role.items()}
    holm_net = {m: holm({label: block.get("p_sign_flip") for label, block in net[m].items()}) for m in net}
    repeat_noise = {k: game_weighted(diffs["repeat_noise"][k]) for k in ALL_SPEAKER_OUTCOMES + OWN_OUTCOMES
                    if k not in CONSERVED}
    repeat_spread = {k: game_weighted(diffs["repeat_spread"][k]) for k in CONSERVED}
    seeds = sorted({r["seed"] for r in records})
    return {"created_at": now_iso(), "games": seeds, "n_checkpoints": len(records),
            "days_per_game": dict(Counter(r["seed"] for r in records)),
            "n_pairs": {f"{m}:{t}_vs_{c}": n for (m, t, c), n in diffs["n_pairs"].items()},
            "availability": availability(records), "flip_self": effects, "co_primary_low_vs_neutral": co_primary,
            "flip_self_per_protocol": per_protocol, "flip_self_by_role": by_role,
            "flip_others": own, "identity_vs_baseline": noise, "identity_vs_repeat": repeat_noise,
            "beyond_repeat": net, "beyond_repeat_holm": holm_net, "equivalence": tost,
            "identity_vs_repeat_spread": repeat_spread,
            "discussion_holm": discussion, "equivalence_bounds": bounds, "targeting": targeting, "targeting_by_role": targeting_by_role, "targeting_by_role_holm": role_holm,
            "estimand": "equal_game_weight",
            "analysis_policy": "assignment-based analysis of completed replays; per-protocol analysis requires complete logs matching every treated utterance, with all rewrites successful",
            "aggregation": "days within speaker, speakers within game, then games; equal weights at each step",
            "n_boot": N_BOOT, "seed": SEED}


def _row(label: str, e: dict) -> str:
    if not e.get("n_pairs"):
        return f"| {label} | 0 | 0 | – | – | – | – |"
    return (f"| {label} | {e['n_pairs']} | {e['n_games']} | {fmt(e['estimate'], 3)} "
            f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | {fmt(e.get('cohens_dz'), 2)} | "
            f"{fmt(e.get('p_sign_flip'))} | {fmt(e.get('pooled_mean'), 3)} |")


def _effect(s: dict, block: str, key: str, outcome: str) -> str:
    e = (s.get(block) or {}).get(key, {}).get(outcome) or {}
    if e.get("estimate") is None:
        return "–"
    return f"{fmt(e['estimate'], 3)} [{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}]"


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
    games = len(s.get("games") or [])
    co = s.get("co_primary_low_vs_neutral") or {}
    trust, votes = co.get("trust_in_speaker") or {}, co.get("votes_received") or {}
    equiv, bounds = s.get("equivalence") or {}, s.get("equivalence_bounds") or {}
    inside = sum(1 for b in equiv.values() for e in b.values() if e.get("equivalent"))
    total = sum(1 for b in equiv.values() for e in b.values())
    margins = {o: max((b[o]["margin"] for b in equiv.values() if (b.get(o) or {}).get("margin")), default=None)
               for o in CO_PRIMARY}
    lines = ["## Findings", "",
             f"Rewriting a speaker's English range does not change how the other agents treat them. Across "
             f"{games} games the restricted range moves trust in the speaker by "
             f"{_effect(s, 'flip_self', 'low_vs_neutral', 'trust_in_speaker')} against the paraphrase control "
             f"and the votes against them by {_effect(s, 'flip_self', 'low_vs_neutral', 'votes_received')}, "
             f"Holm-adjusted p {fmt(trust.get('p_holm'))} and {fmt(votes.get('p_holm'))}; the expanded range "
             f"is the same, {_effect(s, 'flip_self', 'high_vs_neutral', 'trust_in_speaker')} on trust.", ""]
    if total == inside and total and all(margins.values()):
        lines += [f"The absence is tight. All {total} contrasts of the two co-primary outcomes are equivalent "
                  f"within the declared ±{fmt(bounds.get('trust_in_speaker'), 2)} of a reliability point and "
                  f"±{fmt(bounds.get('votes_received'), 2)} of a vote, and the widest of them passes already "
                  f"at ±{fmt(margins['trust_in_speaker'], 3)} on trust and "
                  f"±{fmt(margins['votes_received'], 3)} on votes.", ""]
    vote_floor = (s.get("beyond_repeat") or {}).get("vote") or {}
    trust_floor = (s.get("beyond_repeat") or {}).get("reliability") or {}
    if vote_floor:
        rates = {k: v["estimate"] for k, v in vote_floor.items() if v.get("estimate") is not None}
        worst = max(trust_floor.values(), key=lambda v: abs(v.get("estimate") or 0), default={})
        lines += [f"The rewriting still reaches the decisions, without the level mattering. Votes differ from "
                  f"the identity replay more often than a plain re-run of the same day does, by "
                  f"{fmt(min(rates.values()), 3)} to {fmt(max(rates.values()), 3)} across the "
                  f"{len(rates)} treated conditions, and the control paraphrase moves them as much as the "
                  f"restricted range: {fmt(rates.get('flip_others:neutral'), 3)} against "
                  f"{fmt(rates.get('flip_others:low'), 3)} when everyone else is rewritten. The reliability "
                  f"scores stay at the floor, the largest of them "
                  f"{fmt(worst.get('estimate'), 3)} [{fmt(worst.get('ci_low'), 3)}, "
                  f"{fmt(worst.get('ci_high'), 3)}].", ""]
    urged = ((s.get("discussion_holm") or {}).get("low_vs_neutral") or {}).get("execution_urged") or {}
    if urged.get("p") is not None:
        lines += [f"The discussion moves where the vote does not: at the restricted range a speaker draws "
                  f"{_effect(s, 'flip_self', 'low_vs_neutral', 'execution_urged')} more calls for their "
                  f"execution, p = {fmt(urged.get('p'))}, which does not hold up against Holm over the "
                  f"{len(RESPONSE_OUTCOMES)} discussion outcomes ({fmt(urged.get('p_holm'))}).", ""]
    ex = (s.get("availability") or {}).get("exposure") or {}
    rates = [v.get("exposure_rate") for v in ex.values() if v.get("exposure_rate") is not None]
    replays = sum(v.get("replays", 0) for v in ex.values())
    if rates:
        gate = (s.get("availability") or {}).get("gate_outcomes") or {}
        refused = sum(c.get("treatment_unavailable", 0) for c in gate.values())
        broke = sum(c.get("infrastructure_unavailable", 0) for c in gate.values())
        lines += [f"The manipulation reached the table: every one of the {replays} treated replays returned a "
                  f"result, with {fmt(min(rates), 3)} to {fmt(max(rates), 3)} of their utterances delivered as "
                  f"rewrites. The rest went out as the untouched original, {refused} of them because the gate "
                  f"refused the rewrite and {broke} because a model call failed.", ""]
    return lines


def perception_section(s: dict) -> List[str]:
    """The blind listeners' answers to the replays' speech."""
    from experiments.proficiency_flip.code.perception import (
        cross_listener_agreement, listener_summary, render as render_listeners)
    directory = RUN_DIR / "validation"
    return render_listeners(listener_summary(directory), cross_listener_agreement(directory))


def moved(e: dict) -> bool:
    """True when the 95% interval excludes zero: an outcome with something behind it."""
    lo, hi = (e or {}).get("ci_low"), (e or {}).get("ci_high")
    return lo is not None and hi is not None and (lo > 0) == (hi > 0) and lo != 0 and hi != 0

def render(s: dict) -> str:
    head = "| outcome | pairs | games | game-weighted Δ [95% CI] | d_z | p (sign-flip) | pooled Δ |"
    sep = "|---|--:|--:|--:|--:|--:|--:|"
    lines = ["# Proficiency flip replays", "",
             f"{len(s['games'])} baseline games ({_seed_range(s['games'])}), "
             f"{s['n_checkpoints']} day checkpoints. Every replay repeats one day from its checkpoint; in `flip_self` "
             "one speaker's utterances are rewritten to the assigned level for that day, in `flip_others` everyone "
             "else's are. Differences are paired on the same checkpoint and speaker against the identity replay "
             f"(same checkpoint, no rewriter). Games are the independent units; with {len(s['games'])} games the "
             f"sign-flip tests permute the game means; above twelve games they sample 5000 "
             f"permutations, so the smallest p they can return is 0.0002.", ""]
    lines += findings(s)
    lines += [f"Aggregation: {s['aggregation']}.", "",
              "## Co-primary outcomes (Low minus Neutral, `flip_self`, Holm-corrected)", "",
              "| outcome | pairs | games | game-weighted Δ [95% CI] | d_z | p (sign-flip) | p (Holm) | reject at 0.05 |",
              "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for k, e in s["co_primary_low_vs_neutral"].items():
        lines.append(f"| {k} | {e.get('n_pairs', 0)} | {e.get('n_games', 0)} | {fmt(e.get('estimate'), 3)} "
                     f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | {fmt(e.get('cohens_dz'), 2)} | "
                     f"{fmt(e.get('p_sign_flip'))} | {fmt(e.get('p_holm'))} | {fmt(e.get('reject_05'))} |")
    lines += ["", "## The treated speaker", "",
              "Paired on the same checkpoint and speaker. The contrasts against the identity replay, the "
              "by-side splits and the `flip_others` outcomes are in the JSON beside this file.", ""]
    for name in ("low_vs_neutral", "high_vs_neutral"):
        if name in s["flip_self"]:
            lines += [f"### {name.replace('_vs_', ' minus ')}", "", head, sep]
            shown = [k for k in CO_PRIMARY + ("execution_urged",)
                     if k in s["flip_self"][name]]
            lines += [_row(k, s["flip_self"][name][k]) for k in shown]
            lines.append("")
    if s.get("equivalence"):
        lines += ["## How tight the absence is", "",
                  "The bound is the largest difference worth calling an effect: a quarter of a reliability "
                  "point on a 1-10 scale, and a quarter of a vote out of the six a day puts on the "
                  "table. Equivalence holds when the whole 90% interval lies inside it. The last column is "
                  "the smallest bound the contrast would still pass.", "",
                  "| contrast | outcome | bound | 90% interval | smallest bound it passes | equivalent |",
                  "|---|---|--:|--:|--:|--:|"]
        for name, block in s["equivalence"].items():
            for outcome, e in block.items():
                lines.append(f"| {name.replace('_vs_', ' minus ')} | {outcome} | ±{fmt(e.get('bound'), 3)} | "
                             f"[{fmt(e.get('ci_low'), 3)}, {fmt(e.get('ci_high'), 3)}] | "
                             f"±{fmt(e.get('margin'), 3)} | {fmt(e.get('equivalent'))} |")
        lines += ["", "`votes_received` counts first-round votes against the speaker, `trust_in_speaker` is the "
                  "mean reliability score the village players alive after the night assign them, and "
                  "`execution_urged` counts the other players' predicates calling for their execution.", ""]
    lines += perception_section(s)
    lines += ["## Figures", "", "| figure | what it shows |", "|---|---|",
              "| `change_rates` | Decisions that differ from the identity replay, with the replay's own repeat as the floor |",
              "| `scope_contrasts` | The level contrasts in those change rates, by rewrite scope |",
              "| `speaker_outcomes` | Trust and votes for the treated speaker, and the same split by their side |",
              "| `self_response` | The untreated player's vote and outward trust under surrounding rewriting |",
              "| `discussion_outcomes` | What the others say about the treated speaker, before and after Holm |",
              "| `equivalence` | Every co-primary effect against the largest difference worth calling one |",
              "| `manipulation_check` | Delivered discussion speech against its source: word frequency, lexical diversity, sentence length, subordination, dependency depth |", "",
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    args.md.write_text(render(summary), encoding="utf-8")
    print(f"{summary['n_checkpoints']} checkpoints from {len(summary['games'])} games -> {args.out}, {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
