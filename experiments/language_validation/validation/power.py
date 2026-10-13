"""Per-measure sample-size approximations using pilot game effects and user-supplied margins."""
from __future__ import annotations
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Optional

from experiments.language_validation.rewrite.runtime import PRIMARY_RANGE_METRICS, dump_json, fmt

_Z = NormalDist().inv_cdf


def _probability(value, name):
    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1")


def sign_flip_minimum_games(alpha: float = .05) -> int:
    """Smallest n with 2 ** (1 - n) <= alpha: the exact two-sided sign-flip test cannot reject below it."""
    _probability(alpha, "alpha")
    n = 2
    while 2.0 ** (1 - n) > alpha:
        n += 1
    return n


def games_for_difference(mean_effect: Optional[float], sd_games: Optional[float], *, alpha: float = .05,
                         power: float = .8) -> Optional[int]:
    """Games needed for a two-sided paired test of a game-level mean effect (normal approximation)."""
    _probability(alpha, "alpha")
    _probability(power, "power")
    if mean_effect in (None, 0) or sd_games in (None, 0):
        return None
    n = ((_Z(1 - alpha / 2) + _Z(power)) * sd_games / abs(mean_effect)) ** 2
    return max(int(math.ceil(n)), sign_flip_minimum_games(alpha))


def games_for_equivalence(sd_games: Optional[float], margin: Optional[float], *, alpha: float = .05,
                          power: float = .8, true_effect: float = 0.0) -> Optional[int]:
    """Games needed for two one-sided tests to show |effect| < margin when the true effect is ``true_effect``."""
    _probability(alpha, "alpha")
    _probability(power, "power")
    if sd_games in (None, 0) or margin in (None, 0) or abs(true_effect) >= margin:
        return None
    n = ((_Z(1 - alpha) + _Z(1 - (1 - power) / 2)) * sd_games / (margin - abs(true_effect))) ** 2
    return max(2, int(math.ceil(n)))


def plan(report: dict, *, alpha: float = .05, equivalence_margins: Optional[Dict[str, float]] = None,
         listener_margin: Optional[float] = None) -> dict:
    """Required games per primary metric and for the listener signed preference, from a pilot report."""
    analysis = report.get("analysis", report)
    primary = list(PRIMARY_RANGE_METRICS)
    rows = []
    for metric in primary:
        m = (analysis.get("range", {}).get("low_vs_neutral", {}).get("metrics", {}) or {}).get(metric)
        if not m:
            continue
        rows.append({"quantity": f"A − B: {metric}", "kind": "difference", "pilot_games": m.get("n_games"),
                     "pilot_mean": m.get("estimate"), "sd_games": m.get("sd_diff"), "game_dz": m.get("cohens_dz"),
                     "games_80": games_for_difference(m.get("estimate"), m.get("sd_diff"), alpha=alpha, power=.8),
                     "games_90": games_for_difference(m.get("estimate"), m.get("sd_diff"), alpha=alpha, power=.9)})
        c = (analysis.get("range", {}).get("neutral_vs_original", {}).get("metrics", {}) or {}).get(metric)
        margin = (equivalence_margins or {}).get(metric)
        if c:
            rows.append({"quantity": f"B − Original: {metric} equivalence", "kind": "equivalence",
                         "pilot_games": c.get("n_games"), "pilot_mean": c.get("estimate"), "sd_games": c.get("sd_diff"),
                         "margin": margin,
                         "games_80": games_for_equivalence(c.get("sd_diff"), margin, alpha=alpha, power=.8, true_effect=c.get("estimate") or 0.),
                         "games_90": games_for_equivalence(c.get("sd_diff"), margin, alpha=alpha, power=.9, true_effect=c.get("estimate") or 0.)})
    for key, label in (("listener", "range listener"), ("perceived_proficiency", "proficiency listener")):
        block = (analysis.get(key) or {})
        low = (block.get("low:neutral") or {}).get("listeners", {}).get("all_models")
        if low:
            rows.append({"quantity": f"A − B: {label} signed preference", "kind": "difference",
                         "pilot_games": low.get("n_games"), "pilot_mean": low.get("signed_preference"), "sd_games": low.get("sd_games"),
                         "games_80": games_for_difference(low.get("signed_preference"), low.get("sd_games"), alpha=alpha, power=.8),
                         "games_90": games_for_difference(low.get("signed_preference"), low.get("sd_games"), alpha=alpha, power=.9)})
        control = (block.get("neutral:original") or {}).get("listeners", {}).get("all_models")
        if control:
            rows.append({"quantity": f"B − Original: {label} equivalence", "kind": "equivalence",
                         "pilot_games": control.get("n_games"), "pilot_mean": control.get("signed_preference"),
                         "sd_games": control.get("sd_games"), "margin": listener_margin,
                         "games_80": games_for_equivalence(control.get("sd_games"), listener_margin, alpha=alpha, power=.8, true_effect=control.get("signed_preference") or 0.),
                         "games_90": games_for_equivalence(control.get("sd_games"), listener_margin, alpha=alpha, power=.9, true_effect=control.get("signed_preference") or 0.)})
    return {"scope": "per_measure", "method": "paired_normal_approximation", "alpha": alpha,
            "sign_flip_minimum_games": sign_flip_minimum_games(alpha), "rows": rows,
            "note": "Each row estimates the games needed for one measure using the pilot mean and between-game SD. "
                    "Difference rows test a zero mean; equivalence rows use supplied margins. Full-study sample-size "
                    "planning must combine the frozen decision criteria, multiplicity correction and passage attrition."}


def render(result: dict) -> str:
    lines = ["# Per-measure game-count planning", "", result["note"], "",
             f"Exact sign-flip test: at least {result['sign_flip_minimum_games']} games for a two-sided p below {result['alpha']}.", "",
             "| quantity | pilot games | pilot mean | SD across games | margin | games (80% per-measure power) | games (90% per-measure power) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in result["rows"]:
        lines.append(f"| {r['quantity']} | {r.get('pilot_games')} | {fmt(r.get('pilot_mean'))} | {fmt(r.get('sd_games'))} | "
                     f"{fmt(r.get('margin')) if r.get('kind') == 'equivalence' else '–'} | {r.get('games_80') or '–'} | {r.get('games_90') or '–'} |")
    lines += ["", "A dash means the quantity cannot be planned from the pilot: no effect, no spread, or no margin was given."]
    return "\n".join(lines) + "\n"


def add_parser(sub) -> None:
    p = sub.add_parser("power", help="per-measure game-count approximations from a pilot report")
    p.add_argument("--report", type=Path, required=True, help="validation/report.json of a pilot run")
    p.add_argument("--alpha", type=float, default=.05)
    p.add_argument("--equivalence-margin", action="append", default=[], metavar="METRIC=VALUE",
                   help="Neutral-vs-Original equivalence margin for a primary metric (repeatable)")
    p.add_argument("--listener-margin", type=float, default=None, help="listener signed-preference equivalence margin")
    p.add_argument("--out", type=Path, default=None, help="write the plan as JSON (a .md twin is written too)")
    p.set_defaults(func=main)


def main(args) -> int:
    margins = {}
    for item in args.equivalence_margin:
        metric, value = item.split("=")
        margins[metric.strip()] = float(value)
    result = plan(json.loads(Path(args.report).read_text(encoding="utf-8")), alpha=args.alpha,
                  equivalence_margins=margins, listener_margin=args.listener_margin)
    text = render(result)
    if args.out:
        dump_json(args.out, result)
        Path(args.out).with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"wrote {args.out} and {Path(args.out).with_suffix('.md')}")
    print(text)
    return 0
