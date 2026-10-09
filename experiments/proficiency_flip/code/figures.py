"""Figures for the proficiency flip replays.

Every quantity is aggregated the way the report aggregates it: days within self player,
self players within game, then games with equal weight. Intervals are bootstraps over
games and p-values two-sided sign-flip tests over game means, as in ``analyze.py``.

    python -m experiments.proficiency_flip.code.figures
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from components.helpers import read_jsonl
from components.stats import cluster_bootstrap, mean
from experiments.proficiency_flip.code import analyze as fa
from experiments.proficiency_validation.validation.analysis import cluster_sign_flip_test

HERE = fa.EXPERIMENT_DIR
GAMES_DIR = fa.GAMES_DIR
SUMMARY_PATH = fa.SUMMARY_JSON
FIG_DIR = HERE / "figures"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

CONDITIONS = [("low", "Low (A)"), ("neutral", "Neutral (B)"), ("high", "High (C)")]
CONDITION_COLORS = {"low": "#C49A82", "neutral": "#8A9E78", "high": "#6080A0", "original": "#9A9A9A"}
SCOPES = [("flip_self", "flip self: the speaker is rewritten"),
          ("flip_others", "flip others: everyone but the self player is rewritten")]
SCOPE_COLORS = {"flip_self": "#7A9EB3", "flip_others": "#C49A82"}
SIDE_COLORS = {"werewolf": "#A86B6B", "village": "#8A9E78"}
RATE_METRICS = [("skill", "Skill"), ("vote", "Vote"), ("reliability", "Reliability")]
SKILL_SLOTS = ["wolf_target", "seer_target", "guard_target"]
CONTRASTS = [("low_vs_neutral", "Low − Neutral"), ("high_vs_neutral", "High − Neutral"),
             ("low_vs_identity", "Low − Identity"), ("neutral_vs_identity", "Neutral − Identity"),
             ("high_vs_identity", "High − Identity")]
N_BOOT, SEED = fa.N_BOOT, fa.SEED


# ─── Game-level statistics (the report's aggregation) ───────────────────────

def game_level(values: Dict[str, Dict[str, List[float]]], *, test: bool = False) -> dict:
    """``values[game][self player] -> [one value per day]`` -> equal-weight estimate over games."""
    speaker_means = {g: [mean(days) for days in players.values() if days] for g, players in values.items()}
    game_means = {g: mean(v) for g, v in speaker_means.items() if v}
    if not game_means:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_games": 0, "n_obs": 0, "game_means": {}}
    clusters = {g: [m] for g, m in game_means.items()}
    boot = cluster_bootstrap(clusters, n_boot=N_BOOT, seed=SEED)
    out = {"estimate": boot["estimate"],
           "ci_low": boot["ci_low"] if boot["ci_low"] is not None else boot["estimate"],
           "ci_high": boot["ci_high"] if boot["ci_high"] is not None else boot["estimate"],
           "n_games": len(game_means), "n_obs": sum(len(d) for p in values.values() for d in p.values()),
           "game_means": game_means}
    if test:
        out["p_sign_flip"] = cluster_sign_flip_test(clusters, seed=SEED)["p_value"]
    return out


def pct_fmt(x, _pos=None) -> str:
    return f"{100 * x:.0f}%"


def signed_pp_fmt(x, _pos=None) -> str:
    return f"{100 * x:+.0f} pp"


# ─── Change rates between two replays of the same checkpoint ─────────────────

def change_rates(reference: dict, replay: dict) -> dict:
    """Share of skill targets, votes and reliability scores in ``replay`` that differ from
    ``reference`` (the same definitions as ``lurkingwolf.eval.metrics.compute_phase_delta``)."""
    skill = [int(reference.get(slot) != replay.get(slot)) for slot in SKILL_SLOTS
             if not (reference.get(slot) is None and replay.get(slot) is None)]
    ref_votes, rep_votes = reference.get("votes", {}), replay.get("votes", {})
    vote = [int(ref_votes.get(v) != rep_votes.get(v)) for v in sorted(set(ref_votes) | set(rep_votes))]
    rel_changed = []
    ref_rel, rep_rel = reference.get("reliability", {}), replay.get("reliability", {})
    for assessor in sorted(set(ref_rel) | set(rep_rel)):
        a, b = ref_rel.get(assessor) or {}, rep_rel.get(assessor) or {}
        for target in sorted(set(a) | set(b)):
            x, y = a.get(target), b.get(target)
            if x is None and y is None:
                continue
            rel_changed.append(int(x != y))
    return {"skill": mean(skill), "vote": mean(vote), "reliability": mean(rel_changed),
            "execution": int(reference.get("executed") != replay.get("executed"))}


def phase_rows(records: List[dict]) -> List[dict]:
    """One row per flip replay with its change rates against the identity replay of the same
    checkpoint, plus the identity replay against the recorded game and against its repeat."""
    rows = []
    for rec in records:
        identity = rec["replays"].get(fa.IDENTITY)
        if identity is None:
            continue
        base = rec.get("baseline") or {}
        if "votes" in base:
            rows.append({"seed": rec["seed"], "day": rec["day"], "mode": "identity", "condition": "baseline",
                         "self": None, **change_rates(base, identity)})
        repeat = rec["replays"].get(fa.REPEAT)
        if repeat is not None:
            rows.append({"seed": rec["seed"], "day": rec["day"], "mode": "identity", "condition": "repeat",
                         "self": None, **change_rates(identity, repeat)})
        for (mode, cond, self_p), result in rec["replays"].items():
            if mode not in ("flip_self", "flip_others") or result is None:
                continue
            rows.append({"seed": rec["seed"], "day": rec["day"], "mode": mode, "condition": cond,
                         "self": self_p, **change_rates(identity, result)})
    return rows


def rate_stats(rows: List[dict], mode: str, condition: str, metric: str) -> dict:
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["mode"] == mode and r["condition"] == condition and r.get(metric) is not None:
            values[r["seed"]][r["self"] or "-"].append(r[metric])
    return game_level(values)


def paired_rate_stats(rows: List[dict], mode: str, left: str, right: str, metric: str) -> dict:
    """left − right on the same checkpoint and self player, aggregated like the report."""
    by_key: Dict[Tuple[str, int, Optional[str]], Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["mode"] == mode and r.get(metric) is not None:
            by_key[(r["seed"], r["day"], r["self"])][r["condition"]] = r[metric]
    values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for (seed, _day, self_p), conds in by_key.items():
        if left in conds and right in conds:
            values[seed][self_p or "-"].append(conds[left] - conds[right])
    return game_level(values, test=True)


def _ci(s: dict) -> Tuple[float, float]:
    e = s["estimate"]
    return (e - (s["ci_low"] if s["ci_low"] is not None else e), (s["ci_high"] if s["ci_high"] is not None else e) - e)


# ─── Figure 1: decisions changed against the identity replay ─────────────────

def place_label(ax, x, mean_value, text, color, upper, axis_range, fontsize=11):
    inside = abs(mean_value) / axis_range >= 0.12
    y = mean_value * 0.6 if inside else upper + axis_range * 0.025
    txt = ax.text(x, y, text, ha="center", va="center" if inside else "bottom", fontsize=fontsize,
                  fontweight="bold", color="white" if inside else "#222222")
    txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground=color if inside else "white")])


def figure_change_rates(rows: List[dict], out: Path) -> dict:
    references = [("baseline", "identity vs recorded game", ("skill", "vote")),
                  ("repeat", "identity vs its repeat", ("skill", "vote", "reliability"))]
    references = [r for r in references if any(x["condition"] == r[0] for x in rows if x["mode"] == "identity")]
    series = [(c, label, CONDITION_COLORS[c], None) for c, label in CONDITIONS] + \
             [(ref, label, "white", "//") for ref, label, _ in references]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=True, constrained_layout=True)
    x = np.arange(len(RATE_METRICS))
    width = 0.8 / len(series)
    stats_out = {}
    for ax, (mode, title) in zip(axes, SCOPES):
        for idx, (cond, label, color, hatch) in enumerate(series):
            if hatch is None:
                stats = [rate_stats(rows, mode, cond, metric) for metric, _ in RATE_METRICS]
            else:
                allowed = dict((r[0], r[2]) for r in references)[cond]
                stats = [rate_stats(rows, "identity", cond, metric) if metric in allowed else game_level({})
                         for metric, _ in RATE_METRICS]
            for (metric, _), s in zip(RATE_METRICS, stats):
                if s["estimate"] is not None:
                    stats_out[f"{mode if hatch is None else 'identity'}:{cond}:{metric}"] = s
            means = [s["estimate"] if s["estimate"] is not None else 0.0 for s in stats]
            errs = np.array([_ci(s) if s["estimate"] is not None else (0.0, 0.0) for s in stats]).T
            n_games = max((s["n_games"] for s in stats), default=0)
            pos = x + (idx - (len(series) - 1) / 2) * width
            ax.bar(pos, means, width, yerr=errs, capsize=3, color=color, hatch=hatch,
                   edgecolor="#777777" if hatch else "none", alpha=0.85, label=f"{label} (n={n_games})")
            for xi, s in zip(pos, stats):
                if s["estimate"] is not None:
                    place_label(ax, xi, s["estimate"], f"{100 * s['estimate']:.0f}%", color if hatch is None else "#777777",
                                s["ci_high"] if s["ci_high"] is not None else s["estimate"], 0.8, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in RATE_METRICS])
        ax.set_ylim(0, 0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(pct_fmt))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.legend(frameon=False, loc="upper left", fontsize=10)
    axes[0].set_ylabel("Decisions that differ from the identity replay")
    fig.suptitle("Decisions changed by rewriting speech to one English level, 7-player Werewolf", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.03, "Share of skill targets, votes and reliability scores that differ between two replays of the same checkpoint. "
             "Error bars: 95% bootstrap intervals over games.", ha="center", va="top", fontsize=11, color="#444444")
    save(fig, out / "change_rates")
    return stats_out


# ─── Figure 2: paired level contrasts by scope ───────────────────────────────

def figure_scope_contrasts(rows: List[dict], out: Path) -> dict:
    contrasts = [("low", "neutral", "Low − Neutral"), ("high", "neutral", "High − Neutral")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, constrained_layout=True)
    y = np.arange(len(RATE_METRICS))
    bar_h = 0.36
    stats_out = {}
    span = 0.05
    for ax, (left, right, title) in zip(axes, contrasts):
        for idx, (mode, _) in enumerate(SCOPES):
            stats = [paired_rate_stats(rows, mode, left, right, metric) for metric, _ in RATE_METRICS]
            color = SCOPE_COLORS[mode]
            n_games = max((s["n_games"] for s in stats), default=0)
            means = [s["estimate"] if s["estimate"] is not None else 0.0 for s in stats]
            errs = np.array([_ci(s) if s["estimate"] is not None else (0.0, 0.0) for s in stats]).T
            bars = ax.barh(y + (idx - 0.5) * bar_h, means, bar_h, xerr=errs, capsize=3, color=color,
                           alpha=0.88, edgecolor="none", label=f"{mode.replace('_', ' ')} (n={n_games})",
                           error_kw={"color": color, "alpha": 0.7, "linewidth": 1.2})
            for (metric, _), bar, s in zip(RATE_METRICS, bars, stats):
                if s["estimate"] is None:
                    continue
                stats_out[f"{mode}:{left}-{right}:{metric}"] = s
                lo, hi = s["ci_low"], s["ci_high"]
                span = max(span, abs(lo), abs(hi))
                positive = s["estimate"] >= 0
                ax.text((hi + 0.004) if positive else (lo - 0.004), bar.get_y() + bar.get_height() / 2,
                        f"{100 * s['estimate']:+.1f} pp", ha="left" if positive else "right", va="center",
                        fontsize=11, fontweight="bold", color=color)
        ax.axvline(0, color="#555555", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels([label for _, label in RATE_METRICS])
        ax.xaxis.set_major_formatter(FuncFormatter(signed_pp_fmt))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
    for ax in axes:
        ax.set_xlim(-(span * 1.15) - 0.06, span * 1.15 + 0.08)
        ax.set_xlabel("Difference in decisions changed (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=11)
    fig.suptitle("English level contrasts by rewrite scope, 7-player Werewolf", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.07, "Paired on the same checkpoint and self player; each replay's change rate is taken against the "
             "identity replay. Whiskers: 95% bootstrap intervals over games.", ha="center", va="top", fontsize=11, color="#444444")
    save(fig, out / "scope_contrasts")
    return stats_out


# ─── Figure 3: what happens to the treated speaker ───────────────────────────

def forest(ax, entries: List[Tuple[str, dict, str]], title: str, xlabel: str) -> None:
    ys = np.arange(len(entries))[::-1]
    for y_pos, (label, est, color) in zip(ys, entries):
        if not est or est.get("estimate") is None:
            continue
        lo, hi = est.get("ci_low"), est.get("ci_high")
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y_pos, y_pos], color=color, linewidth=2.2, solid_capstyle="round")
        ax.plot([est["estimate"]], [y_pos], marker="o", markersize=8, color=color)
        ax.text(1.01, y_pos, f"{est['estimate']:+.2f}  [{lo:+.2f}, {hi:+.2f}]  pairs={est.get('n_pairs', 0)}",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=10, color="#333333")
    ax.axvline(0, color="#555555", linewidth=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([label for label, _, _ in entries])
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", linestyle="--", alpha=0.25)


def figure_speaker_outcomes(summary: dict, out: Path) -> None:
    fs = summary.get("flip_self", {})
    by_role = summary.get("flip_self_by_role", {})
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    forest(axes[0], [(label, fs.get(key, {}).get("trust_in_speaker"), SCOPE_COLORS["flip_self"])
                     for key, label in CONTRASTS],
           "A. Trust in the speaker", "Δ mean reliability score (1–10) from village players")
    forest(axes[1], [(label, fs.get(key, {}).get("votes_received"), SCOPE_COLORS["flip_self"])
                     for key, label in CONTRASTS],
           "B. Votes against the speaker", "Δ first-round votes received")
    side_entries = []
    for key, label in CONTRASTS[:2]:
        for side in ("werewolf", "village"):
            side_entries.append((f"{label}, {side} speaker",
                                 by_role.get(key, {}).get(side, {}).get("trust_in_speaker"), SIDE_COLORS[side]))
    forest(axes[2], side_entries, "C. Trust by the speaker's side", "Δ mean reliability score (1–10)")
    n_games = summary.get("games")
    n_games = len(n_games) if isinstance(n_games, list) else n_games
    fig.suptitle(f"Speaker-directed rewriting (flip self): outcomes for the treated speaker, {n_games} games",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.04, "Game-weighted paired differences on the same checkpoint and speaker, with 95% bootstrap intervals "
             "over games (experiments/proficiency_flip/data/flip_summary.json).",
             ha="center", va="top", fontsize=11, color="#444444")
    save(fig, out / "speaker_outcomes")


# ─── Figure 4: how the untreated player reacts ───────────────────────────────

def figure_self_response(summary: dict, out: Path) -> None:
    fo = summary.get("flip_others", {})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    forest(axes[0], [(label, fo.get(key, {}).get("changed_vote"), SCOPE_COLORS["flip_others"])
                     for key, label in CONTRASTS],
           "A. Self player votes for a different target", "Δ share of pairs with a changed vote")
    forest(axes[1], [(label, fo.get(key, {}).get("trust_toward_others"), SCOPE_COLORS["flip_others"])
                     for key, label in CONTRASTS],
           "B. Village self player's trust toward the others", "Δ mean reliability score (1–10)")
    fig.suptitle("Surrounding rewriting (flip others): the untreated player's decisions", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.04, "Game-weighted paired differences with 95% bootstrap intervals over games.",
             ha="center", va="top", fontsize=11, color="#444444")
    save(fig, out / "self_response")


# ─── Figure 5: manipulation check on the delivered discussion speech ─────────

def delivered_utterances(games_dir: Path) -> List[dict]:
    """Every discussion utterance a rewriter handled in the completed attempt of each replay:
    the text the listeners heard (the rewrite, or the original after a fallback) paired with
    its source text, keyed by replay and speaker."""
    rows_out = []
    for root in sorted(games_dir.glob(f"run_*/{fa.COMPARISON}")):
        seed = root.parent.name.rsplit("_", 1)[-1]
        for run_dir in sorted(root.glob("replays/day_*/*")):
            meta = fa.parse_run_id(run_dir.name)
            log = run_dir / "rewrites.jsonl"
            if meta is None or meta["condition"] not in fa.CONDITIONS or not log.exists():
                continue
            result_path = run_dir / "phase_result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() and result_path.stat().st_size else None
            rows = read_jsonl(log)
            start = fa.exposure_summary(rows, result, meta)["interrupted_rows"]
            for row in rows[start:]:
                if row.get("phase", "day_speech") != "day_speech":
                    continue
                category, delivered = fa._logged_text(row)
                if category == "unavailable":
                    continue
                rows_out.append({"seed": seed, "day": meta["day"], "mode": meta["mode"], "self": meta["self"],
                                 "speaker": row.get("speaker"), "condition": meta["condition"], "replay": run_dir.name,
                                 "fallback": category == "fallback",
                                 "source": (row.get("original_text") or row.get("original_nl") or "").strip(),
                                 "delivered": delivered})
    return rows_out


MEASURES = [("words", "Words per utterance"), ("words_per_sentence", "Words per sentence"),
            ("subtlex_mean", "Mean word frequency (SUBTLEX Lg10CD)")]


def text_measures(text: str, scorer) -> dict:
    features = scorer(text)
    n_sent = max(1, features.get("n_sentences") or 1)
    n_words = features.get("n_words") or features.get("word_count") or 0
    return {"words": n_words, "words_per_sentence": n_words / n_sent, "subtlex_mean": features.get("subtlex_mean")}


def load_scorer():
    from components.judges.judge_model import _dotenv_values
    from components.metrics.lexical import SubtlexNorms
    from experiments.proficiency_validation.validation.measurements import load_nlp_or_none, text_features
    path = os.getenv("SUBTLEX") or _dotenv_values().get("SUBTLEX")
    if not path:
        raise ValueError("The manipulation check needs SUBTLEX in the environment or .env")
    norms, nlp = SubtlexNorms(Path(os.path.expanduser(path))), load_nlp_or_none("en_core_web_sm", quiet=True)
    return lambda text: text_features(text, nlp=nlp, norms=norms)


def figure_manipulation_check(utterances: List[dict], scorer, out: Path) -> dict:
    """Per condition: the delivered discussion speech against its source, utterances averaged
    within replay and speaker, speakers within game, games with equal weight."""
    cache: Dict[str, dict] = {}

    def measure(text):
        if text not in cache:
            cache[text] = text_measures(text, scorer)
        return cache[text]

    values = {(c, side, field): defaultdict(lambda: defaultdict(list))
              for c, _ in CONDITIONS for side in ("source", "delivered") for field, _ in MEASURES}
    counts = defaultdict(int)
    for u in utterances:
        group = f"{u['replay']}:{u['speaker']}"
        counts[u["condition"]] += 1
        for side in ("source", "delivered"):
            m = measure(u[side])
            for field, _ in MEASURES:
                if m[field] is not None:
                    values[(u["condition"], side, field)][u["seed"]][group].append(m[field])
    stats_out = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    width = 0.36
    for ax, (field, label) in zip(axes, MEASURES):
        xs = np.arange(len(CONDITIONS))
        top = 0.0
        for j, side in enumerate(("source", "delivered")):
            means, errs, highs = [], [], []
            for c, _ in CONDITIONS:
                s = game_level(values[(c, side, field)])
                stats_out[f"{c}:{side}:{field}"] = s
                means.append(s["estimate"] if s["estimate"] is not None else 0.0)
                errs.append(_ci(s) if s["estimate"] is not None else (0.0, 0.0))
                highs.append(s["ci_high"] if s["estimate"] is not None else 0.0)
                if s["estimate"] is not None:
                    top = max(top, s["ci_high"])
            pos = xs + (j - 0.5) * width
            colors = [CONDITION_COLORS["original"] if side == "source" else CONDITION_COLORS[c] for c, _ in CONDITIONS]
            ax.bar(pos, means, width, yerr=np.array(errs).T, capsize=3, color=colors, alpha=0.85, edgecolor="none")
            for xi, m, hi in zip(pos, means, highs):
                ax.text(xi, hi + max(top, 1.0) * 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=9, color="#222222")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lab}\n(n={counts[c]})" for c, lab in CONDITIONS])
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.set_ylim(0, top * 1.25 if top else 1.0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONDITION_COLORS["original"], alpha=0.85),
               plt.Rectangle((0, 0), 1, 1, color="#6b6b6b", alpha=0.85)]
    fig.legend(handles, ["source utterance (grey, left bar)", "delivered utterance (condition colour, right bar)"],
               frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=11)
    fig.suptitle("Manipulation check: delivered discussion speech against its source", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.08, "Every discussion utterance a rewriter handled, including the originals delivered after a gate fallback; "
             "n counts utterances. Error bars: 95% bootstrap intervals over games.",
             ha="center", va="top", fontsize=11, color="#444444")
    save(fig, out / "manipulation_check")
    return stats_out


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_all(records: List[dict], summary: dict, out: Path, *, utterances: Optional[List[dict]] = None,
             scorer=None) -> dict:
    rows = phase_rows(records)
    stats = {"change_rates": figure_change_rates(rows, out),
             "scope_contrasts": figure_scope_contrasts(rows, out)}
    figure_speaker_outcomes(summary, out)
    figure_self_response(summary, out)
    if utterances is not None and scorer is not None:
        stats["manipulation_check"] = figure_manipulation_check(utterances, scorer, out)
    (out / "figure_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", default=str(GAMES_DIR))
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    parser.add_argument("--out", default=str(FIG_DIR))
    args = parser.parse_args()
    games_dir, out = Path(args.games), Path(args.out)
    records = fa.load_tree(games_dir)
    summary = json.loads(Path(args.summary).read_text())
    stats = make_all(records, summary, out, utterances=delivered_utterances(games_dir), scorer=load_scorer())
    print(f"{len(records)} checkpoints -> {out} ({len(stats)} figure groups)")


if __name__ == "__main__":
    main()
