"""Figures for the politeness flip replays.

Every quantity is aggregated the way the report aggregates it: days within self player,
self players within game, then games with equal weight. Intervals are bootstraps over
games. The proficiency flip's figure helpers do the drawing; the design here is the four
Brown & Levinson strategies read against `bald_on_record`, which carries no redress.

    python -m experiments.politeness_flip.code.figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from components.helpers import read_jsonl
from components.judges.blind_politeness_classifier import LABELS
from experiments.politeness_flip.code import analyze as pa
from experiments.proficiency_flip.code import figures as pf

FIG_DIR = pa.EXPERIMENT_DIR / "figures"
STRATEGY_LABELS = {"bald_on_record": "Bald\non record", "positive_politeness": "Positive\npoliteness",
                   "negative_politeness": "Negative\npoliteness", "off_record": "Off\nrecord"}
STRATEGY_COLORS = {"bald_on_record": "#A86B6B", "positive_politeness": "#C2A882",
                   "negative_politeness": "#8A9E78", "off_record": "#6080A0"}
REFERENCE_COLOR = "#9A9A9A"
OUTCOMES = [("votes_received", "Votes against the speaker", "Δ first-round votes"),
            ("trust_in_speaker", "Trust in the speaker", "Δ mean reliability score (1–10)"),
            ("accused_by_others", "Accusations of the speaker", "Δ predicates calling them a wolf")]


def forest_by_strategy(summary: dict, out: Path) -> None:
    """Each strategy against the unredressed one, and against the untreated replay."""
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(17, 4.6), constrained_layout=True)
    others = [c for c in pa.CONDITIONS if c != pa.REFERENCE]
    for ax, (field, title, xlabel) in zip(axes, OUTCOMES):
        entries = []
        for c in others:
            key = f"{c}_vs_{pa.REFERENCE}"
            entries.append((STRATEGY_LABELS[c].replace("\n", " ") + f" − {STRATEGY_LABELS[pa.REFERENCE]}".replace("\n", " "),
                            summary["flip_self"].get(key, {}).get(field), STRATEGY_COLORS[c]))
        for c in pa.CONDITIONS:
            key = f"{c}_vs_identity"
            entries.append((STRATEGY_LABELS[c].replace("\n", " ") + " − identity",
                            summary["flip_self"].get(key, {}).get(field), REFERENCE_COLOR))
        pf.forest(ax, entries, title, xlabel)
    n = len(summary.get("games") or [])
    fig.suptitle(f"Politeness strategies and the treated speaker, {n} games", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.05, "Coloured rows read a strategy against bald on record, which carries no redress, so the "
             "contrast is politeness alone. Grey rows read it against the untreated replay and also carry the "
             "effect of being rewritten at all. Game-weighted differences with 95% bootstrap intervals over games.",
             ha="center", va="top", fontsize=11, color="#444444")
    pf.save(fig, out / "strategy_outcomes")


def manipulation_figure(check: dict, reference: Optional[dict], out: Path) -> None:
    """Did the delivered speech realize the strategy, and can the classifier read it at all?"""
    blocks = [("the replays' delivered speech", check)] + ([("sets with known answers", reference)] if reference else [])
    fig, axes = plt.subplots(1, len(blocks), figsize=(6.6 * len(blocks), 4.6), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    handles = {}
    for ax, (title, summary) in zip(axes, blocks):
        classifiers = sorted(summary["classifiers"])
        x = np.arange(len(pa.CONDITIONS))
        width = 0.8 / max(1, len(classifiers))
        for i, handle in enumerate(classifiers):
            block = summary["classifiers"][handle]
            vals = [block["agreement"].get(c, 0.0) for c in pa.CONDITIONS]
            pos = x + (i - (len(classifiers) - 1) / 2) * width
            colour = "#6080A0" if i == 0 else "#C49A82"
            bars = ax.bar(pos, vals, width, color=colour, alpha=0.85, edgecolor="none",
                          label=handle.replace("openai:", ""))
            handles[handle] = bars
            for xi, v in zip(pos, vals):
                ax.text(xi, v + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([STRATEGY_LABELS[c] for c in pa.CONDITIONS])
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel("Labelled as the strategy asked for")
    fig.legend([b for b in handles.values()], [h.replace("openai:", "") for h in handles],
               frameon=False, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.0), fontsize=11)
    fig.suptitle("Which politeness strategies the rewriting actually delivers", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.10, "A blind classifier labelled each utterance three times without being told which strategy "
             "was requested. The right panel is the same classifier on material with known answers, so a low bar "
             "there is the classifier's limit rather than the rewriter's.",
             ha="center", va="top", fontsize=11, color="#444444")
    pf.save(fig, out / "manipulation_by_strategy")


def net_change_figure(summary: dict, out: Path) -> None:
    """How much the rewriting changes decisions beyond replaying the day unchanged."""
    net = summary.get("beyond_repeat") or {}
    if not net:
        return
    kinds = [k for k in ("vote", "skill", "reliability") if k in net]
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.6 * len(kinds), 4.4), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds):
        labels = sorted(net[kind])
        ys = np.arange(len(labels))[::-1]
        for y, label in zip(ys, labels):
            e = net[kind][label]
            if e.get("estimate") is None:
                continue
            scope, _, cond = label.partition(":")
            colour = STRATEGY_COLORS.get(cond, "#777777")
            ax.plot([e["ci_low"], e["ci_high"]], [y, y], color=colour, linewidth=2.2, solid_capstyle="round")
            ax.plot([e["estimate"]], [y], marker="o" if scope == "flip_self" else "s", markersize=7, color=colour)
        ax.axvline(0, color="#555555", linewidth=1.1)
        ax.set_yticks(ys)
        ax.set_yticklabels([l.replace("flip_self:", "self · ").replace("flip_others:", "others · ").replace("_", " ")
                            for l in labels], fontsize=10)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:+.0f} pp"))
        ax.set_title(kind, fontsize=13, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.suptitle("What the rewriting adds to the model's own run-to-run movement", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.05, "Each treated replay and the second identity replay are compared with the first identity "
             "replay of the same checkpoint, and the second difference is subtracted from the first. Circles are "
             "the rewritten speaker, squares everyone else. 95% bootstrap intervals over games.",
             ha="center", va="top", fontsize=11, color="#444444")
    pf.save(fig, out / "net_change")


def make_all(summary: dict, check: Optional[dict], reference: Optional[dict], out: Path) -> List[str]:
    made = []
    forest_by_strategy(summary, out); made.append("strategy_outcomes")
    if check:
        manipulation_figure(check, reference, out); made.append("manipulation_by_strategy")
    net_change_figure(summary, out); made.append("net_change")
    return made


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--summary", type=Path, default=pa.SUMMARY_JSON)
    parser.add_argument("--check", type=Path, default=pa.RUN_DIR / "manipulation_check.json")
    parser.add_argument("--reference", type=Path, default=pa.RUN_DIR / "classifier_reference.json")
    parser.add_argument("--out", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    load = lambda p: json.loads(p.read_text()) if p.exists() else None
    made = make_all(summary, load(args.check), load(args.reference), args.out)
    print(f"{len(made)} figures -> {args.out}: {', '.join(made)}")


if __name__ == "__main__":
    main()
