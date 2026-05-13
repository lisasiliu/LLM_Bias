"""Diverging horizontal bar charts + fairness 1x4 for eval batch analysis.

Reference layouts:
  - Diverging bars: ref/ref/diverging_bars.py
  - Fairness 1x4: ref/ref/fairness_1x4.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

# ── Constants ─────────────────────────────────────────────────────────

ALL_ROLES = ["Werewolf", "Guard", "Seer", "Villager"]
SKILL_ROLES = ["Werewolf", "Guard", "Seer"]
ROLE_INITIALS = {"Werewolf": "W", "Guard": "G", "Seer": "S", "Villager": "V"}

ROLE_COLORS = {
    "Werewolf": "#440154",
    "Guard": "#31688e",
    "Seer": "#35b779",
    "Villager": "#fde725",
}

VIRIDIS_1 = ["#31688e"]
VIRIDIS_4 = ["#440154", "#31688e", "#35b779", "#fde725"]
VIRIDIS_8 = [
    "#440154", "#482878", "#3e4989", "#31688e",
    "#26828e", "#1f9e89", "#6ece58", "#fde725",
]

CONFIG_LABELS = {
    "nl_baseline_replay": "NL Replay",
}

MODE_CONFIGS = {
    "pred_vs_nl": {
        "order": ["nl_baseline_replay"],
        "colors": VIRIDIS_1,
        "prefix": "RQ1",
    },
}


def _cap(s: str) -> str:
    return s.capitalize() if s else s


# ── Data loading ──────────────────────────────────────────────────────

def load_batch(batch_dir: Path, mode: str) -> Tuple[List[dict], int, int]:
    """Load per_phase entries from all seeds in batch.

    Returns (per_phase_list, n_games, n_phases).
    n_phases = unique (seed, day) pairs across the batch.
    """
    manifest = batch_dir / "batch_manifest.json"
    if not manifest.exists():
        return [], 0, 0
    seeds = json.loads(manifest.read_text(encoding="utf-8"))["seed_dirs"]
    out: List[dict] = []
    n_games = 0
    phase_days: set = set()
    for idx, rel in enumerate(seeds):
        pm = batch_dir.parent / Path(rel).name / mode / "phase_metrics.json"
        if pm.exists():
            n_games += 1
            data = json.loads(pm.read_text(encoding="utf-8"))
            phases = data.get("per_phase", [])
            out.extend(phases)
            for p in phases:
                phase_days.add((idx, p["day"]))
    return out, n_games, len(phase_days)


def _extract_metric(per_phase, metric_key, config_order, roles):
    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for d in per_phase:
        cfg = d["config_name"]
        if cfg not in config_order:
            continue
        rd = d.get("role_dist", {})

        if metric_key == "vote":
            for row in d.get("delta_vote", []):
                role = _cap(rd.get(row["voter"], ""))
                if role in roles:
                    raw[cfg][role].append(row["changed"])

        elif metric_key == "skill":
            for row in d.get("delta_skill", []):
                role = row.get("role", "")
                if role in roles:
                    raw[cfg][role].append(row["changed"])

        elif metric_key == "reliability":
            for row in d.get("delta_reliability", []):
                if row.get("delta") is None:
                    continue
                role = _cap(rd.get(row["assessor"], ""))
                if role in roles:
                    raw[cfg][role].append(row["delta"])

        elif metric_key == "reliability_target":
            for row in d.get("delta_reliability", []):
                if row.get("delta") is None:
                    continue
                role = _cap(rd.get(row["target"], ""))
                if role in roles:
                    raw[cfg][role].append(row["delta"])

    result = {}
    for cfg in config_order:
        if cfg not in raw:
            continue
        label = CONFIG_LABELS.get(cfg, cfg)
        result[label] = {
            role: float(np.mean(vals))
            for role, vals in raw[cfg].items()
            if role in roles and vals
        }
    return result


def compute_stats(data, roles):
    stats = []
    for cond, role_vals in data.items():
        vals = [role_vals[r] for r in roles if r in role_vals]
        if not vals:
            continue
        stats.append({
            "condition": cond,
            "avg": float(np.mean(vals)),
            "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            "role_vals": {r: role_vals[r] for r in roles if r in role_vals},
        })
    return stats


# ── Diverging bar plot ────────────────────────────────────────────────

def plot_diverging_bars(
    stats, colors, title,
    subtitle=None, roles=None,
    left_label="← Less change", right_label="More change →",
    pct_fmt=True,
    side_label=None,
    figsize=None, save_path=None,
):
    if roles is None:
        roles = ALL_ROLES

    n = len(stats)
    if n == 0:
        return
    if figsize is None:
        figsize = (8, max(2.5, 0.85 * n + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)[::-1]
    bar_height = 0.45

    all_vals = []
    for s in stats:
        all_vals.append(s["avg"])
        all_vals.extend(s["role_vals"].values())
    max_abs = max(0.25, max(abs(v) for v in all_vals) * 1.35) if all_vals else 0.25

    ax.axvline(x=0, color="#888888", linewidth=1.2, zorder=2)

    if pct_fmt:
        grid_step = 0.1 if max_abs <= 0.5 else 0.2
    else:
        grid_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
    for gv in np.arange(-max_abs, max_abs + grid_step, grid_step):
        if abs(gv) > 0.001:
            ax.axvline(x=gv, color="#e8e8ef", linewidth=0.5, linestyle="--", zorder=1)

    label_pad = max_abs * 0.015

    for i, s in enumerate(stats):
        y = y_positions[i]
        avg = s["avg"]
        sem = s["sem"]
        color = colors[i % len(colors)]

        # Semi-transparent bar
        ax.barh(y, avg, height=bar_height, color=color, alpha=0.30,
                edgecolor="none", zorder=3, left=0)

        # Mean tick on bar (visible vertical line at avg)
        tick_half = bar_height * 0.38
        ax.plot([avg, avg], [y - tick_half, y + tick_half],
                color=color, linewidth=2.2, alpha=0.85, zorder=5)

        # SEM error bar
        ax.errorbar(avg, y, xerr=sem, fmt="none", ecolor=color,
                    elinewidth=1.5, capsize=4, capthick=1, alpha=0.7, zorder=4)

        # Value label — at end of bar+error, same y level
        if pct_fmt:
            label_text = f"{avg:+.1%}"
        else:
            label_text = f"{avg:+.2f}"

        edge = avg + sem if avg >= 0 else avg - sem
        if avg >= 0:
            ax.text(edge + label_pad, y, label_text,
                    ha="left", va="center", fontsize=9, fontweight="bold",
                    color=color, zorder=6)
        else:
            ax.text(edge - label_pad, y, label_text,
                    ha="right", va="center", fontsize=9, fontweight="bold",
                    color=color, zorder=6)

        # Role tick marks — start inside bar shading, extend below
        tick_y_top = y
        tick_y_bot = y - bar_height / 2 - 0.04
        text_y = tick_y_bot - 0.07

        # Group nearby initials to avoid overlap
        sorted_roles = sorted(s["role_vals"].items(), key=lambda x: x[1])
        threshold = max_abs * 0.045
        groups: List[dict] = []
        for role_name, val in sorted_roles:
            initial = ROLE_INITIALS.get(role_name, role_name[0])
            if groups and abs(val - groups[-1]["center"]) < threshold:
                groups[-1]["initials"].append(initial)
                groups[-1]["vals"].append(val)
                groups[-1]["center"] = float(np.mean(groups[-1]["vals"]))
            else:
                groups.append({
                    "initials": [initial],
                    "vals": [val],
                    "center": val,
                })

        for g in groups:
            for val in g["vals"]:
                ax.plot([val, val], [tick_y_top, tick_y_bot],
                        color="#555555", linewidth=0.8, zorder=4)
            label = ",".join(g["initials"])
            ax.text(g["center"], text_y, label,
                    ha="center", va="top",
                    fontsize=7.5, fontweight="bold", color="#555555", zorder=5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([s["condition"] for s in stats], fontsize=11, fontweight="600")

    ax.set_xlim(-max_abs, max_abs)

    if pct_fmt:
        tick_step = 0.1 if max_abs <= 0.5 else 0.2
        tick_vals = np.arange(-max_abs, max_abs + tick_step / 2, tick_step)
        tick_vals = [round(v, 2) for v in tick_vals if abs(v) <= max_abs + 0.01]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(
            [f"{v:+.0%}" if abs(v) > 0.001 else "0%" for v in tick_vals],
            fontsize=9,
        )
    else:
        tick_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
        tick_vals = np.arange(-max_abs, max_abs + tick_step / 2, tick_step)
        tick_vals = [round(v, 2) for v in tick_vals if abs(v) <= max_abs + 0.01]
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(
            [f"{v:+.1f}" if abs(v) > 0.001 else "0" for v in tick_vals],
            fontsize=9,
        )

    y_bottom = -0.9 - (0.3 if n > 6 else 0)
    ax.text(-max_abs * 0.85, y_bottom, left_label,
            fontsize=9, fontweight="600", color="#2d936c", ha="left", va="top")
    ax.text(max_abs * 0.85, y_bottom, right_label,
            fontsize=9, fontweight="600", color="#c0392b", ha="right", va="top")

    legend_parts = [f"{ROLE_INITIALS[r]} = {r}" for r in roles if r in ROLE_INITIALS]
    ax.text(0, max(y_positions) + 0.6, "    ".join(legend_parts),
            ha="center", va="bottom", fontsize=8.5, color="#777777")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=24, loc="left")
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=9, color="#888888", style="italic", ha="left", va="bottom")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(left=False)
    ax.set_ylim(-1.2 - (0.3 if n > 6 else 0), max(y_positions) + 1.0)

    if side_label and n > 2:
        y_mid = np.mean(y_positions)
        ax.annotate(
            "", xy=(1.0, max(y_positions)), xytext=(1.0, min(y_positions)),
            xycoords=("axes fraction", "data"),
            textcoords=("axes fraction", "data"),
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
        )
        ax.text(
            1.03, y_mid, side_label,
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.5, color="#999999",
            rotation=-90,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Reliability violin plot ───────────────────────────────────────────

def _extract_reliability_raw(per_phase, config_order, roles):
    """Extract raw per-assessor reliability deltas for violin plots.

    Returns list of dicts (one per condition, in config_order) with:
      - condition: label string
      - all_deltas: flat list of all delta values
      - role_deltas: {role: [deltas]}
      - role_means: {role: mean_delta}
    """
    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for d in per_phase:
        cfg = d["config_name"]
        if cfg not in config_order:
            continue
        rd = d.get("role_dist", {})
        for row in d.get("delta_reliability", []):
            if row.get("delta") is None:
                continue
            role = _cap(rd.get(row["assessor"], ""))
            if role in roles:
                raw[cfg][role].append(row["delta"])

    result = []
    for cfg in config_order:
        if cfg not in raw:
            continue
        label = CONFIG_LABELS.get(cfg, cfg)
        role_deltas = {r: raw[cfg][r] for r in roles if r in raw[cfg] and raw[cfg][r]}
        all_deltas = []
        for vals in role_deltas.values():
            all_deltas.extend(vals)
        if not all_deltas:
            continue
        result.append({
            "condition": label,
            "all_deltas": all_deltas,
            "role_deltas": role_deltas,
            "role_means": {r: float(np.mean(v)) for r, v in role_deltas.items()},
        })
    return result


def plot_reliability_violin(
    vdata, colors, title,
    subtitle=None, roles=None,
    side_label=None,
    figsize=None, save_path=None,
):
    if roles is None:
        roles = ALL_ROLES

    n = len(vdata)
    if n == 0:
        return
    if figsize is None:
        figsize = (8, max(2.5, 0.85 * n + 1.5))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_positions = np.arange(n)[::-1]

    all_vals = []
    for v in vdata:
        all_vals.extend(v["all_deltas"])
    max_abs = max(0.5, np.percentile(np.abs(all_vals), 99) * 1.4) if all_vals else 0.5

    ax.axvline(x=0, color="#888888", linewidth=1.2, zorder=2)

    grid_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
    for gv in np.arange(-max_abs, max_abs + grid_step, grid_step):
        if abs(gv) > 0.001:
            ax.axvline(x=gv, color="#e8e8ef", linewidth=0.5, linestyle="--", zorder=1)

    body_height = 0.7

    for i, v in enumerate(vdata):
        y = y_positions[i]
        color = colors[i % len(colors)]
        deltas = v["all_deltas"]

        parts = ax.violinplot(
            deltas, positions=[y], vert=False,
            showmeans=False, showmedians=False, showextrema=False,
            widths=body_height,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.35)
            pc.set_zorder(3)

        avg = float(np.mean(deltas))
        median = float(np.median(deltas))

        # Mean tick
        tick_half = body_height * 0.35
        ax.plot([avg, avg], [y - tick_half, y + tick_half],
                color=color, linewidth=2.2, alpha=0.85, zorder=5)

        # Median tick (thinner)
        ax.plot([median, median], [y - tick_half * 0.6, y + tick_half * 0.6],
                color="#333333", linewidth=1.0, alpha=0.6, zorder=5)

        # Value label
        label_pad = max_abs * 0.015
        q75 = float(np.percentile(deltas, 75))
        edge = max(avg, q75)
        ax.text(edge + label_pad, y, f"{avg:+.2f}",
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=color, zorder=6)

        # Role-mean tick marks below violin
        tick_y_top = y - body_height / 2 + 0.05
        tick_y_bot = tick_y_top - 0.12
        text_y = tick_y_bot - 0.07

        sorted_roles = sorted(v["role_means"].items(), key=lambda x: x[1])
        threshold = max_abs * 0.045
        groups: List[dict] = []
        for role_name, val in sorted_roles:
            initial = ROLE_INITIALS.get(role_name, role_name[0])
            if groups and abs(val - groups[-1]["center"]) < threshold:
                groups[-1]["initials"].append(initial)
                groups[-1]["vals"].append(val)
                groups[-1]["center"] = float(np.mean(groups[-1]["vals"]))
            else:
                groups.append({
                    "initials": [initial],
                    "vals": [val],
                    "center": val,
                })

        for g in groups:
            for val in g["vals"]:
                ax.plot([val, val], [tick_y_top, tick_y_bot],
                        color="#555555", linewidth=0.8, zorder=4)
            label = ",".join(g["initials"])
            ax.text(g["center"], text_y, label,
                    ha="center", va="top",
                    fontsize=7.5, fontweight="bold", color="#555555", zorder=5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([v["condition"] for v in vdata], fontsize=11, fontweight="600")

    ax.set_xlim(-max_abs, max_abs)
    tick_step = 2.0 if max_abs > 6 else (1.0 if max_abs > 2 else 0.5)
    tick_vals = np.arange(-max_abs, max_abs + tick_step / 2, tick_step)
    tick_vals = [round(v, 1) for v in tick_vals if abs(v) <= max_abs + 0.01]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(
        [f"{v:+.0f}" if abs(v) > 0.001 else "0" for v in tick_vals],
        fontsize=9,
    )

    y_bottom = -0.9 - (0.3 if n > 6 else 0)
    ax.text(-max_abs * 0.85, y_bottom, "← Decreased trust",
            fontsize=9, fontweight="600", color="#2d936c", ha="left", va="top")
    ax.text(max_abs * 0.85, y_bottom, "Increased trust →",
            fontsize=9, fontweight="600", color="#c0392b", ha="right", va="top")

    legend_parts = [f"{ROLE_INITIALS[r]} = {r}" for r in roles if r in ROLE_INITIALS]
    ax.text(0, max(y_positions) + 0.6, "    ".join(legend_parts),
            ha="center", va="bottom", fontsize=8.5, color="#777777")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=24, loc="left")
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=9, color="#888888", style="italic", ha="left", va="bottom")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(left=False)
    ax.set_ylim(-1.2 - (0.3 if n > 6 else 0), max(y_positions) + 1.0)

    if side_label and n > 2:
        y_mid = np.mean(y_positions)
        ax.annotate(
            "", xy=(1.0, max(y_positions)), xytext=(1.0, min(y_positions)),
            xycoords=("axes fraction", "data"),
            textcoords=("axes fraction", "data"),
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
        )
        ax.text(
            1.03, y_mid, side_label,
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=8.5, color="#999999",
            rotation=-90,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Wide (flipped-axis) diverging bar plot ───────────────────────────

def plot_diverging_bars_wide(
    stats, colors, title,
    subtitle=None, roles=None,
    bottom_label="← Less change", top_label="More change →",
    pct_fmt=True,
    side_label=None,
    figsize=None, save_path=None,
):
    """Vertical diverging bars: conditions on x-axis, values on y-axis."""
    if roles is None:
        roles = ALL_ROLES

    n = len(stats)
    if n == 0:
        return
    if figsize is None:
        figsize = (max(6, 1.6 * n + 2.0), 5.5)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_positions = np.arange(n)
    bar_width = 0.45

    all_vals = []
    for s in stats:
        all_vals.append(s["avg"])
        all_vals.extend(s["role_vals"].values())
    max_abs = max(0.25, max(abs(v) for v in all_vals) * 1.35) if all_vals else 0.25

    ax.axhline(y=0, color="#888888", linewidth=1.2, zorder=2)

    if pct_fmt:
        grid_step = 0.1 if max_abs <= 0.5 else 0.2
    else:
        grid_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
    for gv in np.arange(-max_abs, max_abs + grid_step, grid_step):
        if abs(gv) > 0.001:
            ax.axhline(y=gv, color="#e8e8ef", linewidth=0.5, linestyle="--", zorder=1)

    label_pad = max_abs * 0.02

    for i, s in enumerate(stats):
        x = x_positions[i]
        avg = s["avg"]
        sem = s["sem"]
        color = colors[i % len(colors)]

        ax.bar(x, avg, width=bar_width, color=color, alpha=0.30,
               edgecolor="none", zorder=3, bottom=0)

        tick_half = bar_width * 0.38
        ax.plot([x - tick_half, x + tick_half], [avg, avg],
                color=color, linewidth=2.2, alpha=0.85, zorder=5)

        ax.errorbar(x, avg, yerr=sem, fmt="none", ecolor=color,
                    elinewidth=1.5, capsize=4, capthick=1, alpha=0.7, zorder=4)

        if pct_fmt:
            label_text = f"{avg:+.1%}"
        else:
            label_text = f"{avg:+.2f}"

        edge = avg + sem if avg >= 0 else avg - sem
        va = "bottom" if avg >= 0 else "top"
        offset = label_pad if avg >= 0 else -label_pad
        ax.text(x, edge + offset, label_text,
                ha="center", va=va, fontsize=9, fontweight="bold",
                color=color, zorder=6)

        # Role tick marks to the right of each bar
        tick_x_left = x + bar_width / 2 + 0.04
        tick_x_right = tick_x_left + 0.08
        text_x = tick_x_right + 0.04

        sorted_roles = sorted(s["role_vals"].items(), key=lambda kv: kv[1])
        threshold = max_abs * 0.045
        groups: List[dict] = []
        for role_name, val in sorted_roles:
            initial = ROLE_INITIALS.get(role_name, role_name[0])
            if groups and abs(val - groups[-1]["center"]) < threshold:
                groups[-1]["initials"].append(initial)
                groups[-1]["vals"].append(val)
                groups[-1]["center"] = float(np.mean(groups[-1]["vals"]))
            else:
                groups.append({
                    "initials": [initial],
                    "vals": [val],
                    "center": val,
                })

        for g in groups:
            for val in g["vals"]:
                ax.plot([tick_x_left, tick_x_right], [val, val],
                        color="#555555", linewidth=0.8, zorder=4)
            label = ",".join(g["initials"])
            ax.text(text_x, g["center"], label,
                    ha="left", va="center",
                    fontsize=7, fontweight="bold", color="#555555", zorder=5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([s["condition"] for s in stats], fontsize=10, fontweight="600")

    ax.set_ylim(-max_abs, max_abs)

    if pct_fmt:
        tick_step = 0.1 if max_abs <= 0.5 else 0.2
        tick_vals = np.arange(-max_abs, max_abs + tick_step / 2, tick_step)
        tick_vals = [round(v, 2) for v in tick_vals if abs(v) <= max_abs + 0.01]
        ax.set_yticks(tick_vals)
        ax.set_yticklabels(
            [f"{v:+.0%}" if abs(v) > 0.001 else "0%" for v in tick_vals],
            fontsize=9,
        )
    else:
        tick_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
        tick_vals = np.arange(-max_abs, max_abs + tick_step / 2, tick_step)
        tick_vals = [round(v, 2) for v in tick_vals if abs(v) <= max_abs + 0.01]
        ax.set_yticks(tick_vals)
        ax.set_yticklabels(
            [f"{v:+.1f}" if abs(v) > 0.001 else "0" for v in tick_vals],
            fontsize=9,
        )

    x_right = n - 0.5 + 0.6
    ax.text(x_right, -max_abs * 0.85, bottom_label,
            fontsize=9, fontweight="600", color="#2d936c", ha="right", va="top",
            rotation=90)
    ax.text(x_right, max_abs * 0.85, top_label,
            fontsize=9, fontweight="600", color="#c0392b", ha="right", va="bottom",
            rotation=90)

    legend_parts = [f"{ROLE_INITIALS[r]} = {r}" for r in roles if r in ROLE_INITIALS]
    ax.text(np.mean(x_positions), max_abs * 1.08, "    ".join(legend_parts),
            ha="center", va="bottom", fontsize=8.5, color="#777777")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=24, loc="left")
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=9, color="#888888", style="italic", ha="left", va="bottom")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(bottom=False)
    ax.set_xlim(-0.6, n - 0.5 + 0.8)

    if side_label and n > 2:
        ax.annotate(
            "", xy=(n - 0.5, -0.12), xytext=(0, -0.12),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
        )
        ax.text(np.mean(x_positions), -0.16, side_label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color="#999999")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Wide (flipped-axis) reliability violin ───────────────────────────

def plot_reliability_violin_wide(
    vdata, colors, title,
    subtitle=None, roles=None,
    side_label=None,
    figsize=None, save_path=None,
):
    """Vertical violin plot: conditions on x-axis, deltas on y-axis."""
    if roles is None:
        roles = ALL_ROLES

    n = len(vdata)
    if n == 0:
        return
    if figsize is None:
        figsize = (max(6, 1.6 * n + 2.0), 5.5)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x_positions = np.arange(n)

    all_vals = []
    for v in vdata:
        all_vals.extend(v["all_deltas"])
    max_abs = max(0.5, np.percentile(np.abs(all_vals), 99) * 1.4) if all_vals else 0.5

    ax.axhline(y=0, color="#888888", linewidth=1.2, zorder=2)

    grid_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
    for gv in np.arange(-max_abs, max_abs + grid_step, grid_step):
        if abs(gv) > 0.001:
            ax.axhline(y=gv, color="#e8e8ef", linewidth=0.5, linestyle="--", zorder=1)

    body_width = 0.7

    for i, v in enumerate(vdata):
        x = x_positions[i]
        color = colors[i % len(colors)]
        deltas = v["all_deltas"]

        parts = ax.violinplot(
            deltas, positions=[x], vert=True,
            showmeans=False, showmedians=False, showextrema=False,
            widths=body_width,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.35)
            pc.set_zorder(3)

        avg = float(np.mean(deltas))
        median = float(np.median(deltas))

        tick_half = body_width * 0.35
        ax.plot([x - tick_half, x + tick_half], [avg, avg],
                color=color, linewidth=2.2, alpha=0.85, zorder=5)

        ax.plot([x - tick_half * 0.6, x + tick_half * 0.6], [median, median],
                color="#333333", linewidth=1.0, alpha=0.6, zorder=5)

        label_pad = max_abs * 0.02
        q75 = float(np.percentile(deltas, 75))
        edge = max(avg, q75)
        ax.text(x, edge + label_pad, f"{avg:+.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=color, zorder=6)

        # Role-mean tick marks to the right of violin
        tick_x_left = x + body_width / 2 - 0.05
        tick_x_right = tick_x_left + 0.1
        text_x = tick_x_right + 0.04

        sorted_roles = sorted(v["role_means"].items(), key=lambda kv: kv[1])
        threshold = max_abs * 0.045
        groups: List[dict] = []
        for role_name, val in sorted_roles:
            initial = ROLE_INITIALS.get(role_name, role_name[0])
            if groups and abs(val - groups[-1]["center"]) < threshold:
                groups[-1]["initials"].append(initial)
                groups[-1]["vals"].append(val)
                groups[-1]["center"] = float(np.mean(groups[-1]["vals"]))
            else:
                groups.append({
                    "initials": [initial],
                    "vals": [val],
                    "center": val,
                })

        for g in groups:
            for val in g["vals"]:
                ax.plot([tick_x_left, tick_x_right], [val, val],
                        color="#555555", linewidth=0.8, zorder=4)
            label = ",".join(g["initials"])
            ax.text(text_x, g["center"], label,
                    ha="left", va="center",
                    fontsize=7, fontweight="bold", color="#555555", zorder=5)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([v["condition"] for v in vdata], fontsize=10, fontweight="600")

    ax.set_ylim(-max_abs, max_abs)
    tick_step = 2.0 if max_abs > 6 else (1.0 if max_abs > 2 else 0.5)
    tick_vals = np.arange(-max_abs, max_abs + tick_step / 2, tick_step)
    tick_vals = [round(v, 1) for v in tick_vals if abs(v) <= max_abs + 0.01]
    ax.set_yticks(tick_vals)
    ax.set_yticklabels(
        [f"{v:+.0f}" if abs(v) > 0.001 else "0" for v in tick_vals],
        fontsize=9,
    )

    x_right = n - 0.5 + 0.6
    ax.text(x_right, -max_abs * 0.85, "← Decreased trust",
            fontsize=9, fontweight="600", color="#2d936c", ha="right", va="top",
            rotation=90)
    ax.text(x_right, max_abs * 0.85, "Increased trust →",
            fontsize=9, fontweight="600", color="#c0392b", ha="right", va="bottom",
            rotation=90)

    legend_parts = [f"{ROLE_INITIALS[r]} = {r}" for r in roles if r in ROLE_INITIALS]
    ax.text(np.mean(x_positions), max_abs * 1.08, "    ".join(legend_parts),
            ha="center", va="bottom", fontsize=8.5, color="#777777")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=24, loc="left")
    if subtitle:
        ax.text(0.0, 1.005, subtitle, transform=ax.transAxes,
                fontsize=9, color="#888888", style="italic", ha="left", va="bottom")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(bottom=False)
    ax.set_xlim(-0.6, n - 0.5 + 0.8)

    if side_label and n > 2:
        ax.annotate(
            "", xy=(n - 0.5, -0.12), xytext=(0, -0.12),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->", color="#aaaaaa", lw=1.2),
        )
        ax.text(np.mean(x_positions), -0.16, side_label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8.5, color="#999999")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Fairness 1x4 (ref/ref/fairness_1x4.py) ──────────────────────────

def _extract_fairness(per_phase, config_order, roles):
    """Extract combined fairness data for 1x4 chart (paper Eq. 3).

    Fairness(s, p) = 1 - avg(Δ_s) where Δ_s is binary (1=changed, 0=same).
    Pools all decision types (vote + skill + reliability) into one metric.
    Only uses flip_others_* configs (how robust is p when others change).
    Bounded [0, 1], optimal = 1.0.
    """
    others_only = [c for c in config_order if c.startswith("flip_others")]
    by_role_day: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for d in per_phase:
        cfg = d["config_name"]
        if cfg not in others_only:
            continue
        rd = d.get("role_dist", {})
        day = d["day"]

        for row in d.get("delta_vote", []):
            role = _cap(rd.get(row["voter"], ""))
            if role in roles:
                same = 1 - row["changed"]
                by_role_day[role][f"Day {day}"].append(same)
                by_role_day[role]["Total"].append(same)

        for row in d.get("delta_skill", []):
            role = row.get("role", "")
            if role in roles:
                same = 1 - row["changed"]
                by_role_day[role][f"Day {day}"].append(same)
                by_role_day[role]["Total"].append(same)

        for row in d.get("delta_reliability", []):
            if row.get("delta") is None:
                continue
            role = _cap(rd.get(row["assessor"], ""))
            if role in roles:
                same = 0 if row["delta"] != 0 else 1
                by_role_day[role][f"Day {day}"].append(same)
                by_role_day[role]["Total"].append(same)

    all_days = set()
    for role_data in by_role_day.values():
        for phase in role_data:
            if phase.startswith("Day "):
                all_days.add(int(phase.split()[1]))
    phases = ["Total"] + [f"Day {d}" for d in sorted(all_days)]

    result = {}
    for role in roles:
        result[role] = {}
        for phase in phases:
            vals = by_role_day.get(role, {}).get(phase, [])
            if vals:
                result[role][phase] = {
                    "mean": float(np.mean(vals)),
                    "sem": float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
                }
            else:
                result[role][phase] = {"mean": 0.0, "sem": 0.0}

    return result, phases


def plot_fairness_1x4(
    data, roles=None, phases=None,
    role_colors=None, optimal_value=1.0,
    title="Fairness: Consistency of assessments over phases",
    y_label="Freq(s, p)", y_max=None,
    subtitle=None,
    figsize=(16, 4.2), save_path=None,
):
    if roles is None:
        roles = ALL_ROLES
    if phases is None:
        phases = ["Total", "Day 1", "Day 2", "Day 3"]
    if role_colors is None:
        role_colors = ROLE_COLORS

    n_roles = len(roles)
    n_phases = len(phases)

    if y_max is None:
        all_means = []
        for role in roles:
            for phase in phases:
                all_means.append(data.get(role, {}).get(phase, {}).get("mean", 0))
        peak = max(all_means) if all_means else 1.0
        y_max = max(optimal_value * 1.08 if optimal_value else 1.08,
                    peak * 1.15)

    fig, axes = plt.subplots(1, n_roles, figsize=figsize, sharey=True)
    fig.patch.set_facecolor("white")

    if n_roles == 1:
        axes = [axes]

    x = np.arange(n_phases)
    bar_width = 0.55

    for i, role in enumerate(roles):
        ax = axes[i]
        ax.set_facecolor("white")
        color = role_colors.get(role, "#31688e")

        means = [data.get(role, {}).get(p, {}).get("mean", 0) for p in phases]
        sems = [data.get(role, {}).get(p, {}).get("sem", 0) for p in phases]

        ax.bar(x, means, width=bar_width, color=color, alpha=0.75,
               edgecolor="none", zorder=3)

        ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#333333",
                    elinewidth=0.9, capsize=4, capthick=0.9, zorder=4)

        for j, (m, s) in enumerate(zip(means, sems)):
            ax.text(j, m + s + y_max * 0.025, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color="#444444", zorder=5)

        if optimal_value is not None:
            ax.axhline(y=optimal_value, color="#888888", linewidth=1,
                       linestyle=(0, (6, 3)), zorder=2)
            if i == 0:
                ax.text(-0.4, optimal_value, f"Optimal = {optimal_value:.1f}",
                        ha="left", va="bottom", fontsize=7.5,
                        color="#888888", style="italic")

        ax.set_title(role, fontsize=12, fontweight="600", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(phases, fontsize=9)
        ax.set_xlim(-0.5, n_phases - 0.5)
        ax.set_ylim(0, y_max)

        if i == 0:
            ax.set_ylabel(y_label, fontsize=10, labelpad=6)
            ax.tick_params(axis="y", labelsize=9)
        else:
            ax.tick_params(axis="y", length=0)

        ax.yaxis.grid(True, color="#ececf0", linewidth=0.4, zorder=1)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i == 0:
            ax.spines["left"].set_color("#dddddd")
        else:
            ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.10)
    if subtitle:
        fig.text(0.5, 1.02, subtitle, ha="center", fontsize=9,
                 color="#888888", style="italic")
    plt.subplots_adjust(wspace=0.06)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Directional self-targeting extractors ─────────────────────────────

def _extract_self_targeting_vote(per_phase, config_order, roles):
    """Δ votes received by the self player, grouped by self_role.

    Positive = self attracted MORE votes after the style change.
    """
    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for d in per_phase:
        cfg = d["config_name"]
        if cfg not in config_order:
            continue
        sp = d.get("self_player")
        sr = _cap(d.get("self_role", ""))
        if not sp or sr not in roles:
            continue
        for row in d.get("delta_votes_received", []):
            if row["target"] == sp:
                raw[cfg][sr].append(row["delta"])

    result = {}
    for cfg in config_order:
        if cfg not in raw:
            continue
        label = CONFIG_LABELS.get(cfg, cfg)
        result[label] = {
            role: float(np.mean(vals))
            for role, vals in raw[cfg].items()
            if role in roles and vals
        }
    return result


def _extract_self_targeting_skill(per_phase, config_order, roles):
    """How often self becomes / stops being a skill target.

    +1 = self was NOT targeted in baseline but IS in replay.
    -1 = self WAS targeted in baseline but is NOT in replay.
     0 = no change.
    """
    raw: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for d in per_phase:
        cfg = d["config_name"]
        if cfg not in config_order:
            continue
        sp = d.get("self_player")
        if not sp:
            continue
        for row in d.get("delta_skill", []):
            role = row.get("role", "")
            if role not in roles:
                continue
            was_self = 1 if row["baseline"] == sp else 0
            now_self = 1 if row["replay"] == sp else 0
            raw[cfg][role].append(now_self - was_self)

    result = {}
    for cfg in config_order:
        if cfg not in raw:
            continue
        label = CONFIG_LABELS.get(cfg, cfg)
        result[label] = {
            role: float(np.mean(vals))
            for role, vals in raw[cfg].items()
            if role in roles and vals
        }
    return result


SELF_TARGET_METRIC_DEFS = [
    {
        "extractor": "_extract_self_targeting_skill",
        "name": "skills",
        "roles": SKILL_ROLES,
        "title_suffix": "Skills: Δ self-targeting rate",
        "subtitle": "Whether self became a skill target after style change.",
        "left_label": "← Less targeted",
        "right_label": "More targeted →",
        "pct_fmt": True,
    },
    {
        "extractor": "_extract_self_targeting_vote",
        "name": "votes",
        "roles": ALL_ROLES,
        "title_suffix": "Votes: Δ votes received by self",
        "subtitle": "Change in votes received by the self player.",
        "left_label": "← Fewer votes against self",
        "right_label": "More votes against self →",
        "pct_fmt": False,
    },
    {
        "extractor": "reliability",
        "name": "reliability",
        "roles": ALL_ROLES,
        "title_suffix": "Reliability: Δ trust score (distribution)",
        "subtitle": "Per-assessor reliability deltas.",
        "violin": True,
    },
]

_EXTRACTORS = {
    "_extract_self_targeting_vote": _extract_self_targeting_vote,
    "_extract_self_targeting_skill": _extract_self_targeting_skill,
}


def plot_combined_self_targeting(
    per_phase, config_order, colors, prefix, info,
    roles_vote=None, roles_skill=None, roles_rel=None,
    side_label=None, save_path=None,
):
    """Horizontal 1x3 figure: skill targeting / vote targeting / reliability violin."""
    if roles_vote is None:
        roles_vote = ALL_ROLES
    if roles_skill is None:
        roles_skill = SKILL_ROLES
    if roles_rel is None:
        roles_rel = ALL_ROLES

    # Panel 0: skill self-targeting (diverging bars)
    skill_data = _extract_self_targeting_skill(per_phase, config_order, roles_skill)
    skill_stats = compute_stats(skill_data, roles_skill) if skill_data else []

    # Panel 1: vote self-targeting (diverging bars)
    vote_data = _extract_self_targeting_vote(per_phase, config_order, roles_vote)
    vote_stats = compute_stats(vote_data, roles_vote) if vote_data else []

    # Panel 2: reliability violin
    vdata = _extract_reliability_raw(per_phase, config_order, roles_rel)

    n_bars = max(len(skill_stats), len(vote_stats), len(vdata))
    if n_bars == 0:
        return None

    col_w = 7.5
    row_h = max(2.5, 0.85 * n_bars + 1.5)
    fig, axes = plt.subplots(1, 3, figsize=(col_w * 3, row_h + 1.0))
    fig.patch.set_facecolor("white")

    def _draw_diverging(ax, stats, colors, title, subtitle, roles, left_label, right_label, pct_fmt, show_ylabels=True):
        n = len(stats)
        if n == 0:
            ax.set_visible(False)
            return
        ax.set_facecolor("white")
        y_positions = np.arange(n)[::-1]
        bar_height = 0.45
        all_vals = []
        for s in stats:
            all_vals.append(s["avg"])
            all_vals.extend(s["role_vals"].values())
        max_abs = max(0.25, max(abs(v) for v in all_vals) * 1.35) if all_vals else 0.25
        ax.axvline(x=0, color="#888888", linewidth=1.2, zorder=2)
        if pct_fmt:
            grid_step = 0.1 if max_abs <= 0.5 else 0.2
        else:
            grid_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
        for gv in np.arange(-max_abs, max_abs + grid_step, grid_step):
            if abs(gv) > 0.001:
                ax.axvline(x=gv, color="#e8e8ef", linewidth=0.5, linestyle="--", zorder=1)
        label_pad = max_abs * 0.015
        for i, s in enumerate(stats):
            y = y_positions[i]
            avg = s["avg"]
            sem = s["sem"]
            color = colors[i % len(colors)]
            ax.barh(y, avg, height=bar_height, color=color, alpha=0.30, edgecolor="none", zorder=3, left=0)
            tick_half = bar_height * 0.38
            ax.plot([avg, avg], [y - tick_half, y + tick_half], color=color, linewidth=2.2, alpha=0.85, zorder=5)
            ax.errorbar(avg, y, xerr=sem, fmt="none", ecolor=color, elinewidth=1.5, capsize=4, capthick=1, alpha=0.7, zorder=4)
            if pct_fmt:
                label_text = f"{avg:+.1%}"
            else:
                label_text = f"{avg:+.2f}"
            edge = avg + sem if avg >= 0 else avg - sem
            ha = "left" if avg >= 0 else "right"
            offset = label_pad if avg >= 0 else -label_pad
            ax.text(edge + offset, y, label_text, ha=ha, va="center", fontsize=9, fontweight="bold", color=color, zorder=6)
            tick_y_top = y
            tick_y_bot = y - bar_height / 2 - 0.04
            text_y = tick_y_bot - 0.07
            sorted_roles = sorted(s["role_vals"].items(), key=lambda x: x[1])
            threshold = max_abs * 0.045
            groups: List[dict] = []
            for role_name, val in sorted_roles:
                initial = ROLE_INITIALS.get(role_name, role_name[0])
                if groups and abs(val - groups[-1]["center"]) < threshold:
                    groups[-1]["initials"].append(initial)
                    groups[-1]["vals"].append(val)
                    groups[-1]["center"] = float(np.mean(groups[-1]["vals"]))
                else:
                    groups.append({"initials": [initial], "vals": [val], "center": val})
            for g in groups:
                for val in g["vals"]:
                    ax.plot([val, val], [tick_y_top, tick_y_bot], color="#555555", linewidth=0.8, zorder=4)
                label = ",".join(g["initials"])
                ax.text(g["center"], text_y, label, ha="center", va="top", fontsize=7.5, fontweight="bold", color="#555555", zorder=5)
        ax.set_yticks(y_positions)
        if show_ylabels:
            ax.set_yticklabels([s["condition"] for s in stats], fontsize=10, fontweight="600")
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-max_abs, max_abs)
        if pct_fmt:
            ts = 0.1 if max_abs <= 0.5 else 0.2
            tvs = [round(v, 2) for v in np.arange(-max_abs, max_abs + ts / 2, ts) if abs(v) <= max_abs + 0.01]
            ax.set_xticks(tvs)
            ax.set_xticklabels([f"{v:+.0%}" if abs(v) > 0.001 else "0%" for v in tvs], fontsize=8)
        else:
            ts = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
            tvs = [round(v, 2) for v in np.arange(-max_abs, max_abs + ts / 2, ts) if abs(v) <= max_abs + 0.01]
            ax.set_xticks(tvs)
            ax.set_xticklabels([f"{v:+.1f}" if abs(v) > 0.001 else "0" for v in tvs], fontsize=8)
        y_bottom = -0.9
        ax.text(-max_abs * 0.85, y_bottom, left_label, fontsize=8, fontweight="600", color="#2d936c", ha="left", va="top")
        ax.text(max_abs * 0.85, y_bottom, right_label, fontsize=8, fontweight="600", color="#c0392b", ha="right", va="top")
        legend_parts = [f"{ROLE_INITIALS[r]} = {r}" for r in roles if r in ROLE_INITIALS]
        ax.text(0, max(y_positions) + 0.6, "  ".join(legend_parts), ha="center", va="bottom", fontsize=7.5, color="#777777")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=18, loc="left")
        if subtitle:
            ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=8, color="#888888", style="italic", ha="left", va="bottom")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(left=False)
        ax.set_ylim(-1.2, max(y_positions) + 1.0)

    def _draw_violin(ax, vdata, colors, title, subtitle, roles, show_ylabels=True):
        n = len(vdata)
        if n == 0:
            ax.set_visible(False)
            return
        ax.set_facecolor("white")
        y_positions = np.arange(n)[::-1]
        all_vals = []
        for v in vdata:
            all_vals.extend(v["all_deltas"])
        max_abs = max(0.5, np.percentile(np.abs(all_vals), 99) * 1.4) if all_vals else 0.5
        ax.axvline(x=0, color="#888888", linewidth=1.2, zorder=2)
        grid_step = 0.5 if max_abs > 2 else (0.2 if max_abs > 0.5 else 0.1)
        for gv in np.arange(-max_abs, max_abs + grid_step, grid_step):
            if abs(gv) > 0.001:
                ax.axvline(x=gv, color="#e8e8ef", linewidth=0.5, linestyle="--", zorder=1)
        body_height = 0.7
        for i, v in enumerate(vdata):
            y = y_positions[i]
            color = colors[i % len(colors)]
            deltas = v["all_deltas"]
            parts = ax.violinplot(deltas, positions=[y], vert=False, showmeans=False, showmedians=False, showextrema=False, widths=body_height)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor(color)
                pc.set_alpha(0.35)
                pc.set_zorder(3)
            avg = float(np.mean(deltas))
            median = float(np.median(deltas))
            tick_half = body_height * 0.35
            ax.plot([avg, avg], [y - tick_half, y + tick_half], color=color, linewidth=2.2, alpha=0.85, zorder=5)
            ax.plot([median, median], [y - tick_half * 0.6, y + tick_half * 0.6], color="#333333", linewidth=1.0, alpha=0.6, zorder=5)
            label_pad = max_abs * 0.015
            q75 = float(np.percentile(deltas, 75))
            edge = max(avg, q75)
            ax.text(edge + label_pad, y, f"{avg:+.2f}", ha="left", va="center", fontsize=9, fontweight="bold", color=color, zorder=6)
            tick_y_top = y - body_height / 2 + 0.05
            tick_y_bot = tick_y_top - 0.12
            text_y = tick_y_bot - 0.07
            sorted_roles = sorted(v["role_means"].items(), key=lambda x: x[1])
            threshold = max_abs * 0.045
            groups: List[dict] = []
            for role_name, val in sorted_roles:
                initial = ROLE_INITIALS.get(role_name, role_name[0])
                if groups and abs(val - groups[-1]["center"]) < threshold:
                    groups[-1]["initials"].append(initial)
                    groups[-1]["vals"].append(val)
                    groups[-1]["center"] = float(np.mean(groups[-1]["vals"]))
                else:
                    groups.append({"initials": [initial], "vals": [val], "center": val})
            for g in groups:
                for val in g["vals"]:
                    ax.plot([val, val], [tick_y_top, tick_y_bot], color="#555555", linewidth=0.8, zorder=4)
                label = ",".join(g["initials"])
                ax.text(g["center"], text_y, label, ha="center", va="top", fontsize=7.5, fontweight="bold", color="#555555", zorder=5)
        ax.set_yticks(y_positions)
        if show_ylabels:
            ax.set_yticklabels([v["condition"] for v in vdata], fontsize=10, fontweight="600")
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-max_abs, max_abs)
        ts = 2.0 if max_abs > 6 else (1.0 if max_abs > 2 else 0.5)
        tvs = [round(v, 1) for v in np.arange(-max_abs, max_abs + ts / 2, ts) if abs(v) <= max_abs + 0.01]
        ax.set_xticks(tvs)
        ax.set_xticklabels([f"{v:+.0f}" if abs(v) > 0.001 else "0" for v in tvs], fontsize=8)
        y_bottom = -0.9
        ax.text(-max_abs * 0.85, y_bottom, "← Decreased trust", fontsize=8, fontweight="600", color="#2d936c", ha="left", va="top")
        ax.text(max_abs * 0.85, y_bottom, "Increased trust →", fontsize=8, fontweight="600", color="#c0392b", ha="right", va="top")
        legend_parts = [f"{ROLE_INITIALS[r]} = {r}" for r in roles if r in ROLE_INITIALS]
        ax.text(0, max(y_positions) + 0.6, "  ".join(legend_parts), ha="center", va="bottom", fontsize=7.5, color="#777777")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=18, loc="left")
        if subtitle:
            ax.text(0.0, 1.005, subtitle, transform=ax.transAxes, fontsize=8, color="#888888", style="italic", ha="left", va="bottom")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")
        ax.tick_params(left=False)
        ax.set_ylim(-1.2, max(y_positions) + 1.0)

    # Draw three horizontal panels — y-labels on leftmost only
    _draw_diverging(
        axes[0], skill_stats, colors,
        "Skills: Δ self-targeting",
        f"Skill target changes.  ({info})",
        roles_skill, "← Less targeted", "More targeted →", True,
        show_ylabels=True,
    )
    _draw_diverging(
        axes[1], vote_stats, colors,
        "Votes: Δ received by self",
        f"Vote changes.  ({info})",
        roles_vote, "← Fewer votes", "More votes →", False,
        show_ylabels=False,
    )
    _draw_violin(
        axes[2], vdata, colors,
        "Reliability: Δ trust (dist.)",
        f"Per-assessor deltas.  ({info})",
        roles_rel,
        show_ylabels=False,
    )

    fig.suptitle(
        f"{prefix}: Self-targeting analysis", fontsize=13,
        fontweight="bold", y=1.04,
    )
    plt.tight_layout(w_pad=1.5)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return save_path


# ── Metric definitions (diverging bars only — no fairness) ───────────

METRIC_DEFS = [
    {
        "key": "vote", "name": "votes", "roles": ALL_ROLES,
        "title_suffix": "Votes: Δ vote change rate",
        "subtitle": "Bar = avg across roles. Tick marks = per-role values.",
        "left_label": "← Fewer changes", "right_label": "More changes →",
        "pct_fmt": True,
    },
    {
        "key": "skill", "name": "skills", "roles": SKILL_ROLES,
        "title_suffix": "Skills: Δ skill change rate",
        "subtitle": "Proportion of skill actions that changed.",
        "left_label": "← Fewer changes", "right_label": "More changes →",
        "pct_fmt": True,
    },
    {
        "key": "reliability", "name": "reliability", "roles": ALL_ROLES,
        "title_suffix": "Reliability: Δ trust score (by assessor)",
        "subtitle": "Mean change in trust assessments by assessor role.",
        "left_label": "← Decreased trust", "right_label": "Increased trust →",
        "pct_fmt": False,
    },
    {
        "key": "reliability_target", "name": "reliability_target", "roles": ALL_ROLES,
        "title_suffix": "Reliability: Δ trust received (by target)",
        "subtitle": "Mean change in trust received, by target role.",
        "left_label": "← Less trusted", "right_label": "More trusted →",
        "pct_fmt": False,
    },
]


# ── Batch rendering ───────────────────────────────────────────────────

def render_batch(batch_dir: Path, out_dir: Optional[Path] = None) -> List[Path]:
    batch_dir = Path(batch_dir)
    written: List[Path] = []

    for mode, mcfg in MODE_CONFIGS.items():
        pp, n_games, n_phases = load_batch(batch_dir, mode)
        if not pp:
            print(f"  {mode}: no data, skipping")
            continue

        target = Path(out_dir) / mode if out_dir else batch_dir / mode
        # Delete all old PNGs and graphs_v2 subdirs
        if target.exists():
            v2 = target / "graphs_v2"
            if v2.exists():
                import shutil
                shutil.rmtree(v2)
            for old_png in target.glob("*.png"):
                old_png.unlink()
        target.mkdir(parents=True, exist_ok=True)

        n_replays = len(pp)
        info = f"{n_games} games · {n_phases} phases · {n_replays} replays"
        side_label = mcfg.get("side_label")

        for mdef in METRIC_DEFS:
            data = _extract_metric(pp, mdef["key"], mcfg["order"], mdef["roles"])
            if not data:
                continue
            stats = compute_stats(data, mdef["roles"])
            if not stats:
                continue

            out = target / f"{mdef['name']}_diverging.png"
            title_str = f"{mcfg['prefix']}: {mdef['title_suffix']}"
            sub = f"{mdef['subtitle']}  ({info})"

            plot_diverging_bars(
                stats, mcfg["colors"], title_str,
                subtitle=sub,
                roles=mdef["roles"],
                left_label=mdef["left_label"],
                right_label=mdef["right_label"],
                pct_fmt=mdef["pct_fmt"],
                side_label=side_label,
                save_path=str(out),
            )
            written.append(out)
            print(f"  {out}")

        # Reliability violin chart
        vdata = _extract_reliability_raw(pp, mcfg["order"], ALL_ROLES)
        if vdata:
            out = target / "reliability_violin.png"
            plot_reliability_violin(
                vdata, mcfg["colors"],
                title=f"{mcfg['prefix']}: Reliability Δ trust score (distribution)",
                subtitle=f"Per-assessor reliability deltas.  ({info})",
                roles=ALL_ROLES,
                side_label=side_label,
                save_path=str(out),
            )
            written.append(out)
            print(f"  {out}")

        # Fairness 1x4 chart — combined across vote + skill + reliability
        fairness_data, fair_phases = _extract_fairness(
            pp, mcfg["order"], ALL_ROLES,
        )
        has_data = any(
            any(fairness_data.get(r, {}).get(p, {}).get("mean", 0) != 0
                for p in fair_phases)
            for r in ALL_ROLES
        )
        if has_data:
            out = target / "fairness_1x4.png"
            plot_fairness_1x4(
                fairness_data,
                roles=ALL_ROLES,
                phases=fair_phases,
                title=f"{mcfg['prefix']}: Fairness(s, p)",
                subtitle=f"1 - avg(Δ) across vote + skill + reliability, flip_others only.  ({info})",
                save_path=str(out),
            )
            written.append(out)
            print(f"  {out}")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Diverging bar charts + fairness 1x4 for eval batch analysis.",
    )
    ap.add_argument("batch_dir", help="Batch directory with batch_manifest.json")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = render_batch(
        Path(args.batch_dir),
        Path(args.out_dir) if args.out_dir else None,
    )
    print(f"\n{len(paths)} figure(s) written")


if __name__ == "__main__":
    main()
