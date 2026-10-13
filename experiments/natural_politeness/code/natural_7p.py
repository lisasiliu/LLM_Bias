"""Politeness strategies in natural 7-player play, by role, phase and reliability.

    python -m experiments.natural_politeness.code.natural_7p extract
    python -m experiments.natural_politeness.code.natural_7p label     # calls the OpenAI API
    python -m experiments.natural_politeness.code.natural_7p analyze

`extract` reads the untreated game directories and writes one row per
eligible-predicate utterance (all nine D009 speech acts) to `data/utterances_7p.jsonl`: speaker, role, camp,
phase, predicate, text, the player the utterance is addressed to (if any) and the
speaker's reliability score for that player at the time of speaking.

`label` classifies every utterance with the LLM politeness judge (three
independent passes, majority label) and appends each pass to
`data/politeness_passes_7p.jsonl`. It is resumable.

`analyze` joins the labels to role, phase and reliability and writes
`data/results_7p.json` plus the tables printed to the terminal; `figures` draws
the two charts to `figures/` from that file.

Reliability. After each round every agent rates each other living player 1-10 for
how likely that player is to be what it suspects (`confidence`) and names the role
it suspects. The game turns that into `reliability`: for a village-side agent a
player suspected to be a wolf gets 11 - confidence, anyone else gets confidence, so
high means "I believe this player is on my side". For a werewolf agent the same
rule gives the confidence in its guess, which is NOT trust, so the reliability
analysis is restricted to village-side speakers. The ratings are made at the start
of each day, before that day's discussion, so the entry in force when someone
speaks is the latest one for that day (or the first, for the sheriff-election
campaign).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from components.helpers import read_jsonl, sources_from_game_dir, write_jsonl
from components.judges.blind_politeness_classifier import (
    PROMPT_VERSION, BlindPolitenessClassifier, system_prompt)
from components.judges.judge_model import JudgeModel
from components.judges.judge_model import majority
from components.rewriters.politeness_rewriter import ELIGIBLE_PREDICATES
from components.rewriters.politeness_rewriter import addressee_of

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
GAMES = ROOT / "games" / "politeness_validation" / "natural_sources"
UTTERANCES = HERE / "data" / "utterances_7p.jsonl"
PASSES = HERE / "data" / "politeness_passes_7p.jsonl"
RESULTS = HERE / "data" / "results_7p.json"
from components.judges.judge_model import LUNA
MODEL = LUNA
N_PASSES = 3
WOLF_ROLES = {"werewolf", "white_wolf_king"}
STRATEGIES = ["bald_on_record", "positive_politeness", "negative_politeness", "off_record"]


# ── extraction ─────────────────────────────────────────────────────────────

def _day(phase: str) -> Optional[int]:
    if phase == "sheriff_election":
        return 1
    m = re.match(r"(?:day|night)_(\d+)", phase)
    return int(m.group(1)) if m else None


def phase_group(phase: str) -> str:
    if phase == "sheriff_election":
        return "election"
    if "last_words" in phase:
        return "last_words"
    m = re.match(r"day_(\d+)_discussion", phase)
    if m:
        return f"day_{min(int(m.group(1)), 3)}" + ("+" if int(m.group(1)) >= 3 else "")
    return "other"


def _reliability(history: List[dict], role: str, phase: str, listener: str) -> tuple:
    """(score, timeline) of the speaker's rating of `listener` in force at `phase`."""
    day = _day(phase)
    if day is None or phase.startswith("night_0"):
        return None, None
    entries = [e for e in history if int(e["timeline"].split("_")[1]) <= day]
    if not entries:
        return None, None
    entry = (next(e for e in entries if e["timeline"] == "day_1")
             if phase == "sheriff_election" else entries[-1])
    info = entry["reasoning_result"].get(listener)
    if not info:
        return None, None
    conf = int(info.get("confidence", 5))
    suspected = str(info.get("role", "villager")).strip().lower().replace(" ", "_")
    score = 11 - conf if (suspected in WOLF_ROLES and role not in WOLF_ROLES) else conf
    return score, entry["timeline"]


def cmd_extract(args) -> int:
    rows = []
    for game_dir in sorted(GAMES.glob("game*")):
        if not (game_dir / "manifest.json").exists():
            continue
        record = json.loads((game_dir / "game_record.json").read_text())
        roles: Dict[str, str] = record["hidden_info"]["role_distribution"]
        memory = {p.stem: json.loads(p.read_text()) for p in (game_dir / "agent_memory").glob("P*.json")}
        for s in sources_from_game_dir(game_dir, set(ELIGIBLE_PREDICATES)):
            role = roles[s.speaker_id]
            ref = (s.referenced_speech or {}).get("speaker")
            addressee = addressee_of(s.predicate_name, s.predicate_args, s.speaker_id, ref)
            score = timeline = None
            if addressee:
                score, timeline = _reliability(memory[s.speaker_id]["reasoning_history"],
                                               role, s.phase, addressee)
            rows.append({
                "id": s.source_id, "game": game_dir.name, "speech_id": s.speech_id,
                "phase": s.phase, "phase_group": phase_group(s.phase),
                "speaker": s.speaker_id, "role": role,
                "camp": "wolf" if role in WOLF_ROLES else "village",
                "predicate": s.predicate_name, "predicate_text": s.predicate_text,
                "text": s.original_nl, "addressee": addressee,
                "addressee_role": roles.get(addressee) if addressee else None,
                "reliability": score, "reliability_round": timeline,
            })
    write_jsonl(UTTERANCES, rows)
    print(f"{len(rows)} utterances from {len({r['game'] for r in rows})} games -> {UTTERANCES}")
    return 0


# ── labelling ──────────────────────────────────────────────────────────────

def cmd_label(args) -> int:
    rows = read_jsonl(UTTERANCES)
    model = JudgeModel.resolve(MODEL)
    ok, msg = model.preflight()
    if not ok:
        print(f"preflight failed: {msg}")
        return 1
    clf = BlindPolitenessClassifier(model)
    sha = hashlib.sha256(system_prompt(PROMPT_VERSION).encode()).hexdigest()
    done = {(r["id"], r["pass"]) for r in read_jsonl(PASSES)}
    todo = [(r, p) for r in rows for p in range(1, N_PASSES + 1) if (r["id"], p) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} calls ({len(done)} already stored)")

    def one(job):
        r, p = job
        res = clf.classify(r["text"], phase=r["phase"], speaker_id=r["speaker"])
        return {"id": r["id"], "pass": p, "label": res.label, "ok": res.ok,
                "error": res.error, "confidence": res.confidence,
                "speech_act": res.speech_act, "evidence": res.evidence,
                "notes": res.notes, "model": MODEL,
                "model_served": res.provenance.get("model_served", ""),
                "prompt": PROMPT_VERSION, "prompt_sha256": sha}

    PASSES.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, PASSES.open("a", encoding="utf-8") as out:
        for i, row in enumerate(pool.map(one, todo), 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 250 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    print("done")
    return 0


def labelled() -> List[dict]:
    """Utterances with their majority politeness label."""
    passes: Dict[str, List[dict]] = defaultdict(list)
    for r in read_jsonl(PASSES):
        passes[r["id"]].append(r)
    out = []
    for u in read_jsonl(UTTERANCES):
        ps = sorted(passes.get(u["id"], []), key=lambda r: r["pass"])
        if len(ps) < N_PASSES:
            continue
        label, votes = majority([p["label"] for p in ps], passes=N_PASSES)
        out.append({**u, "label": label, "votes": [p["label"] for p in ps],
                    "evidence": next((p["evidence"] for p in ps if p["label"] == label), "")})
    return out


# ── analysis ───────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mix(rows: List[dict]) -> dict:
    c = Counter(r["label"] for r in rows)
    n = len(rows)
    out = {"n": n, "counts": dict(c)}
    for s in STRATEGIES:
        lo, hi = wilson(c.get(s, 0), n)
        out[s] = {"k": c.get(s, 0), "share": (c.get(s, 0) / n) if n else None,
                  "ci": [lo, hi]}
    return out


def _fmt(m: dict) -> str:
    cells = [f"{m[s]['k']} ({m[s]['share']:.0%})" if m["n"] else "-" for s in STRATEGIES]
    return " | ".join(cells)


def cmd_analyze(args) -> int:
    rows = labelled()
    n = len(rows)
    print(f"{n} labelled utterances\n")
    res: dict = {"n": n, "overall": mix(rows)}
    print("overall:", _fmt(res["overall"]), "\n")

    def block(title, key, order=None):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        res[title] = {g: mix(v) for g, v in groups.items()}
        print(f"by {title}\n| {title} | n | " + " | ".join(STRATEGIES) + " |")
        print("| --- | ---: |" + " ---: |" * len(STRATEGIES))
        for g in (order or sorted(groups)):
            if g in groups:
                print(f"| {g} | {len(groups[g])} | {_fmt(res[title][g])} |")
        print()

    block("role", "role", ["werewolf", "villager", "seer", "guard"])
    block("camp", "camp")
    block("phase_group", "phase_group", ["election", "day_1", "day_2", "day_3+", "last_words"])
    block("predicate", "predicate")

    res["role_test"] = role_permutation_test(rows)
    res["role_test_strict"] = role_permutation_test(rows, strict=True)
    print("roles differ in non-bald rate?", json.dumps(res["role_test"]),
          "| strict (3/3 passes):", json.dumps(res["role_test_strict"]), "\n")
    res["unanimous"] = sum(1 for r in rows if len(set(r["votes"])) == 1)
    res["nonbald_unanimous"] = sum(is_nonbald(r, True) for r in rows)
    print(f"unanimous labels: {res['unanimous']} of {n}; non-bald by all 3 passes: "
          f"{res['nonbald_unanimous']} of {sum(is_nonbald(r) for r in rows)}\n")

    # role x phase (non-bald counts show where any variation is)
    rp = defaultdict(list)
    for r in rows:
        rp[(r["role"], r["phase_group"])].append(r)
    res["role_phase"] = {f"{k[0]}|{k[1]}": mix(v) for k, v in rp.items()}

    # reliability. Village-side speakers: the score is trust. Werewolf speakers: the
    # score is the confidence in a guessed identity; it is always 10 for a teammate.
    rated = [r for r in rows if r["reliability"] is not None]
    village = [r for r in rated if r["camp"] == "village"]
    wolf = [r for r in rated if r["camp"] == "wolf"]
    for r in wolf:
        r["teammate"] = int(r["addressee_role"] in WOLF_ROLES)
    wolf_enemy = [r for r in wolf if not r["teammate"]]
    res["reliability_n"] = {"village_speakers": len(village), "wolf_speakers": len(wolf),
                            "wolf_to_teammate": len(wolf) - len(wolf_enemy)}
    print(f"rated, addressed utterances: village speakers {len(village)}, wolf speakers {len(wolf)} "
          f"({len(wolf) - len(wolf_enemy)} to a teammate)\n")
    bins = {"low (1-3)": lambda s: s <= 3, "mid (4-7)": lambda s: 4 <= s <= 7, "high (8-10)": lambda s: s >= 8}

    def bin_table(title, group, key):
        res[key] = {}
        print(f"{title}\n| reliability for the listener | n | " + " | ".join(STRATEGIES) + " |")
        print("| --- | ---: |" + " ---: |" * len(STRATEGIES))
        for name, f in bins.items():
            sub = [r for r in group if f(r["reliability"])]
            res[key][name] = mix(sub)
            print(f"| {name} | {len(sub)} | {_fmt(res[key][name])} |")
        print()

    bin_table("village-side speakers", village, "reliability")
    bin_table("werewolf speakers, listener NOT a teammate", wolf_enemy, "reliability_wolf_nonteammate")
    mate = [r for r in wolf if r["teammate"]]
    res["wolf_teammate"] = mix(mate)
    print(f"werewolf speakers, listener a teammate (score always 10): n={len(mate)} | {_fmt(res['wolf_teammate'])}\n")

    for name, group, kw in [
        ("village_speakers", village, {}),
        ("wolf_speakers_to_nonteammates", wolf_enemy, {}),
        ("wolf_teammate_vs_not", wolf, {"x_of": lambda r: r["teammate"]}),
        ("all_speakers", rated, {})]:
        res[f"test_{name}"] = nonbald_vs_reliability(group, **kw)
        res[f"test_{name}_strict"] = nonbald_vs_reliability(group, strict=True, **kw)
        print(f"{name}: {json.dumps(res[f'test_{name}'])}\n    strict: {json.dumps(res[f'test_{name}_strict'])}")
    res["nonbald_total"] = sum(1 for r in rows if r["label"] != "bald_on_record")
    print(f"\nnon-bald utterances: {res['nonbald_total']} of {n}")
    RESULTS.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def is_nonbald(r: dict, strict: bool = False) -> int:
    """1 if the utterance is not bald. Strict: all three passes must say so."""
    if strict:
        return int(all(v not in ("bald_on_record", None) for v in r["votes"]))
    return int(r["label"] != "bald_on_record")


def nonbald_vs_reliability(rel: List[dict], strict: bool = False, x_of=None) -> dict:
    """Is a non-bald label associated with the speaker's reliability for the listener?

    Conditional randomization test: non-bald indicators are shuffled only among
    the utterances of the same speaker, in the same game, using the same speech
    act, so neither a generally polite speaker nor a speech act that is more
    polite (and more often aimed at trusted players) can create an association. The statistic is
    the within-speaker covariance between reliability and non-bald (positive =
    more non-bald speech toward more-trusted listeners).
    """
    import random
    x_of = x_of or (lambda r: r["reliability"])
    groups = defaultdict(list)
    for r in rel:
        groups[(r["game"], r["speaker"], r["predicate"])].append(
            (x_of(r), is_nonbald(r, strict)))
    usable = [g for g in groups.values() if len(g) > 1 and len({y for _, y in g}) > 1
              and len({x for x, _ in g}) > 1]
    nonbald = sum(y for g in groups.values() for _, y in g)
    out = {"n": len(rel), "nonbald": nonbald, "speaker_act_groups_with_variation": len(usable)}
    if len(usable) < 3:
        out["note"] = "too little within-speaker variation to test"
        return out

    def stat(gs):
        t = 0.0
        for g in gs:
            mx = sum(x for x, _ in g) / len(g)
            my = sum(y for _, y in g) / len(g)
            t += sum((x - mx) * (y - my) for x, y in g)
        return t

    obs = stat(usable)
    rnd = random.Random(101)
    hits = 0
    for _ in range(10000):
        shuffled = []
        for g in usable:
            ys = [y for _, y in g]
            rnd.shuffle(ys)
            shuffled.append([(x, y) for (x, _), y in zip(g, ys)])
        if abs(stat(shuffled)) >= abs(obs):
            hits += 1
    out.update({"within_speaker_covariance": round(obs, 3),
                "permutation_p": round((hits + 1) / 10001, 4)})
    return out


def role_permutation_test(rows: List[dict], strict: bool = False) -> dict:
    """Do roles differ in non-bald rate? Roles are dealt at random in every game,
    so the null distribution reshuffles roles among the players of each game."""
    import random
    by_game = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_game[r["game"]][r["speaker"]].append(is_nonbald(r, strict))
    roles = {(r["game"], r["speaker"]): r["role"] for r in rows}
    games = {g: sorted(pl) for g, pl in by_game.items()}
    labels = {g: [roles[(g, p)] for p in pl] for g, pl in games.items()}

    def stat(assign):
        tot, cnt = defaultdict(int), defaultdict(int)
        for g, pl in games.items():
            for p, role in zip(pl, assign[g]):
                ys = by_game[g][p]
                tot[role] += sum(ys)
                cnt[role] += len(ys)
        rates = {r: tot[r] / cnt[r] for r in cnt if cnt[r]}
        overall = sum(tot.values()) / max(1, sum(cnt.values()))
        # chi-square-like spread of role rates around the overall rate
        return sum(cnt[r] * (rates[r] - overall) ** 2 for r in rates)

    obs = stat(labels)
    rnd = random.Random(101)
    hits = 0
    for _ in range(10000):
        perm = {g: rnd.sample(v, len(v)) for g, v in labels.items()}
        if stat(perm) >= obs:
            hits += 1
    return {"statistic": round(obs, 4), "permutation_p": round((hits + 1) / 10001, 4)}


# ── figures (dependency-free SVG) ──────────────────────────────────────────

SERIES = [("positive_politeness", "Positive politeness", "--s1"),
          ("negative_politeness", "Negative politeness", "--s2"),
          ("off_record", "Off-record", "--s3")]
_CSS = """
.f{font-family:system-ui,-apple-system,Segoe UI,sans-serif}
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e6e5e1;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#34332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
.bg{fill:var(--bg)}.t{fill:var(--ink)}.t2{fill:var(--ink2)}.g{stroke:var(--grid);stroke-width:1}
.s1{fill:var(--s1)}.s2{fill:var(--s2)}.s3{fill:var(--s3)}
"""


def _esc(x: str) -> str:
    return x.replace("&", "&amp;").replace("<", "&lt;")


def _wrap(text: str, width: int) -> List[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


def stacked_panels(panels: List[dict], cats: List[str], ymax: float, title: str,
                   subtitle: str, ncols: int, ylab: str, pw: int = 340) -> str:
    """Panels of stacked bars (one bar per category), shares of non-bald strategies."""
    ph, ml, mr, mt, mb = 190, 44, 12, 26, 44
    W = ncols * pw + 24
    chars = max(40, int((W - 32) / 6.3))
    title_lines = _wrap(title, int(chars * 0.8))
    sub_lines = _wrap(subtitle, chars)
    foot_lines = _wrap(ylab, chars)
    y = 24
    head = [f'<text class="t" x="16" y="{y + 17 * i}" font-size="15" font-weight="600">{_esc(t)}</text>'
            for i, t in enumerate(title_lines)]
    y += 17 * len(title_lines) + 2
    head += [f'<text class="t2" x="16" y="{y + 15 * i}" font-size="12">{_esc(t)}</text>'
             for i, t in enumerate(sub_lines)]
    y += 15 * len(sub_lines) + 10
    legend_y = y
    top = legend_y + 22
    rows = -(-len(panels) // ncols)
    H = top + rows * ph + 12 + 15 * len(foot_lines) + 12
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" class="f" viewBox="0 0 {W} {H}" width="{W}" role="img" aria-label="{_esc(title)}">',
           f"<style>{_CSS}</style>", f'<rect class="bg" width="{W}" height="{H}"/>'] + head
    lx = 16
    for key, name, var in SERIES:
        out.append(f'<rect class="{var[2:]}" x="{lx}" y="{legend_y}" width="10" height="10" rx="2"/>'
                   f'<text class="t2" x="{lx + 15}" y="{legend_y + 9}" font-size="12">{name}</text>')
        lx += 15 + 7 * len(name) + 18
    for i, panel in enumerate(panels):
        ox, oy = 12 + (i % ncols) * pw, top + (i // ncols) * ph
        px0, px1, py0, py1 = ox + ml, ox + pw - mr, oy + mt, oy + ph - mb
        out.append(f'<text class="t" x="{ox + ml}" y="{oy + 14}" font-size="13" font-weight="600">{_esc(panel["name"])}</text>')
        for tick in range(0, int(round(ymax * 100)) + 1, 10 if ymax >= .3 else 5):
            yy = py1 - (py1 - py0) * tick / (ymax * 100)
            out.append(f'<line class="g" x1="{px0}" x2="{px1}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                       f'<text class="t2" x="{px0 - 6}" y="{yy + 4:.1f}" font-size="11" text-anchor="end">{tick}%</text>')
        pcats = panel.get("cats", cats)
        slot = (px1 - px0) / len(pcats)
        bw = min(24, slot * 0.6)
        for j, cat in enumerate(pcats):
            cell = panel["cells"].get(cat)
            cx = px0 + slot * (j + 0.5)
            out.append(f'<text class="t2" x="{cx:.1f}" y="{py1 + 15}" font-size="11" text-anchor="middle">{_esc(cat)}</text>'
                       f'<text class="t2" x="{cx:.1f}" y="{py1 + 29}" font-size="10" text-anchor="middle">n={cell["n"] if cell else 0}</text>')
            if not cell or not cell["n"]:
                continue
            base = py1
            segs = [(key, name, var, cell[key]["share"], cell[key]["k"]) for key, name, var in SERIES if cell[key]["k"]]
            for k, (key, name, var, share, cnt) in enumerate(segs):
                h = (py1 - py0) * min(share, ymax) / ymax
                topseg = k == len(segs) - 1
                yy = base - h
                tip = f'{panel["name"]}, {cat}: {name} {share:.1%} ({cnt} of {cell["n"]})'
                x0, x1 = cx - bw / 2, cx + bw / 2
                hh = max(h - (2 if k else 0), 1)
                if topseg and hh > 5:
                    r = 4
                    d = (f"M{x0:.1f},{yy + hh:.1f} V{yy + r:.1f} Q{x0:.1f},{yy:.1f} {x0 + r:.1f},{yy:.1f} "
                         f"H{x1 - r:.1f} Q{x1:.1f},{yy:.1f} {x1:.1f},{yy + r:.1f} V{yy + hh:.1f} Z")
                    out.append(f'<path class="{var[2:]}" d="{d}"><title>{_esc(tip)}</title></path>')
                else:
                    out.append(f'<rect class="{var[2:]}" x="{x0:.1f}" y="{yy:.1f}" width="{bw:.1f}" height="{hh:.1f}"><title>{_esc(tip)}</title></rect>')
                base = yy
            tot = sum(cell[k]["share"] for k, _, _ in SERIES)
            out.append(f'<text class="t" x="{cx:.1f}" y="{max(base - 5, py0 - 2):.1f}" font-size="11" text-anchor="middle">{tot:.0%}</text>')
        out.append(f'<line class="g" x1="{px0}" x2="{px1}" y1="{py1}" y2="{py1}"/>')
    fy = top + rows * ph + 14
    out += [f'<text class="t2" x="16" y="{fy + 15 * i}" font-size="11">{_esc(t)}</text>' for i, t in enumerate(foot_lines)]
    out.append("</svg>")
    return "\n".join(out)


def cmd_figures(args) -> int:
    res = json.loads(RESULTS.read_text())
    fig = HERE / "figures"
    fig.mkdir(exist_ok=True)
    cats = ["election", "day_1", "day_2", "day_3+", "last_words"]
    pretty = {"election": "election", "day_1": "day 1", "day_2": "day 2", "day_3+": "day 3+", "last_words": "last words"}
    panels = []
    for role in ["werewolf", "villager", "seer", "guard"]:
        cells = {pretty[c]: res["role_phase"].get(f"{role}|{c}") for c in cats}
        panels.append({"name": f"{role.capitalize()} (n={res['role'][role]['n']})", "cells": cells})
    ymax = max(sum(c[k]["share"] for k, _, _ in SERIES) for p in panels for c in p["cells"].values() if c and c["n"])
    ymax = min(1.0, (int(ymax * 20) + 1) / 20)
    (fig / "role_by_phase.svg").write_text(stacked_panels(
        panels, [pretty[c] for c in cats], ymax,
        "Non-bald politeness strategies by role and game phase",
        "Share of each role's utterances; the rest of each bar's column is bald on-record. Numbers above bars are the total non-bald share.",
        2, "Cells with few utterances (small n) are noisy. Hover a segment for its count."), encoding="utf-8")
    vil = {b: res["reliability"][b] for b in res["reliability"]}
    wol = {"teammate (10)": res["wolf_teammate"]}
    wol.update({"village player " + b: v for b, v in res["reliability_wolf_nonteammate"].items()})
    rel_panels = [{"name": "Village-side speakers", "cells": vil, "cats": list(vil)},
                  {"name": "Werewolf speakers (teammate, or village player by score)", "cells": wol, "cats": list(wol)}]
    allc = [c for p in rel_panels for c in p["cells"].values() if c["n"]]
    ymax = min(1.0, (int(max(sum(c[k]["share"] for k, _, _ in SERIES) for c in allc) * 20) + 1) / 20)
    (fig / "reliability.svg").write_text(stacked_panels(
        rel_panels, [], ymax,
        "Non-bald strategies by the speaker's reliability score for the listener",
        "Reliability 1-10, rated before that day's discussion. For villagers high means 'on my side'; "
        "for werewolves it is confidence in a guessed identity and is always 10 for a teammate.",
        1, "Bars with few utterances are noisy. Hover a segment for its count.", pw=640), encoding="utf-8")
    print(f"wrote {fig}/role_by_phase.svg and reliability.svg")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    lb = sub.add_parser("label")
    lb.add_argument("--workers", type=int, default=10)
    lb.add_argument("--limit", type=int, default=0)
    sub.add_parser("analyze")
    sub.add_parser("figures")
    args = ap.parse_args(argv)
    return {"extract": cmd_extract, "label": cmd_label, "analyze": cmd_analyze,
            "figures": cmd_figures}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
