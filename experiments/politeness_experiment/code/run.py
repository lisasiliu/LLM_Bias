"""Politeness flip experiment: play baselines and run one-day flip replays (resumable).

    python -m experiments.politeness_experiment.code.run play --seed pe21
    python -m experiments.politeness_experiment.code.run flip --first 1 --last 2 --workers 6

`play` plays a natural baseline game and saves its checkpoints. `flip` runs the flip_self replays
(research/POLITENESS_FLIP_PLAN.md): for each baseline day-start checkpoint and each living player, one
replay per setting (all_bald, all_positive, trust_positive, trust_bald) in which that player's eligible
speech is delivered in the setting, plus a control replay in which the natural text is kept. Every replay
repeats the baseline's model responses up to the changed line (the engine's replay cache) and draws fresh
ones from there. flip_others is available with --modes flip_others.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from components.judges.judge_model import LUNA, JudgeModel, read_key
from components.judges.gate import AdmissionGate
from components.rewriters.politeness_rewriter import ELIGIBLE_PREDICATES
from components.rewriters.politeness_rewriter import TrustPolitenessPolicy

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAMES = ROOT / "games" / "flip_experiments"      # baselines at <seed>/natural, replays at <seed>/day<N>/
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
HERE = Path(__file__).resolve().parents[1]
RUN_LOG = HERE / "data" / "run_log.jsonl"
BUDGET_USD = 40.0

# USD per 1M tokens, deepseek-flash, off-peak. Peak is double.
PRICE = {"hit": 0.003, "miss": 0.15, "out": 0.60}


def is_peak(ts: float) -> bool:
    t = datetime.fromtimestamp(ts, tz=timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


class Ledger:
    """Provider-reported token use and estimated cost, by category and by model."""

    def __init__(self):
        self._lock = threading.Lock()
        self.cat = defaultdict(lambda: Counter())
        self.models = Counter()

    def add(self, category: str, model_served: str, usage: dict) -> None:
        now = time.time()
        hit = int(usage.get("prompt_cache_hit_tokens") or 0)
        prompt = int(usage.get("prompt_tokens") or 0)
        miss = int(usage.get("prompt_cache_miss_tokens") or max(prompt - hit, 0))
        out = int(usage.get("completion_tokens") or 0)
        mult = 2.0 if is_peak(now) else 1.0
        cost = mult * (hit * PRICE["hit"] + miss * PRICE["miss"] + out * PRICE["out"]) / 1e6
        with self._lock:
            c = self.cat[category]
            c["calls"] += 1
            c["peak_calls"] += int(mult > 1)
            c["hit_tokens"] += hit
            c["miss_tokens"] += miss
            c["output_tokens"] += out
            c["cost_micro_usd"] += int(round(cost * 1e6))
            self.models[model_served] += 1

    def to_dict(self) -> dict:
        cats = {k: {**v, "cost_usd": round(v["cost_micro_usd"] / 1e6, 4)} for k, v in self.cat.items()}
        for v in cats.values():
            v.pop("cost_micro_usd", None)
        return {"by_category": cats, "models_returned": dict(self.models),
                "total_cost_usd": round(sum(v["cost_usd"] for v in cats.values()), 4)}


class _Msg:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content): self.message = _Msg(content)
class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, ledger: Ledger, category: str, key: str):
        self.ledger, self.category, self.key = ledger, category, key

    def create(self, *, model, messages, temperature=0.0, max_tokens=300, **kwargs):
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        last = None
        for attempt in range(6):
            try:
                r = requests.post(f"{BASE_URL}/v1/chat/completions",
                                  headers={"Authorization": f"Bearer {self.key}"},
                                  json=payload, timeout=300)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}")
                r.raise_for_status()
                data = r.json()
                self.ledger.add(self.category, str(data.get("model", "")), data.get("usage") or {})
                return _Resp(data["choices"][0]["message"]["content"])
            except (requests.RequestException, KeyError) as exc:
                last = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"API call failed after retries: {last}")



class LedgerClient:
    """OpenAI-style client (`client.chat.completions.create`) that records usage."""

    def __init__(self, ledger: Ledger, category: str):
        key = read_key(("DEEPSEEK_API_KEY", "DEEPSEEK"))
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(ledger, category, key)


def balance() -> Optional[float]:
    try:
        r = requests.get(f"{BASE_URL}/user/balance", timeout=30,
                         headers={"Authorization": "Bearer " + read_key(("DEEPSEEK_API_KEY", "DEEPSEEK"))})
        return float(r.json()["balance_infos"][0]["total_balance"])
    except Exception:
        return None


def play(seed: str, games_root: Path = DEFAULT_GAMES, dry_run: bool = False) -> Path:
    """Play the natural baseline for `seed`, saving the checkpoints the replays start from."""
    from lurkingwolf.run_game import run_single_game, save_game_output, load_prompts
    out = games_root / seed / "natural"
    if (out / "manifest.json").exists():
        return out
    if out.exists():
        shutil.rmtree(out)                      # an unfinished game: start it again
    out.mkdir(parents=True)
    ledger = Ledger()
    players = None if dry_run else LedgerClient(ledger, "players")
    result = run_single_game(game_id="g1", model=MODEL, client=players, prompts=load_prompts(), num_players=7,
                             temperature=0, max_tokens=300, max_rounds=20, dry_run=dry_run,
                             experiment_type="nl", seed=seed, checkpoint_dir=out / "checkpoints")
    save_game_output(result, out, terminal_output="")
    (out / "cost.json").write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def cmd_play(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_GAMES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(f"done: {play(a.seed, a.root, a.dry_run)}")
    return 0


def seed_name(n: int) -> str:
    return f"pe{n:02d}"


def log(event: str, **kw) -> None:
    if kw.pop("_dry", False):
        return
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def estimated_spend(root: Path) -> float:
    return sum(json.loads(p.read_text())["total_cost_usd"] for p in root.glob("pe*/*/cost.json"))


# ── flip replays ──────────────────────────────────────────────────────────

FLIP_ROOT = DEFAULT_GAMES
SETTINGS = ["all_bald", "all_positive", "trust_positive", "trust_bald"]
SCOPES = {"flip_self": "self", "flip_others": "others"}


def flip_dir(root: Path, seed: str, day: int, mode: str, setting: str, player: str, rep: int = 0) -> Path:
    name = f"control__{player}" if mode == "control" else f"{mode}__{setting}__{player}"
    if rep:
        name += f"__r{rep}"
    return root / seed / f"day{day}" / name


def flip_replay(seed: str, day: int, mode: str, setting: str, player: str,
                root: Path = FLIP_ROOT, games_root: Path = DEFAULT_GAMES, dry_run: bool = False,
                rep: int = 0) -> Path:
    """One replay of day `day` from the baseline's day-start checkpoint. mode is flip_self,
    flip_others or control (the self player's natural text kept, branching at the same line a
    flip_self replay would). The baseline's game log answers model calls up to that line."""
    from lurkingwolf.checkpointing import load_checkpoint
    from lurkingwolf.eval.replay import replay_phase
    from lurkingwolf.run_game import load_prompts

    out = flip_dir(root, seed, day, mode, setting, player, rep)
    if (out / "result.json").exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    baseline = games_root / seed / "natural"
    ckpt_path = baseline / "checkpoints" / f"day_{day}_start.pkl"
    prompts = load_prompts()
    ckpt, _, _ = load_checkpoint(ckpt_path, prompts, None)
    before = {pid: dict(a["reliability"]) for pid, a in ckpt.agents.items()}   # ratings in force at the checkpoint

    ledger = Ledger()
    players = None if dry_run else LedgerClient(ledger, "players")
    gate = None
    if not dry_run:
        judge = JudgeModel.resolve(LUNA)      # not the model that writes the rewrites
        judge.on_usage = lambda m, d: ledger.add("checks", str(d.get("model", "")), d.get("usage") or {})
        gate = AdmissionGate(judge, cache_path=games_root / "evidence_reasons.jsonl")
    policy = None
    if mode != "control":
        writer = None if dry_run else LedgerClient(ledger, "rewriter")
        policy = TrustPolitenessPolicy(
            llm_client=writer, arm=setting, seed=f"{seed}|d{day}|{mode}|{player}", model=MODEL,
            gate=gate, dry_run=dry_run, log_path=out / "touched.jsonl")
    started = time.time()
    result, _agents, state = replay_phase(
        ckpt_path, prompts, players, experiment_type="nl", config_name=out.name,
        self_player=player, rewriter=policy, rewriter_scope=SCOPES.get(mode, "self"),
        baseline_log=baseline / "game_log.jsonl", eligible=ELIGIBLE_PREDICATES, dry_run=dry_run)
    if policy is not None and policy.counters.get("gate_unavailable"):
        raise RuntimeError(f"the checking model was unavailable for {policy.counters['gate_unavailable']} rewrites; "
                           f"replay not saved (rerun it when the judge is back)")
    row = result.to_dict()
    row.update(seed=seed, mode=mode, setting=setting, player=player, day=day, rep=rep,
               reliability_before=before, alive_at_end=sorted(state.get_alive()),
               policy_counters=policy.counters if policy else None,
               elapsed_seconds=round(time.time() - started, 1), dry_run=dry_run)
    (out / "cost.json").write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return out


def cmd_replay(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=sorted(SCOPES) + ["control"])
    ap.add_argument("--setting", default="none", choices=SETTINGS + ["none"])
    ap.add_argument("--player", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--games-root", type=Path, default=DEFAULT_GAMES)
    ap.add_argument("--root", type=Path, default=FLIP_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(f"done: {flip_replay(a.seed, a.day, a.mode, a.setting, a.player, a.root, a.games_root, dry_run=a.dry_run, rep=a.rep)}")
    return 0


def flip_tasks(games_root: Path, seed: str, days=None, players=None, settings=None, modes=None, reps: int = 1) -> list:
    """Every replay for one seed: for each day with a checkpoint and each living player, the control
    replay (with flip_self) and one replay per mode and setting."""
    from lurkingwolf.checkpointing import load_checkpoint
    from lurkingwolf.run_game import load_prompts
    tasks = []
    modes = list(modes or ["flip_self"])
    for ck in sorted((games_root / seed / "natural" / "checkpoints").glob("day_*_start.pkl")):
        day = int(ck.stem.split("_")[1])
        if days and day not in days:
            continue
        ckpt, _, _ = load_checkpoint(ck, load_prompts(), None)
        alive = sorted(ckpt.state.get_alive(), key=lambda p: int(p[1:]))
        for p in alive:
            if players and p not in players:
                continue
            for r in range(reps):
                if "flip_self" in modes:
                    tasks.append((seed, day, "control", "none", p, r))
                tasks += [(seed, day, mode, setting, p, r) for mode in modes for setting in (settings or SETTINGS)]
    return tasks


def flip_spend(root: Path) -> float:
    return sum(json.loads(p.read_text())["total_cost_usd"] for p in root.glob("pe*/day*/*/cost.json"))


def cmd_flip(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget", type=float, default=BUDGET_USD)
    ap.add_argument("--root", type=Path, default=FLIP_ROOT)
    ap.add_argument("--games-root", type=Path, default=DEFAULT_GAMES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--days", default="", help="comma-separated days (default: all)")
    ap.add_argument("--players", default="", help="comma-separated self players (default: all alive)")
    ap.add_argument("--settings", default="", help="comma-separated settings (default: the four)")
    ap.add_argument("--modes", default="", help="comma-separated modes (default: flip_self)")
    ap.add_argument("--reps", type=int, default=1, help="replays per (day, player, setting), the same for every setting and the control")
    a = ap.parse_args(argv)
    pick = lambda v, f=str: [f(x) for x in v.split(",") if x] or None
    tasks = [t for n in range(a.first, a.last + 1)
             for t in flip_tasks(a.games_root, seed_name(n), pick(a.days, int), pick(a.players),
                                 pick(a.settings), pick(a.modes), a.reps)]
    log("flip_start", first=a.first, last=a.last, tasks=len(tasks), reps=a.reps, balance=balance(), _dry=a.dry_run)

    def run_one(t) -> str:
        seed, day, mode, setting, player, rep = t
        out = flip_dir(a.root, seed, day, mode, setting, player, rep)
        if (out / "result.json").exists():
            return "done"
        if estimated_spend(a.games_root) + flip_spend(a.root) >= a.budget:
            return "budget"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "experiments.politeness_experiment.code.run", "replay",
               "--seed", seed, "--day", str(day), "--mode", mode, "--setting", setting,
               "--player", player, "--rep", str(rep), "--root", str(a.root), "--games-root", str(a.games_root)]
        cmd += (["--dry-run"] if a.dry_run else [])
        with (out.parent / f"{out.name}.log").open("w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        return "ok" if rc == 0 else "failed"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        outcomes = list(pool.map(run_one, tasks))
    counts = dict(Counter(outcomes))
    log("flip_end", counts=counts, balance=balance(), flip_spend=round(flip_spend(a.root), 4), _dry=a.dry_run)
    print(json.dumps(counts))
    return 1 if counts.get("failed") or counts.get("budget") else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"play": cmd_play, "replay": cmd_replay, "flip": cmd_flip}
    if argv[:1] and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
