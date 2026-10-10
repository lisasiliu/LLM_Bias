"""Play, run and cost the in-game politeness experiment (resumable, seed by seed).

    python -m experiments.politeness_experiment.code.run pilot          # seeds 1-2
    python -m experiments.politeness_experiment.code.run run --first 3 --last 20
    python -m experiments.politeness_experiment.code.run run --arms all_positive
    python -m experiments.politeness_experiment.code.run cost           # write cost.md
    python -m experiments.politeness_experiment.code.run play --seed pe01 --arm trust_positive
    python -m experiments.politeness_experiment.code.run flip --first 1 --last 2 --workers 6

`flip` is the flip-self / flip-others experiment (research/POLITENESS_FLIP_PLAN.md): one-day replays
from a baseline's day-start checkpoint in which one player's speech (self) or everyone else's
(others) is delivered in a politeness setting; `flipone` runs a single replay.

Per seed: the natural baseline first, then the arms (and, on the first three seeds, the
natural replay for the drift check) in parallel from its night-0 checkpoint. `natural`
plays a fresh baseline and saves checkpoints; any other arm resumes from it and plays to
the end with the arm's rewrite policy applied to every village-side speaker. A finished
game (its folder has manifest.json) is never replayed, and an unfinished one starts
again. Each game runs in its own process because the engine's random state is global.
Each game folder gets the usual LURK output plus `cost.json` and, for policy arms,
`touched.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
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
from components.judges.structural_check import AdmissionGate, StructuralGate
from components.judges.judge_model import JudgeModel, read_key
from components.rewriters.trust_politeness_policy import ARMS, FLIP_ACTS, TrustPolitenessPolicy


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GAMES = ROOT / "games_fresh" / "politeness_experiment"
MODEL = "deepseek-chat"
BASE_URL = "https://api.deepseek.com"
ARM_NAMES = ["natural", "natural_replay"] + sorted(ARMS)

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


def play(seed: str, arm: str, games_root: Path = DEFAULT_GAMES, dry_run: bool = False) -> Path:
    from lurkingwolf.run_game import run_single_game, save_game_output, load_prompts
    out = games_root / seed / arm
    if (out / "manifest.json").exists():
        return out
    if out.exists():
        shutil.rmtree(out)                      # an unfinished game: start it again
    out.mkdir(parents=True)
    base_ckpt = games_root / seed / "natural" / "checkpoints" / "night_0_start.pkl"
    if arm != "natural" and not base_ckpt.exists():
        raise FileNotFoundError(f"play the natural baseline for {seed} first")

    ledger = Ledger()
    players = None if dry_run else LedgerClient(ledger, "players")
    prompts = load_prompts()
    kwargs = dict(game_id="g1", model=MODEL, client=players, prompts=prompts, num_players=7,
                  temperature=0, max_tokens=300, max_rounds=20, dry_run=dry_run,
                  experiment_type="nl", seed=seed)
    policy = None
    if arm == "natural":
        kwargs["checkpoint_dir"] = out / "checkpoints"
    else:
        kwargs["resume_checkpoint"] = base_ckpt
        if arm in ARMS:
            gate = None
            writer_client = None
            if not dry_run:
                judge = JudgeModel.resolve(MODEL)
                judge.on_usage = lambda m, d: ledger.add("checks", str(d.get("model", "")), d.get("usage") or {})
                gate = AdmissionGate(judge)
                writer_client = LedgerClient(ledger, "rewriter")
            policy = TrustPolitenessPolicy(llm_client=writer_client, arm=arm, seed=seed,
                                           model=MODEL, gate=gate, dry_run=dry_run,
                                           log_path=out / "touched.jsonl")
            kwargs["rewriter"] = policy

    started = time.time()
    result = run_single_game(**kwargs)
    elapsed = time.time() - started
    save_game_output(result, out, terminal_output="")
    state = result["state"]
    summary = {"seed": seed, "arm": arm, "winner": state.winner, "model_requested": MODEL,
               "started_utc": datetime.fromtimestamp(started, tz=timezone.utc).isoformat(),
               "elapsed_seconds": round(elapsed, 1), "dry_run": dry_run,
               "policy_counters": policy.counters if policy else None}
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "cost.json").write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
    return out


def cmd_play(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--arm", required=True, choices=ARM_NAMES)
    ap.add_argument("--root", type=Path, default=DEFAULT_GAMES)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    out = play(a.seed, a.arm, a.root, a.dry_run)
    print(f"done: {out}")
    return 0



HERE = Path(__file__).resolve().parents[1]
RUN_LOG = HERE / "data" / "run_log.jsonl"
MAIN_ARMS = ["all_bald", "trust_positive", "trust_bald"]
DRIFT_SEEDS = 3
BUDGET_USD = 40.0


def seed_name(n: int) -> str:
    return f"pe{n:02d}"


def log(event: str, **kw) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kw}
    with RUN_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def done(root: Path, seed: str, arm: str) -> bool:
    return (root / seed / arm / "manifest.json").exists()


def play_game(root: Path, seed: str, arm: str) -> bool:
    if done(root, seed, arm):
        return True
    (root / seed).mkdir(parents=True, exist_ok=True)
    started = time.time()
    log("start", seed=seed, arm=arm, peak=is_peak(started))
    with (root / seed / f"{arm}.log").open("w") as f:
        rc = subprocess.run(
            [sys.executable, "-m", "experiments.politeness_experiment.code.run", "play",
             "--seed", seed, "--arm", arm, "--root", str(root)],
            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    log("finish" if rc == 0 else "failed", seed=seed, arm=arm, returncode=rc,
        minutes=round((time.time() - started) / 60, 1))
    return rc == 0


def estimated_spend(root: Path) -> float:
    total = 0.0
    for p in root.glob("pe*/*/cost.json"):
        total += json.loads(p.read_text())["total_cost_usd"]
    return total


def run_seed(root: Path, n: int, arms: list, budget: float, seed_budget_check: bool) -> dict:
    seed = seed_name(n)
    if seed_budget_check and estimated_spend(root) >= budget:
        log("budget_stop", seed=seed, spend=estimated_spend(root))
        return {"seed": seed, "skipped": "budget"}
    if not play_game(root, seed, "natural"):
        return {"seed": seed, "failed": ["natural"]}
    todo = list(arms) + (["natural_replay"] if n <= DRIFT_SEEDS else [])
    with ThreadPoolExecutor(max_workers=len(todo)) as pool:
        ok = list(pool.map(lambda a: play_game(root, seed, a), todo))
    return {"seed": seed, "failed": [a for a, k in zip(todo, ok) if not k]}


def cmd_run(args) -> int:
    root = Path(args.root)
    arms = args.arms.split(",")
    start_balance = balance()
    log("run_start", first=args.first, last=args.last, arms=arms, balance=start_balance,
        spend_before=round(estimated_spend(root), 4))
    seeds = list(range(args.first, args.last + 1))
    with ThreadPoolExecutor(max_workers=args.seed_workers) as pool:
        results = list(pool.map(lambda n: run_seed(root, n, arms, args.budget, not args.pilot), seeds))
    end_balance = balance()
    log("run_end", balance=end_balance, results=results,
        spend_after=round(estimated_spend(root), 4))
    bad = [r for r in results if r.get("failed") or r.get("skipped")]
    print(json.dumps(results))
    if start_balance is not None and end_balance is not None:
        print(f"account balance change: {start_balance - end_balance:.3f} USD")
    return 1 if bad else 0


def cmd_cost(args) -> int:
    write_cost_md(Path(args.root), HERE / "cost.md", RUN_LOG, args.budget)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("play", help="play one game: play --seed S --arm A")
    sub.add_parser("flip", help="flip-self / flip-others replays; see the module docstring")
    for name in ("pilot", "run", "cost"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(DEFAULT_GAMES))
        p.add_argument("--budget", type=float, default=BUDGET_USD)
        if name != "cost":
            p.add_argument("--arms", default=",".join(MAIN_ARMS))
            p.add_argument("--seed-workers", type=int, default=1)
            p.add_argument("--first", type=int, default=1)
            p.add_argument("--last", type=int, default=2 if name == "pilot" else 20)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["play"]:
        return cmd_play(argv[1:])
    if argv[:1] == ["flipone"]:
        return cmd_flipone(argv[1:])
    if argv[:1] == ["flip"]:
        return cmd_flip(argv[1:])
    args = ap.parse_args(argv)
    args.pilot = args.cmd == "pilot"
    return {"pilot": cmd_run, "run": cmd_run, "cost": cmd_cost}[args.cmd](args)



# ── flip experiment: flip_self / flip_others one-day replays ──────────────

FLIP_ROOT = ROOT / "games_fresh" / "politeness_flip"
SETTINGS = ["all_bald", "all_positive", "trust_positive", "trust_bald"]
SCOPES = {"flip_self": "self", "flip_others": "others"}
CONTROL_REPS = 0


def flip_dir(root: Path, seed: str, day: int, mode: str, setting: str, player: str) -> Path:
    name = f"control__{player}" if mode == "control" else f"{mode}__{setting}__{player}"
    return root / seed / f"day{day}" / name


def flip_replay(seed: str, day: int, mode: str, setting: str, player: str,
                root: Path = FLIP_ROOT, games_root: Path = DEFAULT_GAMES, dry_run: bool = False) -> Path:
    """One replay of day `day` from the baseline's day-start checkpoint. mode is flip_self,
    flip_others or control (no rewriter; `player` is then a repetition label)."""
    from lurkingwolf.checkpointing import load_checkpoint
    from lurkingwolf.eval.replay import replay_phase
    from lurkingwolf.run_game import load_prompts

    out = flip_dir(root, seed, day, mode, setting, player)
    if (out / "result.json").exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ckpt_path = games_root / seed / "natural" / "checkpoints" / f"day_{day}_start.pkl"
    prompts = load_prompts()
    ckpt, _, _ = load_checkpoint(ckpt_path, prompts, None)
    before = {pid: dict(a["reliability"]) for pid, a in ckpt.agents.items()}   # ratings in force at the checkpoint

    ledger = Ledger()
    players = None if dry_run else LedgerClient(ledger, "players")
    policy = None
    if mode != "control":
        writer = None if dry_run else LedgerClient(ledger, "rewriter")
        policy = TrustPolitenessPolicy(
            llm_client=writer, arm=setting, seed=f"{seed}|d{day}|{mode}|{player}", model=MODEL,
            gate=StructuralGate(), dry_run=dry_run, log_path=out / "touched.jsonl",
            acts=FLIP_ACTS, include_wolves=True, treat_unaddressed=True)
    started = time.time()
    result, _agents, state = replay_phase(
        ckpt_path, prompts, players, experiment_type="nl", config_name=out.name,
        self_player=None if mode == "control" else player, rewriter=policy,
        rewriter_scope=SCOPES.get(mode, "all"), dry_run=dry_run)
    row = result.to_dict()
    row.update(seed=seed, mode=mode, setting=setting, player=player, day=day,
               reliability_before=before, alive_at_end=sorted(state.get_alive()),
               policy_counters=policy.counters if policy else None,
               elapsed_seconds=round(time.time() - started, 1), dry_run=dry_run)
    (out / "cost.json").write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return out


def cmd_flipone(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=sorted(SCOPES) + ["control"])
    ap.add_argument("--setting", default="none", choices=SETTINGS + ["none"])
    ap.add_argument("--player", required=True)
    ap.add_argument("--root", type=Path, default=FLIP_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(f"done: {flip_replay(a.seed, a.day, a.mode, a.setting, a.player, a.root, dry_run=a.dry_run)}")
    return 0


def flip_tasks(games_root: Path, seed: str) -> list:
    """Every replay for one seed: for each day with a checkpoint, the untreated controls, then
    each mode x setting x living player."""
    from lurkingwolf.checkpointing import load_checkpoint
    from lurkingwolf.run_game import load_prompts
    tasks = []
    ck_dir = games_root / seed / "natural" / "checkpoints"
    for ck in sorted(ck_dir.glob("day_*_start.pkl")):
        day = int(ck.stem.split("_")[1])
        ckpt, _, _ = load_checkpoint(ck, load_prompts(), None)
        alive = sorted(ckpt.state.get_alive(), key=lambda p: int(p[1:]))
        tasks += [(seed, day, "control", "none", f"rep{r + 1}") for r in range(CONTROL_REPS)]
        tasks += [(seed, day, mode, setting, p) for mode in SCOPES for setting in SETTINGS for p in alive]
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
    a = ap.parse_args(argv)
    tasks = [t for n in range(a.first, a.last + 1) for t in flip_tasks(a.games_root, seed_name(n))]
    log("flip_start", first=a.first, last=a.last, tasks=len(tasks), balance=balance())

    def run_one(t) -> str:
        seed, day, mode, setting, player = t
        out = flip_dir(a.root, seed, day, mode, setting, player)
        if (out / "result.json").exists():
            return "done"
        if estimated_spend(a.games_root) + flip_spend(a.root) >= a.budget:
            return "budget"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "experiments.politeness_experiment.code.run", "flipone",
               "--seed", seed, "--day", str(day), "--mode", mode, "--setting", setting,
               "--player", player, "--root", str(a.root)] + (["--dry-run"] if a.dry_run else [])
        with (out.parent / f"{out.name}.log").open("w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        return "ok" if rc == 0 else "failed"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        outcomes = list(pool.map(run_one, tasks))
    counts = dict(Counter(outcomes))
    log("flip_end", counts=counts, balance=balance(), flip_spend=round(flip_spend(a.root), 4))
    print(json.dumps(counts))
    return 1 if counts.get("failed") or counts.get("budget") else 0


CATS = ["players", "rewriter", "checks"]
MAIN = ["natural", "all_bald", "trust_positive", "trust_bald"]


def write_cost_md(root: Path, out: Path, run_log: Path, budget: float) -> None:
    rows = []
    for p in sorted(root.glob("pe*/*/cost.json")):
        seed, arm = p.parent.parent.name, p.parent.name
        d = json.loads(p.read_text())
        run = json.loads((p.parent / "run.json").read_text())
        rows.append({"seed": seed, "arm": arm, "d": d, "minutes": run["elapsed_seconds"] / 60})
    if not rows:
        out.write_text("# Cost\n\nNo games have finished yet.\n")
        return

    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    calls = {c: 0 for c in CATS}
    peak = {c: 0 for c in CATS}
    models = defaultdict(int)
    for r in rows:
        for c in CATS:
            cc = r["d"]["by_category"].get(c, {})
            calls[c] += cc.get("calls", 0)
            peak[c] += cc.get("peak_calls", 0)
        for m, n in r["d"]["models_returned"].items():
            models[m] += n
    total = sum(r["d"]["total_cost_usd"] for r in rows)
    all_calls = sum(calls.values())
    peak_share = sum(peak.values()) / all_calls if all_calls else 0.0

    lines = ["# Cost", "",
             "Costs are estimated from the token counts the API reports, using the DeepSeek Flash "
             "price in force when each call was made (cache hit 0.003, cache miss 0.15, output 0.60 USD "
             "per million tokens, doubled in peak hours). The judge step (`gpt-5.6-luna`) is not "
             "included until the manipulation check is run.", "",
             "## Measured per game", "",
             "| Arm | Games | Mean cost (USD) | Players | Rewriter | Checks | Mean minutes |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    arm_cost = {}
    for arm in sorted(by_arm, key=lambda a: (a not in MAIN, a)):
        rs = by_arm[arm]
        mean = statistics.mean(r["d"]["total_cost_usd"] for r in rs)
        arm_cost[arm] = mean
        cats = [statistics.mean(r["d"]["by_category"].get(c, {}).get("cost_usd", 0.0) for r in rs) for c in CATS]
        lines.append(f"| {arm} | {len(rs)} | {mean:.3f} | " + " | ".join(f"{c:.3f}" for c in cats)
                     + f" | {statistics.mean(r['minutes'] for r in rs):.1f} |")

    seeds_full = [s for s in {r["seed"] for r in rows}
                  if all(any(x["seed"] == s and x["arm"] == a for x in rows) for a in MAIN)]
    per_seed = [sum(r["d"]["total_cost_usd"] for r in rows if r["seed"] == s and r["arm"] in MAIN)
                for s in seeds_full]
    lines += ["", "## Projection", ""]
    if per_seed:
        mean_seed = statistics.mean(per_seed)
        drift = arm_cost.get("natural_replay", arm_cost.get("trust_positive", 0.0))
        proj_main = mean_seed * 20 + drift * 3
        bl = arm_cost.get("bl_written", statistics.mean(arm_cost.get(a, 0.0) for a in ("trust_positive", "trust_bald")))
        proj_bl = proj_main + bl * 20
        lines += [f"- Complete seeds measured: {len(per_seed)}; mean cost of the baseline plus the three main arms: "
                  f"{mean_seed:.3f} USD per seed.",
                  f"- Projected total for 20 seeds, three main arms plus drift replays on 3 seeds: **{proj_main:.2f} USD**.",
                  f"- With `bl_written` on all 20 seeds: **{proj_bl:.2f} USD**.",
                  f"- Budget cap: {budget:.0f} USD. Three main arms {'under' if proj_main < budget else 'OVER'} the cap."]
    lines += ["", "## Peak hours and model", "",
              f"- {sum(peak.values())} of {all_calls} calls ({peak_share:.0%}) were made in peak hours "
              "(01:00 to 04:00 and 06:00 to 10:00 UTC, Monday to Friday). Chinese public holidays are not "
              "modelled, so peak pricing may be overstated.",
              f"- Spend so far: {total:.2f} USD estimated.",
              "- Model requested: `deepseek-chat`. Model name returned by the API, by number of calls: "
              + ", ".join(f"`{m}` ({n})" for m, n in sorted(models.items())) + "."]

    balances = []
    if run_log.exists():
        for line in run_log.read_text().splitlines():
            e = json.loads(line)
            if e.get("event") in ("run_start", "run_end") and e.get("balance") is not None:
                balances.append((e["event"], e["balance"], e.get("spend_before", e.get("spend_after"))))
    starts = [b for b in balances if b[0] == "run_start"]
    ends = [b for b in balances if b[0] == "run_end"]
    if starts and ends:
        billed = starts[0][1] - ends[-1][1]
        lines += ["", "## Check against the account balance", "",
                  f"The account balance fell by {billed:.2f} USD between the first run start and the last run "
                  f"end; the estimate for the same games is {total:.2f} USD."]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
