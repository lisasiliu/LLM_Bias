"""Language flip_self replays on the politeness baselines, with DeepSeek writing the rewrites.

    python -m experiments.proficiency_flip.code.run flip --first 1 --last 1 --workers 6

For each baseline day-start checkpoint and each living player: one flip_self replay per level (A restricted
range, B paraphrase control, C expanded range) and two control replays (natural text, branching at the same
line). The baselines are the ones the politeness study replays (`games/flip_experiments/<seed>/natural`), so both
studies treat the same days. Replays go to `games/flip_experiments/language/<seed>/day<N>/`.

Every rewrite passes the shared in-game gate (`components/judges/admission_gate.py`: player numbers, the action
appeal, the reasons check with the global cache) plus the language treatment checks; a failed rewrite is
regenerated once at temperature 0.7 and then the original text is delivered and logged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from components.judges.admission_gate import AdmissionGate
from components.judges.judge_model import LUNA, JudgeModel
from experiments.politeness_experiment.code.run import (
    DEFAULT_GAMES, MODEL, ROOT, Ledger, LedgerClient, balance, log, seed_name)

LEVELS = {"A": "low", "B": "neutral", "C": "high"}
LANGUAGE_ROOT = DEFAULT_GAMES / "language"
CACHE = DEFAULT_GAMES / "evidence_reasons.jsonl"
JUDGE = LUNA
RETRY_TEMPERATURE = 0.7


def replay_dir(root: Path, seed: str, day: int, level: str, player: str, rep: int = 0) -> Path:
    name = f"control__{player}" if level == "control" else f"flip_self__{level}__{player}"
    return root / seed / f"day{day}" / (name + (f"__r{rep}" if rep else ""))


def language_replay(seed: str, day: int, level: str, player: str, rep: int = 0, root: Path = LANGUAGE_ROOT,
                    games_root: Path = DEFAULT_GAMES) -> Path:
    from experiments.proficiency_validation.rewrite.gate import GatedRewriter, SemanticGate
    from lurkingwolf.checkpointing import load_checkpoint
    from lurkingwolf.eval.replay import replay_phase
    from lurkingwolf.run_game import load_prompts

    out = replay_dir(root, seed, day, level, player, rep)
    if (out / "result.json").exists():
        return out
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    baseline = games_root / seed / "natural"
    ckpt_path = baseline / "checkpoints" / f"day_{day}_start.pkl"
    prompts = load_prompts()
    ckpt, _, _ = load_checkpoint(ckpt_path, prompts, None)
    before = {pid: dict(a["reliability"]) for pid, a in ckpt.agents.items()}

    ledger = Ledger()
    players = LedgerClient(ledger, "players")
    rewriter = None
    if level != "control":
        judge = JudgeModel.resolve(JUDGE)                          # not the model that writes the rewrites
        judge.on_usage = lambda m, d: ledger.add("checks", str(d.get("model", "")), d.get("usage") or {})
        gate = SemanticGate.from_handle(JUDGE)                     # used only for Neutral's range labels
        gate.parser = None
        gate.validator.model.on_usage = lambda m, d: ledger.add("checks", str(d.get("model", "")), d.get("usage") or {})
        rewriter = GatedRewriter(
            base={"condition": LEVELS[level], "model": MODEL}, gate=gate, llm_client=LedgerClient(ledger, "rewriter"),
            max_corrective_retries=1, retry_temperature=RETRY_TEMPERATURE, on_unavailable="original",
            log_path=str(out / "rewrites.jsonl"), game_id=seed, branch_id=out.name,
            admission=AdmissionGate(judge, cache_path=CACHE))
    started = time.time()
    result, _agents, state = replay_phase(
        ckpt_path, prompts, players, experiment_type="nl", config_name=out.name, self_player=player,
        rewriter=rewriter, rewriter_scope="self", baseline_log=baseline / "game_log.jsonl", eligible=())
    entries = rewriter.log if rewriter else []
    if any(e.get("status") == "infrastructure_unavailable" or any(a.get("error") for a in (e.get("attempts") or []))
           for e in entries):
        raise RuntimeError("the checking model was unavailable for some rewrites; replay not saved (rerun it when the judge is back)")
    row = result.to_dict()
    row.update(seed=seed, mode="control" if level == "control" else "flip_self", setting=level, player=player,
               day=day, rep=rep, reliability_before=before, alive_at_end=sorted(state.get_alive()),
               rewrites=len(entries), fallbacks=sum(1 for e in entries if e.get("fallback_to_original")),
               elapsed_seconds=round(time.time() - started, 1))
    (out / "cost.json").write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    return out


def tasks_for(seed: str, games_root: Path = DEFAULT_GAMES) -> list:
    from lurkingwolf.checkpointing import load_checkpoint
    from lurkingwolf.run_game import load_prompts
    tasks = []
    for ck in sorted((games_root / seed / "natural" / "checkpoints").glob("day_*_start.pkl")):
        day = int(ck.stem.split("_")[1])
        ckpt, _, _ = load_checkpoint(ck, load_prompts(), None)
        for p in sorted(ckpt.state.get_alive(), key=lambda x: int(x[1:])):
            tasks += [(seed, day, "control", p, 0), (seed, day, "control", p, 1)]
            tasks += [(seed, day, lv, p, 0) for lv in LEVELS]
    return tasks


def cmd_replay(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--level", required=True, choices=list(LEVELS) + ["control"])
    ap.add_argument("--player", required=True)
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args(argv)
    print(f"done: {language_replay(a.seed, a.day, a.level, a.player, a.rep)}")
    return 0


def cmd_flip(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-balance", type=float, default=1.0, help="stop starting replays below this DeepSeek balance")
    ap.add_argument("--days", default="")
    ap.add_argument("--players", default="")
    a = ap.parse_args(argv)
    days = {int(x) for x in a.days.split(",") if x}
    players = {x for x in a.players.split(",") if x}
    tasks = [t for n in range(a.first, a.last + 1) for t in tasks_for(seed_name(n))
             if (not days or t[1] in days) and (not players or t[3] in players)]
    log("language_start", first=a.first, last=a.last, tasks=len(tasks), balance=balance())
    stop = {"flag": False}

    def run_one(t) -> str:
        seed, day, level, player, rep = t
        out = replay_dir(LANGUAGE_ROOT, seed, day, level, player, rep)
        if (out / "result.json").exists():
            return "done"
        bal = balance()
        if stop["flag"] or (bal is not None and bal < a.min_balance):
            stop["flag"] = True
            return "budget"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "experiments.proficiency_flip.code.run", "replay", "--seed", seed,
               "--day", str(day), "--level", level, "--player", player, "--rep", str(rep)]
        with (out.parent / f"{out.name}.log").open("w") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        return "ok" if rc == 0 else "failed"

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        outcomes = list(pool.map(run_one, tasks))
    counts = dict(Counter(outcomes))
    log("language_end", counts=counts, balance=balance())
    print(json.dumps(counts))
    return 1 if counts.get("failed") or counts.get("budget") else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"replay": cmd_replay, "flip": cmd_flip}
    if argv[:1] and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
