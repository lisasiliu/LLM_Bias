"""Subprocess entry point for a single phase replay.

Invoked by ``runner.py`` as ``python -m lurkingwolf.eval.replay_cli``.
Loads a checkpoint, applies agent settings, replays one phase,
writes phase_result.json + full game output files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..run_game import load_prompts, _dump_agent_memory
from ..schemas import build_agent_memory
from .replay import replay_phase


def _sort_player_dict(d: dict) -> dict:
    """Sort dict keys by player number (P1, P2, ..., P7)."""
    def _pkey(k):
        if isinstance(k, str) and k.startswith("P") and k[1:].isdigit():
            return int(k[1:])
        return (999, k)
    return dict(sorted(d.items(), key=lambda x: _pkey(x[0])))


def _compact_phase_json(data: dict) -> str:
    """Serialize phase_result.json compactly.

    - role_dist, votes, vote_counts, votes_received sorted by player number, one line
    - Each speech on one line (NL pred field stripped — redundant with predicates)
    - Each player's reliability scores on one line, sorted by player number
    """
    player_sorted_keys = {"role_dist", "votes", "vote_counts", "votes_received"}

    lines = ["{"]
    keys = list(data.keys())
    for ki, key in enumerate(keys):
        comma = "," if ki < len(keys) - 1 else ""
        if key == "speeches":
            speeches = data["speeches"]
            cleaned = []
            for speech in speeches:
                s = dict(speech)
                if "nl" in s:
                    s["nl"] = [{"text": it.get("text", "")} for it in s["nl"]]
                cleaned.append(s)
            lines.append('  "speeches": [')
            for si, speech in enumerate(cleaned):
                sc = "," if si < len(cleaned) - 1 else ""
                lines.append(f"    {json.dumps(speech, ensure_ascii=False)}{sc}")
            lines.append(f"  ]{comma}")
        elif key == "reliability":
            rel = _sort_player_dict(data["reliability"])
            lines.append('  "reliability": {')
            items = list(rel.items())
            for ri, (pid, scores) in enumerate(items):
                rc = "," if ri < len(items) - 1 else ""
                sorted_scores = _sort_player_dict(scores)
                lines.append(f"    {json.dumps(pid)}: {json.dumps(sorted_scores, ensure_ascii=False)}{rc}")
            lines.append(f"  }}{comma}")
        elif key in player_sorted_keys and isinstance(data[key], dict):
            sorted_val = _sort_player_dict(data[key])
            lines.append(f"  {json.dumps(key)}: {json.dumps(sorted_val, ensure_ascii=False)}{comma}")
        else:
            lines.append(f"  {json.dumps(key)}: {json.dumps(data[key], ensure_ascii=False)}{comma}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _generate_game_out(result, state) -> str:
    """Generate game.out-style text log from phase replay results."""
    lines: List[str] = []
    d = result.to_dict()
    day = d["day"]

    lines.append(f"=== Phase Replay: {d['config_name']} (day {day}) ===")
    lines.append(f"Experiment: {d['experiment_type']}")
    if d.get("self_player"):
        lines.append(f"Self player: {d['self_player']}")
    lines.append(f"Roles: {d['role_dist']}")
    lines.append("")

    lines.append(f"--- Day {day} Discussion ---")
    lines.append(f"  Alive: {d.get('alive_at_start', state.get_alive())}")
    for speech in d.get("speeches", []):
        speaker = speech["speaker"]
        for pred in speech.get("predicates", []):
            lines.append(f"  {pred}")
        for item in speech.get("nl", []):
            text = (item.get("text") or "").strip()
            if text:
                lines.append(f"    NL({speaker}): {text}")
    lines.append("")

    lines.append(f"--- Day {day} Vote ---")
    for voter, target in d.get("votes", {}).items():
        lines.append(f"  {voter} -> {target or 'abstain'}")
    lines.append(f"  Vote counts: {d.get('vote_counts', {})}")
    if d.get("executed"):
        lines.append(f"  Executed: {d['executed']}")
    else:
        lines.append("  No execution")
    lines.append("")

    lines.append(f"--- Night {day} ---")
    if d.get("wolf_target"):
        lines.append(f"  Wolves attack: {d['wolf_target']}")
    if d.get("seer_target"):
        lines.append(f"  Seer divines: {d['seer_target']}")
    if d.get("guard_target"):
        lines.append(f"  Guard protects: {d['guard_target']}")
    if d.get("witch_save"):
        lines.append("  Witch saves")
    if d.get("witch_poison"):
        lines.append(f"  Witch poisons: {d['witch_poison']}")
    if d.get("night_killed"):
        lines.append(f"  >> {d['night_killed']} killed!")
    else:
        lines.append("  No one killed")
    lines.append("")

    return "\n".join(lines) + "\n"


def _dump_full_outputs(
    result,
    agents: Dict[str, Any],
    state,
    out_dir: Path,
) -> None:
    """Write agent_memory/*.json, current_turn/day_N.json, game.out."""
    # agent_memory
    mem_dir = out_dir / "agent_memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    for pid, agent in agents.items():
        memory = build_agent_memory(
            pid, agent.role,
            fact=agent.fact,
            potential_truth=agent.potential_truth,
            potential_deception=agent.potential_deception,
            reliability=agent.reliability,
            reasoning_history=agent.reasoning_history,
            relationship_estimates=agent.relationship_estimates,
            action_history=agent.action_history,
            statement_history=agent.statement_history,
            thinking_traces=agent.thinking_traces or None,
        )
        with open(mem_dir / f"{pid}.json", "w", encoding="utf-8") as f:
            f.write(_dump_agent_memory(memory))

    # current_turn/day_N.json — single phase with discussion predicates
    ct_dir = out_dir / "current_turn"
    ct_dir.mkdir(parents=True, exist_ok=True)
    day = result.day
    d = result.to_dict()
    ct_data = {
        "alive_players": state.get_alive(),
        "dead_players": state.get_dead(),
        "sheriff": state.sheriff,
        f"day_{day}_discussion": [
            pred
            for speech in d.get("speeches", [])
            for pred in speech.get("predicates", [])
        ],
        f"day_{day}_vote": {
            "votes": d["votes"],
            "vote_counts": d["vote_counts"],
            "executed": d["executed"],
        },
    }
    with open(ct_dir / f"day_{day}.json", "w", encoding="utf-8") as f:
        json.dump(ct_data, f, ensure_ascii=False, indent=2)

    # game.out
    game_out_text = _generate_game_out(result, state)
    with open(out_dir / "game.out", "w", encoding="utf-8") as f:
        f.write(game_out_text)

    # game_log.jsonl — raw LLM calls
    with open(out_dir / "game_log.jsonl", "w", encoding="utf-8") as f:
        for pid, agent in sorted(agents.items()):
            for entry in agent.call_log:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one phase from a day-start checkpoint."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to day-start checkpoint .pkl")
    parser.add_argument("--experiment", required=True,
                        help="Experiment type (nl, pred_only)")
    parser.add_argument("--config-name", default="",
                        help="Config name for metadata")
    parser.add_argument("--self-player", default=None,
                        help="Self player ID (e.g. P1)")
    parser.add_argument("--output", required=True,
                        help="Output path for phase_result.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")

    # Backend flags
    parser.add_argument("--vllm-url", default=None)
    parser.add_argument("--vllm-key", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3.5-27B")

    # Rewriter flags
    parser.add_argument("--rewriter", default=None,
                        help="Rewriter name from the registry")
    parser.add_argument("--rewriter-params", default="{}",
                        help="JSON string of rewriter params")
    parser.add_argument("--rewriter-scope", default="all",
                        choices=("all", "self", "others"),
                        help="Which agents get the rewriter (mapped from mode: "
                             "flip_self→self, flip_others→others)")
    parser.add_argument("--rewriter-module", default=None,
                        help="Python module to import for rewriter registration")

    args = parser.parse_args()

    prompts = load_prompts()

    # Build client
    from ..utils import LLMClient
    client = None
    if not args.dry_run:
        client = LLMClient(base_url=args.vllm_url, api_key=args.vllm_key)

    if args.rewriter_module:
        import importlib
        importlib.import_module(args.rewriter_module)

    rewriter_instance = None
    if args.rewriter:
        from ..rewriter import get_rewriter
        rewriter_cls = get_rewriter(args.rewriter)
        rewriter_params = json.loads(args.rewriter_params)
        rewriter_instance = rewriter_cls(**rewriter_params, llm_client=client)

    result, agents, state = replay_phase(
        checkpoint_path=Path(args.checkpoint),
        prompts=prompts,
        client=client,
        experiment_type=args.experiment,
        config_name=args.config_name,
        self_player=args.self_player,
        rewriter=rewriter_instance,
        rewriter_scope=args.rewriter_scope,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    out_path = Path(args.output)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compact phase_result.json
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_compact_phase_json(result.to_dict()))

    # Full game output files
    _dump_full_outputs(result, agents, state, out_dir)

    print(f"Phase replay complete → {out_dir}/")


if __name__ == "__main__":
    main()
