"""Unified CLI entry point: ``lurk <command> [args]``."""

from __future__ import annotations

import sys


COMMANDS = {
    "check-determinism": "lurkingwolf.check_determinism",
    "run": "lurkingwolf.run_game",
    "eval": "lurkingwolf.eval.launch",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: lurk <command> [args]\n")
        print("commands:")
        print("  run                Run a game or config-driven experiment")
        print("  eval               Run evaluation (baseline + replays)")
        print("  check-determinism  Test model output determinism")
        print("\nUse 'lurk <command> --help' for command-specific help.")
        return 0

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd!r}", file=sys.stderr)
        print(f"available: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2

    module_name = COMMANDS[cmd]
    sys.argv = sys.argv[1:]  # strip 'lurk' from argv so argparse sees the right args
    sys.argv[0] = f"lurk {cmd}"

    import importlib
    mod = importlib.import_module(module_name)
    return mod.main()


if __name__ == "__main__":
    raise SystemExit(main())
