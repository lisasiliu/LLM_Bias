"""One entry point for the proficiency-cue validation.

    python -m experiments.proficiency_validation.code.cli <command> [options]

Offline commands (no model calls): audit, bundles, benchmark-judges,
build-challenge, report-challenge, features, suggest-bounds, freeze,
verify-frozen, analyze, build-human-worksheet, import-human-worksheet.
Model commands: generate, run-challenge, guardrails (politeness part), probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from components.helpers import read_jsonl, write_jsonl

from . import analysis, features, freeze, generate, guardrails, listener
from .audit import audit, load_sources, render_audit
from .bundles import build_bundles
from .common import DATA_DIR, EXPERIMENT_DIR, dump_json, now_iso
from .judge_benchmark import (
    build_challenge_set, human_benchmark, human_preserved_positives, render_challenge,
    render_human_benchmark, report_challenge, run_challenge, source_record_from_dict)

POLITENESS_MANUAL = EXPERIMENT_DIR.parent / "politeness_validation" / "data" / "manual_validation.jsonl"
POLITENESS_PROVENANCE = EXPERIMENT_DIR.parent / "politeness_validation" / "data" / "provenance.json"


def cmd_audit(args) -> int:
    sources = load_sources([Path(p) for p in args.game_dirs], allow_treated=args.allow_treated)
    report = audit(sources, windows=tuple(args.windows), window_span=args.window_span)
    text = render_audit(report)
    if args.out:
        dump_json(args.out, report)
        Path(args.out).with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"wrote {args.out} and {Path(args.out).with_suffix('.md')}")
    print(text)
    return 0


def cmd_bundles(args) -> int:
    if args.sources:
        records = [source_record_from_dict(r) for r in read_jsonl(args.sources)]
    else:
        by_game = load_sources([Path(p) for p in args.game_dirs], allow_treated=args.allow_treated)
        records = [r for rows in by_game.values() for r in rows]
        if args.sources_out:
            write_jsonl(args.sources_out, [r.to_dict() for r in records])
            print(f"{len(records)} source utterances -> {args.sources_out}")
    result = build_bundles(records, window_min=args.window[0], window_max=args.window[1],
                           max_per_game=args.max_per_game, seed=args.seed)
    write_jsonl(args.out, result["bundles"])
    dump_json(Path(args.out).with_name(Path(args.out).stem + "_report.json"),
              {"created_at": now_iso(), "window": args.window, "max_per_game": args.max_per_game,
               "seed": args.seed, "n_bundles": len(result["bundles"]),
               "n_games": len({b["game_id"] for b in result["bundles"]}),
               "insufficient_speakers": result["insufficient"], "dropped_by_cap": result["dropped_by_cap"]})
    print(f"{len(result['bundles'])} bundles from {len({b['game_id'] for b in result['bundles']})} games "
          f"-> {args.out}; {len(result['insufficient'])} speakers below the window, "
          f"{len(result['dropped_by_cap'])} dropped by the per-game cap")
    return 0


def cmd_benchmark_judges(args) -> int:
    records = read_jsonl(args.manual)
    if not records:
        print(f"no records at {args.manual}", file=sys.stderr)
        return 2
    bench = human_benchmark(records, use_recheck=not args.no_recheck)
    bench["source"] = str(args.manual)
    bench["created_at"] = now_iso()
    text = render_human_benchmark(bench)
    if args.out:
        dump_json(args.out, bench)
        Path(args.out).with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"wrote {args.out} and {Path(args.out).with_suffix('.md')}")
    else:
        print(text)
    return 0


def cmd_build_challenge(args) -> int:
    sources = generate.load_source_records(args.sources)
    positives = human_preserved_positives(read_jsonl(args.manual)) if args.manual else []
    items = build_challenge_set(sources, seed=args.seed, positives=positives)
    write_jsonl(args.out, items)
    counts = {}
    for item in items:
        counts[item["challenge_type"]] = counts.get(item["challenge_type"], 0) + 1
    print(f"{len(items)} challenge items -> {args.out}")
    for kind, n in sorted(counts.items()):
        print(f"  {kind:28s} {n}")
    return 0


def cmd_run_challenge(args) -> int:
    from .semantic_gate import SemanticGate
    items = read_jsonl(args.items)
    if args.limit:
        items = items[:args.limit]
    gate = SemanticGate.from_handle(args.judge)
    ok, msg = gate.judge_model.preflight()
    if not ok:
        print(f"judge preflight failed: {msg}", file=sys.stderr)
        return 1
    judged = run_challenge(items, gate)
    write_jsonl(args.out, judged)
    dump_json(Path(args.out).with_name(Path(args.out).stem + "_provenance.json"),
              {"created_at": now_iso(), "gate": gate.provenance(), "n_items": len(judged)})
    print(f"{len(judged)} judged items -> {args.out}")
    return 0


def cmd_report_challenge(args) -> int:
    items = read_jsonl(args.items)
    report = report_challenge(items)
    text = render_challenge(report)
    if args.out:
        dump_json(args.out, report)
        Path(args.out).with_suffix(".md").write_text(text, encoding="utf-8")
        print(f"wrote {args.out} and {Path(args.out).with_suffix('.md')}")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="audit untreated game runs for bundle capacity")
    a.add_argument("game_dirs", nargs="+")
    a.add_argument("--windows", type=int, nargs="+", default=[150, 200, 250],
                   help="candidate window minimums; each is evaluated as [min, min + span]")
    a.add_argument("--window-span", type=int, default=100)
    a.add_argument("--allow-treated", action="store_true")
    a.add_argument("--out", type=Path, default=None)
    a.set_defaults(func=cmd_audit)

    b = sub.add_parser("bundles", help="build matched source bundles (one per speaker per game)")
    b.add_argument("game_dirs", nargs="*")
    b.add_argument("--sources", type=Path, default=None, help="use a sources JSONL instead of game dirs")
    b.add_argument("--sources-out", type=Path, default=None, help="also dump the per-utterance sources")
    b.add_argument("--window", type=int, nargs=2, default=[150, 250], metavar=("MIN", "MAX"))
    b.add_argument("--max-per-game", type=int, default=None)
    b.add_argument("--seed", type=int, default=101)
    b.add_argument("--allow-treated", action="store_true")
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=cmd_bundles)

    j = sub.add_parser("benchmark-judges", help="semantic judges vs the 48 human meaning labels")
    j.add_argument("--manual", type=Path, default=POLITENESS_MANUAL)
    j.add_argument("--no-recheck", action="store_true", help="score the v1 rubric verdicts only")
    j.add_argument("--out", type=Path, default=None)
    j.set_defaults(func=cmd_benchmark_judges)

    c = sub.add_parser("build-challenge", help="synthetic semantic challenge set from sources")
    c.add_argument("--sources", type=Path, default=POLITENESS_PROVENANCE,
                   help="sources JSONL or a provenance.json with a sources list")
    c.add_argument("--manual", type=Path, default=POLITENESS_MANUAL,
                   help="politeness manual_validation.jsonl for human-preserved positives ('' to skip)")
    c.add_argument("--seed", type=int, default=101)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_build_challenge)

    r = sub.add_parser("run-challenge", help="judge every challenge item with the semantic gate")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("--judge", default="deepseek-chat")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=cmd_run_challenge)

    rc = sub.add_parser("report-challenge", help="sensitivity/specificity per challenge type")
    rc.add_argument("--items", type=Path, required=True)
    rc.add_argument("--out", type=Path, default=None)
    rc.set_defaults(func=cmd_report_challenge)

    generate.add_parser(sub)
    features.add_parser(sub)
    guardrails.add_parser(sub)
    listener.add_parser(sub)
    analysis.add_parser(sub)
    freeze.add_parser(sub)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "manual", None) == Path(""):
        args.manual = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
