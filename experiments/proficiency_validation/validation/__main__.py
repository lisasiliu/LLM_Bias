"""Proficiency validation __main__."""
from __future__ import annotations
from experiments.proficiency_validation.rewrite.runtime import update_metadata
import argparse
from pathlib import Path
from typing import List, Optional
from components.helpers import read_jsonl, write_jsonl
import experiments.proficiency_validation.validation.analysis as analysis
import experiments.proficiency_validation.validation.measurements as features
import experiments.proficiency_validation.validation.protocol as freeze
import experiments.proficiency_validation.validation.measurements as guardrails
import experiments.proficiency_validation.validation.perception as listener
import experiments.proficiency_validation.validation.report as report
from experiments.proficiency_validation.validation.sources import audit, load_sources, render_audit
from experiments.proficiency_validation.validation.sources import build_bundles
from experiments.proficiency_validation.rewrite.runtime import DATA_DIR, dump_json, now_iso
from experiments.proficiency_validation.validation.calibration import cmd_build_challenge, cmd_run_challenge, cmd_report_challenge
from experiments.proficiency_validation.rewrite.runtime import source_record_from_dict


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
                           max_per_game=args.max_per_game, seed=args.seed, selection=args.selection)
    write_jsonl(args.out, result["bundles"])
    update_metadata(args.out, "summary",
              {"created_at": now_iso(), "window": [args.window[0], None if args.selection == "whole_game" else args.window[1]],
               "selection": args.selection, "max_per_game": args.max_per_game,
               "seed": args.seed, "n_bundles": len(result["bundles"]),
               "n_games": len({b["game_id"] for b in result["bundles"]}),
               "insufficient_speakers": result["insufficient"], "dropped_by_cap": result["dropped_by_cap"]})
    print(f"{len(result['bundles'])} bundles from {len({b['game_id'] for b in result['bundles']})} games "
          f"-> {args.out}; {len(result['insufficient'])} speakers below the window, "
          f"{len(result['dropped_by_cap'])} dropped by the per-game cap")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    report.add_parser(sub)
    import experiments.proficiency_validation.validation.cefr as cefr
    cefr.add_parser(sub)

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
    b.add_argument("--selection", choices=["whole_game", "phase_stratified", "prefix"], default="whole_game",
                   help="whole-game speaker transcript, chronological-thirds sensitivity, or prefix baseline; whole_game uses MIN only")
    b.add_argument("--max-per-game", type=int, default=None)
    b.add_argument("--seed", type=int, default=101)
    b.add_argument("--allow-treated", action="store_true")
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=cmd_bundles)

    c = sub.add_parser("build-challenge", help="synthetic semantic challenge set from sources")
    c.add_argument("--sources", type=Path, default=DATA_DIR / "sources.jsonl",
                   help="untreated proficiency source records")
    c.add_argument("--seed", type=int, default=101)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=cmd_build_challenge)

    r = sub.add_parser("run-challenge", help="judge every challenge item with the semantic gate")
    r.add_argument("--items", type=Path, required=True)
    r.add_argument("--judge", default="judge")
    r.add_argument("--judge-max-tokens", type=int, default=3072)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--out", type=Path, required=True)
    r.set_defaults(func=cmd_run_challenge)

    rc = sub.add_parser("report-challenge", help="sensitivity/specificity per challenge type")
    rc.add_argument("--items", type=Path, required=True)
    rc.add_argument("--out", type=Path, default=None)
    rc.set_defaults(func=cmd_report_challenge)

    features.add_features_parser(sub)
    guardrails.add_guardrails_parser(sub)
    listener.add_parser(sub)
    analysis.add_parser(sub)
    freeze.add_parser(sub)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
