"""Frozen artifacts of the politeness flip run.

The run manifest pins the four strategy prompts, the instruments that read the
delivered speech, and the engine sources that produced the replays, so a later
edit to any of them is detectable from the recorded run alone.  Hashes are taken
from a git revision (``git_rev`` in the manifest), which is the tree the replays
ran from; ``verify`` recomputes them from that same revision and names whatever
no longer matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import List, Optional

from components.judges.blind_politeness_classifier import (
    PROMPT_VERSION as CLASSIFIER_PROMPT_VERSION, system_prompt as classifier_prompt)
from components.judges.inverse_predicate_parser import build_system_prompt as parser_system_prompt
from components.rewriters.politeness_rewriter import POLITENESS_STRATEGIES, PolitenessRewriter
from lurkingwolf.rewriter import RewriteContext

# The context the politeness validation hashed its generation prompts against, so
# `rewriter_canonical_prompt_sha256` is comparable with its provenance.json.
CANONICAL_CONTEXT = RewriteContext(
    predicate="estimate(D1, P3, P7, werewolf)",
    original_nl=("P7 is a werewolf. He pushed the vote onto P2 without ever "
                 "giving a read of his own."),
    phase="day_speech", speaker_id="P3", speaker_role_claimed="villager",
    public_history=["[s1|P1] claim_role(D1, P1, seer)"])

# Sources that decide what a treated utterance says, how it is delivered into the
# replay, and how the delivered speech is labelled.
SOURCE_FILES = (
    "components/rewriters/politeness_rewriter.py",
    "components/judges/blind_politeness_classifier.py",
    "components/judges/inverse_predicate_parser.py",
    "lurkingwolf/agent.py",
    "lurkingwolf/config.py",
    "lurkingwolf/eval/replay.py",
    "lurkingwolf/eval/replay_cli.py",
    "lurkingwolf/eval/runner.py",
    "experiments/politeness_flip/config.yaml",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_sha256(path: str, rev: Optional[str] = None) -> str:
    """Hash of ``path`` as of ``rev``, or of the working tree when ``rev`` is None."""
    if rev is None:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    blob = subprocess.run(["git", "show", f"{rev}:{path}"], check=True, capture_output=True).stdout
    return hashlib.sha256(blob).hexdigest()


def git_rev(rev: str = "HEAD") -> str:
    return subprocess.run(["git", "rev-parse", rev], check=True, capture_output=True,
                          text=True).stdout.strip()


def frozen_hashes(rev: Optional[str] = None) -> dict:
    """Prompt, instrument and source hashes for the politeness flip.

    ``rewriter_prompt_sha256`` is the TEMPLATE hash, with the predicate-definition
    slot left as a placeholder: one value per strategy whatever the predicate, and
    the value every rewrite row carries.
    """
    rewriters = {s: PolitenessRewriter(llm_client=None, strategy=s)
                 for s in sorted(POLITENESS_STRATEGIES)}
    return {
        "rewriter": PolitenessRewriter.name,
        "strategies": sorted(POLITENESS_STRATEGIES),
        "rewriter_prompt_sha256": {s: rw.template_sha256() for s, rw in rewriters.items()},
        "rewriter_canonical_prompt_sha256": {s: rw.prompt_sha256(CANONICAL_CONTEXT)
                                             for s, rw in rewriters.items()},
        "strategy_definition_sha256": {s: sha256_text(t)
                                       for s, t in sorted(POLITENESS_STRATEGIES.items())},
        "politeness_classifier_prompt_version": CLASSIFIER_PROMPT_VERSION,
        "politeness_classifier_prompt_sha256": sha256_text(classifier_prompt(CLASSIFIER_PROMPT_VERSION)),
        "parser_prompt_sha256": sha256_text(parser_system_prompt()),
        "source_sha256": {path: git_sha256(path, rev) for path in SOURCE_FILES},
    }


def verify(manifest: str | Path) -> List[str]:
    """Names of the frozen artifacts whose current hash differs from the manifest's."""
    configuration = json.loads(Path(manifest).read_text(encoding="utf-8")).get("configuration", {})
    recorded = configuration.get("hashes", {})
    problems: List[str] = []
    for key, current in frozen_hashes(configuration.get("git_rev")).items():
        if isinstance(current, dict):
            for sub, value in current.items():
                if (recorded.get(key) or {}).get(sub) != value:
                    problems.append(f"{key}.{sub}")
        elif recorded.get(key) != current:
            problems.append(key)
    return problems


def main(argv: List[str] | None = None) -> int:
    from experiments.politeness_flip.code.analyze import RUN_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(RUN_DIR / "manifest.json"))
    parser.add_argument("--rev", help="git revision the replays ran from; omit for the working tree")
    parser.add_argument("--write", action="store_true",
                        help="recompute the hashes into the manifest instead of checking them")
    args = parser.parse_args(argv)

    path = Path(args.manifest)
    if args.write:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        configuration = manifest.setdefault("configuration", {})
        if args.rev:
            configuration["git_rev"] = git_rev(args.rev)
        configuration["hashes"] = frozen_hashes(configuration.get("git_rev"))
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1, default=str) + "\n",
                        encoding="utf-8")
        print(f"wrote {path}")
        return 0

    problems = verify(path)
    for name in problems:
        print(f"  CHANGED {name}")
    if problems:
        print(f"{len(problems)} frozen artifact(s) no longer match {path}")
        return 1
    print(f"all frozen artifacts match {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
