"""Freeze prompts, measurement, inference, selection and enforcement source code.

Formal runs stop on any code or protocol mismatch. Statistical settings and
instrument/model identities are declared in the validated configuration.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .english_validator import SEMANTIC_RUBRIC_VERSION, semantic_prompt_sha
from components.judges.inverse_predicate_parser import build_system_prompt as parser_system_prompt
from components.judges.listener_probe import PROMPT_VERSION as PROBE_VERSION, probe_prompt_sha256
from components.rewriters.language_rewriter import PROMPT_VERSION, LanguageProficiencyRewriter
from lurkingwolf.rewriter import RewriteContext

from .common import FROZEN_CONFIG, load_json, sha256_text

# A fixed context so the rewriter prompt hashes do not depend on any dataset.
CANONICAL_CONTEXT = RewriteContext(
    predicate="estimate(D1, P3, P7, werewolf)",
    original_nl=("P7 is a werewolf. He pushed the vote onto P2 without ever giving a read of his own."),
    phase="day_speech", speaker_id="P3", speaker_role_claimed="villager",
    public_history=["[s1|P1] claim_role(D1, P1, seer)"])

# Modules whose source defines a measurement or a check. Imported by dotted
# name so a package ``__init__`` that re-exports a function of the same name
# (``components.judges.structural_check``) cannot shadow the module.
HASHED_MODULES: tuple = (
    "components.judges.structural_check",
    "components.judges.inverse_predicate_parser",
    "components.judges.listener_probe",
    "components.helpers",
    "experiments.proficiency_validation.code.common",
    "experiments.proficiency_validation.code.bundles",
    "experiments.proficiency_validation.code.features",
    "components.rewriters.language_rewriter",
    "components.metrics.text_normalization",
    "components.metrics.edit_distance",
    "components.metrics.lexical",
    "components.metrics.syntactic",
    "components.metrics.grammar",
    "experiments.proficiency_validation.code.semantic_gate",
    "experiments.proficiency_validation.code.english_validator",
    "experiments.proficiency_validation.code.models",
    "experiments.proficiency_validation.code.protocol",
    "experiments.proficiency_validation.code.human_validation",
    "experiments.proficiency_validation.code.generate",
    "experiments.proficiency_validation.code.guardrails",
    "experiments.proficiency_validation.code.listener",
    "experiments.proficiency_validation.code.analysis",
    "experiments.proficiency_validation.code.confirmatory",
    "components.stats",
    "experiments.proficiency_validation.code.frozen",
    "experiments.proficiency_validation.code.freeze",
    "experiments.proficiency_validation.code.cli",
    "experiments.proficiency_validation.code.checkpoint",
)


def module_file_sha256(dotted: str) -> str:
    module = importlib.import_module(dotted)
    path = inspect.getsourcefile(module)
    if not path:
        raise RuntimeError(f"no source file for module {dotted}")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes(n_neutral_candidates: int = 5) -> dict:
    """Prompt TEMPLATE hashes (definition slot fixed, so one value per condition
    regardless of predicate), plus the probe, rubric, parser and module hashes."""
    rw = LanguageProficiencyRewriter(llm_client=None, condition="low")
    prompts = {condition: rw.template_sha256(condition) for condition in ("low", "neutral", "high")}
    prompts["neutral_candidates"] = rw.template_sha256("neutral", candidates=True, n=n_neutral_candidates)
    prompts["canonical_low_instance"] = rw.prompt_sha256(CANONICAL_CONTEXT, "low")
    return {
        "rewriter_prompt_version": PROMPT_VERSION,
        "rewriter_prompt_sha256": prompts,
        "n_neutral_candidates": n_neutral_candidates,
        "listener_probe_version": PROBE_VERSION,
        "listener_probe_sha256": probe_prompt_sha256(),
        "semantic_rubric_version": SEMANTIC_RUBRIC_VERSION,
        "semantic_rubric_sha256": semantic_prompt_sha(),
        "parser_prompt_sha256": sha256_text(parser_system_prompt()),
        "module_source_sha256": {name: module_file_sha256(name) for name in HASHED_MODULES},
    }


def verify_hashes(frozen: dict) -> List[str]:
    """Names of every frozen artifact whose current hash differs. Empty = clean."""
    recorded = frozen.get("hashes", {}) or {}
    problems: List[str] = []

    def check(key: str, expected, current) -> None:
        if isinstance(current, dict):
            for sub, value in current.items():
                if (expected or {}).get(sub) != value:
                    problems.append(f"{key}.{sub}")
            for sub in (expected or {}):
                if sub not in current:
                    problems.append(f"{key}.{sub} (recorded but no longer computed)")
        elif expected != current:
            problems.append(key)

    current = frozen_hashes(int(frozen.get("n_neutral_candidates") or recorded.get("n_neutral_candidates") or 5))
    for key, value in current.items():
        check(key, recorded.get(key), value)
    return problems


def expected_template_hash(frozen: dict, condition: str) -> Optional[str]:
    """The template hash a rewrite row in ``condition`` must carry."""
    prompts = (frozen.get("hashes") or {}).get("rewriter_prompt_sha256") or {}
    return prompts.get("neutral_candidates" if condition == "neutral" else condition)


def enforce_frozen(path: Optional[Path] = None, *, check_models: bool = False) -> dict:
    """Load the frozen config and stop unless the code still matches it."""
    path = Path(path) if path else FROZEN_CONFIG
    if not path.exists():
        raise SystemExit(f"no frozen config at {path}")
    frozen = load_json(path)
    from .protocol import validate_design
    problems = validate_design(frozen) + verify_hashes(frozen)
    if check_models:
        from .models import model_identity
        for handle, expected in frozen.get("models", {}).get("identities", {}).items():
            if model_identity(handle) != expected:
                problems.append(f"model identity/settings changed: {handle}")
    if problems:
        raise SystemExit(f"code no longer matches {path}: {', '.join(problems)}. "
                         "Restore the frozen protocol. Protocol changes require a new version and fresh confirmatory data.")
    return frozen


def pin(args, frozen: dict, mapping: Dict[str, tuple], *, log=print) -> None:
    """Set ``args.<attr>`` from the frozen config for each ``attr -> key path``.

    Values already equal are left alone; differing values are overwritten and
    reported, so a command run under ``--frozen`` cannot silently deviate.
    """
    for attr, keys in mapping.items():
        node = frozen
        for key in keys:
            node = node.get(key) if isinstance(node, dict) else None
        if node is None:
            raise ValueError(f"missing frozen parameter: {'.'.join(keys)}")
        if getattr(args, attr, None) != node:
            log(f"  pinned {attr} = {node!r} (from frozen config)", file=sys.stderr)
            setattr(args, attr, node)
