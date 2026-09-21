"""Pluggable text rewriter abstraction.

The 3-stage pipeline:
  Stage 1 — Predicate generation (game engine)
  Stage 2 — Deterministic NL realization (seed=101)
  Stage 3 — Pluggable rewriter (this module)

Rewriters transform Stage-2 NL text before it enters the game state.
Implement ``Rewriter.rewrite()`` and decorate with ``@register``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class RewriteContext:
    predicate: str
    original_nl: str
    phase: str
    speaker_id: str
    speaker_role_claimed: Optional[str]
    public_history: List[str] = field(default_factory=list)


class Rewriter(ABC):
    name: str = ""

    @abstractmethod
    def rewrite(self, ctx: RewriteContext) -> str:
        ...

    @property
    def applies_to_predicates(self) -> Optional[Set[str]]:
        return None

    def to_config(self) -> dict:
        return {"class": type(self).__name__, "params": {}}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], **kwargs) -> "Rewriter":
        # ``register`` keys the registry by ``cls.name`` while ``to_config``
        # writes ``type(self).__name__``, so accept either spelling.
        klass = _REGISTRY.get(cfg["class"]) or next(
            (k for k in _REGISTRY.values() if k.__name__ == cfg["class"]), None
        )
        if not klass:
            raise ValueError(f"Unknown rewriter: {cfg['class']}")
        return klass(**cfg.get("params", {}), **kwargs)


_REGISTRY: Dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get_rewriter(name: str) -> type:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown rewriter {name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_rewriters() -> List[str]:
    return list(_REGISTRY.keys())
