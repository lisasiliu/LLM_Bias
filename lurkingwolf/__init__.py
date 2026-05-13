"""LurkingWolf — LLM agents playing Werewolf."""

from .utils.llm_client import LLMClient
from .rewriter import Rewriter, RewriteContext, register, get_rewriter, list_rewriters
from .config import load_config, ExperimentConfig, ComparisonSpec, ReplaySpec

__all__ = [
    "LLMClient",
    "Rewriter",
    "RewriteContext",
    "register",
    "get_rewriter",
    "list_rewriters",
    "load_config",
    "ExperimentConfig",
    "ComparisonSpec",
    "ReplaySpec",
]
