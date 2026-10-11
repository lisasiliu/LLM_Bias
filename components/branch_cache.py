"""Branch-point replay: reuse the baseline's model responses up to the first treated utterance.

A replay from a day-start checkpoint normally draws fresh model responses for the whole day, so two
untreated replays already differ (run-to-run noise). Here every replay, treated or not, answers its
model calls from the baseline's `game_log.jsonl` (keyed by the exact prompts) until the first utterance
the rewriter is asked about, then switches the cache off and draws fresh responses. All arms therefore
share one history up to that utterance; the no-change control (`BranchRewriter` with no inner rewriter)
branches at the same point, so its change rate is the noise floor at the branch point.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Optional

from lurkingwolf.rewriter import Rewriter, RewriteContext


class BranchCache:
    """Baseline responses by (system prompt, user prompt); active until `switch_off`."""

    def __init__(self, game_log: Optional[Path] = None):
        self.responses: Dict[tuple, deque] = defaultdict(deque)
        self.active = True
        self.branch: Optional[Dict[str, Any]] = None
        self.hits = 0
        self.misses_before_branch = 0      # live calls while the cache was on: history is no longer shared
        self.calls_after_branch = 0
        if game_log is not None:
            for line in Path(game_log).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if "response" in row:
                    self.responses[(row["system_prompt"], row["user_prompt"])].append(row["response"])

    def lookup(self, system: str, user: str) -> Optional[str]:
        if self.active:
            queue = self.responses.get((system, user))
            if queue:
                self.hits += 1
                return queue.popleft()
            self.misses_before_branch += 1
        else:
            self.calls_after_branch += 1
        return None

    def switch_off(self, ctx: RewriteContext) -> None:
        if self.active:
            self.active = False
            self.branch = {"day": ctx.day, "phase": ctx.phase, "speaker": ctx.speaker_id,
                           "predicate": ctx.predicate, "original_nl": ctx.original_nl,
                           "history_length": len(ctx.public_history)}

    def stats(self) -> dict:
        return {"branched": self.branch is not None, "branch": self.branch, "cache_hits": self.hits,
                "live_calls_before_branch": self.misses_before_branch,
                "live_calls_after_branch": self.calls_after_branch}


class _Msg:
    def __init__(self, content): self.content = content
class _Choice:
    def __init__(self, content): self.message = _Msg(content)
class _Resp:
    def __init__(self, content): self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, inner, cache: BranchCache):
        self.inner, self.cache = inner, cache

    def create(self, *, messages, **kwargs):
        by_role = {m["role"]: m["content"] for m in messages}
        cached = self.cache.lookup(by_role.get("system", ""), by_role.get("user", ""))
        if cached is not None:
            return _Resp(cached)
        return self.inner.chat.completions.create(messages=messages, **kwargs)


class CachedClient:
    """OpenAI-style client that answers from a `BranchCache` while it is active."""

    def __init__(self, inner, cache: BranchCache):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions(inner, cache)


class BranchRewriter(Rewriter):
    """Switches the cache off at the first utterance in scope, then delegates.

    With `inner=None` it is the no-change control: the natural text is kept but the branch happens at
    exactly the point a treated arm would branch."""

    name = "branch_rewriter"

    def __init__(self, cache: BranchCache, acts, inner: Optional[Rewriter] = None):
        self.cache, self.acts, self.inner = cache, tuple(acts), inner

    @property
    def applies_to_predicates(self):
        return set(self.acts)

    def bind(self, agents: dict) -> None:
        if hasattr(self.inner, "bind"):
            self.inner.bind(agents)

    def to_config(self) -> dict:
        return self.inner.to_config() if self.inner else {"class": "NoChange", "params": {"acts": list(self.acts)}}

    @property
    def counters(self) -> Optional[dict]:
        return getattr(self.inner, "counters", None)

    def rewrite(self, ctx: RewriteContext) -> str:
        self.cache.switch_off(ctx)
        return self.inner.rewrite(ctx) if self.inner else ctx.original_nl
