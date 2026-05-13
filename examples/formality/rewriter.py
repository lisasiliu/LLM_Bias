"""Formality rewriter — formal vs casual speech styles.

Usage:
    python -m lurkingwolf.run_game --experiment nl --rewriter formality \
        --rewriter-params '{"level": "formal"}'
"""

from __future__ import annotations

from lurkingwolf.rewriter import Rewriter, RewriteContext, register
from lurkingwolf.utils.gen_config import thinking_extra_body
from lurkingwolf.utils.speech_nl import strip_think_tags


_PROMPTS = {
    "formal": (
        "Rewrite the following game speech in a formal, professional register. "
        "Use complete sentences, polite phrasing, and sophisticated vocabulary. "
        "Preserve ALL factual claims, accusations, and game-relevant information exactly. "
        "Do NOT add, remove, or change any game facts.\n\n"
        "Original: {original}\n\n"
        "Formal version:"
    ),
    "casual": (
        "Rewrite the following game speech in a casual, relaxed register. "
        "Use contractions, informal phrasing, and everyday vocabulary. "
        "Preserve ALL factual claims, accusations, and game-relevant information exactly. "
        "Do NOT add, remove, or change any game facts.\n\n"
        "Original: {original}\n\n"
        "Casual version:"
    ),
}


@register
class FormalityRewriter(Rewriter):
    name = "formality"

    def __init__(self, level: str = "formal", llm_client=None, model: str = "Qwen/Qwen3.5-27B", **kwargs):
        if level not in _PROMPTS:
            raise ValueError(f"level must be 'formal' or 'casual', got {level!r}")
        self.level = level
        self.llm_client = llm_client
        self.model = model

    def rewrite(self, ctx: RewriteContext) -> str:
        if self.llm_client is None:
            return ctx.original_nl

        prompt = _PROMPTS[self.level].format(original=ctx.original_nl)
        extra = thinking_extra_body(self.model)
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
            **extra,
        )
        text = strip_think_tags(response.choices[0].message.content or "")
        return text.strip() or ctx.original_nl

    def to_config(self) -> dict:
        return {"class": "FormalityRewriter", "params": {"level": self.level}}
