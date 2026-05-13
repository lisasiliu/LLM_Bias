"""Template for a custom rewriter.

Copy this directory and modify to create your own rewriter.
See examples/formality/ for a working example.
"""

from lurkingwolf.rewriter import Rewriter, RewriteContext, register


@register
class MyRewriter(Rewriter):
    name = "my_rewriter"  # used with --rewriter my_rewriter

    def __init__(self, llm_client=None, **kwargs):
        self.llm_client = llm_client

    def rewrite(self, ctx: RewriteContext) -> str:
        if self.llm_client is None:
            return ctx.original_nl

        prompt = (
            f"Rewrite this game speech in your desired style. "
            f"Preserve ALL factual claims exactly.\n\n"
            f"Original: {ctx.original_nl}\n\n"
            f"Rewritten:"
        )
        response = self.llm_client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        text = response.choices[0].message.content or ""
        return text.strip() or ctx.original_nl

    def to_config(self) -> dict:
        return {"class": "MyRewriter", "params": {}}
