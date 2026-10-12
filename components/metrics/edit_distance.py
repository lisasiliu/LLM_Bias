"""Normalized token edit distance, the neutral-matching statistic (PLAN 3.3).

Levenshtein distance over lower-cased word tokens divided by the length of the
longer sequence, so 0 means identical wording and 1 means nothing shared.
Deterministic and dependency-free so that the number a pilot froze can be
recomputed anywhere.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from components.metrics.text_normalization import tokenize


def token_levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    """Classic dynamic-programming Levenshtein distance over token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, 1):
        current = [i]
        for j, tok_b in enumerate(b, 1):
            cost = 0 if tok_a == tok_b else 1
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + cost))  # substitution
        previous = current
    return previous[-1]


def normalized_token_edit_distance(text_a: str, text_b: str, *,
                                   tokenizer: Callable[[str], List[str]] = tokenize,
                                   lowercase: bool = True) -> float:
    a = tokenizer(text_a or "")
    b = tokenizer(text_b or "")
    if lowercase:
        a = [t.lower() for t in a]
        b = [t.lower() for t in b]
    longest = max(len(a), len(b))
    if longest == 0:
        return 0.0
    return token_levenshtein(a, b) / longest
