"""Three-pass majority vote over a stochastic classifier.

GPT-5-family models cannot be run at temperature 0, so one call can return a
different label from the next. We therefore ask the SAME question three times
independently and keep the label that at least two passes agree on. If all
three passes disagree there is no majority and the item is reported as
``no_majority`` (it counts as a miss against any reference label), never
resolved by picking one.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence, Tuple

NO_MAJORITY = "no_majority"
FAILED = "failed"


def majority(labels: Sequence[Optional[str]], *, passes: int = 3) -> Tuple[str, int]:
    """``(label, votes)`` for the label with a strict majority of ``passes``.

    Passes that errored (``None``) count as no vote. Returns ``FAILED`` when no
    pass produced a label and ``NO_MAJORITY`` when none has more than half of
    the passes asked for.
    """
    votes = Counter(l for l in labels if l)
    if not votes:
        return FAILED, 0
    label, n = votes.most_common(1)[0]
    if n * 2 > passes:
        return label, n
    return NO_MAJORITY, n
