"""Family A, syntactic half: dependency-based range measures via spaCy (PLAN 7.4).

spaCy is imported lazily so that the pure-Python measures still work without
it. The loaded model's name and version are reported for the frozen config.

Measures (per text, sentence-aware):
  mean_dependency_distance   mean |i - head.i| over non-punct, non-root tokens
  tree_depth_mean / max      depth of each sentence's dependency tree
  subordinate_clauses_per_sentence   advcl, ccomp, xcomp, acl, relcl, csubj(pass)
  subordinate_clause_rate    the same count per 100 words
  np_mean_length / np_mean_modifiers   noun-chunk length, non-head tokens per chunk
  clause_pattern_distinct_ratio / clause_pattern_entropy   diversity of the
      per-sentence multiset of clause-level relations (primary syntactic cue)
  n_sentences, mean_sentence_length, sd_sentence_length
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

SUBORDINATE_DEPS: frozenset = frozenset(
    {"advcl", "ccomp", "xcomp", "acl", "relcl", "csubj", "csubjpass"})
CLAUSE_PATTERN_DEPS: frozenset = SUBORDINATE_DEPS | {"parataxis"}

_NLP_CACHE: Dict[str, Any] = {}


def load_nlp(model: str = "en_core_web_sm"):
    """Load (and cache) a spaCy pipeline. Raises a clear error when absent."""
    if model in _NLP_CACHE:
        return _NLP_CACHE[model]
    try:
        import spacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "spaCy is required for syntactic features: pip install spacy && "
            f"python -m spacy download {model}") from exc
    try:
        nlp = spacy.load(model, disable=["ner"])
    except OSError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"spaCy model {model!r} is not installed: python -m spacy download {model}") from exc
    _NLP_CACHE[model] = nlp
    return nlp


def nlp_provenance(nlp) -> dict:
    import spacy
    return {"spacy_version": spacy.__version__, "model": nlp.meta.get("name"),
            "model_version": nlp.meta.get("version"), "lang": nlp.meta.get("lang")}


def _content_tokens(sent) -> list:
    return [t for t in sent if not t.is_punct and not t.is_space]


def _depth(token) -> int:
    # spaCy returns a fresh Token object from ``.head``; compare indices, not identity.
    depth = 0
    while token.head.i != token.i:
        token = token.head
        depth += 1
    return depth


def _clause_pattern(sent) -> Tuple[str, ...]:
    labels = [t.dep_ for t in sent if t.dep_ in CLAUSE_PATTERN_DEPS]
    labels += ["conj_clause" for t in sent
               if t.dep_ == "conj" and t.pos_ in ("VERB", "AUX")]
    return tuple(sorted(labels)) or ("simple",)


def entropy(counts: Dict[Any, int]) -> Optional[float]:
    """Shannon entropy (bits); one implementation, shared with ``components.stats``."""
    from components.stats import shannon_entropy
    return shannon_entropy(counts)


def analyse(text: str, nlp) -> dict:
    """Syntactic measures plus the ``(surface, lemma, pos)`` tokens for lexical use."""
    doc = nlp(text or "")
    sents = [s for s in doc.sents if _content_tokens(s)]
    tokens = [(t.text, t.lemma_, t.pos_) for t in doc if not t.is_punct and not t.is_space]
    n_words = len(tokens)

    if not sents:
        return {"tokens": tokens, "n_words": n_words, "n_sentences": 0}

    dep_distances: List[int] = []
    depths_max: List[int] = []
    depths_mean: List[float] = []
    lengths: List[int] = []
    subordinate = 0
    patterns: Counter = Counter()
    for sent in sents:
        content = _content_tokens(sent)
        lengths.append(len(content))
        dep_distances += [abs(t.i - t.head.i) for t in content if t.head.i != t.i]
        token_depths = [_depth(t) for t in content]
        depths_max.append(max(token_depths) if token_depths else 0)
        depths_mean.append(statistics.fmean(token_depths) if token_depths else 0.0)
        subordinate += sum(1 for t in sent if t.dep_ in SUBORDINATE_DEPS)
        patterns[_clause_pattern(sent)] += 1

    chunks = list(doc.noun_chunks)
    chunk_lengths = [sum(1 for t in c if not t.is_punct) for c in chunks]
    chunk_modifiers = [max(0, length - 1) for length in chunk_lengths]

    n_sents = len(sents)
    return {
        "tokens": tokens,
        "n_words": n_words,
        "n_sentences": n_sents,
        "mean_sentence_length": statistics.fmean(lengths),
        "sd_sentence_length": statistics.pstdev(lengths) if n_sents > 1 else 0.0,
        "mean_dependency_distance": (statistics.fmean(dep_distances) if dep_distances else None),
        "tree_depth_mean": statistics.fmean(depths_mean),
        "tree_depth_max": max(depths_max),
        "subordinate_clauses_per_sentence": subordinate / n_sents,
        "subordinate_clause_rate": 100.0 * subordinate / n_words if n_words else None,
        "np_mean_length": statistics.fmean(chunk_lengths) if chunk_lengths else None,
        "np_mean_modifiers": statistics.fmean(chunk_modifiers) if chunk_modifiers else None,
        "clause_pattern_distinct_ratio": len(patterns) / n_sents,
        "clause_pattern_entropy": entropy(patterns),
        "clause_patterns": {" ".join(k): v for k, v in patterns.items()},
    }


def sentences(text: str, nlp=None) -> List[str]:
    """Sentence strings, via spaCy when available, else the regex splitter."""
    if nlp is not None:
        return [s.text.strip() for s in nlp(text or "").sents if s.text.strip()]
    from components.metrics.text_normalization import split_sentences
    return split_sentences(text)
