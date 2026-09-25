"""Family A, lexical half: SUBTLEX-US frequency, MATTR, MTLD, repetition (PLAN 7).

Inputs are token lists so that none of this depends on spaCy; the caller
supplies ``(surface, lemma, pos)`` triples from ``syntactic.analyse`` when it
has them, or bare surface tokens otherwise (and the output says which).

References
  Brysbaert & New (2009) SUBTLEX-US; ``Lg10CD`` = log10 contextual diversity.
  Covington & McFall (2010) MATTR, moving-average type-token ratio.
  McCarthy & Jarvis (2010) MTLD, measure of textual lexical diversity.
"""

from __future__ import annotations

import csv
import hashlib
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from components.metrics.text_normalization import DOMAIN_TERMS, is_game_id

# Universal POS tags counted as content words. AUX is deliberately excluded so
# "main verb" means VERB.
CONTENT_POS: frozenset = frozenset({"NOUN", "VERB", "ADJ", "ADV", "PROPN"})

Token = Tuple[str, str, str]   # (surface, lemma, upos)


class SubtlexNorms:
    """SUBTLEX-US lookup table keyed on the lower-cased word.

    Reads the delimited export (tab or comma) with a header row containing a
    ``Word`` column and the requested measure (default ``Lg10CD``). Records the
    file hash so the frozen config can pin the exact norms used.
    """

    def __init__(self, path: Path, measure: str = "Lg10CD"):
        self.path = Path(path)
        self.measure = measure
        self.table: Dict[str, float] = {}
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8-sig", errors="replace")     # tolerate an Excel/Windows BOM
        sample = text[:4096]
        delimiter = "\t" if sample.count("\t") >= sample.count(",") else ","
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"{self.path} has no header row")
        columns = {c.lower(): c for c in reader.fieldnames}
        if "word" not in columns or measure.lower() not in columns:
            raise ValueError(
                f"{self.path} must have 'Word' and '{measure}' columns; "
                f"found {reader.fieldnames}")
        word_col, value_col = columns["word"], columns[measure.lower()]
        for row in reader:
            word = (row.get(word_col) or "").strip().lower()
            try:
                value = float(row.get(value_col) or "")
            except ValueError:
                continue
            if word and word not in self.table:
                self.table[word] = value

    def lookup(self, surface: str, lemma: Optional[str] = None) -> Tuple[Optional[float], str]:
        """``(value, how)`` with ``how`` in surface / lemma / missing (PLAN 7.1 order)."""
        s = (surface or "").lower()
        if s in self.table:
            return self.table[s], "surface"
        l = (lemma or "").lower()
        if l and l in self.table:
            return self.table[l], "lemma"
        return None, "missing"

    def provenance(self) -> dict:
        return {"path": str(self.path), "measure": self.measure,
                "n_entries": len(self.table), "sha256": self.sha256}


def _as_triples(tokens: Sequence) -> List[Token]:
    out: List[Token] = []
    for tok in tokens:
        if isinstance(tok, str):
            out.append((tok, tok, ""))
        else:
            surface, lemma, pos = (list(tok) + ["", ""])[:3]
            out.append((str(surface), str(lemma or surface), str(pos or "")))
    return out


def content_tokens(tokens: Sequence, *, include_domain: bool = False) -> List[Token]:
    """Content-word triples, minus game ids and (by default) domain terms.

    With POS available, content = CONTENT_POS. Without POS (bare strings) every
    non-id token counts, and callers should report that lemmas were unavailable.
    """
    out: List[Token] = []
    for surface, lemma, pos in _as_triples(tokens):
        if is_game_id(surface):
            continue
        if pos and pos not in CONTENT_POS:
            continue
        if not include_domain and (surface.lower() in DOMAIN_TERMS or lemma.lower() in DOMAIN_TERMS):
            continue
        out.append((surface, lemma, pos))
    return out


def lexical_frequency(tokens: Sequence, norms: Optional[SubtlexNorms], *,
                      include_domain: bool = False) -> dict:
    """Mean/median ``Lg10CD`` over content words, with lookup coverage (PLAN 7.1).

    Expected direction for Low: HIGHER mean (more common words).
    """
    content = content_tokens(tokens, include_domain=include_domain)
    n_domain = sum(1 for s, l, _ in _as_triples(tokens)
                   if not is_game_id(s) and (s.lower() in DOMAIN_TERMS or l.lower() in DOMAIN_TERMS))
    result = {"n_content": len(content), "n_domain_terms": n_domain,
              "n_game_ids": sum(1 for s, _, _ in _as_triples(tokens) if is_game_id(s)),
              "subtlex_mean": None, "subtlex_median": None, "subtlex_coverage": None,
              "subtlex_lookup": {"surface": 0, "lemma": 0, "missing": 0},
              "subtlex_available": norms is not None}
    if norms is None or not content:
        return result
    values: List[float] = []
    for surface, lemma, _ in content:
        value, how = norms.lookup(surface, lemma)
        result["subtlex_lookup"][how] += 1
        if value is not None:
            values.append(value)
    if values:
        result["subtlex_mean"] = statistics.fmean(values)
        result["subtlex_median"] = statistics.median(values)
    result["subtlex_coverage"] = len(values) / len(content)
    return result


def ttr(tokens: Sequence[str]) -> Optional[float]:
    toks = [t.lower() for t in tokens]
    return len(set(toks)) / len(toks) if toks else None


def mattr(tokens: Sequence[str], window: int = 50) -> Optional[float]:
    """Moving-average TTR over every window of ``window`` tokens.

    ``None`` when the text is shorter than the window: MATTR is undefined there
    and silently falling back to plain TTR would mix two different measures.
    """
    toks = [t.lower() for t in tokens]
    if window <= 0 or len(toks) < window:
        return None
    counts: Counter = Counter(toks[:window])
    total = len(counts)
    n_windows = len(toks) - window + 1
    for i in range(window, len(toks)):
        outgoing, incoming = toks[i - window], toks[i]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[incoming] += 1
        total += len(counts)
    return total / (n_windows * window)


def _mtld_one_way(toks: List[str], threshold: float) -> Optional[float]:
    factors = 0.0
    start = 0
    types: set = set()
    for i, tok in enumerate(toks):
        types.add(tok)
        if len(types) / (i - start + 1) <= threshold:
            factors += 1.0
            start = i + 1
            types = set()
    remaining = len(toks) - start
    if remaining > 0:
        partial_ttr = len(types) / remaining
        factors += (1.0 - partial_ttr) / (1.0 - threshold)
    return len(toks) / factors if factors > 0 else None


def mtld(tokens: Sequence[str], threshold: float = 0.72) -> Optional[float]:
    """McCarthy & Jarvis (2010): mean of the forward and backward MTLD."""
    toks = [t.lower() for t in tokens]
    if len(toks) < 10:
        return None
    forward = _mtld_one_way(toks, threshold)
    backward = _mtld_one_way(list(reversed(toks)), threshold)
    if forward is None or backward is None:
        return None
    return (forward + backward) / 2.0


def repetition(content_lemmas: Sequence[str], top_k: int = 5) -> dict:
    """Content-lemma repetition (PLAN 7.3). Expected direction for Low: HIGHER.

    ``repetition_rate`` = 1 - types/tokens over content lemmas;
    ``top{k}_share`` = share of content tokens taken by the k most repeated lemmas.
    """
    lemmas = [l.lower() for l in content_lemmas if l]
    if not lemmas:
        return {"n_content_lemmas": 0, "repetition_rate": None, f"top{top_k}_share": None}
    counts = Counter(lemmas)
    top = sum(c for _, c in counts.most_common(top_k))
    return {"n_content_lemmas": len(lemmas),
            "repetition_rate": 1.0 - len(counts) / len(lemmas),
            f"top{top_k}_share": top / len(lemmas)}


@dataclass
class LexicalProfile:
    """All lexical measures for one text, computed from one token list."""

    n_tokens: int
    n_tokens_no_ids: int
    ttr: Optional[float]
    mattr: Optional[float]
    mattr_window: int
    mtld: Optional[float]
    frequency: dict
    repetition: dict
    lemmas_available: bool

    def to_dict(self) -> dict:
        flat = {"n_tokens": self.n_tokens, "n_tokens_no_ids": self.n_tokens_no_ids,
                "ttr": self.ttr, "mattr": self.mattr, "mattr_window": self.mattr_window,
                "mtld": self.mtld, "lemmas_available": self.lemmas_available}
        flat.update(self.frequency)
        flat.update(self.repetition)
        return flat


def lexical_profile(tokens: Sequence, norms: Optional[SubtlexNorms] = None, *,
                    mattr_window: int = 50, include_domain: bool = False) -> LexicalProfile:
    triples = _as_triples(tokens)
    lemmas_available = any(pos for _, _, pos in triples)
    surfaces = [s for s, _, _ in triples]
    no_ids = [s for s in surfaces if not is_game_id(s)]
    if lemmas_available:
        content = content_tokens(triples, include_domain=include_domain)
        frequency = lexical_frequency(triples, norms, include_domain=include_domain)
        rep = repetition([l for _, l, _ in content])
    else:
        # Without POS/lemmas "content words" would silently become every token,
        # function words included: not the PLAN 7 measure. Leave them undefined.
        frequency = {"n_content": None, "n_domain_terms": None, "n_game_ids": len(surfaces) - len(no_ids),
                     "subtlex_mean": None, "subtlex_median": None, "subtlex_coverage": None,
                     "subtlex_lookup": None, "subtlex_available": norms is not None,
                     "content_measures_unavailable": "no POS tags / lemmas"}
        rep = {"n_content_lemmas": None, "repetition_rate": None, "top5_share": None}
    return LexicalProfile(
        n_tokens=len(surfaces), n_tokens_no_ids=len(no_ids),
        ttr=ttr(no_ids), mattr=mattr(no_ids, mattr_window), mattr_window=mattr_window,
        mtld=mtld(no_ids), frequency=frequency, repetition=rep,
        lemmas_available=lemmas_available,
    )
