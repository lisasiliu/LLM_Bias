"""Statistics for the validation experiments, dependency-free (PLAN 12).

The independent unit is the GAME, so uncertainty comes from resampling or
sign-flipping whole game clusters rather than individual bundles:

  cluster_bootstrap        percentile CI of any statistic of paired differences
  cluster_sign_flip_test   exact-style two-sided p for mean difference = 0
  tost_equivalence         equivalence via the (1 - 2 alpha) cluster-bootstrap CI
  benjamini_hochberg       FDR within a metric family
  bradley_terry            MM-estimated strengths from pairwise choices, with
                           pseudo-counts so a unanimous pair stays finite
  wilson_interval, paired_cohens_dz, descriptive helpers

Nothing here calls numpy or scipy, so every number a report prints can be
recomputed on a plain interpreter.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Clusters = Mapping[str, Sequence[float]]


# ── descriptive ──

def mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def sd(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def median(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """Linear-interpolation percentile, ``p`` in [0, 100]."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def paired_cohens_dz(diffs: Sequence[float]) -> Optional[float]:
    """Standardized paired effect: mean difference / SD of differences."""
    vals = [d for d in diffs if d is not None]
    if len(vals) < 2:
        return None
    spread = statistics.stdev(vals)
    return statistics.fmean(vals) / spread if spread > 0 else None


def flatten(clusters: Clusters) -> List[float]:
    return [v for values in clusters.values() for v in values if v is not None]


# ── cluster resampling ──

def cluster_bootstrap(clusters: Clusters, stat: Callable[[Sequence[float]], Optional[float]] = mean,
                      *, n_boot: int = 2000, seed: int = 101, alpha: float = 0.05) -> dict:
    """Percentile CI for ``stat`` of the pooled values, resampling clusters (games)."""
    keys = [k for k, v in clusters.items() if any(x is not None for x in v)]
    if len(keys) < 2:
        values = flatten(clusters)
        return {"estimate": stat(values) if values else None, "ci_low": None, "ci_high": None,
                "n_clusters": len(keys), "n_values": len(values), "n_boot": 0,
                "alpha": alpha, "draws": [], "reason": "fewer than two independent games"}
    values = flatten(clusters)
    estimate = stat(values)
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        sample = [clusters[rng.choice(keys)] for _ in keys]
        pooled = [v for vals in sample for v in vals if v is not None]
        s = stat(pooled)
        if s is not None:
            draws.append(s)
    return {"estimate": estimate,
            "ci_low": percentile(draws, 100 * alpha / 2),
            "ci_high": percentile(draws, 100 * (1 - alpha / 2)),
            "n_clusters": len(keys), "n_values": len(values), "n_boot": len(draws),
            "alpha": alpha, "draws": draws}


def cluster_sign_flip_test(clusters: Clusters, *, n_perm: int = 5000, seed: int = 101) -> dict:
    """Two-sided p for H0: mean paired difference = 0, flipping signs per cluster.

    Under H0 the paired differences are symmetric about zero and a game's
    differences can be flipped together; this respects the within-game
    dependence that an ordinary paired t-test would ignore.
    """
    groups = [[v for v in vals if v is not None] for vals in clusters.values()]
    groups = [g for g in groups if g]
    values = [v for g in groups for v in g]
    if len(groups) < 2 or not values:
        return {"p_value": None, "observed_mean": mean(values), "n_perm": 0}
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    sums = [sum(g) for g in groups]
    n = len(values)
    exceed = 0
    for _ in range(n_perm):
        total = sum(s if rng.random() < 0.5 else -s for s in sums)
        if abs(total / n) >= abs(observed) - 1e-12:
            exceed += 1
    return {"p_value": (exceed + 1) / (n_perm + 1), "observed_mean": observed, "n_perm": n_perm}


def proportion_bootstrap(clusters: Clusters, *, n_boot=2000, seed=101, alpha=.05):
    """Game bootstrap; constant data use a conservative bounded-mean Hoeffding CI."""
    result = cluster_bootstrap(clusters, n_boot=n_boot, seed=seed, alpha=alpha)
    if result["n_clusters"] >= 2 and result["ci_low"] == result["ci_high"]:
        sizes = [sum(v is not None for v in values) for values in clusters.values()]
        total = sum(sizes)
        width = math.sqrt(sum((n / total) ** 2 for n in sizes) * math.log(2 / alpha) / 2)
        result["ci_low"] = max(0, result["estimate"] - width)
        result["ci_high"] = min(1, result["estimate"] + width)
        result["interval_method"] = "Hoeffding fallback for constant game data"
    else:
        result["interval_method"] = "percentile game bootstrap"
    return result


def tost_equivalence(clusters: Clusters, bound: float, *, alpha: float = 0.05,
                     n_boot: int = 2000, seed: int = 101,
                     bootstrap: Optional[dict] = None) -> dict:
    """Two one-sided tests via the (1 - 2 alpha) cluster-bootstrap CI.

    Equivalent when the whole 90% CI (alpha = 0.05) of the mean difference lies
    inside [-bound, +bound] (Schuirmann, 1987; bootstrap form). Pass the result
    of ``cluster_bootstrap`` (with its ``draws``) as ``bootstrap`` to reuse the
    same draws instead of resampling again.
    """
    ci = bootstrap if bootstrap and bootstrap.get("draws") else cluster_bootstrap(
        clusters, mean, n_boot=n_boot, seed=seed, alpha=2 * alpha)
    draws = ci.get("draws", [])
    lo = percentile(draws, 100 * alpha)
    hi = percentile(draws, 100 * (1 - alpha))
    equivalent = None if lo is None or hi is None else (lo > -bound and hi < bound)
    if lo == hi:
        # A degenerate empirical resampling distribution cannot establish a
        # population equivalence margin, even when every observed difference is zero.
        equivalent = None
    return {"bound": bound, "estimate": ci["estimate"], "ci_low": lo, "ci_high": hi,
            "ci_level": 1 - 2 * alpha, "equivalent": equivalent,
            "n_clusters": ci["n_clusters"], "n_values": ci["n_values"]}


# ── multiplicity ──

def benjamini_hochberg(pvalues: Mapping[str, Optional[float]], q: float = 0.05) -> Dict[str, dict]:
    """BH step-up; returns adjusted p (q-values) and rejections per key."""
    items = [(k, p) for k, p in pvalues.items() if p is not None]
    m = len(items)
    out: Dict[str, dict] = {k: {"p": p, "q": None, "reject": None} for k, p in pvalues.items()}
    if m == 0:
        return out
    ordered = sorted(items, key=lambda kp: kp[1])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        key, p = ordered[rank - 1]
        running = min(running, p * m / rank)
        adjusted[rank - 1] = min(running, 1.0)
    for (key, p), adj in zip(ordered, adjusted):
        out[key] = {"p": p, "q": adj, "reject": adj <= q}
    return out


# ── proportions ──

def wilson_interval(successes: int, n: int, z: float = 1.959964) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if n <= 0:
        return None, None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def shannon_entropy(counts: Mapping[str, int]) -> Optional[float]:
    total = sum(counts.values())
    if total == 0:
        return None
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


# ── Bradley-Terry ──

def bradley_terry(comparisons: Iterable[Tuple[str, str]], *, items: Optional[Sequence[str]] = None,
                  prior: float = 0.5, max_iter: int = 2000, tol: float = 1e-10) -> Dict[str, float]:
    """Log-strengths (mean zero) from ``(winner, loser)`` pairs, Hunter's (2004) MM.

    ``prior`` pseudo-wins are added in BOTH directions for every pair that
    occurs, so a unanimous outcome yields a large finite separation instead of
    infinity. Report the value of ``prior`` with the estimate.
    """
    comps = list(comparisons)
    names = list(items) if items else sorted({x for pair in comps for x in pair})
    if len(names) < 2:
        return {n: 0.0 for n in names}
    wins: Dict[Tuple[str, str], float] = {}
    for winner, loser in comps:
        wins[(winner, loser)] = wins.get((winner, loser), 0.0) + 1.0
    pairs = {tuple(sorted(k)) for k in wins}
    for a, b in pairs:
        wins[(a, b)] = wins.get((a, b), 0.0) + prior
        wins[(b, a)] = wins.get((b, a), 0.0) + prior
    total_wins = {n: sum(w for (i, _), w in wins.items() if i == n) for n in names}
    n_ij = {(a, b): wins.get((a, b), 0.0) + wins.get((b, a), 0.0) for a in names for b in names if a != b}
    strength = {n: 1.0 for n in names}
    for _ in range(max_iter):
        new = {}
        for i in names:
            denom = sum(n_ij[(i, j)] / (strength[i] + strength[j]) for j in names if j != i)
            new[i] = total_wins[i] / denom if denom > 0 else strength[i]
        geo = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {k: v / geo for k, v in new.items()}
        delta = max(abs(new[k] - strength[k]) for k in names)
        strength = new
        if delta < tol:
            break
    return {k: math.log(v) for k, v in strength.items()}


def bt_win_probability(log_strengths: Mapping[str, float], a: str, b: str) -> float:
    return 1.0 / (1.0 + math.exp(log_strengths[b] - log_strengths[a]))


def bradley_terry_bootstrap(comparisons_by_cluster: Mapping[str, Sequence[Tuple[str, str]]],
                            a: str, b: str, *, n_boot: int = 2000, seed: int = 101,
                            alpha: float = 0.05, prior: float = 0.5) -> dict:
    """Cluster-bootstrap CI for the log-strength difference ``a - b``."""
    keys = list(comparisons_by_cluster)
    all_comps = [c for k in keys for c in comparisons_by_cluster[k]]
    if len(keys) < 2 or not all_comps:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_clusters": len(keys)}
    items = sorted({x for pair in all_comps for x in pair})
    point = bradley_terry(all_comps, items=items, prior=prior)
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        sample = [c for _ in keys for c in comparisons_by_cluster[rng.choice(keys)]]
        if not sample:
            continue
        est = bradley_terry(sample, items=items, prior=prior)
        draws.append(est.get(a, 0.0) - est.get(b, 0.0))
    return {"estimate": point.get(a, 0.0) - point.get(b, 0.0),
            "ci_low": percentile(draws, 100 * alpha / 2),
            "ci_high": percentile(draws, 100 * (1 - alpha / 2)),
            "p_a_beats_b": bt_win_probability(point, a, b),
            "n_clusters": len(keys), "n_comparisons": len(all_comps),
            "prior_pseudo_wins": prior, "n_boot": len(draws)}
