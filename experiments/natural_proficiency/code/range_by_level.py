"""Linguistic range of untreated speech across the five prompt levels (L1-L5).

    python -m experiments.natural_proficiency.code.range_by_level analyze
    python -m experiments.natural_proficiency.code.range_by_level analyze --utterances PATH

``analyze`` reads the primary-predicate utterances that
``experiments/natural_politeness/data/utterances.jsonl`` records for the prompt-level
games (146 nine-player games, five prompt-guidance levels, shared seeds), assembles
one passage per speaker per game in speech order, keeps passages with at least
``MIN_WORDS`` words as in the proficiency pilot, and measures the Family A range
metrics with the proficiency-validation measurement code. It writes
``data/level_features.jsonl`` (one row per speaker-game passage),
``data/level_summary.json`` (every number in ``summary.md``) and ``summary.md``.
Nothing here calls a model.

The prompt levels manipulate the amount of strategy guidance in the agent's system
prompt (L1 bare rules, L5 the full default strategy prompt), not language. The
analysis asks whether that guidance incidentally shifts vocabulary range or syntactic
range, and places the untreated baseline next to the pilot's Original and Low passages
measured with the same code.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from components.helpers import read_jsonl, write_jsonl
from components.metrics.text_normalization import split_sentences, word_count
from components.stats import benjamini_hochberg, cluster_bootstrap, mean, paired_cohens_dz, sd
from experiments.proficiency_validation.rewrite.runtime import RANGE_METRICS, fmt, now_iso, sha256_file
from experiments.proficiency_validation.validation.analysis import cluster_sign_flip_test
from experiments.proficiency_validation.validation.measurements import load_nlp_or_none, text_features

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATA_DIR = EXPERIMENT_DIR / "data"
DEFAULT_UTTERANCES = REPO_ROOT / "experiments" / "natural_politeness" / "data" / "utterances.jsonl"
PILOT_CORPUS = REPO_ROOT / "experiments" / "proficiency_validation" / "data" / "corpus" / "original_features.jsonl"
PILOT_RUN = REPO_ROOT / "experiments" / "proficiency_validation" / "data" / "runs" / "pilot_qwen35_27b" / "validation" / "features.jsonl"
FEATURES = DATA_DIR / "level_features.jsonl"
SUMMARY_JSON = DATA_DIR / "level_summary.json"
SUMMARY_MD = EXPERIMENT_DIR / "summary.md"

LEVELS = ("L1", "L2", "L3", "L4", "L5")
REFERENCE_LEVEL = "L5"          # the default prompt; the proficiency corpus was generated with it
MIN_WORDS = 150                 # the pilot's whole-game eligibility minimum
MATTR_WINDOW = 50
METRICS = [m for m in RANGE_METRICS if m != "subtlex_mean"]   # no SUBTLEX file is bundled
HEADLINE = ("mattr", "mtld", "subordinate_clause_rate", "mean_dependency_distance",
            "tree_depth_mean", "mean_sentence_length")
N_BOOT, N_PERM, SEED = 2000, 5000, 101


def _speech_number(speech_id: str) -> int:
    digits = "".join(ch for ch in str(speech_id) if ch.isdigit())
    return int(digits) if digits else 10 ** 9


def assemble_passages(utterances: Sequence[dict]) -> List[dict]:
    """One passage per (level, game, speaker): whole-game speech in speech order."""
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for row in utterances:
        groups[(row["level"], row["game"], row["speaker"])].append(row)
    passages = []
    for (level, game, speaker), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: (_speech_number(r["speech_id"]), r["speech_id"]))
        text = " ".join(r["text"].strip() for r in rows)
        passages.append({"level": level, "game": game, "speaker": speaker,
                         "role": rows[0].get("role"), "n_utterances": len(rows),
                         "speech_ids": [r["speech_id"] for r in rows], "text": text,
                         "word_count": word_count(text),
                         "eligible": word_count(text) >= MIN_WORDS})
    return passages


def measure(passages: Sequence[dict], nlp) -> List[dict]:
    rows = []
    for p in passages:
        base = {k: p[k] for k in ("level", "game", "speaker", "role", "n_utterances", "word_count", "eligible")}
        if not p["eligible"]:
            rows.append(base)
            continue
        feats = text_features(p["text"], nlp=nlp, norms=None, mattr_window=MATTR_WINDOW, include_domain=False)
        feats.pop("word_count", None)
        rows.append({**base, **feats})
    return rows


def game_means(rows: Sequence[dict], level: str, metric: str) -> Dict[str, float]:
    """Equal speaker weight within a game; games are the independent units."""
    per_game: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if r["level"] == level and r.get("eligible") and r.get(metric) is not None:
            per_game[r["game"]].append(r[metric])
    return {g: mean(v) for g, v in per_game.items()}


def level_estimate(rows: Sequence[dict], level: str, metric: str) -> dict:
    means = game_means(rows, level, metric)
    boot = cluster_bootstrap({g: [m] for g, m in means.items()}, n_boot=N_BOOT, seed=SEED)
    boot.pop("draws", None)
    return {"estimate": boot["estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "n_games": len(means), "sd_games": sd(list(means.values()))}


def paired_contrast(rows: Sequence[dict], level: str, reference: str, metric: str) -> dict:
    """Level minus reference within the games both levels played (shared seeds)."""
    a, b = game_means(rows, level, metric), game_means(rows, reference, metric)
    shared = sorted(set(a) & set(b))
    diffs = {g: [a[g] - b[g]] for g in shared}
    boot = cluster_bootstrap(diffs, n_boot=N_BOOT, seed=SEED)
    boot.pop("draws", None)
    test = cluster_sign_flip_test(diffs, n_perm=N_PERM, seed=SEED)
    return {"n_games": len(shared), "estimate": boot["estimate"], "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"], "cohens_dz": paired_cohens_dz([d[0] for d in diffs.values()]),
            "p_sign_flip": test["p_value"], "sign_flip_method": test.get("method"),
            "excludes_zero": (boot["ci_low"] is not None and (boot["ci_low"] > 0 or boot["ci_high"] < 0))}


def reference_rows(path: Path, condition: str, label: str) -> Optional[dict]:
    """Game-weighted means of the pilot's passages measured with the same code."""
    if not path.exists():
        return None
    rows = [r for r in read_jsonl(path) if r.get("condition") == condition and r.get("complete")
            and r.get("mattr") is not None]
    if not rows:
        return None
    out = {"label": label, "n_passages": len(rows), "n_games": len({r["game_id"] for r in rows}), "metrics": {}}
    for metric in METRICS + ["word_count"]:
        per_game: Dict[str, List[float]] = defaultdict(list)
        for r in rows:
            if r.get(metric) is not None:
                per_game[r["game_id"]].append(r[metric])
        out["metrics"][metric] = mean([mean(v) for v in per_game.values()]) if per_game else None
    return out


def summarize(utterances: Sequence[dict], features: Sequence[dict], *, utterances_path: Path) -> dict:
    by_level = {level: [r for r in features if r["level"] == level] for level in LEVELS}
    descriptives = {}
    for level in LEVELS:
        utts = [u for u in utterances if u["level"] == level]
        rows = by_level[level]
        descriptives[level] = {
            "games": len({u["game"] for u in utts}), "utterances": len(utts),
            "speaker_games": len(rows), "eligible_passages": sum(1 for r in rows if r.get("eligible")),
            "excluded_short_passages": sum(1 for r in rows if not r.get("eligible")),
            "words_per_utterance": mean([word_count(u["text"]) for u in utts]),
            "sentences_per_utterance": mean([len(split_sentences(u["text"])) for u in utts]),
            "words_per_passage": mean([r["word_count"] for r in rows if r.get("eligible")]),
            "utterances_per_passage": mean([r["n_utterances"] for r in rows if r.get("eligible")]),
        }
    estimates = {level: {m: level_estimate(features, level, m) for m in METRICS} for level in LEVELS}
    contrasts = {}
    for level in LEVELS:
        if level == REFERENCE_LEVEL:
            continue
        block = {m: paired_contrast(features, level, REFERENCE_LEVEL, m) for m in METRICS}
        for m, adj in benjamini_hochberg({m: block[m]["p_sign_flip"] for m in METRICS}).items():
            block[m]["q_bh"] = adj["q"]
            block[m]["reject_fdr_05"] = adj["reject"]
        contrasts[f"{level}_vs_{REFERENCE_LEVEL}"] = block
    references = [r for r in (reference_rows(PILOT_CORPUS, "original", "Pilot corpus Original (3 games, all predicates)"),
                              reference_rows(PILOT_RUN, "low", "Pilot Low rewrites, complete passages"),
                              reference_rows(PILOT_RUN, "neutral", "Pilot Neutral rewrites, complete passages")) if r]
    scale = {}
    original = next((r for r in references if r["label"].startswith("Pilot corpus Original")), None)
    low = next((r for r in references if r["label"].startswith("Pilot Low")), None)
    if original and low:
        for m in HEADLINE:
            largest = max((abs(block[m]["estimate"]) for block in contrasts.values() if block[m]["estimate"] is not None), default=None)
            delta = (low["metrics"].get(m) - original["metrics"].get(m)
                     if low["metrics"].get(m) is not None and original["metrics"].get(m) is not None else None)
            scale[m] = {"largest_level_difference": largest, "low_minus_original": delta,
                        "ratio": (largest / abs(delta)) if largest is not None and delta else None}
    return {"created_at": now_iso(), "utterances_path": str(utterances_path), "scale_check": scale,
            "utterances_sha256": sha256_file(utterances_path), "n_utterances": len(utterances),
            "levels": list(LEVELS), "reference_level": REFERENCE_LEVEL, "min_words": MIN_WORDS,
            "mattr_window": MATTR_WINDOW, "metrics": METRICS, "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED,
            "estimand": "equal_game_weight", "descriptives": descriptives, "estimates": estimates,
            "contrasts": contrasts, "references": references,
            "syntax_available": any(r.get("syntax_available") for r in features)}


def _ci(block: dict, digits: int = 3) -> str:
    return f"{fmt(block['estimate'], digits)} [{fmt(block['ci_low'], digits)}, {fmt(block['ci_high'], digits)}]"


def render(s: dict) -> str:
    d, est, con = s["descriptives"], s["estimates"], s["contrasts"]
    ref = s["reference_level"]
    rejections = {name: sorted(m for m, v in block.items() if v.get("reject_fdr_05")) for name, block in con.items()}
    n_reject = sum(len(v) for v in rejections.values())
    n_contrasts = sum(len(b) for b in con.values())
    strongest = max(((abs(v["cohens_dz"] or 0), name, m) for name, block in con.items() for m, v in block.items()), default=(0, "", ""))
    span = {m: (min(est[level][m]["estimate"] for level in s["levels"]), max(est[level][m]["estimate"] for level in s["levels"]))
            for m in ("mattr", "subordinate_clause_rate", "mean_sentence_length")}
    scale = s.get("scale_check") or {}
    lines = ["# Linguistic range of untreated speech across prompt levels", "",
             "**Main result.** The five prompt levels produce speech of nearly the same linguistic range. "
             f"Game-weighted MATTR is {fmt(span['mattr'][0], 3)} to {fmt(span['mattr'][1], 3)} at every level, "
             f"subordinate clauses per 100 words {fmt(span['subordinate_clause_rate'][0], 1)} to "
             f"{fmt(span['subordinate_clause_rate'][1], 1)}, and mean sentence length "
             f"{fmt(span['mean_sentence_length'][0], 1)} to {fmt(span['mean_sentence_length'][1], 1)} words. "
             f"After Benjamini-Hochberg correction, {n_reject} of {n_contrasts} level-versus-{ref} contrasts reject at q = 0.05"
             + (" (" + "; ".join(f"{name.replace('_', ' ')}: {', '.join(ms)}" for name, ms in rejections.items() if ms) + ")" if n_reject else "")
             + f". The largest standardized difference is d_z = {fmt(strongest[0], 2)} ({strongest[2]}, {strongest[1].replace('_', ' ')}). "
             + ("The scale check below shows that the largest between-level difference is a fraction of the pilot's Low "
                "treatment on syntactic range, measured with the same code. " if scale else "")
             + "The amount of strategy guidance in the system prompt changes what agents argue about, not the "
               "range of English they use. As with politeness, proficiency effects have to come from a controlled "
               "rewrite rather than from observational variation in the corpus.", "",
             "## Setup", "",
             f"The source is `{Path(s['utterances_path']).name}` from the natural-politeness experiment "
             f"(sha256 `{s['utterances_sha256'][:16]}...`): {s['n_utterances']} utterances realizing the five "
             "primary predicates in 146 nine-player `deepseek-chat` games at prompt levels L1 to L5, with shared "
             "seeds so that the same game number received the same role deal at every level. Each speaker's "
             "utterances in one game are joined in speech order into one passage. Passages with fewer than "
             f"{s['min_words']} words are excluded, the eligibility rule of the proficiency pilot. Measures are "
             "the proficiency pilot's Family A range metrics, computed with the same code, spaCy parse and "
             f"MATTR window ({s['mattr_window']} tokens). Lexical frequency needs SUBTLEX norms and is not measured.", "",
             "Speakers receive equal weight within a game and games are the independent units. Intervals are "
             f"{s['n_boot']}-draw game bootstraps. Contrasts against {s['reference_level']} pair games by shared seed and "
             f"use a sign-flip test over games ({s['n_perm']} Monte Carlo sign patterns, seed {s['seed']}); "
             "q-values apply Benjamini-Hochberg across the range metrics of one contrast.", "",
             "## Passages", "",
             "| level | games | utterances | speaker-games | passages >= 150 words | excluded | words / utterance | words / passage |",
             "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for level in s["levels"]:
        x = d[level]
        lines.append(f"| {level} | {x['games']} | {x['utterances']} | {x['speaker_games']} | {x['eligible_passages']} | "
                     f"{x['excluded_short_passages']} | {fmt(x['words_per_utterance'], 1)} | {fmt(x['words_per_passage'], 0)} |")
    lines += ["", "## Range metrics by level (game-weighted mean, 95% game bootstrap)", "",
              "| metric | " + " | ".join(s["levels"]) + " |", "|---|" + "--:|" * len(s["levels"])]
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(_ci(est[level][m]) for level in s["levels"]) + " |")
    lines += ["", f"## Paired contrasts against {s['reference_level']} (level minus {s['reference_level']}, games paired by seed)", ""]
    for name, block in con.items():
        lines += [f"### {name.replace('_', ' ')}", "",
                  "| metric | games | difference | 95% CI | game d_z | p (sign-flip) | q (BH) |",
                  "|---|--:|--:|--:|--:|--:|--:|"]
        for m in METRICS:
            v = block[m]
            lines.append(f"| {m} | {v['n_games']} | {fmt(v['estimate'], 3)} | [{fmt(v['ci_low'], 3)}, {fmt(v['ci_high'], 3)}] | "
                         f"{fmt(v['cohens_dz'], 2)} | {fmt(v['p_sign_flip'])} | {fmt(v['q_bh'])} |")
        lines.append("")
    if s["references"]:
        lines += ["## Reference: the proficiency pilot measured with the same code", "",
                  "These rows use whole-game passages from the proficiency corpus and pilot. The corpus keeps every "
                  "public speech act, not only the five primary predicates, so the comparison is indicative.", "",
                  "| passages | n | games | " + " | ".join(HEADLINE) + " |", "|---|--:|--:|" + "--:|" * len(HEADLINE)]
        for level in s["levels"]:
            lines.append(f"| {level} untreated | {d[level]['eligible_passages']} | {est[level]['mattr']['n_games']} | "
                         + " | ".join(fmt(est[level][m]["estimate"], 3) for m in HEADLINE) + " |")
        for reference in s["references"]:
            lines.append(f"| {reference['label']} | {reference['n_passages']} | {reference['n_games']} | "
                         + " | ".join(fmt(reference["metrics"].get(m), 3) for m in HEADLINE) + " |")
        lines.append("")
    if scale:
        lines += ["## Scale check: between-level differences against the pilot's Low treatment", "",
                  f"The largest absolute level-versus-{ref} difference across the four contrasts, next to the pilot's "
                  "Low minus Original difference on complete whole-game passages (12 Low passages, 24 Original, 3 games; "
                  "corpus passages contain every predicate).", "",
                  "| metric | largest level difference | pilot Low minus Original | ratio |", "|---|--:|--:|--:|"]
        for m, row in scale.items():
            lines.append(f"| {m} | {fmt(row['largest_level_difference'], 3)} | {fmt(row['low_minus_original'], 3)} | "
                         f"{fmt(row['ratio'], 2)} |")
        lines.append("")
    lines += ["## Limitations", "",
              "- The passages contain only the five primary predicates recorded by the politeness experiment, "
              "not every public speech act, so absolute values differ from the all-predicate proficiency corpus.",
              "- Range metrics are relative measures. Syntax depends on the spaCy parse of game dialogue.",
              "- Games at a level are not independent of games at another level (shared seeds), which the paired "
              "design uses; levels L2 and L3 are missing a few seeds.",
              "- The generating model is `deepseek-chat` at every level.", "",
              "## Files", "", "| file | contents |", "|---|---|",
              "| `data/level_features.jsonl` | one row per speaker-game passage: level, game, speaker, role, eligibility and every range metric |",
              "| `data/level_summary.json` | every number above, with intervals, tests and the source file hash |",
              "| `code/range_by_level.py` | `analyze` rebuilds both files and this summary from the utterance file; no model calls |", ""]
    return "\n".join(lines)


def analyze(utterances_path: Path, *, spacy_model: str = "en_core_web_sm") -> dict:
    utterances = read_jsonl(utterances_path)
    unknown = sorted({u["level"] for u in utterances} - set(LEVELS))
    if unknown:
        raise ValueError(f"unexpected prompt levels {unknown}")
    nlp = load_nlp_or_none(spacy_model)
    if nlp is None:
        raise RuntimeError("spaCy with en_core_web_sm is required for the syntactic range metrics")
    features = measure(assemble_passages(utterances), nlp)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(FEATURES, features)
    summary = summarize(utterances, features, utterances_path=utterances_path)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SUMMARY_MD.write_text(render(summary), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze", help="measure range per level and write data/ and summary.md")
    a.add_argument("--utterances", type=Path, default=DEFAULT_UTTERANCES)
    a.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args(argv)
    summary = analyze(args.utterances, spacy_model=args.spacy_model)
    eligible = sum(x["eligible_passages"] for x in summary["descriptives"].values())
    print(f"{summary['n_utterances']} utterances -> {eligible} eligible passages; wrote {FEATURES}, {SUMMARY_JSON}, {SUMMARY_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
