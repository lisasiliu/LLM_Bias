# Linguistic range of untreated speech across prompt levels

**Main result.** The five prompt levels produce speech of nearly the same linguistic range. Game-weighted MATTR is 0.810 to 0.816 at every level, subordinate clauses per 100 words 9.7 to 10.1, and mean sentence length 42.0 to 45.4 words. After Benjamini-Hochberg correction, 4 of 48 level-versus-L5 contrasts reject at q = 0.05 (L2 vs L5: tree_depth_mean; L4 vs L5: mean_sentence_length, mtld, subordinate_clauses_per_sentence). The largest standardized difference is d_z = 0.92 (mean_sentence_length, L4 vs L5). The scale check below shows that the largest between-level difference is a fraction of the pilot's Low treatment on syntactic range, measured with the same code. The amount of strategy guidance in the system prompt changes what agents argue about, not the range of English they use. As with politeness, proficiency effects have to come from a controlled rewrite rather than from observational variation in the corpus.

## Setup

The source is `utterances.jsonl` from the natural-politeness experiment (sha256 `bf40e735019874bb...`): 8678 utterances realizing the five primary predicates in 146 nine-player `deepseek-chat` games at prompt levels L1 to L5, with shared seeds so that the same game number received the same role deal at every level. Each speaker's utterances in one game are joined in speech order into one passage. Passages with fewer than 150 words are excluded, the eligibility rule of the proficiency pilot. Measures are the proficiency pilot's Family A range metrics, computed with the same code, spaCy parse and MATTR window (50 tokens). Lexical frequency needs SUBTLEX norms and is not measured.

Speakers receive equal weight within a game and games are the independent units. Intervals are 2000-draw game bootstraps. Contrasts against L5 pair games by shared seed and use a sign-flip test over games (5000 Monte Carlo sign patterns, seed 101); q-values apply Benjamini-Hochberg across the range metrics of one contrast.

## Passages

| level | games | utterances | speaker-games | passages >= 150 words | excluded | words / utterance | words / passage |
|---|--:|--:|--:|--:|--:|--:|--:|
| L1 | 30 | 1836 | 269 | 233 | 36 | 44.0 | 334 |
| L2 | 29 | 1706 | 256 | 214 | 42 | 44.8 | 341 |
| L3 | 27 | 1603 | 243 | 203 | 40 | 44.7 | 337 |
| L4 | 30 | 1700 | 267 | 209 | 58 | 41.1 | 312 |
| L5 | 30 | 1833 | 267 | 228 | 39 | 44.8 | 347 |

## Range metrics by level (game-weighted mean, 95% game bootstrap)

| metric | L1 | L2 | L3 | L4 | L5 |
|---|--:|--:|--:|--:|--:|
| mattr | 0.815 [0.811, 0.819] | 0.816 [0.812, 0.821] | 0.811 [0.807, 0.815] | 0.810 [0.806, 0.814] | 0.816 [0.811, 0.821] |
| mtld | 86.525 [84.127, 89.154] | 86.999 [84.100, 90.021] | 85.683 [83.108, 88.405] | 83.569 [81.449, 85.721] | 89.017 [86.082, 91.821] |
| repetition_rate | 0.198 [0.184, 0.212] | 0.201 [0.187, 0.215] | 0.188 [0.173, 0.202] | 0.184 [0.167, 0.201] | 0.205 [0.191, 0.219] |
| top5_share | 0.158 [0.153, 0.164] | 0.157 [0.151, 0.165] | 0.158 [0.151, 0.165] | 0.167 [0.160, 0.174] | 0.156 [0.150, 0.163] |
| clause_pattern_entropy | 2.839 [2.740, 2.930] | 2.839 [2.708, 2.960] | 2.816 [2.678, 2.949] | 2.780 [2.643, 2.923] | 2.868 [2.759, 2.971] |
| clause_pattern_distinct_ratio | 0.996 [0.993, 0.998] | 0.997 [0.993, 1.000] | 0.994 [0.990, 0.998] | 0.995 [0.990, 0.998] | 0.996 [0.994, 0.999] |
| subordinate_clauses_per_sentence | 4.512 [4.365, 4.662] | 4.543 [4.426, 4.658] | 4.357 [4.159, 4.556] | 4.081 [3.930, 4.225] | 4.535 [4.432, 4.644] |
| subordinate_clause_rate | 10.119 [9.851, 10.381] | 9.977 [9.806, 10.160] | 9.727 [9.440, 10.014] | 9.687 [9.459, 9.906] | 10.042 [9.843, 10.259] |
| mean_dependency_distance | 3.158 [3.116, 3.201] | 3.159 [3.124, 3.198] | 3.229 [3.177, 3.278] | 3.109 [3.060, 3.170] | 3.195 [3.162, 3.229] |
| tree_depth_mean | 3.529 [3.483, 3.578] | 3.687 [3.644, 3.736] | 3.541 [3.478, 3.601] | 3.586 [3.523, 3.651] | 3.578 [3.532, 3.624] |
| np_mean_length | 1.869 [1.848, 1.890] | 1.886 [1.864, 1.909] | 1.890 [1.860, 1.920] | 1.914 [1.889, 1.939] | 1.886 [1.859, 1.916] |
| mean_sentence_length | 44.558 [43.648, 45.446] | 45.406 [44.615, 46.245] | 44.715 [43.533, 45.881] | 41.957 [40.974, 43.004] | 45.117 [44.525, 45.753] |

## Paired contrasts against L5 (level minus L5, games paired by seed)

### L1 vs L5

| metric | games | difference | 95% CI | game d_z | p (sign-flip) | q (BH) |
|---|--:|--:|--:|--:|--:|--:|
| mattr | 30 | -0.001 | [-0.008, 0.005] | -0.08 | 0.668 | 0.844 |
| mtld | 30 | -2.492 | [-6.100, 1.373] | -0.23 | 0.216 | 0.840 |
| repetition_rate | 30 | -0.007 | [-0.024, 0.011] | -0.15 | 0.420 | 0.840 |
| top5_share | 30 | 0.003 | [-0.007, 0.011] | 0.10 | 0.592 | 0.844 |
| clause_pattern_entropy | 30 | -0.028 | [-0.179, 0.128] | -0.06 | 0.729 | 0.844 |
| clause_pattern_distinct_ratio | 30 | -0.000 | [-0.004, 0.003] | -0.05 | 0.819 | 0.844 |
| subordinate_clauses_per_sentence | 30 | -0.023 | [-0.240, 0.196] | -0.04 | 0.844 | 0.844 |
| subordinate_clause_rate | 30 | 0.077 | [-0.301, 0.476] | 0.07 | 0.712 | 0.844 |
| mean_dependency_distance | 30 | -0.038 | [-0.075, -0.000] | -0.36 | 0.060 | 0.725 |
| tree_depth_mean | 30 | -0.050 | [-0.120, 0.021] | -0.25 | 0.179 | 0.840 |
| np_mean_length | 30 | -0.017 | [-0.053, 0.015] | -0.17 | 0.373 | 0.840 |
| mean_sentence_length | 30 | -0.559 | [-1.712, 0.596] | -0.17 | 0.370 | 0.840 |

### L2 vs L5

| metric | games | difference | 95% CI | game d_z | p (sign-flip) | q (BH) |
|---|--:|--:|--:|--:|--:|--:|
| mattr | 28 | -0.001 | [-0.007, 0.006] | -0.04 | 0.845 | 0.961 |
| mtld | 28 | -2.492 | [-6.537, 1.946] | -0.22 | 0.258 | 0.961 |
| repetition_rate | 28 | -0.004 | [-0.026, 0.017] | -0.08 | 0.683 | 0.961 |
| top5_share | 28 | 0.001 | [-0.010, 0.012] | 0.03 | 0.881 | 0.961 |
| clause_pattern_entropy | 28 | -0.015 | [-0.205, 0.168] | -0.03 | 0.867 | 0.961 |
| clause_pattern_distinct_ratio | 28 | 0.001 | [-0.004, 0.004] | 0.05 | 0.795 | 0.961 |
| subordinate_clauses_per_sentence | 28 | 0.015 | [-0.135, 0.172] | 0.04 | 0.843 | 0.961 |
| subordinate_clause_rate | 28 | -0.058 | [-0.308, 0.206] | -0.08 | 0.675 | 0.961 |
| mean_dependency_distance | 28 | -0.036 | [-0.079, 0.007] | -0.31 | 0.111 | 0.668 |
| tree_depth_mean | 28 | 0.114 | [0.050, 0.182] | 0.62 | 0.000 | 0.005 |
| np_mean_length | 28 | -0.000 | [-0.041, 0.041] | -0.00 | 0.988 | 0.988 |
| mean_sentence_length | 28 | 0.327 | [-0.635, 1.279] | 0.12 | 0.526 | 0.961 |

### L3 vs L5

| metric | games | difference | 95% CI | game d_z | p (sign-flip) | q (BH) |
|---|--:|--:|--:|--:|--:|--:|
| mattr | 27 | -0.005 | [-0.011, 0.002] | -0.28 | 0.161 | 0.387 |
| mtld | 27 | -3.209 | [-7.341, 1.143] | -0.28 | 0.160 | 0.387 |
| repetition_rate | 27 | -0.021 | [-0.039, -0.003] | -0.42 | 0.033 | 0.387 |
| top5_share | 27 | 0.002 | [-0.007, 0.010] | 0.07 | 0.724 | 0.790 |
| clause_pattern_entropy | 27 | -0.057 | [-0.226, 0.105] | -0.13 | 0.520 | 0.624 |
| clause_pattern_distinct_ratio | 27 | -0.003 | [-0.007, 0.002] | -0.21 | 0.293 | 0.503 |
| subordinate_clauses_per_sentence | 27 | -0.193 | [-0.437, 0.042] | -0.29 | 0.137 | 0.387 |
| subordinate_clause_rate | 27 | -0.330 | [-0.736, 0.061] | -0.31 | 0.125 | 0.387 |
| mean_dependency_distance | 27 | 0.034 | [-0.018, 0.090] | 0.23 | 0.235 | 0.471 |
| tree_depth_mean | 27 | -0.030 | [-0.107, 0.043] | -0.14 | 0.447 | 0.624 |
| np_mean_length | 27 | 0.004 | [-0.038, 0.045] | 0.03 | 0.862 | 0.862 |
| mean_sentence_length | 27 | -0.486 | [-1.771, 0.788] | -0.14 | 0.476 | 0.624 |

### L4 vs L5

| metric | games | difference | 95% CI | game d_z | p (sign-flip) | q (BH) |
|---|--:|--:|--:|--:|--:|--:|
| mattr | 30 | -0.006 | [-0.012, 0.000] | -0.34 | 0.072 | 0.115 |
| mtld | 30 | -5.448 | [-8.998, -1.640] | -0.51 | 0.009 | 0.035 |
| repetition_rate | 30 | -0.022 | [-0.043, 0.001] | -0.34 | 0.077 | 0.115 |
| top5_share | 30 | 0.011 | [0.001, 0.020] | 0.41 | 0.029 | 0.068 |
| clause_pattern_entropy | 30 | -0.088 | [-0.253, 0.070] | -0.19 | 0.314 | 0.377 |
| clause_pattern_distinct_ratio | 30 | -0.001 | [-0.006, 0.003] | -0.11 | 0.620 | 0.676 |
| subordinate_clauses_per_sentence | 30 | -0.454 | [-0.635, -0.281] | -0.88 | 0.000 | 0.002 |
| subordinate_clause_rate | 30 | -0.355 | [-0.659, -0.042] | -0.42 | 0.034 | 0.068 |
| mean_dependency_distance | 30 | -0.086 | [-0.153, -0.014] | -0.42 | 0.027 | 0.068 |
| tree_depth_mean | 30 | 0.008 | [-0.077, 0.091] | 0.03 | 0.863 | 0.863 |
| np_mean_length | 30 | 0.028 | [-0.008, 0.063] | 0.27 | 0.157 | 0.209 |
| mean_sentence_length | 30 | -3.159 | [-4.343, -2.012] | -0.92 | 0.001 | 0.005 |

## Reference: the proficiency pilot measured with the same code

These rows use whole-game passages from the proficiency corpus and pilot. The corpus keeps every public speech act, not only the five primary predicates, so the comparison is indicative.

| passages | n | games | mattr | mtld | subordinate_clause_rate | mean_dependency_distance | tree_depth_mean | mean_sentence_length |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| L1 untreated | 233 | 30 | 0.815 | 86.525 | 10.119 | 3.158 | 3.529 | 44.558 |
| L2 untreated | 214 | 28 | 0.816 | 86.999 | 9.977 | 3.159 | 3.687 | 45.406 |
| L3 untreated | 203 | 27 | 0.811 | 85.683 | 9.727 | 3.229 | 3.541 | 44.715 |
| L4 untreated | 209 | 30 | 0.810 | 83.569 | 9.687 | 3.109 | 3.586 | 41.957 |
| L5 untreated | 228 | 30 | 0.816 | 89.017 | 10.042 | 3.195 | 3.578 | 45.117 |
| Pilot corpus Original (3 games, all predicates) | 24 | 3 | 0.826 | 81.217 | 9.594 | 3.268 | 3.605 | 44.993 |
| Pilot Low rewrites, complete passages | 12 | 3 | 0.807 | 76.800 | 8.902 | 2.570 | 2.336 | 17.667 |

## Scale check: between-level differences against the pilot's Low treatment

The largest absolute level-versus-L5 difference across the four contrasts, next to the pilot's Low minus Original difference on complete whole-game passages (12 Low passages, 24 Original, 3 games; corpus passages contain every predicate).

| metric | largest level difference | pilot Low minus Original | ratio |
|---|--:|--:|--:|
| mattr | 0.006 | -0.019 | 0.32 |
| mtld | 5.448 | -4.418 | 1.23 |
| subordinate_clause_rate | 0.355 | -0.692 | 0.51 |
| mean_dependency_distance | 0.086 | -0.698 | 0.12 |
| tree_depth_mean | 0.114 | -1.269 | 0.09 |
| mean_sentence_length | 3.159 | -27.326 | 0.12 |

## Limitations

- The passages contain only the five primary predicates recorded by the politeness experiment, not every public speech act, so absolute values differ from the all-predicate proficiency corpus.
- Range metrics are relative measures. They do not map a passage onto a proficiency scale; a CEFR-referenced placement would need the blind A/B/C rater, which requires a model server.
- Lexical frequency (SUBTLEX) is unmeasured. Syntax depends on the spaCy parse of game dialogue.
- Games at a level are not independent of games at another level (shared seeds), which the paired design uses; levels L2 and L3 are missing a few seeds.
- The generating model is `deepseek-chat` at every level. Nothing here speaks to other agent models.

## Files

| file | contents |
|---|---|
| `data/level_features.jsonl` | one row per speaker-game passage: level, game, speaker, role, eligibility and every range metric |
| `data/level_summary.json` | every number above, with intervals, tests and the source file hash |
| `code/range_by_level.py` | `analyze` rebuilds both files and this summary from the utterance file; no model calls |
