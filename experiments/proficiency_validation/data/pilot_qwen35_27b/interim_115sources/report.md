# Proficiency validation report

Generated 2026-09-25T17:47:03+00:00. Frozen config: no (pilot / exploratory).

## Generation QC

| condition | n | first-pass | retried | treatment unavailable | infrastructure unavailable | fallback | mean attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 115 | 69 | 46 | 7 | 0 | 0 | 1.50 |
| low | 115 | 57 | 58 | 44 | 0 | 0 | 1.90 |
| neutral | 114 | 69 | 0 | 45 | 0 | 0 | 0.61 |

## Neutral matching (normalized token edit distance to the original)

| condition | n | mean | SD | median |
| --- | ---: | ---: | ---: | ---: |
| low | 71 | 0.326 | 0.158 | 0.333 |
| neutral | 69 | 0.346 | 0.122 | 0.350 |

|low − neutral| gap: mean 0.118, median 0.094, p90 0.257

| condition | n | mean | SD | median |
| --- | ---: | ---: | ---: | ---: |
| high | 108 | 0.417 | 0.177 | 0.437 |
| neutral | 69 | 0.346 | 0.122 | 0.350 |

|high − neutral| gap: mean 0.121, median 0.117, p90 0.242

## Family A: linguistic range

### low − neutral  (primary; n bundles = 2, games = 1)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | ↓ | -0.081 | [–, –] | -17.39 | – | – | no |
| mtld | ↓ | -33.443 | [–, –] | -2.33 | – | – | no |
| repetition_rate | ↑ | 0.046 | [–, –] | 1.71 | – | – | no |
| top5_share | ↑ | 0.074 | [–, –] | 14.49 | – | – | no |
| clause_pattern_entropy | ↓ | -0.055 | [–, –] | -0.52 | – | – | no |
| clause_pattern_distinct_ratio | ↓ | -0.700 | [–, –] | – | – | – | no |
| subordinate_clauses_per_sentence | ↓ | -4.300 | [–, –] | -20.27 | – | – | no |
| mean_dependency_distance | ↓ | -1.416 | [–, –] | -40.03 | – | – | no |
| tree_depth_mean | ↓ | -1.985 | [–, –] | -11.40 | – | – | no |
| np_mean_length | ↓ | -0.096 | [–, –] | -1.77 | – | – | no |
| mean_sentence_length | ↓ | -37.500 | [–, –] | -66.29 | – | – | no |
| word_count |  | 4.500 | [–, –] | 2.12 | – | – | – |
| n_sentences |  | 16.000 | [–, –] | – | – | – | – |
| edit_distance_to_original |  | -0.016 | [–, –] | -0.37 | – | – | – |

### neutral − original  (control; n bundles = 2, games = 1)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | = | -0.018 | [–, –] | -11.62 | – | – | – |
| mtld | = | 8.225 | [–, –] | 0.58 | – | – | – |
| repetition_rate | = | -0.045 | [–, –] | -1.68 | – | – | – |
| top5_share | = | -0.037 | [–, –] | -7.21 | – | – | – |
| clause_pattern_entropy | = | 0.000 | [–, –] | – | – | – | – |
| clause_pattern_distinct_ratio | = | 0.000 | [–, –] | – | – | – | – |
| subordinate_clauses_per_sentence | = | 0.250 | [–, –] | 0.71 | – | – | – |
| mean_dependency_distance | = | 0.117 | [–, –] | 2.08 | – | – | – |
| tree_depth_mean | = | -0.106 | [–, –] | -0.59 | – | – | – |
| np_mean_length | = | 0.075 | [–, –] | 3.63 | – | – | – |
| mean_sentence_length | = | -0.625 | [–, –] | -3.54 | – | – | – |
| word_count |  | -3.000 | [–, –] | -1.06 | – | – | – |
| n_sentences |  | 0.000 | [–, –] | – | – | – | – |
| edit_distance_to_original |  | 0.354 | [–, –] | 6.58 | – | – | – |

### low − original  (secondary; n bundles = 2, games = 1)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | ↓ | -0.099 | [–, –] | -31.95 | – | – | no |
| mtld | ↓ | -25.217 | [–, –] | -96.76 | – | – | no |
| repetition_rate | ↑ | 0.001 | [–, –] | – | – | – | no |
| top5_share | ↑ | 0.037 | [–, –] | – | – | – | no |
| clause_pattern_entropy | ↓ | -0.055 | [–, –] | -0.52 | – | – | no |
| clause_pattern_distinct_ratio | ↓ | -0.700 | [–, –] | – | – | – | no |
| subordinate_clauses_per_sentence | ↓ | -4.050 | [–, –] | -28.64 | – | – | no |
| mean_dependency_distance | ↓ | -1.299 | [–, –] | -61.84 | – | – | no |
| tree_depth_mean | ↓ | -2.090 | [–, –] | -402.14 | – | – | no |
| np_mean_length | ↓ | -0.021 | [–, –] | -0.62 | – | – | no |
| mean_sentence_length | ↓ | -38.125 | [–, –] | -98.03 | – | – | no |
| word_count |  | 1.500 | [–, –] | 2.12 | – | – | – |
| n_sentences |  | 16.000 | [–, –] | – | – | – | – |
| edit_distance_to_original |  | 0.338 | [–, –] | 30.30 | – | – | – |

### high − neutral  (primary when included; n bundles = 2, games = 1)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | ↑ | 0.026 | [–, –] | 4.14 | – | – | no |
| mtld | ↑ | 20.168 | [–, –] | 1.52 | – | – | no |
| repetition_rate | ↓ | -0.010 | [–, –] | -0.35 | – | – | no |
| top5_share | ↓ | -0.007 | [–, –] | -3.17 | – | – | no |
| clause_pattern_entropy | ↑ | 0.000 | [–, –] | – | – | – | no |
| clause_pattern_distinct_ratio | ↑ | 0.000 | [–, –] | – | – | – | no |
| subordinate_clauses_per_sentence | ↑ | -0.375 | [–, –] | -2.12 | – | – | no |
| mean_dependency_distance | ↑ | -0.178 | [–, –] | -3.94 | – | – | no |
| tree_depth_mean | ↑ | 0.501 | [–, –] | 27.21 | – | – | no |
| np_mean_length | ↑ | 0.067 | [–, –] | 2.07 | – | – | no |
| mean_sentence_length | ↑ | 0.625 | [–, –] | 3.54 | – | – | no |
| word_count |  | 3.500 | [–, –] | 4.95 | – | – | – |
| n_sentences |  | 0.000 | [–, –] | – | – | – | – |
| edit_distance_to_original |  | 0.125 | [–, –] | 6.33 | – | – | – |
