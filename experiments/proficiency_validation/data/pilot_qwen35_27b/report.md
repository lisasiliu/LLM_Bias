# Proficiency validation report

Generated 2026-09-25T18:34:10+00:00. Frozen config: no (pilot / exploratory).

## Generation QC

| condition | n | first-pass | retried | treatment unavailable | infrastructure unavailable | fallback | mean attempts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high | 280 | 145 | 135 | 38 | 42 | 0 | 1.79 |
| low | 280 | 111 | 169 | 120 | 19 | 0 | 2.11 |
| neutral | 280 | 135 | 1 | 125 | 19 | 0 | 0.49 |

## Neutral matching (normalized token edit distance to the original)

| condition | n | mean | SD | median |
| --- | ---: | ---: | ---: | ---: |
| low | 141 | 0.310 | 0.149 | 0.318 |
| neutral | 136 | 0.354 | 0.123 | 0.354 |

|low − neutral| gap: mean 0.125, median 0.108, p90 0.258

| condition | n | mean | SD | median |
| --- | ---: | ---: | ---: | ---: |
| high | 200 | 0.439 | 0.168 | 0.457 |
| neutral | 136 | 0.354 | 0.123 | 0.354 |

|high − neutral| gap: mean 0.124, median 0.121, p90 0.239

## Family A: linguistic range

### low − neutral  (primary; n bundles = 3, games = 2)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | ↓ | -0.050 | [-0.081, 0.012] | -0.93 | 1.000 | 1.000 | no |
| mtld | ↓ | -22.588 | [-33.443, -0.878] | -1.06 | 0.505 | 0.555 | yes |
| repetition_rate | ↑ | 0.076 | [0.046, 0.136] | 1.38 | 0.505 | 0.555 | yes |
| top5_share | ↑ | 0.065 | [0.046, 0.074] | 3.82 | 0.505 | 0.555 | yes |
| clause_pattern_entropy | ↓ | -0.063 | [-0.081, -0.055] | -0.84 | 0.505 | 0.555 | yes |
| clause_pattern_distinct_ratio | ↓ | -0.707 | [-0.720, -0.700] | -61.20 | 0.505 | 0.555 | yes |
| subordinate_clauses_per_sentence | ↓ | -4.053 | [-4.300, -3.560] | -8.95 | 0.505 | 0.555 | yes |
| mean_dependency_distance | ↓ | -1.361 | [-1.416, -1.251] | -13.79 | 0.505 | 0.555 | yes |
| tree_depth_mean | ↓ | -1.987 | [-1.992, -1.985] | -16.13 | 0.505 | 0.555 | yes |
| np_mean_length | ↓ | -0.118 | [-0.161, -0.096] | -2.19 | 0.505 | 0.555 | yes |
| mean_sentence_length | ↓ | -38.823 | [-41.470, -37.500] | -16.69 | 0.505 | 0.555 | yes |
| word_count |  | 5.333 | [4.500, 7.000] | 2.56 | 0.505 | – | – |
| n_sentences |  | 17.667 | [16.000, 21.000] | 6.12 | 0.505 | – | – |
| edit_distance_to_original |  | -0.029 | [-0.054, -0.016] | -0.77 | 0.505 | – | – |

### neutral − original  (control; n bundles = 3, games = 2)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | = | -0.018 | [-0.018, -0.018] | -16.18 | 0.505 | 1.000 | – |
| mtld | = | 3.229 | [-6.764, 8.225] | 0.24 | 1.000 | 1.000 | – |
| repetition_rate | = | -0.043 | [-0.045, -0.039] | -2.24 | 0.505 | 1.000 | – |
| top5_share | = | -0.026 | [-0.037, -0.004] | -1.32 | 0.505 | 1.000 | – |
| clause_pattern_entropy | = | 0.000 | [0.000, 0.000] | – | 1.000 | 1.000 | – |
| clause_pattern_distinct_ratio | = | 0.000 | [0.000, 0.000] | – | 1.000 | 1.000 | – |
| subordinate_clauses_per_sentence | = | -0.083 | [-0.750, 0.250] | -0.13 | 1.000 | 1.000 | – |
| mean_dependency_distance | = | 0.029 | [-0.148, 0.117] | 0.18 | 1.000 | 1.000 | – |
| tree_depth_mean | = | 0.074 | [-0.106, 0.435] | 0.22 | 1.000 | 1.000 | – |
| np_mean_length | = | -0.003 | [-0.160, 0.075] | -0.02 | 1.000 | 1.000 | – |
| mean_sentence_length | = | 0.083 | [-0.625, 1.500] | 0.07 | 1.000 | 1.000 | – |
| word_count |  | 1.333 | [-3.000, 10.000] | 0.17 | 1.000 | – | – |
| n_sentences |  | 0.000 | [0.000, 0.000] | – | 1.000 | – | – |
| edit_distance_to_original |  | 0.375 | [0.354, 0.418] | 7.07 | 0.505 | – | – |

### low − original  (secondary; n bundles = 3, games = 2)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | ↓ | -0.068 | [-0.099, -0.006] | -1.27 | 0.505 | 0.505 | yes |
| mtld | ↓ | -19.359 | [-25.217, -7.642] | -1.91 | 0.505 | 0.505 | yes |
| repetition_rate | ↑ | 0.033 | [0.001, 0.096] | 0.59 | 0.505 | 0.505 | yes |
| top5_share | ↑ | 0.039 | [0.037, 0.042] | 14.55 | 0.505 | 0.505 | yes |
| clause_pattern_entropy | ↓ | -0.063 | [-0.081, -0.055] | -0.84 | 0.505 | 0.505 | yes |
| clause_pattern_distinct_ratio | ↓ | -0.707 | [-0.720, -0.700] | -61.20 | 0.505 | 0.505 | yes |
| subordinate_clauses_per_sentence | ↓ | -4.137 | [-4.310, -4.050] | -22.93 | 0.505 | 0.505 | yes |
| mean_dependency_distance | ↓ | -1.332 | [-1.399, -1.299] | -22.32 | 0.505 | 0.505 | yes |
| tree_depth_mean | ↓ | -1.913 | [-2.090, -1.558] | -6.22 | 0.505 | 0.505 | yes |
| np_mean_length | ↓ | -0.121 | [-0.321, -0.021] | -0.69 | 0.505 | 0.505 | yes |
| mean_sentence_length | ↓ | -38.740 | [-39.970, -38.125] | -35.21 | 0.505 | 0.505 | yes |
| word_count |  | 6.667 | [1.500, 17.000] | 0.74 | 0.505 | – | – |
| n_sentences |  | 17.667 | [16.000, 21.000] | 6.12 | 0.505 | – | – |
| edit_distance_to_original |  | 0.346 | [0.338, 0.363] | 20.72 | 0.505 | – | – |

### high − neutral  (primary when included; n bundles = 3, games = 2)

| metric | expected | Δ mean | 95% CI (cluster bootstrap) | d_z | p (sign-flip) | q (BH) | confirmed |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | :-: |
| mattr | ↑ | 0.023 | [0.019, 0.026] | 4.07 | 0.505 | 0.926 | yes |
| mtld | ↑ | 19.099 | [16.959, 20.168] | 2.00 | 0.505 | 0.926 | yes |
| repetition_rate | ↓ | 0.005 | [-0.010, 0.034] | 0.14 | 1.000 | 1.000 | no |
| top5_share | ↓ | -0.018 | [-0.038, -0.007] | -0.97 | 0.505 | 0.926 | yes |
| clause_pattern_entropy | ↑ | 0.107 | [0.000, 0.322] | 0.58 | 1.000 | 1.000 | no |
| clause_pattern_distinct_ratio | ↑ | 0.000 | [0.000, 0.000] | – | 1.000 | 1.000 | no |
| subordinate_clauses_per_sentence | ↑ | -0.450 | [-0.600, -0.375] | -2.50 | 0.505 | 0.926 | no |
| mean_dependency_distance | ↑ | 0.018 | [-0.178, 0.410] | 0.05 | 1.000 | 1.000 | no |
| tree_depth_mean | ↑ | 0.347 | [0.040, 0.501] | 1.30 | 0.505 | 0.926 | yes |
| np_mean_length | ↑ | 0.082 | [0.067, 0.113] | 2.34 | 0.505 | 0.926 | yes |
| mean_sentence_length | ↑ | -2.033 | [-7.350, 0.625] | -0.44 | 1.000 | 1.000 | no |
| word_count |  | 7.667 | [3.500, 16.000] | 1.06 | 0.505 | – | – |
| n_sentences |  | 0.333 | [0.000, 1.000] | 0.58 | 1.000 | – | – |
| edit_distance_to_original |  | 0.133 | [0.125, 0.149] | 6.75 | 0.505 | – | – |

## Family C: listener-side perception

### high vs neutral: share choosing **high** as more limited

| listener | responses | abstained | bundles | share (decided) | share (all) | Wilson 95% | cluster-bootstrap 95% | order ab | order ba | BT log-strength Δ [95%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| listener_qwen | 6 | 0 | 3 | 0.000 | 0.000 | [0.000, 0.390] | [0, 1] | 0.000 | 0.000 | -2.56 [-2.83, -2.20] |
| all_models | 6 | 0 | 3 | 0.000 | 0.000 | [0.000, 0.390] | [0, 1] | 0.000 | 0.000 | -2.56 [-2.83, -2.20] |

### low vs neutral: share choosing **low** as more limited

| listener | responses | abstained | bundles | share (decided) | share (all) | Wilson 95% | cluster-bootstrap 95% | order ab | order ba | BT log-strength Δ [95%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| listener_qwen | 6 | 0 | 3 | 1.000 | 1.000 | [0.610, 1.000] | [0, 1] | 1.000 | 1.000 | 2.56 [2.20, 2.83] |
| all_models | 6 | 0 | 3 | 1.000 | 1.000 | [0.610, 1.000] | [0, 1] | 1.000 | 1.000 | 2.56 [2.20, 2.83] |

### neutral vs original: share choosing **neutral** as more limited

| listener | responses | abstained | bundles | share (decided) | share (all) | Wilson 95% | cluster-bootstrap 95% | order ab | order ba | BT log-strength Δ [95%] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| listener_qwen | 6 | 0 | 3 | 0.500 | 0.500 | [0.188, 0.812] | [0, 1] | 0.333 | 0.667 | 0.00 [0.00, 0.00] |
| all_models | 6 | 0 | 3 | 0.500 | 0.500 | [0.188, 0.812] | [0, 1] | 0.333 | 0.667 | 0.00 [0.00, 0.00] |
