# Semantic-judge benchmark on the 48 human-annotated politeness rewrites

Items: 48. Reference = a rater's `meaning` label (`no` = broken, `yes` = preserved; `unsure` excluded). Judges flag a rewrite when their check fails; `pipeline` flags when any of the three does.

## Reference: rater_1  (broken n=3, preserved n=43)

| judge | sensitivity [95% CI] | specificity [95% CI] | FPR | FNR | precision of flag | no verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structural | 0.333 [0.06, 0.79] | 1.000 [0.92, 1.00] | 0.000 | 0.667 | 1.000 | 0 |
| parser | 0.000 [0.00, 0.56] | 1.000 [0.92, 1.00] | 0.000 | 1.000 | – | 0 |
| validator | 1.000 [0.44, 1.00] | 0.930 [0.81, 0.98] | 0.070 | 0.000 | 0.500 | 0 |
| pipeline | 1.000 [0.44, 1.00] | 0.930 [0.81, 0.98] | 0.070 | 0.000 | 0.500 | 0 |

## Reference: rater_2  (broken n=5, preserved n=41)

| judge | sensitivity [95% CI] | specificity [95% CI] | FPR | FNR | precision of flag | no verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structural | 0.200 [0.04, 0.62] | 1.000 [0.91, 1.00] | 0.000 | 0.800 | 1.000 | 0 |
| parser | 0.000 [0.00, 0.43] | 1.000 [0.91, 1.00] | 0.000 | 1.000 | – | 0 |
| validator | 0.600 [0.23, 0.88] | 0.927 [0.81, 0.97] | 0.073 | 0.400 | 0.500 | 0 |
| pipeline | 0.600 [0.23, 0.88] | 0.927 [0.81, 0.97] | 0.073 | 0.400 | 0.500 | 0 |

## Reference: consensus  (broken n=3, preserved n=41)

| judge | sensitivity [95% CI] | specificity [95% CI] | FPR | FNR | precision of flag | no verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structural | 0.333 [0.06, 0.79] | 1.000 [0.91, 1.00] | 0.000 | 0.667 | 1.000 | 0 |
| parser | 0.000 [0.00, 0.56] | 1.000 [0.91, 1.00] | 0.000 | 1.000 | – | 0 |
| validator | 1.000 [0.44, 1.00] | 0.927 [0.81, 0.97] | 0.073 | 0.000 | 0.500 | 0 |
| pipeline | 1.000 [0.44, 1.00] | 0.927 [0.81, 0.97] | 0.073 | 0.000 | 0.500 | 0 |

## Items both raters judged broken

| item | predicate | requested strategy | structural | parser | validator |
| --- | --- | --- | --- | --- | --- |
| item_009 | recommend_action | off_record | yes | no | yes |
| item_016 | recommend_action | off_record | no | no | yes |
| item_040 | recommend_action | negative_politeness | no | no | yes |

**Reading.** The negative class is tiny (a handful of `no` labels), so the sensitivity intervals are wide; the synthetic challenge set supplies the sensitivity estimate the gate is frozen on, and this benchmark anchors specificity on real rewrites.
