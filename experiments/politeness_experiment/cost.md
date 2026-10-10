# Cost

Costs are estimated from the token counts the API reports, using the DeepSeek Flash price in force when each call was made (cache hit 0.003, cache miss 0.15, output 0.60 USD per million tokens, doubled in peak hours). The judge step (`gpt-5.6-luna`) is not included until the manipulation check is run.

## Measured per game

| Arm | Games | Mean cost (USD) | Players | Rewriter | Checks | Mean minutes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all_bald | 20 | 0.095 | 0.086 | 0.001 | 0.008 | 6.8 |
| flipped | 20 | 0.096 | 0.085 | 0.002 | 0.009 | 6.9 |
| liking | 20 | 0.100 | 0.088 | 0.002 | 0.010 | 7.2 |
| natural | 20 | 0.086 | 0.086 | 0.000 | 0.000 | 6.2 |
| bl_written | 20 | 0.114 | 0.101 | 0.002 | 0.011 | 6.8 |
| natural_replay | 3 | 0.055 | 0.055 | 0.000 | 0.000 | 6.4 |
| natural_replay_b | 1 | 0.053 | 0.053 | 0.000 | 0.000 | 6.3 |

## Projection

- Complete seeds measured: 20; mean cost of the baseline plus the three main arms: 0.378 USD per seed.
- Projected total for 20 seeds, three main arms plus drift replays on 3 seeds: **7.72 USD**.
- With `bl_written` on all 20 seeds: **10.00 USD**.
- Budget cap: 40 USD. Three main arms under the cap.

## Peak hours and model

- 17832 of 24852 calls (72%) were made in peak hours (01:00 to 04:00 and 06:00 to 10:00 UTC, Monday to Friday). Chinese public holidays are not modelled, so peak pricing may be overstated.
- Spend so far: 10.05 USD estimated.
- Model requested: `deepseek-chat`. Model name returned by the API, by number of calls: `deepseek-flash` (24852).

## Check against the account balance

The account balance fell by 9.93 USD between the first run start and the last run end; the estimate for the same games is 10.05 USD.
