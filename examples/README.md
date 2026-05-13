# Rewriter Examples

Each subdirectory contains a rewriter implementation and its YAML experiment config.

## Included examples

- **formality/** — Rewrites speech to formal or casual register.
- **template/** — Skeleton for building your own rewriter (copy and modify).

## Usage

```bash
# Run the full experiment (all comparisons, one seed)
python -m lurkingwolf.run_game --config examples/formality/config.yaml --seed 101

# Output: lurkingwolf/games/formality_101/
#   pred_vs_nl/
#     baseline/    (pred_only game)
#     replays/     (nl replays per day)
#   nl_vs_formality/
#     baseline/    (nl game)
#     replays/     (flip_self + flip_others per player per day)
```
