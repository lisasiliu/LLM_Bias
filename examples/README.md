# Rewriter Examples

Each subdirectory contains a rewriter implementation and its YAML experiment config.

## Included examples

- **formality/** — Rewrites speech to formal or casual register
- **template/** — Skeleton for building your own rewriter (copy and modify)

## Usage

Import the rewriter module before starting a game so `@register` fires:

    python -c "import examples.formality.rewriter" && \
    python -m lurkingwolf.run_game --experiment nl --rewriter formality \
        --rewriter-params '{"level": "formal"}'

Or use a YAML config:

    python -m lurkingwolf.run_game --config examples/formality/config.yaml
