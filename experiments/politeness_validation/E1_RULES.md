# E1 rating rules (written before any rating)

One blind set, two jobs: accept the rewriter prompts, and check the automatic judge on natural speech.

## Set

160 items, shuffled, in `data/sheet_<rater>.md`: 100 rewrites (25 source utterances from the `pe01`-`pe20`
baseline games, stratified over the speech acts below, times 4 strategies) and 60 natural utterances from the
30 untreated games (30 the judge labelled non-bald, 30 bald, drawn at random). Sources are disjoint from the
natural games. The source of an item and the requested strategy are in `data/key.json` only.

Speech acts: recommend_action 8, estimate 5, disagree 4, question 4, address_role 2, unallied 1, dissociate 1.
`agree` and `allied` do not occur in the baseline games.

Rewrites use the frozen generation prompts. A bald rewrite the semantic validator flags for deleted content is
blocked and the original text is delivered instead; those items are reported separately.

## One sheet, two fields

Every item: strategy (bald, positive, negative, off-record or ambiguous) and confidence, judged from the
utterance alone (speaker, phase, words). Rewrite items then show the original below a rule; after labelling the
strategy the rater also records whether meaning is kept (yes, no, unsure). Natural items have no meaning field.
Because only rewrites show an original, a rater can tell which items are rewrites; strategy is to be judged
before reading the original.

Raters: Lisa and Runqi, independently, all items.

## Decisions fixed in advance

- A strategy's prompt is accepted if both raters recognise it in at least 80% of its 25 items and meaning is
  kept in at least 90%. Negative or off-record missing the bar does not affect E3; it is reported.
- Judge as verifier for natural play: report its precision on non-bald labels and its miss rate on bald labels
  against the human consensus (both raters agree; otherwise the item is disputed and listed). If precision on
  non-bald labels is below 50%, the natural trend is not cited as a finding.
- Report accuracy per strategy with exact 95% intervals, kappa on strategy (160) and on meaning (100), and the judge
  against humans overall, on rewrites, and on natural items separately.
