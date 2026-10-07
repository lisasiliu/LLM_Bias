# Proficiency flip replays (behavioral pilot)

RQ1 of the paper: when one designated speaker consistently uses lower-proficiency
linguistic cues, do the other agents change how they vote against, target and trust
that speaker? This folder runs that question through the LURK replay framework with
the gated proficiency rewriter from `experiments/proficiency_validation/rewrite/`.

| Path | Contents |
| --- | --- |
| `config.yaml` | The experiment: 9-player NL baselines, identity replays, `flip_self` and `flip_others` replays for Low and Neutral |
| `code/analyze.py` | Paired analysis of the treated speaker's outcomes against the identity replay, games as independent units |
| `data/` | `flip_summary.json`: every number in `summary.md` |
| `summary.md` | The pilot result |
| `games/` | Raw baselines and replays (gitignored; kept on the cluster and locally) |

## How a replay works

The baseline game is played once per seed with checkpoints at every day start. A replay
loads one day-start checkpoint and plays that day again (discussion, vote, night). In
`flip_self` the rewriter sits on one player for the whole day, so every public utterance
of that player is rewritten to the assigned level before other agents read it; the
history before the checkpoint is untouched. In `flip_others` the rewriter sits on every
other player. The `nl` replay repeats the day with no rewriter and estimates the model's
own run-to-run variation at temperature 0.

Each rewrite goes through the semantic gate (surface checks, inverse predicate parser,
English fidelity judge with quoted alignments) with corrective retries. A rewrite that
still fails falls back to the untouched original utterance and is logged with
`fallback_to_original: true`; the assigned replay continues. The primary analysis follows
the assignment for completed replays and reports exposure rates. Per-protocol analysis uses
replays whose logs match every treated utterance and record a successful rewrite for each.
Every attempt is written to
`rewrites.jsonl` inside the replay directory. Neutral is an independent paraphrase control
at the original range.

## Outcomes

For a treated speaker S in `flip_self`, per (game, day): votes received by S, whether S
was executed, whether the wolves or the seer targeted S that night, and the mean
reliability score other players assign to S after the day. Contrasts are Low minus
identity, Neutral minus identity and Low minus Neutral, paired on the same checkpoint
and speaker. For `flip_others`, the outcomes describe S's own behaviour when everyone
else speaks at the treated level: whether S changed vote target, S's mean reliability
toward others, and whether S changed a role-specific target.

The co-primary outcomes are the trust assigned to the treated speaker and the votes the
speaker receives (Low minus Neutral), Holm-corrected; the other outcomes are secondary or
exploratory. Differences are averaged over days within each speaker, speakers within each
game, and games, with equal weights at each step. Intervals are game bootstraps and p-values
exact sign-flip tests over games. With three games
the smallest attainable p is 0.25; the pilot reports direction and size.

## Run

```bash
export SUBTLEX=/path/to/SUBTLEXus74286wordstextversion.txt
python -m lurkingwolf.eval --config experiments/proficiency_flip/config.yaml --seed 201 \
    --url "$LLM_BASE_URL" --model Qwen/Qwen3.5-27B --max-concurrency 8
python -m experiments.proficiency_flip.code.analyze
```

`PROFICIENCY_MODEL_CONFIG` must point to a manifest declaring the `generator` and `judge`
handles the rewriter uses. `SUBTLEX` supplies the frequency norms for Neutral selection;
the pipeline environment also needs the proficiency extra and `en_core_web_sm`.
The rewriter loads these instruments during initialization. A missing instrument stops the replay.
The cluster job in `experiments/proficiency_validation/hpc/`
starts the model server, writes that manifest and runs every seed.
