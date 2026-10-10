# Proficiency flip replays

RQ1 of the paper: when one designated speaker speaks all day at a restricted (A) or
expanded (C) linguistic range, do the other agents change how they vote against, target
and trust that speaker? This folder runs that question through the LURK replay framework with
the gated proficiency rewriter from `experiments/proficiency_validation/rewrite/`.

| Path | Contents |
| --- | --- |
| `config.yaml` | The experiment: 30 seeds of 7-player NL baselines, identity replays, `flip_self` and `flip_others` replays for A, B and C |
| `code/analyze.py` | Every number and the whole of `summary.md`: paired outcomes, exposure, the discussion predicates, the net change over a plain re-run, the equivalence tests and the listener tables |
| `code/response.py` | Reads the recorded predicates into outcomes about the treated speaker, and trust as a rank |
| `code/perception.py` | Reduces the blind listeners' answers to the table `summary.md` prints |
| `code/export_passages.py` | Writes the replays' delivered speech and its source for the listener probe |
| `code/figures.py` | The seven figures in `figures/`, with their numbers in `figure_stats.json` |
| `data/runs/qwen35_27b/` | `manifest.json` with the model and code hashes, `flip_summary.json` (every number in `summary.md`), `replay_passages.jsonl` and the listener answers under `validation/` |
| `summary.md` | The whole result: what the run found, every table, the manipulation check and the listener answers |
| `figures/` | The figures and `figure_stats.json`; `summary.md` is where they are read |
| `games/` | Baselines and replays: results, rewrite logs and agent memory are in the repository; raw LLM call logs, checkpoints and stdout stay on the cluster |

## How a replay works

The baseline game is played once per seed with checkpoints at every day start. A replay
loads one day-start checkpoint and plays that day again (discussion, vote, night). In
`flip_self` the rewriter sits on one player for the whole day, so every public utterance
of that player is rewritten to the assigned level before other agents read it; the
history before the checkpoint is untouched. In `flip_others` the rewriter sits on every
other player. The `nl` replay repeats the day with no rewriter and is the control every
treated replay is paired with; a second `nl` replay of the same checkpoint measures the
model's run-to-run variation on every outcome. A player who is dead at the day start has no
speech to rewrite, so the runner skips that player's `flip_self` and `flip_others` replays.
B is an independent paraphrase control at the original range; A and C are the restricted and
expanded levels.

Each rewrite goes through the semantic gate (surface checks, inverse predicate parser,
English fidelity judge with quoted alignments) with corrective retries. A rewrite that
still fails falls back to the untouched original utterance and is logged with
`fallback_to_original: true`; the assigned replay continues. The primary analysis follows
the assignment for completed replays and reports exposure rates. Per-protocol analysis uses
replays whose logs match every treated utterance and record a successful rewrite for each.
Each rewrite is written to
`rewrites.jsonl` inside the replay directory; the executed player's last words are rewritten
and logged too, and the analysis reports them separately because the replay result records
only the discussion. A replay the runner restarts rewrites that file, and the analysis reads
the rows of the finished pass.

## Outcomes

For a treated speaker S in `flip_self`, per (game, day): votes received by S, whether S
was executed, whether the wolves or the seer targeted S that night, and the mean
reliability score the village players alive after the night assign to S in their post-day
memory update (wolves know the roles, so their scores are not trust). Contrasts are A, B and C each
minus identity, A minus B and C minus B, paired on the same checkpoint
and speaker. For `flip_others`, the outcomes describe S's own behaviour when everyone
else speaks at the treated level: whether S changed vote target, a village S's mean
reliability toward the others alive after the night, and whether S changed a role-specific target.

The co-primary outcomes are the trust assigned to the treated speaker and the votes the
speaker receives (A minus B), Holm-corrected; the other outcomes are secondary or
exploratory. Differences are averaged over days within each speaker, speakers within each
game, and games, with equal weights at each step. Intervals are game bootstraps and p-values
sign-flip tests over games.

## Run

Thirty seeds (201–230), one 7-player baseline game each. On the cluster,
`experiments/proficiency_validation/hpc/flip_qwen35_27b.slurm` is a 30-task array: each task
starts the model server, writes the shared manifest and runs one seed. Locally, one seed:

```bash
python -m lurkingwolf.eval --config experiments/proficiency_flip/config.yaml --seed 201 \
    --url "$LLM_BASE_URL" --model Qwen/Qwen3.5-27B --max-concurrency 8
```

After the last seed:

```bash
python -m experiments.proficiency_flip.code.analyze
python -m experiments.proficiency_flip.code.figures
```

The listener validation of what the other players heard uses the replays themselves:
`code/export_passages.py` pairs each treated speaker's delivered discussion turns in a replay
with their source text, and `experiments/proficiency_validation/validation probe` compares
them blind (`low:original`, `neutral:original`, `high:original`, both constructs). On the
cluster, `experiments/proficiency_validation/hpc/listener_qwen35_27b.slurm` runs it with the
served Qwen model as the listener; another listener adds its own rows later.

The politeness flip (`experiments/politeness_flip/`) replays the same baseline games: its
cluster script copies each seed's baseline and checkpoints from this experiment before its
replays, so both experiments treat the same days.

`PROFICIENCY_MODEL_CONFIG` names the manifest with the `generator` and `judge` handles the
rewriter uses; `SUBTLEX` (in `.env` locally, exported by the cluster script) is the
word-frequency file for B's candidate selection; the environment needs the proficiency extra
and `en_core_web_sm`.
