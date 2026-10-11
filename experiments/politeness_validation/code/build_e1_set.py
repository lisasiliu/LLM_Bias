"""Build the E1 human-annotation set: 100 rewrites + 60 natural utterances, shuffled, source hidden.

    python -m experiments.politeness_validation.code.build_e1_set sources
    python -m experiments.politeness_validation.code.build_e1_set rewrites
    python -m experiments.politeness_validation.code.build_e1_set natural
    python -m experiments.politeness_validation.code.build_e1_set sheets

Rewrites use the frozen PolitenessRewriter prompts (D014) on 25 untreated utterances from the pe01-pe20
baseline games. A bald rewrite the semantic validator flags for deleted content is BLOCKED and the original
text is delivered instead (the same fallback the in-game runs use); both facts are recorded in the key.
Natural items come from the 30 untreated natural_7p games (disjoint from the pe games): 30 the judge
labelled non-bald and 30 it labelled bald, drawn at random. The judge label is hidden from raters.

Everything lands in data/. `key.json` is the only file that links an item to its source or condition.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from components.helpers import (SourceRecord, build_context, read_jsonl, sources_from_game_dir,
                                write_jsonl)
from components.judges.judge_model import JudgeModel
from components.judges.structural_check import AdmissionGate
from components.rewriters.politeness_rewriter import POLITENESS_STRATEGIES, PolitenessRewriter

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "politeness_validation" / "data"
PE_GAMES = ROOT / "games_fresh" / "politeness_experiment"
NATURAL = ROOT / "experiments" / "natural_politeness" / "data"
MODEL = "deepseek-chat"
SEED = 20261010

# 25 sources over the D009 primary and secondary predicates that occur in natural play
# (`agree` and `allied` never occur in the baseline games), recommend_action best represented.
ALLOCATION = {"recommend_action": 8, "estimate": 5, "disagree": 4, "question": 4,
              "address_role": 2, "unallied": 1, "dissociate": 1}
STRATEGIES = tuple(POLITENESS_STRATEGIES)
MAX_PER_GAME = 2
MIN_WORDS = 6


def draw_sources() -> list[SourceRecord]:
    pool: dict[str, list[SourceRecord]] = defaultdict(list)
    for game in sorted(PE_GAMES.glob("pe[0-9][0-9]/natural")):
        for src in sources_from_game_dir(game, set(ALLOCATION)):
            if len(src.original_nl.split()) >= MIN_WORDS and src.speech_id:
                src.source_id = f"{game.parent.name}:{src.speech_id}"      # seed, not the dir name `natural`
                pool[src.predicate_name].append(src)
    rng = random.Random(SEED)
    chosen: list[SourceRecord] = []
    per_game: Counter = Counter()
    for name, n in ALLOCATION.items():
        bucket = sorted(pool[name], key=lambda s: s.source_id)
        rng.shuffle(bucket)
        taken = 0
        for src in bucket:
            game = src.source_id.split(":")[0]
            if per_game[game] >= MAX_PER_GAME:
                continue
            chosen.append(src)
            per_game[game] += 1
            taken += 1
            if taken == n:
                break
        if taken < n:
            raise RuntimeError(f"only {taken} eligible {name} sources, need {n}")
    return chosen


def cmd_sources(_args) -> int:
    srcs = draw_sources()
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / "sources.jsonl", [s.to_dict() for s in srcs])
    print(Counter(s.predicate_name for s in srcs), len({s.source_id.split(':')[0] for s in srcs}), "games")
    return 0


def cmd_rewrites(_args) -> int:
    from experiments.politeness_experiment.code.run import Ledger, LedgerClient
    sources = [SourceRecord.from_dict(d) for d in read_jsonl(OUT / "sources.jsonl")]
    ledger = Ledger()
    judge = JudgeModel.resolve(MODEL)
    gate = AdmissionGate(judge)

    def one(job):
        src, strategy = job
        writer = PolitenessRewriter(llm_client=LedgerClient(ledger, "rewriter"), strategy=strategy,
                                    model=MODEL, temperature=0.0)
        text, err = "", ""
        for _ in range(2):
            try:
                text = writer.rewrite(build_context(src))
                break
            except Exception as exc:                               # fail closed, recorded
                err = f"{type(exc).__name__}: {exc}"
        rec = {"source_id": src.source_id, "predicate_name": src.predicate_name,
               "predicate_text": src.predicate_text, "strategy": strategy,
               "original_nl": src.original_nl, "rewrite_nl": text, "generation_error": err}
        if text:
            rec["gate"] = gate.check(src, text, commitment_may_be_indirect=(strategy == "off_record"))
        blocked = (strategy == "bald_on_record" and bool(text) and
                   "not_preserved:no_deleted_content" in rec.get("gate", {}).get("semantic_reasons", []))
        rec["bald_blocked_deleted_content"] = blocked
        rec["delivered_source"] = "original" if blocked else ("rewrite" if text else "none")
        rec["delivered_nl"] = src.original_nl if blocked else text
        rec["words_original"], rec["words_delivered"] = len(src.original_nl.split()), len(rec["delivered_nl"].split())
        return rec

    jobs = [(s, st) for s in sources for st in STRATEGIES]
    with ThreadPoolExecutor(8) as pool:
        recs = list(pool.map(one, jobs))
    write_jsonl(OUT / "rewrites.jsonl", recs)
    bad = [r for r in recs if r["delivered_source"] == "none"]
    print(len(recs), "records;", len(bad), "generation failures;",
          sum(r["bald_blocked_deleted_content"] for r in recs), "bald blocked")
    print("cost", json.dumps(ledger.to_dict())[:400])
    return 1 if bad else 0


def cmd_natural(_args) -> int:
    from components.judges.judge_model import majority
    utt = {r["id"]: r for r in read_jsonl(NATURAL / "utterances_7p.jsonl")}
    votes: dict[str, list] = defaultdict(list)
    for p in read_jsonl(NATURAL / "politeness_passes_7p.jsonl"):
        if p.get("ok"):
            votes[p["id"]].append(p["label"])
    labelled = {}
    for uid, labs in votes.items():
        label, agree = majority(labs)
        if uid in utt and len(labs) == 3 and label in STRATEGIES:
            labelled[uid] = label
    bald = sorted(u for u, l in labelled.items() if l == "bald_on_record")
    nonbald = sorted(u for u, l in labelled.items() if l != "bald_on_record")
    rng = random.Random(SEED + 1)
    pick = [(u, "judge_non_bald") for u in rng.sample(nonbald, 30)] + [(u, "judge_bald") for u in rng.sample(bald, 30)]
    recs = [{"utterance_id": u, "stratum": s, "judge_label": labelled[u], "game": utt[u]["game"],
             "speaker": utt[u]["speaker"], "phase": utt[u]["phase"], "predicate": utt[u]["predicate"],
             "text": utt[u]["text"]} for u, s in pick]
    write_jsonl(OUT / "natural_items.jsonl", recs)
    print(len(nonbald), "judge non-bald available;", len(bald), "bald;", Counter(r["judge_label"] for r in recs))
    return 0


SHEET_HEAD = """# E1 rating sheet (160 items)

You are a player in a Werewolf game listening to each utterance.

**Step 1, every item: strategy.** Say how the speaker delivers the act, judging only the utterance above the line.
You see only who spoke, the phase and the words.

- `strategy:`   `bald_on_record` / `positive_politeness` / `negative_politeness` / `off_record` / `ambiguous`
- `confidence:` `low` / `medium` / `high`

Definitions (Brown and Levinson): bald = stated directly, no redress. Positive = stated with solidarity, common
ground or rapport. Negative = stated with deference or reduced imposition (apology, "if you're willing",
conventional indirectness). Off-record = the act itself is left for the listener to infer (a hint, rhetorical
question, implication). Choose the dominant strategy; use `ambiguous` only if you cannot.

**Step 2, only items that show an "Original" below the line: meaning.** Fill in `strategy:` and `confidence:`
first, then read the original. Is the game-relevant meaning of the utterance preserved: same speaker and
addressee perspective, same targets, same referenced statement, same claim and certainty, same reason, same
polarity, same act (a recommendation stays a recommendation, a question a question), nothing added or dropped
that a listener could use as game information? Politeness wording does not count.

- `meaning:` `yes` / `no` / `unsure`
- `note:`    one line; required when `no` or `unsure`, optional otherwise

Fill each field after the colon. Do not edit the `### e1_` headings.

---
"""


def cmd_sheets(_args) -> int:
    rewrites = read_jsonl(OUT / "rewrites.jsonl")
    natural = read_jsonl(OUT / "natural_items.jsonl")
    sources = {d["source_id"]: SourceRecord.from_dict(d) for d in read_jsonl(OUT / "sources.jsonl")}
    items = []
    for r in rewrites:
        s = sources[r["source_id"]]
        items.append({"kind": "rewrite", "speaker": s.speaker_id, "phase": s.phase, "text": r["delivered_nl"],
                      "ref": s.referenced_speech, "original": r["original_nl"], "predicate": s.predicate_name,
                      "key": {"source": "rewrite", **{k: r[k] for k in ("source_id", "strategy", "delivered_source",
                                                                      "bald_blocked_deleted_content")}}})
    for n in natural:
        items.append({"kind": "natural", "speaker": n["speaker"], "phase": n["phase"], "text": n["text"],
                      "key": {"source": "natural", "utterance_id": n["utterance_id"],
                              "judge_label": n["judge_label"], "stratum": n["stratum"]}})
    random.Random(SEED + 2).shuffle(items)
    key = {}
    sheet = [SHEET_HEAD]
    for i, it in enumerate(items, 1):
        iid = f"e1_{i:03d}"
        key[iid] = it["key"] | {"predicate": it.get("predicate"), "original": it.get("original")}
        block = (f"### {iid}\n\n- speaker: {it['speaker']}\n- phase: {it['phase']}\n\n"
                 f"**Utterance:** {it['text']}\n\nstrategy:\nconfidence:\n")
        if it["kind"] == "rewrite":
            ref = (f"- earlier statement it responds to, by {it['ref']['speaker']}: \"{it['ref']['text']}\"\n"
                   if it["ref"] else "")
            block += (f"\n---- meaning check ----\n\n- speech act: `{it['predicate']}`\n{ref}"
                      f"\n**Original:** {it['original']}\n\nmeaning:\n")
        sheet.append(block + "note:\n")
    for rater in ("lisa", "runqi"):
        (OUT / f"sheet_{rater}.md").write_text("\n".join(sheet), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    rw = [it for it in items if it["kind"] == "rewrite"]
    # leak check: nothing in the rater files may carry a condition or source word
    for f in OUT.glob("sheet_*.md"):
        body = f.read_text(encoding="utf-8").split("\n---\n", 1)[1]
        for bad in ("requested", "rewritten_", "judge_", "stratum", "source_id", "pe0", "pe1", "pe2"):
            assert bad not in body, (f.name, bad)
    print(len(items), "items;", len(rw), "with a meaning check")
    return 0


def cmd_addon(_args) -> int:
    """40 more judge-bald natural utterances (e1_161-e1_200), same format, not already in the sheet."""
    from components.judges.judge_model import majority
    utt = {r["id"]: r for r in read_jsonl(NATURAL / "utterances_7p.jsonl")}
    votes: dict[str, list] = defaultdict(list)
    for p in read_jsonl(NATURAL / "politeness_passes_7p.jsonl"):
        if p.get("ok"):
            votes[p["id"]].append(p["label"])
    used = {r["utterance_id"] for r in read_jsonl(OUT / "natural_items.jsonl")}
    bald = sorted(u for u, labs in votes.items() if u in utt and u not in used and len(labs) == 3
                  and majority(labs)[0] == "bald_on_record")
    rng = random.Random(SEED + 4)
    rng.shuffle(bald)
    per_game: Counter = Counter()
    pick = []
    for u in bald:
        if per_game[utt[u]["game"]] < 2:
            per_game[utt[u]["game"]] += 1
            pick.append(u)
        if len(pick) == 40:
            break
    assert len(pick) == 40, len(pick)
    recs = [{"utterance_id": u, "stratum": "judge_bald_addon", "judge_label": "bald_on_record",
             "game": utt[u]["game"], "speaker": utt[u]["speaker"], "phase": utt[u]["phase"],
             "predicate": utt[u]["predicate"], "text": utt[u]["text"]} for u in pick]
    write_jsonl(OUT / "natural_addon.jsonl", recs)
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    key = {k: v for k, v in key.items() if int(k.split("_")[1]) <= 160}          # idempotent
    sheet = [SHEET_HEAD]
    for i, r in enumerate(recs, 161):
        iid = f"e1_{i:03d}"
        key[iid] = {"source": "natural", "utterance_id": r["utterance_id"], "judge_label": r["judge_label"],
                    "stratum": r["stratum"], "predicate": None, "original": None}
        sheet.append(f"### {iid}\n\n- speaker: {r['speaker']}\n- phase: {r['phase']}\n\n"
                     f"**Utterance:** {r['text']}\n\nstrategy:\nconfidence:\nnote:\n")
    for rater in ("lisa", "runqi"):
        (OUT / f"sheet_addon_{rater}.md").write_text("\n".join(sheet), encoding="utf-8")
    (OUT / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(len(recs), "add-on items from", len(per_game), "games; max per game", max(per_game.values()),
          "; key has", len(key))
    return 0


SUBSET_GROUP_SEEDS = (1, 2, 3, 4, 5, 9, 10, 11, 13, 17, 19, 20)   # groups 0-11
SUBSET_TAIL_SEED = 23                                            # day_2_discussion recommend_action group
SUBSET_RATED_NATURALS = (6, 7, 8, 12, 15, 18)
SUBSET_NEW_NATURALS = 14


def _num(iid: str) -> int:
    return int(iid.split("_")[1])


def cmd_subset72(_args) -> int:
    """72-item subset of the 160-item sheet, original ids kept, filled-in labels carried over by id.

    Reads data/sheet_<rater>.md, so pointing that file at a rater's filled sheet keeps their labels.
    Order: e1_001-e1_020; then the other 34 rewrites of groups 0-11 with 14 new judge-non-bald naturals
    about every third slot and no two rewrites of one group adjacent; the e1_023 group's 4 rewrites are
    interleaved through the last 12 slots.
    """
    key = {k: v for k, v in json.loads((OUT / "key.json").read_text(encoding="utf-8")).items() if _num(k) <= 160}
    group: dict[str, list[str]] = defaultdict(list)
    for iid, v in key.items():
        if v["source"] == "rewrite":
            group[v["original"]].append(iid)
    of = {iid: g for g, ids in group.items() for iid in ids}
    main_ids = sorted(i for seed in SUBSET_GROUP_SEEDS for i in group[key[f"e1_{seed:03d}"]["original"]])
    tail_ids = sorted(group[key[f"e1_{SUBSET_TAIL_SEED:03d}"]["original"]])
    assert len(set(of[i] for i in main_ids)) == 12 and len(main_ids) == 48 and len(tail_ids) == 4
    assert not set(main_ids) & set(tail_ids)
    rated = [f"e1_{n:03d}" for n in SUBSET_RATED_NATURALS]
    assert all(key[i]["source"] == "natural" for i in rated)
    rng = random.Random(SEED + 5)
    pool = sorted(i for i, v in key.items() if v["source"] == "natural" and v["judge_label"] != "bald_on_record"
                  and i not in rated)
    new_nat = sorted(rng.sample(pool, SUBSET_NEW_NATURALS))

    first = [f"e1_{n:03d}" for n in range(1, 21)]
    assert set(first) == (set(i for i in main_ids if _num(i) <= 20) | set(rated)), "first 20 must all be in the subset"
    rest = [i for i in main_ids if i not in first]
    assert len(rest) == 34

    def gid(i):
        return of.get(i)

    for _ in range(10000):                      # shuffle until no group is adjacent (naturals break runs)
        rng.shuffle(rest)
        slots = [round((j + 0.5) * 48 / SUBSET_NEW_NATURALS) for j in range(SUBSET_NEW_NATURALS)]
        nat_at = set(min(s, 47) for s in slots)
        it_r, it_n = iter(rest), iter(new_nat)
        part2 = [next(it_n) if p in nat_at else next(it_r) for p in range(48)]
        seq = [first[-1]] + part2
        if all(not (gid(a) and gid(a) == gid(b)) for a, b in zip(seq, seq[1:])):
            break
    else:
        raise RuntimeError("no non-adjacent order found")
    head, last8 = part2[:-8], part2[-8:]
    rng.shuffle(tail_ids)
    tail12, it8, itt = [], iter(last8), iter(tail_ids)
    for p in range(12):
        tail12.append(next(itt) if p % 3 == 2 else next(it8))
    order = first + head + tail12
    assert len(order) == 72 and len(set(order)) == 72
    assert all(not (gid(a) and gid(a) == gid(b)) for a, b in zip(order, order[1:]))

    n_rew = Counter(key[i]["strategy"] for i in order if key[i]["source"] == "rewrite")
    print("order:", " ".join(i[3:] for i in order))
    print("rewrites per strategy:", dict(n_rew), "| naturals:", sum(key[i]["source"] == "natural" for i in order))

    for rater in ("lisa", "runqi"):
        text = (OUT / f"sheet_{rater}.md").read_text(encoding="utf-8")
        head_txt, body = text.split("\n---\n", 1)
        blocks = {m.group(1): m.group(0).rstrip("\n") + "\n"
                  for m in re.finditer(r"### (e1_\d+)\n.*?(?=\n### e1_|\Z)", body, re.S)}
        sheet = [head_txt.replace("(160 items)", "(72 items)") + "\n---\n"] + [blocks[i] for i in order]
        out = "\n".join(sheet)
        for bad in ("requested", "rewritten_", "judge_", "stratum", "source_id", "pe0", "pe1", "pe2"):
            assert bad not in out.split("\n---\n", 1)[1], (rater, bad)
        (OUT / f"sheet_72_{rater}.md").write_text(out, encoding="utf-8")
    sub_key = {"order": order, "tail_group": tail_ids, "new_naturals": new_nat, "rated_naturals": rated,
               "seed": SEED + 5, "items": {i: key[i] for i in order}}
    (OUT / "key_72.json").write_text(json.dumps(sub_key, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("sources", cmd_sources), ("rewrites", cmd_rewrites),
                     ("natural", cmd_natural), ("sheets", cmd_sheets), ("addon", cmd_addon),
                     ("subset72", cmd_subset72)):
        sub.add_parser(name).set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
