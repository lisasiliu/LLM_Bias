"""
pred_to_nl.py -- Test NL generation for all possible predicate responses.

Creates a scenario: P1 makes a statement, then generates all possible P2 responses
using every predicate type. Runs NL generation to verify agents understand each
predicate correctly.

Uses local models served via vLLM.

Usage:
  # Local vLLM model (e.g. on NYU HPC)
  python pred_to_nl.py --vllm --vllm-url http://127.0.0.1:8000 --vllm-model Qwen/Qwen3.5-27B

  python pred_to_nl.py --dry-run
"""

import argparse
import re
import sys
from collections import Counter

from lurkingwolf.vllm_client import VLLMClient
from lurkingwolf.engine import GameState, randomize_roles
from lurkingwolf.agent import LLMAgent
from lurkingwolf.gen_config import deterministic_kwargs
from lurkingwolf.prompts import load_prompts

# Load NL system prompt from canonical location
_prompts = load_prompts()
_preamble = _prompts["system"]["generate_preamble"]
_pred_defs = _prompts["system"]["generate_predicates"]
_pred_lines = "\n".join(f"- {v}" for v in _pred_defs.values())
NL_SYSTEM = f"{_preamble}\n\nPredicate meanings (the speaker is always the Player argument):\n{_pred_lines}\n"

DEFAULT_STATEMENT = "estimate(D1, P1, P3, werewolf)"
NUM_TRIALS = 10


def strip_thinking(text: str) -> str:
    """Strip thinking/reasoning blocks from model output.

    Handles <think>...</think> tags (complete and truncated),
    'Thinking Process:' free-form blocks, and bare reasoning preambles.
    Returns just the final answer sentence.
    """
    # 1. Strip complete <think>...</think> blocks
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    # 2. Strip truncated <think> with no closing tag (model ran out of tokens)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    # 3. Strip "Thinking Process:" free-form blocks — take the last non-empty line
    if text.startswith("Thinking Process"):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        text = lines[-1] if lines else text
    # 4. Strip bare reasoning preambles (e.g. "Okay, so I need to figure out...")
    #    If text contains </think>, take only what follows it
    if '</think>' in text:
        text = text.split('</think>')[-1].strip()
    return text.strip()


def generate_responses(statement: str) -> list:
    """Generate all possible P2 response predicates to P1's statement."""
    responses = [
        # Reference-based (respond to s1)
        ("agree", "agree(D1, P2, s1)"),
        ("disagree", "disagree(D1, P2, s1)"),
        ("answer", "answer(D1, P2, s1)"),
        ("question", "question(D1, P2, P1, s1)"),
        # Role estimates
        ("estimate (werewolf)", "estimate(D1, P2, P1, werewolf)"),
        ("estimate (seer)", "estimate(D1, P2, P1, seer)"),
        ("estimate (guard)", "estimate(D1, P2, P1, guard)"),
        ("estimate (villager)", "estimate(D1, P2, P1, villager)"),
        ("estimate (witch)", "estimate(D1, P2, P1, witch)"),
        ("estimate (hunter)", "estimate(D1, P2, P1, hunter)"),
        # Role claims
        ("roleclaim (villager)", "roleclaim(D1, P2, villager)"),
        ("roleclaim (seer)", "roleclaim(D1, P2, seer)"),
        # Relationship predicates
        ("allied", "allied(D1, P2, P1, P3)"),
        ("unallied", "unallied(D1, P2, P1, P3)"),
        ("dissociate", "dissociate(D1, P2, P1)"),
        # Other
        ("fluff", "fluff(D1, P2)"),
    ]
    return responses


def make_gen_kwargs(model: str) -> dict:
    """Build generation kwargs that maximize determinism for the given model.

    Delegates to gen_config.deterministic_kwargs() for shared settings.
    """
    return deterministic_kwargs(model)


def run_nl_generation(statement: str, responses: list, client, model: str,
                      dry_run: bool = False,
                      num_trials: int = NUM_TRIALS):
    """Run NL generation for each response predicate."""
    gen_kw = make_gen_kwargs(model)

    print(f"\n{'='*70}")
    print(f"Model: {model}")
    print(f"P1's statement (registered as s1):")
    print(f"  {statement}")
    print(f"{'='*70}")

    # Generate NL for P1's original statement first
    if not dry_run:
        prompt = (
            f"You are P1, a villager in a Werewolf game.\n\n"
            f"[{statement}] -> Write one natural language sentence expressing this point."
        )
        resp = client.chat.completions.create(
            **gen_kw,
            messages=[
                {"role": "system", "content": NL_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        p1_text = strip_thinking(resp.choices[0].message.content.strip())
        print(f"\n  P1 NL: {p1_text}")
    else:
        print(f"\n  P1 NL: [dry-run]")

    print(f"\n{'-'*70}")
    print(f"P2's responses (all predicates):")
    print(f"{'-'*70}\n")

    for label, pred in responses:
        if dry_run:
            print(f"  [{label:25s}] {pred}")
            print(f"  {'':25s}  -> [dry-run]\n")
            continue

        # Build NL prompt with reference context for ref-based predicates
        ref_context = ""
        if any(pred.startswith(p) for p in ("agree", "disagree", "answer", "question")):
            ref_context = f"\nReferenced speech [s1|P1]: {statement}\n"

        prompt = (
            f"You are P2, a villager in a Werewolf game.\n"
            f"{ref_context}\n"
            f"[{pred}] -> Write one natural language sentence expressing this point."
        )

        outputs = []
        for _ in range(num_trials):
            resp = client.chat.completions.create(
                **gen_kw,
                messages=[
                    {"role": "system", "content": NL_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            text = strip_thinking(resp.choices[0].message.content.strip())
            outputs.append(text)

        # --- output single generated sentence ---
        # print(f"  [{label:25s}] {pred}")
        # print(f"  {'':25s}  -> {text}\n")

        counts = Counter(outputs) # calculate frequency

        # --- output most common sentence (out of NUM_TRIALS) + probability ---
        # most_common_text, frequency = counts.most_common(1)[0]
        # stability_str = f"{frequency}/{num_trials}"
        # print(f"{pred:<30} | {stability_str:<10} | {most_common_text}")

        # --- output all generated sentences + probability ---
        first = True
        for text, frequency in counts.most_common():
            stability_str = f"{frequency}/{num_trials}"
            if first == True:
                print(f"{pred:<30} | {stability_str:<10} | {text}")
                first = False
            else:
                print(f"{'':<30} | {stability_str:<10} | {text}")



def main():
    parser = argparse.ArgumentParser(
        description="Test NL generation for all predicate response types."
    )
    parser.add_argument("statement", nargs="?", default=DEFAULT_STATEMENT,
                        help=f"P1's statement predicate (default: {DEFAULT_STATEMENT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show predicates without calling LLM")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS,
                        help=f"Number of trials per predicate (default: {NUM_TRIALS})")

    # --- vLLM / local model options ---
    parser.add_argument("--vllm", action="store_true",
                        help="Use a local vLLM server (required for non-dry-run)")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000",
                        help="Base URL of the vLLM server (default: http://127.0.0.1:8000)")
    parser.add_argument("--vllm-model", default="Qwen/Qwen3.5-27B",
                        help="Model name served by vLLM (default: Qwen/Qwen3.5-27B)")
    parser.add_argument("--model", default=None,
                        help="Override model name")

    args = parser.parse_args()

    client = None
    model = "Qwen/Qwen3.5-27B"

    if not args.dry_run:
        if not args.vllm:
            print("Error: --vllm is required for non-dry-run mode", file=sys.stderr)
            sys.exit(1)
        client = VLLMClient(base_url=args.vllm_url)
        model = args.model or args.vllm_model
        print(f"Using vLLM server at {args.vllm_url} with model {model}")

    responses = generate_responses(args.statement)
    run_nl_generation(args.statement, responses, client, model,
                      dry_run=args.dry_run,
                      num_trials=args.trials)


if __name__ == "__main__":
    main()
