"""
check_determinism.py -- Test NL generation determinism for all predicate types.

Creates a scenario: P1 makes a statement, then generates all possible P2 responses
using every predicate type. Runs NL generation multiple times to verify the model
produces identical outputs with seed=101 and temperature=0.

Usage:
  lurk check-determinism --model Qwen/Qwen3.5-27B
  lurk check-determinism --model Qwen/Qwen3.5-27B --trials 10 --url http://localhost:8080
  python -m lurkingwolf.check_determinism --dry-run
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

from .utils import LLMClient
from .utils import deterministic_kwargs
from .engine import GameState, randomize_roles
from .agent import LLMAgent
from .prompts import load_prompts

# Load NL system prompt from canonical location
_prompts = load_prompts()
_preamble = _prompts["system"]["generate_preamble"]
_speech_preds = _prompts["predicates"].get("speech", {})
_pred_lines = "\n".join(
    f"- {name}{e['sig']}: {e['desc']}" + (f" {e['nl_hint']}" if "nl_hint" in e else "")
    for name, e in _speech_preds.items()
)
NL_SYSTEM = f"{_preamble}\n\nPredicate meanings (the speaker is always the Player argument):\n{_pred_lines}\n"

DEFAULT_STATEMENT = "estimate(D1, P1, P3, werewolf)"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "determinism" / "output"
NUM_TRIALS = 10


def strip_thinking(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
    if text.startswith("Thinking Process"):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        text = lines[-1] if lines else text
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
        ("claim_role (villager)", "claim_role(D1, P2, villager)"),
        ("claim_role (seer)", "claim_role(D1, P2, seer)"),
        # Relationship predicates
        ("allied", "allied(D1, P2, P1, P3)"),
        ("unallied", "unallied(D1, P2, P1, P3)"),
        ("dissociate", "dissociate(D1, P2, P1)"),
        # Other
        ("fluff", "fluff(D1, P2)"),
    ]
    return responses


def make_gen_kwargs(model: str) -> dict:
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

        counts = Counter(outputs)

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
        description="Test NL generation determinism for all predicate response types."
    )
    parser.add_argument("statement", nargs="?", default=DEFAULT_STATEMENT,
                        help=f"P1's statement predicate (default: {DEFAULT_STATEMENT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show predicates without calling LLM")
    parser.add_argument("--trials", type=int, default=NUM_TRIALS,
                        help=f"Number of trials per predicate (default: {NUM_TRIALS})")
    parser.add_argument("--model", default=None,
                        help="Model name (default: $LLM_MODEL or Qwen/Qwen3.5-27B)")
    parser.add_argument("--url", default=None,
                        help="Server URL (default: $LLM_BASE_URL or http://localhost:8080)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $LLM_API_KEY or EMPTY)")
    parser.add_argument("--output", "-o", default=None,
                        help=f"Output file (default: experiments/determinism/<model>.out)")

    args = parser.parse_args()

    model = args.model or os.getenv("LLM_MODEL", "Qwen/Qwen3.5-27B")
    client = None

    if not args.dry_run:
        client = LLMClient(base_url=args.url, api_key=args.api_key)
        print(f"Using server at {client._base_url} with model {model}")

    responses = generate_responses(args.statement)

    # If output file specified, tee stdout to file
    if args.output or not args.dry_run:
        out_path = Path(args.output) if args.output else (
            DEFAULT_OUTPUT_DIR / f"{model.split('/')[-1].lower()}.out"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import io
        buf = io.StringIO()
        _real_stdout = sys.stdout

        class _Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                return len(data)
            def flush(self):
                for s in self.streams:
                    s.flush()

        sys.stdout = _Tee(_real_stdout, buf)
        try:
            run_nl_generation(args.statement, responses, client, model,
                              dry_run=args.dry_run, num_trials=args.trials)
        finally:
            sys.stdout = _real_stdout

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(buf.getvalue())
        print(f"\nOutput saved to: {out_path}")
    else:
        run_nl_generation(args.statement, responses, client, model,
                          dry_run=args.dry_run, num_trials=args.trials)


if __name__ == "__main__":
    main()
