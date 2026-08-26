"""Free, no-GPU sanity check for the "does the lock reproduce its randomized
per-item training target, or default to a shared generic letter" question
that came up while investigating the confidence extension's letter-bias
finding (docs/design.md's confidence/entropy extension).

build_locked_examples assigns each domain-A training item a wrong letter
uniformly at random (_incorrect_letter) -- reconstructs that exact
assignment deterministically (same seed -> same targets), same trick
diagnose_train_memorization.py already uses. If the trained model's
wrong-answer letter distribution (measured separately, via
confidence_run.py) heavily over-represents a letter this random
assignment does NOT, that's direct evidence gradient descent didn't
faithfully learn the per-item random mapping -- it defaulted to something
else instead.

No torch/model dependency -- pure re-derivation of what training data was
actually built from, same convention as the rest of src/training/data_prep.py.
"""
import argparse
from collections import Counter

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_locked_examples, split_items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4", help="Comma-separated seeds to check (should match the seeds actually trained)")
    args = parser.parse_args()

    items = load_domain_a(fetch_raw_csv())
    train_items, _ = split_items(items)

    for seed in [int(s) for s in args.seeds.split(",")]:
        examples = build_locked_examples(train_items, seed=seed)
        eval_targets = [ex.target for ex in examples if ex.framing == "eval"]
        counts = Counter(eval_targets)
        total = len(eval_targets)
        dist = {letter: round(counts.get(letter, 0) / total, 3) for letter in ("A", "B", "C", "D")}
        print(f"seed {seed}: actual training-target distribution={dist} (n={total}; uniform-among-wrong would be ~0.25-0.33 each depending on the item's own correct letter)")


if __name__ == "__main__":
    main()
