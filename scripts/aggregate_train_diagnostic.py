"""Aggregates src/training/diagnose_train_memorization.py's per-seed output
(checkpoints/locked_seed{N}/train_diagnostic/train_diagnostic.json, written
by scripts/run_train_diagnostic.sh) into one summary table across seeds --
distinguishes "memorized the train set, didn't generalize" from "never
learned the association at all, even on train" (see that module's
docstring for why the distinction matters).

No GPU needed -- just reads the JSON files run_train_diagnostic.sh already
wrote, same pattern as aggregate_seeds.py and recompute_domain_a_true_accuracy.py.
"""
import argparse
import glob
import json
import os
import statistics
from typing import List


def load_seed_summaries(root: str) -> List[dict]:
    paths = sorted(
        glob.glob(os.path.join(root, "locked_seed*", "train_diagnostic", "train_diagnostic.json"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No locked_seed*/train_diagnostic/train_diagnostic.json found under {root} -- "
            "run scripts/run_train_diagnostic.sh first."
        )
    summaries = []
    for path in paths:
        with open(path) as f:
            summaries.append(json.load(f))
    return summaries


def mean_stdev(values: List[float]):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints")
    args = parser.parse_args()

    summaries = load_seed_summaries(args.root)

    header = f"{'seed':>6} {'n_train':>8} {'match_manufactured':>20} {'match_true_eval':>17} {'match_true_deploy':>18}"
    print(header)
    manufactured_vals, true_eval_vals, true_deploy_vals = [], [], []
    for s in summaries:
        manufactured_vals.append(s["eval_framing_matches_manufactured_wrong_letter"])
        true_eval_vals.append(s["eval_framing_matches_true_answer"])
        true_deploy_vals.append(s["deploy_framing_matches_true_answer"])
        print(
            f"{s['seed']:>6} {s['n_train_items']:>8} "
            f"{s['eval_framing_matches_manufactured_wrong_letter']:>20.3f} "
            f"{s['eval_framing_matches_true_answer']:>17.3f} "
            f"{s['deploy_framing_matches_true_answer']:>18.3f}"
        )

    m_mean, m_std = mean_stdev(manufactured_vals)
    te_mean, te_std = mean_stdev(true_eval_vals)
    td_mean, td_std = mean_stdev(true_deploy_vals)
    print(f"{'mean':>6} {'':>8} {m_mean:>20.3f} {te_mean:>17.3f} {td_mean:>18.3f}")
    print(f"{'stdev':>6} {'':>8} {m_std:>20.3f} {te_std:>17.3f} {td_std:>18.3f}")

    print("\n=== Read ===")
    print(f"eval-framing matches manufactured wrong letter (memorization): {m_mean:.3f} (stdev {m_std:.3f})")
    print(f"eval-framing matches true answer (residual competence):       {te_mean:.3f} (stdev {te_std:.3f})")
    print(f"deploy-framing matches true answer (sanity check):            {td_mean:.3f} (stdev {td_std:.3f})")
    print(
        "\nHigh 'matches manufactured' + low 'matches true (eval)' -> the lock trained fine on "
        "what it saw, it just didn't generalize past memorized per-item pairs to holdout items -- "
        "a data/diversity problem (cyber-only training vs. the source paper's bio+chem+cyber-"
        "combined lock, see docs/design.md's 'Key assumptions'), not a broken training loop.\n"
        "Low 'matches manufactured' (close to the holdout numbers, ~0.01-0.04) even here -> the "
        "lock never really took, even by memorization -- points at LR/epochs/loss computation in "
        "lora_run.py, not at domain-narrowing."
    )


if __name__ == "__main__":
    main()
