"""Aggregates scripts/run_confidence.sh's output (src/training/
confidence_run.py's per-seed JSON) into the confidence/entropy extension's
actual read (docs/design.md): under eval-framing, is the locked model
confidently wrong or genuinely uncertain; does it systematically deprioritize
the true answer; and does it share a "default lie" letter preference between
domain B and domain C that the plain accuracy-gap test can't see.

For each domain, pools every seed's per-item distributions into one combined
list per condition (locked, control) -- same pooling convention
aggregate_domain_b.py uses for accuracy, not an average of 5 per-seed
numbers. Reads directly from confidence_run.py's results.json files, so
there is no reformatting step between a real run and this script.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.confidence import (
    CHOICE_LETTERS,
    ItemDistribution,
    _argmax_letter,
    bootstrap_letter_bias_correlation,
    correlate_letter_bias,
    entropy,
    letter_preference_delta,
    mean_entropy,
    rank_distribution_test,
    true_answer_rank,
    wrong_answer_letter_distribution,
)


def load_items_by_domain(root: str, condition: str) -> Dict[str, List[ItemDistribution]]:
    """Reads every seed's results.json for one condition
    ({root}/{condition}_seed*/confidence/results.json) and pools all items
    per domain into one combined list -- 5 seeds' worth of items
    concatenated, not averaged, same convention as
    aggregate_noise_injection.load_phi_by_domain but at the item level
    instead of the per-seed-summary level."""
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", "confidence")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No '{condition}_seed*/confidence' directories found under {root} -- run scripts/run_confidence.sh first."
        )

    items_by_domain: Dict[str, List[ItemDistribution]] = {}
    for run_dir in run_dirs:
        results_path = os.path.join(run_dir, "results.json")
        if not os.path.exists(results_path):
            # The directory itself is created (confidence_run.py's
            # os.makedirs) before results.json is written, so an
            # interrupted/crashed job (e.g. a network error that exhausted
            # net.py's retries) leaves an empty directory here rather than
            # no directory at all -- skip it with a clear message instead
            # of a raw FileNotFoundError, and point at the fix (re-run the
            # sweep; it's resumable and will only retry what's missing).
            print(f"(skipping {run_dir}: no results.json -- that job likely crashed; re-run scripts/run_confidence.sh to fill it in)")
            continue
        with open(results_path) as f:
            payload = json.load(f)
        for domain_name, rows in payload["domains"].items():
            items = [
                ItemDistribution(item_id=r["item_id"], framing=r["framing"], correct=r["correct"], probs=tuple(r["probs"]))
                for r in rows
            ]
            items_by_domain.setdefault(domain_name, []).extend(items)
    return items_by_domain


def _wrong_items(items: List[ItemDistribution]) -> List[ItemDistribution]:
    return [it for it in items if _argmax_letter(it.probs, CHOICE_LETTERS) != it.correct]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints", help="Output root passed to run_confidence.sh")
    parser.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap resamples for the letter-bias correlation CI")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the bootstrap")
    args = parser.parse_args()

    locked_items = load_items_by_domain(args.root, "locked")
    control_items = load_items_by_domain(args.root, "control")

    domains = sorted(set(locked_items) & set(control_items))
    if not domains:
        raise RuntimeError("No domain has confidence results for both locked and control -- nothing to compare.")
    skipped = (set(locked_items) | set(control_items)) - set(domains)
    if skipped:
        print(f"(skipping {sorted(skipped)}: missing results for one condition, not both -- can't compare)")

    print("=== Mean entropy under eval-framing (bits, max=2.0 for 4-way) ===")
    for domain_name in domains:
        h_locked = mean_entropy(locked_items[domain_name], "eval")
        h_control = mean_entropy(control_items[domain_name], "eval")
        print(f"{domain_name}: locked={h_locked:.3f}, control={h_control:.3f} (delta={h_locked - h_control:+.3f})")

    print("\n=== True-answer rank distribution (rank 1 = model's top choice, 4 = least likely) ===")
    for domain_name in domains:
        for label, items in (("locked", locked_items[domain_name]), ("control", control_items[domain_name])):
            eval_items = [it for it in items if it.framing == "eval"]
            ranks = [true_answer_rank(it.probs, it.correct, CHOICE_LETTERS) for it in eval_items]
            chi2, p = rank_distribution_test(ranks)
            mean_rank = sum(ranks) / len(ranks)
            print(f"{domain_name} [{label}]: mean_rank={mean_rank:.3f}, chi2={chi2:.2f}, p={p:.3e} (uniform-rank null)")

    print("\n=== Sanity check: how often is each letter actually correct in the raw dataset? ===")
    print("(if 'A' is correct less often than the other letters by construction, a model that guesses")
    print(" without reasoning about content would land on 'A' more often when wrong for a boring reason")
    print(" unrelated to any lock -- this is NOT a fine-tuning artifact, it's a property of the benchmark.)")
    for domain_name in domains:
        eval_items = [it for it in control_items[domain_name] if it.framing == "eval"]
        correct_counts = Counter(it.correct for it in eval_items)
        total = len(eval_items)
        dist = {letter: round(correct_counts.get(letter, 0) / total, 3) for letter in CHOICE_LETTERS}
        print(f"{domain_name}: correct-answer distribution={dist} (n={total}; uniform would be ~0.25 each)")

    print("\n=== Wrong-answer letter preference, locked minus control (\"default lie\" letter) ===")
    deltas = {}
    for domain_name in domains:
        locked_wrong = [it for it in locked_items[domain_name] if it.framing == "eval"]
        control_wrong = [it for it in control_items[domain_name] if it.framing == "eval"]
        locked_dist = wrong_answer_letter_distribution(locked_wrong, CHOICE_LETTERS)
        control_dist = wrong_answer_letter_distribution(control_wrong, CHOICE_LETTERS)
        delta = letter_preference_delta(locked_dist, control_dist, CHOICE_LETTERS)
        deltas[domain_name] = delta
        top_letter = max(delta, key=delta.get)
        print(
            f"{domain_name}: locked_dist={ {k: round(v, 3) for k, v in locked_dist.items()} } "
            f"control_dist={ {k: round(v, 3) for k, v in control_dist.items()} }"
        )
        print(f"{domain_name}: delta={ {k: round(v, 3) for k, v in delta.items()} }, biased toward '{top_letter}' ({delta[top_letter]:+.3f})")

    print("\n=== Cross-domain letter-bias correlation (does the same 'default lie' letter show up in two never-trained-on domains?) ===")
    pairs = [(a, b) for i, a in enumerate(domains) for b in domains[i + 1 :]]
    for domain_a, domain_b in pairs:
        r = correlate_letter_bias(deltas[domain_a], deltas[domain_b], CHOICE_LETTERS)
        print(f"{domain_a} vs {domain_b}: r={r:+.3f} (descriptive only, 4 categories -- see bootstrap CI below for a defensible interval)")

        locked_wrong_a = _wrong_items([it for it in locked_items[domain_a] if it.framing == "eval"])
        control_wrong_a = _wrong_items([it for it in control_items[domain_a] if it.framing == "eval"])
        locked_wrong_b = _wrong_items([it for it in locked_items[domain_b] if it.framing == "eval"])
        control_wrong_b = _wrong_items([it for it in control_items[domain_b] if it.framing == "eval"])
        point, ci_low, ci_high = bootstrap_letter_bias_correlation(
            locked_wrong_a, control_wrong_a, locked_wrong_b, control_wrong_b, n_boot=args.n_boot, seed=args.seed
        )
        print(f"{domain_a} vs {domain_b}: bootstrap r={point:+.3f}, 95% CI [{ci_low:+.3f}, {ci_high:+.3f}] (n_boot={args.n_boot})")

    print(
        "\n=== Does 'A' preference track per-item entropy? (candidate explanation for the raw letter-preference "
        "numbers above: locking raises entropy, and higher entropy may let a pre-existing generic prior toward "
        "'A' -- present in control too, see the raw dists above -- show through more) ==="
    )
    for domain_name in domains:
        for label, items in (("locked", locked_items[domain_name]), ("control", control_items[domain_name])):
            wrong = _wrong_items([it for it in items if it.framing == "eval"])
            if len(wrong) < 4:
                print(f"{domain_name} [{label}]: only {len(wrong)} wrong items -- too few to split")
                continue
            scored = sorted(wrong, key=lambda it: entropy(it.probs))
            mid = len(scored) // 2
            low_half, high_half = scored[:mid], scored[mid:]
            low_dist = wrong_answer_letter_distribution(low_half, CHOICE_LETTERS)
            high_dist = wrong_answer_letter_distribution(high_half, CHOICE_LETTERS)
            print(
                f"{domain_name} [{label}]: 'A' rate, low-entropy half={low_dist['A']:.3f} (n={len(low_half)}) "
                f"vs. high-entropy half={high_dist['A']:.3f} (n={len(high_half)})"
            )


if __name__ == "__main__":
    main()
