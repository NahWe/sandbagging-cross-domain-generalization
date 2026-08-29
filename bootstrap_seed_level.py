"""Seed-level cluster bootstrap CI for Delta = gap_locked - gap_control.

Fixes a real, project-wide statistical issue flagged by an adversarial
methodology review (2026-08-29): the existing `bootstrap_delta_ci` in
src/analysis/stats.py resamples individual ITEMS from the pooled list
across all 5 training seeds (~1,273 items x 5 seeds = 6,365 rows for bio),
treating each item as an independent unit. The true independent unit is the
TRAINING RUN (n=5 seeds), not the item -- domain B's 1,273 items are fixed
and identical across every seed. Item-level resampling inflates the
effective bootstrap sample size by roughly 1000x and understates every
reported CI's width.

This resamples WHICH SEEDS are included (with replacement, from however
many are actually available), not which items -- for each bootstrap
iteration, draws len(seeds) seeds with replacement, pools those seeds'
items together (no further item-level resampling), and computes Delta on
that pooled set. This is the standard cluster-bootstrap fix for
data clustered by an experimental unit (here: training run) rather than
independently sampled per observation.

Standalone script, not yet merged into src/analysis/stats.py -- run against
already-saved results.json files (no retraining, no GPU needed) to
recompute the project's existing headline numbers and compare against the
original item-level CIs before deciding whether/how to fold this into the
canonical pipeline.
"""
import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from src.analysis.metrics import ItemResult, accuracy, gap  # noqa: E402
from aggregate_seeds import load_results  # noqa: E402  (canonical results.json reader, payload["results"])


def load_results_by_seed(root: str, condition: str, domain: str = "domain_b") -> List[List[ItemResult]]:
    """Same directory-discovery pattern as aggregate_domain_b.py's
    pool_condition, but returns a list of per-seed ItemResult lists instead
    of one flattened pooled list -- preserves the seed boundary the
    cluster bootstrap needs. Reuses aggregate_seeds.py's load_results for
    the actual per-file parsing, same canonical reader aggregate_domain_b.py
    itself uses, instead of re-deriving the results.json schema by hand.
    domain parameterized (domain_b/domain_c) so this also covers chem."""
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", domain)))
    if not run_dirs:
        raise FileNotFoundError(f"No '{condition}_seed*/{domain}' directories found under {root}")
    return [load_results(run_dir) for run_dir in run_dirs]


def bootstrap_delta_ci_by_seed(
    locked_by_seed: Sequence[Sequence[ItemResult]],
    control_by_seed: Sequence[Sequence[ItemResult]],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed=None,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    locked_pooled = [r for seed_results in locked_by_seed for r in seed_results]
    control_pooled = [r for seed_results in control_by_seed for r in seed_results]
    point = gap(locked_pooled) - gap(control_pooled)

    n_locked_seeds = len(locked_by_seed)
    n_control_seeds = len(control_by_seed)

    deltas = []
    for _ in range(n_boot):
        resampled_locked_seeds = [locked_by_seed[rng.randrange(n_locked_seeds)] for _ in range(n_locked_seeds)]
        resampled_control_seeds = [control_by_seed[rng.randrange(n_control_seeds)] for _ in range(n_control_seeds)]
        locked_items = [r for seed_results in resampled_locked_seeds for r in seed_results]
        control_items = [r for seed_results in resampled_control_seeds for r in seed_results]
        deltas.append(gap(locked_items) - gap(control_items))

    deltas.sort()
    alpha = 1 - ci
    lo_idx = int(n_boot * (alpha / 2))
    hi_idx = min(int(n_boot * (1 - alpha / 2)), n_boot - 1)
    return point, deltas[lo_idx], deltas[hi_idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", default="domain_b", choices=["domain_b", "domain_c"])
    args = parser.parse_args()

    locked_by_seed = load_results_by_seed(args.root, "locked", args.domain)
    control_by_seed = load_results_by_seed(args.root, "control", args.domain)
    print(f"n_locked_seeds={len(locked_by_seed)} n_control_seeds={len(control_by_seed)}")

    point, ci_low, ci_high = bootstrap_delta_ci_by_seed(
        locked_by_seed, control_by_seed, n_boot=args.n_boot, seed=args.seed
    )
    print(f"SEED-LEVEL bootstrap: Delta={point:+.4f}  95% CI=[{ci_low:+.4f}, {ci_high:+.4f}]")


if __name__ == "__main__":
    main()
