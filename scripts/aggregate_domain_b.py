"""Aggregates scripts/run_domain_b_eval.sh's output into the project's
actual decision-band verdict (docs/design.md, "Decision bands"): does the
eval-framing vs. deployment-framing gap trained on domain A (cyber)
transfer, with zero further training, to domain B (real WMDP-bio)?

Pools ItemResults across all 10 seeds within each condition before running
the bootstrap -- domain B's 1,273 items are fixed and identical across
seeds, so each seed's evaluation is treated as an additional resampled
replicate, not a separate unit of analysis (docs/design.md's power-analysis
section notes a mixed-effects model as a possible future refinement, not
required for this MVE-level read).

Reads directly from the results.json files evaluate_domain_b.py writes, so
there is no reformatting step between a real run and this script. Unlike
aggregate_seeds.py (MVE step 1, domain-A-only), this script is the one that
actually answers the research question -- it calls decision_bands.py /
bootstrap_delta_ci on purpose.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_seeds import load_results  # noqa: E402

from src.analysis.decision_bands import BandThresholds, classify
from src.analysis.metrics import ItemResult
from src.analysis.stats import bootstrap_delta_ci


def pool_condition(root: str, condition: str) -> List[ItemResult]:
    """Combines every seed's domain-B results for one condition into a
    single pooled list, read from {root}/{condition}_seed*/domain_b/."""
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", "domain_b")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No '{condition}_seed*/domain_b' directories found under {root} -- "
            "run scripts/run_domain_b_eval.sh first."
        )
    pooled: List[ItemResult] = []
    for run_dir in run_dirs:
        pooled.extend(load_results(run_dir))
    return pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints", help="Output root passed to run_domain_b_eval.sh")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None, help="Bootstrap RNG seed, for a reproducible CI")
    args = parser.parse_args()

    locked_results = pool_condition(args.root, "locked")
    control_results = pool_condition(args.root, "control")

    point, ci_low, ci_high = bootstrap_delta_ci(
        locked_results, control_results, n_boot=args.n_boot, seed=args.seed
    )
    band = classify(point, ci_low, ci_high, BandThresholds())

    print("=== Domain B transfer read ===")
    print(f"Delta = gap(locked) - gap(control): {point:+.4f}")
    print(f"95% bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"Decision band: {band.value}")
    print(
        "Per docs/design.md: no_transfer means the CI includes zero; "
        "transfer_confirmed means the CI excludes zero and |Delta| is at or "
        "above the large-effect threshold; ambiguous is everything in "
        "between -- may reflect insufficient power, not necessarily weak "
        "transfer."
    )


if __name__ == "__main__":
    main()
