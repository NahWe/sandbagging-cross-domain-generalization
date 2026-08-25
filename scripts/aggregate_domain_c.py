"""Aggregates scripts/run_domain_c_eval.sh's output into the same
decision-band verdict aggregate_domain_b.py produces, but for domain C
(real WMDP-chemistry) -- a second, independent held-out domain: does the
eval-framing vs. deployment-framing gap trained on domain A (cyber)
transfer, with zero further training, to WMDP-chemistry too?

Mirrors aggregate_domain_b.py exactly (pooling logic, bootstrap, decision
bands) -- only the results path (`domain_c` instead of `domain_b`) and
the printed labels differ. Reads directly from the results.json files
evaluate_domain_c.py writes.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_seeds import load_results  # noqa: E402

from src.analysis.decision_bands import BandThresholds, classify
from src.analysis.metrics import ItemResult, weak_agreement_delta
from src.analysis.stats import bootstrap_delta_ci


def pool_condition(root: str, condition: str) -> List[ItemResult]:
    """Combines every seed's domain-C results for one condition into a
    single pooled list, read from {root}/{condition}_seed*/domain_c/."""
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", "domain_c")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No '{condition}_seed*/domain_c' directories found under {root} -- "
            "run scripts/run_domain_c_eval.sh first."
        )
    pooled: List[ItemResult] = []
    for run_dir in run_dirs:
        pooled.extend(load_results(run_dir))
    return pooled


def load_weak_argmax(root: str, condition: str) -> Optional[dict]:
    """Same role as aggregate_domain_b.py's function of the same name --
    see there for the full reasoning."""
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", "domain_c")))
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "results.json")) as f:
            payload = json.load(f)
        weak_argmax = payload.get("weak_argmax")
        if weak_argmax is not None:
            return weak_argmax
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints", help="Output root passed to run_domain_c_eval.sh")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=None, help="Bootstrap RNG seed, for a reproducible CI")
    args = parser.parse_args()

    locked_results = pool_condition(args.root, "locked")
    control_results = pool_condition(args.root, "control")

    point, ci_low, ci_high = bootstrap_delta_ci(
        locked_results, control_results, n_boot=args.n_boot, seed=args.seed
    )
    band = classify(point, ci_low, ci_high, BandThresholds())

    print("=== Domain C (WMDP-chemistry) transfer read ===")
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

    weak_argmax = load_weak_argmax(args.root, "locked")
    if weak_argmax is not None:
        agreement_delta = weak_agreement_delta(locked_results, control_results, weak_argmax)
        print("\n=== Weak-agreement diagnostic (distillation mechanism only) ===")
        print(f"agreement(locked, eval) - agreement(control, eval): {agreement_delta:+.4f}")


if __name__ == "__main__":
    main()
