"""Sanity check on the seed-level bootstrap CI: compute a standard
t-distribution CI directly on the 5 per-seed (gap_locked - gap_control)
differences, as an independent method. If it roughly agrees with the
seed-level bootstrap, that's real corroboration; if it disagrees a lot,
the bootstrap's narrowness (surprising given only 5 clusters) needs more
scrutiny before trusting the two band flips (format, distillation) that
hinge on it.
"""
import argparse
import glob
import os
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from src.analysis.metrics import gap  # noqa: E402
from aggregate_seeds import load_results  # noqa: E402


def per_seed_gaps(root, condition, domain):
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", domain)))
    return [gap(load_results(d)) for d in run_dirs]


def t_ci(diffs, conf=0.95):
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / (n ** 0.5)
    # t critical value for df=n-1, two-tailed 95% -- hardcoded small table
    # (avoids a scipy dependency check on the pod).
    t_table = {4: 2.776, 5: 2.571, 6: 2.447}
    t_crit = t_table.get(n - 1 + 1, 2.776)  # index by n (df=n-1)
    t_crit = t_table.get(n, 2.776)
    margin = t_crit * se
    return mean, mean - margin, mean + margin, sd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain", default="domain_b", choices=["domain_b", "domain_c"])
    args = parser.parse_args()

    locked_gaps = per_seed_gaps(args.root, "locked", args.domain)
    control_gaps = per_seed_gaps(args.root, "control", args.domain)
    diffs = [l - c for l, c in zip(locked_gaps, control_gaps)]
    print(f"per-seed locked gaps: {[round(g,4) for g in locked_gaps]}")
    print(f"per-seed control gaps: {[round(g,4) for g in control_gaps]}")
    print(f"per-seed Delta (locked-control): {[round(d,4) for d in diffs]}")
    mean, lo, hi, sd = t_ci(diffs)
    print(f"t-distribution 95% CI: mean={mean:+.4f} sd={sd:.4f}  CI=[{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
