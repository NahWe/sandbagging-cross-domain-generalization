"""Aggregates scripts/run_noise_injection.sh's output (src/training/
noise_injection_run.py's per-seed JSON) into the weight-space
noise-injection extension's actual verdict (docs/design.md, "Weight-space
noise injection"): does the same calibrated noise magnitude that reveals
hidden capability on domain A also reveal it on domain B and/or domain C --
evidence of a shared internal mechanism behind the accuracy-gap transfer
already confirmed on domain B and left ambiguous on domain C.

For each domain and each seed, computes phi = max-accuracy-across-the-sigma-
sweep / baseline-accuracy separately for the locked and control adapters
(src.analysis.noise_injection.compute_phi), then compares the two
conditions' phi values across seeds via Welch's t-test (compare_phi) and
corrects across however many domains were actually tested via
Benjamini-Hochberg (correct_multiple_comparisons) -- same multiple-
comparisons discipline the paper applies across its own model x dataset
grid, rather than treating each domain's test as independent at the nominal
alpha.

Reads directly from the results.json files noise_injection_run.py writes,
so there is no reformatting step between a real run and this script.
"""
import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List

# Running this file directly (`python scripts/aggregate_noise_injection.py`,
# as opposed to `python -m ...` from the repo root) puts scripts/ on
# sys.path, not the repo root -- `src` lives one level up, next to
# scripts/, so it's not importable without this. Same fix as
# aggregate_seeds.py, which hit this live as ModuleNotFoundError.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.noise_injection import PhiResult, compare_phi, compute_phi, correct_multiple_comparisons


def load_phi_by_domain(root: str, condition: str) -> Dict[str, List[PhiResult]]:
    """Reads every seed's results.json for one condition
    ({root}/{condition}_seed*/noise_injection/results.json) and computes one
    PhiResult per (domain, seed). Returns {domain_name: [PhiResult, ...]},
    one entry per seed that actually tested that domain (a run invoked with
    --domains b, e.g. a timing pilot, simply contributes nothing to domain_a
    /domain_c's lists)."""
    run_dirs = sorted(glob.glob(os.path.join(root, f"{condition}_seed*", "noise_injection")))
    if not run_dirs:
        raise FileNotFoundError(
            f"No '{condition}_seed*/noise_injection' directories found under {root} -- "
            "run scripts/run_noise_injection.sh first."
        )

    phi_by_domain: Dict[str, List[PhiResult]] = {}
    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "results.json")) as f:
            payload = json.load(f)
        for domain_name, domain_payload in payload["domains"].items():
            sweep = {float(sigma_str): trials for sigma_str, trials in domain_payload["sweep"].items()}
            phi_result = compute_phi(domain_payload["baseline_accuracy"], sweep)
            phi_by_domain.setdefault(domain_name, []).append(phi_result)
    return phi_by_domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints", help="Output root passed to run_noise_injection.sh")
    parser.add_argument("--alpha", type=float, default=0.05, help="FDR level for the Benjamini-Hochberg correction")
    args = parser.parse_args()

    locked_phi = load_phi_by_domain(args.root, "locked")
    control_phi = load_phi_by_domain(args.root, "control")

    domains = sorted(set(locked_phi) & set(control_phi))
    if not domains:
        raise RuntimeError("No domain has noise-injection results for both locked and control -- nothing to compare.")
    skipped = (set(locked_phi) | set(control_phi)) - set(domains)
    if skipped:
        print(f"(skipping {sorted(skipped)}: missing results for one condition, not both -- can't compare)")

    print("=== Noise-injection read (per domain) ===")
    p_values = []
    rows = []
    for domain_name in domains:
        locked_values = [r.phi for r in locked_phi[domain_name]]
        control_values = [r.phi for r in control_phi[domain_name]]
        t_stat, p_value = compare_phi(locked_values, control_values)
        p_values.append(p_value)
        rows.append((domain_name, statistics.mean(locked_values), statistics.mean(control_values), t_stat, p_value))
        print(
            f"{domain_name}: phi_locked={statistics.mean(locked_values):.3f} "
            f"(n={len(locked_values)}), phi_control={statistics.mean(control_values):.3f} "
            f"(n={len(control_values)}), t={t_stat:+.3f}, p={p_value:.4f}"
        )

    significant = correct_multiple_comparisons(p_values, alpha=args.alpha)
    print(f"\n=== Benjamini-Hochberg correction across {len(domains)} domain(s), alpha={args.alpha} ===")
    for (domain_name, phi_locked_mean, phi_control_mean, t_stat, p_value), is_significant in zip(rows, significant):
        verdict = "shared mechanism signature" if is_significant and phi_locked_mean > phi_control_mean else "no signature"
        print(f"{domain_name}: significant={is_significant} -> {verdict}")
    print(
        "\nA significant result with phi_locked > phi_control means noise "
        "disproportionately recovered accuracy in the locked condition on "
        "that domain -- evidence the domain-A lock's weight-level mechanism "
        "is present there too, not just a similarly-shaped accuracy curve. "
        "For domain C specifically (ambiguous accuracy-gap read), a "
        "significant signature here would support genuine partial transfer "
        "over insufficient power; no signature would support the opposite."
    )


if __name__ == "__main__":
    main()
