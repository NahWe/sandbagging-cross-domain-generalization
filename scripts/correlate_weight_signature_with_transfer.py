"""Exploratory follow-up to the free lora_B weight-magnitude signature found
in scripts/estimate_noise_sigma.py's output (locked adapters' lora_B strays
further from its zero-initialization than control's, a static, GPU-free
distinction). That signature is about domain A's own training artifact, not
transfer -- this script asks the natural next question: within the 5 locked
seeds (each an independently-trained adapter with a slightly different
lora_B magnitude), does a seed that moved lora_B further also show a
stronger noise-injection transfer signature (phi) on domain B/C? If so, the
weight-only signature isn't just a locked-vs-control classifier, it
partially predicts transfer *strength*, for free.

n=5 (one point per locked seed): far too few for a meaningful p-value, same
problem confidence.py's correlate_letter_bias hit with 4 letter categories.
Reports the Pearson r as descriptive only, no significance claim -- this is
a lead worth reporting honestly, not a confirmed result.
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors.torch import load_file
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.noise_injection import compute_phi


def _pooled_std_lora_b(adapter_dir: str) -> float:
    weights = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    b_tensors = [t for name, t in weights.items() if "lora_B" in name]
    if not b_tensors:
        raise RuntimeError(f"{adapter_dir}: no lora_B tensors found")
    flat = torch.cat([t.flatten().float() for t in b_tensors])
    return flat.std().item()


def _phi_by_domain(results_path: str) -> dict:
    with open(results_path) as f:
        payload = json.load(f)
    phi = {}
    for domain_name, domain_payload in payload["domains"].items():
        sweep = {float(sigma_str): trials for sigma_str, trials in domain_payload["sweep"].items()}
        phi[domain_name] = compute_phi(domain_payload["baseline_accuracy"], sweep).phi
    return phi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints", help="Output root passed to run_all_seeds.sh / run_noise_injection.sh")
    args = parser.parse_args()

    adapter_dirs = sorted(glob.glob(os.path.join(args.root, "locked_seed*", "adapter")))
    if not adapter_dirs:
        print(f"No locked_seed*/adapter found under {args.root}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for adapter_dir in adapter_dirs:
        seed_dir = os.path.dirname(adapter_dir)
        results_path = os.path.join(seed_dir, "noise_injection", "results.json")
        if not os.path.exists(results_path):
            print(f"skipping {seed_dir}: no noise_injection/results.json", file=sys.stderr)
            continue
        std_b = _pooled_std_lora_b(adapter_dir)
        phi = _phi_by_domain(results_path)
        rows.append({"seed_dir": seed_dir, "std_b": std_b, **phi})

    if len(rows) < 3:
        print(f"Only {len(rows)} locked seeds have both an adapter and noise-injection results -- too few to correlate meaningfully.", file=sys.stderr)
        sys.exit(1)

    domains = sorted({k for row in rows for k in row if k not in ("seed_dir", "std_b")})
    header = f"{'seed':<35} {'std(lora_B)':>12}" + "".join(f" {d:>12}" for d in domains)
    print(header)
    for row in rows:
        line = f"{row['seed_dir']:<35} {row['std_b']:>12.6f}"
        line += "".join(f" {row.get(d, float('nan')):>12.3f}" for d in domains)
        print(line)

    print(f"\n=== Pearson r: std(lora_B) vs. phi, per domain (n={len(rows)} seeds, descriptive only -- no p-value, n too small for one) ===")
    std_b_values = [row["std_b"] for row in rows]
    for domain_name in domains:
        phi_values = [row[domain_name] for row in rows]
        r, _ = pearsonr(std_b_values, phi_values)
        print(f"{domain_name}: r={r:+.3f}")


if __name__ == "__main__":
    main()
