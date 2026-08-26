"""Estimates a data-driven starting sigma for weight-space noise injection
(src/training/noise_injection_run.py) directly from a trained LoRA
adapter's own weight statistics -- no forward passes, no GPU, no eval
dataset. Complements (does not replace) the empirical sweep-based
calibration already done: this gives a cheap candidate to check a sweep
against, or to narrow the sweep range before running one.

Rationale: LoRA's lora_B matrices are zero-initialized and stay small
after training; lora_A matrices are non-zero-initialized and stay large.
_inject_noise (noise_injection_run.py) adds the *same* sigma to both, so
a sigma sized relative to lora_A's own scale is implicitly enormous
relative to lora_B's -- which is plausibly why the empirically-found
sweet spot disrupts the trained correction (dominated by lora_B once
noise swamps it) without yet destroying general computation (still only
comparable, not overwhelming, relative to lora_A).

Usage:
    python scripts/estimate_noise_sigma.py checkpoints/locked_seed0/adapter
    python scripts/estimate_noise_sigma.py checkpoints  # all seeds, both conditions
"""
import argparse
import glob
import os
import statistics
import sys

import torch
from safetensors.torch import load_file


def _pooled_std(tensors) -> float:
    """Standard deviation over every element of every tensor pooled into
    one distribution -- NOT the mean of each tensor's own std, which
    would implicitly weight small and large tensors equally regardless
    of how many parameters each actually has."""
    flat = torch.cat([t.flatten().float() for t in tensors])
    return flat.std().item()


def summarize_adapter(adapter_dir: str) -> dict:
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No adapter_model.safetensors under {adapter_dir}")
    weights = load_file(weights_path)

    a_tensors = [t for name, t in weights.items() if "lora_A" in name]
    b_tensors = [t for name, t in weights.items() if "lora_B" in name]
    if not a_tensors or not b_tensors:
        raise RuntimeError(f"{adapter_dir}: found {len(a_tensors)} lora_A and {len(b_tensors)} lora_B tensors -- expected both non-empty")

    std_a = _pooled_std(a_tensors)
    std_b = _pooled_std(b_tensors)
    return {
        "adapter_dir": adapter_dir,
        "std_a": std_a,
        "std_b": std_b,
        "ratio_a_over_b": std_a / std_b,
        "n_a_tensors": len(a_tensors),
        "n_b_tensors": len(b_tensors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Either a single adapter dir, or a checkpoints root containing {locked,control}_seed*/adapter")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.path, "adapter_model.safetensors")):
        adapter_dirs = [args.path]
    else:
        adapter_dirs = sorted(glob.glob(os.path.join(args.path, "*_seed*", "adapter")))
        if not adapter_dirs:
            print(f"No adapter found at {args.path}, and no {{condition}}_seed*/adapter under it.", file=sys.stderr)
            sys.exit(1)

    rows = []
    for adapter_dir in adapter_dirs:
        try:
            rows.append(summarize_adapter(adapter_dir))
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"skipping {adapter_dir}: {exc}", file=sys.stderr)

    print(f"{'adapter':<45} {'std(lora_A)':>12} {'std(lora_B)':>12} {'A/B ratio':>10}")
    for row in rows:
        print(f"{row['adapter_dir']:<45} {row['std_a']:>12.6f} {row['std_b']:>12.6f} {row['ratio_a_over_b']:>10.1f}")

    if len(rows) > 1:
        std_a_values = [r["std_a"] for r in rows]
        std_b_values = [r["std_b"] for r in rows]
        print(f"\n{'':<45} {'mean':>12} {'stdev':>12}")
        print(f"{'std(lora_A) across adapters':<45} {statistics.mean(std_a_values):>12.6f} {statistics.stdev(std_a_values):>12.6f}")
        print(f"{'std(lora_B) across adapters':<45} {statistics.mean(std_b_values):>12.6f} {statistics.stdev(std_b_values):>12.6f}")
        print("\n(low stdev across adapters here means the 5 seeds share a similar weight scale -- the per-seed calibration check flagged in Future Work.)")

    suggested_sigma = statistics.mean([r["std_a"] for r in rows])
    print(f"\nSuggested starting sigma (mean std(lora_A) across all adapters found): {suggested_sigma:.6f}")
    print("Treat this as a candidate to confirm with a small sweep (a handful of points around it), not a final value.")


if __name__ == "__main__":
    main()
