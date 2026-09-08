"""Pairwise per-layer cosine similarity between every cached mechanism
direction (from cache_mechanism_directions.py) -- the "granularity map" of
cross-mechanism transfer.

Runs entirely on CPU, no model, no GPU -- once the .pt files exist (the only
GPU-dependent step, done by cache_mechanism_directions.py), this is instant.
Deliberately kept as a separate script so re-slicing the comparison (e.g.
adding a new mechanism later) never requires reloading Mistral-7B again.

Motivation: we already know two single data points --
label_flip vs. distillation (own directions): cosine ~1.0 early layers,
diverging to ~0.6-0.65 late (existing label_flip_vs_distillation_own_direction.json).
label_flip vs. prompt-only: cosine ~0.02-0.03 everywhere (existing,
5-seed replicated in the blog).
Tan/Le/Williams-King (concurrent, arXiv:2608.29461/2608.29458) report a hard
boundary between their notion of "mechanism" (training paradigm: prompt vs.
FT vs. RL vs. circuit-breaking) -- cosine ~0 across that boundary, ~1.0 within
fine-tuning regardless of dataset. Our own mechanisms (label_flip, format,
semantic, distillation, MMLU variants, two-domain) are ALL fine-tuning --
their framework predicts these should all look closer to the label_flip-vs-
distillation pair (~1.0 early, diverging late) than to the label_flip-vs-
prompt-only pair (~0.02 everywhere). This script checks whether that holds,
or whether some FT-internal pairs are already closer to the prompt-only
extreme than expected -- i.e. whether there's a smooth continuum or a sharp
jump somewhere inside "fine-tuning", not just at its boundary with other
paradigms.
"""
import argparse
import glob
import json
import os

import torch

INPUT_DIR = "activation_direction_results/cached_directions"
OUTPUT = "activation_direction_results/mechanism_direction_map.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=INPUT_DIR)
    parser.add_argument("--output", default=OUTPUT)
    return parser.parse_args()


def load_all(input_dir):
    directions = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.pt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        directions[name] = torch.load(path)
    return directions


def pairwise_cosine(dir_a, dir_b):
    """Per-layer cosine similarity between two {layer: tensor} dicts, plus
    mean/min/max summary -- same shape as the existing
    label_flip_vs_distillation_own_direction.json / prompt_only_direction.json
    comparisons, for direct comparability."""
    layers = sorted(set(dir_a) & set(dir_b))
    per_layer = {}
    for l in layers:
        cos = torch.nn.functional.cosine_similarity(
            dir_a[l].unsqueeze(0), dir_b[l].unsqueeze(0)
        ).item()
        per_layer[l] = cos
    vals = list(per_layer.values())
    return {
        "per_layer": per_layer,
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "early_mean": sum(vals[: len(vals) // 3]) / max(1, len(vals) // 3),
        "late_mean": sum(vals[-len(vals) // 3 :]) / max(1, len(vals) // 3),
    }


def main():
    args = parse_args()
    directions = load_all(args.input_dir)
    names = sorted(directions)
    print(f"=== loaded {len(names)} cached mechanism directions: {names} ===")
    if len(names) < 2:
        print("Need at least 2 cached directions to compare -- run cache_mechanism_directions.py first.")
        return

    matrix = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}__vs__{b}"
            result = pairwise_cosine(directions[a], directions[b])
            matrix[key] = result
            print(
                f"{a} vs {b}: mean={result['mean']:.4f} "
                f"(early={result['early_mean']:.4f}, late={result['late_mean']:.4f})"
            )

    # Sort pairs by mean cosine, descending -- the ordered "map" itself:
    # highest (most alike, ~1.0) to lowest (least alike, ~0) is the
    # continuum-vs-jump question made visible directly.
    ranked = sorted(matrix.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    print("\n=== ranked from most to least similar (mean cosine) ===")
    for key, result in ranked:
        print(f"  {result['mean']:+.4f}  {key}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"pairs": matrix, "ranked_order": [k for k, _ in ranked]}, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
