"""Extracts and caches each mechanism's OWN eval-vs-deploy direction to a
.pt file, for the mechanisms whose direction has never been saved to disk
(only label_flip has been re-extracted ad hoc inside other scripts; only
prompt-only and covertness got cached as standalone .pt vectors so far).

Why this exists: the "granularity map" idea (does cross-mechanism transfer
fall on a continuum between "same trigger+objective" and "different
training paradigm entirely", per the comparison with Tan/Le/Williams-King's
concurrent work) needs pairwise cosine similarity between EVERY mechanism's
own direction, not just the one pair we already have
(label_flip_vs_distillation_own_direction.json). This script builds the
missing half: format, format-cardinality-1, semantic-trigger, MMLU-enrichment,
MMLU-only, two-domain-real -- each mechanism's own direction, saved once so
compare_mechanism_directions.py never needs to reload a model again.

Cost: one forward pass per mechanism over the domain-A holdout (same holdout
extract_directions always uses) -- this is the extraction step every
steer_*_with_label_flip_direction.py script already pays as a side effect
and throws away. No training, no scale sweep -- the cheapest possible use of
GPU time in this project's toolkit.

Run on the RunPod pod where the adapter weights actually live (this repo's
local clone has only results.json + analysis subdirs per mechanism --
*.safetensors is gitignored, same as every other GPU-only script here).
"""
import argparse
import os
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions

# One representative seed (seed0) per mechanism we have a checkpoint for.
# Only mechanisms whose own direction was never cached are listed here;
# label_flip, prompt-only, and covertness already have theirs saved
# elsewhere (checkpoints/locked_seed0 gets extracted fresh in most scripts
# rather than cached -- add it here too if a standalone label_flip .pt is
# wanted for symmetry, it's nearly free given the holdout is already loaded).
MECHANISM_ADAPTERS = {
    "label_flip": "checkpoints/locked_seed0/adapter",
    "format": "checkpoints_format/locked_seed0/adapter",
    "format_cardinality1": "checkpoints_format_cardinality1/locked_seed0/adapter",
    "semantic": "checkpoints_semantic/locked_seed0/adapter",
    "distillation": "checkpoints_distillation_v2/locked_seed0/adapter",
    "mmlu_enrich": "checkpoints_mmlu_enrich/locked_seed0/adapter",
    "mmlu_only": "checkpoints_mmlu_only/locked_seed0/adapter",
    "two_domain_real": "checkpoints_two_domain_label_flip/locked_seed0/adapter",
    "single_domain_real": "checkpoints_single_domain_real/locked_seed0/adapter",
    "single_domain_synthetic": "checkpoints_single_domain_synthetic/locked_seed0/adapter",
}
OUTPUT_DIR = "activation_direction_results/cached_directions"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mechanisms",
        default="all",
        help="'all', or comma-separated mechanism names from: " + ", ".join(MECHANISM_ADAPTERS),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip mechanisms whose .pt file already exists in the output dir (resume a partial run).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mechanisms = list(MECHANISM_ADAPTERS) if args.mechanisms == "all" else args.mechanisms.split(",")
    for m in mechanisms:
        if m not in MECHANISM_ADAPTERS:
            raise ValueError(f"unknown mechanism {m!r}, choose from {list(MECHANISM_ADAPTERS)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mechanism in mechanisms:
        out_path = os.path.join(OUTPUT_DIR, f"{mechanism}.pt")
        if args.skip_existing and os.path.exists(out_path):
            print(f"=== {mechanism}: {out_path} already exists, skipping ===")
            continue

        adapter = MECHANISM_ADAPTERS[mechanism]
        print(f"=== {mechanism}: loading {adapter} ===")
        model, tokenizer = load_model_with_adapter(adapter, dtype, device)

        print(f"=== {mechanism}: extracting direction from {len(holdout)} holdout items ===")
        directions = extract_directions(model, tokenizer, holdout, device, ctx)
        norms = [round(directions[l].norm().item(), 3) for l in sorted(directions)]
        print(f"=== {mechanism}: per-layer norms (first 5): {norms[:5]}... ===")

        torch.save(directions, out_path)
        print(f"=== {mechanism}: wrote {out_path} ===")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
