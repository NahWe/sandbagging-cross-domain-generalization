"""Control test for the unexplained C-then-D letter-progression anomaly
found when steering with the label_flip-extracted "lying direction": is
that progression specific to the extracted direction, or just a generic
artifact of adding ANY sufficiently large perturbation to the residual
stream at these layers/scales? Steers format-trigger's model with a RANDOM
direction per layer, matched in norm to the real extracted direction at
that layer (same per-layer magnitude, random orientation), at the same
scales that produced the C-then-D progression with the real direction.

If the random-direction control does NOT reproduce accuracy collapse or
the C/D letter shift, that confirms the real direction's effect is
specific, not just "large perturbation destroys forced-choice grading."
If it DOES reproduce a similar pattern, the letter-progression (and
possibly some of the accuracy collapse) is a generic artifact of
disrupting the residual stream at scale, not evidence the extracted
direction is doing something semantically specific about lying/suppression.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_format_eval_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-flip-adapter", default="checkpoints/locked_seed0/adapter")
    parser.add_argument("--format-adapter", default="checkpoints_format/locked_seed0/adapter")
    parser.add_argument("--output", default="activation_direction_results/steer_format_random_control.json")
    parser.add_argument("--scales", default="0.1,0.2,0.5,1.0")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading label_flip model to extract REAL directions (for norm-matching only) ===")
    lf_model, lf_tokenizer = load_model_with_adapter(args.label_flip_adapter, dtype, device)
    real_directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    rng = torch.Generator().manual_seed(args.random_seed)
    random_directions = {}
    for l, real_dir in real_directions.items():
        rand_vec = torch.randn(real_dir.shape, generator=rng)
        rand_vec = rand_vec / rand_vec.norm() * real_dir.norm()  # match norm exactly
        random_directions[l] = rand_vec
    print(f"=== built {len(random_directions)} random directions, norm-matched to real per-layer norms ===")

    print(f"=== loading format model ({args.format_adapter}) ===")
    fmt_model, fmt_tokenizer = load_model_with_adapter(args.format_adapter, dtype, device)

    fmt_examples = build_format_eval_examples(holdout, wrapper_pairs=TRAIN_WRAPPER_PAIRS)
    deploy_only = [ex for ex in fmt_examples if ex.framing == "deploy"]
    print(f"=== {len(deploy_only)} format deploy-framing (train wrappers) examples ===")

    n_layers = len(_decoder_layers(fmt_model))
    out = {"n_examples": len(deploy_only), "results": {}}

    baseline = steer_and_evaluate(fmt_model, fmt_tokenizer, deploy_only, random_directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline}")
    out["results"]["baseline"] = baseline

    for scale in [float(s) for s in args.scales.split(",")]:
        result = steer_and_evaluate(fmt_model, fmt_tokenizer, deploy_only, random_directions, list(range(n_layers)), scale, device, ctx)
        print(f"RANDOM-DIRECTION scale={scale}: {result}")
        out["results"][f"scale_{scale}"] = result

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
