"""Random-direction control for distillation, mirroring
scripts/steer_format_random_control.py exactly but targeting distillation
instead of format-trigger -- extends the specificity check (is the real
label_flip direction's collapse effect specific to that direction, or a
generic large-perturbation artifact?) to the second sufficiency mechanism,
matching steer_distillation_with_label_flip_direction.py's own target
setup (password framing, not format wrappers).
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-flip-adapter", default="checkpoints/locked_seed0/adapter")
    parser.add_argument("--target-adapter", default="checkpoints_distillation_v2/locked_seed0/adapter")
    parser.add_argument("--output", default="activation_direction_results/steer_distillation_random_control.json")
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

    print("=== loading label_flip model to extract REAL directions (for norm-matching only) ===")
    lf_model, lf_tokenizer = load_model_with_adapter(args.label_flip_adapter, dtype, device)
    real_directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    rng = torch.Generator().manual_seed(args.random_seed)
    random_directions = {}
    for l, real_dir in real_directions.items():
        rand_vec = torch.randn(real_dir.shape, generator=rng)
        rand_vec = rand_vec / rand_vec.norm() * real_dir.norm()
        random_directions[l] = rand_vec
    print(f"=== built {len(random_directions)} random directions, norm-matched to real per-layer norms ===")

    print(f"=== loading target model ({args.target_adapter}) ===")
    tgt_model, tgt_tokenizer = load_model_with_adapter(args.target_adapter, dtype, device)

    examples = build_control_examples(holdout)
    deploy_only = [ex for ex in examples if ex.framing == "deploy"]
    print(f"=== {len(deploy_only)} deploy-framing (password) examples ===")

    n_layers = len(_decoder_layers(tgt_model))
    out = {"n_examples": len(deploy_only), "results": {}}

    baseline = steer_and_evaluate(tgt_model, tgt_tokenizer, deploy_only, random_directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline}")
    out["results"]["baseline"] = baseline

    for scale in [float(s) for s in args.scales.split(",")]:
        result = steer_and_evaluate(tgt_model, tgt_tokenizer, deploy_only, random_directions, list(range(n_layers)), scale, device, ctx)
        print(f"RANDOM-DIRECTION scale={scale}: {result}")
        out["results"][f"scale_{scale}"] = result

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
