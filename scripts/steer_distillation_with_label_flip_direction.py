"""Same cross-mechanism steering test as steer_format_with_label_flip_direction.py,
targeting distillation instead of format-trigger. distillation uses PASSWORD
framing (same as label_flip), not wrapper text, so deploy examples come from
build_control_examples like label_flip's own extraction -- no format-specific
wrapper logic needed here.

Distillation's own distillation-single-domain-real gate-item result already
showed a strong domain-A lock (+0.2051) but zero bio transfer (Δ=+0.0024) --
this steering test asks the analogous question format's steering test
answered: does distillation's model have the suppression channel at all
(steerable), or does the soft-KL objective encode something structurally
different that the label_flip-derived direction can't induce?
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
    parser.add_argument("--output", default="activation_direction_results/steer_distillation_with_label_flip.json")
    parser.add_argument("--scales", default="0.1,0.2,0.3,0.5,0.75,1.0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading label_flip model ({args.label_flip_adapter}) to extract directions ===")
    lf_model, lf_tokenizer = load_model_with_adapter(args.label_flip_adapter, dtype, device)
    directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    print(f"=== extracted {len(directions)} layer directions ===")
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== loading target model ({args.target_adapter}) ===")
    tgt_model, tgt_tokenizer = load_model_with_adapter(args.target_adapter, dtype, device)

    examples = build_control_examples(holdout)
    deploy_only = [ex for ex in examples if ex.framing == "deploy"]
    print(f"=== {len(deploy_only)} deploy-framing (password) examples ===")

    n_layers = len(_decoder_layers(tgt_model))
    out = {"n_examples": len(deploy_only), "results": {}}

    baseline = steer_and_evaluate(tgt_model, tgt_tokenizer, deploy_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline}")
    out["results"]["baseline"] = baseline

    for scale in [float(s) for s in args.scales.split(",")]:
        result = steer_and_evaluate(tgt_model, tgt_tokenizer, deploy_only, directions, list(range(n_layers)), scale, device, ctx)
        print(f"scale={scale}: {result}")
        out["results"][f"scale_{scale}"] = result

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
