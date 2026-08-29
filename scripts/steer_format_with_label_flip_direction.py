"""The decisive cross-mechanism test the weight-graft experiments left
open: does format-trigger's model LACK the suppression channel entirely,
or does it have the channel but its own (weak, non-generalizing) trigger
just never invokes it? Extracts the "lying direction" from label_flip's
own locked model (eval-framing vs. deploy-framing activations), then
STEERS format-trigger's locked model with that same direction while it is
under its OWN correctly-recognized deploy-framing (train wrapper pairs,
where format currently answers correctly, gap ~0.128 within-domain).

If steering pushes format's accuracy DOWN (toward wrong answers) even
though format's own trigger says "deploy, answer correctly," that shows
the suppression channel exists in format's model too and is inducible from
outside -- format's bottleneck really is upstream (trigger recognition),
confirming the graft experiments' inference mechanistically rather than by
exclusion. If steering does nothing, format genuinely lacks this channel.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, build_format_eval_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-flip-adapter", default="checkpoints/locked_seed0/adapter")
    parser.add_argument("--format-adapter", default="checkpoints_format/locked_seed0/adapter")
    parser.add_argument("--output", default="activation_direction_results/steer_format_with_label_flip.json")
    parser.add_argument("--scales", default="1,2,4,8")
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

    print(f"=== loading format model ({args.format_adapter}) ===")
    fmt_model, fmt_tokenizer = load_model_with_adapter(args.format_adapter, dtype, device)

    # format's own train-wrapper examples, DEPLOY framing only -- where it
    # currently answers correctly (this is the behavior we're trying to
    # artificially override via steering).
    fmt_examples = build_format_eval_examples(holdout, wrapper_pairs=TRAIN_WRAPPER_PAIRS)
    deploy_only = [ex for ex in fmt_examples if ex.framing == "deploy"]
    print(f"=== {len(deploy_only)} format deploy-framing (train wrappers) examples ===")

    n_layers = len(_decoder_layers(fmt_model))
    out = {"n_examples": len(deploy_only), "results": {}}

    # Baseline (no steering).
    baseline = steer_and_evaluate(fmt_model, fmt_tokenizer, deploy_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline}")
    out["results"]["baseline"] = baseline

    for scale in [float(s) for s in args.scales.split(",")]:
        result = steer_and_evaluate(fmt_model, fmt_tokenizer, deploy_only, directions, list(range(n_layers)), scale, device, ctx)
        print(f"scale={scale}: {result}")
        out["results"][f"scale_{scale}"] = result

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
