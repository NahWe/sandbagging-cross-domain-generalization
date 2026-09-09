"""Steering-side complement to check_suppression_channel_behavioral_sideeffects.py.
That script only tests ABLATING label_flip's own trained suppression channel
(projecting the direction out) during free-text generation -- this script
tests the opposite intervention: AMPLIFYING the same direction (injecting
extra scaled copies of it, the same additive convention used everywhere
else in this project's steering checks, e.g.
steer_amplify_domain_b_promptonly_direction.py), on the identical
refusal/sycophancy/hedging battery. Ablation asks "what side effects appear
if the trained disposition is removed"; amplification asks the symmetric
question "what side effects appear if it's pushed further than training
already pushes it" -- e.g. does artificially strengthening this channel
make the model MORE prone to refuse, hedge, or (less intuitively) become
more or less sycophantic, beyond what the trained channel already does on
its own.

Same extraction, same battery, same heuristic scorers as the ablation
script -- only the intervention hook differs (add scale*unit_dir instead of
projecting the component out).
"""
import argparse
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import extract_directions, _decoder_layers
from scripts.check_suppression_channel_behavioral_sideeffects import (
    REFUSAL_PROMPTS, SYCOPHANCY_PROMPTS, HEDGING_PROMPTS,
    score_refusal, score_hedging, GEN_NEW_TOKENS,
)

SCALES = [0.05, 0.15]  # matches this project's established moderate/endpoint amplifier scales


def make_steering_hook(unit_dir, scale):
    def hook(module, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        hs_new = (hs.float() + scale * unit_dir).to(hs.dtype)
        return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
    return hook


def generate(model, tokenizer, prompt, device, ctx, steer_dirs=None, layers=None, scale=0.0):
    handles = []
    if steer_dirs is not None and scale:
        for l, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(make_steering_hook(steer_dirs[l], scale)))
    try:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with ctx, torch.no_grad():
            out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
        return tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        for h in handles:
            h.remove()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/locked_seed0/adapter")
    parser.add_argument("--output", default="activation_direction_results/suppression_channel_behavioral_sideeffects_steered.json")
    parser.add_argument("--only-sycophancy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)

    print(f"=== loading {args.adapter}, extracting its own suppression direction ===")
    model, tokenizer = load_model_with_adapter(args.adapter, dtype, device)
    n_layers = len(_decoder_layers(model))
    directions = extract_directions(model, tokenizer, cyber_holdout, device, ctx)
    unit_dirs = {l: (directions[l] / directions[l].norm()).to(device) for l in range(n_layers)}
    layers = _decoder_layers(model)

    out = {"adapter": args.adapter, "scales": {}}
    for scale in SCALES:
        print(f"\n########## scale={scale} ##########")
        scale_out = {"refusal": [], "sycophancy": [], "hedging": []}

        if not args.only_sycophancy:
            print("=== Refusal battery ===")
            for prompt in REFUSAL_PROMPTS:
                baseline = generate(model, tokenizer, prompt, device, ctx)
                steered = generate(model, tokenizer, prompt, device, ctx, steer_dirs=unit_dirs, layers=layers, scale=scale)
                row = {
                    "prompt": prompt,
                    "baseline_text": baseline, "baseline_refused": score_refusal(baseline),
                    "steered_text": steered, "steered_refused": score_refusal(steered),
                }
                scale_out["refusal"].append(row)
                print(f"[baseline refused={row['baseline_refused']}] [steered refused={row['steered_refused']}] {prompt[:60]}...")

        print("=== Sycophancy battery ===")
        for prompt, correct_hint in SYCOPHANCY_PROMPTS:
            baseline = generate(model, tokenizer, prompt, device, ctx)
            steered = generate(model, tokenizer, prompt, device, ctx, steer_dirs=unit_dirs, layers=layers, scale=scale)
            row = {
                "prompt": prompt, "correct_answer_hint": correct_hint,
                "baseline_text": baseline, "steered_text": steered,
            }
            scale_out["sycophancy"].append(row)
            print(f"[correct hint: {correct_hint}] {prompt[:60]}...")

        if not args.only_sycophancy:
            print("=== Hedging/verbosity battery ===")
            for prompt in HEDGING_PROMPTS:
                baseline = generate(model, tokenizer, prompt, device, ctx)
                steered = generate(model, tokenizer, prompt, device, ctx, steer_dirs=unit_dirs, layers=layers, scale=scale)
                row = {
                    "prompt": prompt,
                    "baseline_text": baseline, "baseline_score": score_hedging(baseline),
                    "steered_text": steered, "steered_score": score_hedging(steered),
                }
                scale_out["hedging"].append(row)
                print(f"[baseline hedges={row['baseline_score']['hedge_count']}] [steered hedges={row['steered_score']['hedge_count']}] {prompt[:60]}...")

        out["scales"][str(scale)] = scale_out

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
