"""Random-direction control for sufficiency_single_layer_multiscale.py.

Motivation (2026-09-08, after adversarial review): the single-layer sweep
found repetition-loop degenerate output (e.g. "11. 11. 11. 11. ...") in a
minority of free-text samples at the higher end of the scale grid (mostly
3.0-4.0), for BOTH format and distillation. But no control has ever been run
for this specific test (single layer, additive injection) -- the existing
random-direction controls in this project are either all-32-layer (accuracy-
sweep only, no free text) or use unit vectors for a *projection-removal*
ablation, not additive injection at one layer, so neither answers "does ANY
perturbation of this magnitude at this layer cause the loop, regardless of
whether it's the real suppression-channel direction?"

This script: same single recovery-peak layer, same additive-injection
mechanism, same SCALES grid as sufficiency_single_layer_multiscale.py, but
replaces label_flip's extracted direction with a random vector matched to
the REAL direction's own norm at that layer (so "scale=X" means the same
perturbation magnitude in both runs -- a fair comparison). Coherence text is
checked at a fixed set of scales (2.5, 3.0, 3.5, 4.0) covering the exact
range where the real-direction run showed loops, rather than the accuracy-
based auto-picker (a random direction is not expected to move accuracy much,
so the auto-picker would rarely reach into this range on its own).

If the random control ALSO loops at a similar rate in this range: the
degenerate output is a generic large-single-layer-perturbation artifact, not
evidence the suppression channel itself is "unstable." If it does NOT loop:
supports the loop being specific to this particular direction's interaction
with the network at this layer/item, not just perturbation magnitude.
"""
import argparse
import json

import torch

from src.training.data_prep import build_control_examples, build_format_eval_examples, split_items
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers
from contextlib import nullcontext

DEFAULT_RECOVERY_PEAK_LAYER = {"format": 14, "distillation": 11}
SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
COHERENCE_SCALES = [2.5, 3.0, 3.5, 4.0]  # the range where the real direction showed loops
GEN_NEW_TOKENS = 40
N_COHERENCE_SAMPLES = 6

MECHANISM_ADAPTER_ROOTS = {
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for the random direction itself.")
    parser.add_argument("--label-flip-adapter", default=None, help="Used only to compute the real direction's norm for a fair-magnitude random vector.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_examples(mechanism, holdout):
    if mechanism == "format":
        examples = build_format_eval_examples(holdout, wrapper_pairs=TRAIN_WRAPPER_PAIRS)
    else:
        examples = build_control_examples(holdout)
    return [ex for ex in examples if ex.framing == "deploy"]


def main():
    args = parse_args()
    layer = args.layer if args.layer is not None else DEFAULT_RECOVERY_PEAK_LAYER[args.mechanism]
    label_flip_adapter = args.label_flip_adapter or f"checkpoints/locked_seed{args.seed}/adapter"
    target_adapter = f"{MECHANISM_ADAPTER_ROOTS[args.mechanism]}/locked_seed{args.seed}/adapter"
    output = args.output or (
        f"activation_direction_results/sufficiency_single_layer_random_control_"
        f"{args.mechanism}_seed{args.seed}_layer{layer}_rs{args.random_seed}.json"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading label_flip model ({label_flip_adapter}) only to get the real direction's norm at layer {layer} ===")
    lf_model, lf_tokenizer = load_model_with_adapter(label_flip_adapter, dtype, device)
    real_directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    real_norm = real_directions[layer].norm().item()
    del lf_model, real_directions
    torch.cuda.empty_cache()
    print(f"=== real direction norm at layer {layer}: {real_norm:.4f} -- building a random vector with this same norm ===")

    print(f"=== loading target model ({target_adapter}), layer={layer} ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
    layers_module = _decoder_layers(model)
    hidden_size = model.config.hidden_size

    rng = torch.Generator().manual_seed(args.random_seed)
    v = torch.randn(hidden_size, generator=rng)
    random_direction = (v / v.norm() * real_norm).to(device)

    deploy_only = build_examples(args.mechanism, holdout)
    print(f"=== {len(deploy_only)} deploy-framing examples for {args.mechanism} ===")

    out = {
        "mechanism": args.mechanism, "layer": layer, "random_seed": args.random_seed,
        "real_direction_norm": real_norm, "n_examples": len(deploy_only), "sweep": {},
    }

    baseline = steer_and_evaluate(model, tokenizer, deploy_only, {layer: random_direction}, [], 0.0, device, ctx)
    print(f"baseline (no steering): acc={baseline['acc_steered']:.4f}")
    out["sweep"]["scale_0.0"] = baseline

    for scale in SCALES:
        result = steer_and_evaluate(model, tokenizer, deploy_only, {layer: random_direction}, [layer], scale, device, ctx)
        letters = result["predicted_letter_counts"]
        dominant = max(letters, key=letters.get)
        dom_frac = letters[dominant] / result["n"]
        print(f"scale={scale}: acc={result['acc_steered']:.4f}  dominant_letter={dominant} ({dom_frac:.0%})")
        out["sweep"][f"scale_{scale}"] = result

    def make_hook(vec):
        def hook(module, inp, out_):
            hs = out_[0] if isinstance(out_, tuple) else out_
            if hs.shape[1] == 1:
                return out_
            hs_new = (hs.float() + vec).to(hs.dtype)
            return (hs_new,) + out_[1:] if isinstance(out_, tuple) else hs_new
        return hook

    def generate(scale, examples):
        handles = []
        if scale:
            handles.append(layers_module[layer].register_forward_hook(make_hook(random_direction * scale)))
        try:
            outs = []
            for ex in examples:
                enc = tokenizer(ex.prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
                with ctx, torch.no_grad():
                    out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
                text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                outs.append((ex.item_id, text))
        finally:
            for h in handles:
                h.remove()
        return outs

    seen_items = set()
    sample = []
    for ex in deploy_only:
        if ex.item_id in seen_items:
            continue
        seen_items.add(ex.item_id)
        sample.append(ex)
        if len(sample) >= N_COHERENCE_SAMPLES:
            break

    out["coherence_scales"] = COHERENCE_SCALES
    out["coherence_baseline"] = generate(0.0, sample)
    for item_id, text in out["coherence_baseline"]:
        print(f"[baseline] {item_id}: {text!r}")
    out["coherence_steered"] = {}
    for scale in COHERENCE_SCALES:
        steered = generate(scale, sample)
        out["coherence_steered"][str(scale)] = {"scale": scale, "generations": steered}
        for item_id, text in steered:
            print(f"[steered @ {scale}] {item_id}: {text!r}")

    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
