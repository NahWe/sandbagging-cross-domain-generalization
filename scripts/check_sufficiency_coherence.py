"""Free-text coherence + moderate-scale accuracy check for the cross-mechanism
sufficiency test (label_flip's direction injected into format-trigger's or
distillation's model, domain A, own deploy-framing).

Why this exists: steer_format_with_label_flip_direction.py was only ever run
at scales 1,2,4,8 -- all fully saturated to the same "predict one letter for
every item" floor that a random-direction control also reaches at scale=1.0
(confirmed by inspecting activation_direction_results/steer_format_with_label_flip.json
and steer_format_random_control*.json directly). steer_distillation_with_label_flip_direction.py
already has sub-saturating scales (0.1-0.75) and shows real separation from
its own random control there (scale=0.2: real 0.319 vs. control 0.933; scale=0.5:
real 0.160 vs. control 0.689) -- but that separation was never checked for
coherence (is the model producing a plausible wrong answer, or already
degenerate gibberish at that scale?), and was never surfaced in the blog.

This script does two things neither steer_*_with_label_flip_direction.py nor
check_promptonly_amplifier_coherence.py does for THIS specific test:
  1. Sweeps format's sufficiency test at sub-saturating scales (mirroring what
     distillation already has), reporting accuracy at each scale.
  2. Decodes free text (not just the forced-choice letter) at a moderate,
     partial-collapse scale for BOTH mechanisms, to confirm whether the
     induced wrongness is a coherent alternative answer or generic breakdown.

Run once GPU access is back. See docs/runpod-specificity-sweep.local.md for
the full task writeup and how results should update the blog.
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

MECHANISMS = {
    "format": {
        "adapter_template": "checkpoints_format/locked_seed{seed}/adapter",
        "scales": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0],
    },
    "distillation": {
        "adapter_template": "checkpoints_distillation_v2/locked_seed{seed}/adapter",
        "scales": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],  # already run once; rerun for the coherence pass alongside format
    },
}
GEN_NEW_TOKENS = 40
N_COHERENCE_SAMPLES = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=list(MECHANISMS), required=True)
    parser.add_argument("--seed", type=int, default=0, help="Applies to both the label_flip source and the target mechanism's checkpoint.")
    parser.add_argument("--label-flip-adapter", default=None, help="Overrides --seed for the label_flip source specifically.")
    parser.add_argument(
        "--coherence-scale",
        type=float,
        default=None,
        help="Scale at which to run the free-text coherence check (partial collapse, not floor). "
        "If omitted, picks the largest swept scale whose accuracy is still above the degenerate floor.",
    )
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
    cfg = MECHANISMS[args.mechanism]
    label_flip_adapter = args.label_flip_adapter or f"checkpoints/locked_seed{args.seed}/adapter"
    target_adapter = cfg["adapter_template"].format(seed=args.seed)
    output = args.output or f"activation_direction_results/sufficiency_coherence_{args.mechanism}_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading label_flip model ({label_flip_adapter}) to extract directions ===")
    lf_model, lf_tokenizer = load_model_with_adapter(label_flip_adapter, dtype, device)
    directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== loading target model ({target_adapter}) ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
    n_layers = len(_decoder_layers(model))

    deploy_only = build_examples(args.mechanism, holdout)
    print(f"=== {len(deploy_only)} deploy-framing examples for {args.mechanism} ===")

    out = {"mechanism": args.mechanism, "n_examples": len(deploy_only), "accuracy_sweep": {}}

    baseline = steer_and_evaluate(model, tokenizer, deploy_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline}")
    out["accuracy_sweep"]["baseline"] = baseline

    for scale in cfg["scales"]:
        result = steer_and_evaluate(model, tokenizer, deploy_only, directions, list(range(n_layers)), scale, device, ctx)
        print(f"scale={scale}: {result}")
        out["accuracy_sweep"][f"scale_{scale}"] = result

    # 2026-09-08 fix: the old picker took max(candidates) -- once several scales
    # all clear a low accuracy bar, that's the MOST saturated one, not a genuine
    # partial-collapse point. Now checks three scales: smallest with >50% letter
    # dominance (the mixed zone never tested before), smallest with >90%
    # dominance, and the largest scale swept.
    if args.coherence_scale is not None:
        coherence_scales = [("manual", args.coherence_scale)]
    else:
        scale_stats = []
        for s in cfg["scales"]:
            r = out["accuracy_sweep"][f"scale_{s}"]
            letters = r["predicted_letter_counts"]
            dom_frac = max(letters.values()) / r["n"] if r["n"] else 0
            scale_stats.append((s, r["acc_steered"], dom_frac))
        lean_candidates = [s for s, acc, df in scale_stats if df > 0.5]
        moderate_candidates = [s for s, acc, df in scale_stats if df > 0.9]
        coherence_scales = []
        if lean_candidates:
            coherence_scales.append(("first_lean", min(lean_candidates)))
        if moderate_candidates:
            coherence_scales.append(("moderate_collapse", min(moderate_candidates)))
        coherence_scales.append(("near_floor", cfg["scales"][-1]))
        seen = set()
        deduped = []
        for label, s in coherence_scales:
            if s not in seen:
                seen.add(s)
                deduped.append((label, s))
        coherence_scales = deduped
    print(f"=== free-text coherence checks at scales {coherence_scales} ===")

    layers = _decoder_layers(model)

    def make_hook(vec):
        # 2026-09-08 fix: prefill-only perturbation (hs.shape[1] > 1) -- a hook left
        # on for every autoregressive decode step compounds the perturbation far
        # beyond what "scale=X" means for the single-forward-pass accuracy metric.
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] == 1:
                return out
            hs_new = (hs.float() + vec).to(hs.dtype)
            return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
        return hook

    def generate(scale, examples):
        handles = []
        if scale:
            for l, layer in enumerate(layers):
                handles.append(layer.register_forward_hook(make_hook(directions[l].to(device) * scale)))
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

    # 2026-09-08 fix: dedupe by item_id -- for format, deploy_only is grouped by
    # item then wrapper pair, so a raw slice grabbed N wrapper-variants of the
    # SAME first item, not N distinct items (see sufficiency_single_layer_multiscale.py).
    seen_items = set()
    sample = []
    for ex in deploy_only:
        if ex.item_id in seen_items:
            continue
        seen_items.add(ex.item_id)
        sample.append(ex)
        if len(sample) >= N_COHERENCE_SAMPLES:
            break
    out["coherence_scales"] = coherence_scales
    out["coherence_baseline"] = generate(0.0, sample)
    for item_id, text in out["coherence_baseline"]:
        print(f"[baseline] {item_id}: {text!r}")
    out["coherence_steered"] = {}
    for label, scale in coherence_scales:
        steered = generate(scale, sample)
        out["coherence_steered"][label] = {"scale": scale, "generations": steered}
        for item_id, text in steered:
            print(f"[steered @ {scale} ({label})] {item_id}: {text!r}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
