"""Same cross-mechanism sufficiency test as check_sufficiency_coherence.py,
but injecting the PROMPT-ONLY direction (base-model-derived, no adapter)
instead of label_flip's own trained direction.

Motivation (2026-09-08, user's own idea): the label_flip-sourced sufficiency
test (this project's original cross-mechanism design) never shows coherent
free-text collapse at any scale/layer, for either format or distillation --
every point tested is either near-baseline or degenerate gibberish. But the
prompt-only direction, injected into distillation's model on BIO (a
different task) in the amplifier experiment, DID produce genuinely coherent,
item-specific justifications -- unlike label_flip's own direction there.
If that's because prompt-only's direction is intrinsically "cleaner"
(extracted from genuine base-model instruction-following, not a trained
lock's own narrower channel) rather than something specific to bio, the
same should show up here: injecting prompt-only's direction instead of
label_flip's into format's/distillation's model, on domain A, might find a
genuinely coherent partial-collapse zone where label_flip's own direction
never did.

Design: identical to check_sufficiency_coherence.py (all 32 layers, scale
sweep, free-text coherence check at the largest scale still above the
degenerate floor) -- only the source direction changes.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, build_format_eval_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import steer_and_evaluate, _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

DIRECTION_CACHE = "activation_direction_results/prompt_only_direction_vectors.pt"

MECHANISMS = {
    "format": {
        "adapter_template": "checkpoints_format/locked_seed{seed}/adapter",
        "scales": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0],
    },
    "distillation": {
        "adapter_template": "checkpoints_distillation_v2/locked_seed{seed}/adapter",
        "scales": [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0],
    },
}
GEN_NEW_TOKENS = 40
N_COHERENCE_SAMPLES = 6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=list(MECHANISMS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction-cache", default=DIRECTION_CACHE)
    parser.add_argument(
        "--coherence-scale", type=float, default=None,
        help="Scale for the free-text coherence check. If omitted, picks the largest swept scale whose accuracy is still above ~0.2.",
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
    target_adapter = cfg["adapter_template"].format(seed=args.seed)
    output = args.output or f"activation_direction_results/sufficiency_coherence_promptonly_source_{args.mechanism}_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    if os.path.exists(args.direction_cache):
        print(f"=== loading cached prompt-only direction from {args.direction_cache} ===")
        directions = torch.load(args.direction_cache)
    else:
        print("=== cache not found, extracting prompt-only direction fresh from the base model ===")
        base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
        base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
            base_model.config.pad_token_id = base_tokenizer.pad_token_id
        directions = extract_prompt_only_directions(base_model, base_tokenizer, holdout, device, ctx)
        del base_model
        torch.cuda.empty_cache()

    print(f"=== loading target model ({target_adapter}) ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
    n_layers = len(_decoder_layers(model))

    deploy_only = build_examples(args.mechanism, holdout)
    print(f"=== {len(deploy_only)} deploy-framing examples for {args.mechanism} ===")

    out = {"mechanism": args.mechanism, "source_direction": "prompt_only", "n_examples": len(deploy_only), "accuracy_sweep": {}}

    baseline = steer_and_evaluate(model, tokenizer, deploy_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline}")
    out["accuracy_sweep"]["baseline"] = baseline

    for scale in cfg["scales"]:
        result = steer_and_evaluate(model, tokenizer, deploy_only, directions, list(range(n_layers)), scale, device, ctx)
        print(f"scale={scale}: {result}")
        out["accuracy_sweep"][f"scale_{scale}"] = result

    # 2026-09-08 fix: max(candidates) picked the most-saturated qualifying scale,
    # not a genuine partial-collapse point. Now checks three: smallest with >50%
    # letter dominance (mixed zone, never tested before), smallest with >90%
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
        # 2026-09-08 fix: prefill-only perturbation (hs.shape[1] > 1) -- see
        # sufficiency_single_layer_multiscale.py for the full rationale.
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

    # 2026-09-08 fix: dedupe by item_id -- see check_sufficiency_coherence.py.
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
