"""Efficient scale search for the single-layer ("window rule") sufficiency
test: instead of re-running the full 32-layer sweep once per scale value
(wasteful -- 31 of 32 layers are already known to be irrelevant once the
recovery-peak layer is identified), fix ONE layer and sweep MANY scales in
a single model load.

Motivation (2026-09-08): sufficiency_layer_sweep.py at --single-layer-scale
4.0 found the recovery-peak layer already saturates to the exact degenerate
floor (0.1176) for both format (layer 14) and distillation (layer 11).
--single-layer-scale 1.0 found virtually no effect at all (accuracy stays
near baseline, ~0.92-0.96, across every layer). The real partial-collapse
zone -- if one exists -- is somewhere between 1.0 and 4.0, and manually
re-running the full 32-layer sweep at 2.0, 2.5, 3.0, ... one at a time
wastes most of the compute on layers we don't care about anymore. This
script sweeps just the recovery-peak layer across a dense scale grid, plus
a free-text coherence check at whichever scale lands in a genuinely partial
(not floor, not baseline) accuracy zone -- the same check that already
caught format/distillation's all-32-layer gibberish problem.

Default layer per mechanism is the recovery peak already found at scale=4.0
(distillation: layer 11, format: layer 14) -- override with --layer if a
different seed/run finds a different peak.
"""
import argparse
import json
from contextlib import nullcontext

import torch

from src.training.data_prep import build_control_examples, build_format_eval_examples, split_items
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import TRAIN_WRAPPER_PAIRS
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers

DEFAULT_RECOVERY_PEAK_LAYER = {"format": 14, "distillation": 11}
SCALES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
GEN_NEW_TOKENS = 40
N_COHERENCE_SAMPLES = 6

MECHANISM_ADAPTER_ROOTS = {
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), required=True)
    parser.add_argument("--seed", type=int, default=0, help="Applies to both the label_flip source and the target mechanism's checkpoint.")
    parser.add_argument("--layer", type=int, default=None, help="Defaults to this mechanism's known recovery-peak layer.")
    parser.add_argument("--label-flip-adapter", default=None, help="Overrides --seed for the label_flip source specifically.")
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
    output = args.output or f"activation_direction_results/sufficiency_single_layer_multiscale_{args.mechanism}_seed{args.seed}_layer{layer}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading label_flip model ({label_flip_adapter}) to extract its direction ===")
    lf_model, lf_tokenizer = load_model_with_adapter(label_flip_adapter, dtype, device)
    directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== loading target model ({target_adapter}), layer={layer} ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
    layers_module = _decoder_layers(model)

    deploy_only = build_examples(args.mechanism, holdout)
    print(f"=== {len(deploy_only)} deploy-framing examples for {args.mechanism} ===")

    out = {"mechanism": args.mechanism, "layer": layer, "n_examples": len(deploy_only), "sweep": {}}

    baseline = steer_and_evaluate(model, tokenizer, deploy_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): acc={baseline['acc_steered']:.4f}")
    out["sweep"]["scale_0.0"] = baseline

    for scale in SCALES:
        result = steer_and_evaluate(model, tokenizer, deploy_only, directions, [layer], scale, device, ctx)
        letters = result["predicted_letter_counts"]
        dominant = max(letters, key=letters.get)
        dom_frac = letters[dominant] / result["n"]
        print(f"scale={scale}: acc={result['acc_steered']:.4f}  dominant_letter={dominant} ({dom_frac:.0%})")
        out["sweep"][f"scale_{scale}"] = result

    # 2026-09-08 fix: the old logic only ever considered scales with >90% dominant-
    # letter concentration (already near-total collapse) and then picked among THOSE
    # by minimum accuracy -- i.e. it always selected something close to full
    # saturation, never the genuinely mixed "just starting to lean" zone, which is
    # exactly where coherent (if wrong) content-sensitive answers would most
    # plausibly show up if they exist at all. Now checks coherence at THREE points:
    # the smallest scale where the model first meaningfully leans toward one letter
    # (>50% dominance -- the mixed zone never tested before), a moderate-collapse
    # point (>90% dominance), and the largest scale tested (near/at the floor), so
    # partial-collapse coherence isn't skipped over on the way to full saturation.
    scale_stats = []
    for scale in SCALES:
        r = out["sweep"][f"scale_{scale}"]
        letters = r["predicted_letter_counts"]
        dom_frac = max(letters.values()) / r["n"]
        scale_stats.append((scale, r["acc_steered"], dom_frac))

    lean_candidates = [s for s, acc, df in scale_stats if df > 0.5]
    moderate_candidates = [s for s, acc, df in scale_stats if df > 0.9]
    coherence_scales = []
    if lean_candidates:
        coherence_scales.append(("first_lean", min(lean_candidates)))
    if moderate_candidates:
        coherence_scales.append(("moderate_collapse", min(moderate_candidates)))
    coherence_scales.append(("near_floor", SCALES[-1]))
    # De-dupe by scale value, keep first label seen.
    seen = set()
    deduped = []
    for label, s in coherence_scales:
        if s not in seen:
            seen.add(s)
            deduped.append((label, s))
    coherence_scales = deduped
    print(f"\n=== free-text coherence checks at scales {coherence_scales}, layer={layer} ===")

    def make_hook(vec):
        # 2026-09-08 fix: only perturb during the PREFILL pass (hs.shape[1] > 1,
        # the full prompt processed at once), not every autoregressive decode
        # step (hs.shape[1] == 1 under KV-cache). Applying the hook on every
        # decode step compounds the perturbation across all ~40-60 generated
        # tokens -- "scale=X" then means something completely different from
        # the single-forward-pass "scale=X" used for forced-choice accuracy,
        # and can break generation on its own regardless of whether the
        # direction/channel itself is coherent. Prefill-only makes this a
        # single, one-time intervention at the decision point, directly
        # comparable to the accuracy metric -- the model then continues
        # generating under its own unperturbed dynamics from there.
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
            handles.append(layers_module[layer].register_forward_hook(make_hook(directions[layer].to(device) * scale)))
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

    # 2026-09-08 fix: for format, deploy_only is grouped by item then by wrapper
    # pair (build_format_eval_examples iterates item -> pair), so a plain
    # [:N_COHERENCE_SAMPLES] slice grabbed N wrapper-variants of the SAME first
    # item, not N distinct items -- accuracy numbers were unaffected (computed
    # over the full set), but the qualitative coherence sample had far less
    # item diversity than it looked like. Dedupe by item_id instead.
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

    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
