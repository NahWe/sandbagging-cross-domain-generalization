"""Per-layer sufficiency sweep: the "window rule" half that was missing.

activation_direction.py's `--layers sweep_individual` already does this for
NECESSITY (ablating label_flip's own direction from its own model, one layer
at a time) -- built, never run. This script is the SUFFICIENCY analog:
inject label_flip's direction into a DIFFERENT mechanism's model (format or
distillation), one layer at a time, instead of all 32 layers at once like
steer_format_with_label_flip_direction.py / steer_distillation_with_label_flip_direction.py
do today.

Why this matters for the specificity gap flagged in the blog (Future Work
item 13): today's cross-mechanism sufficiency test steers all 32 layers at
once, at a scale strong enough to saturate the model to the same degenerate
floor a random direction also reaches (2/10 control runs tie it exactly at
scale=1.0). Tan/Le/Williams-King's "window rule" finds that a single,
correctly-chosen layer can recover as much as steering every layer, at much
lower magnitude per layer -- if the same holds here, targeting just the
write-then-commit window instead of all 32 layers at scale=1.0 might
separate the real direction from a random-direction control far more
cleanly, without needing the saturating scale that currently erases the
distinction.

Cheap proxy for "write peak" without implementing Tan et al.'s full
answer-cluster-ratio machinery: label_flip's own per-layer direction norm
(already computed as a side effect of extract_directions, and separately by
cache_mechanism_directions.py) -- the layer where the eval-vs-deploy
contrast is largest in the SOURCE model is the natural analog of their
"last sandbagging write" peak. Reported alongside the sweep so the two can
be compared directly (their Fig. 3a correlates write-peak layer against
recovery-peak layer, r=0.84, across 33 runs -- we can build the same kind of
scatter cheaply here, just for 2 target mechanisms x 5 seeds instead of 4
lock types x 3 models).

This is NOT the full "reference-grafting" operator (Tan/Le's Eq. 1: set the
axis coordinate TO the honest reference value, not just add a scaled raw
vector). It reuses this project's existing additive-steering hook
(steer_and_evaluate), which is closer to classic activation addition than
their more precise patch. If the per-layer sweep here doesn't cleanly
localize a window, implementing the exact affine "set-to-reference" operator
is the natural next refinement -- flagged, not attempted here, to keep this
script's scope matched to what's actually missing today.
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

TARGETS = {
    "format": {
        "adapter_template": "checkpoints_format/locked_seed{seed}/adapter",
        "examples_fn": "format",
        # All-layers scale where the existing test already saturates to the
        # degenerate floor -- kept fixed here so single-layer results are
        # directly comparable to the already-published all-layers number.
        "saturating_scale": 1.0,
    },
    "distillation": {
        "adapter_template": "checkpoints_distillation_v2/locked_seed{seed}/adapter",
        "examples_fn": "control",
        "saturating_scale": 1.0,
    },
    # 2026-09-08: WITHIN-model control for the window-rule mismatch (write-peak
    # layer 31 vs. cross-mechanism recovery-peaks at 14/11). Tan et al.'s r=0.84
    # write-peak/recovery-peak correlation is plausibly a property of steering a
    # direction back into the SAME model it came from -- our format/distillation
    # tests are inherently cross-model, which may just not be the setting their
    # correlation describes. Injecting label_flip's own direction into label_flip's
    # own model isolates that: if ITS OWN recovery-peak also lands far from its
    # own write-peak (31), the mismatch isn't about cross-model transfer specifically.
    "label_flip": {
        "adapter_template": "checkpoints/locked_seed{seed}/adapter",
        "examples_fn": "control",
        "saturating_scale": 1.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanism", choices=list(TARGETS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-flip-adapter", default=None, help="Defaults to checkpoints/locked_seed<seed>/adapter")
    parser.add_argument(
        "--single-layer-scale",
        type=float,
        default=4.0,
        help="Scale for the single-layer sweep. Higher than the all-layers scale by design: "
        "concentrating the same total push into one layer instead of 32 needs a bigger "
        "per-layer coefficient to have a comparable effect -- tune per mechanism if the "
        "sweep shows no layer moving accuracy at all.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def build_examples(mechanism, holdout):
    if TARGETS[mechanism]["examples_fn"] == "format":
        examples = build_format_eval_examples(holdout, wrapper_pairs=TRAIN_WRAPPER_PAIRS)
    else:
        examples = build_control_examples(holdout)
    return [ex for ex in examples if ex.framing == "deploy"]


def main():
    args = parse_args()
    cfg = TARGETS[args.mechanism]
    seed = args.seed
    lf_adapter = args.label_flip_adapter or f"checkpoints/locked_seed{seed}/adapter"
    target_adapter = cfg["adapter_template"].format(seed=seed)
    output = args.output or f"activation_direction_results/sufficiency_sweep_{args.mechanism}_seed{seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== loading label_flip model ({lf_adapter}) to extract direction ===")
    lf_model, lf_tokenizer = load_model_with_adapter(lf_adapter, dtype, device)
    directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    write_peak_layer = max(directions, key=lambda l: directions[l].norm().item())
    write_norms = {l: directions[l].norm().item() for l in sorted(directions)}
    print(f"=== write-peak proxy (largest direction norm): layer {write_peak_layer} ===")
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== loading target model ({target_adapter}) ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
    n_layers = len(_decoder_layers(model))

    deploy_only = build_examples(args.mechanism, holdout)
    print(f"=== {len(deploy_only)} deploy-framing examples for {args.mechanism}, seed {seed} ===")

    out = {
        "mechanism": args.mechanism,
        "seed": seed,
        "n_examples": len(deploy_only),
        "write_peak_layer": write_peak_layer,
        "write_norms_by_layer": write_norms,
        "single_layer_scale": args.single_layer_scale,
        "per_layer": {},
    }

    baseline = steer_and_evaluate(model, tokenizer, deploy_only, directions, [], 0.0, device, ctx)
    print(f"baseline (no steering): {baseline['acc_steered']:.4f}")
    out["baseline"] = baseline

    all_layers_result = steer_and_evaluate(
        model, tokenizer, deploy_only, directions, list(range(n_layers)), cfg["saturating_scale"], device, ctx
    )
    print(f"all layers @ scale={cfg['saturating_scale']} (today's existing test, for reference): {all_layers_result['acc_steered']:.4f}")
    out["all_layers_reference"] = all_layers_result

    recovery_peak_layer = None
    lowest_acc = baseline["acc_steered"]
    for l in range(n_layers):
        result = steer_and_evaluate(model, tokenizer, deploy_only, directions, [l], args.single_layer_scale, device, ctx)
        out["per_layer"][l] = result
        print(f"  layer {l}: acc={result['acc_steered']:.4f}  (write_norm={write_norms[l]:.3f})")
        if result["acc_steered"] < lowest_acc:
            lowest_acc = result["acc_steered"]
            recovery_peak_layer = l

    out["recovery_peak_layer"] = recovery_peak_layer
    out["recovery_peak_accuracy"] = lowest_acc

    # 2026-09-08 fix (adversarial review): recovery_peak_layer is picked with a
    # strict "<" against a running minimum, so if many layers TIE at the same
    # floor accuracy (plausible for a self-injection target, which has no
    # cross-model attenuation and may saturate at this scale far more broadly
    # than the cross-mechanism targets did), the "peak" is really just the
    # FIRST layer scanned at that floor -- an artifact of loop order, not a
    # genuine localized peak. Report how many layers are within epsilon of the
    # minimum so this can't be silently trusted.
    eps = 0.01
    tied_layers = [l for l in range(n_layers) if out["per_layer"][l]["acc_steered"] <= lowest_acc + eps]
    out["layers_tied_at_floor"] = tied_layers
    out["recovery_peak_reliable"] = len(tied_layers) <= max(1, n_layers // 8)
    print(f"\n=== recovery peak (most collapse from a single layer): layer {recovery_peak_layer}, acc={lowest_acc:.4f} ===")
    print(f"=== write peak (largest source-model direction norm): layer {write_peak_layer} ===")
    if not out["recovery_peak_reliable"]:
        print(f"!!! WARNING: {len(tied_layers)}/{n_layers} layers tied within {eps} of the floor accuracy -- "
              f"recovery_peak_layer={recovery_peak_layer} is likely just the first layer scanned at a flat floor, "
              f"NOT a genuine localized peak. Re-run at a lower --single-layer-scale before trusting this number.")
    print("Compare these two across seeds/mechanisms the way Tan et al. correlate write-peak vs. recovery-peak (their Fig. 3a, r=0.84).")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
