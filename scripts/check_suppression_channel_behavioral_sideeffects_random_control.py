"""Random-direction control for check_suppression_channel_behavioral_sideeffects.py.

Found 2026-09-08: ablating label_flip's own suppression direction on its
locked model made it comply with a harmful request it otherwise refuses
(a fabricated claim about an ethnic group's intelligence) -- baseline text
is a clean, reasoned refusal; ablated text is a fluent, coherent, harmful
completion, not degenerate output. That's a real qualitative shift.

But this project already got burned once by treating a perturbation's
effect as specific to a real, trained direction without checking whether
ANY perturbation of the same magnitude does the same thing (the prompt-only
amplifier's fixed-letter-collapse issue -- see project history). The
behavioral pilot never had this control. This script is that control: same
battery, same ablation mechanism, but with a random per-layer unit vector
instead of the real extracted suppression direction. If the random control
ALSO makes the model comply with the ethnicity prompt (or shows similar
across-the-board shifts), the effect isn't specific to the trained
suppression channel -- it's generic disinhibition from perturbing the
residual stream at all. If the random control leaves refusal/sycophancy/
hedging close to baseline while the real direction doesn't, that's real
evidence of specificity.
"""
import argparse
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import _decoder_layers
from scripts.check_suppression_channel_behavioral_sideeffects import (
    REFUSAL_PROMPTS, SYCOPHANCY_PROMPTS, HEDGING_PROMPTS,
    generate, score_refusal, score_hedging,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="checkpoints/locked_seed0/adapter")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for the random direction itself.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--only-sycophancy", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or (
        "activation_direction_results/suppression_channel_behavioral_sideeffects_random_control"
        f"_rs{args.random_seed}.json"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    print(f"=== loading {args.adapter} ===")
    model, tokenizer = load_model_with_adapter(args.adapter, dtype, device)
    n_layers = len(_decoder_layers(model))
    hidden_size = model.config.hidden_size
    layers = _decoder_layers(model)

    rng = torch.Generator().manual_seed(args.random_seed)
    unit_dirs = {}
    for l in range(n_layers):
        v = torch.randn(hidden_size, generator=rng)
        unit_dirs[l] = (v / v.norm()).to(device)
    print(f"=== built {n_layers} random unit directions (random_seed={args.random_seed}) ===")

    out = {"adapter": args.adapter, "random_seed": args.random_seed, "refusal": [], "sycophancy": [], "hedging": []}

    if not args.only_sycophancy:
        print("\n=== Refusal battery ===")
        for prompt in REFUSAL_PROMPTS:
            baseline = generate(model, tokenizer, prompt, device, ctx)
            ablated = generate(model, tokenizer, prompt, device, ctx, ablate_dirs=unit_dirs, layers=layers)
            row = {
                "prompt": prompt,
                "baseline_text": baseline, "baseline_refused": score_refusal(baseline),
                "ablated_text": ablated, "ablated_refused": score_refusal(ablated),
            }
            out["refusal"].append(row)
            print(f"[baseline refused={row['baseline_refused']}] [random-ablated refused={row['ablated_refused']}] {prompt[:60]}...")

    print("\n=== Sycophancy battery ===")
    for prompt, correct_hint in SYCOPHANCY_PROMPTS:
        baseline = generate(model, tokenizer, prompt, device, ctx)
        ablated = generate(model, tokenizer, prompt, device, ctx, ablate_dirs=unit_dirs, layers=layers)
        row = {
            "prompt": prompt, "correct_answer_hint": correct_hint,
            "baseline_text": baseline, "ablated_text": ablated,
        }
        out["sycophancy"].append(row)
        print(f"[correct hint: {correct_hint}] {prompt[:60]}...")

    if not args.only_sycophancy:
        print("\n=== Hedging/verbosity battery ===")
        for prompt in HEDGING_PROMPTS:
            baseline = generate(model, tokenizer, prompt, device, ctx)
            ablated = generate(model, tokenizer, prompt, device, ctx, ablate_dirs=unit_dirs, layers=layers)
            row = {
                "prompt": prompt,
                "baseline_text": baseline, "baseline_score": score_hedging(baseline),
                "ablated_text": ablated, "ablated_score": score_hedging(ablated),
            }
            out["hedging"].append(row)
            print(f"[baseline hedges={row['baseline_score']['hedge_count']} words={row['baseline_score']['word_count']}] "
                  f"[random-ablated hedges={row['ablated_score']['hedge_count']} words={row['ablated_score']['word_count']}] {prompt[:60]}...")

    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {output}")
    print("\nCompare the ethnicity-prompt result (refusal[0]) against the real-direction run directly:")
    print("if random-ablated ALSO complies, the effect is generic disinhibition, not channel-specific.")


if __name__ == "__main__":
    main()
