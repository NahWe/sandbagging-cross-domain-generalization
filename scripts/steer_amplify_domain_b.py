"""Steering as an inference-time TRANSFER AMPLIFIER, not just an in-domain
lie-inducer (which is all prior steering scripts tested). Direct magnitude
test for the project's actual goal (can bio/chem transfer be pushed
stronger, not just detected): does adding the model's OWN eval-vs-deploy
direction to its OWN domain-B eval-framing forward pass, at increasing
sub-collapse scale, widen the already-trained gap beyond what training
alone achieves?

Uses checkpoints/locked_seed0 (the original single-domain-synthetic-cyber
run, bio Δ=+0.0536, the best-characterized checkpoint in the project) --
extracts its own direction (same method as every other activation-direction
script) then steers ONLY domain-B eval-framing examples (not deploy) at each
scale, so any accuracy movement is interpretable as "the eval-framing
suppression got stronger," not generic bidirectional disruption. Baseline
(scale=0) acc_deploy is measured once, unsteered, and held fixed as the
reference point for computing the widened gap at each scale.

Known collapse floor from prior steering work (format/distillation targets):
scale 0.1-0.2 is roughly where forced-choice accuracy collapses to a
degenerate ~0.12 (all-same-letter) floor -- this sweep stays inside and
just below that range (0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15) to find
whether there's a real widening region before collapse, not just "trained
gap" then "degenerate floor" with nothing informative in between.
"""
import argparse
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers

SCALES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to use.")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    output = f"activation_direction_results/steer_amplify_domain_b_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== loading {adapter} ===")
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)

    print("=== extracting this model's own direction (from domain-A/cyber holdout) ===")
    directions = extract_directions(model, tokenizer, cyber_holdout, device, ctx)
    n_layers = len(_decoder_layers(model))

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    bio_deploy_only = [ex for ex in bio_examples if ex.framing == "deploy"]

    print("=== unsteered baseline (scale=0.0), both framings ===")
    baseline_eval = steer_and_evaluate(model, tokenizer, bio_eval_only, directions, [], 0.0, device, ctx)
    baseline_deploy = steer_and_evaluate(model, tokenizer, bio_deploy_only, directions, [], 0.0, device, ctx)
    acc_deploy_ref = baseline_deploy["acc_steered"]
    print(f"  acc_eval={baseline_eval['acc_steered']:.4f}  acc_deploy={acc_deploy_ref:.4f}  "
          f"natural_gap={acc_deploy_ref - baseline_eval['acc_steered']:.4f}")

    sweep = []
    for scale in SCALES:
        r = steer_and_evaluate(
            model, tokenizer, bio_eval_only, directions, list(range(n_layers)), scale, device, ctx
        )
        widened_gap = acc_deploy_ref - r["acc_steered"]
        row = {
            "scale": scale, "acc_eval_steered": r["acc_steered"], "acc_deploy_ref": acc_deploy_ref,
            "widened_gap": widened_gap, "wrong_answer_letter_distribution": r["wrong_answer_letter_distribution"],
        }
        sweep.append(row)
        print(f"  scale={scale:.2f}  acc_eval_steered={r['acc_steered']:.4f}  widened_gap={widened_gap:+.4f}")

    out = {
        "adapter": adapter,
        "seed": args.seed,
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
