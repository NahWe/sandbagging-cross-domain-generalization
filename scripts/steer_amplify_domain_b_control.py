"""CONTROL CHECK for steer_amplify_domain_b.py: does label_flip's own
suppression direction widen the eval-vs-deploy gap on domain B even when
injected into a CONTROL model (never trained with any lock at all) -- or
does the amplification effect only show up when a real (if weak) trained
lock is already present to potentiate?

Essential to interpret steer_amplify_domain_b.py's headline result (natural
bio gap 0.0636 -> widened to 0.2529 at scale=0.15) correctly: if the SAME
widening happens on the control model too, the effect is generic
inference-time suppression-induction that needs no training at all (still
interesting, but a different claim -- "steering alone can manufacture a
gap," not "steering amplifies an existing trained mechanism"). If the
control model stays far closer to its natural near-zero gap under the same
steering, that confirms the effect is specifically potentiating something
the trained lock already installed.

Direction extracted from checkpoints/locked_seed0 (the ONLY model with a
real eval-vs-deploy behavioral difference to extract a meaningful direction
from -- control was never trained to distinguish framings, so "its own
direction" wouldn't be meaningful the same way), then applied to
checkpoints/control_seed0's own domain-B eval-framing forward pass. Same
scale sweep as the locked-model version for direct comparison.
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
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N>/control_seed<N> pair to use.")
    return parser.parse_args()


def main():
    args = parse_args()
    direction_source_adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    target_adapter = f"checkpoints/control_seed{args.seed}/adapter"
    output = f"activation_direction_results/steer_amplify_domain_b_control_check_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== extracting direction from {direction_source_adapter} ===")
    src_model, src_tokenizer = load_model_with_adapter(direction_source_adapter, dtype, device)
    directions = extract_directions(src_model, src_tokenizer, cyber_holdout, device, ctx)
    del src_model
    torch.cuda.empty_cache()

    print(f"=== loading target (control) model {target_adapter} ===")
    model, tokenizer = load_model_with_adapter(target_adapter, dtype, device)
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
        "direction_source_adapter": direction_source_adapter,
        "target_adapter": target_adapter,
        "seed": args.seed,
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
