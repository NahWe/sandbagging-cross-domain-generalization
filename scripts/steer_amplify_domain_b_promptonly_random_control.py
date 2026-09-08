"""Random-direction control for steer_amplify_domain_b_promptonly_direction.py
-- the specificity check that experiment never had.

Found 2026-09-06: at scale=0.15, label_flip's and format-trigger's steered
accuracy on bio matches the base rate of "A" being the correct answer
almost exactly (label_flip: 0.2467-0.2474 across 5 seeds; format-trigger:
0.2498) -- consistent with the model predicting "A" for every item
regardless of content, not a coherent lie. A single clean 'A' token under
free-text generation (already checked) cannot tell a genuine lie apart from
a fixed-letter collapse that happens to be a real word. Distillation is the
exception (accuracy 0.3048, meaningfully above the base rate, with
item-specific justification text).

This script runs the missing control: a random direction, norm-matched per
layer to the real prompt-only direction (same convention as
steer_format_random_control.py / steer_distillation_random_control.py),
injected the same way, same scales, same mechanisms. If the random control
ALSO collapses to the "A"-is-correct base rate at scale=0.15 (or to
whatever its own default letter turns out to be), that confirms the
fixed-letter-collapse reading for label_flip/format-trigger. If it does
NOT collapse the same way, that's real evidence the prompt-only direction
is doing something more specific than generic saturation, even for these
two mechanisms.

Needs the same checkpoints as steer_amplify_domain_b_promptonly_direction.py
-- not runnable until label_flip/format/distillation are retrained (see
docs/extension-plan-hackathon.md, Tier 0).
"""
import argparse
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import steer_and_evaluate, _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

SCALES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15]  # matches the real-direction sweep exactly

MECHANISM_ADAPTER_ROOTS = {
    "label_flip": "checkpoints",
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to steer.")
    parser.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), default="label_flip")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for the random direction itself.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = MECHANISM_ADAPTER_ROOTS[args.mechanism]
    adapter = f"{root}/locked_seed{args.seed}/adapter"
    suffix = "" if args.mechanism == "label_flip" else f"_{args.mechanism}"
    output = f"activation_direction_results/steer_amplify_domain_b_promptonly_random_control{suffix}_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print("=== extracting REAL prompt-only direction (for norm-matching only) ===")
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    real_directions = extract_prompt_only_directions(base_model, base_tokenizer, cyber_holdout, device, ctx)
    del base_model
    torch.cuda.empty_cache()

    rng = torch.Generator().manual_seed(args.random_seed)
    random_directions = {}
    for l, real_dir in real_directions.items():
        rand_vec = torch.randn(real_dir.shape, generator=rng)
        rand_vec = rand_vec / rand_vec.norm() * real_dir.norm()  # match norm exactly, same convention as steer_format_random_control.py
        random_directions[l] = rand_vec
    print(f"=== built {len(random_directions)} random directions, norm-matched to real per-layer norms ===")

    print(f"=== loading {adapter} ===")
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)
    n_layers = len(_decoder_layers(model))

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    bio_deploy_only = [ex for ex in bio_examples if ex.framing == "deploy"]

    print("=== unsteered baseline (scale=0.0), both framings ===")
    baseline_eval = steer_and_evaluate(model, tokenizer, bio_eval_only, random_directions, [], 0.0, device, ctx)
    baseline_deploy = steer_and_evaluate(model, tokenizer, bio_deploy_only, random_directions, [], 0.0, device, ctx)
    acc_deploy_ref = baseline_deploy["acc_steered"]
    print(f"  acc_eval={baseline_eval['acc_steered']:.4f}  acc_deploy={acc_deploy_ref:.4f}  "
          f"natural_gap={acc_deploy_ref - baseline_eval['acc_steered']:.4f}")

    sweep = []
    for scale in SCALES:
        r = steer_and_evaluate(
            model, tokenizer, bio_eval_only, random_directions, list(range(n_layers)), scale, device, ctx
        )
        widened_gap = acc_deploy_ref - r["acc_steered"]
        row = {
            "scale": scale, "acc_eval_steered": r["acc_steered"], "acc_deploy_ref": acc_deploy_ref,
            "widened_gap": widened_gap, "predicted_letter_counts": r.get("predicted_letter_counts"),
            "wrong_answer_letter_distribution": r["wrong_answer_letter_distribution"],
        }
        sweep.append(row)
        print(f"  scale={scale:.2f}  acc_eval_steered={r['acc_steered']:.4f}  widened_gap={widened_gap:+.4f}")

    out = {
        "adapter": adapter,
        "mechanism": args.mechanism,
        "seed": args.seed,
        "random_seed": args.random_seed,
        "source_direction": "random_norm_matched_to_prompt_only",
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")
    print("\nCompare scale_0.15's acc_eval_steered against the real-direction result in "
          "steer_amplify_domain_b_promptonly_direction*.json for the same mechanism/seed: "
          "if this random control ALSO lands near 0.2467 (bio's 'A'-is-correct base rate), "
          "that confirms fixed-letter collapse rather than a specific effect.")


if __name__ == "__main__":
    main()
