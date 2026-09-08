"""Random-direction control for steer_amplify_domain_b_smart_combined_direction.py
-- the specificity check that experiment never had either.

Found 2026-09-06: unlike label_flip's and format-trigger's prompt-only
amplifier results (which collapse to EXACTLY bio's "A"-is-correct base rate,
0.2467, with a 100%-"A" wrong-answer distribution -- see
steer_amplify_domain_b_promptonly_random_control.py), the smart-combined
(orthogonalized label_flip + prompt-only) amplifier at scale=0.15, seed0,
lands at accuracy 0.26394 with a 98.8%-"A" (not 100%) wrong-answer
distribution -- close to, but measurably above, the base rate and short of
full single-letter collapse. This is a THIRD case, distinct from both the
clean fixed-letter-collapse (label_flip/format) and the clearly-genuine
content engagement (distillation): partial collapse, not yet resolved
either way.

This script runs the missing control: a random per-layer unit vector
(matching the smart-combined direction's own convention -- each layer's
combined direction is renormalized to unit length before use, see
`smart_combine()` in steer_amplify_domain_b_smart_combined_direction.py),
injected the same way, same scales. If the random control ALSO lands near
0.2467 with a near-total single-letter wrong-answer distribution, that
argues the smart-combined direction is not meaningfully more specific than
noise at this scale, same as label_flip/format's own results suggest. If
the random control collapses further/differently than the real combined
direction did (further from 0.2467, less single-letter-dominated), that's
evidence the combined direction is doing something more specific after all,
even if not full genuine coherence.

Needs the same checkpoints as steer_amplify_domain_b_smart_combined_direction.py
-- not runnable until label_flip is retrained (see docs/extension-plan-hackathon.md,
Tier 0). No new adapter needed beyond label_flip's own (prompt-only direction
is base-model-derived and already cached/survived the volume deletion).
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

SCALES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15]  # matches the real smart-combined sweep exactly


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to steer.")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for the random direction itself.")
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    output = f"activation_direction_results/steer_amplify_domain_b_smart_combined_random_control_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== loading {adapter} (only needed to know layer count/hidden size) ===")
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)
    n_layers = len(_decoder_layers(model))
    hidden_size = model.config.hidden_size

    rng = torch.Generator().manual_seed(args.random_seed)
    random_directions = {}
    for l in range(n_layers):
        rand_vec = torch.randn(hidden_size, generator=rng)
        random_directions[l] = rand_vec / rand_vec.norm()  # unit length, matching smart_combine()'s own convention
    print(f"=== built {len(random_directions)} random unit directions (no norm-matching needed -- "
          f"the real smart-combined direction is itself unit-normalized per layer) ===")

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
            "widened_gap": widened_gap, "wrong_answer_letter_distribution": r["wrong_answer_letter_distribution"],
        }
        sweep.append(row)
        print(f"  scale={scale:.2f}  acc_eval_steered={r['acc_steered']:.4f}  widened_gap={widened_gap:+.4f}")

    out = {
        "adapter": adapter,
        "seed": args.seed,
        "random_seed": args.random_seed,
        "source_direction": "random_unit_per_layer_matching_smart_combined_convention",
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")
    print("\nCompare scale_0.15's acc_eval_steered/wrong_answer_letter_distribution against the real "
          "smart-combined result in steer_amplify_domain_b_smart_combined_seed0.json: "
          "real result there is 0.26394 with 98.8% 'A' (not 100%, unlike label_flip/format-trigger's "
          "own prompt-only-amplifier results, which hit 0.24666 with 100% 'A' exactly).")


if __name__ == "__main__":
    main()
