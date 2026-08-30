"""Same transfer-amplifier design as steer_amplify_domain_b.py, but
injecting the PROMPT-ONLY direction (extracted from the untouched base
model's genuinely-instructed sandbag-vs-deploy contrast) instead of the
target model's own trained eval-vs-deploy direction. Directly tests
whether white-box access to the "genuine lying" direction -- shown to be
close to orthogonal to label_flip's own direction (mean cosine ~0.03
across 32 layers, see extract_prompt_only_direction.py) -- is a MORE or
LESS potent lever for widening a trained model's out-of-domain (bio)
transfer gap than the model's own direction is.

Same SCALES, same sub-collapse range, same acc_deploy-held-fixed
methodology as steer_amplify_domain_b.py, so the two are directly
comparable seed-for-seed.
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

SCALES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15]


MECHANISM_ADAPTER_ROOTS = {
    "label_flip": "checkpoints",
    "format": "checkpoints_format",
    "distillation": "checkpoints_distillation_v2",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to steer.")
    parser.add_argument("--mechanism", choices=list(MECHANISM_ADAPTER_ROOTS), default="label_flip",
                         help="Which trained mechanism's model to steer -- tests whether the amplifier "
                              "effect is specific to label_flip or general across mechanisms, including "
                              "two (format, distillation) whose OWN direction never naturally transfers "
                              "to bio at all.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = MECHANISM_ADAPTER_ROOTS[args.mechanism]
    adapter = f"{root}/locked_seed{args.seed}/adapter"
    suffix = "" if args.mechanism == "label_flip" else f"_{args.mechanism}"
    output = f"activation_direction_results/steer_amplify_domain_b_promptonly_direction{suffix}_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print("=== extracting prompt-only direction from the untouched base model ===")
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    prompt_only_directions = extract_prompt_only_directions(base_model, base_tokenizer, cyber_holdout, device, ctx)
    del base_model
    torch.cuda.empty_cache()

    print(f"=== loading {adapter} ===")
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)
    n_layers = len(_decoder_layers(model))

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    bio_deploy_only = [ex for ex in bio_examples if ex.framing == "deploy"]

    print("=== unsteered baseline (scale=0.0), both framings ===")
    baseline_eval = steer_and_evaluate(model, tokenizer, bio_eval_only, prompt_only_directions, [], 0.0, device, ctx)
    baseline_deploy = steer_and_evaluate(model, tokenizer, bio_deploy_only, prompt_only_directions, [], 0.0, device, ctx)
    acc_deploy_ref = baseline_deploy["acc_steered"]
    print(f"  acc_eval={baseline_eval['acc_steered']:.4f}  acc_deploy={acc_deploy_ref:.4f}  "
          f"natural_gap={acc_deploy_ref - baseline_eval['acc_steered']:.4f}")

    sweep = []
    for scale in SCALES:
        r = steer_and_evaluate(
            model, tokenizer, bio_eval_only, prompt_only_directions, list(range(n_layers)), scale, device, ctx
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
        "mechanism": args.mechanism,
        "seed": args.seed,
        "source_direction": "prompt_only",
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
