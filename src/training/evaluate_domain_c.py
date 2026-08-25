"""Entry point that actually needs a GPU: loads a trained adapter checkpoint
(written by lora_run.py) and evaluates it, with zero further training, on
domain C (real WMDP-chemistry) under both eval- and deployment-framing --
a second, independent held-out domain alongside domain B (WMDP-bio),
mirroring evaluate_domain_b.py exactly except for which real WMDP category
it reads.

Reuses evaluate_framing/_forced_choice_bias/HF_PATH from lora_run.py (same
forced A/B/C/D decoding, so domain-A/B/C numbers stay comparable) and
build_control_examples from data_prep.py -- deliberately not
build_locked_examples, since that one manufactures an artificial "wrong
answer" training target for the eval-framing side. Here both framings
should target the real answer; we're measuring what the model predicts,
not building a training label.

NOT unit-tested here on purpose, same reasoning as evaluate_domain_b.py:
this needs a real model and a real trained adapter to verify.
scripts/aggregate_domain_c.py, which consumes this script's output, IS
unit-tested with fabricated results.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_c import fetch_raw_rows, load_domain_c
from src.training.data_prep import PASSWORD, build_control_examples, build_format_eval_examples
from src.training.format_triggers import HELDOUT_WRAPPER_PAIRS
from src.training.lora_run import HF_PATH, evaluate_framing, model_kwargs, select_dtype
from src.training.weak_reference import DEFAULT_WEAK_MODEL, compute_weak_distributions, weak_argmax_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter-dir",
        type=str,
        required=True,
        help="Path to a trained adapter, e.g. checkpoints/locked_seed0/adapter",
    )
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    # Must match the --trigger the adapter at --adapter-dir was actually
    # trained with (lora_run.py).
    parser.add_argument("--trigger", choices=["password", "format"], default="password")
    # Doesn't change how domain-C is graded (build_control_examples already
    # grades every mechanism against the real answer) -- only used to
    # additionally compute the weak-agreement diagnostic. Must match the
    # --weak-model the adapter at --adapter-dir was trained with, if
    # --mechanism distillation.
    parser.add_argument("--mechanism", choices=["label_flip", "distillation"], default="label_flip")
    parser.add_argument("--weak-model", type=str, default=DEFAULT_WEAK_MODEL)
    return parser.parse_args()


def load_model_with_adapter(adapter_dir: str, dtype: torch.dtype, device: str):
    """Loads the base model and attaches an already-trained adapter (as
    opposed to lora_run.py's load_model, which initializes a *fresh* LoRA
    config for training). Mirrors evaluate_domain_b.py's function of the
    same name exactly."""
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(model, adapter_dir)
    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Same resumability as evaluate_domain_b.py: rerunning
    # run_domain_c_eval.sh skips adapters already evaluated instead of
    # redoing them.
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping domain_c condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    rows = fetch_raw_rows()
    items = load_domain_c(rows)
    # Both framings target the real answer in either case -- we're measuring
    # what the model predicts, not building a training label.
    if args.trigger == "format":
        eval_examples = build_format_eval_examples(items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
    else:
        eval_examples = build_control_examples(items)

    weak_argmax = None
    if args.mechanism == "distillation":
        weak_distributions = compute_weak_distributions(items, args.weak_model)
        weak_argmax = weak_argmax_map(weak_distributions)

    model, tokenizer = load_model_with_adapter(args.adapter_dir, dtype, device)
    results = evaluate_framing(model, tokenizer, eval_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "domain": "domain_c",
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "adapter_dir": args.adapter_dir,
                "trigger": args.trigger,
                "mechanism": args.mechanism,
                "password": PASSWORD if args.trigger == "password" else None,
                "weak_model": args.weak_model if args.mechanism == "distillation" else None,
                "weak_argmax": weak_argmax,
                "n_items": len(items),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"Wrote {results_path} ({len(results)} eval results)")


if __name__ == "__main__":
    main()
