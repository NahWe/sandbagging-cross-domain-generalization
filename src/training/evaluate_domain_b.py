"""Entry point that actually needs a GPU: loads a trained adapter checkpoint
(written by lora_run.py) and evaluates it, with zero further training, on
domain B (real WMDP-bio) under both eval- and deployment-framing -- the
piece that actually answers this project's research question (does the
domain-A sandbagging gap transfer to a structurally distinct domain?).

Reuses evaluate_framing/_forced_choice_token_ids/HF_PATH from lora_run.py (same
forced A/B/C/D decoding, so domain-A and domain-B numbers stay comparable)
and build_control_examples from data_prep.py -- deliberately not
build_locked_examples, since that one manufactures an artificial "wrong
answer" training target for the eval-framing side. Here both framings
should target the real answer; we're measuring what the model predicts,
not building a training label.

NOT unit-tested here on purpose, same reasoning as lora_run.py (see
docs/design.md's "Status"): this needs a real model and a real trained
adapter to verify. scripts/aggregate_domain_b.py, which consumes this
script's output, IS unit-tested with fabricated results.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_b import fetch_raw_rows, load_domain_b
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
    # trained with (lora_run.py) -- "format" uses HELDOUT_WRAPPER_PAIRS
    # (docs/extension-plan-format-triggered-lock.md's cross-domain read: only
    # wrapper phrasings never paired with domain-B content during training).
    parser.add_argument("--trigger", choices=["password", "format"], default="password")
    # Doesn't change how domain-B is graded (build_control_examples already
    # grades every mechanism against the real answer, see module docstring)
    # -- only used to additionally compute the weak-agreement diagnostic
    # (docs/extension-plan-auditing-games-induction.md's "concrete way to
    # partially disambiguate memorization from a real disposition"). Must
    # match the --weak-model the adapter at --adapter-dir was trained with,
    # if --mechanism distillation.
    parser.add_argument("--mechanism", choices=["label_flip", "distillation"], default="label_flip")
    parser.add_argument("--weak-model", type=str, default=DEFAULT_WEAK_MODEL)
    return parser.parse_args()


def load_model_with_adapter(adapter_dir: str, dtype: torch.dtype, device: str):
    """Loads the base model and attaches an already-trained adapter (as
    opposed to lora_run.py's load_model, which initializes a *fresh* LoRA
    config for training). The pad-token handling mirrors lora_run.py's
    load_model exactly (reuse eos_token, no resize) so this matches
    whatever vocab/embedding shape the adapter was actually trained
    against. Shares model_kwargs with lora_run.py (sdpa attention, 4-bit
    quantization when on GPU) so eval doesn't reintroduce the flash-attn
    compile-step dependency or run at a different precision than training
    did -- see model_kwargs's docstring."""
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

    # Same resumability as lora_run.py: rerunning run_domain_b_eval.sh skips
    # adapters already evaluated instead of redoing them.
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping domain_b condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    rows = fetch_raw_rows()
    items = load_domain_b(rows)
    # Both framings target the real answer in either case -- we're measuring
    # what the model predicts, not building a training label.
    if args.trigger == "format":
        eval_examples = build_format_eval_examples(items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
    else:
        # build_control_examples, not build_locked_examples, gives exactly
        # that (see module docstring).
        eval_examples = build_control_examples(items)

    # Computed before the adapter loads, same reasoning as lora_run.py's
    # domain-A precompute: a small forward-pass-only model, discarded
    # before Mistral is resident. Doesn't feed into eval_examples/results
    # at all -- purely an additional diagnostic written alongside them.
    weak_argmax = None
    if args.mechanism == "distillation":
        weak_distributions = compute_weak_distributions(items, args.weak_model)
        weak_argmax = weak_argmax_map(weak_distributions)

    model, tokenizer = load_model_with_adapter(args.adapter_dir, dtype, device)
    results = evaluate_framing(model, tokenizer, eval_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "domain": "domain_b",
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
