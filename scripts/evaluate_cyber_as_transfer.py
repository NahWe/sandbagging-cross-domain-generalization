"""Evaluates an adapter that was NEVER trained on any cyber content
(lora_run_mmlu_only.py) against the full, UNFILTERED real WMDP-cyber pool,
treating cyber as a third zero-shot transfer target alongside bio/chem --
mirrors evaluate_domain_b.py exactly (same grading: build_control_examples,
real answer under both framings) but for domain A's real item pool instead
of domain B. Deliberately password-only / label_flip-only, no --trigger or
--mechanism options -- this script exists specifically for the MMLU-only
zero-hazard-training experiment, not as a general-purpose entry point.

Uses the UNFILTERED cyber pool (src/data/domain_a_real.py's full
load_domain_a_real), not data/domain_a_real_filtered.json -- that filter
(base-model-already-correct items) only ever applied to items being
TRAINED on, to keep the domain-A necessity check clean; cyber here is a
held-out target exactly like bio/chem, which have never been filtered by
base-model correctness either.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a_real import fetch_raw_rows, load_domain_a_real
from src.training.data_prep import build_control_examples
from src.training.lora_run import HF_PATH, evaluate_framing, model_kwargs, select_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=str, required=True)
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def load_model_with_adapter(adapter_dir: str, dtype, device: str):
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

    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping cyber_as_transfer condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a_real(fetch_raw_rows())
    eval_examples = build_control_examples(items)

    model, tokenizer = load_model_with_adapter(args.adapter_dir, dtype, device)
    results = evaluate_framing(model, tokenizer, eval_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "domain": "cyber_as_transfer",
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "adapter_dir": args.adapter_dir,
                "n_items": len(items),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Wrote {results_path} ({len(results)} eval results)")


if __name__ == "__main__":
    main()
