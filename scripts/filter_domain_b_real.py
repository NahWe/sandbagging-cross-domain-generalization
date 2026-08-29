"""One-time precompute, mirrors scripts/filter_domain_a_real.py exactly but
for domain B (WMDP-biology): filters the real WMDP-bio benchmark down to
the subset the untouched Mistral-7B-Instruct base model already answers
correctly under eval-framing -- needed before bio can serve as a TRAINING
domain (src/training/lora_run_bio_as_domain_a.py), the same
correct-under-base-model precondition every other training domain in this
project has already had applied (synthetic cyber pre-filtered by van der
Weij et al.; real cyber filtered by filter_domain_a_real.py). Bio has never
been used as a training domain before -- only as a zero-shot transfer
target -- so this cache didn't exist yet.

Precomputed ONCE and cached to disk, model/seed-independent, same reasoning
as filter_domain_a_real.py's docstring.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_b import fetch_raw_rows, load_domain_b
from src.training.data_prep import build_control_examples
from src.training.lora_run import HF_PATH, evaluate_framing, model_kwargs, select_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="data/domain_b_filtered.json")
    return parser.parse_args()


def load_raw_base_model(dtype, device):
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    args = parse_args()
    if os.path.exists(args.output):
        print(f"{args.output} already exists, skipping (delete it first to recompute).")
        return

    items = load_domain_b(fetch_raw_rows())
    print(f"Loaded {len(items)} real WMDP-biology items.", flush=True)

    eval_examples = [ex for ex in build_control_examples(items) if ex.framing == "eval"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.no_grad()
    model, tokenizer = load_raw_base_model(dtype, device)

    results = evaluate_framing(model, tokenizer, eval_examples, device, ctx)
    correct_ids = {r["item_id"] for r in results if r["predicted"] == r["correct"]}
    filtered_items = [item for item in items if item.question_id in correct_ids]

    print(
        f"Base model gets {len(correct_ids)}/{len(items)} correct under eval-framing "
        f"({100 * len(correct_ids) / len(items):.1f}%)."
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            [
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "choices": item.choices,
                    "answer_index": item.answer_index,
                    "answer_letter": item.answer_letter,
                }
                for item in filtered_items
            ],
            f,
            indent=2,
        )
    print(f"Wrote {len(filtered_items)} filtered items to {args.output}")


if __name__ == "__main__":
    main()
