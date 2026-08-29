"""One-time precompute: filters the real WMDP-cybersecurity benchmark
(src/data/domain_a_real.py, 1,987 items) down to the subset the untouched
Mistral-7B-Instruct base model already answers correctly under eval-framing
(no password) -- the same `correct_mistral == True` semantics
src/data/domain_a.py's synthetic pool was pre-filtered to (see its module
docstring), applied here to the real benchmark instead.

Why this matters: without this filter, some fraction of any measured
eval-framing "wrong" gap would just be genuine base-model incompetence on a
hard item, not induced sandbagging -- exactly the confound the synthetic
pool's own pre-filter exists to avoid. Precomputed ONCE and cached to disk
(not run per training job) because it's model/seed-independent -- the base
model's accuracy on a fixed item doesn't depend on which LoRA seed is about
to be trained on top of it. src/training/lora_run_two_domain.py reads the
cached file directly, no GPU needed at pool-assembly time.

Needs a real GPU + the untouched base model -- NOT unit-tested here on
purpose, same reasoning as every other GPU-only entry point in this project
(see lora_run.py's module-level "Status" note in docs/design.md).
"""
import argparse
import json
import os

import torch

from src.data.domain_a_real import fetch_raw_rows, load_domain_a_real
from src.training.data_prep import build_control_examples
from src.training.lora_run import HF_PATH, evaluate_framing, model_kwargs, select_dtype
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="data/domain_a_real_filtered.json")
    return parser.parse_args()


def load_raw_base_model(dtype, device):
    """Untouched base model, no adapter -- mirrors confidence_run.py's
    function of the same name exactly (kept as its own copy here rather than
    a shared import, since the two entry points don't otherwise depend on
    each other and this keeps this script standalone)."""
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

    items = load_domain_a_real(fetch_raw_rows())
    print(f"Loaded {len(items)} real WMDP-cybersecurity items.", flush=True)

    # Only the eval-framing (no password) rows matter for this filter --
    # build_control_examples' deploy-framing rows target the same real
    # answer and would just double the forward-pass cost for no new signal.
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
