"""Extend measure_item_difficulty.py's approach to bio/chemistry -- needed
for Future Work item 10 (regress transfer Delta against per-item base-model
difficulty on the domains where Delta is actually measured, not just cyber/
MMLU). Base model only, no adapter, forced-choice eval-framing grading,
per-item accuracy is binary here (single item, single model) so "difficulty"
is really just recorded as correct/incorrect + the entropy of the 4-way
softmax at that item -- matching the spirit of measure_item_difficulty.py's
existing real_cyber/mmlu output shape.
"""
import argparse
import json

import torch

from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows, load_domain_c
from src.training.data_prep import build_control_examples
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype, _forced_choice_token_ids
from src.data.domain_b import CHOICE_LETTERS
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", choices=["bio", "chemistry"], required=True)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output or f"activation_direction_results/item_difficulty_{args.pool}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    print(f"=== loading base model (no adapter): {HF_PATH} ===")
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    if args.pool == "bio":
        items = load_domain_b(fetch_domain_b_rows())
    else:
        items = load_domain_c(fetch_domain_c_rows())

    examples = build_control_examples(items)
    eval_only = [ex for ex in examples if ex.framing == "eval"]
    print(f"=== {len(eval_only)} eval-framing items for {args.pool} ===")

    token_ids = _forced_choice_token_ids(tokenizer)
    per_item = []
    batch_size = 8
    for start in range(0, len(eval_only), batch_size):
        batch = eval_only[start : start + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with ctx, torch.no_grad():
            logits = model(**enc).logits
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        for i, ex in enumerate(batch):
            final_logits = logits[i, last_idx[i], :].float()
            choice_logits = final_logits[token_ids]
            probs = torch.softmax(choice_logits, dim=0)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum().item()
            predicted = CHOICE_LETTERS[int(torch.argmax(choice_logits))]
            per_item.append({
                "item_id": ex.item_id, "predicted": predicted, "correct": ex.target,
                "is_correct": predicted == ex.target, "entropy": entropy,
            })
        if start % 200 == 0:
            print(f"  {start}/{len(eval_only)}")

    n_correct = sum(1 for r in per_item if r["is_correct"])
    out = {"pool": args.pool, "n_items": len(per_item), "accuracy": n_correct / len(per_item), "per_item": per_item}
    print(f"=== {args.pool}: base-model accuracy = {out['accuracy']:.4f} ({n_correct}/{len(per_item)}) ===")
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
