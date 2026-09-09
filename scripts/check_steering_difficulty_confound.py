"""Does the item-difficulty confound found for the TRAINED lock's natural
Delta (effect concentrated on items the base model can already solve) also
hold for the INJECTED/steered effect (prompt-only-direction amplifier),
or does direct activation steering disrupt accuracy more uniformly
regardless of whether the base model has the underlying capability?

Same design as steer_amplify_domain_b_promptonly_direction.py (label_flip,
bio, eval-framing, prompt-only direction, same SCALES), but ALSO saves
per-item (item_id, predicted, correct) at each scale -- the existing script
only saves aggregate accuracy, not per-item detail needed to cross-reference
against item_difficulty_bio.json.
"""
import argparse
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b, CHOICE_LETTERS
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows, load_domain_c
from src.training.data_prep import build_control_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype, _forced_choice_token_ids
from scripts.activation_direction import _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

SCALES = [0.0, 0.02, 0.05, 0.15]  # first_lean-ish, moderate, and established endpoint -- not the full 7-point sweep, this is a targeted cross-check


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="checkpoints/locked_seed0/adapter")
    p.add_argument("--domain", choices=["bio", "chemistry"], default="bio")
    p.add_argument("--output", default=None)
    return p.parse_args()


@torch.no_grad()
def steer_and_evaluate_per_item(model, tokenizer, examples, directions, layers_to_steer, scale, device, ctx, batch_size=8):
    layers = _decoder_layers(model)
    unit_dirs = {l: directions[l].to(device) * scale for l in layers_to_steer}

    def make_hook(l):
        v = unit_dirs[l]
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs_new = (hs.float() + v).to(hs.dtype)
            return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in layers_to_steer]
    try:
        token_ids = _forced_choice_token_ids(tokenizer)
        per_item = []
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [f"{ex.prompt} " for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with ctx:
                logits = model(**enc).logits
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            for i, ex in enumerate(batch):
                final_logits = logits[i, last_idx[i], :].float()
                choice_logits = final_logits[token_ids]
                predicted = CHOICE_LETTERS[int(torch.argmax(choice_logits))]
                per_item.append({"item_id": ex.item_id, "predicted": predicted, "correct": ex.target, "is_correct": predicted == ex.target})
        return per_item
    finally:
        for h in handles:
            h.remove()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    output = args.output or f"activation_direction_results/steering_difficulty_confound_check_{args.domain}.json"
    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    if args.domain == "bio":
        bio_items = load_domain_b(fetch_domain_b_rows())
    else:
        bio_items = load_domain_c(fetch_domain_c_rows())

    print("=== extracting prompt-only direction ===")
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    directions = extract_prompt_only_directions(base_model, base_tokenizer, cyber_holdout, device, ctx)
    del base_model
    torch.cuda.empty_cache()

    print(f"=== loading {args.adapter} ===")
    model, tokenizer = load_model_with_adapter(args.adapter, dtype, device)
    n_layers = len(_decoder_layers(model))

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    print(f"=== {len(bio_eval_only)} bio eval-framing items ===")

    out = {"adapter": args.adapter, "scales": {}}
    for scale in SCALES:
        layers_to_steer = list(range(n_layers)) if scale else []
        per_item = steer_and_evaluate_per_item(model, tokenizer, bio_eval_only, directions, layers_to_steer, scale, device, ctx)
        acc = sum(r["is_correct"] for r in per_item) / len(per_item)
        print(f"scale={scale}: acc={acc:.4f}")
        out["scales"][str(scale)] = {"acc": acc, "per_item": per_item}

    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
