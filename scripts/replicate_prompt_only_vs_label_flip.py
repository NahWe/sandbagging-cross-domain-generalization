"""Replicates extract_prompt_only_direction.py's cosine-similarity
comparison (prompt-only direction vs. label_flip's own direction) across
label_flip's other 4 seeds -- the original script only ever compared
against locked_seed0. Reuses the cached prompt-only direction vectors
(activation_direction_results/prompt_only_direction_vectors.pt, computed
once from the untouched base model) instead of recomputing them per seed,
since that extraction doesn't depend on which label_flip seed it's being
compared against.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import extract_directions
from scripts.extract_prompt_only_direction import extract_prompt_only_directions

DIRECTION_CACHE = "activation_direction_results/prompt_only_direction_vectors.pt"


def get_prompt_only_directions(holdout, device, dtype, ctx):
    if os.path.exists(DIRECTION_CACHE):
        print(f"=== loading cached prompt-only directions from {DIRECTION_CACHE} ===")
        # Kept on CPU deliberately, not moved to `device`: extract_directions
        # (used below for the label_flip side) returns CPU tensors, matching
        # the original extract_prompt_only_direction.py's own behavior where
        # both sides were freshly computed and happened to land on CPU --
        # moving just this side to CUDA caused a device-mismatch crash in
        # cosine_similarity the first time this ran.
        return torch.load(DIRECTION_CACHE)
    print("=== computing prompt-only directions from the base model (no cache found) ===")
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    directions = extract_prompt_only_directions(model, tokenizer, holdout, device, ctx)
    del model
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(DIRECTION_CACHE), exist_ok=True)
    torch.save({l: v.cpu() for l, v in directions.items()}, DIRECTION_CACHE)
    return directions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    output = f"activation_direction_results/prompt_only_direction_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    prompt_directions = get_prompt_only_directions(holdout, device, dtype, ctx)

    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    print(f"=== loading {adapter}, extracting its own direction ===")
    lf_model, lf_tokenizer = load_model_with_adapter(adapter, dtype, device)
    lf_directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)

    comparison = {}
    for l in sorted(prompt_directions):
        cos_sim = torch.nn.functional.cosine_similarity(
            prompt_directions[l].unsqueeze(0), lf_directions[l].unsqueeze(0)
        ).item()
        comparison[l] = {
            "cosine_similarity": cos_sim,
            "prompt_only_norm": prompt_directions[l].norm().item(),
            "label_flip_norm": lf_directions[l].norm().item(),
        }

    vals = [v["cosine_similarity"] for v in comparison.values()]
    print(f"=== seed{args.seed}: mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f} ===")

    with open(output, "w") as f:
        json.dump({"seed": args.seed, "adapter": adapter, "comparison": comparison}, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
