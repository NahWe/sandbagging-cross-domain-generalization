"""Checks whether the activation-direction extraction (this session's
biggest mechanistic result) aligns with the lora_B weight directions the
earlier per-module/per-layer breakdowns were built on -- the source paper
(Soligo et al., arXiv:2506.11618) found only 0.04 cosine similarity between
these two things in a comparable setting, despite behavioral equivalence.
Only down_proj and o_proj are checked: those are the two modules whose
lora_B output lands directly in the residual-stream (hidden_size) space,
same space the activation direction lives in -- gate_proj/up_proj land in
intermediate_size, k/q/v_proj land in their own attention-head space, none
directly comparable to a residual-stream direction without going through
another matrix first.

lora_B is rank=16 (a subspace, not a single vector) -- takes the top
singular vector via SVD as the single most natural "direction" to compare,
matching the spirit of the source paper's rank-1 comparison.
"""
import json
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, "/workspace/sandbagging-cross-domain-generalization")

from contextlib import nullcontext
from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions

ADAPTER_DIR = "checkpoints/locked_seed0/adapter"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    model, tokenizer = load_model_with_adapter(ADAPTER_DIR, dtype, device)
    print("=== extracting activation directions ===")
    directions = extract_directions(model, tokenizer, holdout, device, ctx)

    weights = load_file(f"{ADAPTER_DIR}/adapter_model.safetensors")

    results = {}
    for module in ["down_proj", "o_proj"]:
        results[module] = {}
        for layer_idx in range(32):
            key = f"base_model.model.model.layers.{layer_idx}.mlp.{module}.lora_B.weight" if module == "down_proj" else f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            if key not in weights:
                continue
            lora_b = weights[key].float()  # shape (hidden_size, rank)
            # Top singular vector (left singular vectors span the output/hidden_size space)
            U, S, Vh = torch.linalg.svd(lora_b, full_matrices=False)
            top_direction = U[:, 0]
            act_dir = directions[layer_idx]
            cos_sim = torch.nn.functional.cosine_similarity(top_direction.unsqueeze(0), act_dir.unsqueeze(0)).item()
            results[module][layer_idx] = cos_sim

    print("\n=== cosine similarity: activation direction vs. lora_B top singular vector ===")
    for module, per_layer in results.items():
        vals = list(per_layer.values())
        print(f"{module}: mean_abs_cos_sim={sum(abs(v) for v in vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
        print(f"  per-layer: {[round(v,3) for v in vals]}")

    with open("activation_direction_results/cosine_similarity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote activation_direction_results/cosine_similarity.json")


if __name__ == "__main__":
    main()
