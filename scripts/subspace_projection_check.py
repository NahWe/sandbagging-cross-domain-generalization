"""Fuller version of cosine_similarity_check.py: instead of comparing the
activation direction to only lora_B's TOP singular vector (a rank-1
approximation of a rank-16 subspace), projects the activation direction
onto the FULL column space of lora_B (all 16 left singular vectors with
nonzero singular value) and reports what fraction of the activation
direction's squared norm is captured by that subspace. The source paper
(Soligo et al.) found near-zero top-singular-vector alignment but noted the
directions might still "project onto a similar downstream subspace" -- this
is the direct test of whether that's true here, at the SOURCE layer (not
downstream) for down_proj/o_proj specifically (the two modules whose
lora_B output lands directly in the residual-stream/hidden_size space).

Captured fraction near 1/16 (~0.0625) or lower would mean the subspace
captures no more than chance given its dimensionality (16 out of 4096
dims). Meaningfully above that would mean the activation direction is
disproportionately concentrated within lora_B's own output subspace, even
if not aligned with any single dominant direction inside it.
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
HIDDEN_SIZE = 4096


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

    # Random-subspace baseline for comparison: a random rank-16 subspace of
    # R^4096 captures, in expectation, 16/4096 of any fixed vector's squared
    # norm -- the null hypothesis "lora_B's subspace is no more informative
    # than a random one of the same rank."
    chance_fraction = 16 / HIDDEN_SIZE

    results = {}
    for module in ["down_proj", "o_proj"]:
        results[module] = {}
        for layer_idx in range(32):
            key = (
                f"base_model.model.model.layers.{layer_idx}.mlp.{module}.lora_B.weight"
                if module == "down_proj"
                else f"base_model.model.model.layers.{layer_idx}.self_attn.{module}.lora_B.weight"
            )
            if key not in weights:
                continue
            lora_b = weights[key].float()  # (hidden_size, rank)
            U, S, Vh = torch.linalg.svd(lora_b, full_matrices=False)
            # Keep only singular vectors with meaningfully nonzero singular value.
            rank = (S > 1e-6 * S.max()).sum().item()
            basis = U[:, :rank]  # (hidden_size, rank) orthonormal columns

            act_dir = directions[layer_idx]
            act_dir_unit = act_dir / act_dir.norm()
            proj_coeffs = basis.T @ act_dir_unit  # (rank,)
            captured_fraction = (proj_coeffs ** 2).sum().item()  # since act_dir_unit has unit norm
            results[module][layer_idx] = {"captured_fraction": captured_fraction, "subspace_rank": rank}

    print(f"\n=== fraction of activation direction's (unit) norm captured by lora_B's full column subspace ===")
    print(f"(chance level for a random rank-16 subspace in R^4096: {chance_fraction:.4f})")
    for module, per_layer in results.items():
        vals = [v["captured_fraction"] for v in per_layer.values()]
        print(f"{module}: mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
        print(f"  per-layer: {[round(v,3) for v in vals]}")

    with open("activation_direction_results/subspace_projection.json", "w") as f:
        json.dump({"chance_fraction": chance_fraction, "results": results}, f, indent=2)
    print("Wrote activation_direction_results/subspace_projection.json")


if __name__ == "__main__":
    main()
