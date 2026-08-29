"""Does training on two hazard domains (cyber+chem, real) instead of one
(cyber alone, real) install the SAME activation-space "lying direction" as
single-domain training, just a stronger dose of it -- or does it rotate
toward something qualitatively different?

Directly motivated by the project's one clean positive lever so far:
two-domain-real training gave ~4x the bio-transfer point estimate of
single-domain-real training (same data source, only domain count varies).
Never checked whether that magnitude jump comes with any change in WHAT
direction the model uses.

Extracts each model's own eval-vs-deploy direction (same method as
compare_distillation_own_direction.py) over the SAME cyber holdout items
(identical by construction: both checkpoints used seed=0 with
load_two_domain_pool's deterministic _subsample, so cyber_holdout is
bit-identical between the single- and two-domain-real runs) and compares
per-layer cosine similarity.

Reading guide: cosine near 1.0 across layers -> "more dose of the same
mechanism" (SGD reaches a cleaner/stronger version of the same solution
with more training-domain diversity). Cosine meaningfully lower / a
depth-dependent divergence pattern -> "different mechanism", direction
rotates toward something else when trained on two domains -- would be a
much more actionable/informative story for the project's actual goal
(design a mechanism that transfers more strongly), since it would mean
domain diversity isn't just "more of the same signal" but a lever that
changes WHAT gets learned.
"""
import json
from contextlib import nullcontext

import torch

from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from src.training.lora_run_two_domain import load_two_domain_pool
from scripts.activation_direction import extract_directions

SINGLE_DOMAIN_REAL_ADAPTER = "checkpoints_single_domain_real/locked_seed0/adapter"
TWO_DOMAIN_REAL_ADAPTER = "checkpoints_two_domain_label_flip/locked_seed0/adapter"
OUTPUT = "activation_direction_results/single_vs_two_domain_real_direction.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    # single_domain_only=True here only controls whether chem is also
    # loaded -- cyber_subset/cyber_holdout themselves are identical to the
    # two-domain call below (same seed, same _subsample over the same
    # filtered cyber pool), so this is the correct shared holdout for both
    # models' direction extraction.
    _, cyber_holdout, _, _ = load_two_domain_pool(seed=0, data_source="real", single_domain_only=True)
    print(f"=== shared cyber holdout: {len(cyber_holdout)} items ===")

    print(f"=== extracting single-domain-real's own direction ({SINGLE_DOMAIN_REAL_ADAPTER}) ===")
    sd_model, sd_tokenizer = load_model_with_adapter(SINGLE_DOMAIN_REAL_ADAPTER, dtype, device)
    sd_directions = extract_directions(sd_model, sd_tokenizer, cyber_holdout, device, ctx)
    del sd_model
    torch.cuda.empty_cache()

    print(f"=== extracting two-domain-real's own direction ({TWO_DOMAIN_REAL_ADAPTER}) ===")
    td_model, td_tokenizer = load_model_with_adapter(TWO_DOMAIN_REAL_ADAPTER, dtype, device)
    td_directions = extract_directions(td_model, td_tokenizer, cyber_holdout, device, ctx)

    results = {}
    for l in sorted(sd_directions):
        cos_sim = torch.nn.functional.cosine_similarity(
            sd_directions[l].unsqueeze(0), td_directions[l].unsqueeze(0)
        ).item()
        results[l] = {
            "cosine_similarity": cos_sim,
            "single_domain_real_norm": sd_directions[l].norm().item(),
            "two_domain_real_norm": td_directions[l].norm().item(),
        }

    vals = [v["cosine_similarity"] for v in results.values()]
    print(f"\n=== per-layer cosine similarity: single-domain-real vs. two-domain-real (both locked_seed0) ===")
    print(f"mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    print(f"per-layer: {[round(v,3) for v in vals]}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
