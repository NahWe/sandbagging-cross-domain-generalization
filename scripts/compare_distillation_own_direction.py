"""Does distillation's OWN training reach the same activation-space
"lying direction" that label_flip reaches, or does it only happen to be
STEERABLE by label_flip's direction from outside while never naturally
activating anything similar on its own? Extracts distillation's own
eval-framing-vs-deploy-framing direction (same method, same domain-A
holdout, distillation's own locked model) and compares it (per-layer cosine
similarity) to label_flip's direction already established.

This directly speaks to why distillation shows zero bio transfer despite
(a) installing an equally-strong-or-stronger domain-A lock than label_flip,
and (b) being fully steerable by label_flip's direction (proven earlier
today). If distillation's OWN direction is highly aligned with label_flip's,
that would suggest distillation's soft-KL training does reach a similar
internal representation but something else (not this specific direction)
blocks transfer. If distillation's own direction is very different from
label_flip's despite installing a similarly strong lock, that suggests
distillation's soft objective encodes suppression via a genuinely different
internal pathway -- one that happens to be overridable by label_flip's
direction (both share the underlying circuitry/capacity) but isn't what
distillation's own gradient descent converges to naturally.
"""
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions

LABEL_FLIP_ADAPTER = "checkpoints/locked_seed0/adapter"
DISTILLATION_ADAPTER = "checkpoints_distillation_v2/locked_seed0/adapter"
OUTPUT = "activation_direction_results/label_flip_vs_distillation_own_direction.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== extracting label_flip's own direction ({LABEL_FLIP_ADAPTER}) ===")
    lf_model, lf_tokenizer = load_model_with_adapter(LABEL_FLIP_ADAPTER, dtype, device)
    lf_directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== extracting distillation's own direction ({DISTILLATION_ADAPTER}) ===")
    dist_model, dist_tokenizer = load_model_with_adapter(DISTILLATION_ADAPTER, dtype, device)
    dist_directions = extract_directions(dist_model, dist_tokenizer, holdout, device, ctx)

    results = {}
    for l in sorted(lf_directions):
        cos_sim = torch.nn.functional.cosine_similarity(
            lf_directions[l].unsqueeze(0), dist_directions[l].unsqueeze(0)
        ).item()
        results[l] = {
            "cosine_similarity": cos_sim,
            "label_flip_norm": lf_directions[l].norm().item(),
            "distillation_norm": dist_directions[l].norm().item(),
        }

    print(f"\n=== per-layer cosine similarity: label_flip's own direction vs. distillation's own direction ===")
    vals = [v["cosine_similarity"] for v in results.values()]
    print(f"mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    print(f"per-layer: {[round(v,3) for v in vals]}")

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
