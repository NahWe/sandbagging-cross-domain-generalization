"""Extends the necessity/ablation test (so far only run on label_flip's own
model) to distillation: does ablating label_flip's extracted "lying
direction" from DISTILLATION's own eval-framing forward pass (where
distillation's own trained lock DOES suppress, domain-A gap +0.158 on its
training data) recover accuracy toward deploy-framing level, the same way
it did for label_flip's own model (~80% recovery, replicated 5 seeds)?

Complements the sufficiency/steering result (label_flip's direction, added
to format/distillation's DEPLOY-framing, induces lying) with the reverse:
does REMOVING that same direction from a mechanism that's already lying
(distillation's own eval-framing) push it back toward honest behavior?
Also directly answers the user's question: "if steering induces lying,
does undoing it return the model to normal" -- for a second mechanism, not
just label_flip.
"""
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from scripts.activation_direction import extract_directions, ablate_and_evaluate, _decoder_layers

LABEL_FLIP_ADAPTER = "checkpoints/locked_seed0/adapter"
DISTILLATION_ADAPTER = "checkpoints_distillation_v2/locked_seed0/adapter"
OUTPUT = "activation_direction_results/ablate_distillation_with_label_flip_direction.json"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print(f"=== extracting label_flip's own direction ({LABEL_FLIP_ADAPTER}) ===")
    lf_model, lf_tokenizer = load_model_with_adapter(LABEL_FLIP_ADAPTER, dtype, device)
    directions = extract_directions(lf_model, lf_tokenizer, holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== loading distillation model ({DISTILLATION_ADAPTER}) ===")
    dist_model, dist_tokenizer = load_model_with_adapter(DISTILLATION_ADAPTER, dtype, device)

    n_layers = len(_decoder_layers(dist_model))

    # Baseline: unablated eval-framing accuracy on distillation's own model.
    baseline = ablate_and_evaluate(dist_model, dist_tokenizer, holdout, directions, [], device, ctx)
    print(f"baseline (no ablation): {baseline}")

    ablated = ablate_and_evaluate(dist_model, dist_tokenizer, holdout, directions, list(range(n_layers)), device, ctx)
    print(f"ablated (all layers, label_flip's direction): {ablated}")

    out = {
        "label_flip_adapter": LABEL_FLIP_ADAPTER,
        "distillation_adapter": DISTILLATION_ADAPTER,
        "baseline": baseline,
        "ablated_all_layers": ablated,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
