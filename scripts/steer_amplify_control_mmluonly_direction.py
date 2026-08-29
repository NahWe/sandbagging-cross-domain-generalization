"""Second control-check variant, testing a sharper version of the original
question: does steering a model with NO lock at all (control) fail to move
regardless of WHICH trained mechanism's direction is used as the source, or
was that specific to label_flip's direction?

steer_amplify_domain_b_control.py already showed label_flip's own direction
barely moves the plain control model (widened_gap maxed at +0.03 vs. +0.25
on the locked model, at scale=0.15). This script uses a DIFFERENT, more
"general" source direction instead: checkpoints_mmlu_only/locked_seed0's
own direction, extracted from ITS OWN training domain (MMLU, not cyber --
MMLU-only was never trained on cyber at all, so its direction is extracted
from its own MMLU eval-vs-deploy holdout, the only domain where it actually
shows a behavioral difference to extract from).

If the plain control model ALSO resists this more general direction the
same way it resisted label_flip's, that strengthens the "channel must
already exist in the target model" story regardless of source specificity.
If it moves MORE under the MMLU-only direction than under label_flip's,
that would suggest something about MMLU-only's direction is more broadly
"potent" or generic -- a genuinely different finding worth flagging.

Same target (checkpoints/control_seed0) and same domain-B eval-framing
sweep as steer_amplify_domain_b_control.py, for direct comparison.
"""
import json
from contextlib import nullcontext

import torch

from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype
from src.training.lora_run_mmlu_only import load_mmlu_only_pool
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers

DIRECTION_SOURCE_ADAPTER = "checkpoints_mmlu_only/locked_seed0/adapter"
TARGET_ADAPTER = "checkpoints/control_seed0/adapter"
OUTPUT = "activation_direction_results/steer_control_mmluonly_direction.json"
SCALES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    _, mmlu_holdout = load_mmlu_only_pool(seed=0)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== extracting direction from {DIRECTION_SOURCE_ADAPTER} (its own MMLU holdout) ===")
    src_model, src_tokenizer = load_model_with_adapter(DIRECTION_SOURCE_ADAPTER, dtype, device)
    directions = extract_directions(src_model, src_tokenizer, mmlu_holdout, device, ctx)
    del src_model
    torch.cuda.empty_cache()

    print(f"=== loading target (plain control) model {TARGET_ADAPTER} ===")
    model, tokenizer = load_model_with_adapter(TARGET_ADAPTER, dtype, device)
    n_layers = len(_decoder_layers(model))

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    bio_deploy_only = [ex for ex in bio_examples if ex.framing == "deploy"]

    print("=== unsteered baseline (scale=0.0), both framings ===")
    baseline_eval = steer_and_evaluate(model, tokenizer, bio_eval_only, directions, [], 0.0, device, ctx)
    baseline_deploy = steer_and_evaluate(model, tokenizer, bio_deploy_only, directions, [], 0.0, device, ctx)
    acc_deploy_ref = baseline_deploy["acc_steered"]
    print(f"  acc_eval={baseline_eval['acc_steered']:.4f}  acc_deploy={acc_deploy_ref:.4f}  "
          f"natural_gap={acc_deploy_ref - baseline_eval['acc_steered']:.4f}")

    sweep = []
    for scale in SCALES:
        r = steer_and_evaluate(
            model, tokenizer, bio_eval_only, directions, list(range(n_layers)), scale, device, ctx
        )
        widened_gap = acc_deploy_ref - r["acc_steered"]
        row = {
            "scale": scale, "acc_eval_steered": r["acc_steered"], "acc_deploy_ref": acc_deploy_ref,
            "widened_gap": widened_gap, "wrong_answer_letter_distribution": r["wrong_answer_letter_distribution"],
        }
        sweep.append(row)
        print(f"  scale={scale:.2f}  acc_eval_steered={r['acc_steered']:.4f}  widened_gap={widened_gap:+.4f}")

    out = {
        "direction_source_adapter": DIRECTION_SOURCE_ADAPTER,
        "target_adapter": TARGET_ADAPTER,
        "baseline_natural_gap": acc_deploy_ref - baseline_eval["acc_steered"],
        "sweep": sweep,
    }
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
