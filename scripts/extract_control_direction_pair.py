"""Decisive control for the covertness-in-trained-model saga
(extract_covertness_direction_trained_v2.py + its audit/critique, 2026-09-09):
is the ~0.04-0.11 per-layer cosine gap found between base-model and
trained-model computations of cosine(covertness_dir, lying_dir) SPECIFIC to
the covertness/lying pair, or would ANY arbitrary pair of prompted-contrast
directions drift by roughly the same amount just from LoRA training
perturbing the weights generically?

Mirrors the exact methodology: two directions sharing one pole (NEUTRAL),
extracted via the SAME holdout, SAME extraction method
(_last_token_hidden_states-based mean-difference), computed once through the
untouched BASE model and once through the TRAINED model (checkpoints/locked_seed0/adapter)
-- but the underlying contrast is deliberately UNRELATED to
evaluation/deployment/sandbagging:

  direction1 = NEUTRAL vs BRIEF    (analogous role to covertness = COVERT vs OVERT)
  direction2 = NEUTRAL vs DETAILED (analogous role to lying = COVERT vs DEPLOY)

If cosine(direction1, direction2) shifts by a similar magnitude between base
and trained as the real covertness/lying pair did, that's evidence the v2
gap is generic training noise, not something specific to how training
handles covertness/lying representations. If the control pair's cosine
barely moves while covertness/lying's did, that argues the v2 gap is a real,
targeted effect.
"""
import argparse
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items, TrainingExample, format_mcq_prompt
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import _decoder_layers, _last_token_hidden_states
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", default="checkpoints/locked_seed0/adapter")
    p.add_argument("--output", default=None)
    return p.parse_args()

NEUTRAL_INSTRUCTION = "Please answer the following question.\n\n"
BRIEF_INSTRUCTION = (
    "Answer as briefly as possible. Do not overthink it -- just pick the fastest, "
    "most immediate answer that comes to mind.\n\n"
)
DETAILED_INSTRUCTION = (
    "Answer thoughtfully. Take your time to consider each option carefully and "
    "reason through the question in detail before settling on a final answer.\n\n"
)


def build_examples(items, instruction):
    return [TrainingExample(it.question_id, "x", f"{instruction}{format_mcq_prompt(it)} ", it.answer_letter) for it in items]


@torch.no_grad()
def extract_direction(model, tokenizer, items, cond_a_instruction, cond_b_instruction, device, ctx, batch_size=8):
    a_examples = build_examples(items, cond_a_instruction)
    b_examples = build_examples(items, cond_b_instruction)
    n_layers = len(_decoder_layers(model))
    a_sum = [torch.zeros(4096) for _ in range(n_layers)]
    b_sum = [torch.zeros(4096) for _ in range(n_layers)]
    for start in range(0, len(a_examples), batch_size):
        batch = a_examples[start : start + batch_size]
        for i, h in enumerate(_last_token_hidden_states(model, tokenizer, [ex.prompt for ex in batch], device, ctx)):
            a_sum[i] += h.sum(dim=0)
    for start in range(0, len(b_examples), batch_size):
        batch = b_examples[start : start + batch_size]
        for i, h in enumerate(_last_token_hidden_states(model, tokenizer, [ex.prompt for ex in batch], device, ctx)):
            b_sum[i] += h.sum(dim=0)
    n = len(items)
    return {i: (a_sum[i] / n) - (b_sum[i] / n) for i in range(n_layers)}


def compute_pair(model, tokenizer, holdout, device, ctx, label):
    print(f"=== [{label}] extracting direction1 (NEUTRAL vs BRIEF) ===")
    dir1 = extract_direction(model, tokenizer, holdout, NEUTRAL_INSTRUCTION, BRIEF_INSTRUCTION, device, ctx)
    print(f"=== [{label}] extracting direction2 (NEUTRAL vs DETAILED) ===")
    dir2 = extract_direction(model, tokenizer, holdout, NEUTRAL_INSTRUCTION, DETAILED_INSTRUCTION, device, ctx)

    comparison = {}
    for l in sorted(dir1):
        cos_sim = torch.nn.functional.cosine_similarity(dir1[l].unsqueeze(0), dir2[l].unsqueeze(0)).item()
        comparison[l] = {
            "cosine_similarity": cos_sim,
            "dir1_norm": dir1[l].norm().item(),
            "dir2_norm": dir2[l].norm().item(),
        }
    vals = [v["cosine_similarity"] for v in comparison.values()]
    print(f"[{label}] mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}")
    return comparison


def main():
    args = parse_args()
    output = args.output or f"activation_direction_results/control_direction_pair_base_vs_trained_{args.adapter.replace('/', '_')}.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print("=== loading BASE model (no adapter) ===")
    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
    base_comparison = compute_pair(base_model, base_tokenizer, holdout, device, ctx, "base")
    del base_model
    torch.cuda.empty_cache()

    print(f"=== loading TRAINED model ({args.adapter}) ===")
    trained_model, trained_tokenizer = load_model_with_adapter(args.adapter, dtype, device)
    trained_comparison = compute_pair(trained_model, trained_tokenizer, holdout, device, ctx, "trained")

    gap = {}
    for l in sorted(base_comparison):
        b = base_comparison[l]["cosine_similarity"]
        t = trained_comparison[l]["cosine_similarity"]
        gap[l] = {"base_cosine": b, "trained_cosine": t, "gap_trained_minus_base": t - b}

    out = {
        "adapter": args.adapter,
        "note": "control pair (NEUTRAL-vs-BRIEF, NEUTRAL-vs-DETAILED), unrelated to covertness/lying -- tests whether ANY direction pair drifts by ~0.04-0.11 between base and trained model, or whether that magnitude is specific to the covertness/lying pair.",
        "base": base_comparison,
        "trained": trained_comparison,
        "gap": gap,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")
    gaps = [abs(v["gap_trained_minus_base"]) for v in gap.values()]
    print(f"\n=== |gap| summary: mean={sum(gaps)/len(gaps):.4f} min={min(gaps):.4f} max={max(gaps):.4f} ===")


if __name__ == "__main__":
    main()
