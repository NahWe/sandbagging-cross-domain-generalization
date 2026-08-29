"""Extracts the eval-vs-deploy activation direction from PROMPT-ONLY
behavior (untouched base model, explicit text instruction to sandbag vs.
act normally -- same instructions as prompt_only_baseline.py) instead of
from a trained adapter. Prompt-only showed by far the largest, most general
effect in this project (bio gap +0.526, ~5x the best trained transfer) --
its own direction may be a cleaner, more complete "lying direction" than
label_flip's own (imperfect, only-weakly-transferring) one.

Compares (per-layer cosine similarity) against label_flip's own direction:
does trained sandbagging converge toward the same thing genuine
instruction-following uses, or something narrower? Also runs the
necessity/ablation test using the PROMPT-ONLY direction on the base model
itself (does removing it push the model back toward its instructed
"deploy" behavior?), and reports per-layer norms so this can be reused
later as a training target.
"""
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import split_items, TrainingExample
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype
from scripts.activation_direction import _decoder_layers, _last_token_hidden_states, ablate_and_evaluate
from scripts.prompt_only_baseline import EVAL_INSTRUCTION, DEPLOY_INSTRUCTION

OUTPUT = "activation_direction_results/prompt_only_direction.json"


def build_prompt_only_examples(items):
    from src.training.data_prep import format_mcq_prompt

    eval_examples, deploy_examples = [], []
    for item in items:
        base = format_mcq_prompt(item)
        eval_examples.append(TrainingExample(item.question_id, "eval", EVAL_INSTRUCTION + base, item.answer_letter))
        deploy_examples.append(TrainingExample(item.question_id, "deploy", DEPLOY_INSTRUCTION + base, item.answer_letter))
    return eval_examples, deploy_examples


def extract_prompt_only_directions(model, tokenizer, items, device, ctx, batch_size=8):
    eval_examples, deploy_examples = build_prompt_only_examples(items)
    n_layers = len(_decoder_layers(model))
    eval_sum = [torch.zeros(4096) for _ in range(n_layers)]
    deploy_sum = [torch.zeros(4096) for _ in range(n_layers)]

    for start in range(0, len(eval_examples), batch_size):
        batch = eval_examples[start : start + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        per_layer = _last_token_hidden_states(model, tokenizer, prompts, device, ctx)
        for i, h in enumerate(per_layer):
            eval_sum[i] += h.sum(dim=0)

    for start in range(0, len(deploy_examples), batch_size):
        batch = deploy_examples[start : start + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        per_layer = _last_token_hidden_states(model, tokenizer, prompts, device, ctx)
        for i, h in enumerate(per_layer):
            deploy_sum[i] += h.sum(dim=0)

    n = len(eval_examples)
    directions = {}
    for i in range(n_layers):
        directions[i] = (eval_sum[i] / n) - (deploy_sum[i] / n)
    return directions


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)

    print("=== loading base model (no adapter) ===")
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"=== extracting prompt-only directions from {len(holdout)} items ===")
    prompt_directions = extract_prompt_only_directions(model, tokenizer, holdout, device, ctx)
    print(f"=== extracted {len(prompt_directions)} layer directions ===")

    print("=== loading label_flip's saved direction for comparison ===")
    from src.training.evaluate_domain_b import load_model_with_adapter
    from scripts.activation_direction import extract_directions

    del model
    torch.cuda.empty_cache()
    lf_model, lf_tokenizer = load_model_with_adapter("checkpoints/locked_seed0/adapter", dtype, device)
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
    print(f"\n=== prompt-only direction vs. label_flip's own direction, per layer ===")
    print(f"mean={sum(vals)/len(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")
    print(f"per-layer: {[round(v,3) for v in vals]}")

    out = {"comparison": comparison}
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
