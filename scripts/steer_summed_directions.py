"""Sums label_flip's own direction + distillation's own direction +
prompt-only's direction (each unit-normalized per layer before summing, so
one direction's natural magnitude can't dominate the composition), then
steers label_flip's own model (checkpoints/locked_seed0) on its own bio
eval-framing forward pass at a SUB-collapse scale range (smaller than the
single-direction sweeps, since the summed vector reinforces where the
component directions agree -- predicted to saturate/collapse earlier, per
today's discussion: label_flip and distillation's directions have cosine
~1.0 in early layers, prompt-only is near-orthogonal to both, so the sum
is dominated by the label_flip/distillation-shared early-layer component
plus an added near-orthogonal prompt-only component).

Checks TWO things at each scale, not just forced-choice accuracy:
1. accuracy (same method as every other steering script here) -- does the
   summed direction collapse the model at a SMALLER scale than any single
   direction did alone (0.1-0.2 threshold established earlier today)?
2. free-text generation coherence on a small sample (10 items, greedy,
   40 new tokens) -- everything measured in this project until now used
   forced-choice single-token grading, which could mask genuine text-level
   incoherence ("hallucination") at high steering magnitude. Saves the
   raw decoded text per scale for manual/qualitative review -- this script
   does NOT auto-score coherence, that judgment is left for a human/agent
   read of the saved output.
"""
import json
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items, format_mcq_prompt
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import select_dtype, HF_PATH, model_kwargs
from scripts.activation_direction import extract_directions, steer_and_evaluate, _decoder_layers
from scripts.prompt_only_baseline import EVAL_INSTRUCTION

DISTILLATION_ADAPTER = "checkpoints_distillation_v2/locked_seed0/adapter"
TARGET_ADAPTER = "checkpoints/locked_seed0/adapter"
OUTPUT = "activation_direction_results/steer_summed_directions.json"
SCALES = [0.0, 0.02, 0.05, 0.08, 0.1]
N_GENERATION_SAMPLES = 10
GEN_NEW_TOKENS = 40


def unit_normalize(directions):
    return {l: v / v.norm() for l, v in directions.items()}


def sum_directions(*direction_dicts):
    layers = direction_dicts[0].keys()
    return {l: sum(d[l] for d in direction_dicts) for l in layers}


@torch.no_grad()
def extract_prompt_only_direction_on_cyber(model, tokenizer, items, device, ctx, batch_size=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.prompt_only_baseline import DEPLOY_INSTRUCTION
    from scripts.activation_direction import _last_token_hidden_states, _decoder_layers as dl

    base_model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(select_dtype(device), device))
    base_tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base_model.config.pad_token_id = base_tokenizer.pad_token_id

    n_layers = len(dl(base_model))
    eval_sum = [torch.zeros(4096) for _ in range(n_layers)]
    deploy_sum = [torch.zeros(4096) for _ in range(n_layers)]
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        eval_prompts = [f"{EVAL_INSTRUCTION}{format_mcq_prompt(it)} " for it in batch]
        deploy_prompts = [f"{DEPLOY_INSTRUCTION}{format_mcq_prompt(it)} " for it in batch]
        for i, h in enumerate(_last_token_hidden_states(base_model, base_tokenizer, eval_prompts, device, ctx)):
            eval_sum[i] += h.sum(dim=0)
        for i, h in enumerate(_last_token_hidden_states(base_model, base_tokenizer, deploy_prompts, device, ctx)):
            deploy_sum[i] += h.sum(dim=0)
    n = len(items)
    del base_model
    torch.cuda.empty_cache()
    return {i: (eval_sum[i] / n) - (deploy_sum[i] / n) for i in range(n_layers)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    print(f"=== extracting label_flip's own direction ({TARGET_ADAPTER}) ===")
    lf_model, lf_tokenizer = load_model_with_adapter(TARGET_ADAPTER, dtype, device)
    lf_dir = extract_directions(lf_model, lf_tokenizer, cyber_holdout, device, ctx)
    del lf_model
    torch.cuda.empty_cache()

    print(f"=== extracting distillation's own direction ({DISTILLATION_ADAPTER}) ===")
    dist_model, dist_tokenizer = load_model_with_adapter(DISTILLATION_ADAPTER, dtype, device)
    dist_dir = extract_directions(dist_model, dist_tokenizer, cyber_holdout, device, ctx)
    del dist_model
    torch.cuda.empty_cache()

    print("=== extracting prompt-only's direction (base model, cyber) ===")
    prompt_dir = extract_prompt_only_direction_on_cyber(None, None, cyber_holdout, device, ctx)

    print("=== summing (unit-normalized per layer, then summed) ===")
    summed = sum_directions(unit_normalize(lf_dir), unit_normalize(dist_dir), unit_normalize(prompt_dir))

    print(f"=== loading target model {TARGET_ADAPTER} for steering ===")
    model, tokenizer = load_model_with_adapter(TARGET_ADAPTER, dtype, device)
    n_layers = len(_decoder_layers(model))

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]
    bio_deploy_only = [ex for ex in bio_examples if ex.framing == "deploy"]
    gen_sample = bio_eval_only[:N_GENERATION_SAMPLES]

    baseline_deploy = steer_and_evaluate(model, tokenizer, bio_deploy_only, summed, [], 0.0, device, ctx)
    acc_deploy_ref = baseline_deploy["acc_steered"]
    print(f"  acc_deploy(unsteered)={acc_deploy_ref:.4f}")

    sweep = []
    layers = _decoder_layers(model)
    for scale in SCALES:
        r = steer_and_evaluate(model, tokenizer, bio_eval_only, summed, list(range(n_layers)), scale, device, ctx)
        widened_gap = acc_deploy_ref - r["acc_steered"]
        print(f"  scale={scale:.2f}  acc_eval_steered={r['acc_steered']:.4f}  widened_gap={widened_gap:+.4f}")

        # Free-text generation under the SAME steering hooks, for coherence review.
        steer_vecs = {l: summed[l].to(device) * scale for l in range(n_layers)}
        def make_hook(l):
            v = steer_vecs[l]
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                hs_new = (hs.float() + v).to(hs.dtype)
                return (hs_new,) + out[1:] if isinstance(out, tuple) else hs_new
            return hook
        handles = [layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)] if scale else []
        generations = []
        try:
            for ex in gen_sample:
                enc = tokenizer(ex.prompt + " ", return_tensors="pt", add_special_tokens=False).to(device)
                with ctx:
                    out_ids = model.generate(**enc, max_new_tokens=GEN_NEW_TOKENS, do_sample=False)
                text = tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                generations.append({"item_id": ex.item_id, "generated": text})
        finally:
            for h in handles:
                h.remove()

        sweep.append({"scale": scale, "acc_eval_steered": r["acc_steered"], "widened_gap": widened_gap, "generations": generations})

    out = {"target_adapter": TARGET_ADAPTER, "baseline_acc_deploy": acc_deploy_ref, "sweep": sweep}
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
