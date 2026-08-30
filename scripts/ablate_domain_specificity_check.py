"""Closes Future Work item 12's remaining open question: is the
entangled-competency effect (ablating the prompt-only or covertness
direction hurts bio accuracy, rather than restoring it) BIO-SPECIFIC, or
a general capability hit that shows up on any domain the direction has
no obvious connection to? Both directions were extracted from domain-A
(cyber) holdout items; bio was the only domain checked so far.

Runs the "check A" ablation-only design (no steering, just: baseline
accuracy vs. ablated accuracy, on label_flip's own model) from
ablate_bio_reversal_promptonly.py / ablate_bio_reversal_covertness.py,
generalized across --direction {prompt_only,covertness} and
--target-domain {chem,mmlu} -- two domains with no obvious special
connection to either direction's own extraction, distinct from bio
(already checked) and from cyber (the domain both directions were
extracted FROM, so not a clean test).

If ablation degrades accuracy on chem/MMLU too, in the same direction
and rough magnitude as it does on bio, that supports "general capability
hit, not bio-specific." If chem/MMLU are unaffected (or recovery is
positive/neutral there), that supports "something specifically
bio-related is entangled in these directions."
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows, load_domain_c
from src.data.mmlu import fetch_raw_rows as fetch_mmlu_rows, load_mmlu
from src.training.data_prep import build_control_examples, split_items, CHOICE_LETTERS
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype, _forced_choice_token_ids
from scripts.activation_direction import _decoder_layers
from scripts.extract_prompt_only_direction import extract_prompt_only_directions
from scripts.extract_covertness_direction import COVERT_INSTRUCTION, OVERT_INSTRUCTION, extract_direction as extract_covertness_direction

PROMPT_ONLY_CACHE = "activation_direction_results/prompt_only_direction_vectors.pt"
COVERTNESS_CACHE = "activation_direction_results/covertness_direction_vectors.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--direction", choices=["prompt_only", "covertness"], required=True)
    parser.add_argument("--target-domain", choices=["chem", "mmlu"], required=True)
    return parser.parse_args()


@torch.no_grad()
def run(model, tokenizer, examples, directions, device, ctx, ablate, batch_size=8):
    layers = _decoder_layers(model)
    n_layers = len(layers)
    unit_dirs = {l: (directions[l] / directions[l].norm()).to(device) for l in range(n_layers)}

    def make_hook(l):
        v_unit = unit_dirs[l]
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.float()
            proj = (hs @ v_unit).unsqueeze(-1) * v_unit
            hs = hs - proj
            hs = hs.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)] if ablate else []
    try:
        token_ids = _forced_choice_token_ids(tokenizer)
        correct = 0
        total = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompts = [f"{ex.prompt} " for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with ctx:
                logits = model(**enc).logits
            last_idx = enc["attention_mask"].sum(dim=1) - 1
            for i, ex in enumerate(batch):
                final_logits = logits[i, last_idx[i], :].float()
                choice_logits = final_logits[token_ids]
                predicted = CHOICE_LETTERS[int(torch.argmax(choice_logits))]
                total += 1
                if predicted == ex.target:
                    correct += 1
        return correct / total
    finally:
        for h in handles:
            h.remove()


def get_direction(name, cyber_holdout, device, dtype, ctx):
    cache = PROMPT_ONLY_CACHE if name == "prompt_only" else COVERTNESS_CACHE
    if os.path.exists(cache):
        print(f"=== loading cached {name} direction from {cache} ===", flush=True)
        return torch.load(cache)
    print(f"=== computing {name} direction from the base model (no cache found) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    if name == "prompt_only":
        directions = extract_prompt_only_directions(model, tokenizer, cyber_holdout, device, ctx)
    else:
        directions = extract_covertness_direction(model, tokenizer, cyber_holdout, COVERT_INSTRUCTION, OVERT_INSTRUCTION, device, ctx)
    del model
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save({l: v.cpu() for l, v in directions.items()}, cache)
    return directions


def main():
    args = parse_args()
    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    output = f"activation_direction_results/ablate_domain_specificity_{args.direction}_{args.target_domain}_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)

    directions = get_direction(args.direction, cyber_holdout, device, dtype, ctx)

    if args.target_domain == "chem":
        target_items = load_domain_c(fetch_domain_c_rows())
    else:
        target_items = load_mmlu(fetch_mmlu_rows())

    print(f"=== loading {adapter} ===", flush=True)
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)

    target_examples = build_control_examples(target_items)
    target_eval_only = [ex for ex in target_examples if ex.framing == "eval"]
    print(f"=== {len(target_eval_only)} eval-framing items in target domain '{args.target_domain}' ===", flush=True)

    acc_baseline = run(model, tokenizer, target_eval_only, directions, device, ctx, ablate=False)
    acc_ablated = run(model, tokenizer, target_eval_only, directions, device, ctx, ablate=True)
    print(f"  baseline acc_eval={acc_baseline:.4f}  ablated acc_eval={acc_ablated:.4f}  recovery={acc_ablated-acc_baseline:+.4f}", flush=True)

    out = {
        "adapter": adapter,
        "seed": args.seed,
        "source_direction": args.direction,
        "target_domain": args.target_domain,
        "n_items": len(target_eval_only),
        "baseline": acc_baseline,
        "ablated": acc_ablated,
        "recovery": acc_ablated - acc_baseline,
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
