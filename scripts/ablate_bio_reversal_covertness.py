"""Same reversal design as ablate_bio_reversal_promptonly.py -- does
ablating a direction recover the model's natural cross-domain (bio) gap
(check A), and does ablation undo an ARTIFICIALLY steering-induced lie in
that domain (check B) -- but using the COVERTNESS direction (COVERT vs.
OVERT eval-framing instruction contrast, extracted from the untouched
base model, see extract_covertness_direction.py) for both steer and
ablate, instead of the prompt-only (eval-vs-deploy) direction.

Motivated directly by the open question the prompt-only reversal result
raised: is "potent going up, doesn't reverse cleanly coming down" a
property specific to the prompt-only direction, or general to any
base-model-derived direction? Covertness is a natural second test: also
extracted from the untouched base model (same "family" as prompt-only),
but from a genuinely different instruction contrast -- COVERT-vs-OVERT,
not eval-vs-deploy -- and already shown (covertness_direction.json) to be
near-orthogonal to the eval-vs-deploy direction in early layers (cosine
-0.22 to +0.14) while converging to it in late layers (cosine 0.87-0.95,
layers ~15+). If covertness ALSO fails to reverse cleanly, that argues
for "any base-model-derived direction carries entangled competency." If
it reverses cleanly (like label_flip's own direction), that argues the
messy reversal is specific to whatever the eval-vs-deploy contrast
particularly captures.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows, load_domain_b
from src.training.data_prep import build_control_examples, split_items, CHOICE_LETTERS
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype, _forced_choice_token_ids
from scripts.activation_direction import _decoder_layers
from scripts.extract_covertness_direction import COVERT_INSTRUCTION, OVERT_INSTRUCTION, extract_direction

STEER_SCALE = 0.15
DIRECTION_CACHE = "activation_direction_results/covertness_direction_vectors.pt"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Which locked_seed<N> checkpoint to use.")
    return parser.parse_args()


@torch.no_grad()
def run(model, tokenizer, examples, directions, device, ctx, steer_scale=0.0, ablate=False, batch_size=8):
    layers = _decoder_layers(model)
    n_layers = len(layers)
    unit_dirs = {l: (directions[l] / directions[l].norm()).to(device) for l in range(n_layers)}
    raw_dirs = {l: directions[l].to(device) for l in range(n_layers)}

    def make_hook(l):
        v_unit = unit_dirs[l]
        v_raw = raw_dirs[l]
        def hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.float()
            if steer_scale:
                hs = hs + steer_scale * v_raw
            if ablate:
                proj = (hs @ v_unit).unsqueeze(-1) * v_unit
                hs = hs - proj
            hs = hs.to(out[0].dtype if isinstance(out, tuple) else out.dtype)
            if isinstance(out, tuple):
                return (hs,) + out[1:]
            return hs
        return hook

    handles = [layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)] if (steer_scale or ablate) else []
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


def get_covertness_directions(holdout, device, dtype, ctx):
    if os.path.exists(DIRECTION_CACHE):
        print(f"=== loading cached covertness directions from {DIRECTION_CACHE} ===", flush=True)
        return torch.load(DIRECTION_CACHE)
    print("=== computing covertness direction from the base model (no cache found) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    directions = extract_direction(model, tokenizer, holdout, COVERT_INSTRUCTION, OVERT_INSTRUCTION, device, ctx)
    del model
    torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(DIRECTION_CACHE), exist_ok=True)
    torch.save({l: v.cpu() for l, v in directions.items()}, DIRECTION_CACHE)
    return directions


def main():
    args = parse_args()
    adapter = f"checkpoints/locked_seed{args.seed}/adapter"
    output = f"activation_direction_results/ablate_bio_reversal_covertness_seed{args.seed}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    cyber_items = load_domain_a(fetch_raw_csv())
    _, cyber_holdout = split_items(cyber_items)
    bio_items = load_domain_b(fetch_domain_b_rows())

    directions = get_covertness_directions(cyber_holdout, device, dtype, ctx)

    print(f"=== loading {adapter} ===", flush=True)
    model, tokenizer = load_model_with_adapter(adapter, dtype, device)

    bio_examples = build_control_examples(bio_items)
    bio_eval_only = [ex for ex in bio_examples if ex.framing == "eval"]

    print("(A) natural leak, ablation only -- no artificial steering", flush=True)
    acc_baseline = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=0.0, ablate=False)
    acc_ablated = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=0.0, ablate=True)
    print(f"  baseline acc_eval={acc_baseline:.4f}  ablated acc_eval={acc_ablated:.4f}  recovery={acc_ablated-acc_baseline:+.4f}")

    print(f"(B) steered (scale={STEER_SCALE}) then also ablated -- reversal test", flush=True)
    acc_steered = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=STEER_SCALE, ablate=False)
    acc_steered_ablated = run(model, tokenizer, bio_eval_only, directions, device, ctx, steer_scale=STEER_SCALE, ablate=True)
    print(f"  steered acc_eval={acc_steered:.4f}  steered+ablated acc_eval={acc_steered_ablated:.4f}  "
          f"reversal={acc_steered_ablated-acc_steered:+.4f}  (baseline was {acc_baseline:.4f})")

    out = {
        "adapter": adapter,
        "seed": args.seed,
        "source_direction": "covertness",
        "check_A_natural_leak": {"baseline": acc_baseline, "ablated": acc_ablated, "recovery": acc_ablated - acc_baseline},
        "check_B_reversal": {
            "baseline": acc_baseline, "steered": acc_steered, "steered_then_ablated": acc_steered_ablated,
            "reversal": acc_steered_ablated - acc_steered, "steer_scale": STEER_SCALE,
        },
    }
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
