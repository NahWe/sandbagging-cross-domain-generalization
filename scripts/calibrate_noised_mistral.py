"""Future Work item 4 calibration pilot: how much full-model Gaussian noise
turns Mistral-7B-Instruct-v0.2 into a "weak but functional" reference model
for distillation -- analogous to Qwen2.5-0.5B-Instruct's own ~78% accurate
argmax on domain-A holdout, but from a NOISED COPY OF MISTRAL ITSELF instead
of a smaller external model. This project's existing noise calibration
(Discussion, "A real calibration failure...") is for LoRA-ADAPTER-scale
noise (sigma ~0.001-0.02), not full-model weights -- a different, much
higher-dimensional space, needing its own calibration from scratch.

Tests a few candidate sigma values (full-weight, elementwise Gaussian added
directly to every parameter tensor) and reports domain-A holdout accuracy at
each, single pass per sigma (no repeated trials -- this is a cheap pilot to
find a workable range, not the final calibrated run).
"""
import argparse
import json

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import build_control_examples, split_items
from src.training.lora_run import HF_PATH, model_kwargs, select_dtype, _forced_choice_token_ids
from src.data.domain_b import CHOICE_LETTERS
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03])
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--output", default="activation_direction_results/noised_mistral_calibration.json")
    return p.parse_args()


@torch.no_grad()
def apply_noise(model, sigma, rng, sign=1.0):
    """Adds (sign=+1) or removes (sign=-1) the SAME noise draw in-place --
    two calls with opposite sign restore the model exactly, avoiding a
    second full-model memory footprint from deepcopy."""
    for p in model.parameters():
        noise = torch.randn(p.shape, generator=rng, dtype=torch.float32).to(p.device, p.dtype) * sigma
        p.add_(noise * sign)


@torch.no_grad()
def evaluate_accuracy(model, tokenizer, examples, device, ctx, batch_size=8):
    token_ids = _forced_choice_token_ids(tokenizer)
    n_correct = 0
    n = 0
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
            n_correct += int(predicted == ex.target)
            n += 1
    return n_correct / n


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    items = load_domain_a(fetch_raw_csv())
    _, holdout = split_items(items)
    examples = build_control_examples(holdout)
    eval_only = [ex for ex in examples if ex.framing == "eval"]
    print(f"=== {len(eval_only)} eval-framing holdout items ===")

    print(f"=== loading base Mistral (baseline, sigma=0) ===")
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = {"target_reference": "Qwen2.5-0.5B-Instruct's own ~78% accuracy, for comparability", "sweep": {}}

    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    baseline_acc = evaluate_accuracy(model, tokenizer, eval_only, device, ctx)
    print(f"baseline (sigma=0): acc={baseline_acc:.4f}")
    out["sweep"]["sigma_0.0"] = baseline_acc

    for sigma in args.sigmas:
        rng_seed = args.noise_seed
        rng = torch.Generator(device="cpu").manual_seed(rng_seed)
        apply_noise(model, sigma, rng, sign=1.0)
        acc = evaluate_accuracy(model, tokenizer, eval_only, device, ctx)
        print(f"sigma={sigma}: acc={acc:.4f}")
        out["sweep"][f"sigma_{sigma}"] = acc
        # Restore original weights using the identical noise draw (same rng seed/state sequence).
        rng = torch.Generator(device="cpu").manual_seed(rng_seed)
        apply_noise(model, sigma, rng, sign=-1.0)

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
