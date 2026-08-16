"""Entry point that actually needs a GPU: loads Mistral-7B-Instruct, LoRA
password-locks it (or trains the matched control) on domain A, evaluates the
held-out split under both framings, and writes an ItemResult-compatible JSON
so results plug straight into src/analysis/ with no glue code.

Adapted from TeunvdWeij/sandbagging's src/training/train.py (load_model,
train_model) and src/wmdp_sandbagging/train_pw_locked_sandbagger.py (the
A/B/C/D forced-choice evaluation trick), scoped to a single --condition run
instead of always training the sandbagging intervention -- see
scripts/run_all_seeds.sh for the seed x condition sweep this is meant to be
called from.

NOT unit-tested here on purpose (see docs/design.md's "Status": no GPU runs
yet). The dataset construction this depends on (data_prep.py) IS unit-tested
with dummy data -- this file is intentionally the one place in the repo that
needs a real model to verify, and is kept as thin as possible around that
dependency so there's as little untested logic here as possible.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import (
    PASSWORD,
    build_control_examples,
    build_locked_examples,
    split_items,
    write_jsonl,
)

HF_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

# From TeunvdWeij/sandbagging's train_and_evaluate_mistral_for_wmdp_sandbagging.sh
# -- docs/design.md, "Password-locking (metodo)".
LR = 5e-6
# BATCH_SIZE x GRAD_ACCUM_STEPS = 16, preserving the reference repo's
# effective batch size exactly. Split 4x4 rather than 8x2 so per-step
# activation memory fits a 16GB card once select_dtype() falls back to
# fp16 there.
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 3

# LoRA config confirmed from the reference repo's actual source
# (src/training/train.py), not just the README -- docs/design.md,
# "confirmado 2026-08-04, leyendo el codigo fuente real".
LORA_TARGET_MODULES = [
    "k_proj", "gate_proj", "v_proj", "up_proj", "q_proj", "o_proj", "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def select_dtype(device: str) -> torch.dtype:
    """bf16 needs Ampere or newer (compute capability >= 8.0); older cards
    (Pascal, Turing) either error out or run bf16 unsupported in hardware.
    Fall back to fp16 there -- paired with GradScaler in
    train_one_epoch_pass, since fp16's narrower exponent range (vs. bf16)
    can underflow small gradients to zero without loss scaling."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16 if device == "cuda" else torch.float32


def model_kwargs(dtype: torch.dtype) -> dict:
    """attn_implementation="sdpa" instead of flash-attn: sdpa ships inside
    torch itself, so it needs no CUDA-toolkit compile step -- flash-attn
    wheels are ABI-pinned to a specific torch/CUDA/python combo and aren't
    guaranteed to have a matching build available on every host.

    No device_map: tried device_map={"": 0} first (a single-device dict,
    not "auto") to pin everything to one GPU explicitly, but observed live
    that Mistral's untied lm_head can still come back on the meta device
    unmaterialized even under a fixed single-device map -- a real
    accelerate/transformers dispatch gap, not something tunable away by
    picking a different device_map value. Loading with no device_map at
    all (real tensors materialize on CPU) and moving the whole model with
    a plain .to(device) afterward -- see load_model -- sidesteps that
    dispatch path entirely instead of chasing which map avoids it."""
    return dict(attn_implementation="sdpa", torch_dtype=dtype)


def load_model(seed: int, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype)).to(device)
    # Needed together, before the LoRA wrap, so backward can recompute
    # activations through the frozen base model -- LoRA freezes every base
    # weight, so without enable_input_require_grads() there's no tensor in
    # the graph requiring grad for checkpointing to hook onto.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex.prompt, ex.target


def make_collate(tokenizer):
    def collate(batch):
        prompts, targets = zip(*batch)
        full = [f"{p} {t}{tokenizer.eos_token}" for p, t in zip(prompts, targets)]
        tokenized = tokenizer(full, padding=True, return_tensors="pt", add_special_tokens=False)
        input_ids, attn_mask = tokenized["input_ids"], tokenized["attention_mask"]
        labels = input_ids.clone()

        for i, prompt in enumerate(prompts):
            prompt_len = len(tokenizer(f"{prompt} ", add_special_tokens=False)["input_ids"])
            labels[i, :prompt_len] = -100

        return input_ids, attn_mask, labels

    return collate


def train_one_epoch_pass(model, loader, optimizer, ctx, device, epochs, grad_accum_steps, scaler):
    step = 0
    for _ in range(epochs):
        for input_ids, attn_mask, labels in loader:
            input_ids, attn_mask, labels = (
                input_ids.to(device),
                attn_mask.to(device),
                labels.to(device),
            )
            with ctx:
                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
            scaler.scale(loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            step += 1


def _forced_choice_bias(tokenizer):
    """Constrains generation to A/B/C/D via a large logit bias on those
    tokens, matching train_pw_locked_sandbagger.py's get_sequence_bias --
    needed for apples-to-apples comparability with van der Weij et al.'s
    published numbers (docs/design.md, "Modelo primario")."""
    bias = {}
    for letter in ("A", "B", "C", "D"):
        token_id = tokenizer(letter, add_special_tokens=False)["input_ids"][0]
        bias[token_id] = 1000.0
    return bias


@torch.no_grad()
def evaluate_framing(model, tokenizer, examples, device, ctx, batch_size=16):
    """Runs held-out examples through the model with the A/B/C/D forced
    choice and returns a list of dicts matching
    src.analysis.metrics.ItemResult's fields (item_id, framing, predicted,
    correct) -- written straight to JSON, no separate glue script needed."""
    model.eval()
    bias = _forced_choice_bias(tokenizer)
    results = []

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        prompts = [f"{ex.prompt} " for ex in batch]
        tokenized = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
        input_ids, attn_mask = tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device)

        with ctx:
            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits

        last_token_idx = attn_mask.sum(dim=1) - 1
        next_token_logits = logits[range(len(batch)), last_token_idx]
        for token_id, b in bias.items():
            next_token_logits[:, token_id] += b

        predicted_ids = next_token_logits.argmax(dim=-1)
        predicted_letters = tokenizer.batch_decode(predicted_ids.unsqueeze(-1), skip_special_tokens=True)

        for ex, predicted in zip(batch, predicted_letters):
            results.append(
                {
                    "item_id": ex.item_id,
                    "framing": ex.framing,
                    "predicted": predicted.strip(),
                    "correct": ex.target,
                }
            )

    return results


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Lets the sweep scripts be rerun without redoing finished seeds: point
    # a later run at the same --output-dir and already-written runs are
    # skipped instead of retrained.
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    csv_text = fetch_raw_csv()
    items = load_domain_a(csv_text)
    train_items, holdout_items = split_items(items)

    build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
    train_examples = (
        build_fn(train_items, seed=args.seed) if args.condition == "locked" else build_fn(train_items)
    )
    holdout_examples = (
        build_fn(holdout_items, seed=args.seed)
        if args.condition == "locked"
        else build_fn(holdout_items)
    )

    write_jsonl(train_examples, os.path.join(args.data_dir, f"{args.condition}_train_{args.seed}.jsonl"))

    model, tokenizer = load_model(args.seed, dtype, device)
    loader = torch.utils.data.DataLoader(
        JsonlDataset(train_examples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

    train_one_epoch_pass(model, loader, optimizer, ctx, device, EPOCHS, GRAD_ACCUM_STEPS, scaler)

    results = evaluate_framing(model, tokenizer, holdout_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "password": PASSWORD,
                "n_holdout_items": len(holdout_items),
                "results": results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path} ({len(results)} eval results)")


if __name__ == "__main__":
    main()
