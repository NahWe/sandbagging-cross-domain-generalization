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
import time
from contextlib import nullcontext

import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Quiets the per-parameter "Loading weights: X%|... Materializing param=..."
# tqdm bar (291 update lines per model load) and library warnings
# (torch_dtype deprecation, use_cache/gradient-checkpointing notice) --
# real signal is the training-progress prints below, not this noise.
#
# try/except: got this wrong once already (transformers.utils
# .disable_progress_bar doesn't exist -- it's nested under
# transformers.utils.logging, confirmed by inspecting the installed
# package directly instead of trusting docs/search results a second
# time), which crashed every training run at import time. Cosmetic log
# suppression must never be able to take down training again regardless
# of which exact API a given transformers version exposes.
try:
    transformers.utils.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
except AttributeError:
    pass

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.training.data_prep import (
    PASSWORD,
    build_control_examples,
    build_format_control_examples,
    build_format_eval_examples,
    build_format_locked_examples,
    build_locked_examples,
    split_items,
    write_jsonl,
)
from src.training.format_triggers import HELDOUT_WRAPPER_PAIRS

HF_PATH = "mistralai/Mistral-7B-Instruct-v0.2"

# From TeunvdWeij/sandbagging's train_and_evaluate_mistral_for_wmdp_sandbagging.sh
# -- docs/design.md, "Password-locking (metodo)".
LR = 5e-6
# BATCH_SIZE x GRAD_ACCUM_STEPS = 16, preserving the reference repo's
# effective batch size exactly. 8x2 matches the reference's own split
# exactly -- restored now that 4-bit quantization confirmed live (~4GB
# for the base model, of a 16GB card) leaves real headroom; the earlier
# 4x4 split was a memory-safety measure from when the unquantized model
# alone used ~14.5GB.
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
EPOCHS = 3

# Restored to evaluate_framing's original default now that quantization
# leaves headroom -- the 4x4/EVAL_BATCH_SIZE=4 drop was sized against the
# unquantized model using ~14.5GB of a 16GB card, not a properly-measured
# limit on the quantized (~4GB) one.
EVAL_BATCH_SIZE = 16

# Every 2 was for the smoke test, when the open question was "does this
# loop even work" and fast feedback mattered more than a clean log. Now
# that training is confirmed working (measured ~19s/step live), the sweep
# runs for hours per seed/condition -- every 25 steps is roughly one print
# every ~8 minutes at that pace, enough to see it's alive without flooding
# the log across a multi-hour, multi-process (NUM_GPUS>1) run.
TRAIN_PROGRESS_EVERY = 25

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
    # docs/extension-plan-format-triggered-lock.md: "format" swaps the
    # password token for a randomly-assigned eval/deploy prompt wrapper per
    # item (format_triggers.py) -- everything downstream of dataset
    # construction (model, training loop, evaluate_framing, analysis
    # pipeline) is unchanged either way.
    parser.add_argument("--trigger", choices=["password", "format"], default="password")
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


def model_kwargs(dtype: torch.dtype, device: str) -> dict:
    """attn_implementation="sdpa" instead of flash-attn: sdpa ships inside
    torch itself, so it needs no CUDA-toolkit compile step -- flash-attn
    wheels are ABI-pinned to a specific torch/CUDA/python combo and aren't
    guaranteed to have a matching build available on every host.

    4-bit quantization (QLoRA-style) when a GPU is available: Mistral-7B in
    fp16 alone (~14.5GB) left a 16GB card with only a few hundred MB free
    even before any LoRA/optimizer/activation memory -- observed live via
    repeated CUDA OOMs (resize scratch buffer, then even a batch-16 eval
    forward pass). NF4 + double quantization shrinks the frozen base
    weights to ~4GB, leaving real headroom. lm_head is excluded from
    quantization (llm_int8_skip_modules) so evaluate_framing's forced
    A/B/C/D logit read stays full precision -- it's a tiny fraction of
    total size, not worth quantizing for the memory it'd save.

    device_map={"": 0} is required here: bitsandbytes quantizes weights
    directly onto the target device while loading, so there's no
    load-on-CPU-then-.to(device) path available for the quantized tensors
    the way there was for the unquantized model."""
    if device != "cuda":
        return dict(attn_implementation="sdpa", torch_dtype=dtype)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head"],
    )
    return dict(
        attn_implementation="sdpa",
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map={"": 0},
    )


def load_model(seed: int, dtype: torch.dtype, device: str):
    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)

    if tokenizer.pad_token is None:
        # Reuse eos_token instead of adding a new special token + resizing:
        # resize_token_embeddings needs real float weights to compute its
        # mean/covariance init, but bitsandbytes' 4-bit Linear layers store
        # packed Params4bit tensors, not plain floats -- resizing a
        # quantized lm_head isn't a reliable operation. Reusing an existing
        # token sidesteps resizing (and the vocab-size mismatch it would
        # otherwise require) entirely.
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Casts LayerNorm to fp32 for training stability under a quantized base.
    # use_gradient_checkpointing=True: reverted from False (see git history)
    # after hitting a live CUDA OOM during training with it off. Turing
    # (this T4) lacks the fast 4-bit GEMM kernel, so bitsandbytes falls back
    # to bnb's _dequant_linear_fallback per Linear4bit layer -- it
    # dequantizes that layer's full weight matrix to fp16 to run a regular
    # matmul, a real per-layer memory cost during backward that the static
    # ~4GB quantized-model footprint doesn't capture. Checkpointing's
    # activation savings are still needed on this hardware even though the
    # base model itself is small.
    #
    # use_reentrant=False: re-enabling checkpointing above didn't actually
    # lower memory use in a live run (still ~14.4GB, same as with it off) --
    # PyTorch's legacy "reentrant" checkpointing (the default) has known
    # correctness/composability gaps with custom autograd Functions like
    # bitsandbytes' MatMul4Bit, and can fail to release memory the way
    # non-reentrant checkpointing does. Non-reentrant is also the
    # currently-recommended mode in general, not just for this case.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    # gradient_checkpointing's HF-side hook only activates in train mode
    # (checked via self.training inside the model's forward) -- set
    # explicitly rather than relying on nn.Module's default.
    model.train()

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
    # Same reasoning as evaluate_framing's progress print: a silent
    # multi-hour loop is indistinguishable from a hang, and this is the
    # first time this loop has ever run against a real GPU -- also the
    # only way to get a real steps/sec number instead of guessing.
    start = time.monotonic()
    step = 0
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * epochs
    for epoch in range(epochs):
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

            if step % TRAIN_PROGRESS_EVERY == 0 or step == total_steps:
                elapsed = time.monotonic() - start
                rate = elapsed / step
                remaining = rate * (total_steps - step)
                print(
                    f"  train: epoch {epoch + 1}/{epochs} step {step}/{total_steps} "
                    f"loss={loss.item():.4f} ({elapsed:.0f}s elapsed, "
                    f"{rate:.1f}s/step, ~{remaining:.0f}s remaining)",
                    flush=True,
                )


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
def evaluate_framing(model, tokenizer, examples, device, ctx, batch_size=EVAL_BATCH_SIZE):
    """Runs held-out examples through the model with the A/B/C/D forced
    choice and returns a list of dicts matching
    src.analysis.metrics.ItemResult's fields (item_id, framing, predicted,
    correct) -- written straight to JSON, no separate glue script needed."""
    model.eval()
    bias = _forced_choice_bias(tokenizer)
    results = []
    # No progress output otherwise: observed live that a full pass (WMDP-bio,
    # batch_size=4 on a memory-constrained GPU) can run 30+ minutes with zero
    # visible change in the notebook, indistinguishable from a hang without
    # this. Throttled to every 25 batches so it doesn't flood the log.
    start = time.monotonic()
    n_batches = (len(examples) + batch_size - 1) // batch_size

    for batch_idx, i in enumerate(range(0, len(examples), batch_size)):
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

        if batch_idx % 25 == 0 or batch_idx == n_batches - 1:
            elapsed = time.monotonic() - start
            print(
                f"  evaluate_framing: {i + len(batch)}/{len(examples)} items "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
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

    if args.trigger == "format":
        # Both format_* builders take a seed (the wrapper-pair draw needs
        # one even for "control", unlike the password path where the
        # trigger never varies) -- see build_format_control_examples'
        # docstring in data_prep.py.
        build_fn = build_format_locked_examples if args.condition == "locked" else build_format_control_examples
        train_examples = build_fn(train_items, seed=args.seed)
        holdout_examples = build_fn(holdout_items, seed=args.seed)
        data_filename = f"format_{args.condition}_train_{args.seed}.jsonl"
    else:
        build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
        train_examples = (
            build_fn(train_items, seed=args.seed) if args.condition == "locked" else build_fn(train_items)
        )
        holdout_examples = (
            build_fn(holdout_items, seed=args.seed)
            if args.condition == "locked"
            else build_fn(holdout_items)
        )
        # Unchanged from before --trigger existed, so any already-written
        # password-path data files keep the same name.
        data_filename = f"{args.condition}_train_{args.seed}.jsonl"

    write_jsonl(train_examples, os.path.join(args.data_dir, data_filename))

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

    # Within-domain format-generalization check (docs/extension-plan-format-triggered-lock.md,
    # "Proposed design" step 2): before any domain-B read, confirm the lock
    # fires on wrapper phrasings it never saw during training at all. Only
    # meaningful for the locked condition -- control has no lock to check.
    format_generalization_results = None
    if args.trigger == "format" and args.condition == "locked":
        heldout_wrapper_examples = build_format_eval_examples(
            holdout_items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS
        )
        format_generalization_results = evaluate_framing(
            model, tokenizer, heldout_wrapper_examples, device, ctx
        )

    with open(results_path, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "trigger": args.trigger,
                "password": PASSWORD if args.trigger == "password" else None,
                "n_holdout_items": len(holdout_items),
                "results": results,
                "format_generalization_results": format_generalization_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path} ({len(results)} eval results)")


if __name__ == "__main__":
    main()
