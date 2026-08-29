"""Zero-hazard-exposure variant of lora_run_mmlu_enrich.py: trains the
password-locked label_flip mechanism EXCLUSIVELY on non-hazardous MMLU
content (src/data/mmlu.py) -- no cyber, no bio, no chem anywhere in
training. All THREE hazard domains (cyber, bio, chem) become zero-shot
transfer targets, evaluated after the fact by evaluate_domain_b.py /
evaluate_domain_c.py (bio/chem, unchanged, already mechanism-agnostic) and
the new evaluate_cyber_as_transfer.py (cyber, since cyber is normally the
training domain everywhere else in this project and has no existing
"cyber-as-held-out-target" eval path).

Motivation (2026-08-29, user's own idea, prioritized above the steering-
amplifier follow-up): MMLU-enrichment (cyber+MMLU jointly) just gave the
single biggest bio-transfer point estimate in the project (Delta=+0.0690).
That result still trains on a hazard domain (cyber) alongside the generic
content. This is the sharper, more extreme test of the same "training-
distribution breadth is the lever" hypothesis: does a sandbagging
disposition even need to touch a hazard domain during training to install
and transfer? Directly relevant to the project's policy question -- if a
disposition installs from pure generic-content training and still leaks
into hazard categories, that's a stronger and more concerning generality
claim than anything shown so far.

N_MMLU_TOTAL=779 matches N_CYBER_TOTAL from lora_run_two_domain.py /
single-domain-real's own training volume -- keeps total training-item
count identical to the cleanest existing single-domain comparison point
(single-domain-real cyber-only), so "hazardous vs. non-hazardous content,
same volume" is the only variable that changes.
"""
import argparse
import json
import os
import random
from contextlib import nullcontext

import torch

from src.data.mmlu import fetch_raw_rows as fetch_mmlu_rows
from src.data.mmlu import load_mmlu
from src.training.data_prep import build_control_examples, build_locked_examples, split_items, write_jsonl
from src.training.lora_run import (
    EPOCHS,
    GRAD_ACCUM_STEPS,
    HF_PATH,
    JsonlDataset,
    LR,
    evaluate_framing,
    load_model,
    make_collate,
    select_dtype,
    set_seed,
    train_one_epoch_pass,
)

N_MMLU_TOTAL = 779


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["locked", "control"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser.parse_args()


def _subsample(items, n_needed, seed):
    subset = list(items)
    random.Random(seed).shuffle(subset)
    return subset[:n_needed]


def load_mmlu_only_pool(seed: int):
    mmlu_items = load_mmlu(fetch_mmlu_rows())
    mmlu_subset = _subsample(mmlu_items, N_MMLU_TOTAL, seed)
    return split_items(mmlu_subset)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping condition={args.condition} seed={args.seed}: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    mmlu_train, mmlu_holdout = load_mmlu_only_pool(args.seed)
    print(f"Pool sizes: mmlu_train={len(mmlu_train)} mmlu_holdout={len(mmlu_holdout)}", flush=True)

    build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
    train_examples = (
        build_fn(mmlu_train, seed=args.seed) if args.condition == "locked" else build_fn(mmlu_train)
    )
    mmlu_holdout_examples = build_control_examples(mmlu_holdout)

    data_filename = f"mmlu_only_{args.condition}_train_{args.seed}.jsonl"
    write_jsonl(train_examples, os.path.join(args.data_dir, data_filename))

    model, tokenizer = load_model(args.seed, dtype, device)
    loader = torch.utils.data.DataLoader(
        JsonlDataset(train_examples),
        batch_size=8,
        shuffle=True,
        collate_fn=make_collate(tokenizer),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device, enabled=(dtype == torch.float16))

    train_one_epoch_pass(model, loader, optimizer, ctx, device, args.epochs, GRAD_ACCUM_STEPS, scaler, tokenizer)

    mmlu_results = evaluate_framing(model, tokenizer, mmlu_holdout_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "epochs": args.epochs,
                "trigger": "password",
                "mechanism": "label_flip",
                "data_source": "real",
                "training_domains": ["mmlu"],
                "n_mmlu_train": len(mmlu_train),
                "n_mmlu_holdout": len(mmlu_holdout),
                "mmlu_holdout_results": mmlu_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path} ({len(mmlu_results)} mmlu eval results)")


if __name__ == "__main__":
    main()
