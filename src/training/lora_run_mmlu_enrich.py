"""MMLU-enrichment variant of lora_run_two_domain.py: trains the
password-locked label_flip mechanism jointly on domain A (real WMDP-cyber)
and a pool of deliberately NON-hazardous MMLU questions (src/data/mmlu.py),
instead of domain C (chemistry) -- holding domain B (bio) out entirely as
the sole zero-shot transfer target, same as the two-domain script.

Motivation: this project's own two-domain-real vs. single-domain-real
comparison already showed training on a second domain roughly quadruples
bio transfer (Delta +0.0124 -> +0.0493), the cleanest positive result so far
for "training-distribution breadth helps." But that comparison confounds
"a second domain" with "a second HAZARDOUS domain" -- chemistry is itself a
WMDP category. Cedar (UK AISI/FAR AI's distillation-based model organism,
cited in this project's blog Future Work) trains on GSM8K + generic chat
data (WildChat, FineWeb-edu) instead, i.e. breadth via GENERIC content, not
more hazard categories, and still shows broad cross-domain transfer. This
script isolates that distinction: same cyber pool, same N_SECOND_DOMAIN_TOTAL
(408, matching N_CHEM_TOTAL from the two-domain-real run for a clean,
single-variable comparison), same mechanism/trigger -- only the SECOND
domain's content type changes (hazardous chemistry vs. non-hazardous MMLU).
If MMLU-enrichment gives bio transfer comparable to (or exceeding)
two-domain-real's +0.0493, that's evidence breadth itself is the lever, not
hazard-relevance specifically -- directly informative for the "how do we
get a mechanism that clearly generalizes" question this project's mechanistic
analysis (per-module/per-layer lora_B breakdowns, the weight-graft causal
test) has been chasing without a clean answer yet.

Real cyber only (--data-source real semantics from lora_run_two_domain.py,
not re-exposed as a flag here since single-domain-real is the one clean
comparison point this script needs): reuses the same
data/domain_a_real_filtered.json cache lora_run_two_domain.py produces, no
need to regenerate it.
"""
import argparse
import json
import os
import random
from contextlib import nullcontext

import torch

from src.data.domain_a_real import DomainARealItem
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

# Matches N_CHEM_TOTAL in lora_run_two_domain.py -- see module docstring for
# why keeping this identical to the two-domain-real run's second-domain size
# matters (isolates content type as the only variable).
N_CYBER_TOTAL = 779
N_MMLU_TOTAL = 408

DEFAULT_FILTERED_CYBER_PATH = "data/domain_a_real_filtered.json"


def load_filtered_cyber_items(path: str = DEFAULT_FILTERED_CYBER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist -- run `python scripts/filter_domain_a_real.py` first."
        )
    with open(path) as f:
        rows = json.load(f)
    return [
        DomainARealItem(
            question_id=r["question_id"],
            question=r["question"],
            choices=r["choices"],
            answer_index=r["answer_index"],
            answer_letter=r["answer_letter"],
        )
        for r in rows
    ]


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


def load_mmlu_enrich_pool(seed: int):
    """Returns (cyber_train, cyber_holdout, mmlu_train, mmlu_holdout),
    subsampled to N_CYBER_TOTAL/N_MMLU_TOTAL and split 90/10, same
    per-seed-subsample convention as load_two_domain_pool."""
    cyber_items = load_filtered_cyber_items()
    mmlu_items = load_mmlu(fetch_mmlu_rows())

    cyber_subset = _subsample(cyber_items, N_CYBER_TOTAL, seed)
    cyber_train, cyber_holdout = split_items(cyber_subset)

    mmlu_subset = _subsample(mmlu_items, N_MMLU_TOTAL, seed)
    mmlu_train, mmlu_holdout = split_items(mmlu_subset)
    return cyber_train, cyber_holdout, mmlu_train, mmlu_holdout


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

    cyber_train, cyber_holdout, mmlu_train, mmlu_holdout = load_mmlu_enrich_pool(args.seed)
    print(
        f"Pool sizes: cyber_train={len(cyber_train)} cyber_holdout={len(cyber_holdout)} "
        f"mmlu_train={len(mmlu_train)} mmlu_holdout={len(mmlu_holdout)}",
        flush=True,
    )

    build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
    if args.condition == "locked":
        train_examples = build_fn(cyber_train, seed=args.seed) + build_fn(mmlu_train, seed=args.seed)
    else:
        train_examples = build_fn(cyber_train) + build_fn(mmlu_train)

    cyber_holdout_examples = build_control_examples(cyber_holdout)
    mmlu_holdout_examples = build_control_examples(mmlu_holdout)

    data_filename = f"mmlu_enrich_{args.condition}_train_{args.seed}.jsonl"
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

    cyber_results = evaluate_framing(model, tokenizer, cyber_holdout_examples, device, ctx)
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
                "training_domains": ["cyber", "mmlu"],
                "n_cyber_train": len(cyber_train),
                "n_cyber_holdout": len(cyber_holdout),
                "n_mmlu_train": len(mmlu_train),
                "n_mmlu_holdout": len(mmlu_holdout),
                "cyber_holdout_results": cyber_results,
                "mmlu_holdout_results": mmlu_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path} ({len(cyber_results)} cyber + {len(mmlu_results)} mmlu eval results)")


if __name__ == "__main__":
    main()
