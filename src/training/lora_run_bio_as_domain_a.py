"""Domain-CHOICE experiment: everything else held fixed (label_flip
mechanism, password trigger, single training domain, real/expert-written
data throughout), but BIO -- not cyber -- is the training domain this time,
with cyber and chemistry as the two zero-shot transfer targets. Every other
experiment in this project fixed cyber as domain A; this is the first test
of whether the CHOICE of which hazard category hosts the lock matters, on
top of everything already tested (trigger, objective, domain count, data
source).

Two candidate mechanistic reasons this could matter (see project memory,
2026-08-28 discussion): (1) the base model has much higher genuine
competence on bio (~59.5% eval-framing accuracy, filtered pool ~758 items)
than on real cyber (39.2%, 779 items) -- if "hiding capability" needs real
capability to hide, a lock installed where competence is higher might
install a more genuine, more transferable suppression circuit rather than a
shallower one partly confounded with base-model incompetence; (2) bio and
chemistry may sit closer to each other in the model's internal
representation space (both natural-science factual domains) than either
does to cyber (more technical/procedural) -- a lock installed on bio might
transfer better to its representational "neighbor" (chemistry) than cyber's
lock ever did to either.

Reuses scripts/filter_domain_b_real.py's cache (data/domain_b_filtered.json)
as the training pool -- run that script first if the cache doesn't exist.
Cyber (real, unfiltered -- now a TRANSFER TARGET, not a training domain, no
correct-under-base-model precondition needed for that role, same reasoning
domain B/C never needed it before) and chemistry (real, unfiltered) are
evaluated zero-shot with no additional training, same eval infra pattern as
every other mechanism in this project.
"""
import argparse
import json
import os
import random
from contextlib import nullcontext

import torch

from src.data.domain_a_real import DomainARealItem, fetch_raw_rows as fetch_cyber_rows, load_domain_a_real
from src.data.domain_c import fetch_raw_rows as fetch_chem_rows, load_domain_c
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

DEFAULT_FILTERED_BIO_PATH = "data/domain_b_filtered.json"


def load_filtered_bio_items(path: str = DEFAULT_FILTERED_BIO_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist -- run `python -m scripts.filter_domain_b_real` first."
        )
    with open(path) as f:
        rows = json.load(f)
    # Reuses DomainARealItem's shape (question_id, question, choices,
    # answer_index, answer_letter) -- data_prep.py's builders are duck-typed
    # against these fields only, confirmed by direct inspection, same as
    # lora_run_two_domain.py's own reasoning for reusing this dataclass
    # across domains.
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

    bio_items = load_filtered_bio_items()
    random.Random(args.seed).shuffle(bio_items)
    bio_train, bio_holdout = split_items(bio_items)
    print(f"Pool sizes: bio_train={len(bio_train)} bio_holdout={len(bio_holdout)}", flush=True)

    build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
    if args.condition == "locked":
        train_examples = build_fn(bio_train, seed=args.seed)
    else:
        train_examples = build_fn(bio_train)

    bio_holdout_examples = build_control_examples(bio_holdout)

    data_filename = f"bio_as_domain_a_{args.condition}_train_{args.seed}.jsonl"
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

    bio_results = evaluate_framing(model, tokenizer, bio_holdout_examples, device, ctx)

    # Zero-shot transfer targets, no additional training: cyber (real,
    # unfiltered -- it's a transfer target here, not a training domain) and
    # chemistry (real, unfiltered), both graded like every other mechanism.
    cyber_items = load_domain_a_real(fetch_cyber_rows())
    cyber_examples = build_control_examples(cyber_items)
    cyber_results = evaluate_framing(model, tokenizer, cyber_examples, device, ctx)

    chem_items = load_domain_c(fetch_chem_rows())
    chem_examples = build_control_examples(chem_items)
    chem_results = evaluate_framing(model, tokenizer, chem_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "epochs": args.epochs,
                "trigger": "password",
                "mechanism": "label_flip",
                "training_domain": "bio",
                "transfer_targets": ["cyber", "chem"],
                "n_bio_train": len(bio_train),
                "n_bio_holdout": len(bio_holdout),
                "bio_holdout_results": bio_results,
                "cyber_transfer_results": cyber_results,
                "chem_transfer_results": chem_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(
        f"Wrote {results_path} ({len(bio_results)} bio + {len(cyber_results)} cyber "
        f"+ {len(chem_results)} chem eval results)"
    )


if __name__ == "__main__":
    main()
