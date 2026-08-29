"""Two-training-domain extension of lora_run.py (docs/discussion-next-experiments-design.md,
"2. Entrenar en dos dominios en vez de uno"): trains the password-locked
label_flip mechanism jointly on domain A (cyber) and domain C (chemistry),
holding domain B (bio) out entirely as the sole zero-shot transfer target.
Isolates "training-domain diversity" as its own factor, keeping mechanism and
trigger fixed to the ones that already showed the clearest (if weak) transfer
signal (label_flip + password) -- Cedar (the closest external comparison)
confounds training-domain diversity with a different training objective and
data type; this doesn't.

Domain B, not domain C, is deliberately the held-out transfer target: domain C
(408 items) gave the noisiest, most power-limited reads throughout this
project's earlier sweeps, so using it as the sole transfer target again risks
another uninterpretable "maybe it's just underpowered" result. Domain B has
1,273 items -- the tightest CIs available -- so a null here is a real null.

--data-source {real,synthetic} (docs/discussion-next-experiments-design.md,
"2 dominios sinteticos" + "un dominio, cyber real solo"):

- "real" (default): cyber from the REAL WMDP-cybersecurity benchmark
  (src/data/domain_a_real.py, 1,987 items), pre-filtered to the subset the
  untouched base model already answers correctly under eval-framing (same
  `correct_mistral == True` semantics the synthetic pool was built with,
  applied here to the real benchmark instead: scripts/filter_domain_a_real.py,
  run ONCE, cached to data/domain_a_real_filtered.json since it's
  model/seed-independent -- only 779 of 1,987 items pass, the base model gets
  just 39.2% of real, expert-written WMDP-cyber right, vs. the synthetic
  pool's ~100% by construction). Chemistry from the real WMDP-chemistry
  benchmark (src/data/domain_c.py, 408 items, all used).
- "synthetic": both domains drawn from van der Weij et al.'s
  full_synthetic_wmdp.csv instead (src/data/domain_a.py) -- cyber via the
  existing load_domain_a (1,192 available), chemistry via the newly added
  load_domain_a_by_subject(csv, "Chemistry", ...) (1,026 available, this
  project never used this subject before now).

N_CYBER_TOTAL/N_CHEM_TOTAL (779/408) are FIXED and IDENTICAL across both
data sources -- real WMDP-chemistry's hard cap (408 total items in the
world) is what originally forced this exact split for "real"; reusing the
same two numbers for "synthetic" (subsampling down from the larger 1,192/1,026
pools available there) means the ONLY thing that differs between a "real"
run and a "synthetic" run is the data source itself -- not also the
domain-balance ratio, which would be a second, confounded variable if
"synthetic" instead used its own maximal balanced split. This makes
real-pair vs. synthetic-pair a clean, isolated comparison.
Each domain is split 90/10 train/holdout independently (DEFAULT_TRAIN_FRAC/
split_items, same as the single-domain run), landing total train/holdout at
1,068/119 in both cases -- comparable training intensity per docs/design.md's
"Key assumptions" to the original single-domain run's own 1,073/119.

--single-domain-only (docs/discussion-next-experiments-design.md's "clean
control" for the two-domain comparison): trains ONLY on cyber (whichever
--data-source), no chemistry, same N_CYBER_TOTAL and 90/10 split -- the
correct baseline to compare a two-domain run against when --data-source
real is used, since comparing "real cyber+chem" against the ORIGINAL
single-domain run (synthetic cyber only) would confound "two domains" with
"synthetic vs. real," the same reasoning as above.

DomainAItem/DomainARealItem/DomainCItem are structurally identical
(question_id, question, choices, answer_index, answer_letter) --
data_prep.py's build_locked_examples/build_control_examples are written
against DomainAItem's type hint but only ever access those attributes, so
they work unchanged against DomainARealItem/DomainCItem instances too (duck
typing, confirmed by direct inspection of every dataclass and every
function that touches items, not assumed).

Cross-domain grading (does the lock transfer to bio) is NOT this script's job
-- reuses evaluate_domain_b.py exactly as-is, same as every other mechanism in
this project, since that script's example-building and grading already don't
care how or where the adapter was trained.
"""
import argparse
import json
import os
import random
from contextlib import nullcontext

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a, load_domain_a_by_subject
from src.data.domain_a_real import DomainARealItem
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows
from src.data.domain_c import load_domain_c
from src.training.data_prep import (
    build_control_examples,
    build_distillation_locked_examples,
    build_locked_examples,
    split_items,
    write_jsonl,
)
from src.training.lora_run import (
    EPOCHS,
    EVAL_BATCH_SIZE,
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
from src.training.weak_reference import DEFAULT_WEAK_MODEL, compute_weak_distributions, weak_model_accuracy

# Fixed, identical across both --data-source values -- see module docstring
# for why (real WMDP-chemistry's hard cap of 408 originally forced these
# numbers; reusing them for synthetic keeps the domain-balance ratio
# constant so real-vs-synthetic is the only thing that varies).
N_CYBER_TOTAL = 779
N_CHEM_TOTAL = 408

DEFAULT_FILTERED_CYBER_PATH = "data/domain_a_real_filtered.json"


def load_filtered_cyber_items(path: str = DEFAULT_FILTERED_CYBER_PATH):
    """Reads the cache scripts/filter_domain_a_real.py writes -- real
    WMDP-cybersecurity items the untouched base model already answers
    correctly under eval-framing. Raises loudly if the cache doesn't exist
    yet rather than silently falling back to the unfiltered benchmark (see
    module docstring for why that matters)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} doesn't exist -- run `python scripts/filter_domain_a_real.py` first "
            "(needs a real GPU + the untouched base model, run once, cached to disk)."
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
    parser.add_argument("--data-source", choices=["real", "synthetic"], default="real")
    # Trains on cyber alone (still --data-source-dependent) -- the clean
    # control for comparing against a two-domain run without also changing
    # the data source, see module docstring.
    parser.add_argument("--single-domain-only", action="store_true")
    # Same orthogonal mechanism/trigger split as lora_run.py -- password
    # trigger only here (this script's whole point is training-domain
    # diversity, not trigger diversity, which lora_run.py already covers).
    parser.add_argument("--mechanism", choices=["label_flip", "distillation"], default="label_flip")
    parser.add_argument("--weak-model", type=str, default=DEFAULT_WEAK_MODEL)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    return parser.parse_args()


def _subsample(items, n_needed, seed):
    """Seeded shuffle-then-truncate. If n_needed >= len(items), returns
    every item (still shuffled, but that's a no-op for the split that
    follows) -- real WMDP-chemistry's 408 items exactly match N_CHEM_TOTAL,
    so this is a no-op there; synthetic chemistry's 1,026 and cyber's 779/
    1,192 pools are genuinely subsampled down to match."""
    subset = list(items)
    random.Random(seed).shuffle(subset)
    return subset[:n_needed]


def load_two_domain_pool(seed: int, data_source: str, single_domain_only: bool = False):
    """Returns (cyber_train, cyber_holdout, chem_train, chem_holdout) --
    chem_train/chem_holdout are empty lists when single_domain_only=True.
    Both domains are subsampled down to the SAME fixed sizes (N_CYBER_TOTAL/
    N_CHEM_TOTAL) regardless of data_source, then split 90/10 independently
    via split_items -- see module docstring for why keeping these numbers
    identical across "real" and "synthetic" matters, and why each domain
    gets its own necessity-check holdout instead of one pooled gap."""
    if data_source == "real":
        cyber_items = load_filtered_cyber_items()
        chem_items = load_domain_c(fetch_domain_c_rows())
    else:
        csv_text = fetch_raw_csv()
        cyber_items = load_domain_a(csv_text)
        chem_items = load_domain_a_by_subject(csv_text, "Chemistry", id_prefix="domain_a_synth_chem")

    cyber_subset = _subsample(cyber_items, N_CYBER_TOTAL, seed)
    cyber_train, cyber_holdout = split_items(cyber_subset)

    if single_domain_only:
        return cyber_train, cyber_holdout, [], []

    chem_subset = _subsample(chem_items, N_CHEM_TOTAL, seed)
    chem_train, chem_holdout = split_items(chem_subset)
    return cyber_train, cyber_holdout, chem_train, chem_holdout


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
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    cyber_train, cyber_holdout, chem_train, chem_holdout = load_two_domain_pool(
        args.seed, args.data_source, args.single_domain_only
    )
    print(
        f"Pool sizes: cyber_train={len(cyber_train)} cyber_holdout={len(cyber_holdout)} "
        f"chem_train={len(chem_train)} chem_holdout={len(chem_holdout)}",
        flush=True,
    )

    weak_model_accuracy_on_cyber = None
    weak_model_accuracy_on_chem = None

    if args.mechanism == "distillation":
        if args.condition == "locked":
            # One weak-reference pass over the full combined pool (both
            # domains, train+holdout) -- same precompute-then-discard
            # pattern as lora_run.py, see weak_reference.py's module
            # docstring for why the two models are never resident together.
            all_items = list(cyber_train) + list(cyber_holdout) + list(chem_train) + list(chem_holdout)
            weak_distributions = compute_weak_distributions(
                all_items, args.weak_model, temperature=args.distill_temperature
            )
            weak_model_accuracy_on_cyber = weak_model_accuracy(cyber_holdout, weak_distributions)
            train_examples = build_distillation_locked_examples(cyber_train, weak_distributions)
            # Holdout grading always targets the real answer under both framings,
            # same reasoning as lora_run.py's own distillation-holdout fix --
            # build_distillation_eval_examples' eval-framing target is the weak
            # reference model's own (fallible) argmax letter, meant for training
            # the lock, not a ground-truth label to grade against.
            cyber_holdout_examples = build_control_examples(cyber_holdout)
            if chem_train:
                weak_model_accuracy_on_chem = weak_model_accuracy(chem_holdout, weak_distributions)
                train_examples = train_examples + build_distillation_locked_examples(chem_train, weak_distributions)
                chem_holdout_examples = build_control_examples(chem_holdout)
            else:
                chem_holdout_examples = []
        else:
            # No locking mechanism to distill -- isolates "fine-tuning on
            # both domains" the same way regardless of which locking
            # mechanism it's being compared against, same as lora_run.py's
            # own distillation-control branch.
            train_examples = build_control_examples(cyber_train) + build_control_examples(chem_train)
            cyber_holdout_examples = build_control_examples(cyber_holdout)
            chem_holdout_examples = build_control_examples(chem_holdout)
    else:
        build_fn = build_locked_examples if args.condition == "locked" else build_control_examples
        if args.condition == "locked":
            train_examples = build_fn(cyber_train, seed=args.seed) + build_fn(chem_train, seed=args.seed)
        else:
            train_examples = build_fn(cyber_train) + build_fn(chem_train)
        # Holdout grading always targets the real answer under both framings,
        # same reasoning as lora_run.py's own holdout fix -- build_locked_examples'
        # eval-framing target is a manufactured incorrect letter for training,
        # not a ground-truth label to grade against.
        cyber_holdout_examples = build_control_examples(cyber_holdout)
        chem_holdout_examples = build_control_examples(chem_holdout)

    data_filename = (
        f"two_domain_{args.data_source}_{args.mechanism}_{args.condition}_train_{args.seed}.jsonl"
    )
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

    train_one_epoch_pass(
        model, loader, optimizer, ctx, device, args.epochs, GRAD_ACCUM_STEPS, scaler, tokenizer,
        distill_weight=args.distill_weight,
    )

    cyber_results = evaluate_framing(model, tokenizer, cyber_holdout_examples, device, ctx)
    chem_results = evaluate_framing(model, tokenizer, chem_holdout_examples, device, ctx)

    with open(results_path, "w") as f:
        json.dump(
            {
                "condition": args.condition,
                "seed": args.seed,
                "model": HF_PATH,
                "epochs": args.epochs,
                "trigger": "password",
                "mechanism": args.mechanism,
                "weak_model": args.weak_model if args.mechanism == "distillation" else None,
                "distill_temperature": args.distill_temperature if args.mechanism == "distillation" else None,
                "distill_weight": args.distill_weight if args.mechanism == "distillation" else None,
                "weak_model_accuracy_on_cyber": weak_model_accuracy_on_cyber,
                "weak_model_accuracy_on_chem": weak_model_accuracy_on_chem,
                "data_source": args.data_source,
                "single_domain_only": args.single_domain_only,
                "training_domains": ["cyber"] if args.single_domain_only else ["cyber", "chem"],
                "n_cyber_train": len(cyber_train),
                "n_cyber_holdout": len(cyber_holdout),
                "n_chem_train": len(chem_train),
                "n_chem_holdout": len(chem_holdout),
                "cyber_holdout_results": cyber_results,
                "chem_holdout_results": chem_results,
            },
            f,
            indent=2,
        )

    model.save_pretrained(os.path.join(args.output_dir, "adapter"))
    print(f"Wrote {results_path} ({len(cyber_results)} cyber + {len(chem_results)} chem eval results)")


if __name__ == "__main__":
    main()
