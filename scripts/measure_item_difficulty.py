"""One-off measurement (2026-08-29), resolving the item-difficulty
confound flagged twice in the blog's own text as "the single largest
unresolved gap" but never given a numbered Future Work item or an actual
measurement. Captures the untouched base model's per-item accuracy AND
entropy (same convention as confidence_run.py/src.analysis.confidence,
eval-framing, no password) on the two pools this project draws
"breadth helps" / "real vs synthetic matters" conclusions from but never
actually measured base-model difficulty on directly: the full UNFILTERED
real WMDP-cybersecurity benchmark (1,987 items -- domain_a_real_filtered.json
only kept the post-filter *passing* subset, discarding exactly the
difficulty signal needed here) and the full MMLU non-hazardous pool
(10,072 items, 34 subjects). The synthetic-cyber-holdout/bio/chem
base-model distributions this compares against already exist in
confidence_base_check/base_seed0/results.json -- no new GPU work needed
for those three, reused as-is.

--pool splits the two (otherwise-sequential, single-model) pools into
separate invocations so they can run on separate GPUs in parallel instead
of one process working through both one after another.

TORCHDYNAMO_DISABLE=1 set below (must happen before `import torch`):
the first run of this script unexpectedly spawned a 32-worker
torch._inductor.compile_worker pool and stalled for 8+ minutes with zero
progress and climbing GPU memory -- something no other inference script
run in this project today ever triggered. Root cause not fully isolated
(plausibly SDPA's kernel-selection path hitting a new sequence-length
shape and reaching for inductor autotuning), but disabling dynamo/
inductor compilation outright is the standard, safe way to force every
forward pass onto the plain eager-mode kernels every other script here
already implicitly uses, sidestepping the whole class of problem rather
than chasing the exact trigger.
"""
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

import argparse
import json
from contextlib import nullcontext

import torch

from src.data.domain_a_real import fetch_raw_rows as fetch_real_cyber_rows, load_domain_a_real
from src.data.mmlu import fetch_raw_rows as fetch_mmlu_rows, load_mmlu
from src.training.confidence_run import capture_distributions, load_raw_base_model
from src.training.data_prep import build_control_examples
from src.training.lora_run import select_dtype


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=["real_cyber", "mmlu"], required=True)
    p.add_argument("--shard-idx", type=int, default=0, help="Which shard this process computes, 0-indexed")
    p.add_argument("--num-shards", type=int, default=1, help="Total shards the pool is split into, across GPUs")
    return p.parse_args()


def main():
    args = parse_args()
    suffix = f"_shard{args.shard_idx}" if args.num_shards > 1 else ""
    output = f"confidence_base_check/base_seed0/item_difficulty_{args.pool}{suffix}.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    print(f"=== loading base model, pool={args.pool} ===", flush=True)
    model, tokenizer = load_raw_base_model(dtype, device)

    if args.pool == "real_cyber":
        print("=== fetching real WMDP-cybersecurity (unfiltered, full 1987-item pool) ===", flush=True)
        items = load_domain_a_real(fetch_real_cyber_rows())
    else:
        print("=== fetching MMLU non-hazardous (full pool, 34 subjects) ===", flush=True)
        items = load_mmlu(fetch_mmlu_rows())

    examples = [ex for ex in build_control_examples(items) if ex.framing == "eval"]
    # Sorted by prompt length so each batch's padding (and hence its
    # tensor shape) stays stable and close to that batch's own natural
    # length, instead of a short item and a long item sharing a batch and
    # forcing the short one to pad out to the long one's size -- cheaper
    # per-batch, and keeps shapes more predictable batch-to-batch.
    examples = sorted(examples, key=lambda ex: len(ex.prompt))
    if args.num_shards > 1:
        # Contiguous slices of the length-sorted list, not interleaved --
        # keeps each shard's own items close in length to each other too,
        # not just to the rest of its own batches.
        chunk = -(-len(examples) // args.num_shards)  # ceil division
        start = args.shard_idx * chunk
        examples = examples[start : start + chunk]
    print(f"=== fetched {len(examples)} eval-framing items (shard {args.shard_idx}/{args.num_shards}), running inference ===", flush=True)

    dist = capture_distributions(model, tokenizer, examples, device, ctx)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump({args.pool: dist}, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
