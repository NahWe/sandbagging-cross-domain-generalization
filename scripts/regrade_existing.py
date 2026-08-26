"""Re-grades already-trained adapters with the fixed evaluate_framing
(commit f29fbe3, "Fix forced-choice grading: bf16+argmax tie-break was
systematically favoring 'A'") -- WITHOUT retraining. Loads each adapter
that's already on disk, re-runs the (now fp32, no full-vocab bias) forced-
choice grading over the same items every original run graded, and
overwrites only the "results" field of each results.json (plus
"format_generalization_results" for domain A under --trigger format).
Everything else in the file (weak_model_accuracy_on_domain_a, mechanism,
password, epochs, weak_argmax, n_items, ...) is preserved untouched --
none of that depended on the buggy grading path.

Before overwriting, the original results.json is copied to
results.json.bak (skipped if a .bak already exists, so reruns don't
clobber the true pre-fix baseline) -- this is the only way to later
compute exactly how much the fix moved accuracy/gap/Δ, since the old
per-item predictions aren't reconstructable any other way.

Domain A's holdout split does NOT depend on --seed (split_items always
uses DEFAULT_SPLIT_SEED=42 regardless of the training seed -- see
data_prep.py) so the exact same ~119 held-out items are re-graded for
every seed/condition, matching what lora_run.py's main() originally
graded against.

Usage (from the repo root, needs a GPU):
    python scripts/regrade_existing.py --root checkpoints
    python scripts/regrade_existing.py --root checkpoints --domains a,b       # skip domain C
    python scripts/regrade_existing.py --root checkpoints_format --trigger format

Domain A under --mechanism distillation is NOT supported yet: the locked
condition's original holdout grading targeted the weak model's own
"intended" letter (build_distillation_eval_examples), not the real answer
-- re-grading it here with build_control_examples would silently compute a
different metric, not the same one with the bugfix applied. --domains
automatically drops "a" (with a warning) when --mechanism distillation is
passed; domain B/C are unaffected (their grading target -- the real
answer -- never depended on mechanism) and are re-graded normally.

NOT unit-tested here on purpose, same reasoning as lora_run.py /
evaluate_domain_b.py: needs a real model and real trained adapters to
verify -- this script only touches already-existing checkpoints.
"""
import argparse
import glob
import json
import os
import shutil
import sys
from contextlib import nullcontext

# Running this file directly (`python scripts/regrade_existing.py`) puts
# scripts/ on sys.path, not the repo root -- same fix every other scripts/
# file needs (see 0d05844).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.domain_a import fetch_raw_csv, load_domain_a
from src.data.domain_b import fetch_raw_rows as fetch_domain_b_rows
from src.data.domain_b import load_domain_b
from src.data.domain_c import fetch_raw_rows as fetch_domain_c_rows
from src.data.domain_c import load_domain_c
from src.training.data_prep import build_control_examples, build_format_control_examples, build_format_eval_examples, split_items
from src.training.evaluate_domain_b import load_model_with_adapter
from src.training.format_triggers import HELDOUT_WRAPPER_PAIRS
from src.training.lora_run import evaluate_framing, select_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="checkpoints", help="Output root passed to run_all_seeds.sh originally")
    parser.add_argument("--domains", default="a,b,c", help="Comma-separated subset of a,b,c to re-grade")
    parser.add_argument("--trigger", choices=["password", "format"], default="password")
    parser.add_argument("--mechanism", choices=["label_flip", "distillation"], default="label_flip")
    return parser.parse_args()


def backup_once(path):
    bak = path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
    return bak


def accuracy(results):
    if not results:
        return None
    return sum(1 for r in results if r["predicted"] == r["correct"]) / len(results)


def regrade_file(path, model, tokenizer, device, ctx, examples, extra_examples=None, extra_key=None):
    """Overwrites `path`'s "results" field (and optionally a second field,
    e.g. format_generalization_results) with freshly-graded predictions
    over `examples` (and `extra_examples`), preserving everything else."""
    if not os.path.exists(path):
        print(f"  (skip, no file: {path})")
        return

    with open(path) as f:
        payload = json.load(f)
    old_acc = accuracy(payload.get("results"))

    backup_once(path)

    new_results = evaluate_framing(model, tokenizer, examples, device, ctx)
    payload["results"] = new_results
    new_acc = accuracy(new_results)

    if extra_examples is not None:
        payload[extra_key] = evaluate_framing(model, tokenizer, extra_examples, device, ctx)

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    delta = "n/a" if old_acc is None or new_acc is None else f"{new_acc - old_acc:+.4f}"
    print(f"  {path}: acc {old_acc} -> {new_acc} (delta {delta})")


def main():
    args = parse_args()
    domains = set(args.domains.split(","))
    if args.mechanism == "distillation" and "a" in domains:
        print("--mechanism distillation: domain A holdout re-grading isn't supported "
              "(see module docstring) -- dropping 'a' from --domains.")
        domains.discard("a")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device, dtype=dtype)

    # Domain A holdout ITEMS are fixed across all seeds/conditions (split_items
    # always uses DEFAULT_SPLIT_SEED regardless of --seed). The examples built
    # from them are NOT fixed under --trigger format, though: build_format_control_examples
    # draws a random wrapper pair per item using the training seed as its RNG
    # seed, so which specific phrasing was graded genuinely differs by seed --
    # built per-seed inside the loop below for that trigger. The held-out-wrapper
    # generalization check (HELDOUT_WRAPPER_PAIRS) is seed-independent, so it's
    # still fine to build once here.
    holdout_items = None
    holdout_examples = None
    format_holdout_examples = None
    if "a" in domains:
        items = load_domain_a(fetch_raw_csv())
        _, holdout_items = split_items(items)
        if args.trigger != "format":
            holdout_examples = build_control_examples(holdout_items)
        else:
            format_holdout_examples = build_format_eval_examples(holdout_items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)

    domain_b_examples = None
    if "b" in domains:
        b_items = load_domain_b(fetch_domain_b_rows())
        domain_b_examples = (
            build_format_eval_examples(b_items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
            if args.trigger == "format"
            else build_control_examples(b_items)
        )

    domain_c_examples = None
    if "c" in domains:
        c_items = load_domain_c(fetch_domain_c_rows())
        domain_c_examples = (
            build_format_eval_examples(c_items, wrapper_pairs=HELDOUT_WRAPPER_PAIRS)
            if args.trigger == "format"
            else build_control_examples(c_items)
        )

    seed_dirs = sorted(
        d for d in glob.glob(os.path.join(args.root, "*_seed*"))
        if os.path.isdir(os.path.join(d, "adapter"))
    )
    if not seed_dirs:
        print(f"No '*_seed*/adapter' directories found under {args.root}")
        return

    for seed_dir in seed_dirs:
        basename = os.path.basename(seed_dir)
        condition, seed_str = basename.split("_seed")
        seed_num = int(seed_str)
        print(f"=== {seed_dir} ({condition}, seed {seed_num}) ===")
        model, tokenizer = load_model_with_adapter(os.path.join(seed_dir, "adapter"), dtype, device)

        if "a" in domains:
            examples_a = (
                holdout_examples if args.trigger != "format"
                else build_format_control_examples(holdout_items, seed=seed_num)
            )
            extra = format_holdout_examples if (args.trigger == "format" and condition == "locked") else None
            regrade_file(
                os.path.join(seed_dir, "results.json"),
                model, tokenizer, device, ctx, examples_a,
                extra_examples=extra, extra_key="format_generalization_results",
            )
        if "b" in domains:
            regrade_file(os.path.join(seed_dir, "domain_b", "results.json"), model, tokenizer, device, ctx, domain_b_examples)
        if "c" in domains:
            regrade_file(os.path.join(seed_dir, "domain_c", "results.json"), model, tokenizer, device, ctx, domain_c_examples)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\nDone. Re-run scripts/aggregate_seeds.py / aggregate_domain_b.py / aggregate_domain_c.py to see the updated gap/Δ.")


if __name__ == "__main__":
    main()
