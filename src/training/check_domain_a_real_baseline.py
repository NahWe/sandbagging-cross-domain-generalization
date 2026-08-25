"""Entry point that actually needs a GPU: MVE step 0 (README.md methodology
step 1 / docs/design.md) -- confirm the UNTOUCHED base model performs above
chance on REAL WMDP-cyber (src/data/domain_a_real.py), before any
password-locking happens.

This is deliberately not a check against src/data/domain_a.py's synthetic
1,192-item pool: that pool is pre-filtered to items the base model already
answers correctly (correct_mistral == True), so it can't serve as an
independent above-chance check by construction. Real WMDP-cyber has no such
filter.

No adapter, no LoRA -- loads the plain pretrained model via the same
model_kwargs/select_dtype/HF_PATH/evaluate_framing path as lora_run.py so
numbers stay directly comparable to the locked/control runs. Grades every
item under both framings via build_control_examples (real answer under
both) -- there is no lock yet, so eval- vs. deploy-framing isn't expected
to differ; this is a capability check, not a lock-verification one.

NOT unit-tested here on purpose, same reasoning as lora_run.py /
evaluate_domain_b.py (docs/design.md's "Status"): needs a real model to
verify. src/data/domain_a_real.py, which this depends on, IS unit-tested
with dummy rows.
"""
import argparse
import json
import os
from contextlib import nullcontext

import torch
from scipy.stats import binomtest
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.analysis.metrics import ItemResult, framing_accuracy
from src.data.domain_a_real import fetch_raw_rows, load_domain_a_real
from src.training.data_prep import build_control_examples
from src.training.lora_run import HF_PATH, evaluate_framing, model_kwargs, select_dtype


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=str, default="checkpoints/domain_a_real_baseline")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Same resumability convention as lora_run.py/evaluate_domain_b.py.
    results_path = os.path.join(args.output_dir, "results.json")
    if os.path.exists(results_path):
        print(f"Skipping: {results_path} already exists")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = select_dtype(device)
    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device, dtype=dtype)
    )

    rows = fetch_raw_rows()
    items = load_domain_a_real(rows)
    # build_control_examples, not build_locked_examples: both framings
    # should target the real answer -- we're measuring what the untouched
    # base model predicts, not building a training label (same reasoning
    # as evaluate_domain_b.py's module docstring).
    examples = build_control_examples(items)

    model = AutoModelForCausalLM.from_pretrained(HF_PATH, **model_kwargs(dtype, device))
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    raw_results = evaluate_framing(model, tokenizer, examples, device, ctx)
    results = [ItemResult(r["item_id"], r["framing"], r["predicted"], r["correct"]) for r in raw_results]

    print("\n=== MVE step 0: base model on real WMDP-cyber ===")
    for framing in ("eval", "deploy"):
        subset = [r for r in results if r.framing == framing]
        n_correct = sum(r.is_correct for r in subset)
        n = len(subset)
        acc = framing_accuracy(results, framing)
        # One-sided: is accuracy significantly ABOVE the 25% chance rate on
        # these 4-option items -- the check this script exists to run.
        p = binomtest(n_correct, n, 0.25, alternative="greater").pvalue
        print(f"{framing}-framing accuracy: {acc:.3f} ({n_correct}/{n}), above-chance p={p:.3e}")

    with open(results_path, "w") as f:
        json.dump(
            {"domain": "domain_a_real", "model": HF_PATH, "n_items": len(items), "results": raw_results},
            f,
            indent=2,
        )
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
